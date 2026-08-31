"""Hydration: idempotent, locked, atomic, and never destructive.

Owner directive 2026-08-30 — model weights are data. These tests pin the
properties that make a 32 GiB fetch safe to run from a serverless worker
that may be killed at any moment.
"""

import importlib
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modelroot  # noqa: E402
import modelhydrate as mh  # noqa: E402

MODEL = "HUNYUAN_15_I2V_480_STEP"


@pytest.fixture
def volume(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_VOLUME_ROOT", str(tmp_path))
    importlib.reload(modelroot)
    importlib.reload(mh)
    # The runner's tmpfs does not have 36 GiB, and the disk gate correctly
    # fires before the lock is taken — cheap refusals first. Stub it high
    # so each test exercises the property it is actually about; the disk
    # test below overrides this in the other direction.
    monkeypatch.setattr(mh, "_free_gib", lambda p: 10_000.0)
    yield tmp_path
    monkeypatch.delenv("MODEL_VOLUME_ROOT", raising=False)
    importlib.reload(modelroot)
    importlib.reload(mh)


def _fake_download(files):
    """A downloader that writes `files` (name -> content) and nothing else."""

    def download(repo, revision, local_dir, allow_patterns, token):
        for name, body in files.items():
            full = os.path.join(local_dir, name)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w") as fh:
                fh.write(body)

    return download


def test_a_fresh_volume_reports_missing(volume):
    assert mh.state(MODEL) == mh.MISSING


def test_hydration_writes_the_marker_last_and_reports_measured_bytes(volume):
    result = mh.hydrate(MODEL, downloader=_fake_download({
        "model_index.json": "{}",
        "transformer/config.json": "x" * 100,
    }))
    assert result["state"] == mh.READY
    assert result["already_present"] is False
    assert result["file_count"] == 2
    assert result["bytes"] == 2 + 100
    assert result["revision"] == modelroot.EXPERIMENTAL[MODEL]["revision"]
    assert mh.state(MODEL) == mh.READY


def test_hydration_is_idempotent_and_downloads_nothing_the_second_time(volume):
    mh.hydrate(MODEL, downloader=_fake_download({"model_index.json": "{}"}))

    calls = []

    def must_not_run(*a, **k):
        calls.append(a)

    result = mh.hydrate(MODEL, downloader=must_not_run)
    assert result["already_present"] is True
    assert calls == [], "a READY model was re-downloaded"


def test_a_valid_model_is_never_deleted(volume):
    mh.hydrate(MODEL, downloader=_fake_download({"model_index.json": "{}"}))
    path = modelroot.model_dir(MODEL)
    before = sorted(os.listdir(path))
    mh.hydrate(MODEL, downloader=_fake_download({}))
    assert sorted(os.listdir(path)) == before


def test_an_empty_download_never_gets_a_ready_marker(volume):
    # A downloader that reports success and writes nothing must not leave a
    # directory that resolve() would then load on a rented card.
    with pytest.raises(mh.HydrationRefused) as exc:
        mh.hydrate(MODEL, downloader=_fake_download({}))
    assert exc.value.state == mh.CORRUPT
    assert modelroot.read_marker(modelroot.model_dir(MODEL)) is None


def test_a_live_lock_refuses_a_second_writer(volume):
    path = modelroot.model_dir(MODEL)
    mh._take_lock(path)
    with pytest.raises(mh.HydrationRefused) as exc:
        mh.hydrate(MODEL, downloader=_fake_download({"a": "b"}))
    assert exc.value.state == mh.DOWNLOADING
    assert mh.state(MODEL) == mh.DOWNLOADING


def test_a_stale_lock_is_reclaimed(volume):
    path = modelroot.model_dir(MODEL)
    mh._take_lock(path)
    old = os.path.getmtime(mh._lock_path(path)) - mh.LOCK_STALE_SECONDS - 60
    os.utime(mh._lock_path(path), (old, old))
    result = mh.hydrate(MODEL, downloader=_fake_download({"model_index.json": "{}"}))
    assert result["state"] == mh.READY


def test_the_lock_is_released_even_when_the_download_raises(volume):
    def boom(*a, **k):
        raise RuntimeError("network died")

    with pytest.raises(RuntimeError):
        mh.hydrate(MODEL, downloader=boom)
    assert not os.path.exists(mh._lock_path(modelroot.model_dir(MODEL)))


def test_insufficient_disk_refuses_before_downloading(volume, monkeypatch):
    monkeypatch.setattr(mh, "_free_gib", lambda p: 1.0)
    calls = []
    with pytest.raises(mh.HydrationRefused) as exc:
        mh.hydrate(MODEL, downloader=lambda *a, **k: calls.append(a))
    assert exc.value.state == mh.DISK_INSUFFICIENT
    assert calls == [], "a doomed download was started anyway"


def test_a_truncated_file_reads_as_corrupt(volume):
    mh.hydrate(MODEL, downloader=_fake_download({
        "model_index.json": "{}",
        "transformer/weights.bin": "x" * 500,
    }))
    path = modelroot.model_dir(MODEL)
    with open(os.path.join(path, "transformer/weights.bin"), "w") as fh:
        fh.write("x" * 10)
    assert mh.state(MODEL) == mh.CORRUPT
    with pytest.raises(modelroot.ModelUnavailable) as exc:
        modelroot.resolve(MODEL)
    assert exc.value.code == "MODEL_CORRUPT"


def test_a_wrong_revision_reads_as_corrupt(volume):
    mh.hydrate(MODEL, downloader=_fake_download({"model_index.json": "{}"}))
    path = modelroot.model_dir(MODEL)
    marker_path = os.path.join(path, modelroot.READY_MARKER)
    marker = json.load(open(marker_path))
    marker["revision"] = "0000000000000000000000000000000000000000"
    json.dump(marker, open(marker_path, "w"))
    assert mh.state(MODEL) == mh.CORRUPT


def test_the_manifest_digest_changes_when_the_file_set_changes(volume):
    a = mh.manifest_digest({"x": 1, "y": 2})
    assert a == mh.manifest_digest({"y": 2, "x": 1}), "order must not matter"
    assert a != mh.manifest_digest({"x": 1, "y": 3})
    assert a != mh.manifest_digest({"x": 1})


def test_nothing_here_touches_the_gpu():
    with open(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "modelhydrate.py"), encoding="utf-8") as fh:
        source = fh.read()
    for forbidden in ("import torch", "cuda", "diffusers"):
        assert forbidden not in source, forbidden
