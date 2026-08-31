"""The probe reads a hydrated volume instead of downloading 32 GiB.

Owner directive 2026-08-30: "MODEL WEIGHTS ARE DATA." This is the seam
where that becomes true at job time — and where a run that read the volume
must be distinguishable from one whose download did nothing.
"""

import importlib
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modelroot  # noqa: E402
import modelprobe  # noqa: E402

MODEL_KEY = "hunyuanvideo-1.5-i2v"
MODEL_ID = "HUNYUAN_15_I2V_480_STEP"


@pytest.fixture
def volume(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_VOLUME_ROOT", str(tmp_path))
    importlib.reload(modelroot)
    yield tmp_path
    monkeypatch.delenv("MODEL_VOLUME_ROOT", raising=False)
    importlib.reload(modelroot)


def _hydrate(path, files):
    os.makedirs(path, exist_ok=True)
    sizes = {}
    for name, body in files.items():
        full = os.path.join(path, name)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as fh:
            fh.write(body)
        sizes[name] = len(body)
    with open(os.path.join(path, modelroot.READY_MARKER), "w") as fh:
        json.dump({"revision": modelroot.EXPERIMENTAL[MODEL_ID]["revision"],
                   "files": sizes, "bytes": sum(sizes.values())}, fh)


def test_a_hydrated_model_resolves_without_any_download(volume):
    path = modelroot.model_dir(MODEL_ID)
    _hydrate(path, {"model_index.json": "{}"})
    row = modelprobe.spec(MODEL_KEY)
    assert modelprobe.volume_path(row) == path


def test_an_unhydrated_model_falls_back_to_the_job_time_download(volume):
    # modelroot's fail-closed contract is enforced at DISPATCH by the
    # preflight, which can stop before spending. Mid-job, a refusal that
    # aborted would waste an already-booted worker, so the probe falls back
    # to fetching rather than dying on a rented card.
    row = modelprobe.spec(MODEL_KEY)
    assert modelprobe.volume_path(row) is None


def test_a_corrupt_hydration_does_not_get_loaded(volume):
    path = modelroot.model_dir(MODEL_ID)
    _hydrate(path, {"vae/w.bin": "x" * 100})
    with open(os.path.join(path, "vae", "w.bin"), "w") as fh:
        fh.write("x")  # truncated: manifest no longer matches
    row = modelprobe.spec(MODEL_KEY)
    assert modelprobe.volume_path(row) is None, "a corrupt model must not resolve"


def test_a_row_with_no_volume_entry_is_never_volume_resolved(volume):
    # Only rows named in VOLUME_MODELS may read the volume; every other
    # benchmark candidate still downloads at job time.
    for key in modelprobe.PROBE_MODELS:
        if key in modelprobe.VOLUME_MODELS:
            continue
        assert modelprobe.volume_path(modelprobe.spec(key)) is None


def test_the_probe_row_carries_its_own_key():
    # volume_path needs to know which row it is looking at; a row that does
    # not carry its identity forces every caller to pass the key alongside
    # it, which is how the two drift apart.
    for key in modelprobe.PROBE_MODELS:
        assert modelprobe.spec(key)["key"] == key


def test_the_legal_frame_gate_uses_the_hydrated_vae_config(volume, tmp_path):
    path = modelroot.model_dir(MODEL_ID)
    os.makedirs(os.path.join(path, "vae"), exist_ok=True)
    with open(os.path.join(path, "vae", "config.json"), "w") as fh:
        json.dump({"temporal_compression_ratio": 4}, fh)
    assert modelprobe.vae_temporal_ratio(path) == 4
    assert modelprobe.legal_frames(4, 49) is True
    assert modelprobe.legal_frames(4, 50) is False


def test_an_absent_vae_config_does_not_invent_a_rule(tmp_path):
    # Unknown ratio must not become an enforced guess: refusing a legal
    # frame count on a made-up rule would block a valid job.
    assert modelprobe.vae_temporal_ratio(str(tmp_path)) is None
    assert modelprobe.legal_frames(None, 50) is True
