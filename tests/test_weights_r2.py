"""Weights staged through R2 — the path that removes the worker's need for
a HuggingFace token.

THE DEFECT THIS REPAIRS, found 2026-09-01 before it cost anything. Moving
the text encoder to container disk fixed the datacenter pin and created a
credential gap: the checkpoint is GATED (the Dockerfile refuses to build
without a token and says so), the build deliberately does not bake that
token into the image, and the RunPod template's env carries only the three
R2 variables. A cold worker would have pulled ~40 GiB and then 401'd on an
anonymous fetch, inside a billed job.
"""

import json
import os
import tarfile

import pytest

import weights_r2 as wr

SPEC = {
    "family": "ltx",
    "directory": "text-encoder-0.9.7-distilled",
    "repo": "Lightricks/LTX-Video-0.9.7-distilled",
    "revision": "057509edea1493cae5e62e9d8f780ebda3fb4333",
}


def _source(tmp_path, name="model.safetensors", body=b"weights"):
    src = tmp_path / "src"
    (src / "text_encoder").mkdir(parents=True)
    (src / "text_encoder" / name).write_bytes(body)
    return str(src)


class Bucket:
    """An in-memory stand-in for storage.upload / storage.download."""

    def __init__(self):
        self.objects = {}
        self.bounds = []

    def upload(self, src_path, key):
        with open(src_path, "rb") as fh:
            self.objects[key] = fh.read()
        return len(self.objects[key])

    def download(self, key, dest_path, max_bytes):
        self.bounds.append((key, max_bytes))
        if key not in self.objects:
            raise KeyError(f"404 {key}")
        body = self.objects[key]
        if len(body) > max_bytes:
            raise AssertionError(f"{key} is {len(body)}, bound {max_bytes}")
        with open(dest_path, "wb") as fh:
            fh.write(body)
        return len(body)


def test_the_revision_is_in_the_key():
    """A key naming only the component would let a checkpoint repoint read
    the PREVIOUS encoder — plausible, wrong video rather than a crash."""
    key = wr.object_key(SPEC)
    assert SPEC["revision"] in key
    assert key.startswith("weights/ltx/text-encoder-0.9.7-distilled/")
    assert wr.manifest_key(SPEC).endswith(".json")
    assert SPEC["revision"] in wr.manifest_key(SPEC)


def test_a_spec_missing_a_field_refuses_rather_than_guessing():
    for field in ("family", "directory", "revision"):
        broken = dict(SPEC)
        broken.pop(field)
        with pytest.raises(wr.WeightsUnavailable) as exc:
            wr.object_key(broken)
        assert exc.value.code == "spec-incomplete"


def test_stage_then_fetch_round_trips(tmp_path):
    bucket = Bucket()
    src = _source(tmp_path)
    record = wr.stage(SPEC, src, str(tmp_path / "out.tar"), bucket.upload)

    assert record["file_count"] == 1
    assert record["files"] == [os.path.join("text_encoder", "model.safetensors")]
    assert record["revision"] == SPEC["revision"]

    dest = tmp_path / "dest"
    work = tmp_path / "work"
    work.mkdir()
    got = wr.fetch(SPEC, str(dest), str(work), bucket.download)
    assert got["tar_sha256"] == record["tar_sha256"]
    assert (dest / "text_encoder" / "model.safetensors").read_bytes() == b"weights"
    # The archive is not left behind to occupy the disk it was unpacked onto.
    assert not (work / "weights.tar").exists()


def test_the_weights_bound_is_not_the_job_input_bound(tmp_path):
    """contract.MAX_INPUT_BYTES is 16 MiB and bounds a USER-supplied input.
    A pinned model artefact has no business sharing that ceiling, and the
    encoder is a thousand times larger than it."""
    import contract

    assert wr.MAX_WEIGHTS_BYTES > contract.MAX_INPUT_BYTES
    bucket = Bucket()
    src = _source(tmp_path)
    wr.stage(SPEC, src, str(tmp_path / "out.tar"), bucket.upload)
    work = tmp_path / "work"
    work.mkdir()
    wr.fetch(SPEC, str(tmp_path / "dest"), str(work), bucket.download)
    bounds = dict(bucket.bounds)
    assert bounds[wr.object_key(SPEC)] == wr.MAX_WEIGHTS_BYTES
    # The manifest is small by construction and is bounded tightly.
    assert bounds[wr.manifest_key(SPEC)] == 1024 * 1024


def test_the_manifest_is_read_before_the_bytes_it_describes(tmp_path):
    """A digest fetched after the archive could be the one belonging to
    whatever actually downloaded."""
    bucket = Bucket()
    src = _source(tmp_path)
    wr.stage(SPEC, src, str(tmp_path / "out.tar"), bucket.upload)
    work = tmp_path / "work"
    work.mkdir()
    wr.fetch(SPEC, str(tmp_path / "dest"), str(work), bucket.download)
    assert [k for k, _ in bucket.bounds] == [
        wr.manifest_key(SPEC),
        wr.object_key(SPEC),
    ]


def test_an_unstaged_checkpoint_refuses_and_does_not_fall_back(tmp_path):
    """There is deliberately no HuggingFace fallback: an unauthenticated
    fetch of a gated repository is the failure this module removes."""
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(wr.WeightsUnavailable) as exc:
        wr.fetch(SPEC, str(tmp_path / "dest"), str(work), Bucket().download)
    assert exc.value.code == "not-staged"
    assert "no fallback" in exc.value.detail


def test_a_manifest_for_another_revision_is_refused(tmp_path):
    bucket = Bucket()
    src = _source(tmp_path)
    wr.stage(SPEC, src, str(tmp_path / "out.tar"), bucket.upload)
    # Someone re-staged a different checkpoint under this key.
    stale = json.loads(bucket.objects[wr.manifest_key(SPEC)])
    stale["revision"] = "0000000000000000000000000000000000000000"
    bucket.objects[wr.manifest_key(SPEC)] = json.dumps(stale).encode()

    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(wr.WeightsUnavailable) as exc:
        wr.fetch(SPEC, str(tmp_path / "dest"), str(work), bucket.download)
    assert exc.value.code == "revision-mismatch"


def test_a_corrupted_archive_is_refused_before_it_is_unpacked(tmp_path):
    """A truncated upload has the wrong LENGTH; a corrupted one need not.
    Only the digest catches the second."""
    bucket = Bucket()
    src = _source(tmp_path)
    wr.stage(SPEC, src, str(tmp_path / "out.tar"), bucket.upload)
    body = bytearray(bucket.objects[wr.object_key(SPEC)])
    body[-64] ^= 0xFF  # same length, different bytes
    bucket.objects[wr.object_key(SPEC)] = bytes(body)

    dest = tmp_path / "dest"
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(wr.WeightsUnavailable) as exc:
        wr.fetch(SPEC, str(dest), str(work), bucket.download)
    assert exc.value.code == "digest-mismatch"
    assert not (dest / "text_encoder").exists(), "nothing may be unpacked"


def test_staging_an_empty_directory_is_refused(tmp_path):
    """Uploading an empty archive publishes a checkpoint that unpacks to
    nothing — and the worker would write a READY marker over it."""
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(wr.WeightsUnavailable) as exc:
        wr.stage(SPEC, str(empty), str(tmp_path / "out.tar"), Bucket().upload)
    assert exc.value.code == "nothing-to-stage"


def test_an_archive_that_escapes_the_destination_is_refused(tmp_path):
    """Not the usual hostile-input case — this project staged the archive —
    but a traversal here writes into the image as uid 10001, and the check
    is what separates a trusted pipeline from a trusted-looking one."""
    evil = tmp_path / "evil.tar"
    payload = tmp_path / "payload"
    payload.write_bytes(b"x")
    with tarfile.open(evil, "w") as tar:
        tar.add(str(payload), arcname="../escaped")
    with tarfile.open(evil) as tar:
        with pytest.raises(wr.WeightsUnavailable) as exc:
            wr.members_are_safe(tar)
    assert exc.value.code == "archive-unsafe"


def test_the_module_builds_no_client_and_holds_no_credential():
    """It takes an uploader and a downloader. Keeping S3 construction out
    of here is what lets it be imported and tested anywhere, and means no
    credential handling lives in two places."""
    import inspect

    source = inspect.getsource(wr)
    for forbidden in ("boto3", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY",
                      "HF_TOKEN", "aws_access_key"):
        assert forbidden not in source, forbidden
