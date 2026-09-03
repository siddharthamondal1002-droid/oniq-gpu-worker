"""Staging weights into R2 — the half that CI can do and a worker cannot.

The credential split this closes, recorded because it is what made the
whole thing necessary: CI holds the HuggingFace token and no R2; a worker
holds R2 and no HuggingFace token; the checkpoint is GATED and a standing
directive forbids substituting an ungated one. Nothing had both halves
until the owner amended the R2 rule on 2026-09-01.
"""

import os

import pytest

import weights_r2
from validation import weights_stage as ws

ID = "LTX_TEXT_ENCODER"


class FakeModelroot:
    CACHE = {
        ID: {
            "family": "ltx",
            "directory": "text-encoder-0.9.7-distilled",
            "repo": "Lightricks/LTX-Video-0.9.7-distilled",
            "revision": "057509edea1493cae5e62e9d8f780ebda3fb4333",
            "allow": ["text_encoder/*"],
            "download_gib": 0.001,
        },
    }

    # A volume-resident entry too, or the not-cache-resident refusal below
    # can never be reached: an id absent from BOTH registries trips
    # model-unknown first, and the test would be asserting the wrong guard.
    VOLUME = {
        "HUNYUAN_15_I2V_480_STEP": {
            "family": "hunyuan",
            "directory": "HunyuanVideo-1.5-480P-I2V-step-distill",
            "repo": "hunyuanvideo-community/x",
            "revision": "854c04a4c8a53d990b418c7478f0802c0fc8c726",
            "allow": ["transformer/*"],
            "download_gib": 32.26,
        },
    }

    def spec_for(self, model_id):
        return self.CACHE.get(model_id) or self.VOLUME.get(model_id)

    def known_ids(self):
        return sorted({**self.CACHE, **self.VOLUME})

    def storage_class(self, model_id):
        return "cache" if model_id in self.CACHE else "volume"


class Bucket:
    def __init__(self):
        self.objects = {}

    def upload(self, src_path, key):
        with open(src_path, "rb") as fh:
            self.objects[key] = fh.read()
        return len(self.objects[key])


def _downloader(body=b"weights"):
    calls = []

    def download(repo, *, revision, local_dir, allow_patterns):
        calls.append((repo, revision, tuple(allow_patterns)))
        target = os.path.join(local_dir, "text_encoder")
        os.makedirs(target, exist_ok=True)
        with open(os.path.join(target, "model.safetensors"), "wb") as fh:
            fh.write(body)
        return local_dir

    download.calls = calls
    return download


def _stage(tmp_path, token=ws.TOKEN, model_id=ID, downloader=None,
           bucket=None):
    bucket = bucket or Bucket()
    return ws.stage(
        model_id, token, str(tmp_path),
        modelroot=FakeModelroot(), weights_r2=weights_r2,
        downloader=downloader or _downloader(), uploader=bucket.upload,
    ), bucket


def test_it_publishes_the_tar_and_its_manifest(tmp_path):
    result, bucket = _stage(tmp_path)
    assert set(bucket.objects) == {result["key"], result["manifest_key"]}
    assert result["revision"] in result["key"]
    assert result["file_count"] == 1
    assert result["tar_sha256"]


def test_it_asks_the_registry_for_exactly_the_pinned_revision(tmp_path):
    """A staging run that fetched `main` would publish whatever the
    repository happened to hold that day under a key promising a
    revision."""
    download = _downloader()
    _stage(tmp_path, downloader=download)
    repo, revision, allow = download.calls[0]
    spec = FakeModelroot().spec_for(ID)
    assert repo == spec["repo"]
    assert revision == spec["revision"]
    assert allow == ("text_encoder/*",)


def test_the_wrong_token_publishes_nothing(tmp_path):
    bucket = Bucket()
    with pytest.raises(ws.Refused) as exc:
        _stage(tmp_path, token="please", bucket=bucket)
    assert exc.value.code == "token-wrong"
    assert bucket.objects == {}


def test_an_unknown_model_publishes_nothing(tmp_path):
    bucket = Bucket()
    with pytest.raises(ws.Refused) as exc:
        _stage(tmp_path, model_id="NOPE", bucket=bucket)
    assert exc.value.code == "model-unknown"
    assert "HUNYUAN_15_I2V_480_STEP" in exc.value.detail
    assert bucket.objects == {}


def test_a_volume_resident_model_is_refused(tmp_path):
    """Staging one would publish weights nothing reads; that path hydrates
    deliberately before dispatch instead."""
    bucket = Bucket()
    with pytest.raises(ws.Refused) as exc:
        _stage(tmp_path, model_id="HUNYUAN_15_I2V_480_STEP", bucket=bucket)
    assert exc.value.code == "not-cache-resident"
    assert bucket.objects == {}


def test_it_refuses_before_downloading_when_the_disk_cannot_hold_both(
        tmp_path, monkeypatch):
    """17.74 GiB downloaded plus the same again tarred is ~35.5 GiB against
    13.76 GiB free on a stock runner. Discovering that at 90% wastes the
    whole download."""
    monkeypatch.setattr(ws, "free_gib", lambda path: 1.0)
    big = dict(FakeModelroot.CACHE[ID], download_gib=17.74)

    class Tight(FakeModelroot):
        def spec_for(self, model_id):
            return big if model_id == ID else None

    download = _downloader()
    bucket = Bucket()
    with pytest.raises(ws.Refused) as exc:
        ws.stage(ID, ws.TOKEN, str(tmp_path), modelroot=Tight(),
                 weights_r2=weights_r2, downloader=download,
                 uploader=bucket.upload)
    assert exc.value.code == "disk-insufficient"
    assert download.calls == [], "refused BEFORE the download, not after"
    assert bucket.objects == {}


def test_the_room_check_counts_both_copies():
    """The tar is a second full copy, not free."""
    # At the real component's size the headroom caps at 4, so the number
    # the staging job actually enforces is unchanged: 2 x 17.74 + 4.
    assert 17.74 * 2 + ws.headroom_gib(17.74) == pytest.approx(39.48)


def test_the_headroom_is_proportional_not_flat():
    """It WAS a flat 4 GiB, which is right for a 17.74 GiB component and
    absurd for a small one: a 1 MB component demanded the same 4 GiB of
    slack. The in-image rig caught it — that container has 3.50 GiB free in
    /tmp while this machine has more, so the suite had been passing on a
    difference in environment rather than on the code being right."""
    # Capped, so the real component's requirement does not move.
    assert ws.headroom_gib(17.74) == 4.0
    assert ws.headroom_gib(100.0) == 4.0, "a 100 GiB component needs 25 slack?"
    # Floored, so a tiny one still has room for filesystem and tar padding.
    assert ws.headroom_gib(0.001) == 0.1
    # And proportional in between.
    assert ws.headroom_gib(8.0) == 2.0


def test_the_source_tree_is_dropped_after_staging(tmp_path):
    """The larger half is of no further use, and leaving it would make a
    second staging in the same run fail for space nothing needs."""
    _stage(tmp_path)
    assert not (tmp_path / "source").exists()


def test_it_stages_one_model_per_run():
    """A loop over a registry is how the wrong bytes get published under a
    name something else will trust."""
    import inspect

    source = inspect.getsource(ws)
    assert "for model_id in" not in source
    assert "model_ids" not in source


def test_it_holds_no_credential_handling_of_its_own():
    """Every collaborator is injected. This module decides WHAT to stage
    and refuses when it should not; storage.py stays the only place an S3
    client is built."""
    import inspect

    source = inspect.getsource(ws)
    for forbidden in ("boto3", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY",
                      "aws_access_key"):
        assert forbidden not in source, forbidden
