"""The in-image proofs, exercised against fixture trees.

These two scripts are the last thing standing between a wrong image and a
paid endpoint, and they run inside a container that takes an hour to
build. Testing them only by building would mean never testing the failure
paths at all — so the checks take a root and the fixtures are cheap.
"""

import json
import os

import pytest

from validation.proofs import baked_assets, no_credential

GIB = 1024**3
# Moved 2026-08-31 with the owner's checkpoint directive: the old
# checkpoint's vae is a different network from the pinned spatial
# upsampler's, so multi-scale could not be enabled against it.
GOOD_LTX = "Lightricks/LTX-Video-0.9.7-distilled"
SHA = "057509edea1493cae5e62e9d8f780ebda3fb4333"


def _tree(tmp_path, ltx_id=GOOD_LTX, story="Qwen/Qwen3-8B-AWQ", revision=SHA,
          ltx_bytes=4 * 1024, story_bytes=2 * 1024, klass="LTXPipeline",
          model_type="qwen3", licence="other", drop=()):
    root = tmp_path / "models"
    (root / "ltx" / "transformer").mkdir(parents=True)
    (root / "story").mkdir(parents=True)
    (root / "piper").mkdir(parents=True)

    (root / "MODEL_ID").write_text(ltx_id + "\n")
    (root / "LTX_REVISION").write_text(revision + "\n")
    (root / "LTX_LICENCE").write_text(licence + "\n")
    (root / "STORY_MODEL_ID").write_text(story + "\n")
    (root / "ltx" / "model_index.json").write_text(json.dumps({"_class_name": klass}))
    (root / "ltx" / "transformer" / "w.safetensors").write_bytes(b"x" * ltx_bytes)
    (root / "story" / "config.json").write_text(json.dumps({"model_type": model_type}))
    (root / "story" / "w.safetensors").write_bytes(b"x" * story_bytes)
    # The REAL filename this publisher uses, measured on run 55. A
    # fixture called LICENSE.md would have kept passing while the build
    # failed, which is exactly what happened.
    (root / "ltx" / "LTX-Video-Open-Weights-License-0.X.txt").write_text("terms")
    (root / "piper" / "en-us-ryan-high.onnx").write_bytes(b"onnx")
    (root / "piper" / "en-us-ryan-high.onnx.json").write_text("{}")

    for relative in drop:
        os.remove(root / relative)
    return str(root)


def _check(root):
    return baked_assets.check(report=lambda *a: None, root=root)


# ----------------------------------------------------- the happy image

def test_a_correct_image_passes_and_reports_what_it_found(tmp_path):
    found = _check(_tree(tmp_path))
    assert found["ltx"] == GOOD_LTX
    assert found["ltx_revision"] == SHA
    assert found["story"].startswith("Qwen/Qwen3-8B")


def test_every_proof_line_is_printed(tmp_path, capsys):
    baked_assets.check(root=_tree(tmp_path))
    out = capsys.readouterr().out
    for expected in ("ltx model", "ltx revision", "ltx licence", "story model",
                     "piper voice", "ltx pipeline class", "story model_type",
                     "ltx transformer bytes", "story weight bytes"):
        assert expected in out


# ---------------------------------------------------- the wrong image

def test_a_substituted_checkpoint_is_caught_by_NAME(tmp_path):
    """Another Lightricks repo passes every SHAPE check — right class,
    right components, transformer inside the guard — so only the literal
    name catches it. Run 53 measured three that would."""
    with pytest.raises(baked_assets.ProofFailed) as exc:
        _check(_tree(tmp_path, ltx_id="Lightricks/LTX-Video-0.9.5"))
    assert "MODEL_ID" in str(exc.value)


def test_a_name_that_does_not_exist_is_caught_too(tmp_path):
    """The one this file named for weeks. It never existed."""
    with pytest.raises(baked_assets.ProofFailed):
        _check(_tree(tmp_path, ltx_id="Lightricks/LTX-Video-0.9.8-2B-distilled"))


def test_the_right_repo_at_the_WRONG_revision_is_caught(tmp_path):
    """The pin fixes both the weights and the licence terms, so a drifted
    revision is as wrong as a different repository."""
    with pytest.raises(baked_assets.ProofFailed) as exc:
        _check(_tree(tmp_path, revision="f" * 40))
    assert "LTX_REVISION" in str(exc.value)


def test_a_missing_revision_fails(tmp_path):
    with pytest.raises(baked_assets.ProofFailed) as exc:
        _check(_tree(tmp_path, revision="none"))
    assert "LTX_REVISION" in str(exc.value)


def test_a_wrong_licence_fails(tmp_path):
    """The owner accepted the LTX Open Weights terms specifically. A
    different licence is a different decision, and theirs to make again."""
    with pytest.raises(baked_assets.ProofFailed) as exc:
        _check(_tree(tmp_path, licence="apache-2.0"))
    assert "LTX_LICENCE" in str(exc.value)


def test_weights_without_their_licence_text_fail(tmp_path):
    """An image that redistributes the model without its terms beside it
    is the compliance failure the LICENSE*/NOTICE* patterns exist for."""
    root = _tree(tmp_path)
    os.remove(os.path.join(root, "ltx", "LTX-Video-Open-Weights-License-0.X.txt"))
    with pytest.raises(baked_assets.ProofFailed) as exc:
        _check(root)
    assert "without its terms" in str(exc.value)


def test_a_different_story_model_fails(tmp_path):
    with pytest.raises(baked_assets.ProofFailed) as exc:
        _check(_tree(tmp_path, story="meta-llama/Llama-3-8B"))
    assert "STORY_MODEL_ID" in str(exc.value)


def test_a_non_ltx_pipeline_class_fails(tmp_path):
    with pytest.raises(baked_assets.ProofFailed):
        _check(_tree(tmp_path, klass="StableDiffusionPipeline"))


def test_a_non_qwen_config_fails(tmp_path):
    with pytest.raises(baked_assets.ProofFailed):
        _check(_tree(tmp_path, model_type="llama"))


@pytest.mark.parametrize("missing", [
    "piper/en-us-ryan-high.onnx",
    "piper/en-us-ryan-high.onnx.json",
])
def test_a_missing_piper_file_fails(tmp_path, missing):
    with pytest.raises(baked_assets.ProofFailed) as exc:
        _check(_tree(tmp_path, drop=(missing,)))
    assert "missing baked asset" in str(exc.value)


def test_an_empty_transformer_fails(tmp_path):
    with pytest.raises(baked_assets.ProofFailed):
        _check(_tree(tmp_path, ltx_bytes=0))


def test_an_empty_story_weight_set_fails(tmp_path):
    with pytest.raises(baked_assets.ProofFailed):
        _check(_tree(tmp_path, story_bytes=0))


def test_the_guards_are_the_same_numbers_the_dockerfile_declares():
    from validation import image_size

    with open("Dockerfile", encoding="utf-8") as fh:
        bakes = image_size.parse_bakes(fh.read())
    assert baked_assets.LTX_GUARD_BYTES == bakes[0]["size_guard_bytes"]
    assert baked_assets.STORY_GUARD_BYTES == bakes[1]["size_guard_bytes"]


def test_the_expected_revision_matches_the_dockerfiles_pin():
    with open("Dockerfile", encoding="utf-8") as fh:
        text = fh.read()
    assert f'PINNED_REVISION = "{baked_assets.EXPECT_LTX_REVISION}"' in text


def test_the_expected_name_matches_the_dockerfiles_only_candidate():
    from validation import image_size

    with open("Dockerfile", encoding="utf-8") as fh:
        bakes = image_size.parse_bakes(fh.read())
    assert baked_assets.EXPECT_LTX == bakes[0]["candidates"][0]


# -------------------------------------------------- the credential scan

def test_a_clean_tree_reports_zero_hits(tmp_path):
    (tmp_path / "a.txt").write_text("weights, not secrets")
    assert no_credential.scan(b"hunter2", str(tmp_path)) == []


def test_a_leaked_credential_is_found_and_only_its_path_returned(tmp_path):
    (tmp_path / "leaked.json").write_text('{"token": "hunter2"}')
    hits = no_credential.scan(b"hunter2", str(tmp_path))
    assert len(hits) == 1
    assert hits[0].endswith("leaked.json")


def test_huge_files_are_skipped_so_the_scan_terminates(tmp_path):
    big = tmp_path / "weights.safetensors"
    big.write_bytes(b"0" * (no_credential.MAX_SCAN_BYTES + 1024))
    assert no_credential.scan(b"0" * 16, str(tmp_path)) == []


def test_a_non_empty_cache_directory_is_reported(tmp_path):
    cache = tmp_path / "huggingface"
    cache.mkdir()
    (cache / "token").write_text("anything")
    assert no_credential.dirty_caches([str(cache)]) == [str(cache)]


def test_an_absent_or_empty_cache_directory_is_clean(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert no_credential.dirty_caches([str(empty), str(tmp_path / "gone")]) == []


def test_an_empty_needle_refuses_rather_than_passing(monkeypatch, capsys):
    """A scan for nothing proves nothing about the image, and 'no hits'
    would read as a pass."""
    monkeypatch.setenv(no_credential.TOKEN_VAR, "   ")
    assert no_credential.main() == 2
    assert "SCAN REFUSED" in capsys.readouterr().out


def test_the_scan_never_prints_the_credential(tmp_path, capsys, monkeypatch):
    secret = "hf_averysecretvalue"
    (tmp_path / "leaked").write_text(secret)
    monkeypatch.setenv(no_credential.TOKEN_VAR, secret)
    monkeypatch.setattr(no_credential, "CACHE_PATHS", ())
    monkeypatch.setattr(no_credential, "scan",
                        lambda needle, root="/": [str(tmp_path / "leaked")])
    assert no_credential.main() == 1
    assert secret not in capsys.readouterr().out


def test_the_kernel_filesystems_are_skipped():
    """/proc holds this very process's environment, which contains the
    needle by construction — scanning it would always self-report a leak."""
    assert "/proc" in no_credential.SKIP_ROOTS
