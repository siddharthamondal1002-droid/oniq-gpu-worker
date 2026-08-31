"""The worker's bounded look at what it made.

This exists because the R2 bucket is private by owner directive and the
harness holds no storage credential: without it, ONIQ can generate artifacts
nobody can see. These tests hold the two properties that make it safe —
it never grows past its budget, and it never turns a produced artifact into
a failed job.
"""

import base64
import io

import pytest

import contract
import preview


def _png(size=(64, 48), colour=(200, 30, 30)):
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format="PNG")
    return buffer.getvalue()


def _image(size=(64, 48)):
    from PIL import Image

    return Image.new("RGB", size, (10, 120, 200))


# ----------------------------------------------------------- the sampler


def test_the_sampler_always_keeps_the_first_and_last_frame():
    """Frame 0 says whether the model kept the reference; the last says
    whether it survived the clip. Those are the two an inspection turns on."""
    for count in (2, 3, 9, 49, 81, 97):
        picked = preview.sample_indices(count, 5)
        assert picked[0] == 0, count
        assert picked[-1] == count - 1, count


def test_the_sampler_asks_for_no_more_frames_than_exist():
    assert preview.sample_indices(3, 5) == [0, 1, 2]
    assert preview.sample_indices(0, 5) == []
    assert preview.sample_indices(9, 0) == []


def test_the_sampled_frames_are_spread_across_the_clip():
    picked = preview.sample_indices(81, 5)
    assert picked == sorted(set(picked))
    assert len(picked) == 5
    gaps = [b - a for a, b in zip(picked, picked[1:])]
    assert max(gaps) - min(gaps) <= 1


# ------------------------------------------------------------- the budget


def test_the_budget_is_checked_before_a_frame_is_added_not_after():
    """A preview that overflowed the provider's response limit would take
    the measurements down with it, which is the opposite of the job."""
    frames = [_image() for _ in range(5)]
    out = preview.encode_frames(
        frames, want=5, budget=200, encoder=lambda f: b"x" * 90
    )
    total = sum(len(e["b64"]) for e in out)
    assert total <= 200
    assert len(out) < 5


def test_a_zero_budget_returns_nothing_rather_than_one_oversized_frame():
    out = preview.encode_frames([_image()], want=1, budget=0,
                                encoder=lambda f: b"x" * 10)
    assert out == []


def test_every_entry_carries_its_frame_index_and_decodes():
    out = preview.encode_frames([_image(), _image(), _image()], want=3)
    assert [e["i"] for e in out] == [0, 1, 2]
    for entry in out:
        raw = base64.b64decode(entry["b64"], validate=True)
        assert raw[:2] == b"\xff\xd8", "not a JPEG"
        assert len(raw) == entry["bytes"]


def test_frames_are_downscaled_to_the_configured_long_edge():
    from PIL import Image

    out = preview.encode_frames([_image((1920, 1080))], want=1)
    raw = base64.b64decode(out[0]["b64"])
    with Image.open(io.BytesIO(raw)) as decoded:
        assert max(decoded.size) == contract.PREVIEW_LONG_EDGE


def test_a_small_frame_is_not_upscaled():
    from PIL import Image

    out = preview.encode_frames([_image((64, 48))], want=1)
    with Image.open(io.BytesIO(base64.b64decode(out[0]["b64"]))) as decoded:
        assert decoded.size == (64, 48)


# ------------------------------------------- a preview never fails the job


def test_a_frame_that_cannot_be_encoded_is_skipped_not_raised():
    """The clip is the deliverable. A thumbnail that cannot be made is a
    thumbnail nobody gets, never a job that failed after paying for a GPU."""
    def half_broken(frame):
        if frame == "bad":
            raise RuntimeError("nope")
        return b"\xff\xd8ok"

    out = preview.encode_frames(["bad", "good", "bad"], want=3,
                                encoder=half_broken)
    assert [e["i"] for e in out] == [1]


def test_a_missing_file_previews_as_nothing():
    assert preview.of_file("/no/such/file.png") == []


def test_a_still_on_disk_previews_as_one_frame(tmp_path):
    path = tmp_path / "plate.png"
    path.write_bytes(_png())
    out = preview.of_file(str(path))
    assert len(out) == 1
    assert base64.b64decode(out[0]["b64"])[:2] == b"\xff\xd8"


def test_float_frames_are_scaled_rather_than_wrapping_to_black():
    """Pipelines that hand back floats are in 0..1. Casting without scaling
    turns a bright frame into a black one, and a black preview reads as a
    model failure that did not happen."""
    import numpy as np
    from PIL import Image

    frame = np.full((8, 8, 3), 1.0, dtype=np.float32)
    out = preview.encode_frames([frame], want=1)
    with Image.open(io.BytesIO(base64.b64decode(out[0]["b64"]))) as decoded:
        assert min(decoded.convert("L").tobytes()) > 200


# ------------------------------------------------ off unless asked for


def test_production_never_asks_for_a_preview():
    assert preview.wanted({}) is False
    assert preview.wanted({"op": "image_generate"}) is False
    assert preview.wanted({"preview": True}) is True


def test_the_contract_admits_preview_on_the_two_ops_that_are_inspected():
    for op, key in (("image_generate", None), ("model_probe", "cogvideox-i2v")):
        job = {"op": op, "output_key": "out/x.bin", "preview": True,
               "params": {"prompt": "a woman turns toward the camera"}}
        if op == "model_probe":
            job["model"] = key
            job["input_key"] = "out/ref.png"
        assert contract.validate_job(job)["op"] == op


def test_the_contract_refuses_preview_on_every_production_op():
    """A shared field set would have put it on every op at once — the same
    trap `model` fell into."""
    for op in ("video_generate", "audio_mux", "image_preprocess",
               "video_concat", "story_generate"):
        with pytest.raises(contract.ContractError) as exc:
            contract.validate_job({
                "op": op, "input_key": "in/x.bin", "output_key": "out/x.bin",
                "preview": True, "params": {},
            })
        assert "preview" in str(exc.value)


def test_the_preview_survives_the_output_filter():
    kept = contract.filter_output({"ok": True, "preview_frames": [{"i": 0, "b64": "AA"}]})
    assert kept["preview_frames"] == [{"i": 0, "b64": "AA"}]


# ------------------------------------- the harness side: saved and printed


def test_the_harness_writes_and_prints_what_came_back(tmp_path, capsys):
    from validation import spend_run

    written = spend_run.save_previews(
        {"preview_frames": [{"i": 0, "b64": base64.b64encode(b"\xff\xd8ab").decode()},
                            {"i": 48, "b64": base64.b64encode(b"\xff\xd8cd").decode()}]},
        "validation/out/probe-cogvideox-i2v.mp4",
        out_dir=str(tmp_path),
    )
    assert len(written) == 2
    assert written[0].endswith("probe-cogvideox-i2v.preview-0000.jpg")
    assert written[1].endswith("probe-cogvideox-i2v.preview-0048.jpg")
    out = capsys.readouterr().out
    assert "=== PREVIEW b64 probe-cogvideox-i2v.preview-0000.jpg 4 ===" in out
    assert "2 frame(s) returned by the worker" in out


def test_a_reply_with_no_preview_writes_nothing(tmp_path):
    from validation import spend_run

    assert spend_run.save_previews({}, "out/x.mp4", out_dir=str(tmp_path)) == []
    assert spend_run.save_previews(None, "out/x.mp4", out_dir=str(tmp_path)) == []
    assert list(tmp_path.iterdir()) == []


def test_undecodable_base64_is_reported_and_skipped(tmp_path, capsys):
    from validation import spend_run

    written = spend_run.save_previews(
        {"preview_frames": [{"i": 0, "b64": "!!!not base64!!!"},
                            {"i": 1, "b64": base64.b64encode(b"ok").decode()}]},
        "out/x.mp4", out_dir=str(tmp_path),
    )
    assert len(written) == 1
    assert "could not be decoded" in capsys.readouterr().out


def test_the_status_dump_elides_the_base64_rather_than_burying_the_numbers():
    """The status payload is printed twice. A megabyte of base64 through it
    would bury every measurement beside it."""
    from validation import spend_run

    shown = spend_run.redact({"preview_frames": [{"i": 0, "b64": "A" * 5000}],
                              "vram_peak_mb": 13837})
    assert shown["preview_frames"][0]["b64"] == "<5000 base64 chars, printed once below>"
    assert shown["vram_peak_mb"] == 13837


def test_the_benchmark_asks_for_a_preview_and_production_does_not():
    import inspect

    from validation import spend_run

    src = inspect.getsource(spend_run.main)
    # The probe and the drawn reference both ask; nothing else does.
    assert src.count("preview=True") == 2
    assert 'payload["preview"] = True' in inspect.getsource(spend_run.one_job)


def test_preview_is_a_flag_not_anything_truthy():
    """Every other field in this contract is refused rather than guessed at."""
    with pytest.raises(contract.ContractError) as exc:
        contract.validate_job({
            "op": "image_generate", "output_key": "out/x.png",
            "preview": "yes", "params": {"prompt": "a woman"},
        })
    assert "preview must be true or false" in str(exc.value)


# --------------------- the field must SURVIVE validation, not just pass it


def test_the_preview_flag_reaches_the_worker_not_just_the_validator():
    """Every branch of validate_job builds an explicit dict, so a field that
    is checked above and left out below is accepted and then silently
    discarded. That is exactly what happened on 2026-08-29: the flag was
    validated, the job ran, the worker never saw it, and the reference came
    back invisible — one dispatch to find out."""
    job = contract.validate_job({
        "op": "image_generate", "output_key": "out/probe-reference.png",
        "preview": True, "params": {"prompt": "one adult woman, plain background"},
    })
    assert job["preview"] is True
    assert preview.wanted(job) is True


def test_the_probe_job_carries_it_too():
    job = contract.validate_job({
        "op": "model_probe", "model": "cogvideox-i2v",
        "input_key": "out/probe-reference.png", "output_key": "out/p.mp4",
        "preview": True, "params": {"prompt": "she turns toward the camera"},
    })
    assert job["preview"] is True


def test_a_job_that_did_not_ask_carries_false_not_a_missing_key():
    """Production never sends the flag. The worker must read a definite
    False rather than guessing from an absent key."""
    job = contract.validate_job({
        "op": "image_generate", "output_key": "out/x.png",
        "params": {"prompt": "a still"},
    })
    assert job["preview"] is False
    assert preview.wanted(job) is False


def test_every_op_that_admits_the_flag_also_returns_it():
    """Walked rather than spot-checked: the bug was one branch out of seven
    that admitted the field and dropped it."""
    admitting = []
    for op, extra in (
        ("image_generate", {"output_key": "o/x.png",
                            "params": {"prompt": "p"}}),
        ("model_probe", {"model": "cogvideox-i2v", "input_key": "i/r.png",
                         "output_key": "o/x.mp4", "params": {"prompt": "p"}}),
    ):
        job = contract.validate_job({"op": op, "preview": True, **extra})
        admitting.append(op)
        assert job.get("preview") is True, op
    assert admitting == ["image_generate", "model_probe"]
