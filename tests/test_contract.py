import pytest

import contract


def _job(**overrides):
    base = {
        "op": "image_preprocess",
        "input_key": "in/test.png",
        "output_key": "out/test.jpg",
    }
    base.update(overrides)
    return base


def test_valid_job_normalizes_defaults():
    job = contract.validate_job(_job())
    assert job["op"] == "image_preprocess"
    assert job["params"] == {
        "target_max_dim": contract.DEFAULT_TARGET_DIM,
        "format": "jpeg",
        "quality": contract.DEFAULT_QUALITY,
    }


def test_non_dict_input_refused():
    with pytest.raises(contract.ContractError) as exc:
        contract.validate_job(None)
    assert exc.value.code == "invalid-input"


def test_unknown_op_refused():
    with pytest.raises(contract.ContractError) as exc:
        contract.validate_job(_job(op="train_model"))
    assert exc.value.code == "op-not-allowed"


def test_no_model_field_exists_in_contract():
    with pytest.raises(contract.ContractError) as exc:
        contract.validate_job(_job(model="anything"))
    assert exc.value.code == "invalid-input"
    assert "model" in exc.value.message


def test_no_production_op_accepts_a_model_field():
    """model_probe admits `model`; nothing a user can reach does.

    Widened 2026-08-29 when the benchmark op arrived. The original guard used
    one production op, and adding "model" to the shared field set would have
    retired it for every op at once — so the guard now walks them all, and the
    exception is one op that no user request can produce."""
    for op, params in (
        ("image_preprocess", {"target_max_dim": 512, "format": "png"}),
        ("image_generate", {"prompt": "x"}),
        ("video_generate", {"prompt": "x"}),
        ("audio_mux", {"narration": "x"}),
    ):
        with pytest.raises(contract.ContractError) as exc:
            contract.validate_job({
                "op": op, "input_key": "a.png", "output_key": "b.mp4",
                "params": params, "model": "wan21-i2v-480p",
            })
        assert "model" in exc.value.message, op


def test_no_gpu_field_exists_in_contract():
    with pytest.raises(contract.ContractError) as exc:
        contract.validate_job(_job(gpu="H100"))
    assert exc.value.code == "invalid-input"


def test_missing_keys_refused():
    with pytest.raises(contract.ContractError):
        contract.validate_job({"op": "image_preprocess"})


def test_key_with_traversal_refused():
    with pytest.raises(contract.ContractError):
        contract.validate_job(_job(input_key="a/../../etc/passwd"))


def test_key_with_leading_slash_refused():
    with pytest.raises(contract.ContractError):
        contract.validate_job(_job(output_key="/absolute/path"))


def test_key_too_long_refused():
    with pytest.raises(contract.ContractError):
        contract.validate_job(_job(input_key="a" * 513))


def test_unknown_param_refused():
    with pytest.raises(contract.ContractError):
        contract.validate_job(_job(params={"steps": 50}))


def test_target_dim_out_of_bounds_refused():
    with pytest.raises(contract.ContractError):
        contract.validate_job(_job(params={"target_max_dim": contract.MAX_TARGET_DIM + 1}))
    with pytest.raises(contract.ContractError):
        contract.validate_job(_job(params={"target_max_dim": contract.MIN_TARGET_DIM - 1}))


def test_bool_is_not_an_int_for_bounds():
    with pytest.raises(contract.ContractError):
        contract.validate_job(_job(params={"target_max_dim": True}))


def test_format_whitelist():
    with pytest.raises(contract.ContractError):
        contract.validate_job(_job(params={"format": "tiff"}))
    job = contract.validate_job(_job(params={"format": "webp"}))
    assert job["params"]["format"] == "webp"


def test_quality_bounds():
    with pytest.raises(contract.ContractError):
        contract.validate_job(_job(params={"quality": 0}))
    with pytest.raises(contract.ContractError):
        contract.validate_job(_job(params={"quality": 101}))


def test_runtime_ceiling_is_900():
    assert contract.RUNTIME_CEILING_SECONDS == 900


def test_input_byte_bound_is_positive_and_finite():
    assert 0 < contract.MAX_INPUT_BYTES <= 64 * 1024 * 1024


def _video_job(**overrides):
    base = {
        "op": "video_generate",
        "input_key": "in/face.jpg",
        "output_key": "out/clip.mp4",
        "params": {"prompt": "  the subject turns toward the camera  "},
    }
    base.update(overrides)
    return base


def test_video_job_normalizes_to_prompt_and_watermark():
    job = contract.validate_job(_video_job())
    assert job["op"] == "video_generate"
    # Absent watermark normalizes to TRUE — the fail-safe: an old caller
    # can only ever produce the watermarked product.
    assert job["params"] == {
        "prompt": "the subject turns toward the camera",
        "watermark": True,
    }


def test_video_watermark_is_a_strict_boolean():
    clean = contract.validate_job(
        _video_job(params={"prompt": "ok", "watermark": False})
    )
    assert clean["params"]["watermark"] is False
    marked = contract.validate_job(
        _video_job(params={"prompt": "ok", "watermark": True})
    )
    assert marked["params"]["watermark"] is True
    # A present non-bool is a malformed contract: refused, never guessed.
    for bad in (1, 0, "false", "true", None, [], {}):
        with pytest.raises(contract.ContractError) as exc:
            contract.validate_job(
                _video_job(params={"prompt": "ok", "watermark": bad})
            )
        assert exc.value.code == "invalid-input"


def test_video_job_requires_a_prompt():
    with pytest.raises(contract.ContractError) as exc:
        contract.validate_job(_video_job(params={}))
    assert exc.value.code == "invalid-input"
    with pytest.raises(contract.ContractError):
        contract.validate_job(_video_job(params={"prompt": "   "}))
    with pytest.raises(contract.ContractError):
        contract.validate_job(_video_job(params={"prompt": 42}))


def test_video_prompt_is_bounded():
    long = "x" * (contract.MAX_PROMPT_CHARS + 1)
    with pytest.raises(contract.ContractError) as exc:
        contract.validate_job(_video_job(params={"prompt": long}))
    assert exc.value.code == "invalid-input"
    ok = contract.validate_job(
        _video_job(params={"prompt": "y" * contract.MAX_PROMPT_CHARS})
    )
    assert len(ok["params"]["prompt"]) == contract.MAX_PROMPT_CHARS


def test_video_job_refuses_every_knob_but_the_prompt():
    # Resolution, length, steps, model — all server decisions. A caller
    # naming any of them is refused, not silently ignored.
    for knob in ("width", "num_frames", "steps", "model", "target_max_dim"):
        with pytest.raises(contract.ContractError) as exc:
            contract.validate_job(
                _video_job(params={"prompt": "ok", knob: 1})
            )
        assert exc.value.code == "invalid-input"
        assert knob in exc.value.message


def test_video_constants_are_the_recorded_server_decisions():
    assert (contract.VIDEO_WIDTH, contract.VIDEO_HEIGHT) == (704, 1248)
    assert contract.VIDEO_WIDTH % 32 == 0 and contract.VIDEO_HEIGHT % 32 == 0
    assert contract.VIDEO_NUM_FRAMES % 8 == 1  # LTX's 8k+1 rule
    assert contract.VIDEO_FPS == 24


def test_the_canvas_is_portrait_and_matches_the_film():
    """The audit's primary finding, as an executable guard.

    A landscape canvas feeding a 1080x1920 film cost a 4.00x upscale and threw
    away 61.6% of every frame's width. This asserts the canvas is portrait AND
    that its aspect is close enough to the film's that the assembly's
    cover-crop removes almost nothing.
    """
    assert contract.VIDEO_HEIGHT > contract.VIDEO_WIDTH, "canvas must be portrait"
    film_w, film_h = 1080, 1920
    canvas = contract.VIDEO_WIDTH / contract.VIDEO_HEIGHT
    film = film_w / film_h
    assert abs(canvas - film) < 0.01, f"aspect {canvas:.4f} vs film {film:.4f}"

    # The crop the assembly will actually perform.
    scale = max(film_w / contract.VIDEO_WIDTH, film_h / contract.VIDEO_HEIGHT)
    kept = film_w / (contract.VIDEO_WIDTH * scale)
    assert kept > 0.99, f"cover-crop keeps only {kept:.1%} of the width"

    # And the pixel deficit, which was 16.00x before this change.
    visible = min(contract.VIDEO_WIDTH, round(film_w / scale)) * min(
        contract.VIDEO_HEIGHT, round(film_h / scale)
    )
    assert (film_w * film_h) / visible < 3.0


def test_video_evidence_fields_are_whitelisted():
    assert {
        "model",
        "model_load_ms",
        "inference_ms",
        "encode_ms",
        "frames",
        "fps",
        "video_seconds",
    } <= contract.OUTPUT_WHITELIST


def test_filter_output_drops_everything_not_whitelisted():
    filtered = contract.filter_output(
        {"ok": True, "device": "cuda", "aws_secret": "LEAK", "env": {"x": 1}}
    )
    assert filtered == {"ok": True, "device": "cuda"}


def test_output_whitelist_is_explicit_and_closed():
    assert "aws_secret" not in contract.OUTPUT_WHITELIST
    assert {"ok", "code", "error", "device", "gpu_name"} <= contract.OUTPUT_WHITELIST


# ------------------------------------------------------------ video_concat


def _concat_job(**overrides):
    segments = ["clips/a.mp4", "clips/b.mp4", "clips/c.mp4"]
    base = {
        "op": "video_concat",
        "input_key": segments[0],
        "output_key": "films/final.mp4",
        "params": {"segment_keys": list(segments)},
    }
    base.update(overrides)
    return base


def test_concat_job_normalizes_ordered_segments():
    job = contract.validate_job(_concat_job())
    assert job["op"] == "video_concat"
    assert job["params"] == {
        "segment_keys": ["clips/a.mp4", "clips/b.mp4", "clips/c.mp4"]
    }


def test_concat_segment_count_is_bounded_both_ways():
    with pytest.raises(contract.ContractError):
        contract.validate_job(
            _concat_job(
                input_key="clips/a.mp4",
                params={"segment_keys": ["clips/a.mp4"]},
            )
        )
    too_many = [f"clips/{i}.mp4" for i in range(contract.MAX_CONCAT_SEGMENTS + 1)]
    with pytest.raises(contract.ContractError):
        contract.validate_job(
            _concat_job(input_key=too_many[0], params={"segment_keys": too_many})
        )
    at_cap = [f"clips/{i}.mp4" for i in range(contract.MAX_CONCAT_SEGMENTS)]
    ok = contract.validate_job(
        _concat_job(input_key=at_cap[0], params={"segment_keys": at_cap})
    )
    assert len(ok["params"]["segment_keys"]) == contract.MAX_CONCAT_SEGMENTS


def test_concat_refuses_repeats_bad_keys_and_mismatched_input():
    with pytest.raises(contract.ContractError):
        contract.validate_job(
            _concat_job(params={"segment_keys": ["clips/a.mp4", "clips/a.mp4"]})
        )
    with pytest.raises(contract.ContractError):
        contract.validate_job(
            _concat_job(params={"segment_keys": ["clips/a.mp4", "../etc/x"]})
        )
    with pytest.raises(contract.ContractError) as exc:
        contract.validate_job(_concat_job(input_key="clips/b.mp4"))
    assert "segment_keys[0]" in exc.value.message


def test_concat_refuses_every_extra_knob():
    for knob in ("watermark", "prompt", "format", "fps"):
        with pytest.raises(contract.ContractError):
            contract.validate_job(
                _concat_job(
                    params={
                        "segment_keys": ["clips/a.mp4", "clips/b.mp4"],
                        knob: 1,
                    },
                    input_key="clips/a.mp4",
                )
            )


def test_concat_evidence_fields_are_whitelisted():
    assert {"segments", "concat_ms", "watermarked"} <= contract.OUTPUT_WHITELIST


# ------------------------------------------------- image_generate (in-house)
# ONIQ's own image engine (fully in-house directive, 2026-08-27). The op that
# replaced an outsourced image API, so its contract is pinned as tightly as
# the video one: text-only, prompt-bounded, and no watermark field at all.


def _image_job(**overrides):
    job = {"op": "image_generate", "output_key": "out/still.png",
           "params": {"prompt": "a lantern in the rain"}}
    job.update(overrides)
    return job


def test_image_generate_is_allowed_and_normalizes():
    out = contract.validate_job(_image_job())
    assert out["op"] == "image_generate"
    assert out["input_key"] is None
    assert out["params"] == {"prompt": "a lantern in the rain"}


def test_image_generate_refuses_an_input_key():
    # Refused, never ignored: a caller that believes it is conditioning on
    # an image must never be told silently that it was.
    with pytest.raises(contract.ContractError) as exc:
        contract.validate_job(_image_job(input_key="in/frame.png"))
    assert exc.value.code == "invalid-input"


def test_image_generate_requires_a_prompt():
    for bad in (None, "", "   ", 7):
        with pytest.raises(contract.ContractError):
            contract.validate_job(_image_job(params={"prompt": bad}))


def test_image_generate_bounds_the_prompt():
    long_prompt = "x" * (contract.MAX_PROMPT_CHARS + 1)
    with pytest.raises(contract.ContractError) as exc:
        contract.validate_job(_image_job(params={"prompt": long_prompt}))
    assert exc.value.code == "invalid-input"


def test_image_generate_takes_no_watermark_field():
    # A conditioning frame is an intermediate. The mark belongs to the
    # delivered film, burned by the stage that knows the entitlement.
    assert contract._IMAGE_GEN_PARAM_FIELDS == frozenset(
        {"prompt", "seed", "negative_prompt"}
    )
    assert "watermark" not in contract._IMAGE_GEN_PARAM_FIELDS
    with pytest.raises(contract.ContractError):
        contract.validate_job(
            _image_job(params={"prompt": "ok", "watermark": False})
        )


def test_image_generate_canvas_matches_the_video_canvas():
    # The still exists to be animated; a mismatched canvas would be
    # rescaled at the seam between the two stages.
    assert contract.IMAGE_GEN_NUM_FRAMES % 8 == 1
    assert contract.IMAGE_GEN_FORMAT in contract.ALLOWED_FORMATS
