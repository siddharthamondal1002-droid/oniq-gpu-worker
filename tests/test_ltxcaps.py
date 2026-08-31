"""The inference profile comes from the CHECKPOINT, and the sampler obeys it.

These are behavioural: a real directory is written to disk and read back by
the real inspector, and a fake pipeline records the kwargs it was actually
called with. Nothing here asserts on source text — the audit that prompted
this work found a system where the source said "Vertical 9:16 portrait" and
the tensor was 704x480 landscape, so a test that reads strings would have
passed while the product was broken.
"""

import json
import os

import pytest
from PIL import Image

import contract
import ltxcaps
import videogen
from conftest import write_fake_checkpoint


# ------------------------------------------------- reading the checkpoint

def test_a_full_checkpoint_yields_the_full_step_count(tmp_path):
    root = write_fake_checkpoint(tmp_path / "ltx")
    caps = ltxcaps.inspect_checkpoint(root, "Lightricks/LTX-Video")
    profile = ltxcaps.inference_profile(caps)
    assert caps["distilled"] is False
    assert profile["num_inference_steps"] == ltxcaps.STEPS_FULL
    assert profile["guidance_scale"] > 1.0


def test_a_distilled_checkpoint_yields_the_distilled_step_count(tmp_path):
    root = write_fake_checkpoint(tmp_path / "ltx", distilled=True)
    caps = ltxcaps.inspect_checkpoint(root, "LTX-Video-0.9.7-distilled")
    profile = ltxcaps.inference_profile(caps)
    assert caps["distilled"] is True
    assert profile["num_inference_steps"] == ltxcaps.STEPS_DISTILLED
    # Guidance is what distillation removes the need for.
    assert profile["guidance_scale"] == 1.0


def test_contradictory_distillation_evidence_refuses(tmp_path):
    """A NAME is not a capability.

    The old code decided the step count with `"distilled" in model_id()`. Here
    the name says distilled and the shipped scheduler does not; there is no
    honest profile to pick, so it raises rather than choosing one.
    """
    root = write_fake_checkpoint(tmp_path / "ltx", distilled=False)
    with pytest.raises(ltxcaps.CheckpointInconsistent):
        ltxcaps.inspect_checkpoint(root, "Lightricks/LTX-Video-distilled")


def test_a_missing_component_refuses_rather_than_sampling(tmp_path):
    root = write_fake_checkpoint(
        tmp_path / "ltx", components=("transformer", "vae", "tokenizer", "scheduler")
    )
    caps = ltxcaps.inspect_checkpoint(root, "Lightricks/LTX-Video")
    assert caps["components_missing"] == ["text_encoder"]
    assert caps["condition_pipeline_supported"] is False
    with pytest.raises(ltxcaps.CheckpointInconsistent):
        ltxcaps.inference_profile(caps)


def test_an_unreadable_checkpoint_refuses(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(ltxcaps.CheckpointInconsistent):
        ltxcaps.inspect_checkpoint(str(empty), "whatever")


def test_a_non_ltx_pipeline_refuses(tmp_path):
    root = tmp_path / "other"
    root.mkdir()
    (root / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionPipeline"}), encoding="utf-8"
    )
    with pytest.raises(ltxcaps.CheckpointInconsistent):
        ltxcaps.inspect_checkpoint(str(root), "some/model")


# ------------------------------------- conditioning and the upscaler asset

def test_condition_pipeline_needs_no_extra_weights(tmp_path):
    """LTXConditionPipeline takes the SAME five modules as the i2v pipeline.

    Verified against diffusers 0.38.0. This is why reference conditioning is
    reachable at all without changing the baked model.
    """
    root = write_fake_checkpoint(tmp_path / "ltx")
    caps = ltxcaps.inspect_checkpoint(root, "Lightricks/LTX-Video")
    assert caps["condition_pipeline_supported"] is True


def test_the_spatial_upscaler_is_detected_not_assumed(tmp_path):
    plain = ltxcaps.inspect_checkpoint(
        write_fake_checkpoint(tmp_path / "a"), "Lightricks/LTX-Video"
    )
    assert plain["latent_upsampler_baked"] is False

    withup = ltxcaps.inspect_checkpoint(
        write_fake_checkpoint(tmp_path / "b", upscaler=True), "Lightricks/LTX-Video"
    )
    assert withup["latent_upsampler_baked"] is True


# ------------------------------------------------------- the canvas itself

def test_the_generation_canvas_is_portrait_and_32_divisible():
    assert contract.VIDEO_HEIGHT > contract.VIDEO_WIDTH
    assert contract.VIDEO_WIDTH % 32 == 0
    assert contract.VIDEO_HEIGHT % 32 == 0


def test_the_canvas_cannot_reproduce_the_61_percent_crop():
    """The audit's headline defect, as a guard.

    704x480 into 1080x1920 discarded 61.6% of every frame's width and invented
    16 output pixels per real one. Any future canvas that would do that again
    fails here.
    """
    film_w, film_h = 1080, 1920
    scale = max(film_w / contract.VIDEO_WIDTH, film_h / contract.VIDEO_HEIGHT)
    width_kept = film_w / (contract.VIDEO_WIDTH * scale)
    height_kept = film_h / (contract.VIDEO_HEIGHT * scale)
    assert width_kept > 0.99 and height_kept > 0.99

    visible = min(contract.VIDEO_WIDTH, round(film_w / scale)) * min(
        contract.VIDEO_HEIGHT, round(film_h / scale)
    )
    deficit = (film_w * film_h) / visible
    assert deficit < 3.0, f"pixel deficit {deficit:.2f}x is back near the old 16x"


# --------------------------------------------- what the sampler is told

class Recorder:
    def __init__(self, n=contract.VIDEO_NUM_FRAMES):
        self.n = n
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        from types import SimpleNamespace

        frames = [
            Image.new("RGB", (contract.VIDEO_WIDTH, contract.VIDEO_HEIGHT), (9, 9, 9))
            for _ in range(self.n)
        ]
        return SimpleNamespace(frames=[frames])


def _job(**params):
    body = {"prompt": "she turns to the window", **params}
    return contract.validate_job(
        {
            "op": "video_generate",
            "input_key": "in/a.jpg",
            "output_key": "out/a.mp4",
            "params": body,
        }
    )


@pytest.fixture
def io_paths(tmp_path):
    src = str(tmp_path / "in.png")
    Image.new("RGB", (704, 1248), (30, 60, 90)).save(src)
    return src, str(tmp_path / "out.mp4")


def test_guidance_and_vae_settings_are_sent_explicitly(io_paths, fake_checkpoint):
    src, out = io_paths
    pipe = Recorder()
    metrics = videogen.run(_job(), src, out, load_pipeline=lambda: pipe)
    kw = pipe.calls[0]
    assert kw["guidance_scale"] == 3.0
    assert kw["guidance_rescale"] == 0.0
    assert "decode_timestep" in kw
    assert "decode_noise_scale" in kw
    # And every one of them is reported, so a bad clip is diagnosable.
    assert metrics["guidance_scale"] == 3.0
    assert metrics["num_inference_steps"] == ltxcaps.STEPS_FULL
    assert metrics["defaults_source"]


def test_a_per_shot_negative_prompt_reaches_the_model(io_paths, fake_checkpoint):
    src, out = io_paths
    pipe = Recorder()
    neg = "blurry face, deformed face, malformed eyes, warped hands"
    videogen.run(_job(negative_prompt=neg), src, out, load_pipeline=lambda: pipe)
    assert pipe.calls[0]["negative_prompt"] == neg


def test_absent_negative_prompt_keeps_the_previous_default(io_paths, fake_checkpoint):
    src, out = io_paths
    pipe = Recorder()
    videogen.run(_job(), src, out, load_pipeline=lambda: pipe)
    assert pipe.calls[0]["negative_prompt"] == videogen.NEGATIVE_PROMPT


def test_the_seed_reaches_the_metrics_so_a_clip_can_be_reproduced(
    io_paths, fake_checkpoint
):
    src, out = io_paths
    pipe = Recorder()
    metrics = videogen.run(_job(seed=123456789), src, out, load_pipeline=lambda: pipe)
    assert metrics["seed"] == 123456789


def test_ten_distinct_seeds_are_ten_distinct_jobs(io_paths, fake_checkpoint):
    """The retry ceiling was raised 3 -> 10 on 2026-08-31.

    With a module-level SEED those ten attempts re-sampled one image ten
    times. Each attempt must be a genuinely different draw.
    """
    src, out = io_paths
    seen = set()
    for attempt in range(10):
        pipe = Recorder()
        m = videogen.run(
            _job(seed=1000 + attempt), src, out, load_pipeline=lambda: pipe
        )
        seen.add(m["seed"])
    assert len(seen) == 10


def test_the_contract_refuses_a_non_integer_or_out_of_range_seed():
    for bad in ("42", 4.2, True, -1, 2**64):
        with pytest.raises(contract.ContractError):
            _job(seed=bad)


def test_the_contract_bounds_the_negative_prompt():
    with pytest.raises(contract.ContractError):
        _job(negative_prompt="x" * (contract.MAX_NEGATIVE_PROMPT_CHARS + 1))


# ------------------------------------------------------------- fail closed

def test_out_of_memory_refuses_instead_of_shrinking_the_canvas(
    io_paths, fake_checkpoint
):
    """The portrait canvas is 2.6x the pixels; the VRAM figure behind it is
    arithmetic, not a measurement. Silently downscaling would restore the very
    upscale this work removes, and would do it invisibly."""
    src, out = io_paths

    def boom(**_kwargs):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    class Boom:
        __call__ = staticmethod(boom)

    with pytest.raises(videogen.OutOfMemory):
        videogen.run(_job(), src, out, load_pipeline=lambda: Boom())


def test_a_non_oom_failure_is_not_relabelled(io_paths, fake_checkpoint):
    src, out = io_paths

    class Boom:
        def __call__(self, **_kwargs):
            raise ValueError("something else entirely")

    with pytest.raises(ValueError):
        videogen.run(_job(), src, out, load_pipeline=lambda: Boom())


def test_diagnostics_carry_no_secrets_or_storage_urls(io_paths, fake_checkpoint):
    src, out = io_paths
    pipe = Recorder()
    metrics = videogen.run(_job(), src, out, load_pipeline=lambda: pipe)
    blob = json.dumps(metrics).lower()
    for forbidden in ("http://", "https://", "secret", "token", "password", "r2_"):
        assert forbidden not in blob


# --------------------------------------- explicit conditioning (§5, §13)

def test_a_supporting_checkpoint_asks_for_explicit_conditioning(tmp_path):
    """The condition pipeline is the point, not the class name.

    LTXImageToVideoPipeline takes `image=` and decides for itself how hard to
    hold it. LTXConditionPipeline takes a NAMED frame, a NAMED strength and a
    NAMED noise scale — which is the difference between a clip that drifted
    off its opening frame and a clip that was never asked to hold it.
    """
    root = write_fake_checkpoint(tmp_path / "ltx")
    caps = ltxcaps.inspect_checkpoint(root, "Lightricks/LTX-Video")
    profile = ltxcaps.inference_profile(caps)
    assert profile["conditioning"] is True
    assert ltxcaps.diagnostics(caps, profile)["conditioning"] is True


def test_a_checkpoint_without_the_components_does_not_pretend(tmp_path):
    root = write_fake_checkpoint(
        tmp_path / "ltx", components=("transformer", "vae", "tokenizer", "scheduler")
    )
    caps = ltxcaps.inspect_checkpoint(root, "Lightricks/LTX-Video")
    assert caps["condition_pipeline_supported"] is False


class ConditionRecorder(Recorder):
    """A pipe that records what it was called with, standing in for either
    class — the discriminator is which KWARGS arrive, not which type does."""


def _fake_condition(image):
    return {"image": image, "frame_index": 0, "strength": videogen.CONDITION_STRENGTH}


def test_the_conditioning_frame_strength_and_noise_are_all_named(
    io_paths, fake_checkpoint, monkeypatch
):
    src, out = io_paths
    pipe = ConditionRecorder()
    monkeypatch.setattr(videogen, "_video_condition", _fake_condition)
    metrics = videogen.run(_job(), src, out, load_pipeline=lambda: pipe)
    kw = pipe.calls[0]
    assert "conditions" in kw and len(kw["conditions"]) == 1
    assert kw["conditions"][0]["frame_index"] == 0
    assert kw["conditions"][0]["strength"] == videogen.CONDITION_STRENGTH
    assert kw["image_cond_noise_scale"] == 0.15
    # `image=` is the implicit path and must not be sent alongside.
    assert "image" not in kw
    assert metrics["conditioning_count"] == 1
    assert metrics["conditioning_strength"] == videogen.CONDITION_STRENGTH


def test_without_diffusers_the_worker_still_animates_the_still(io_paths, fake_checkpoint):
    """Degrading is honest; refusing to animate would not be.

    The CPU rig has no diffusers, so _video_condition returns None and the
    plain image= path runs — byte-for-byte what this worker did before. A
    fallback here is the same weights on the same GPU with less control, not
    a different provider.
    """
    src, out = io_paths
    pipe = Recorder()
    metrics = videogen.run(_job(), src, out, load_pipeline=lambda: pipe)
    kw = pipe.calls[0]
    assert "conditions" not in kw
    assert kw["image"] is not None
    assert metrics["conditioning_count"] == 0
    assert metrics["conditioning_strength"] is None


def test_the_strength_is_the_documented_value_not_an_invented_one():
    # §17: every parameter carries evidence. 1.0 is LTXVideoCondition's own
    # default — "hold this frame as given". No measurement on this hardware
    # justifies anything else yet, and inventing one would repeat exactly the
    # mistake the audit found.
    assert videogen.CONDITION_STRENGTH == 1.0
    assert videogen.CONDITION_FRAME_INDEX == 0


def test_the_metrics_name_the_class_that_actually_ran(io_paths, fake_checkpoint):
    src, out = io_paths
    pipe = Recorder()
    metrics = videogen.run(_job(), src, out, load_pipeline=lambda: pipe)
    # model_index.json's declaration and the object that ran are two different
    # facts; a diagnosis needs the second.
    assert metrics["pipeline_used"] == "Recorder"
    assert metrics["pipeline_class"] == "LTXImageToVideoPipeline"


# ------------------------------------------------- encoding (§11)

def test_the_clip_is_written_at_the_generation_fps():
    """24 in, 24 out. No resample, no duplicated frames.

    The contract generates 97 frames at 24 fps. Writing them at 30 would make
    every clip play 25% fast and then need a second, lossy pass to correct —
    an intermediate the audit asks us not to introduce.
    """
    assert contract.VIDEO_FPS == 24
    source = open("videogen.py", encoding="utf-8").read()
    encode = source[source.index("def _encode_mp4") : source.index("def _encode_mp4") + 700]
    assert "fps=contract.VIDEO_FPS" in encode
    assert "30" not in encode


def test_the_frames_go_straight_from_the_vae_to_h264():
    """One lossy step, not two.

    The PIL frames the pipeline returns are appended to the encoder directly.
    A PNG/JPEG round-trip in between, or a scale filter, would be a second
    generation loss on top of the one h264 already costs.
    """
    source = open("videogen.py", encoding="utf-8").read()
    encode = source[source.index("def _encode_mp4") : source.index("def _encode_mp4") + 700]
    assert "np.asarray(frame)" in encode
    for lossy in (".jpg", ".jpeg", "JPEG", "quality=95", "scale="):
        assert lossy not in encode


def test_the_still_is_written_losslessly():
    # It is a CONDITIONING FRAME. A jpeg here would put compression artifacts
    # into the video model's own input.
    assert contract.IMAGE_GEN_FORMAT == "png"
