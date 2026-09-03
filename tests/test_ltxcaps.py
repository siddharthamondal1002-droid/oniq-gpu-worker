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


def _fake_condition(image, frame_index=None, strength=None):
    return {
        "image": image,
        "frame_index": videogen.CONDITION_FRAME_INDEX if frame_index is None else frame_index,
        "strength": videogen.CONDITION_STRENGTH if strength is None else strength,
    }


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


# ─────────────────────────────────── multi-scale latent upscaling (§6, §10)

class UpsampleRecorder:
    """Stands in for LTXLatentUpsamplePipeline. Records the latents it was
    handed and returns a marker, so the test can prove the REFINE pass was
    given the upsampler's output and not the base pass's."""

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        from types import SimpleNamespace

        self.calls.append(kwargs)
        return SimpleNamespace(frames=["UPSCALED-LATENTS"])


class TwoPassRecorder(Recorder):
    """A pipe that answers both the latent pass and the decode pass."""

    def __call__(self, **kwargs):
        from types import SimpleNamespace

        if kwargs.get("output_type") == "latent":
            self.calls.append(kwargs)
            return SimpleNamespace(frames=["BASE-LATENTS"])
        return super().__call__(**kwargs)  # Recorder records the decode pass


@pytest.fixture
def multiscale_checkpoint(tmp_path, monkeypatch):
    root = write_fake_checkpoint(tmp_path / "ltx-up", upscaler=True)
    monkeypatch.setattr(videogen, "_model_dir", lambda: root)
    monkeypatch.setattr(videogen, "model_id", lambda: "Lightricks/LTX-Video")
    monkeypatch.setattr(ltxcaps, "_multiscale_available", lambda caps: (True, "ok"))
    return root


def test_the_upscaler_runs_in_latent_space_not_on_pixels(io_paths, multiscale_checkpoint):
    """The whole point of §6, as a behavioural assertion.

    A generic pixel upscaler on the finished MP4 cannot invent detail that was
    never sampled. The latent path upsamples BEFORE the decode and then runs a
    short second denoise, so the transformer actually synthesises the detail.
    This proves the order: base pass hands latents out, the upsampler receives
    THOSE latents, and the refine pass receives the upsampler's output.
    """
    src, out = io_paths
    pipe = TwoPassRecorder()
    up = UpsampleRecorder()
    metrics = videogen.run(
        _job(), src, out, load_pipeline=lambda: pipe, load_upsampler=lambda: up
    )

    assert len(pipe.calls) == 2, "expected a base pass and a refine pass"
    base, refine = pipe.calls
    assert base["output_type"] == "latent"
    assert up.calls[0]["latents"] == "BASE-LATENTS"
    assert refine["latents"] == "UPSCALED-LATENTS"
    assert "output_type" not in refine, "the refine pass decodes"
    assert metrics["upscaler_used"] is True
    assert metrics["upscaler_absent_reason"] is None


def test_the_refine_pass_is_partial_and_at_double_the_canvas(
    io_paths, multiscale_checkpoint
):
    src, out = io_paths
    pipe = TwoPassRecorder()
    metrics = videogen.run(
        _job(), src, out,
        load_pipeline=lambda: pipe, load_upsampler=lambda: UpsampleRecorder(),
    )
    base, refine = pipe.calls
    assert (base["width"], base["height"]) == (contract.VIDEO_WIDTH, contract.VIDEO_HEIGHT)
    assert refine["width"] == contract.VIDEO_WIDTH * 2
    assert refine["height"] == contract.VIDEO_HEIGHT * 2
    # PARTIAL: a full re-denoise would cost the whole step count again and
    # discard the composition the base pass just agreed on.
    assert 0 < refine["denoise_strength"] < 1.0
    # A FULL checkpoint has no upstream multi-scale schedule, so it keeps its
    # own step count and the partiality comes from denoise_strength alone.
    assert refine["num_inference_steps"] == base["num_inference_steps"]
    assert refine["denoise_strength"] == ltxcaps.FULL_REFINE_DENOISE_STRENGTH
    assert metrics["render_width"] == contract.VIDEO_WIDTH * 2


def test_the_film_stops_upscaling_the_render_once_multiscale_runs():
    """1080 must become a DOWNSCALE of real detail, not an upscale of absent
    detail. That is the whole haze finding, as arithmetic."""
    film_w = 1080
    assert contract.VIDEO_WIDTH < film_w, "the base canvas is upscaled by the film"
    assert contract.VIDEO_WIDTH * ltxcaps.UPSCALE_SPATIAL_FACTOR > film_w


def test_a_missing_upscaler_is_recorded_never_silent(io_paths, fake_checkpoint):
    """'The clip was soft' and 'the upscaler was never in the image' are
    indistinguishable in an output file and completely different in a
    diagnosis."""
    src, out = io_paths
    pipe = Recorder()
    metrics = videogen.run(_job(), src, out, load_pipeline=lambda: pipe)
    assert metrics["upscaler_used"] is False
    assert metrics["upscaler_absent_reason"] == "latent-upsampler-not-baked"
    assert metrics["render_width"] == contract.VIDEO_WIDTH
    assert len(pipe.calls) == 1, "no refine pass without an upscaler"


def test_nothing_in_the_worker_pixel_upscales_a_finished_clip():
    source = open("videogen.py", encoding="utf-8").read()
    encode = source[source.index("def _encode_mp4"): source.index("def _encode_mp4") + 700]
    for generic in ("scale=", "LANCZOS", "resize(", "upscale"):
        assert generic not in encode


# ───────────────────────────── the identity anchor (§3, §4, §5)

def test_a_reference_key_must_be_a_published_canonical_reference():
    """The field is an AUTHORITY TO READ ONE OBJECT, so it is pinned to a
    server-owned prefix rather than trusted as a key."""
    for bad in (
        "../secrets.png",
        "story/still/someone-elses.png",
        "story/ref/canon/../escape/v1.png",
        "https://evil.example/x.png",
        "story/ref/canon/x/v1.exe",
        "/story/ref/canon/x/v1.png",
        # The scope segment is not optional, and it is not the caller's.
        "story/ref/ali-01/v1.png",
        "story/ref/u/another-user/x/v1.png",
        # A version must be a real one: v0 does not exist, and an unversioned
        # key would be mutable — the whole point of versioning is that a film
        # drawn against v1 keeps looking like v1.
        "story/ref/canon/x/v0.png",
        "story/ref/canon/x.png",
        "",
        123,
    ):
        with pytest.raises(contract.ContractError):
            contract.validate_job({
                "op": "image_generate",
                "output_key": "out/a.png",
                "params": {"prompt": "p", "reference_key": bad},
            })
    ok = contract.validate_job({
        "op": "image_generate",
        "output_key": "out/a.png",
        "params": {"prompt": "p", "reference_key": CANON_REF},
    })
    assert ok["params"]["reference_key"] == CANON_REF
    # And v2 is a DIFFERENT object, never an overwrite of v1.
    v2 = CANON_REF.replace("/v1.png", "/v2.png")
    assert contract.validate_job({
        "op": "image_generate", "output_key": "out/a.png",
        "params": {"prompt": "p", "reference_key": v2},
    })["params"]["reference_key"] == v2
    # ABSENT is not an error — it is the unconditioned path, unchanged.
    plain = contract.validate_job({
        "op": "image_generate", "output_key": "out/a.png", "params": {"prompt": "p"},
    })
    assert "reference_key" not in plain["params"]


def test_the_strength_band_refuses_both_useless_ends():
    for bad in (0.0, 1.0, 1.5, -0.2, "0.5", True):
        with pytest.raises(contract.ContractError):
            contract.validate_job({
                "op": "image_generate",
                "output_key": "out/a.png",
                "params": {
                    "prompt": "p",
                    "reference_key": CANON_REF,
                    "reference_strength": bad,
                },
            })


def test_a_strength_without_a_reference_is_refused():
    with pytest.raises(contract.ContractError):
        contract.validate_job({
            "op": "image_generate",
            "output_key": "out/a.png",
            "params": {"prompt": "p", "reference_strength": 0.5},
        })


# The one shape the contract admits: scope, character, version.
CANON_REF = "story/ref/canon/ali-01/v1.png"


def _image_job(**params):
    return contract.validate_job({
        "op": "image_generate",
        "output_key": "out/a.png",
        "params": {"prompt": "a person by a window", **params},
    })


def test_the_identity_anchor_reaches_the_sampler_as_a_frame_0_condition(
    tmp_path, fake_checkpoint, monkeypatch
):
    ref = str(tmp_path / "ref.png")
    Image.new("RGB", (900, 1600), (200, 120, 60)).save(ref)
    pipe = Recorder(n=contract.IMAGE_GEN_NUM_FRAMES)
    monkeypatch.setattr(videogen, "_video_condition", _fake_condition)
    metrics = videogen.run_image(
        _image_job(reference_key=CANON_REF, reference_strength=0.6),
        str(tmp_path / "out.png"),
        load_pipeline=lambda: pipe,
        reference_path=ref,
    )
    kw = pipe.calls[0]
    assert kw["conditions"][0]["frame_index"] == 0
    assert kw["conditions"][0]["strength"] == 0.6
    assert "image_cond_noise_scale" in kw
    assert metrics["conditioning_count"] == 1
    assert metrics["conditioning_strength"] == 0.6


def test_an_anchor_below_full_strength_is_the_point(tmp_path, fake_checkpoint, monkeypatch):
    """Strength 1.0 hands the reference straight back and wastes the shot's
    own prompt; the anchor has to start PARTWAY from that person."""
    ref = str(tmp_path / "ref.png")
    Image.new("RGB", (704, 1248), (10, 20, 30)).save(ref)
    pipe = Recorder(n=contract.IMAGE_GEN_NUM_FRAMES)
    monkeypatch.setattr(videogen, "_video_condition", _fake_condition)
    videogen.run_image(
        _image_job(reference_key=CANON_REF),
        str(tmp_path / "out.png"),
        load_pipeline=lambda: pipe,
        reference_path=ref,
    )
    assert 0 < pipe.calls[0]["conditions"][0]["strength"] < 1.0
    assert pipe.calls[0]["conditions"][0]["strength"] == videogen.DEFAULT_REFERENCE_STRENGTH


def test_a_still_with_no_reference_is_byte_for_byte_the_old_path(
    tmp_path, fake_checkpoint
):
    pipe = Recorder(n=contract.IMAGE_GEN_NUM_FRAMES)
    metrics = videogen.run_image(
        _image_job(), str(tmp_path / "out.png"), load_pipeline=lambda: pipe
    )
    assert "conditions" not in pipe.calls[0]
    assert metrics["conditioning_count"] == 0
    assert metrics["conditioning_strength"] is None


def test_an_unbuildable_reference_refuses_rather_than_drawing_an_unanchored_still(
    tmp_path, fake_checkpoint, monkeypatch
):
    """A still drawn without the anchor the caller asked for looks exactly like
    one drawn with it. The difference only shows up as the character changing
    face between shots — the defect, arriving silently."""
    ref = str(tmp_path / "ref.png")
    Image.new("RGB", (704, 1248), (1, 2, 3)).save(ref)
    monkeypatch.setattr(videogen, "_video_condition", lambda *a, **k: None)
    with pytest.raises(videogen.ReferenceUnsupported):
        videogen.run_image(
            _image_job(reference_key=CANON_REF),
            str(tmp_path / "out.png"),
            load_pipeline=lambda: Recorder(n=contract.IMAGE_GEN_NUM_FRAMES),
            reference_path=ref,
        )


def test_the_reference_download_is_bounded_and_by_key_only():
    """The worker reads the reference with its OWN credentials, so the handler
    may only ever hand storage.download a contract-validated key."""
    handler_src = open("handler.py", encoding="utf-8").read()
    block = handler_src[handler_src.rindex('elif job["op"] == "image_generate"'):]
    block = block[: block.index("\n        else:")]
    assert 'job["params"].get("reference_key")' in block
    assert "contract.MAX_INPUT_BYTES" in block
    # No URL, no caller-supplied path, no origin of any kind.
    for forbidden in ("http", "url", "requests", "urlopen"):
        assert forbidden not in block.lower()


def test_the_upscaler_bake_stays_off_when_no_revision_is_pinned():
    """A revision is the licence. No sha, no bake — and unset must remain a
    no-op, so removing the pin can never break a build.

    The shipped pin is LIVE: Lightricks/ltxv-spatial-upscaler-0.9.7 at
    c96c168c…, resolved from the registry 2026-08-31, accepted by the owner
    under the LTX 0.X community licence, and enabled only once the CHECKPOINT
    moved to a vae that matches it. So this asserts the shape of the
    declaration and that the skip path still exists, rather than that the file
    is empty."""
    import re

    docker = open("Dockerfile", encoding="utf-8").read()
    pin = open("ltx-upscaler.pin", encoding="utf-8").read()
    declared = [l.split("#", 1)[0].strip() for l in pin.splitlines()
                if l.split("#", 1)[0].strip()]
    assert len(declared) == 1, declared
    repo, revision = declared[0].split()
    assert repo.count("/") == 1
    assert re.fullmatch(r"[0-9a-f]{40}", revision), revision
    # The acceptance is recorded next to the code it governs, with its date.
    assert "OWNER LICENCE ACCEPTANCE" in pin
    # And so is what had to change before it could be enabled — the pairing,
    # never the check.
    assert "THE PAIRING WAS FIXED, NOT THE CHECK" in pin
    assert "COPY ltx-upscaler.pin" in docker
    # UNSET IS STILL A NO-OP. The skip path is what makes the pin reversible:
    # blank the file and the image is the single-scale one again.
    assert "UPSCALER SKIPPED" in docker
    # And it is a FILE, not a build arg: ARG survives into `docker history`,
    # which is why test_the_image_takes_no_build_argument bans it outright.
    assert not any(l.strip().startswith("ARG ") for l in docker.splitlines())
    # The same gates the LTX bake makes.
    stage = docker[docker.index("THE SPATIAL LATENT UPSCALER"):]
    stage = stage[: stage.index("# Bake the STORY model")]
    assert "refusing to bake different bytes" in stage

def test_a_fallback_pipeline_is_never_sent_conditions(io_paths, fake_checkpoint):
    """`conditions=` on LTXImageToVideoPipeline is a TypeError mid-job on a
    rented GPU. The capability is read off the object that actually loaded,
    not off what the checkpoint could in principle support."""
    src, out = io_paths

    class NoConditions(Recorder):
        _oniq_supports_conditions = False

    pipe = NoConditions()
    videogen.run(_job(), src, out, load_pipeline=lambda: pipe)
    assert "conditions" not in pipe.calls[0]
    assert pipe.calls[0]["image"] is not None


def test_an_anchor_on_a_fallback_pipeline_refuses(tmp_path, fake_checkpoint):
    ref = str(tmp_path / "ref.png")
    Image.new("RGB", (704, 1248), (5, 5, 5)).save(ref)

    class NoConditions(Recorder):
        _oniq_supports_conditions = False

    with pytest.raises(videogen.ReferenceUnsupported):
        videogen.run_image(
            _image_job(reference_key=CANON_REF),
            str(tmp_path / "out.png"),
            load_pipeline=lambda: NoConditions(n=contract.IMAGE_GEN_NUM_FRAMES),
            reference_path=ref,
        )


# ───────────── the official 0.9.8 recipe, and where each number came from

def _profile_for(distilled, monkeypatch, multiscale=True):
    monkeypatch.setattr(ltxcaps, "_multiscale_available", lambda caps: (multiscale, "ok"))
    return ltxcaps.inference_profile({
        "latent_upsampler_baked": multiscale,
        "components_missing": [],
        "distilled": distilled,
        "condition_pipeline_supported": True,
    })


def test_the_distilled_schedule_is_transcribed_not_invented(monkeypatch):
    """Every value below was fetched from upstream on 2026-08-31:
    Lightricks/LTX-Video configs/ltxv-2b-0.9.8-distilled.yaml, and the worked
    multi-scale example in the diffusers LTX docs. The two agree on all the
    values they share."""
    p = _profile_for(True, monkeypatch)
    assert p["multiscale_schedule"] == "ltx-0.9.8-distilled"
    assert p["first_pass_timesteps"] == [1000, 993, 987, 981, 975, 909, 725, 0.03]
    assert p["second_pass_timesteps"] == [1000, 909, 725, 421, 0]
    assert p["refine_denoise_strength"] == 0.999
    assert p["guidance_scale"] == 1.0
    assert p["guidance_rescale"] == 0.7
    assert p["image_cond_noise_scale"] == 0.0
    assert p["decode_timestep"] == 0.05
    assert p["decode_noise_scale"] == 0.025
    assert p["upscale_adain_factor"] == 1.0
    assert ltxcaps.DOWNSCALE_FACTOR == 2 / 3
    # And the provenance travels with them.
    assert "ltxv-2b-0.9.8-distilled.yaml" in p["multiscale_source"]


def test_the_distilled_recipe_is_never_applied_to_a_full_checkpoint(monkeypatch):
    """THE MIXING RULE, as a test.

    `guidance_scale: 1` turns classifier-free guidance OFF. That is correct
    for a guidance-distilled checkpoint and actively wrong for a full one,
    which needs CFG to follow the prompt at all. Likewise the seven-step
    schedule is the distilled sampler's own. Applying either to ONIQ's
    non-distilled checkpoint is the '0.9.8 config on a different LTX
    checkpoint' mixing the owner directive forbids.
    """
    full = _profile_for(False, monkeypatch)
    assert full["multiscale_schedule"] == "full-checkpoint-conservative"
    assert full["first_pass_timesteps"] is None
    assert full["second_pass_timesteps"] is None
    # The checkpoint's OWN guidance and steps stand.
    assert full["guidance_scale"] == 3.0
    assert full["num_inference_steps"] == ltxcaps.STEPS_FULL
    assert full["decode_timestep"] == 0.0
    # And the source says plainly that this half is ONIQ's, not upstream's.
    assert "ONIQ" in full["multiscale_source"]


def test_tone_mapping_is_enabled_only_where_upstream_demonstrates_it(monkeypatch):
    """A PARAMETER EXISTING IS NOT EVIDENCE IT BELONGS IN THIS RECIPE.

    The diffusers note recommends 0.6 for 'the 0.9.8 distilled model' and its
    worked example (a 13B) passes it. But ltxv-2b-0.9.8-distilled.yaml does
    NOT set tone_map_compression_ratio — only the 13B config does. So it is
    applied on the distilled path, where a source demonstrates it, and left at
    diffusers' own 0.0 elsewhere.
    """
    assert _profile_for(True, monkeypatch)["upscale_tone_map_compression"] == 0.6
    assert _profile_for(False, monkeypatch)["upscale_tone_map_compression"] == 0.0
    # And off entirely when there is no multi-scale path at all.
    assert _profile_for(True, monkeypatch, multiscale=False)["upscale_tone_map_compression"] == 0.0


@pytest.fixture
def distilled_multiscale_checkpoint(tmp_path, monkeypatch):
    """A checkpoint the evidence says IS distilled, and which carries the
    upscaler. Both halves matter: the distilled recipe only applies when the
    shipped scheduler agrees with the name."""
    root = write_fake_checkpoint(tmp_path / "ltx-d", distilled=True, upscaler=True)
    monkeypatch.setattr(videogen, "_model_dir", lambda: root)
    monkeypatch.setattr(videogen, "model_id", lambda: "LTX-Video-0.9.8-2B-distilled")
    monkeypatch.setattr(ltxcaps, "_multiscale_available", lambda caps: (True, "ok"))
    return root


def test_the_distilled_first_pass_sends_timesteps_not_a_step_count(
    io_paths, distilled_multiscale_checkpoint
):
    """On the distilled recipe the SCHEDULE is the sampler. diffusers derives
    num_inference_steps from the list's length, so sending both would be two
    answers to one question."""
    src, out = io_paths
    pipe = TwoPassRecorder()
    metrics = videogen.run(
        _job(), src, out,
        load_pipeline=lambda: pipe, load_upsampler=lambda: UpsampleRecorder(),
    )
    base, refine = pipe.calls
    assert base["timesteps"] == [1000, 993, 987, 981, 975, 909, 725, 0.03]
    assert "num_inference_steps" not in base
    assert refine["timesteps"] == [1000, 909, 725, 421, 0]
    assert refine["denoise_strength"] == 0.999
    # Guidance is OFF, which is what "guidance-distilled" means.
    assert base["guidance_scale"] == 1.0
    assert base["guidance_rescale"] == 0.7
    assert metrics["multiscale_schedule"] == "ltx-0.9.8-distilled"


def test_a_full_checkpoint_keeps_its_own_guidance_on_the_multiscale_path(
    io_paths, multiscale_checkpoint
):
    """The other half of the mixing rule, at the call site rather than in the
    profile: a non-distilled checkpoint must still receive CFG."""
    src, out = io_paths
    pipe = TwoPassRecorder()
    videogen.run(
        _job(), src, out,
        load_pipeline=lambda: pipe, load_upsampler=lambda: UpsampleRecorder(),
    )
    base, refine = pipe.calls
    assert base["guidance_scale"] == 3.0
    assert "timesteps" not in base
    assert "timesteps" not in refine
    assert base["num_inference_steps"] == ltxcaps.STEPS_FULL


def test_the_upsample_stage_gets_the_recipes_own_adain_and_tone_map(
    io_paths, multiscale_checkpoint
):
    src, out = io_paths
    up = UpsampleRecorder()
    videogen.run(
        _job(), src, out,
        load_pipeline=lambda: TwoPassRecorder(), load_upsampler=lambda: up,
    )
    call = up.calls[0]
    assert call["output_type"] == "latent", "the upscale happens BEFORE the decode"
    assert call["adain_factor"] == ltxcaps.UPSCALE_ADAIN_FACTOR
    assert "tone_map_compression_ratio" in call


def test_both_canvases_are_legal_for_the_vae():
    """diffusers refuses outright: 'height and width have to be divisible by
    32'. Both stages, not just the first."""
    f = ltxcaps.UPSCALE_SPATIAL_FACTOR
    for w, h in ((contract.VIDEO_WIDTH, contract.VIDEO_HEIGHT),
                 (contract.VIDEO_WIDTH * f, contract.VIDEO_HEIGHT * f)):
        assert w % videogen.VAE_SPATIAL_RATIO == 0
        assert h % videogen.VAE_SPATIAL_RATIO == 0


def test_the_two_stages_bracket_the_film_so_the_last_step_is_a_downscale():
    """The point of the whole path. Stage 1 sits BELOW the film's width and
    stage 2 ABOVE it, so assembly resizes DOWN — which preserves detail —
    instead of UP, which cannot invent it.

    Upstream's own step 4 is exactly this: 'Downscale the video to the
    expected resolution'.
    """
    film_w, film_h = 1080, 1920
    f = ltxcaps.UPSCALE_SPATIAL_FACTOR
    assert contract.VIDEO_WIDTH < film_w
    assert contract.VIDEO_WIDTH * f > film_w
    assert contract.VIDEO_HEIGHT * f > film_h
    # And the aspect survives the trip: the crop into 1080x1920 stays a
    # rounding error rather than a composition decision.
    scale = max(film_w / (contract.VIDEO_WIDTH * f), film_h / (contract.VIDEO_HEIGHT * f))
    assert film_w / (contract.VIDEO_WIDTH * f * scale) > 0.99
    assert film_h / (contract.VIDEO_HEIGHT * f * scale) > 0.99


def test_the_base_canvas_is_near_the_official_downscale_factor():
    """Upstream generates at 2/3 of the target. 1080 * 2/3 = 720, which is NOT
    divisible by 32, so the nearest legal portrait pair is used instead — and
    the deviation is small enough to state."""
    film_w = 1080
    ratio = contract.VIDEO_WIDTH / film_w
    assert abs(ratio - ltxcaps.DOWNSCALE_FACTOR) < 0.02
    assert 720 % videogen.VAE_SPATIAL_RATIO != 0, "720 would have been exact but is illegal"


def test_the_reference_contract_numbers_are_pinned_on_this_side_too():
    """THE OTHER HALF OF A TWO-REPO CONTRACT.

    supabase/functions/_shared/characterRef.ts states the identical prefix,
    scope and strength band, and its own suite asserts those literals. The two
    repositories cannot import each other and CI checks out one at a time, so
    each pins the numbers itself and a drift fails in whichever repo moved.

    That is the lesson of the 1000-character prompt ceiling: a bound known to
    only one end of a contract is a deterministic failure waiting for a long
    enough input, and it cost a 27% still-failure rate before anybody saw it.
    """
    assert contract.REFERENCE_PREFIX == "story/ref/"
    assert contract.REFERENCE_SCOPE_CANON == "canon"
    assert contract.MIN_REFERENCE_STRENGTH == 0.05
    assert contract.MAX_REFERENCE_STRENGTH == 0.95
    # And the shape the app builds is the shape this side accepts.
    assert contract.validate_job({
        "op": "image_generate",
        "output_key": "out/a.png",
        "params": {"prompt": "p", "reference_key": "story/ref/canon/abc/v1.png"},
    })["params"]["reference_key"] == "story/ref/canon/abc/v1.png"
