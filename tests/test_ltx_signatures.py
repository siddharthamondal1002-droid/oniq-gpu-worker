"""EVERY KWARG THE WORKER SENDS IS ONE THE PIPELINE ACTUALLY ACCEPTS.

WHY THIS FILE EXISTS. A pipeline's signature is the only thing that decides
whether a kwarg is accepted, and getting it wrong is not a soft failure: it is
a TypeError raised part-way through a job, on a rented GPU, after the model has
already loaded. Two of those were found by hand in this codebase:

  1. `conditions=` sent to LTXImageToVideoPipeline, which has no such
     parameter — caught by reading the source.
  2. `denoise_strength=` sent to the same fallback on the multi-scale path —
     caught 2026-08-31 by extracting the signatures mechanically. The first
     guard did not cover it, because it guarded the parameter it knew about
     rather than the CLASS's whole accepted set.

The second is the point. Guarding known-bad parameters one at a time will keep
missing the next one. So this checks the whole surface: what videogen actually
sends (captured behaviourally from a real run, not transcribed) against what
each class actually accepts (parsed from the pinned wheel, not remembered).

`ltx_signatures.json` is DATA, extracted by ast from the diffusers 0.38.0
wheel. The pin in requirements.txt is asserted to match it below, so a bump
cannot leave this file quietly stale.
"""

import json
import os

import pytest
from PIL import Image

import contract
import ltxcaps
import videogen
from conftest import write_fake_checkpoint

SIGS = json.load(
    open(os.path.join(os.path.dirname(__file__), "..", "ltx_signatures.json"), encoding="utf-8")
)


def test_the_manifest_matches_the_diffusers_pin():
    """A signature file for a version the worker does not install is worse than
    none: it would pass while describing a different library."""
    with open(os.path.join(os.path.dirname(__file__), "..", "requirements.txt"),
              encoding="utf-8") as fh:
        req = fh.read()
    assert f"diffusers=={SIGS['_diffusers_version']}" in req


def test_the_manifest_records_where_it_came_from():
    assert "diffusers 0.38.0" in SIGS["_source"]
    assert "ast-parsed" in SIGS["_source"]


# ─────────────────────── the facts that drive the guards in videogen

def test_the_fallback_pipeline_accepts_none_of_the_conditioning_parameters():
    """The whole reason `_oniq_supports_conditions` exists, as a fact rather
    than a memory."""
    i2v = set(SIGS["LTXImageToVideoPipeline"])
    for absent in ("conditions", "denoise_strength", "image_cond_noise_scale", "strength"):
        assert absent not in i2v, f"{absent} unexpectedly present — re-check the guards"
    # And the condition pipeline has all four, which is why it is preferred.
    cond = set(SIGS["LTXConditionPipeline"])
    for present in ("conditions", "denoise_strength", "image_cond_noise_scale", "latents"):
        assert present in cond


def test_the_text_pipeline_cannot_take_a_condition_either():
    t2v = set(SIGS["LTXPipeline"])
    assert "conditions" not in t2v
    assert "image_cond_noise_scale" not in t2v


def test_the_upsampler_takes_exactly_what_the_worker_sends_it():
    up = set(SIGS["LTXLatentUpsamplePipeline"])
    for sent in ("latents", "adain_factor", "tone_map_compression_ratio", "output_type"):
        assert sent in up
    # It builds from the pipeline's OWN vae plus the upsampler and nothing
    # else, which is why no new weights beyond the upsampler reach the card.
    assert SIGS["LTXLatentUpsamplePipeline.__init__"] == ["vae", "latent_upsampler"]


# ───────────────────── what videogen ACTUALLY sends, captured and checked

class UnsupportedKwarg(TypeError):
    """What the real pipeline would raise, raised at the same moment."""


class TypedRecorder:
    """A fake that REFUSES what the real class would refuse.

    A fake taking **kwargs and accepting everything is the wrong shape for
    this job: it turns a TypeError into a later assertion, so the failure
    surfaces somewhere other than the call that caused it — and any code path
    the test does not explicitly assert on passes silently. This one raises at
    the call, exactly as diffusers would, from the class's real accepted set.
    """

    def __init__(self, klass, frames=None, supports_conditions=None, latent_first=False):
        self.klass = klass
        self.accepted = set(SIGS[klass])
        self.calls = []
        self.n = frames or contract.VIDEO_NUM_FRAMES
        self.latent_first = latent_first
        if supports_conditions is not None:
            self._oniq_supports_conditions = supports_conditions

    def __call__(self, **kwargs):
        from types import SimpleNamespace

        unsupported = sorted(set(kwargs) - self.accepted)
        if unsupported:
            # The real message shape, so a failure reads like the production
            # one rather than like a test artefact.
            raise UnsupportedKwarg(
                f"{self.klass}.__call__() got an unexpected keyword argument "
                f"{unsupported[0]!r}"
            )
        self.calls.append(kwargs)
        if self.latent_first and kwargs.get("output_type") == "latent":
            return SimpleNamespace(frames=["LATENTS"])
        frames = [
            Image.new("RGB", (contract.VIDEO_WIDTH, contract.VIDEO_HEIGHT), (7, 7, 7))
            for _ in range(self.n)
        ]
        return SimpleNamespace(frames=[frames])


def test_the_fake_itself_refuses_an_unsupported_kwarg():
    """The guard is only worth what the fake refuses. If this ever passes
    silently, every test below is checking nothing."""
    pipe = TypedRecorder("LTXImageToVideoPipeline")
    with pytest.raises(UnsupportedKwarg, match="denoise_strength"):
        pipe(prompt="x", denoise_strength=0.4)
    with pytest.raises(UnsupportedKwarg, match="conditions"):
        pipe(prompt="x", conditions=[])
    # And it accepts what the real class accepts.
    pipe(prompt="x", width=8, height=8, num_frames=9)


def _assert_accepted(recorder):
    accepted = set(SIGS[recorder.klass])
    for i, call in enumerate(recorder.calls):
        unsupported = sorted(set(call) - accepted)
        assert not unsupported, (
            f"{recorder.klass} call {i} would raise TypeError on: {unsupported}"
        )


def _job(**params):
    return contract.validate_job({
        "op": "video_generate",
        "input_key": "in/a.jpg",
        "output_key": "out/a.mp4",
        "params": {"prompt": "she turns to the window", **params},
    })


@pytest.fixture
def io_paths(tmp_path):
    src = str(tmp_path / "in.png")
    Image.new("RGB", (contract.VIDEO_WIDTH, contract.VIDEO_HEIGHT), (30, 60, 90)).save(src)
    return src, str(tmp_path / "out.mp4")


def test_the_condition_pipeline_receives_only_parameters_it_accepts(
    io_paths, fake_checkpoint, monkeypatch
):
    # The CPU rig has no diffusers, so _video_condition returns None and the
    # code correctly degrades to `image=` — which LTXConditionPipeline also
    # accepts. To exercise the conditions branch the condition object is faked.
    monkeypatch.setattr(
        videogen, "_video_condition",
        lambda image, frame_index=None, strength=None: {"image": image},
    )
    src, out = io_paths
    pipe = TypedRecorder("LTXConditionPipeline", supports_conditions=True)
    videogen.run(_job(seed=7, negative_prompt="blurry face"), src, out,
                 load_pipeline=lambda: pipe)
    assert pipe.calls and "conditions" in pipe.calls[0]
    _assert_accepted(pipe)


def test_degrading_to_image_stays_inside_the_signature_too(io_paths, fake_checkpoint):
    """When a condition cannot be built the code sends `image=` instead. That
    is only safe because LTXConditionPipeline accepts BOTH — checked here
    rather than assumed."""
    assert "image" in set(SIGS["LTXConditionPipeline"])
    src, out = io_paths
    pipe = TypedRecorder("LTXConditionPipeline", supports_conditions=True)
    videogen.run(_job(), src, out, load_pipeline=lambda: pipe)
    assert "image" in pipe.calls[0] and "conditions" not in pipe.calls[0]
    _assert_accepted(pipe)


def test_the_fallback_pipeline_receives_only_parameters_it_accepts(
    io_paths, fake_checkpoint
):
    src, out = io_paths
    pipe = TypedRecorder("LTXImageToVideoPipeline", supports_conditions=False)
    videogen.run(_job(), src, out, load_pipeline=lambda: pipe)
    assert "conditions" not in pipe.calls[0]
    assert "image" in pipe.calls[0]
    _assert_accepted(pipe)


def test_the_still_pipeline_receives_only_parameters_it_accepts(
    tmp_path, fake_checkpoint
):
    pipe = TypedRecorder("LTXPipeline", frames=contract.IMAGE_GEN_NUM_FRAMES)
    videogen.run_image(
        contract.validate_job({
            "op": "image_generate", "output_key": "out/a.png",
            "params": {"prompt": "a face by lamplight", "seed": 3},
        }),
        str(tmp_path / "out.png"),
        load_pipeline=lambda: pipe,
    )
    _assert_accepted(pipe)


def test_multiscale_never_reaches_a_pipeline_that_takes_no_denoise_strength(
    io_paths, tmp_path, monkeypatch
):
    """THE BUG THIS FILE WAS WRITTEN FOR.

    The refine pass is DEFINED by denoise_strength — without it the second pass
    is a full re-generation at 4x the pixels rather than a refinement. The
    fallback pipeline has no such parameter, so multi-scale must not run on it
    at all, and must say why rather than degrade in silence.
    """
    root = write_fake_checkpoint(tmp_path / "ltx-up", upscaler=True)
    monkeypatch.setattr(videogen, "_model_dir", lambda: root)
    monkeypatch.setattr(videogen, "model_id", lambda: "Lightricks/LTX-Video")
    monkeypatch.setattr(ltxcaps, "_multiscale_available", lambda caps: (True, "ok"))

    src, out = io_paths
    pipe = TypedRecorder("LTXImageToVideoPipeline", supports_conditions=False,
                         latent_first=True)
    called = []
    metrics = videogen.run(
        _job(), src, out,
        load_pipeline=lambda: pipe,
        load_upsampler=lambda: called.append(1),
    )
    assert called == [], "the upsampler must not even be loaded"
    assert len(pipe.calls) == 1, "no refine pass on a pipeline that cannot refine"
    assert metrics["upscaler_used"] is False
    assert metrics["upscaler_absent_reason"] == "fallback-pipeline-takes-no-denoise-strength"
    _assert_accepted(pipe)


def test_multiscale_on_the_condition_pipeline_stays_inside_the_signature(
    io_paths, tmp_path, monkeypatch
):
    root = write_fake_checkpoint(tmp_path / "ltx-up2", upscaler=True)
    monkeypatch.setattr(videogen, "_model_dir", lambda: root)
    monkeypatch.setattr(videogen, "model_id", lambda: "Lightricks/LTX-Video")
    monkeypatch.setattr(ltxcaps, "_multiscale_available", lambda caps: (True, "ok"))

    class Up:
        def __init__(self):
            self.calls = []

        def __call__(self, **kwargs):
            from types import SimpleNamespace
            self.calls.append(kwargs)
            return SimpleNamespace(frames=["UPSCALED"])

    src, out = io_paths
    pipe = TypedRecorder("LTXConditionPipeline", supports_conditions=True,
                         latent_first=True)
    up = Up()
    videogen.run(_job(), src, out, load_pipeline=lambda: pipe, load_upsampler=lambda: up)
    assert len(pipe.calls) == 2
    _assert_accepted(pipe)
    # The upsampler's own surface, checked the same way.
    accepted = set(SIGS["LTXLatentUpsamplePipeline"])
    for call in up.calls:
        assert not set(call) - accepted


# ───────────── the geometry, against the model's own compression ratios

VAE_SPATIAL = 32   # LTXConditionPipeline refuses anything else, verbatim
VAE_TEMPORAL = 8   # num_latent_frames = (num_frames - 1) // 8 + 1


def test_both_stages_produce_whole_latents_in_space_and_time():
    """Every dimension the sampler is given has to land on a whole latent.

    Space and time are separate ratios and it is possible to satisfy one and
    not the other — 97 frames is 8k+1 for a reason, and a canvas can be
    portrait and still illegal.
    """
    f = ltxcaps.UPSCALE_SPATIAL_FACTOR
    stages = {
        "stage 1": (contract.VIDEO_WIDTH, contract.VIDEO_HEIGHT),
        "stage 2": (contract.VIDEO_WIDTH * f, contract.VIDEO_HEIGHT * f),
    }
    for name, (w, h) in stages.items():
        assert w % VAE_SPATIAL == 0, f"{name} width {w}"
        assert h % VAE_SPATIAL == 0, f"{name} height {h}"
        # The latent grid the transformer actually attends over.
        assert (w // VAE_SPATIAL) * (h // VAE_SPATIAL) > 0

    # Temporal: (n - 1) must divide the temporal ratio exactly, or the last
    # latent frame is partial and the clip's tail is undefined.
    assert (contract.VIDEO_NUM_FRAMES - 1) % VAE_TEMPORAL == 0
    latent_frames = (contract.VIDEO_NUM_FRAMES - 1) // VAE_TEMPORAL + 1
    assert latent_frames == 13
    # The upsampler is SPATIAL only, so the frame count is identical either
    # side of it — a temporal upsampler would change the clip's length and
    # silently break the 97-frame contract.
    assert ltxcaps.UPSCALE_SPATIAL_FACTOR == 2


def test_the_upscale_multiplies_latent_area_by_exactly_four():
    """2x in each spatial axis. Stated because the VRAM and time arithmetic
    downstream depends on it, and because 'it upscales' is not a number."""
    f = ltxcaps.UPSCALE_SPATIAL_FACTOR
    base = (contract.VIDEO_WIDTH // VAE_SPATIAL) * (contract.VIDEO_HEIGHT // VAE_SPATIAL)
    up = ((contract.VIDEO_WIDTH * f) // VAE_SPATIAL) * ((contract.VIDEO_HEIGHT * f) // VAE_SPATIAL)
    assert up == base * 4


def test_the_aspect_survives_both_stages_and_the_film():
    """A crop is a composition decision. This one has to stay a rounding
    error, or the framing fix upstream of it is undone by the assembler."""
    f = ltxcaps.UPSCALE_SPATIAL_FACTOR
    film_w, film_h = 1080, 1920
    for w, h in ((contract.VIDEO_WIDTH, contract.VIDEO_HEIGHT),
                 (contract.VIDEO_WIDTH * f, contract.VIDEO_HEIGHT * f)):
        assert abs((w / h) - (film_w / film_h)) < 0.005, f"{w}x{h}"
