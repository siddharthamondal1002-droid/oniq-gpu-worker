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

class TypedRecorder:
    """A fake that declares which real pipeline class it stands in for, so the
    kwargs it receives can be checked against that class's accepted set."""

    def __init__(self, klass, frames=None, supports_conditions=None, latent_first=False):
        self.klass = klass
        self.calls = []
        self.n = frames or contract.VIDEO_NUM_FRAMES
        self.latent_first = latent_first
        if supports_conditions is not None:
            self._oniq_supports_conditions = supports_conditions

    def __call__(self, **kwargs):
        from types import SimpleNamespace

        self.calls.append(kwargs)
        if self.latent_first and kwargs.get("output_type") == "latent":
            return SimpleNamespace(frames=["LATENTS"])
        frames = [
            Image.new("RGB", (contract.VIDEO_WIDTH, contract.VIDEO_HEIGHT), (7, 7, 7))
            for _ in range(self.n)
        ]
        return SimpleNamespace(frames=[frames])


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
