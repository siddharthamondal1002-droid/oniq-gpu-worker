"""What the BAKED checkpoint can actually do, read from the checkpoint.

WHY THIS MODULE EXISTS. videogen.py used to carry the inference settings as
module constants: SEED = 42, a step count chosen by `"distilled" in
model_id()`, and a negative prompt written once for every shot in every film.
Two of those three are decisions that belong to the CHECKPOINT, not to the
worker, and the third belongs to the caller.

The step count was the clearest case. `"distilled" in model_id()` is a
substring test against a repository NAME, and a name is not a capability:
Lightricks ships distilled 2B weights as a single-file checkpoint whose
repository is called LTX-Video-2B-0.9.6-Distilled, and ships the non-distilled
diffusers snapshot this image actually bakes as plain LTX-Video. The same
substring test would answer "distilled" for a repository this loader cannot
even consume, and answers "not distilled" for the one it does — correctly, but
by luck rather than by evidence.

SO NOTHING HERE IS INFERRED FROM A NAME. Everything is read from files the
bake put on disk: model_index.json for the pipeline class and the component
list, scheduler_config.json for the sampler the checkpoint shipped with, and
the directory listing for whether an optional asset (the spatial upscaler) is
present at all.

FAIL CLOSED, ALWAYS. A checkpoint that cannot be read, or that carries
contradictory evidence, raises. It does not fall back to a default profile:
a wrong step count is not a crash, it is a film that costs money and looks
wrong, which is far harder to notice. `CheckpointInconsistent` is deliberately
not catchable into a default anywhere in this repo.

THE GUIDANCE NUMBERS ARE THE PIPELINE'S OWN DOCUMENTED DEFAULTS, read from the
installed pipeline's signature rather than copied into this file as literals.
"Never rely on a silent default" does not mean "invent a number"; it means
send the value explicitly and record what it was. If diffusers changes its
default, the recorded diagnostics change with it and the change is visible,
which is the property a hardcoded 3.0 would destroy.
"""

from __future__ import annotations

import json
import os

# The component set LTXImageToVideoPipeline and LTXConditionPipeline both
# declare. Verified against diffusers 0.38.0: the two pipelines take the
# IDENTICAL five modules, which is why conditioning needs no new weights.
LTX_COMPONENTS = ("transformer", "vae", "text_encoder", "tokenizer", "scheduler")

# Where a spatial latent upscaler lives once baked. It is a SEPARATE model
# (LTXLatentUpsamplerModel, its own repository), so it is DETECTED rather than
# assumed and baking one turns the path on without another code change.
UPSCALER_DIRS = ("latent_upsampler", "spatial_upscaler", "ltxv-spatial-upscaler")

# ---------------------------------------------------------------- multi-scale
#
# WHAT THE UPSCALER IS FOR, and why it is the largest remaining quality lever.
#
# MEASURED, arithmetic rather than opinion: the generation canvas is 704 wide
# and the delivered film is 1080 wide. Even with the portrait canvas fixed,
# assembly still resamples every frame UP by 1080/704 = 1.534x. A pixel
# upscaler cannot invent detail that was never sampled, so that 53% is
# precisely the softness a viewer reads as "hazy" — it is baked in before
# Remotion ever opens the clip.
#
# The fix upstream ships is MULTI-SCALE: generate latents at the base canvas,
# upsample them IN LATENT SPACE with LTXLatentUpsamplerModel, then run a short
# second denoise pass at the higher resolution so the transformer actually
# synthesises the new detail, and only then decode. VERIFIED against the
# installed diffusers 0.38.0 source:
#
#   LTXLatentUpsamplePipeline(vae, latent_upsampler)   pipeline_ltx_latent_upsample.py
#     __call__(video|latents, height, width, decode_timestep, decode_noise_scale,
#              adain_factor, tone_map_compression_ratio, generator, output_type)
#   LTXConditionPipeline.__call__(..., latents=..., denoise_strength=...)
#
# so the base pass can hand latents out (output_type="latent"), the upsampler
# can take them and hand back 2x latents, and the condition pipeline can take
# THOSE back for a partial re-denoise. No new pipeline class is invented here;
# all three calls are the ones diffusers documents.
#
# The upsampler reuses the VAE ALREADY BAKED — it is the only other module the
# pipeline takes — so the incremental cost is the upsampler weights alone.

# ── THE OFFICIAL 0.9.8 MULTI-SCALE RECIPE ────────────────────────────────────
#
# TRANSCRIBED FROM TWO UPSTREAM SOURCES, fetched 2026-08-31. Not recalled, not
# inferred from a model card, and not invented — the previous draft of this
# file carried placeholder numbers and said so; these replace them.
#
#   A. Lightricks/LTX-Video @ main, configs/ltxv-2b-0.9.8-distilled.yaml
#      raw.githubusercontent.com — the checkpoint family's OWN config.
#   B. huggingface/diffusers @ main, docs/source/en/api/pipelines/ltx_video.md
#      the worked multi-scale example, which is what the pipeline code expects.
#
# The two agree on every value they share (downscale factor, decode timestep,
# decode noise scale, both timestep schedules, guidance 1). B additionally
# supplies the three A does not state: guidance_rescale, image_cond_noise_scale
# and adain_factor.
#
# ── WHY THIS IS GATED ON DISTILLATION, AND WHY THAT IS THE WHOLE POINT ───────
#
# Every number below belongs to a GUIDANCE- AND TIMESTEP-DISTILLED checkpoint.
# `guidance_scale: 1` means classifier-free guidance is OFF, which is correct
# for a distilled model and actively wrong for a full one — CFG is what a
# non-distilled checkpoint uses to follow the prompt at all. The explicit
# timestep lists are the distilled sampler's own schedule; a full checkpoint
# has no business walking seven steps.
#
# ONIQ's baked checkpoint is Lightricks/LTX-Video at revision 8984fa25, and
# inspect_checkpoint() reads its shipped scheduler at runtime rather than
# trusting its name. Applying this recipe to a checkpoint the evidence says is
# NOT distilled would be exactly the "0.9.8 config on a different LTX
# checkpoint" mixing the owner directive forbids. So the profile carries two
# multi-scale schedules and picks by evidence, and records WHICH it used.
DOWNSCALE_FACTOR = 2 / 3

# Source A + B, verbatim. The trailing 0.03 on the first pass and the trailing
# 0 on the second are upstream's, not a typo: the schedules are open at one end.
DISTILLED_FIRST_PASS_TIMESTEPS = [1000, 993, 987, 981, 975, 909, 725, 0.03]
DISTILLED_SECOND_PASS_TIMESTEPS = [1000, 909, 725, 421, 0]
# "Effectively, 4 inference steps out of 5" — upstream's own comment.
DISTILLED_DENOISE_STRENGTH = 0.999
DISTILLED_GUIDANCE_SCALE = 1.0
DISTILLED_GUIDANCE_RESCALE = 0.7
# ZERO on the multi-scale path, where diffusers' standalone default is 0.15.
# The conditioning frame is held by the first pass; re-noising it during the
# refine would fight the latents the upsampler just produced.
DISTILLED_IMAGE_COND_NOISE_SCALE = 0.0
# Decode settings differ from the single-pass defaults (0.0 / None) and both
# sources agree on these.
DISTILLED_DECODE_TIMESTEP = 0.05
DISTILLED_DECODE_NOISE_SCALE = 0.025

# ── THE UPSCALE STAGE ────────────────────────────────────────────────────────
#
# ADAIN 1.0, from source B. The upsampler shifts the latent distribution;
# matching its moments back to the pre-upscale latents keeps colour and
# contrast where the first pass put them. diffusers' standalone default is 0.0
# — that is the "no opinion" value, not the recommended one for this recipe.
UPSCALE_ADAIN_FACTOR = 1.0

# TONE MAPPING — enabled ONLY where upstream actually demonstrates it.
#
# This is the one place the two sources differ, and the difference is
# load-bearing. The diffusers note recommends 0.6 for "the 0.9.8 distilled
# model", and its worked example (a 13B) passes it. The 2B config file does
# NOT set it; only ltxv-13b-0.9.8-distilled.yaml carries
# tone_map_compression_ratio in its second pass.
#
# So a parameter EXISTING is not evidence it belongs in this recipe. It is
# applied on the distilled path, where a source demonstrates it, and left at
# diffusers' own 0.0 elsewhere. A measurement can move it; a plausible-sounding
# default should not.
DISTILLED_TONE_MAP_COMPRESSION = 0.6
UPSCALE_TONE_MAP_COMPRESSION = 0.0

# ── THE NON-DISTILLED SECOND PASS ────────────────────────────────────────────
#
# ONIQ's, and labelled as ONIQ's. No upstream config exists for multi-scale on
# a FULL 0.9.8 checkpoint, so there is nothing to transcribe. What is known is
# the shape: the second pass must be short and partial, or it re-generates at
# 2x cost and discards the composition the first pass agreed on.
#
# With no explicit timestep list, diffusers derives the refine schedule from
# num_inference_steps scaled by denoise_strength — so 0.4 of the checkpoint's
# own 30 steps is roughly 12, which is the same order as the distilled path's
# 4-of-5. Conservative on purpose: a low denoise strength cannot destroy the
# base composition. It is UNMEASURED and a tuning pass moves it WITH a figure.
FULL_REFINE_DENOISE_STRENGTH = 0.4

# The spatial factor LTXLatentUpsamplerModel applies (spatial_upsample=True,
# temporal_upsample=False -> PixelShuffleND(2) over the spatial dims).
UPSCALE_SPATIAL_FACTOR = 2

# The T5 window LTX actually encodes. VERIFIED from the installed
# LTXConditionPipeline.__call__ signature: max_sequence_length defaults to 256.
# Everything past it is DROPPED silently, which is why a prompt budget stated
# in characters has to be reasoned about in tokens.
MAX_SEQUENCE_LENGTH = 256


def _multiscale_available(caps: dict) -> tuple:
    """(supported, reason). Both halves of the answer, always.

    A capability that is merely absent must still say WHY, because "the clip
    was soft" and "the upscaler was never in the image" look identical in an
    output file and completely different in a diagnosis.
    """
    if not caps.get("latent_upsampler_baked"):
        return False, "latent-upsampler-not-baked"
    if caps.get("components_missing"):
        return False, "checkpoint-incomplete"
    try:
        from diffusers import LTXLatentUpsamplePipeline  # noqa: F401
    except Exception:  # noqa: BLE001 - absent or too old, same consequence
        return False, "diffusers-has-no-latent-upsample-pipeline"
    return True, "ok"


class CheckpointInconsistent(RuntimeError):
    """The baked checkpoint cannot be trusted to configure inference."""


def _read_json(path: str):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _distillation_evidence(model_id_text: str, scheduler_cfg: dict | None) -> dict:
    """Every signal that bears on 'is this a timestep-distilled checkpoint'.

    Returned as a dict of named signals rather than a bool so that a
    contradiction is VISIBLE to the caller and to the diagnostics, instead of
    being resolved silently by whichever branch happened to be first.
    """
    name_says = "distilled" in (model_id_text or "").lower()
    # A timestep-distilled LTX checkpoint ships a sampler configured for the
    # very short schedule it was distilled onto. The marker is the scheduler
    # carrying an explicit sigma/timestep schedule of its own, which the stock
    # flow-match scheduler does not.
    sched_says = False
    if isinstance(scheduler_cfg, dict):
        sched_says = bool(
            scheduler_cfg.get("timesteps")
            or scheduler_cfg.get("sigmas")
            or scheduler_cfg.get("is_distilled")
        )
    return {"name_says_distilled": name_says, "scheduler_says_distilled": sched_says}


def inspect_checkpoint(model_dir: str, model_id_text: str = "") -> dict:
    """Read the baked checkpoint. Raises rather than guessing."""
    if not model_dir or not os.path.isdir(model_dir):
        raise CheckpointInconsistent(f"model dir {model_dir!r} is not a directory")

    index = _read_json(os.path.join(model_dir, "model_index.json"))
    if not isinstance(index, dict):
        raise CheckpointInconsistent("model_index.json missing or unreadable")

    pipeline_class = str(index.get("_class_name") or "")
    if "LTX" not in pipeline_class:
        raise CheckpointInconsistent(
            f"model_index.json declares {pipeline_class!r}, which is not an LTX pipeline"
        )

    declared = {k for k, v in index.items() if isinstance(v, list)}
    present = tuple(
        c for c in LTX_COMPONENTS if os.path.isdir(os.path.join(model_dir, c))
    )
    missing = [c for c in LTX_COMPONENTS if c not in present]

    scheduler_cfg = _read_json(
        os.path.join(model_dir, "scheduler", "scheduler_config.json")
    )
    evidence = _distillation_evidence(model_id_text, scheduler_cfg)

    # A CONTRADICTION IS A REFUSAL. If the name and the shipped sampler
    # disagree about distillation, neither profile can be applied honestly:
    # distilled settings on a full checkpoint produce mush, and full settings
    # on a distilled one waste four times the GPU seconds for a worse frame.
    if evidence["name_says_distilled"] != evidence["scheduler_says_distilled"]:
        raise CheckpointInconsistent(
            "distillation evidence disagrees: "
            f"{evidence} — refusing to choose a sampler profile"
        )

    upscaler = next(
        (d for d in UPSCALER_DIRS if os.path.isdir(os.path.join(model_dir, d))),
        None,
    )

    return {
        "model_id": model_id_text or "unknown",
        "pipeline_class": pipeline_class,
        "declared_components": sorted(declared),
        "components_present": list(present),
        "components_missing": missing,
        "scheduler_class": str((scheduler_cfg or {}).get("_class_name") or "unknown"),
        "distilled": evidence["name_says_distilled"],
        "distillation_evidence": evidence,
        # Conditioning needs no extra weights — only the same five modules.
        "condition_pipeline_supported": not missing,
        "latent_upsampler_baked": upscaler is not None,
        "latent_upsampler_dir": upscaler,
    }


def _documented_defaults() -> dict:
    """The installed pipeline's OWN documented defaults, by introspection.

    Read from the signature rather than transcribed, so this file cannot drift
    from the library it configures. Falls back to the values verified against
    diffusers 0.38.0 only when diffusers is absent (the CPU contract rig).
    """
    verified = {
        "guidance_scale": 3.0,
        "guidance_rescale": 0.0,
        "decode_timestep": 0.0,
        "decode_noise_scale": None,
        "image_cond_noise_scale": 0.15,
        "source": "verified-against-diffusers-0.38.0",
    }
    try:
        import inspect as _inspect

        from diffusers import LTXConditionPipeline
    except Exception:
        return verified
    try:
        params = _inspect.signature(LTXConditionPipeline.__call__).parameters
        out = {"source": "introspected-from-installed-diffusers"}
        for key in (
            "guidance_scale",
            "guidance_rescale",
            "decode_timestep",
            "decode_noise_scale",
            "image_cond_noise_scale",
        ):
            p = params.get(key)
            out[key] = verified[key] if p is None or p.default is _inspect.Parameter.empty else p.default
        return out
    except (TypeError, ValueError):
        return verified


# Step counts. These are ONIQ's decisions and are stated here rather than
# borrowed from a default, because the default (50) is a general-purpose
# number and this worker runs one shape of job on one card under a 600s
# endpoint ceiling. 30 was the value the previous constant used for a full
# checkpoint and is retained deliberately: this change is about deriving the
# CHOICE from the checkpoint, not about retuning the number at the same time.
STEPS_DISTILLED = 8
STEPS_FULL = 30


def _multiscale_schedule(multiscale: bool, distilled: bool) -> dict:
    """The two-pass numbers, and WHERE EACH ONE CAME FROM.

    `multiscale_schedule` is the answer to "which recipe ran", which is the
    question a soft clip actually poses. Reporting the values without their
    provenance would leave nobody able to tell an upstream figure from a
    guess — and this file has carried both.
    """
    if not multiscale:
        return {
            "multiscale_schedule": "none",
            "multiscale_source": None,
            "first_pass_timesteps": None,
            "second_pass_timesteps": None,
            "refine_denoise_strength": None,
            "upscale_adain_factor": UPSCALE_ADAIN_FACTOR,
            "upscale_tone_map_compression": UPSCALE_TONE_MAP_COMPRESSION,
        }
    if distilled:
        # Every value transcribed. Nothing here is ONIQ's opinion.
        return {
            "multiscale_schedule": "ltx-0.9.8-distilled",
            "multiscale_source": (
                "Lightricks/LTX-Video configs/ltxv-2b-0.9.8-distilled.yaml + "
                "diffusers docs ltx_video.md, fetched 2026-08-31"
            ),
            "first_pass_timesteps": list(DISTILLED_FIRST_PASS_TIMESTEPS),
            "second_pass_timesteps": list(DISTILLED_SECOND_PASS_TIMESTEPS),
            "refine_denoise_strength": DISTILLED_DENOISE_STRENGTH,
            "guidance_scale": DISTILLED_GUIDANCE_SCALE,
            "guidance_rescale": DISTILLED_GUIDANCE_RESCALE,
            "image_cond_noise_scale": DISTILLED_IMAGE_COND_NOISE_SCALE,
            "decode_timestep": DISTILLED_DECODE_TIMESTEP,
            "decode_noise_scale": DISTILLED_DECODE_NOISE_SCALE,
            "upscale_adain_factor": UPSCALE_ADAIN_FACTOR,
            "upscale_tone_map_compression": DISTILLED_TONE_MAP_COMPRESSION,
        }
    # A FULL checkpoint. The distilled schedule must not be applied to it:
    # guidance 1 turns classifier-free guidance off, and a full checkpoint
    # needs CFG to follow the prompt at all. So the checkpoint's own guidance
    # and step count stand (set by the caller above) and only the refine
    # strength is added — ONIQ's, stated, unmeasured.
    return {
        "multiscale_schedule": "full-checkpoint-conservative",
        "multiscale_source": "ONIQ — no upstream multi-scale config exists for a full 0.9.8 checkpoint",
        "first_pass_timesteps": None,
        "second_pass_timesteps": None,
        "refine_denoise_strength": FULL_REFINE_DENOISE_STRENGTH,
        "upscale_adain_factor": UPSCALE_ADAIN_FACTOR,
        # Not demonstrated for a full checkpoint anywhere upstream.
        "upscale_tone_map_compression": UPSCALE_TONE_MAP_COMPRESSION,
    }


def inference_profile(caps: dict) -> dict:
    """The exact kwargs this checkpoint should be sampled with."""
    if not isinstance(caps, dict) or "distilled" not in caps:
        raise CheckpointInconsistent("inference_profile needs inspect_checkpoint output")
    if caps.get("components_missing"):
        raise CheckpointInconsistent(
            f"checkpoint is missing {caps['components_missing']} — cannot sample"
        )

    d = _documented_defaults()
    distilled = bool(caps["distilled"])
    multiscale, multiscale_reason = _multiscale_available(caps)
    profile = {
        "num_inference_steps": STEPS_DISTILLED if distilled else STEPS_FULL,
        # Classifier-free guidance is what a distilled checkpoint removes the
        # need for; sending 3.0 to one would double the work for a worse frame.
        "guidance_scale": 1.0 if distilled else float(d["guidance_scale"]),
        "guidance_rescale": float(d["guidance_rescale"]),
        "decode_timestep": d["decode_timestep"],
        "decode_noise_scale": d["decode_noise_scale"],
        "image_cond_noise_scale": d["image_cond_noise_scale"],
        # Whether this checkpoint can be sampled through LTXConditionPipeline,
        # which is what makes the conditioning frame EXPLICIT — named frame,
        # named strength, named noise scale — instead of implicit and
        # unrecordable. Derived from the components actually on disk.
        "conditioning": bool(caps.get("condition_pipeline_supported")),
        # NOTE: the multi-scale block below is spread LAST on purpose — on a
        # distilled checkpoint it legitimately overrides guidance and the
        # decode settings with the recipe's own values.
        # MULTI-SCALE. Both halves of the answer travel in the profile, so a
        # soft clip can be told apart from a missing capability without
        # re-running anything.
        "multiscale": multiscale,
        "multiscale_reason": multiscale_reason,
        "upscale_spatial_factor": UPSCALE_SPATIAL_FACTOR if multiscale else 1,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "timesteps": None,
        **_multiscale_schedule(multiscale, distilled),
        "defaults_source": d["source"],
        "profile_for": "distilled" if distilled else "full",
    }
    return profile


def diagnostics(caps: dict, profile: dict) -> dict:
    """What every job records about the engine that produced it.

    NO SECRETS AND NO STORAGE URLS — only the model's public identity and the
    numbers that were actually sent to it.
    """
    return {
        "model_id": caps.get("model_id"),
        "pipeline_class": caps.get("pipeline_class"),
        "scheduler_class": caps.get("scheduler_class"),
        "distilled": caps.get("distilled"),
        "condition_pipeline_supported": caps.get("condition_pipeline_supported"),
        "latent_upsampler_baked": caps.get("latent_upsampler_baked"),
        "num_inference_steps": profile.get("num_inference_steps"),
        "guidance_scale": profile.get("guidance_scale"),
        "guidance_rescale": profile.get("guidance_rescale"),
        "decode_timestep": profile.get("decode_timestep"),
        "decode_noise_scale": profile.get("decode_noise_scale"),
        "image_cond_noise_scale": profile.get("image_cond_noise_scale"),
        "conditioning": profile.get("conditioning"),
        "multiscale": profile.get("multiscale"),
        "multiscale_reason": profile.get("multiscale_reason"),
        "multiscale_schedule": profile.get("multiscale_schedule"),
        "multiscale_source": profile.get("multiscale_source"),
        "first_pass_timesteps": profile.get("first_pass_timesteps"),
        "second_pass_timesteps": profile.get("second_pass_timesteps"),
        "refine_denoise_strength": profile.get("refine_denoise_strength"),
        "upscale_adain_factor": profile.get("upscale_adain_factor"),
        "upscale_tone_map_compression": profile.get("upscale_tone_map_compression"),
        "upscale_spatial_factor": profile.get("upscale_spatial_factor"),
        "max_sequence_length": profile.get("max_sequence_length"),
        "defaults_source": profile.get("defaults_source"),
    }
