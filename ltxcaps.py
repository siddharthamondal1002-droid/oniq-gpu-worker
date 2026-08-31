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

# Where a spatial latent upscaler would live if one were ever baked. It is a
# SEPARATE model (LTXLatentUpsamplerModel, its own repository) and the current
# image does not carry it — see Dockerfile FIRST_PASS/COMPONENTS, neither of
# which names it. Detected rather than assumed so that baking one later turns
# the path on without another code change.
UPSCALER_DIRS = ("latent_upsampler", "spatial_upscaler", "ltxv-spatial-upscaler")


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
        "timesteps": None,
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
        "defaults_source": profile.get("defaults_source"),
    }
