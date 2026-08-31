"""Provider-neutral GPU job contract.

This module is the bounded-input half of the worker: everything a job may
ask for is validated here, before any byte is downloaded and before any GPU
code runs. It imports nothing but the standard library — no torch, no
network, no storage — so it is fully testable on a CPU-only machine.

The contract deliberately has no field for choosing a model, a GPU, a
provider, a Docker image, or a runtime. The allowed operations are
`image_preprocess` and `video_generate`; which card runs them is decided
server-side by the harness (validation/admission.py), which model the
video op loads is decided server-side by videogen.py, and the only
degree of freedom a video caller has is the motion prompt — bounded
text, nothing else.
"""

from __future__ import annotations

import re

# The workloads this worker exists to run.
ALLOWED_OPS = (
    "image_preprocess",
    "image_generate",
    "video_generate",
    "audio_mux",
    "video_concat",
    "story_generate",
    # model_probe joined 2026-08-29 (owner: five-model A5000 probe). A MODEL
    # EVALUATION op, not a production one: it fetches a candidate checkpoint at
    # job time and measures it. Production ops are untouched and still run the
    # baked model with local_files_only, so nothing a user can reach changes.
    "model_probe",
    # model_hydrate joined 2026-08-30 (owner: "MODEL WEIGHTS ARE DATA").
    # It fetches an experimental checkpoint onto the persistent volume and
    # produces NO artifact and NO inference: no GPU work, no input object,
    # no output object. It exists so that changing a model stops meaning
    # rebuilding a 25 GiB image and cold-pulling it for two hours.
    "model_hydrate",
)

# Bounded input: the object referenced from R2 may not exceed this, checked
# against Content-Length BEFORE the download begins.
MAX_INPUT_BYTES = 16 * 1024 * 1024

# Decode bound — a small file can still decompress into a huge image.
MAX_IMAGE_PIXELS = 64_000_000

# Output re-encode bounds.
ALLOWED_FORMATS = ("jpeg", "png", "webp")
MIN_TARGET_DIM = 16
MAX_TARGET_DIM = 4096
DEFAULT_TARGET_DIM = 1024
MIN_QUALITY = 1
MAX_QUALITY = 100
DEFAULT_QUALITY = 85

# Runtime ceiling: a job that is still running at the ceiling is an error,
# not a longer job. Refused, never clamped. 900s is the reservation window
# the financial admission charges for in full.
RUNTIME_CEILING_SECONDS = 900

# THE BENCHMARK'S OWN CEILING — model_probe only, owner directive
# 2026-08-29. Production's 900s is unchanged and this constant never
# touches it.
#
# A production job loads a checkpoint that is already in the image. A probe
# DOWNLOADS one: 20.15 GiB for the smallest candidate and 117.52 GiB for
# the largest, before a single frame is generated. Measured against 900s
# that is not a slow job, it is a job that cannot finish — and the way it
# fails is the expensive way, because handler checks the deadline AFTER the
# work returns: the full window is billed, the clip is discarded, and every
# phase measurement the owner asked for is lost with it.
#
# 1800s at the A5000's live secure rate reserves well inside the $0.50 job
# cap, so this buys measurement time rather than raising exposure past a
# gate. It is a starting figure, not a finding: the probe reports its
# download throughput, and the cheap candidates run first precisely so the
# large ones are dispatched against a MEASURED rate instead of this guess.
PROBE_RUNTIME_CEILING_SECONDS = 1800

# THE PREVIEW BUDGET. See preview.py for why a preview exists at all.
#
# These numbers are a response-size decision, not a quality one. The reply
# travels through the provider's job-status payload, and a preview that
# outgrew it would take every measurement down with it — so the cap is
# enforced before a frame is added, never after.
#
# Five frames at a 640px long edge is enough to judge the things an
# inspection turns on: whether there is one subject, whether the face and
# body hold across the clip, and whether the last frame still resembles the
# first. It is deliberately not enough to be a delivery channel; R2 remains
# the only place the artifact lives.
PREVIEW_MAX_FRAMES = 5
PREVIEW_LONG_EDGE = 640
PREVIEW_JPEG_QUALITY = 82
PREVIEW_MAX_BYTES = 900_000

# Video generation: everything below is a SERVER decision. The caller's
# only degree of freedom is the motion prompt; resolution, frame count,
# fps and the model are constants here and in videogen.py, so no job can
# request a bigger canvas, a longer clip, or a different model.
# THE CANVAS IS PORTRAIT, AND THAT IS THE WHOLE POINT (owner directive,
# 2026-08-31, after the capability audit).
#
# It was 704x480 LANDSCAPE while the film it feeds is 1080x1920 PORTRAIT, and
# nothing in between could reconcile them. The assembly does
# `scale=...:force_original_aspect_ratio=increase,crop=1080:1920`, so every
# frame was scaled 4.00x and then had 61.6% of its WIDTH thrown away. Measured
# on the delivered artifact: 270x480 = 129,600 source pixels were stretched to
# fill 1080x1920 = 2,073,600 — SIXTEEN output pixels invented per real one,
# with no upscaler and no face restoration anywhere in the path. That, and not
# the model, is why faces were hazy.
#
# 704x1248 is chosen, not rounded to:
#   - both axes divisible by 32, which the VAE's spatial compression requires;
#   - 0.5641 against the film's 0.5625, so the crop falls from 61.6% to 0.28%;
#   - 2.60x the pixels of the old canvas, taking the deficit from 16.00x to
#     2.37x — a 6.75x improvement — without the jump to a full 1080-wide
#     latent, which projects past this card's memory at 97 frames.
#
# THE PROJECTION IS NOT A MEASUREMENT. Scaling the measured 13,803 MB peak by
# pixel count gives ~35.9 GB of the A5000-class 48 GB card and ~101s against
# the 600s endpoint ceiling. Attention does not scale linearly, so the true
# figure will differ; videogen fails CLOSED on OOM rather than trusting this.
VIDEO_WIDTH = 704
VIDEO_HEIGHT = 1248
VIDEO_NUM_FRAMES = 97  # LTX wants 8k+1 frames; 97 @ 24fps ≈ 4.0s
VIDEO_FPS = 24
MAX_PROMPT_CHARS = 1000

# The VAE's spatial compression makes a non-multiple-of-32 canvas silently
# resize inside the pipeline, which would reintroduce exactly the resampling
# this change exists to remove. Asserted at import so a future edit to the two
# numbers above cannot land quietly.
assert VIDEO_WIDTH % 32 == 0, "VIDEO_WIDTH must be divisible by 32"
assert VIDEO_HEIGHT % 32 == 0, "VIDEO_HEIGHT must be divisible by 32"
assert VIDEO_HEIGHT > VIDEO_WIDTH, "the film is portrait; the canvas must be too"

# Bounds on a caller-supplied negative prompt. Per-shot negatives are the
# point (a face-artifact list belongs to a shot with a face in it, not to a
# landscape), but an unbounded one would eat the model's own token budget and
# push the positive prompt out of the window.
MAX_NEGATIVE_PROMPT_CHARS = 400

# A caller-supplied seed is a 64-bit unsigned integer. It is DERIVED by the
# app from job/scene/shot/attempt — never random, never a clock — so the same
# attempt reproduces and a different attempt genuinely differs.
MAX_SEED = 2**64 - 1

# story_generate: ONIQ's own causal LLM (owner directive 2026-08-27). The
# prompt is long by nature — a brief, a budget and a schema — so it has
# its own ceiling rather than the motion prompt's. The token budget is
# bounded here too: an unbounded generation is an unbounded bill.
MAX_STORY_PROMPT_CHARS = 20000
MIN_STORY_TOKENS = 256
MAX_STORY_TOKENS = 8192

# image_generate: ONIQ's OWN image engine (fully in-house directive,
# 2026-08-27). It is the SAME baked LTX snapshot as video_generate — the
# text-to-video pipeline over the transformer/vae/text_encoder/tokenizer
# already in this image — sampled for the shortest legal clip, of which
# frame 0 is kept as a still. No second model, no new weights, no
# download: the component that used to be an outsourced image API is a
# different pipeline class over bytes this worker already carries.
#
# The canvas is deliberately VIDEO_WIDTH x VIDEO_HEIGHT: this still's
# whole purpose is to be the conditioning frame a video_generate job
# animates, and a frame that does not match the video canvas would be
# rescaled at the seam.
IMAGE_GEN_NUM_FRAMES = 9  # LTX wants 8k+1; 9 is the cheapest legal pass
IMAGE_GEN_FORMAT = "png"  # lossless — it is a conditioning frame, not a
# delivery artifact, and jpeg ringing would be fed to the video model

# audio_mux: the caller's only degree of freedom is the narration text.
# This bound is an input-size fence; the REAL gate is measured seconds
# against the video's own duration, in audio.py, refused never truncated.
MAX_NARRATION_CHARS = 300

# video_concat: ordered stream-copy assembly of clips this worker itself
# generated (monetization resolution loop, 2026-08-27). 16 segments of the
# 97-frame clip is ~65s of film — the v1 assembly ceiling; longer films
# concat concats. Segments must be uniform (same canvas, same audio
# presence); anything else is refused, never coerced.
MIN_CONCAT_SEGMENTS = 2
MAX_CONCAT_SEGMENTS = 16

# R2 keys are references, not paths: a bounded character set, no leading
# slash, no parent-directory traversal.
_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,511}$")

_TOP_LEVEL_FIELDS = frozenset({"op", "input_key", "output_key", "params"})
# `model` is legal on model_probe AND NOWHERE ELSE.
#
# The standing fence is that a caller cannot choose the model a production job
# runs, and test_no_model_field_exists_in_contract has held that line since the
# contract was written. Adding "model" to the set above would have quietly
# retired that guard for every op at once — the test caught it, which is what
# it is for. So the field is admitted per-op instead, and even on model_probe
# it is a KEY into the worker's own benchmark table, never a repository, path
# or revision: a caller may say which row of an authorised benchmark to run,
# never what to download or how to run it.
_PROBE_TOP_LEVEL_FIELDS = _TOP_LEVEL_FIELDS | {"model", "preview"}


def _sampling_params(params_raw: dict) -> dict:
    """The two per-shot sampler inputs, validated once for both generate ops.

    SEED AND NEGATIVE PROMPT ARE THE CALLER'S, and everything else about the
    sampler stays the server's. They are here because neither can be decided
    by the worker without making the product worse:

      - a module-level SEED made every retry re-sample IDENTICALLY, so the
        ten attempts the owner authorised on 2026-08-31 were ten copies of one
        image. The app derives it from job/scene/shot/attempt, so the same
        attempt reproduces and a different attempt genuinely differs.
      - one global negative prompt cannot serve both a close-up face and an
        empty landscape. A face-artifact list belongs to the shot that has a
        face in it.

    Both are OPTIONAL. Absent means "the server's own default applies", so
    every existing caller keeps its current behaviour byte for byte.
    """
    out: dict = {}

    if "seed" in params_raw:
        seed = params_raw["seed"]
        # bool is an int subclass in Python and True would silently become 1.
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ContractError("invalid-input", "params.seed must be an integer")
        if not 0 <= seed <= MAX_SEED:
            raise ContractError(
                "invalid-input", f"params.seed must be within 0..{MAX_SEED}"
            )
        out["seed"] = seed

    if "negative_prompt" in params_raw:
        neg = params_raw["negative_prompt"]
        if not isinstance(neg, str):
            raise ContractError(
                "invalid-input", "params.negative_prompt must be a string"
            )
        neg = neg.strip()
        if len(neg) > MAX_NEGATIVE_PROMPT_CHARS:
            raise ContractError(
                "invalid-input",
                f"params.negative_prompt may not exceed "
                f"{MAX_NEGATIVE_PROMPT_CHARS} characters",
            )
        # An empty string is a caller asking for NO negative prompt, which is
        # different from not asking at all. Both are legal; they differ, and
        # the difference is preserved rather than collapsed.
        out["negative_prompt"] = neg

    return out
# model_hydrate carries a model id and nothing else. No keys, because it
# reads no object and writes none; no params, because there is nothing to
# tune about a download whose revision is pinned server-side.
_HYDRATE_TOP_LEVEL_FIELDS = frozenset({"op", "model"})
# `preview` is legal on the two ops an INSPECTION reads, and nowhere else.
#
# It changes nothing about what is generated or what it costs: the artifact
# is produced and uploaded identically either way, and the flag only decides
# whether the reply also carries a thumbnail of it (see preview.py). It is
# admitted per-op for the same reason `model` is — a shared field set would
# put it on every production op at once — and production callers never send
# it, so their replies stay byte-for-byte what they were.
_PREVIEWABLE_TOP_LEVEL_FIELDS = _TOP_LEVEL_FIELDS | {"preview"}
_PARAM_FIELDS = frozenset({"target_max_dim", "format", "quality"})
# watermark is a SERVER-derived entitlement relayed by the application —
# absent means TRUE (marked), the fail-safe: an old or malformed caller
# can only ever produce the watermarked product, never a free clean one.
_VIDEO_PARAM_FIELDS = frozenset({"prompt", "watermark", "seed", "negative_prompt"})

# model_probe ONLY. Owner directive 2026-08-30: "Do not rebuild the image
# merely to change frames, steps, checkpoint revision, offload mode or
# model weights. Those values belong to the model configuration/provider
# layer." Baking them cost a two-hour cold pull per edit.
#
# This does NOT widen what a user can reach. model_probe is a server-side
# evaluation op: `model` is legal on it and nowhere else, the browser
# cannot select it, and every production op still takes neither field. The
# standing fence is that a USER may not choose model, precision, runtime
# or offloading strategy; the provider adapter always could, and this is
# the provider adapter.
_PROBE_PARAM_FIELDS = _VIDEO_PARAM_FIELDS | frozenset({"frames", "steps"})

# Bounded, so a typo cannot become an expensive job. The upper frame bound
# is the 121 that OOM'd on the A5000 on 2026-08-30 - measured, so nothing
# above it is worth dispatching on this card.
MIN_PROBE_FRAMES = 5
MAX_PROBE_FRAMES = 121
MIN_PROBE_STEPS = 1
MAX_PROBE_STEPS = 50
_IMAGE_GEN_PARAM_FIELDS = frozenset(
    {"prompt", "seed", "negative_prompt", "reference_key", "reference_strength"}
)
_STORY_PARAM_FIELDS = frozenset({"prompt", "max_tokens"})
_AUDIO_PARAM_FIELDS = frozenset({"narration"})
_CONCAT_PARAM_FIELDS = frozenset({"segment_keys"})

# Every key the handler may return. Anything not named here is dropped by
# filter_output before the response leaves the worker.
OUTPUT_WHITELIST = frozenset(
    {
        "ok",
        "op",
        "output_key",
        "width",
        "height",
        "format",
        "output_bytes",
        "device",
        "gpu_name",
        "vram_total_mb",
        "vram_peak_mb",
        "duration_ms",
        "cleanup_ok",
        "code",
        "error",
        # video_generate evidence — measured on the worker, never inferred
        "model",
        "model_load_ms",
        "inference_ms",
        "encode_ms",
        "frames",
        "fps",
        "video_seconds",
        # story_generate evidence — the text itself plus its measurements
        "story_text",
        "story_chars",
        # QUALITY DIAGNOSTICS (owner directive, 2026-08-31). Every clip must
        # carry enough to diagnose a bad face without re-running it: which
        # weights, which sampler numbers, which seed. The 2026-08-30 film was
        # undiagnosable precisely because none of this was recorded — guidance
        # was never even passed, so no value existed to report.
        #
        # NO SECRETS AND NO STORAGE URLS pass through here; the whitelist is
        # what makes that checkable rather than merely intended.
        "pipeline_class",
        "scheduler_class",
        "distilled",
        "condition_pipeline_supported",
        "latent_upsampler_baked",
        "num_inference_steps",
        "guidance_scale",
        "guidance_rescale",
        "decode_timestep",
        "decode_noise_scale",
        "image_cond_noise_scale",
        "defaults_source",
        "seed",
        "negative_prompt_chars",
        "conditioning_count",
        # WHICH pipeline class actually ran, as opposed to the one the
        # checkpoint's model_index names. `pipeline_class` above is the
        # snapshot's own declaration; this is what was instantiated, and the
        # two differ whenever the condition pipeline is used or falls back.
        "pipeline_used",
        "conditioning",
        "conditioning_strength",
        # MULTI-SCALE evidence. `multiscale` is what the checkpoint CAN do;
        # `upscaler_used` is what this job actually did; `upscaler_absent_reason`
        # says why when the two differ. All three, because a soft clip and a
        # missing component look identical in an output file.
        "multiscale",
        "multiscale_reason",
        # WHICH RECIPE RAN, and where each number came from. The distilled
        # 0.9.8 schedule and a full checkpoint's are different sets of values,
        # and a clip that used the wrong one looks exactly like a clip that
        # used the right one until somebody reads this field.
        "multiscale_schedule",
        "multiscale_source",
        "first_pass_timesteps",
        "second_pass_timesteps",
        "refine_denoise_strength",
        "upscale_adain_factor",
        "upscale_tone_map_compression",
        "upscale_spatial_factor",
        "max_sequence_length",
        "upscaler_used",
        "upscaler_absent_reason",
        # The resolution the LAST pass actually sampled at. The film is 1080
        # wide; whether that is a downscale of real detail or an upscale of
        # absent detail is the whole of the haze question, and this is the
        # number that answers it.
        "render_width",
        "render_height",
        "refine_steps_run",
        # WHICH precision actually loaded. 4-bit and bf16 differ by ~12GB
        # of the card, and the load can silently fall back, so the mode is
        # reported by the job rather than assumed from the config.
        "precision",
        # audio_mux evidence — measured on the worker, never inferred
        "has_audio",
        "narration_seconds",
        "audio_seconds",
        "audio_sample_rate",
        "audio_peak_dbfs",
        "audio_gain_db",
        "tts_ms",
        "mux_ms",
        # watermark evidence — whether the mark was actually burned, so the
        # application can fail closed when entitlement and artifact disagree
        "watermarked",
        # model_probe evidence — owner directive 2026-08-29. EVERY field the
        # benchmark asks for, because filter_output drops anything unlisted
        # and a probe whose measurements were silently discarded would have
        # cost a rented GPU and returned nothing. test_modelprobe walks the
        # report against this set so the two cannot drift apart.
        "label",
        "repo",
        "revision",
        "licence",
        "dtype",
        "offload",
        "failure",
        "detail",
        "total_wall_ms",
        "download_ms",
        "conditioning_load_ms",
        # VRAM read from the device, before and at peak. Separate from
        # vram_peak_mb above: those are the production video fields, and a
        # benchmark that reused them would be comparing rounded megabytes.
        "vram_total_bytes",
        "vram_before_allocated_bytes",
        "vram_before_reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        # Disk, because "the model did not fit on the worker" and "the model
        # does not work" are different findings.
        "disk_free_bytes",
        "disk_total_bytes",
        "download_bytes",
        # The ratio the hydrated checkpoint declared, so the report can
        # say WHY a frame count was legal rather than asserting it.
        "vae_temporal_ratio",
        # Where the weights actually came from. A probe that read a
        # hydrated volume reports download_bytes 0, which is
        # indistinguishable from a download that did nothing unless
        # the source is stated.
        "weights_source", "weights_path",
        # Hydration record fields (model_hydrate, 2026-08-30). A model
        # that is on the volume must be able to prove it: revision,
        # bytes, file count and the manifest digest, or the READY it
        # reports is just a word.
        "model_id", "state", "path", "already_present", "licence",
        "file_count", "manifest_sha256", "download_ms",
        "disk_free_after_gib",
        "preview_frames",
        "steps",
        "guidance",
        "sampling_source",
        # video_concat evidence
        "segments",
        "concat_ms",
    }
)


class ContractError(Exception):
    """A job the contract refuses. `code` is a stable kebab-case token."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _require_key(value, field: str) -> str:
    if not isinstance(value, str) or not _KEY_RE.match(value):
        raise ContractError(
            "invalid-input",
            f"{field} must be a bounded object key "
            "([A-Za-z0-9._/-], max 512 chars, no leading slash)",
        )
    if ".." in value:
        raise ContractError("invalid-input", f"{field} may not contain '..'")
    return value


# THE ONLY SHAPE A REFERENCE MAY ARRIVE IN.
#
# The worker reads a character reference from the media bucket with its OWN
# credentials, so the field naming it is an authority to read one object. It is
# therefore NOT a free object key: it is pinned to a server-owned prefix and a
# bounded id, so the set of objects any caller can name is exactly the set of
# canonical character references the application has published — and nothing
# else in the bucket, of any user's.
#
# `input_key` stays a general key because the caller that supplies it is the
# edge function, which DERIVES it from a job token; this field can arrive from
# further out, so it is narrowed at the contract rather than trusted.
# THE KEY SCHEME, and why it has a scope segment and a version.
#
#     story/ref/canon/<characterId>/v<n>.png
#
# SCOPE (`canon`) because today's references are ONIQ's own published
# characters — shared canon, belonging to no user and no film. The segment
# exists so that per-user references, if they are ever added, land under a
# DIFFERENT scope and the isolation is structural rather than a rule somebody
# has to remember. A regex that admitted only `story/ref/<id>` would have to
# be widened later, and widening an authorisation pattern is exactly the
# change nobody reviews carefully enough.
#
# VERSION because a reference must be IMMUTABLE once a film has used it. A
# shot drawn against v1 keeps looking like v1 even after the character is
# re-published as v2; overwriting one key in place would silently change
# films that were already finished. Versions are integers, not timestamps and
# not random ids — the same reason storySeed derives rather than rolls.
REFERENCE_PREFIX = "story/ref/"
REFERENCE_SCOPE_CANON = "canon"
# The usable band for an identity anchor, and both ends are refusals rather
# than clamps. Above the top the sampler simply returns the reference; below
# the bottom the anchor is indistinguishable from no anchor at all, and a
# caller that asked for one deserves to be told it would have done nothing.
MIN_REFERENCE_STRENGTH = 0.05
MAX_REFERENCE_STRENGTH = 0.95
_REFERENCE_RE = __import__("re").compile(
    r"^story/ref/canon/[A-Za-z0-9][A-Za-z0-9._-]{0,120}/v[1-9][0-9]{0,3}\.png$"
)


def _require_reference_key(value) -> str:
    if not isinstance(value, str) or not _REFERENCE_RE.match(value):
        raise ContractError(
            "invalid-input",
            "params.reference_key must be a canonical character reference: "
            f"{REFERENCE_PREFIX}{REFERENCE_SCOPE_CANON}/<characterId>/v<n>.png",
        )
    if ".." in value:
        raise ContractError("invalid-input", "params.reference_key may not contain '..'")
    return value


def _bounded_int(value, field: str, lo: int, hi: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ContractError("invalid-input", f"{field} must be an integer")
    if value < lo or value > hi:
        raise ContractError(
            "invalid-input", f"{field} must be between {lo} and {hi}"
        )
    return value


def _probe_params(prompt: str, params_raw: dict) -> dict:
    """Validated probe knobs. Absent means "use the row's own default".

    Bounded rather than trusted: an unbounded frame count is a job that
    runs to the ceiling and bills for it, and an out-of-range step count
    measures something nobody asked for.
    """
    out = {"prompt": prompt.strip()}

    frames = params_raw.get("frames")
    if frames is not None:
        if not isinstance(frames, int) or isinstance(frames, bool):
            raise ContractError("invalid-input", "params.frames must be an integer")
        if not MIN_PROBE_FRAMES <= frames <= MAX_PROBE_FRAMES:
            raise ContractError(
                "invalid-input",
                f"params.frames must be between {MIN_PROBE_FRAMES} and "
                f"{MAX_PROBE_FRAMES}; {frames} is outside what this card has "
                "been measured to hold",
            )
        out["frames"] = frames

    steps = params_raw.get("steps")
    if steps is not None:
        if not isinstance(steps, int) or isinstance(steps, bool):
            raise ContractError("invalid-input", "params.steps must be an integer")
        if not MIN_PROBE_STEPS <= steps <= MAX_PROBE_STEPS:
            raise ContractError(
                "invalid-input",
                f"params.steps must be between {MIN_PROBE_STEPS} and "
                f"{MAX_PROBE_STEPS}",
            )
        out["steps"] = steps

    return out


def validate_job(raw) -> dict:
    """Validate an incoming event's input and return the normalized job.

    Raises ContractError for anything outside the bounded contract; never
    mutates or echoes unexpected values back to the caller.
    """
    if not isinstance(raw, dict):
        raise ContractError("invalid-input", "job input must be an object")

    if raw.get("op") == "model_hydrate":
        allowed = _HYDRATE_TOP_LEVEL_FIELDS
    elif raw.get("op") == "model_probe":
        allowed = _PROBE_TOP_LEVEL_FIELDS
    elif raw.get("op") == "image_generate":
        allowed = _PREVIEWABLE_TOP_LEVEL_FIELDS
    else:
        allowed = _TOP_LEVEL_FIELDS
    unknown = set(raw) - allowed
    if unknown:
        raise ContractError(
            "invalid-input",
            "unknown job field(s): " + ", ".join(sorted(unknown)),
        )

    op = raw.get("op")
    if op not in ALLOWED_OPS:
        raise ContractError(
            "op-not-allowed",
            "op must be one of: " + ", ".join(ALLOWED_OPS),
        )

    # `preview` is a flag, and a flag that accepts anything truthy is not a
    # flag. Typed here rather than coerced, because every other field in this
    # contract is refused rather than guessed at.
    if "preview" in raw and not isinstance(raw["preview"], bool):
        raise ContractError("invalid-input", "preview must be true or false")
    preview = bool(raw.get("preview"))

    # image_generate and story_generate are the TEXT-ONLY ops: it draws from a prompt, so
    # it has no source object. An input_key sent with it is refused rather
    # than ignored — a caller that thinks it is conditioning on an image
    # must not be told silently that it was.
    if op in ("image_generate", "story_generate", "model_hydrate"):
        if raw.get("input_key") is not None:
            raise ContractError("invalid-input", f"{op} takes no input_key")
        input_key = None
    else:
        input_key = _require_key(raw.get("input_key"), "input_key")
    output_key = (
        None
        if op in ("story_generate", "model_hydrate")
        else _require_key(raw.get("output_key"), "output_key")
    )

    params_raw = raw.get("params", {})
    if params_raw is None:
        params_raw = {}
    if not isinstance(params_raw, dict):
        raise ContractError("invalid-input", "params must be an object")

    if op == "story_generate":
        unknown_params = set(params_raw) - _STORY_PARAM_FIELDS
        if unknown_params:
            raise ContractError(
                "invalid-input",
                "unknown params field(s): " + ", ".join(sorted(unknown_params)),
            )
        prompt = params_raw.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ContractError(
                "invalid-input", "params.prompt must be a non-empty string"
            )
        if len(prompt) > MAX_STORY_PROMPT_CHARS:
            raise ContractError(
                "invalid-input",
                f"params.prompt exceeds {MAX_STORY_PROMPT_CHARS} characters",
            )
        max_tokens = params_raw.get("max_tokens", MAX_STORY_TOKENS)
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
            raise ContractError("invalid-input", "params.max_tokens must be an integer")
        if not MIN_STORY_TOKENS <= max_tokens <= MAX_STORY_TOKENS:
            raise ContractError(
                "invalid-input",
                f"params.max_tokens must be {MIN_STORY_TOKENS}..{MAX_STORY_TOKENS}",
            )
        # The story comes back IN THE RESPONSE, not as a stored artifact:
        # it is structured text the application validates before anything
        # is rendered, so writing it to the media bucket would create a
        # file nothing tracks.
        return {
            "op": op,
            "input_key": None,
            "output_key": None,
            "params": {"prompt": prompt.strip(), "max_tokens": max_tokens},
        }

    if op == "image_generate":
        unknown_params = set(params_raw) - _IMAGE_GEN_PARAM_FIELDS
        if unknown_params:
            raise ContractError(
                "invalid-input",
                "unknown params field(s): " + ", ".join(sorted(unknown_params)),
            )
        prompt = params_raw.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ContractError(
                "invalid-input", "params.prompt must be a non-empty string"
            )
        if len(prompt) > MAX_PROMPT_CHARS:
            raise ContractError(
                "invalid-input",
                f"params.prompt exceeds {MAX_PROMPT_CHARS} characters",
            )
        # No watermark field, deliberately: a conditioning frame is an
        # INTERMEDIATE, never delivered. The mark is burned by the video
        # stage that consumes it, from the entitlement of record, so a
        # still cannot carry a second mark into the film.
        # THE IDENTITY ANCHOR (owner directive, 2026-08-31 — character
        # identity is not preserved between character creation, scene
        # creation and motion).
        #
        # WHAT THIS IS AND IS NOT. LTX-Video has no identity-transfer
        # mechanism — no IP-Adapter, no face embedding, no reference-only
        # attention; VERIFIED by reading the installed diffusers 0.38.0 LTX
        # pipelines end to end. What it has is FRAME conditioning. So the
        # honest anchor is img2img-shaped: the canonical character frame is
        # supplied as the frame-0 condition at a strength below 1.0, and the
        # sampler starts partway from that person instead of from noise. A
        # strength of 1.0 would hand the reference straight back; low
        # strengths are indistinguishable from drawing the character again.
        #
        # It is bounded here rather than trusted because it is the difference
        # between "a new shot of this person" and "the reference, returned".
        reference_key = params_raw.get("reference_key")
        reference = None
        if reference_key is not None:
            reference = _require_reference_key(reference_key)
        strength = params_raw.get("reference_strength")
        if strength is not None:
            if reference is None:
                raise ContractError(
                    "invalid-input",
                    "params.reference_strength without params.reference_key",
                )
            if isinstance(strength, bool) or not isinstance(strength, (int, float)):
                raise ContractError(
                    "invalid-input", "params.reference_strength must be a number"
                )
            if not MIN_REFERENCE_STRENGTH <= float(strength) <= MAX_REFERENCE_STRENGTH:
                raise ContractError(
                    "invalid-input",
                    "params.reference_strength must be between "
                    f"{MIN_REFERENCE_STRENGTH} and {MAX_REFERENCE_STRENGTH}",
                )
            strength = float(strength)
        return {
            "op": op,
            "input_key": None,
            "output_key": output_key,
            "preview": preview,
            "params": {
                "prompt": prompt.strip(),
                **_sampling_params(params_raw),
                **({"reference_key": reference} if reference else {}),
                **({"reference_strength": strength} if strength is not None else {}),
            },
        }

    if op == "model_hydrate":
        import modelroot

        model = raw.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ContractError("invalid-input", "model must be a non-empty string")
        model = model.strip()
        if model not in modelroot.EXPERIMENTAL:
            # Named, never guessed at. A hydrate that silently fell back to
            # some default would download 32 GiB of the wrong checkpoint
            # and mark it READY.
            raise ContractError(
                "invalid-input",
                "model must be one of: " + ", ".join(sorted(modelroot.EXPERIMENTAL)),
            )
        return {
            "op": op,
            "model": model,
            "input_key": None,
            "output_key": None,
            "params": {},
        }

    if op == "model_probe":
        import modelprobe

        model = raw.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ContractError("invalid-input", "model must be a non-empty string")
        model = model.strip()
        if model in modelprobe.NOT_EVALUATED:
            raise ContractError(
                "invalid-input",
                f"{model} is NOT_EVALUATED: {modelprobe.NOT_EVALUATED[model]}",
            )
        if model not in modelprobe.PROBE_MODELS:
            # Named, not guessed at. An unknown key is a harness bug and the
            # job costs nothing to refuse; silently falling back to some
            # default would spend a GPU measuring the wrong model.
            raise ContractError(
                "invalid-input",
                "model must be one of: " + ", ".join(sorted(modelprobe.PROBE_MODELS)),
            )
        unknown_params = set(params_raw) - _PROBE_PARAM_FIELDS
        if unknown_params:
            raise ContractError(
                "invalid-input",
                "unknown params field(s): " + ", ".join(sorted(unknown_params)),
            )
        prompt = params_raw.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ContractError(
                "invalid-input", "params.prompt must be a non-empty string"
            )
        if len(prompt) > MAX_PROMPT_CHARS:
            raise ContractError(
                "invalid-input",
                f"params.prompt may not exceed {MAX_PROMPT_CHARS} characters",
            )
        return {
            "op": op,
            "model": model,
            "input_key": input_key,
            "output_key": output_key,
            # CARRIED, not just admitted. Every return here builds an explicit
            # dict, so a field that is validated above and left out below is
            # accepted and then silently discarded — which is what happened to
            # `preview` on 2026-08-29: the flag was checked, the job ran, and
            # the worker never saw it, so the reference came back invisible
            # and cost a dispatch to find out.
            "preview": preview,
            "params": _probe_params(prompt, params_raw),
        }

    if op == "video_generate":
        unknown_params = set(params_raw) - _VIDEO_PARAM_FIELDS
        if unknown_params:
            raise ContractError(
                "invalid-input",
                "unknown params field(s): " + ", ".join(sorted(unknown_params)),
            )
        prompt = params_raw.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ContractError(
                "invalid-input", "params.prompt must be a non-empty string"
            )
        if len(prompt) > MAX_PROMPT_CHARS:
            raise ContractError(
                "invalid-input",
                f"params.prompt may not exceed {MAX_PROMPT_CHARS} characters",
            )
        # Absent -> True is the fail-safe; a PRESENT non-bool is a malformed
        # contract and the job is refused outright rather than guessed at.
        watermark = params_raw.get("watermark", True)
        if not isinstance(watermark, bool):
            raise ContractError(
                "invalid-input", "params.watermark must be a boolean"
            )
        return {
            "op": op,
            "input_key": input_key,
            "output_key": output_key,
            "params": {
                "prompt": prompt.strip(),
                "watermark": watermark,
                **_sampling_params(params_raw),
            },
        }

    if op == "video_concat":
        unknown_params = set(params_raw) - _CONCAT_PARAM_FIELDS
        if unknown_params:
            raise ContractError(
                "invalid-input",
                "unknown params field(s): " + ", ".join(sorted(unknown_params)),
            )
        segments_raw = params_raw.get("segment_keys")
        if not isinstance(segments_raw, list):
            raise ContractError(
                "invalid-input", "params.segment_keys must be a list of keys"
            )
        if not (MIN_CONCAT_SEGMENTS <= len(segments_raw) <= MAX_CONCAT_SEGMENTS):
            raise ContractError(
                "invalid-input",
                "params.segment_keys must hold between "
                f"{MIN_CONCAT_SEGMENTS} and {MAX_CONCAT_SEGMENTS} keys",
            )
        segment_keys = [
            _require_key(k, f"params.segment_keys[{i}]")
            for i, k in enumerate(segments_raw)
        ]
        if len(set(segment_keys)) != len(segment_keys):
            raise ContractError(
                "invalid-input", "params.segment_keys may not repeat a key"
            )
        # input_key names the first segment so every op still declares one
        # bounded primary input; a mismatch is a caller bug, refused.
        if input_key != segment_keys[0]:
            raise ContractError(
                "invalid-input",
                "input_key must equal params.segment_keys[0]",
            )
        return {
            "op": op,
            "input_key": input_key,
            "output_key": output_key,
            "params": {"segment_keys": segment_keys},
        }

    if op == "audio_mux":
        unknown_params = set(params_raw) - _AUDIO_PARAM_FIELDS
        if unknown_params:
            raise ContractError(
                "invalid-input",
                "unknown params field(s): " + ", ".join(sorted(unknown_params)),
            )
        narration = params_raw.get("narration")
        if not isinstance(narration, str) or not narration.strip():
            raise ContractError(
                "invalid-input", "params.narration must be a non-empty string"
            )
        if len(narration) > MAX_NARRATION_CHARS:
            raise ContractError(
                "invalid-input",
                f"params.narration may not exceed {MAX_NARRATION_CHARS} characters",
            )
        return {
            "op": op,
            "input_key": input_key,
            "output_key": output_key,
            "params": {"narration": narration.strip()},
        }

    unknown_params = set(params_raw) - _PARAM_FIELDS
    if unknown_params:
        raise ContractError(
            "invalid-input",
            "unknown params field(s): " + ", ".join(sorted(unknown_params)),
        )

    target_max_dim = _bounded_int(
        params_raw.get("target_max_dim", DEFAULT_TARGET_DIM),
        "params.target_max_dim",
        MIN_TARGET_DIM,
        MAX_TARGET_DIM,
    )
    fmt = params_raw.get("format", "jpeg")
    if fmt not in ALLOWED_FORMATS:
        raise ContractError(
            "invalid-input",
            "params.format must be one of: " + ", ".join(ALLOWED_FORMATS),
        )
    quality = _bounded_int(
        params_raw.get("quality", DEFAULT_QUALITY),
        "params.quality",
        MIN_QUALITY,
        MAX_QUALITY,
    )

    return {
        "op": op,
        "input_key": input_key,
        "output_key": output_key,
        "params": {
            "target_max_dim": target_max_dim,
            "format": fmt,
            "quality": quality,
        },
    }


def filter_output(result: dict) -> dict:
    """Drop every key not on the explicit output whitelist."""
    return {k: v for k, v in result.items() if k in OUTPUT_WHITELIST}
