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

# Video generation: everything below is a SERVER decision. The caller's
# only degree of freedom is the motion prompt; resolution, frame count,
# fps and the model are constants here and in videogen.py, so no job can
# request a bigger canvas, a longer clip, or a different model.
VIDEO_WIDTH = 704
VIDEO_HEIGHT = 480
VIDEO_NUM_FRAMES = 97  # LTX wants 8k+1 frames; 97 @ 24fps ≈ 4.0s
VIDEO_FPS = 24
MAX_PROMPT_CHARS = 1000

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
_PROBE_TOP_LEVEL_FIELDS = _TOP_LEVEL_FIELDS | {"model"}
_PARAM_FIELDS = frozenset({"target_max_dim", "format", "quality"})
# watermark is a SERVER-derived entitlement relayed by the application —
# absent means TRUE (marked), the fail-safe: an old or malformed caller
# can only ever produce the watermarked product, never a free clean one.
_VIDEO_PARAM_FIELDS = frozenset({"prompt", "watermark"})
_IMAGE_GEN_PARAM_FIELDS = frozenset({"prompt"})
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


def _bounded_int(value, field: str, lo: int, hi: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ContractError("invalid-input", f"{field} must be an integer")
    if value < lo or value > hi:
        raise ContractError(
            "invalid-input", f"{field} must be between {lo} and {hi}"
        )
    return value


def validate_job(raw) -> dict:
    """Validate an incoming event's input and return the normalized job.

    Raises ContractError for anything outside the bounded contract; never
    mutates or echoes unexpected values back to the caller.
    """
    if not isinstance(raw, dict):
        raise ContractError("invalid-input", "job input must be an object")

    allowed = (
        _PROBE_TOP_LEVEL_FIELDS
        if raw.get("op") == "model_probe"
        else _TOP_LEVEL_FIELDS
    )
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

    # image_generate and story_generate are the TEXT-ONLY ops: it draws from a prompt, so
    # it has no source object. An input_key sent with it is refused rather
    # than ignored — a caller that thinks it is conditioning on an image
    # must not be told silently that it was.
    if op in ("image_generate", "story_generate"):
        if raw.get("input_key") is not None:
            raise ContractError("invalid-input", f"{op} takes no input_key")
        input_key = None
    else:
        input_key = _require_key(raw.get("input_key"), "input_key")
    output_key = (
        None if op == "story_generate" else _require_key(raw.get("output_key"), "output_key")
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
        return {
            "op": op,
            "input_key": None,
            "output_key": output_key,
            "params": {"prompt": prompt.strip()},
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
        return {
            "op": op,
            "model": model,
            "input_key": input_key,
            "output_key": output_key,
            "params": {"prompt": prompt.strip()},
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
            "params": {"prompt": prompt.strip(), "watermark": watermark},
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
