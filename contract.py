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
ALLOWED_OPS = ("image_preprocess", "video_generate", "audio_mux", "video_concat")

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

# Video generation: everything below is a SERVER decision. The caller's
# only degree of freedom is the motion prompt; resolution, frame count,
# fps and the model are constants here and in videogen.py, so no job can
# request a bigger canvas, a longer clip, or a different model.
VIDEO_WIDTH = 704
VIDEO_HEIGHT = 480
VIDEO_NUM_FRAMES = 97  # LTX wants 8k+1 frames; 97 @ 24fps ≈ 4.0s
VIDEO_FPS = 24
MAX_PROMPT_CHARS = 1000

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
_PARAM_FIELDS = frozenset({"target_max_dim", "format", "quality"})
# watermark is a SERVER-derived entitlement relayed by the application —
# absent means TRUE (marked), the fail-safe: an old or malformed caller
# can only ever produce the watermarked product, never a free clean one.
_VIDEO_PARAM_FIELDS = frozenset({"prompt", "watermark"})
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

    unknown = set(raw) - _TOP_LEVEL_FIELDS
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

    input_key = _require_key(raw.get("input_key"), "input_key")
    output_key = _require_key(raw.get("output_key"), "output_key")

    params_raw = raw.get("params", {})
    if params_raw is None:
        params_raw = {}
    if not isinstance(params_raw, dict):
        raise ContractError("invalid-input", "params must be an object")

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
