"""Provider-neutral GPU job contract.

This module is the bounded-input half of the worker: everything a job may
ask for is validated here, before any byte is downloaded and before any GPU
code runs. It imports nothing but the standard library — no torch, no
network, no storage — so it is fully testable on a CPU-only machine.

The contract deliberately has no field for choosing a model and no field
for choosing a GPU. The single allowed operation is `image_preprocess`;
which card runs it is decided server-side by the harness (validation/
admission.py), never by the caller.
"""

from __future__ import annotations

import re

# The one workload this worker exists to run.
ALLOWED_OPS = ("image_preprocess",)

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

# R2 keys are references, not paths: a bounded character set, no leading
# slash, no parent-directory traversal.
_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,511}$")

_TOP_LEVEL_FIELDS = frozenset({"op", "input_key", "output_key", "params"})
_PARAM_FIELDS = frozenset({"target_max_dim", "format", "quality"})

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
