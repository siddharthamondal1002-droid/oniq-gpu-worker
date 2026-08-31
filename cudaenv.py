"""Set the CUDA allocator strategy before torch can read it, and PROVE it.

OWNER DIRECTIVE, 2026-08-30: "set it BEFORE the first CUDA initialization.
Verify this in the running worker. Do not claim it is configured merely
because it appears in source."

PyTorch reads PYTORCH_CUDA_ALLOC_CONF once, when its CUDA allocator is
first initialised. Setting it after that point is a no-op that leaves no
trace — the variable is present, the setting is not applied, and every
report says it was. That is exactly the "green step proving nothing"
failure this codebase has hit before, so the module records WHEN it ran
relative to torch rather than merely that it ran.

Why the setting is wanted: the 2026-08-30 Hunyuan probe OOM'd on the
A5000 with 1.44 GiB reserved-but-unallocated — fragmentation PyTorch's own
error message pointed at. Expandable segments let the allocator grow a
segment instead of stranding it.

This module must be imported BEFORE anything imports torch. handler.py
imports it first, above every other project module, for that reason.
"""

from __future__ import annotations

import os
import sys

VARIABLE = "PYTORCH_CUDA_ALLOC_CONF"
SETTING = "expandable_segments:True"

# Captured at import: was torch already in the interpreter when this ran?
# True means some earlier import beat us to it and the setting may not
# take effect — a fact worth reporting rather than hiding.
TORCH_PRESENT_AT_IMPORT = "torch" in sys.modules

# What the environment already said, before this module touched it. An
# operator-supplied value is respected; this module is a default, not an
# override, so a deliberate template setting still wins.
INHERITED = os.environ.get(VARIABLE)

if not INHERITED:
    os.environ[VARIABLE] = SETTING

APPLIED = os.environ.get(VARIABLE)


def evidence() -> dict:
    """What actually happened, for the worker's startup report.

    `ordering_ok` is the claim that matters. It is False when torch was
    already imported before this module ran, which is the one case where
    the variable is set and the allocator ignores it.
    """
    return {
        "variable": VARIABLE,
        "value": APPLIED,
        "source": "inherited" if INHERITED else "set-by-cudaenv",
        "torch_imported_before_cudaenv": TORCH_PRESENT_AT_IMPORT,
        "torch_cuda_initialized_before_cudaenv": _cuda_was_initialized(),
        "ordering_ok": not TORCH_PRESENT_AT_IMPORT,
    }


def _cuda_was_initialized() -> bool:
    """True only if torch is loaded AND its CUDA context is already up.

    Deliberately does not import torch: importing it here would create the
    very condition the check exists to detect.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_initialized())
    except Exception:
        return False
