"""Where a model component's weights actually live.

Owner directive, 2026-08-30: move the baked weights out of the container
image and onto a RunPod network volume, so that changing the worker stops
costing a full cold pull of a 25 GiB image.

The seam exists so that move can happen WITHOUT a flag day. Each
component is resolved on its own: the volume wins when it actually holds
that component, and the baked path is used otherwise. So a weightless
image with a populated volume works, a baked image with no volume works,
and — the case that matters during the migration — a volume holding only
some components works too, with the rest still served from the image.
Nothing here fails closed on an empty volume, because an empty volume is
what every worker sees before the first hydration finishes.

`MODEL_ROOT` is set by the TEMPLATE, never by a job. The owner's standing
constraint is that a user may not specify GPU, provider, endpoint, model,
checkpoint, runtime, budget, precision or offloading strategy; where the
weights are read from belongs to the same list, so `resolve` takes no
caller input at all and there is no code path from job input to this
module.
"""

from __future__ import annotations

import os

# Baked into the image by the Dockerfile. Still the fallback, and still
# the whole answer for any worker running an image built before the
# migration.
BAKED_ROOT = "/app/models"

# RunPod mounts a serverless network volume here. Overridable by the
# template only, because a provider that changes its mount path should
# not need a code change — but NOT by a job.
VOLUME_ROOT = os.environ.get("MODEL_ROOT") or "/runpod-volume/models"


def _populated(path: str) -> bool:
    """Does this directory hold anything at all?

    Deliberately cheap and deliberately shallow: one listdir, no walk, no
    size arithmetic. The expensive integrity question — are these the
    right files, at the pinned revision, complete — belongs to the
    hydrator, which is the thing that can actually fix a bad answer. A
    per-request deep check on a cold worker would pay that cost on every
    job forever.
    """
    try:
        with os.scandir(path) as it:
            return any(True for _ in it)
    except OSError:
        return False


def resolve(component: str) -> str:
    """Absolute path to one component's directory.

    `component` is a fixed identifier from this codebase ("ltx", "story",
    "piper"), never a caller-supplied string — see the module docstring.
    """
    candidate = os.path.join(VOLUME_ROOT, component)
    if _populated(candidate):
        return candidate
    return os.path.join(BAKED_ROOT, component)


def resolve_file(name: str) -> str:
    """Absolute path to one of the small marker files the bakes write
    (MODEL_ID, STORY_MODEL_ID, LTX_REPO). Same rule as `resolve`, applied
    to a file: the volume's copy wins when it exists."""
    candidate = os.path.join(VOLUME_ROOT, name)
    if os.path.isfile(candidate):
        return candidate
    return os.path.join(BAKED_ROOT, name)


def where() -> dict:
    """What resolved to what — for the worker's own report, so a job that
    ran off the volume can be told apart from one that ran off the image
    without inferring it from timings."""
    return {
        "volume_root": VOLUME_ROOT,
        "baked_root": BAKED_ROOT,
        "volume_present": _populated(VOLUME_ROOT),
        "components": {c: resolve(c) for c in ("ltx", "story", "piper")},
    }
