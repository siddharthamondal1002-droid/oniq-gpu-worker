"""A bounded look at what the worker just made, carried in the job's own reply.

WHY THIS EXISTS. ONIQ can generate artifacts it cannot see. Every quality
verdict in this project has depended on someone handing over a signed URL or
a public read base for the R2 bucket, because the bucket is private by owner
directive 2026-08-29 and the harness holds no storage credential. On
2026-08-29 that stopped the benchmark outright: the repository variable
R2_PUBLIC_BASE_URL was set to the string "on", the frame reader refused it
correctly, and there was no way to look at a reference the worker had
already drawn.

Asking for a URL would have unblocked one run. This unblocks every run: the
worker downscales what it produced and returns it inside the reply it was
already sending, so seeing an artifact needs exactly the credential that
submitting the job needed and nothing more.

IT IS NOT A DELIVERY CHANNEL. The artifact still goes to R2 and R2 remains
the only place it lives. What comes back here is a thumbnail — long edge
capped, JPEG, a handful of frames, and a hard total budget — because a
serverless reply is not a file transfer and a preview that could grow with
the clip would eventually fail the job it was meant to describe.

IT IS OFF UNLESS ASKED. Production callers never set `preview`, so their
replies are byte-for-byte what they were. Only the benchmark harness asks.
"""

from __future__ import annotations

import base64
import io

import contract


def sample_indices(count: int, want: int) -> list[int]:
    """Evenly spaced frame indices, ALWAYS including the first and last.

    The ends carry most of the evidence: frame 0 says whether the model kept
    the reference, and the last frame says whether it survived the clip. A
    sampler that drifted off either end would drop the two frames an
    inspection actually turns on.
    """
    if count <= 0 or want <= 0:
        return []
    if count <= want:
        return list(range(count))
    if want == 1:
        return [0]
    step = (count - 1) / (want - 1)
    return sorted({int(round(i * step)) for i in range(want)})


def _to_jpeg(frame, long_edge: int, quality: int) -> bytes:
    from PIL import Image
    import numpy as np

    if not isinstance(frame, Image.Image):
        array = np.asarray(frame)
        if array.dtype != np.uint8:
            # Pipelines that hand back floats are in 0..1; scaling by 255 is
            # the conversion, and clipping first stops a stray overshoot from
            # wrapping around into black.
            array = (np.clip(array, 0.0, 1.0) * 255).astype(np.uint8)
        frame = Image.fromarray(array)
    frame = frame.convert("RGB")
    width, height = frame.size
    longest = max(width, height)
    if longest > long_edge:
        scale = long_edge / longest
        frame = frame.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.LANCZOS,
        )
    buffer = io.BytesIO()
    frame.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def encode_frames(
    frames,
    *,
    want: int = contract.PREVIEW_MAX_FRAMES,
    long_edge: int = contract.PREVIEW_LONG_EDGE,
    quality: int = contract.PREVIEW_JPEG_QUALITY,
    budget: int = contract.PREVIEW_MAX_BYTES,
    encoder=None,
) -> list[dict]:
    """Sampled frames as base64 JPEG, stopping at the budget.

    The budget is checked BEFORE a frame is added, never after, so the reply
    can never exceed it — a preview that overflowed the provider's response
    limit would take the measurements down with it, which is the opposite of
    the job.
    """
    encode = encoder or (lambda f: _to_jpeg(f, long_edge, quality))
    frames = list(frames)
    out: list[dict] = []
    spent = 0
    for index in sample_indices(len(frames), want):
        try:
            raw = encode(frames[index])
        except Exception:  # noqa: BLE001 - a preview never fails the job
            continue
        text = base64.b64encode(raw).decode("ascii")
        if spent + len(text) > budget:
            break
        spent += len(text)
        out.append({"i": index, "bytes": len(raw), "b64": text})
    return out


def of_file(path: str, **kw) -> list[dict]:
    """One preview of a still already written to disk."""
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.load()
            return encode_frames([image], want=1, **kw)
    except Exception:  # noqa: BLE001 - a preview never fails the job
        return []


def wanted(job: dict) -> bool:
    """Did the CALLER ask for one. Absent means no, and production never asks."""
    return bool(job.get("preview"))
