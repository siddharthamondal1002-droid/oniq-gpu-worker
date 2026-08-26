"""The video_generate workload — LTX-Video image-to-video on CUDA.

Server-controlled by construction: the model is whatever the image baked
at build time into MODEL_DIR (never downloaded at job time —
local_files_only), the resolution/frame-count/fps are contract
constants, and the sampler settings live here. The caller supplies a
motion prompt and nothing else.

Unlike image_preprocess there is NO CPU fallback, ever: a 2B video
diffusion pass on CPU would blow the runtime ceiling and the point of
the workload is the card. A host without working CUDA refuses with
cuda-unavailable regardless of ONIQ_ALLOW_CPU_FALLBACK.

Heavy imports (torch, diffusers, imageio) happen lazily inside the
functions so the contract/handler suites still run on rigs holding only
the base pins; the orchestration, metrics and mp4 encode are tested on
CPU by injecting a fake pipeline through `load_pipeline`.
"""

from __future__ import annotations

import os
import time

from PIL import Image

import contract
from preprocess import GpuUnavailable, _decode

MODEL_DIR = "/app/models/ltx"
MODEL_ID_FILE = "/app/models/MODEL_ID"

# Sampler settings — server decisions, deliberately boring for the first
# measurement: a fixed seed so a re-run is comparable, a stock negative
# prompt, and a step count that respects a distilled checkpoint if the
# build resolved one (recorded in MODEL_ID as "...#distilled").
SEED = 42
NEGATIVE_PROMPT = (
    "worst quality, inconsistent motion, blurry, jittery, distorted"
)
STEPS_DISTILLED = 8
STEPS_FULL = 30


def model_id() -> str:
    try:
        with open(MODEL_ID_FILE, encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return "missing"


def _load_real_pipeline():
    """Load the baked pipeline onto CUDA. Never touches the network."""
    import torch
    from diffusers import LTXImageToVideoPipeline

    pipe = LTXImageToVideoPipeline.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, local_files_only=True
    )
    pipe.to("cuda")
    pipe.vae.enable_tiling()
    return pipe


def _generate(pipe, image, prompt: str):
    """One deterministic I2V pass. Returns a list of PIL frames."""
    steps = (
        STEPS_DISTILLED if "distilled" in model_id() else STEPS_FULL
    )
    generator = None
    try:
        import torch

        if torch.cuda.is_available():
            generator = torch.Generator(device="cuda").manual_seed(SEED)
    except ImportError:
        pass
    result = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        width=contract.VIDEO_WIDTH,
        height=contract.VIDEO_HEIGHT,
        num_frames=contract.VIDEO_NUM_FRAMES,
        num_inference_steps=steps,
        generator=generator,
    )
    return result.frames[0]


def _encode_mp4(frames, output_path: str) -> None:
    """h264/yuv420p at the contract fps — playable everywhere."""
    import numpy as np
    import imageio.v2 as imageio

    writer = imageio.get_writer(
        output_path,
        fps=contract.VIDEO_FPS,
        codec="libx264",
        quality=None,
        pixelformat="yuv420p",
        output_params=["-crf", "23", "-preset", "medium"],
    )
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame))
    finally:
        writer.close()


def _fit_to_canvas(image: Image.Image) -> Image.Image:
    """Center-crop-and-scale the input onto the fixed video canvas."""
    target_w, target_h = contract.VIDEO_WIDTH, contract.VIDEO_HEIGHT
    scale = max(target_w / image.width, target_h / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    left = (resized.width - target_w) // 2
    top = (resized.height - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


def run(job: dict, input_path: str, output_path: str, load_pipeline=None) -> dict:
    """Decode, generate on CUDA, encode mp4. Returns measured metrics only.

    `load_pipeline` exists for the CPU test rig: injecting a fake
    pipeline exercises everything here except the CUDA pass itself.
    """
    started = time.monotonic()
    image = _fit_to_canvas(_decode(input_path))

    if load_pipeline is None:
        try:
            import torch
        except ImportError:
            torch = None
        if torch is None or not torch.cuda.is_available():
            raise GpuUnavailable(
                "video_generate requires CUDA; there is no CPU fallback"
            )
        load_pipeline = _load_real_pipeline

    load_started = time.monotonic()
    pipe = load_pipeline()
    model_load_ms = int((time.monotonic() - load_started) * 1000)

    metrics = {
        "device": "cpu",
        "gpu_name": None,
        "vram_total_mb": None,
        "vram_peak_mb": None,
    }
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    infer_started = time.monotonic()
    frames = _generate(pipe, image, job["params"]["prompt"])
    inference_ms = int((time.monotonic() - infer_started) * 1000)

    if torch is not None and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        metrics = {
            "device": "cuda",
            "gpu_name": torch.cuda.get_device_name(0),
            "vram_total_mb": int(props.total_memory / (1024 * 1024)),
            "vram_peak_mb": int(
                torch.cuda.max_memory_allocated() / (1024 * 1024)
            ),
        }

    encode_started = time.monotonic()
    _encode_mp4(frames, output_path)
    encode_ms = int((time.monotonic() - encode_started) * 1000)

    return {
        "model": model_id(),
        "model_load_ms": model_load_ms,
        "inference_ms": inference_ms,
        "encode_ms": encode_ms,
        "frames": len(frames),
        "fps": contract.VIDEO_FPS,
        "video_seconds": round(len(frames) / contract.VIDEO_FPS, 2),
        "width": contract.VIDEO_WIDTH,
        "height": contract.VIDEO_HEIGHT,
        "format": "mp4",
        "output_bytes": os.path.getsize(output_path),
        "duration_ms": int((time.monotonic() - started) * 1000),
        **metrics,
    }
