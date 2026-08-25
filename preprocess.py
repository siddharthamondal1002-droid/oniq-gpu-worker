"""The image_preprocess workload.

CUDA-only by default: `run_gpu_op(require_cuda=True)` is the default and
raises `cuda-unavailable` rather than falling back, so a CPU host cannot
produce a green job that looks like GPU proof. The one escape hatch is the
explicit env opt-in ONIQ_ALLOW_CPU_FALLBACK=1, which exists for harness
rigs on machines without a card — including machines where torch is not
installed at all, which run_gpu_op treats the same as CUDA being absent.

torch is imported lazily inside run_gpu_op; the CPU fallback resizes with
PIL so the contract/storage/handler suites run on rigs holding only the
image's three pinned deps (runpod, pillow, boto3).
"""

from __future__ import annotations

import os
import time

from PIL import Image

import contract


class GpuUnavailable(Exception):
    code = "cuda-unavailable"

    def __init__(self, message: str = "CUDA is not available on this host"):
        super().__init__(message)
        self.message = message


def _cpu_fallback_allowed() -> bool:
    return os.environ.get("ONIQ_ALLOW_CPU_FALLBACK") == "1"


def run_gpu_op(fn_gpu, fn_cpu, *, require_cuda: bool = True):
    """Run fn_gpu(torch) on CUDA, or refuse.

    Returns (result, metrics). With require_cuda=True (the default) a host
    without working CUDA — torch missing counts — raises GpuUnavailable
    instead of silently computing on CPU. Only the explicit
    ONIQ_ALLOW_CPU_FALLBACK=1 opt-in routes to fn_cpu().
    """
    try:
        import torch  # lazy: the worker image has it, test rigs may not
    except ImportError:
        torch = None

    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        result = fn_gpu(torch)
        props = torch.cuda.get_device_properties(0)
        return result, {
            "device": "cuda",
            "gpu_name": torch.cuda.get_device_name(0),
            "vram_total_mb": int(props.total_memory / (1024 * 1024)),
            "vram_peak_mb": int(
                torch.cuda.max_memory_allocated() / (1024 * 1024)
            ),
        }

    if require_cuda and not _cpu_fallback_allowed():
        raise GpuUnavailable()

    return fn_cpu(), {
        "device": "cpu",
        "gpu_name": None,
        "vram_total_mb": None,
        "vram_peak_mb": None,
    }


def _decode(input_path: str) -> Image.Image:
    Image.MAX_IMAGE_PIXELS = contract.MAX_IMAGE_PIXELS
    try:
        with Image.open(input_path) as im:
            im.load()
            return im.convert("RGB")
    except Image.DecompressionBombError as exc:
        raise contract.ContractError(
            "input-too-large", "image exceeds the decode pixel bound"
        ) from exc
    except Exception as exc:
        raise contract.ContractError(
            "invalid-image", "input is not a decodable image"
        ) from exc


def run(job: dict, input_path: str, output_path: str) -> dict:
    """Decode, resize (CUDA by default), re-encode. Returns metrics only."""
    started = time.monotonic()
    params = job["params"]
    image = _decode(input_path)

    target = params["target_max_dim"]
    scale = min(1.0, target / max(image.width, image.height))
    out_w = max(1, round(image.width * scale))
    out_h = max(1, round(image.height * scale))

    def _gpu_resize(torch):
        import torch.nn.functional as F

        tensor = (
            torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
            .reshape(image.height, image.width, 3)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device="cuda", dtype=torch.float32)
        )
        resized = F.interpolate(
            tensor, size=(out_h, out_w), mode="bilinear", antialias=True
        )
        out_bytes = (
            resized.clamp(0, 255)
            .to(dtype=torch.uint8)
            .squeeze(0)
            .permute(1, 2, 0)
            .contiguous()
            .cpu()
            .numpy()
            .tobytes()
        )
        return Image.frombytes("RGB", (out_w, out_h), out_bytes)

    def _cpu_resize():
        return image.resize((out_w, out_h), Image.Resampling.LANCZOS)

    result_image, metrics = run_gpu_op(_gpu_resize, _cpu_resize)

    fmt = params["format"]
    save_kwargs = {}
    if fmt in ("jpeg", "webp"):
        save_kwargs["quality"] = params["quality"]
    result_image.save(output_path, format=fmt.upper(), **save_kwargs)

    return {
        "width": result_image.width,
        "height": result_image.height,
        "format": fmt,
        "output_bytes": os.path.getsize(output_path),
        "duration_ms": int((time.monotonic() - started) * 1000),
        **metrics,
    }
