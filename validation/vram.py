"""What each candidate would actually need on a GPU, and which GPU that is.

Answers questions 4-6 of the owner's benchmark brief (actual VRAM
requirement / can the A5000 run it / minimum practical GPU) from the byte
counts model_registry measured, plus the shapes read out of each model's own
config. No parameter counts, no round numbers remembered from a README.

WHAT IS MEASURED AND WHAT IS PROJECTED — the distinction the brief insists on
("Treat those numbers as configuration guidance, NOT as proof for ONIQ"):

  MEASURED   weight bytes per component, straight from the registry's file
             listing. Exact. A repository either ships those bytes or not.
  PROJECTED  activation and VAE-decode working sets, computed here from the
             published architecture with the formulas below. Every constant
             is named and adjustable. A projection is a reason to schedule a
             probe, never a reason to declare a model deployable.

So `plan()` returns both, separately, and nothing in this module ever emits a
verdict that hides which half it rests on.

THE THREE PLACES VIDEO MODELS ACTUALLY RUN OUT OF MEMORY, in the order they
bite:

  1. Resident weights. Trivial arithmetic, and the only one a parameter count
     predicts. It is also the one offloading defeats most completely.
  2. The DiT's per-block working set — tokens x hidden x dtype x K. Grows with
     FRAMES and AREA, not with parameters, which is why a 2B model at 720p can
     want more than a 14B model at 480p.
  3. VAE decode. Usually the real ceiling: a full-length latent decoded in one
     pass materialises frames x H x W x channels at full resolution. For a
     97-frame 704x480 clip that is several GiB before the pipeline has done
     anything clever, which is why every one of these models ships tiled or
     chunked decoding. Modelled both ways here, because "it OOMs" and "it OOMs
     unless you enable the thing the model card tells you to enable" are
     different findings.
"""

from __future__ import annotations

from dataclasses import dataclass

GIB = 1024**3

# Peak transient bytes per token per hidden unit inside one transformer block,
# as a multiple of the element size: the residual stream and its normed copy,
# q/k/v, the attention output, and the MLP's expansion. Flash/SDPA attention
# never materialises the T x T matrix, so this stays linear in tokens.
# Deliberately generous — a projection that under-reports VRAM is worse than
# useless, because it is the one that gets believed until the OOM.
ACTIVATION_K = 12

# Feature-map multiplier during VAE decode: how many full-resolution maps the
# up-blocks hold live at once.
VAE_DECODE_K = 6

# TILED DECODE IS BOUNDED BY THE TILE, NOT BY THE CLIP. `enable_tiling()` — which
# ONIQ's own worker already calls — splits the decode SPATIALLY, so the biggest
# feature map is a tile's worth of pixels however large the canvas is. Modelling
# tiling as a frame-count cap alone was wrong by more than an order of magnitude
# against the one real measurement ONIQ owns (see ANCHOR below).
VAE_TILE_PIXELS = 256 * 256
VAE_DECODE_TILED_K = 2

# THE ONE REAL NUMBER. RunPod job b1971009-4b48-46b2-b7b1-5e3905d56d7e-u1:
# LTX-Video 2B on an A5000, bf16, every component resident, vae.enable_tiling()
# on, 9 frames at 704x480, reported vram_peak_mb 13837 (13.51 GiB) with 13.23
# GiB of that being weights. So the entire working set measured 0.28 GiB.
#
# Every tiled figure in this module inherits a constant fitted to THAT SINGLE
# POINT. It is a real anchor and it is one anchor: it pins the model at 9 frames
# on one canvas with one architecture, and says nothing about how the term grows.
# `anchor_check()` prints projection against measurement on every run so the fit
# is visible rather than assumed, and any probe that follows should re-fit it.
ANCHOR = {
    "job": "b1971009-4b48-46b2-b7b1-5e3905d56d7e-u1",
    "model": "LTX-Video 2B",
    "gpu": "NVIDIA RTX A5000",
    "measured_peak_bytes": 13837 * 1024 * 1024,
    "resident_weight_bytes": int(13.23 * GIB),
    "width": 704,
    "height": 480,
    "frames": 9,
}

# CUDA context, cuDNN/cuBLAS workspaces, allocator fragmentation. Nominal VRAM
# is never all usable, and a plan that assumes it is will OOM at 23.6 GiB on a
# "24 GB" card.
USABLE_FRACTION = 0.93

# The hardware the owner listed, nominal GiB. A GPU earns a place in the
# matrix only where a model/configuration actually needs it — the brief is
# explicit that a 14B label is not a reason to pay for an A100.
GPUS: tuple[tuple[str, int], ...] = (
    ("RTX A5000 24GB", 24),
    ("RTX 4090 24GB", 24),
    ("RTX 5090 32GB", 32),
    ("A6000 48GB", 48),
    ("A100 40GB", 40),
    ("A100 80GB", 80),
    ("H100 80GB", 80),
)

# Bytes per element. fp8 halves the transformer and nothing else: text
# encoders and VAEs are not what fp8 checkpoints quantise.
DTYPE_BYTES = {"fp32": 4, "bf16": 2, "fp16": 2, "fp8": 1, "int8": 1}


@dataclass(frozen=True)
class Shape:
    """What the model is being asked to produce."""

    width: int
    height: int
    frames: int


@dataclass(frozen=True)
class Arch:
    """Architecture, read from the model's own configs — never assumed."""

    hidden: int
    layers: int
    patch_spatial: int = 1  # DiT patch size ON TOP of VAE compression
    patch_temporal: int = 1
    vae_spatial: int = 8  # pixel -> latent spatial compression
    vae_temporal: int = 4  # pixel -> latent temporal compression
    vae_channels: int = 128  # widest decoder feature map


@dataclass(frozen=True)
class Config:
    """One benchmark configuration: a model at a precision, on a strategy."""

    precision: str = "bf16"
    offload: str = "none"  # none | model | sequential
    vae_tile_frames: int | None = None  # None = decode the clip in one pass


def latent_grid(shape: Shape, arch: Arch) -> tuple[int, int, int]:
    """Latent frames, height, width after VAE compression and DiT patching."""
    lf = 1 + max(0, shape.frames - 1) // arch.vae_temporal
    lh = shape.height // arch.vae_spatial
    lw = shape.width // arch.vae_spatial
    return (
        max(1, lf // arch.patch_temporal),
        max(1, lh // arch.patch_spatial),
        max(1, lw // arch.patch_spatial),
    )


def tokens(shape: Shape, arch: Arch) -> int:
    """Sequence length the DiT attends over. The driver of cost 2."""
    lf, lh, lw = latent_grid(shape, arch)
    return lf * lh * lw


def activation_bytes(shape: Shape, arch: Arch, *, dtype: str = "bf16",
                     k: int = ACTIVATION_K) -> int:
    """PROJECTED per-block working set. Linear in tokens, not quadratic."""
    return tokens(shape, arch) * arch.hidden * DTYPE_BYTES[dtype] * k


def vae_decode_bytes(shape: Shape, arch: Arch, *, dtype: str = "bf16",
                     tile_frames: int | None = None,
                     k: int = VAE_DECODE_K,
                     tile_pixels: int = VAE_TILE_PIXELS,
                     tiled_k: int = VAE_DECODE_TILED_K) -> int:
    """PROJECTED decode peak, whole-clip or tiled.

    Whole-clip decode is what a naive pipeline does and what OOMs first. Tiled
    decode is what the same model does once `enable_tiling()` is on, and it is
    bounded in BOTH dimensions: the frame chunk caps time, the tile caps area.
    Capping only time overestimated ONIQ's own production configuration by more
    than tenfold — the anchor caught it.
    """
    if tile_frames is None:
        return (shape.frames * shape.height * shape.width
                * arch.vae_channels * DTYPE_BYTES[dtype] * k)
    frames = min(shape.frames, tile_frames)
    area = min(shape.height * shape.width, tile_pixels)
    return frames * area * arch.vae_channels * DTYPE_BYTES[dtype] * tiled_k


def anchor_check(arch: Arch) -> dict:
    """Projection against the one configuration that has a real measurement.

    Reported on every run. A model whose arithmetic cannot reproduce the single
    case ONIQ has actually observed has no business ranking eight it has not.
    """
    shape = Shape(ANCHOR["width"], ANCHOR["height"], ANCHOR["frames"])
    working = max(
        activation_bytes(shape, arch),
        vae_decode_bytes(shape, arch, tile_frames=ANCHOR["frames"]),
    )
    projected = ANCHOR["resident_weight_bytes"] + working
    measured = ANCHOR["measured_peak_bytes"]
    return {
        "projected_bytes": projected,
        "measured_bytes": measured,
        "ratio": round(projected / measured, 3),
        "projected_working_set": working,
        "measured_working_set": measured - ANCHOR["resident_weight_bytes"],
    }


def weight_bytes(roles: dict[str, int], *, precision: str,
                 shipped: str | dict[str, str] = "bf16") -> dict[str, int]:
    """MEASURED weights, rescaled to the precision the run would actually use.

    `shipped` is per-role, because IT VARIES WITHIN ONE REPOSITORY. Wan2.1
    ships its transformer in fp32 and Wan2.2 ships the same-sized transformer
    in fp32 beside a bf16 text encoder; treating a repository as having one
    dtype double-counts some components and halves others. The registry reads
    `torch_dtype` per component and hands the map in here.

    Only the transformer rescales when the target is fp8: an fp8 checkpoint is
    an fp8 DiT beside an unchanged text encoder and VAE, and pretending
    otherwise would flatter every quantised row in the matrix. Everything
    rescales between float widths, because loading a fp32 checkpoint in bf16
    is the ordinary thing to do and halves it.
    """
    out = {}
    for role, size in roles.items():
        role_shipped = shipped.get(role, "bf16") if isinstance(shipped, dict) else shipped
        target = precision
        if precision in ("fp8", "int8") and not role.startswith("transformer"):
            # Quantised builds quantise the DiT only. The rest loads at the
            # widest sensible compute dtype, not at the DiT's.
            target = "bf16"
        ratio = DTYPE_BYTES[target] / DTYPE_BYTES[role_shipped]
        out[role] = int(size * ratio)
    return out


def resident_bytes(weights: dict[str, int], offload: str) -> tuple[int, str]:
    """Weight bytes the GPU must hold at once, and why that is the number.

    `model` offload is the interesting one for the mixture-of-experts case:
    Wan2.2's two experts are used at different denoising stages, never in the
    same forward pass, so one-at-a-time residency is the honest floor — while
    holding both is what a pipeline does by default.
    """
    if not weights:
        return 0, "no weights measured"
    if offload == "none":
        return sum(weights.values()), "every component resident at once"
    if offload == "model":
        role, size = max(weights.items(), key=lambda kv: kv[1])
        return size, f"largest single component resident ({role})"
    if offload == "sequential":
        transformer = max(
            (v for k, v in weights.items() if k.startswith("transformer")),
            default=0,
        )
        return transformer // 8, "transformer streamed per-block (~1/8 resident)"
    raise ValueError(f"unknown offload strategy {offload!r}")


def plan(*, roles: dict[str, int], arch: Arch, shape: Shape,
         config: Config, shipped: str | dict[str, str] = "bf16") -> dict:
    """Everything one configuration needs, measured and projected kept apart."""
    weights = weight_bytes(roles, precision=config.precision, shipped=shipped)
    resident, why = resident_bytes(weights, config.offload)
    compute_dtype = "bf16" if config.precision in ("fp8", "int8") else config.precision
    activations = activation_bytes(shape, arch, dtype=compute_dtype)
    decode = vae_decode_bytes(
        shape, arch, dtype=compute_dtype, tile_frames=config.vae_tile_frames
    )

    # Denoise and decode do not overlap: the DiT's working set is freed before
    # the VAE runs. Peak is therefore the worse of the two stages, not a sum —
    # summing them would invent a ceiling no run ever hits.
    denoise_peak = resident + activations
    decode_peak = resident + decode
    peak = max(denoise_peak, decode_peak)
    return {
        "measured_weight_bytes": sum(weights.values()),
        "resident_weight_bytes": resident,
        "residency_note": why,
        "projected_activation_bytes": activations,
        "projected_decode_bytes": decode,
        "denoise_peak_bytes": denoise_peak,
        "decode_peak_bytes": decode_peak,
        "peak_bytes": peak,
        "binding_stage": "decode" if decode_peak >= denoise_peak else "denoise",
        "tokens": tokens(shape, arch),
        "precision": config.precision,
        "offload": config.offload,
        "vae_tile_frames": config.vae_tile_frames,
    }


def fits(peak_bytes: int, gpu_gib: int, *,
         usable: float = USABLE_FRACTION) -> bool:
    return peak_bytes <= gpu_gib * GIB * usable


def minimum_gpu(peak_bytes: int, gpus=GPUS, *,
                usable: float = USABLE_FRACTION) -> str | None:
    """The smallest listed card this configuration fits on.

    Smallest by VRAM, then by the order the owner listed them — so a 24 GB
    consumer card is never passed over for an 80 GB accelerator that the
    arithmetic does not require.
    """
    ordered = sorted(range(len(gpus)), key=lambda i: (gpus[i][1], i))
    for i in ordered:
        name, gib = gpus[i]
        if fits(peak_bytes, gib, usable=usable):
            return name
    return None


def gib(n: int) -> float:
    return round(n / GIB, 2)
