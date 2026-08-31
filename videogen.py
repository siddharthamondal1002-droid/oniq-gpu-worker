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
import modelroot
import preview
from preprocess import GpuUnavailable, _decode

# Both resolve LAZILY, through modelroot — None means "ask on each call".
# Lazy, not import-time, because a warm worker can see the volume change
# underneath it: hydration populates /runpod-volume/models while the
# process is already running, and the next job on that same worker must
# see the new answer. Tests still override by setting these attributes,
# which is why they stay module-level names rather than becoming calls.
MODEL_DIR = None
MODEL_ID_FILE = None


def _model_dir() -> str:
    return MODEL_DIR or modelroot.resolve_production("ltx")


def _model_id_file() -> str:
    return MODEL_ID_FILE or modelroot.resolve_production_file("MODEL_ID")

# Sampler settings. THE STEP COUNT AND GUIDANCE NOW COME FROM THE CHECKPOINT
# (ltxcaps.py), not from this file, because they are properties of the weights
# rather than of the worker — see that module's header for why a substring test
# against a repository name was never evidence of distillation.
#
# SEED IS A FALLBACK, NOT A POLICY. It stays 42 only for callers that send no
# seed of their own, so nothing that worked before changes. Every production
# caller now derives a seed from job/scene/shot/attempt: a module-level
# constant made the ten retry attempts authorised on 2026-08-31 re-sample the
# SAME image ten times, which is a retry budget that cannot succeed.
SEED = 42

# The fallback negative prompt, for callers that send none. It is deliberately
# the ORIGINAL string: this change gives callers the ability to send a better,
# shot-specific one, and does not silently retune the default underneath a
# caller that did not ask.
NEGATIVE_PROMPT = (
    "worst quality, inconsistent motion, blurry, jittery, distorted"
)


def _profile():
    """The inference settings this baked checkpoint should be sampled with."""
    import ltxcaps

    caps = ltxcaps.inspect_checkpoint(_model_dir(), model_id())
    return caps, ltxcaps.inference_profile(caps)


def _sampler_kwargs(job: dict, profile: dict) -> dict:
    """Merge the checkpoint's profile with this job's own two inputs."""
    params = job.get("params") or {}
    seed = params.get("seed", SEED)
    negative = params.get("negative_prompt")
    if negative is None:
        negative = NEGATIVE_PROMPT
    return {
        "negative_prompt": negative,
        "num_inference_steps": profile["num_inference_steps"],
        "guidance_scale": profile["guidance_scale"],
        "guidance_rescale": profile["guidance_rescale"],
        "seed": seed,
    }


def _generator(seed: int):
    """A CUDA generator pinned to this job's seed, or None off-GPU."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.Generator(device="cuda").manual_seed(int(seed))
    except ImportError:
        pass
    return None


class OutOfMemory(RuntimeError):
    """The canvas did not fit. Refused, never silently downscaled."""


def _is_oom(exc: BaseException) -> bool:
    """Is this the card running out, rather than the code being wrong?

    Matched on the exception TYPE where torch offers one, and on the message
    only as a fallback — a substring test alone would also catch a prompt that
    merely contained the words.
    """
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except (ImportError, AttributeError):
        pass
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text


def model_id() -> str:
    try:
        with open(_model_id_file(), encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return "missing"


# THE CONDITIONING STRENGTH, and it is the DOCUMENTED default, not a tuned one.
#
# LTXConditionPipeline takes conditions as LTXVideoCondition(image|video,
# frame_index, strength); strength 1.0 is "hold this frame as given". That is
# the documented value and the one this worker sends, because no measurement
# on this hardware exists to justify any other and inventing one would be the
# same class of mistake the audit was opened to fix — a number nobody chose,
# nobody recorded, and nobody could compare a good clip against a bad one on.
# A future tuning pass changes this WITH a figure next to it.
CONDITION_STRENGTH = 1.0
CONDITION_FRAME_INDEX = 0


def _video_condition(image):
    """The explicit frame-0 condition, or None when it cannot be built.

    WHY THIS IS A SEPARATE FUNCTION. Two callers need it to degrade rather
    than explode: the CPU contract rig has no diffusers at all, and a
    checkpoint whose components do not support the condition pipeline must
    still be able to animate a still. Returning None puts both on the plain
    image= path, which is exactly what the worker did before.
    """
    try:
        from diffusers import LTXVideoCondition
    except Exception:  # noqa: BLE001 - any import failure means "no condition"
        try:
            from diffusers.pipelines.ltx.pipeline_ltx_condition import (  # type: ignore
                LTXVideoCondition,
            )
        except Exception:  # noqa: BLE001
            return None
    try:
        return LTXVideoCondition(
            image=image,
            frame_index=CONDITION_FRAME_INDEX,
            strength=CONDITION_STRENGTH,
        )
    except TypeError:
        # A diffusers version whose condition object is shaped differently.
        # Degrading is honest; guessing at the shape is not.
        return None


def _load_real_pipeline():
    """Load the baked pipeline onto CUDA. Never touches the network.

    LTXConditionPipeline WHERE THE CHECKPOINT SUPPORTS IT, which this one
    does. It takes the identical five components as LTXImageToVideoPipeline —
    scheduler, vae, text_encoder, tokenizer, transformer — so nothing new is
    downloaded and nothing new is baked; `ltxcaps.inspect_checkpoint` proves
    that from the snapshot on disk rather than from the repository's name.
    What it adds is EXPLICIT control of the conditioning: which frame is being
    conditioned on, how hard to hold it (`strength`), and how much noise is
    added to it (`image_cond_noise_scale`). On the i2v pipeline those are
    implicit, unrecorded, and unavailable to a diagnosis — which is why a clip
    that drifted off its own opening frame could not be told apart from one
    that never held it.

    Falls back to LTXImageToVideoPipeline when the checkpoint or the installed
    diffusers cannot support conditioning. A fallback here is not a provider
    fallback: it is the same weights on the same GPU with less control.
    """
    import torch

    caps, _profile_unused = _profile()
    if caps.get("condition_pipeline_supported"):
        try:
            from diffusers import LTXConditionPipeline

            pipe = LTXConditionPipeline.from_pretrained(
                _model_dir(), torch_dtype=torch.bfloat16, local_files_only=True
            )
            pipe.to("cuda")
            pipe.vae.enable_tiling()
            return pipe
        except (ImportError, AttributeError, TypeError, ValueError, OSError):
            # Named exceptions only: an OOM or a CUDA fault must NOT be caught
            # here and silently retried on a different pipeline class.
            pass

    from diffusers import LTXImageToVideoPipeline

    pipe = LTXImageToVideoPipeline.from_pretrained(
        _model_dir(), torch_dtype=torch.bfloat16, local_files_only=True
    )
    pipe.to("cuda")
    pipe.vae.enable_tiling()
    return pipe


def _load_real_text_pipeline():
    """The SAME baked snapshot, opened as text-to-video.

    ONIQ's in-house image engine is not a second model: LTXPipeline and
    LTXImageToVideoPipeline read the identical transformer/vae/
    text_encoder/tokenizer already in /app/models/ltx. Nothing is
    downloaded, nothing new is baked, and local_files_only keeps that
    true even if the network were reachable.
    """
    import torch
    from diffusers import LTXPipeline

    pipe = LTXPipeline.from_pretrained(
        _model_dir(), torch_dtype=torch.bfloat16, local_files_only=True
    )
    pipe.to("cuda")
    pipe.vae.enable_tiling()
    return pipe


def _generate_still(pipe, prompt: str, sampler: dict):
    """One deterministic T2V pass at the shortest legal length. Returns
    the frames; frame 0 is the still the caller keeps.

    Every sampler value is now passed EXPLICITLY. Guidance in particular was
    never sent before, so the pipeline's own default applied silently and no
    record of it reached the job's metrics — a value nobody chose and nobody
    could see.
    """
    result = pipe(
        prompt=prompt,
        negative_prompt=sampler["negative_prompt"],
        width=contract.VIDEO_WIDTH,
        height=contract.VIDEO_HEIGHT,
        num_frames=contract.IMAGE_GEN_NUM_FRAMES,
        num_inference_steps=sampler["num_inference_steps"],
        guidance_scale=sampler["guidance_scale"],
        guidance_rescale=sampler["guidance_rescale"],
        generator=_generator(sampler["seed"]),
    )
    return result.frames[0]


def _generate(pipe, image, prompt: str, sampler: dict, profile: dict):
    """One deterministic I2V pass. Returns a list of PIL frames.

    The VAE decode settings travel with the sampler now. They are the
    pipeline's own documented values rather than numbers invented here, and
    they are sent explicitly for the same reason guidance is: a value that is
    never passed is a value that cannot be recorded, and an unrecorded value
    cannot be compared between a good clip and a bad one.
    """
    common = dict(
        prompt=prompt,
        negative_prompt=sampler["negative_prompt"],
        width=contract.VIDEO_WIDTH,
        height=contract.VIDEO_HEIGHT,
        num_frames=contract.VIDEO_NUM_FRAMES,
        num_inference_steps=sampler["num_inference_steps"],
        guidance_scale=sampler["guidance_scale"],
        guidance_rescale=sampler["guidance_rescale"],
        decode_timestep=profile["decode_timestep"],
        decode_noise_scale=profile["decode_noise_scale"],
        generator=_generator(sampler["seed"]),
    )
    condition = _video_condition(image) if profile.get("conditioning") else None
    if condition is not None:
        # EXPLICIT CONDITIONING. The frame is named, the strength is named,
        # and the noise added to it is named — all three reach the metrics, so
        # a clip that drifted off its opening frame can be told apart from one
        # that was never asked to hold it.
        result = pipe(
            conditions=[condition],
            image_cond_noise_scale=profile["image_cond_noise_scale"],
            **common,
        )
    else:
        result = pipe(image=image, **common)
    return result.frames[0], 1 if condition is not None else 0


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


# ------------------------------------------------------------- watermark
# The ONIQ mark (monetization resolution loop, 2026-08-27): small, bottom
# right, translucent — present unless the job carried the server-derived
# no-watermark entitlement. Burned at GENERATION so a stream-copy concat
# of clips carries the correct mark through to any long-form final with
# zero re-encode; a later clean purchase re-renders, exactly as the story
# product's watermark removal already does.
WATERMARK_TEXT = "ONIQ"
_WATERMARK_MARGIN = 14
_WATERMARK_SIZE = 22
_WATERMARK_ALPHA = 170


def _watermark_frame(frame: Image.Image) -> Image.Image:
    """One frame, marked. Pure PIL — no font files, no shell, no network."""
    from PIL import ImageDraw, ImageFont

    font = ImageFont.load_default(size=_WATERMARK_SIZE)
    base = frame.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    left, top, right, bottom = draw.textbbox((0, 0), WATERMARK_TEXT, font=font)
    x = base.width - (right - left) - _WATERMARK_MARGIN
    y = base.height - (bottom - top) - _WATERMARK_MARGIN
    # A soft dark shadow keeps the mark readable on light frames without
    # turning it into a box; the mark itself stays translucent white.
    draw.text((x + 1, y + 1), WATERMARK_TEXT, font=font, fill=(0, 0, 0, 110))
    draw.text((x, y), WATERMARK_TEXT, font=font, fill=(255, 255, 255, _WATERMARK_ALPHA))
    return Image.alpha_composite(base, overlay).convert("RGB")


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

    caps, profile = _profile()
    sampler = _sampler_kwargs(job, profile)

    infer_started = time.monotonic()
    try:
        frames, conditioning_count = _generate(
            pipe, image, job["params"]["prompt"], sampler, profile
        )
    except Exception as exc:  # noqa: BLE001 - re-raised, never swallowed
        # FAIL CLOSED ON OOM. The portrait canvas is 2.6x the pixels the old
        # one was, and the VRAM projection behind that choice is arithmetic,
        # not a measurement. A worker that quietly dropped to a smaller canvas
        # would reintroduce the upscale this change removes, and would do it
        # invisibly; refusing is the honest failure and the caller already
        # treats a failed clip as "this shot carries as a still".
        if _is_oom(exc):
            raise OutOfMemory(
                f"{contract.VIDEO_WIDTH}x{contract.VIDEO_HEIGHT} at "
                f"{contract.VIDEO_NUM_FRAMES} frames did not fit: {exc}"
            ) from exc
        raise
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

    # The server-derived entitlement, validated by the contract; absent
    # means marked — old callers can only ever produce the marked product.
    marked = job["params"].get("watermark", True)
    if marked:
        frames = [_watermark_frame(f) for f in frames]

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
        "watermarked": marked,
        "output_bytes": os.path.getsize(output_path),
        "duration_ms": int((time.monotonic() - started) * 1000),
        "seed": sampler["seed"],
        "negative_prompt_chars": len(sampler["negative_prompt"]),
        "conditioning_count": conditioning_count,
        # The class that actually ran, not the one model_index.json declares.
        "pipeline_used": type(pipe).__name__,
        "conditioning_strength": CONDITION_STRENGTH if conditioning_count else None,
        "upscaler_used": False,
        **__import__("ltxcaps").diagnostics(caps, profile),
        **metrics,
    }


# ---------------------------------------------------------------- concat
class ConcatRefused(Exception):
    """A concat input the assembler refuses. `code` is a stable token."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def run_image(job: dict, output_path: str, load_pipeline=None) -> dict:
    """ONIQ's in-house image engine: prompt -> still, on this worker's own
    GPU. Returns measured metrics only.

    The still is a CONDITIONING FRAME for the video stage, so it is
    written at the video canvas, losslessly, and carries no watermark —
    the mark belongs to the delivered film, burned by the stage that
    knows the entitlement of record.

    `load_pipeline` exists for the CPU test rig, exactly as in run().
    """
    started = time.monotonic()

    if load_pipeline is None:
        try:
            import torch
        except ImportError:
            torch = None
        if torch is None or not torch.cuda.is_available():
            raise GpuUnavailable(
                "image_generate requires CUDA; there is no CPU fallback"
            )
        load_pipeline = _load_real_text_pipeline

    load_started = time.monotonic()
    pipe = load_pipeline()
    model_load_ms = int((time.monotonic() - load_started) * 1000)

    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    infer_started = time.monotonic()
    caps, profile = _profile()
    sampler = _sampler_kwargs(job, profile)
    try:
        frames = _generate_still(pipe, job["params"]["prompt"], sampler)
    except Exception as exc:  # noqa: BLE001 - re-raised, never swallowed
        if _is_oom(exc):
            raise OutOfMemory(
                f"still at {contract.VIDEO_WIDTH}x{contract.VIDEO_HEIGHT} "
                f"did not fit: {exc}"
            ) from exc
        raise
    inference_ms = int((time.monotonic() - infer_started) * 1000)

    if not frames:
        raise contract.ContractError(
            "no-frames", "the image engine produced no frames"
        )
    still = frames[0]
    if still.mode != "RGB":
        still = still.convert("RGB")

    encode_started = time.monotonic()
    still.save(output_path, format=contract.IMAGE_GEN_FORMAT.upper())
    encode_ms = int((time.monotonic() - encode_started) * 1000)

    width, height = still.size
    result = {
        "ok": True,
        "op": "image_generate",
        "output_key": job["output_key"],
        "model": model_id(),
        "model_load_ms": model_load_ms,
        "inference_ms": inference_ms,
        "encode_ms": encode_ms,
        "width": width,
        "height": height,
        "format": contract.IMAGE_GEN_FORMAT,
        "output_bytes": os.path.getsize(output_path),
        "duration_ms": int((time.monotonic() - started) * 1000),
        "seed": sampler["seed"],
        "negative_prompt_chars": len(sampler["negative_prompt"]),
        "conditioning_count": 0,
        "pipeline_used": type(pipe).__name__,
        "conditioning_strength": None,
        "upscaler_used": False,
        **__import__("ltxcaps").diagnostics(caps, profile),
        **_gpu_metrics(),
    }
    # ONLY WHEN ASKED. A production still goes to R2 and its reply says so;
    # nothing about that changes here. The benchmark harness asks, because the
    # bucket is private and a reference nobody can look at cannot be approved.
    if preview.wanted(job):
        result["preview_frames"] = preview.encode_frames([still], want=1)
    return result


def _gpu_metrics() -> dict:
    """The same device evidence the generate op reports, measured here."""
    metrics = {
        "device": "cpu",
        "gpu_name": None,
        "vram_total_mb": None,
        "vram_peak_mb": None,
    }
    try:
        import torch
    except ImportError:
        return metrics
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        metrics = {
            "device": "cuda",
            "gpu_name": torch.cuda.get_device_name(0),
            "vram_total_mb": int(props.total_memory / (1024 * 1024)),
            "vram_peak_mb": int(torch.cuda.max_memory_allocated() / (1024 * 1024)),
        }
    return metrics


def run_concat(job: dict, segment_paths: list, output_path: str) -> dict:
    """Ordered stream-copy assembly of clips this worker generated.

    Deterministic by construction: no re-encode, packets are copied with
    monotonic timestamp offsets, so the output is byte-stable for the same
    inputs. Uniformity is enforced, never coerced — mismatched canvases,
    codecs, timebases or audio presence refuse with their own codes. The
    output is then DECODED end to end: a final artifact that cannot decode
    is a failure, not a delivery.
    """
    import av

    started = time.monotonic()

    with av.open(segment_paths[0]) as first:
        if not first.streams.video:
            raise ConcatRefused("concat-no-video", "segment 0 has no video stream")
        v_tpl = first.streams.video[0]
        width, height = v_tpl.codec_context.width, v_tpl.codec_context.height
        codec_name = v_tpl.codec_context.name
        time_base = v_tpl.time_base
        has_audio = len(first.streams.audio) > 0

    out = av.open(output_path, "w")
    try:
        with av.open(segment_paths[0]) as tpl_container:
            out_v = out.add_stream(template=tpl_container.streams.video[0])
            out_a = (
                out.add_stream(template=tpl_container.streams.audio[0])
                if has_audio
                else None
            )

        v_offset = 0
        a_offset = 0
        for index, path in enumerate(segment_paths):
            with av.open(path) as src:
                if not src.streams.video:
                    raise ConcatRefused(
                        "concat-no-video", f"segment {index} has no video stream"
                    )
                vin = src.streams.video[0]
                cc = vin.codec_context
                if (cc.width, cc.height) != (width, height):
                    raise ConcatRefused(
                        "concat-dims-mismatch",
                        f"segment {index} is {cc.width}x{cc.height}, "
                        f"expected {width}x{height}",
                    )
                if cc.name != codec_name:
                    raise ConcatRefused(
                        "concat-codec-mismatch",
                        f"segment {index} is {cc.name}, expected {codec_name}",
                    )
                if vin.time_base != time_base:
                    raise ConcatRefused(
                        "concat-timebase-mismatch",
                        f"segment {index} timebase differs",
                    )
                ain = src.streams.audio[0] if src.streams.audio else None
                if bool(ain) != has_audio:
                    raise ConcatRefused(
                        "concat-mixed-audio",
                        "segments must all carry audio, or none",
                    )

                # Each segment's clock is REBASED, not merely offset: the
                # first dts of each stream (which libx264 may start below
                # zero) maps exactly onto the running end, so timestamps
                # stay monotonic across every joint.
                v_shift = None
                a_shift = None
                v_end = v_offset
                a_end = a_offset
                streams = [vin] + ([ain] if ain else [])
                for packet in src.demux(streams):
                    if packet.dts is None:
                        continue
                    if packet.stream == vin:
                        if v_shift is None:
                            v_shift = v_offset - packet.dts
                        packet.stream = out_v
                        if packet.pts is not None:
                            packet.pts += v_shift
                        packet.dts += v_shift
                        v_end = max(v_end, packet.dts + (packet.duration or 0))
                    else:
                        if a_shift is None:
                            a_shift = a_offset - packet.dts
                        packet.stream = out_a
                        if packet.pts is not None:
                            packet.pts += a_shift
                        packet.dts += a_shift
                        a_end = max(a_end, packet.dts + (packet.duration or 0))
                    out.mux(packet)
                v_offset = v_end
                a_offset = a_end
    finally:
        out.close()
    concat_ms = int((time.monotonic() - started) * 1000)

    # VERIFY BY DECODING, not by trusting the mux: every frame of the
    # final artifact must decode, and the measured duration is what the
    # application settles against.
    frames = 0
    with av.open(output_path) as check:
        v = check.streams.video[0]
        for _ in check.decode(v):
            frames += 1
        rate = float(v.average_rate) if v.average_rate else float(contract.VIDEO_FPS)
    if frames <= 0 or rate <= 0:
        raise ConcatRefused("concat-output-undecodable", "assembled output has no frames")

    return {
        "segments": len(segment_paths),
        "frames": frames,
        "fps": int(round(rate)),
        "video_seconds": round(frames / rate, 2),
        "width": width,
        "height": height,
        "format": "mp4",
        "concat_ms": concat_ms,
        "output_bytes": os.path.getsize(output_path),
        "duration_ms": int((time.monotonic() - started) * 1000),
        **_gpu_metrics(),
    }
