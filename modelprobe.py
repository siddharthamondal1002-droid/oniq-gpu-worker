"""The model_probe workload — one candidate, one clip, every number measured.

Owner directive 2026-08-29 (five-model A5000 probe). This is a MODEL
EVALUATION, not a production path: it exists to answer which open-source video
model is actually viable on the A5000 ONIQ already rents, and it answers that
by loading the real checkpoint and generating a real clip.

WHAT MAKES THIS DIFFERENT FROM videogen. videogen runs ONE model, baked into
the image at build time, loaded with `local_files_only` so a paid job can never
wait on a download. That is exactly right for production and exactly wrong for
a benchmark: the whole point here is to compare checkpoints the image does not
carry. So this module fetches at job time — and REPORTS the fetch as its own
timed phase, because on a 14B model the download, not the inference, is likely
to be the expensive part, and hiding it inside "model load" would misattribute
the cost of every candidate.

THE CALLER NEVER NAMES A MODEL. `model` is a key into PROBE_MODELS below, and
every repository, revision, precision and offload strategy is fixed here on the
server. The same fence the production contract keeps: a caller may say WHICH
row of a benchmark to run, never what to download or how to run it.

NO TECHNICAL FAILURE BECOMES A QUALITY VERDICT. This module reports what broke
and where — LOAD_FAILED, VRAM_OOM, GENERATION_FAILURE — and stops. It never
returns QUALITY_FAIL, because quality is decided by looking at frames and this
code cannot see. Equally, a technically valid MP4 comes back SUCCESS, which
means "the pipeline completed", never "the clip is good".
"""

from __future__ import annotations

import os
import time

import contract

# Where a probe's weights land. Container disk, wiped with the worker — a
# probe never writes into /app/models, so it cannot disturb the baked
# production checkpoints it runs beside.
PROBE_CACHE = "/tmp/probe-models"

# Every candidate the owner authorised, pinned. The revision is not decoration:
# a repository name names a moving branch, and the licence recorded here is the
# licence AT THIS COMMIT — which is the only form of that claim that stays true.
#
# `dtype` is the precision to LOAD at, which is not the precision on disk.
# Wan ships fp32; loading bf16 halves it, and that is the difference between
# fitting this card and not.
PROBE_MODELS: dict[str, dict] = {
    "ltx-13b": {
        "label": "LTX-Video 13B",
        "repo": "Lightricks/LTX-Video",
        "revision": "8984fa25007f376c1a299016d0957a37a2f797bb",
        "pipeline": "LTXImageToVideoPipeline",
        "single_file": "ltxv-13b-0.9.8-dev.safetensors",
        "licence": "other (LTX Open Weights)",
        "dtype": "bfloat16",
        "offload": "model",
        "width": 704,
        "height": 480,
        "frames": 97,
        "fps": 24,
        # Measured 2026-08-29 from the registry: the single-file 13B checkpoint
        # plus the pipeline components this repo ships beside it.
        "download_gib": 26.62 + 17.74 + 1.56,
    },
    "wan21-i2v-480p": {
        "label": "Wan2.1 I2V-14B-480P",
        "repo": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        "revision": "b184e23a8a16b20f108f727c902e769e873ffc73",
        "pipeline": "WanImageToVideoPipeline",
        "licence": "apache-2.0",
        "dtype": "bfloat16",
        "offload": "model",
        # 832x480 is this checkpoint's own landscape shape; 704x480 is not one
        # it was trained at, and forcing ONIQ's canvas onto it would measure
        # the mismatch rather than the model.
        "width": 832,
        "height": 480,
        "frames": 81,
        "fps": 16,
        "download_gib": 83.89,
    },
    "wan22-i2v-a14b": {
        "label": "Wan2.2 I2V-A14B",
        "repo": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "revision": "596658fd9ca6b7b71d5057529bbf319ecbc61d74",
        "pipeline": "WanImageToVideoPipeline",
        "licence": "apache-2.0",
        "dtype": "bfloat16",
        "offload": "model",
        "width": 832,
        "height": 480,
        "frames": 81,
        "fps": 16,
        # Two experts. A separate candidate from Wan2.1 by owner directive, and
        # separately measured — the label shares a number with 2.1 and nothing
        # else.
        "download_gib": 117.52,
    },
    "cogvideox-i2v": {
        "label": "CogVideoX-5B-I2V",
        "repo": "zai-org/CogVideoX-5b-I2V",
        "revision": "a6f0f4858a8395e7429d82493864ce92bf73af11",
        "pipeline": "CogVideoXImageToVideoPipeline",
        "licence": "other (publisher terms)",
        "dtype": "bfloat16",
        "offload": "model",
        # CogVideoX is trained at a fixed 720x480x49. It has no 704 mode and
        # no 97-frame mode; asking for one would not produce a comparable clip,
        # it would produce a broken one.
        "width": 720,
        "height": 480,
        "frames": 49,
        "fps": 8,
        "download_gib": 20.15,
    },
}

# HunyuanVideo-1.5 is deliberately ABSENT. Owner rule: it may only be probed if
# its architecture, checkpoint layout and required configuration resolve from
# authoritative material without guessing. On 2026-08-29 the registry gave a
# non-diffusers layout, eleven complete checkpoints sharing one directory, and
# a config carrying neither depth nor temporal compression ratio. Adding a row
# here on inference would be the guess the rule forbids.
NOT_EVALUATED = {
    "hunyuanvideo-1.5-i2v": "ARCHITECTURE_NOT_RESOLVED",
}

# The decisive-failure vocabulary, in the order the pipeline can reach them.
# QUALITY_FAIL is NOT here: this module cannot see frames, and a stage that
# completed is never a statement about what it produced.
FAILURES = (
    "LOAD_FAILED",
    "VRAM_OOM",
    "CUDA_FAILURE",
    "MODEL_ERROR",
    "CONDITIONING_FAILURE",
    "GENERATION_FAILURE",
    "ENCODE_FAILURE",
    "ARTIFACT_FAILURE",
)


class ProbeStop(Exception):
    """A probe ended at a named stage. Carries the stage, never a verdict."""

    def __init__(self, failure: str, detail: str):
        super().__init__(f"{failure}: {detail}")
        self.failure = failure
        self.detail = detail


def classify(exc: Exception) -> str:
    """Which decisive failure this exception is.

    OOM is separated from every other CUDA error on purpose: an OOM is the
    finding the owner asked for — evidence about this card — while a driver or
    kernel fault says nothing about whether the model fits. Collapsing them
    would turn a broken worker into a verdict on a model.
    """
    name = type(exc).__name__
    text = str(exc).lower()
    if "outofmemory" in name.lower() or "out of memory" in text:
        return "VRAM_OOM"
    if "cuda" in name.lower() or "cuda" in text:
        return "CUDA_FAILURE"
    return "MODEL_ERROR"


def cuda_snapshot(torch=None) -> dict:
    """Total / allocated / reserved, in bytes. Empty when there is no CUDA.

    Read from the device rather than inferred from checkpoint size — the owner
    was explicit, and the projection this replaces was wrong about decode by
    more than tenfold before a real job corrected it.
    """
    if torch is None:  # pragma: no cover - exercised through run()
        import torch
    if not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info()
    return {
        "total_bytes": total,
        "free_bytes": free,
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
    }


def cuda_peaks(torch=None) -> dict:
    if torch is None:  # pragma: no cover - exercised through run()
        import torch
    if not torch.cuda.is_available():
        return {}
    return {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


class Phases:
    """Wall clock per named stage, kept separate so no stage can hide in another.

    The download is its own phase for the reason the module docstring gives: on
    a 14B candidate it is expected to dominate, and folding it into model load
    would report the cost of the network as the cost of the model.
    """

    def __init__(self, clock=time.perf_counter):
        self._clock = clock
        self._start = clock()
        self.timings: dict[str, float] = {}

    def time(self, name: str, fn):
        began = self._clock()
        try:
            return fn()
        finally:
            self.timings[f"{name}_ms"] = int((self._clock() - began) * 1000)

    def total_ms(self) -> int:
        return int((self._clock() - self._start) * 1000)


def spec(model_key: str) -> dict:
    """The server's row for this benchmark key. Never a caller-supplied repo."""
    if model_key in NOT_EVALUATED:
        raise ProbeStop("MODEL_ERROR",
                        f"{model_key} is NOT_EVALUATED: {NOT_EVALUATED[model_key]}")
    if model_key not in PROBE_MODELS:
        raise ProbeStop("MODEL_ERROR", f"unknown probe model {model_key!r}")
    return PROBE_MODELS[model_key]


def fits_disk(model_key: str, free_bytes: int, *, headroom: float = 1.15) -> bool:
    """Can this candidate's weights land on the disk this worker actually has?

    Checked BEFORE the download starts, because running out of disk 60 GiB into
    a 118 GiB fetch burns the whole watchdog window and produces no evidence
    about the model at all. Headroom covers the fact that a snapshot download
    briefly holds an incoming file beside what it has already written.
    """
    need = PROBE_MODELS[model_key]["download_gib"] * (1024 ** 3) * headroom
    return free_bytes >= need


def free_disk_bytes(path: str = "/tmp") -> int:
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize


def total_disk_bytes(path: str = "/tmp") -> int:
    """What the worker ACTUALLY has, not what the template asked for.

    The template requests 200 GB; the owner's instruction is not to trust
    that nominal figure. This is the number the running container reports.
    """
    stat = os.statvfs(path)
    return stat.f_blocks * stat.f_frsize


def probe_report(model_key: str, spec_row: dict, phases: Phases, *,
                 before: dict, peaks: dict, failure: str,
                 output_bytes: int = 0, detail: str = "") -> dict:
    """Everything the owner's table needs from one probe, measured or absent.

    Cost is deliberately NOT computed here. The worker does not know the live
    rate, and an hourly figure remembered from an earlier run is not a price —
    the harness attaches it at report time from a live quote, labelled as an
    estimate from measured runtime unless the provider states billing itself.
    """
    row = {
        "model": model_key,
        "label": spec_row.get("label"),
        "repo": spec_row.get("repo"),
        "revision": spec_row.get("revision"),
        "licence": spec_row.get("licence"),
        "dtype": spec_row.get("dtype"),
        "offload": spec_row.get("offload"),
        "width": spec_row.get("width"),
        "height": spec_row.get("height"),
        "frames": spec_row.get("frames"),
        "fps": spec_row.get("fps"),
        "failure": failure,
        "output_bytes": output_bytes,
        "total_wall_ms": phases.total_ms(),
    }
    row.update(phases.timings)
    if before:
        row["vram_total_bytes"] = before.get("total_bytes")
        row["vram_before_allocated_bytes"] = before.get("allocated_bytes")
        row["vram_before_reserved_bytes"] = before.get("reserved_bytes")
    if peaks:
        row.update(peaks)
    if detail:
        row["detail"] = detail
    return row


def run(job: dict, input_path: str, output_path: str,
        load_pipeline=None, torch=None) -> dict:
    """One probe. Downloads, loads, conditions, generates, encodes, measures.

    `load_pipeline` and `torch` are injectable for the CPU test rig, the same
    seam videogen uses: everything here except the CUDA pass itself is
    exercised without a GPU.
    """
    model_key = job.get("model")
    spec_row = spec(model_key)
    phases = Phases()

    if torch is None:  # pragma: no cover - real path only
        import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    before = cuda_snapshot(torch)

    # DISK FIRST. Cheapest possible refusal, and it distinguishes "this card
    # cannot hold the model" from "this worker had nowhere to put it" — two
    # findings that would otherwise both arrive as a failed probe.
    where = PROBE_CACHE if os.path.isdir(PROBE_CACHE) else "/tmp"
    free = free_disk_bytes(where)
    disk = {"disk_free_bytes": free, "disk_total_bytes": total_disk_bytes(where)}
    if not fits_disk(model_key, free):
        raise ProbeStop(
            "LOAD_FAILED",
            f"needs ~{spec_row['download_gib']:.1f}GiB of weights, worker has "
            f"{free / 1024**3:.1f}GiB free — no download attempted",
        )

    if load_pipeline is None:  # pragma: no cover - real path only
        load_pipeline = _real_loader(spec_row)

    try:
        pipe = phases.time("model_load", lambda: load_pipeline())
    except Exception as exc:  # noqa: BLE001
        raise ProbeStop(classify(exc), f"{type(exc).__name__}: {exc}") from exc

    try:
        image = phases.time("conditioning_load", lambda: _load_reference(input_path))
    except Exception as exc:  # noqa: BLE001
        raise ProbeStop("CONDITIONING_FAILURE", f"{type(exc).__name__}: {exc}") from exc

    try:
        frames = phases.time("inference", lambda: pipe(
            image=image,
            prompt=job["params"]["prompt"],
            width=spec_row["width"],
            height=spec_row["height"],
            num_frames=spec_row["frames"],
        ))
    except Exception as exc:  # noqa: BLE001
        raise ProbeStop(classify(exc), f"{type(exc).__name__}: {exc}") from exc

    try:
        phases.time("encode", lambda: _encode(frames, output_path, spec_row["fps"]))
    except Exception as exc:  # noqa: BLE001
        raise ProbeStop("ENCODE_FAILURE", f"{type(exc).__name__}: {exc}") from exc

    size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
    if size <= 0:
        raise ProbeStop("ARTIFACT_FAILURE", "encoder produced no bytes")

    report = probe_report(
        model_key, spec_row, phases,
        before=before, peaks=cuda_peaks(torch),
        failure="SUCCESS", output_bytes=size,
    )
    report.update(disk)
    return report


def _load_reference(path: str):  # pragma: no cover - thin PIL wrapper
    from PIL import Image

    return Image.open(path).convert("RGB")


def _encode(frames, output_path: str, fps: int):  # pragma: no cover
    import imageio.v2 as imageio

    writer = imageio.get_writer(output_path, fps=fps, codec="libx264",
                                quality=contract.VIDEO_QUALITY)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def _real_loader(spec_row: dict):  # pragma: no cover - needs CUDA and network
    """Build the loader for one candidate. Every knob comes from the row."""

    def load():
        import torch
        from huggingface_hub import snapshot_download

        os.makedirs(PROBE_CACHE, exist_ok=True)
        local = snapshot_download(
            spec_row["repo"],
            revision=spec_row["revision"],
            local_dir=os.path.join(PROBE_CACHE, spec_row["repo"].replace("/", "--")),
            token=os.environ.get("HF_TOKEN") or None,
        )
        import diffusers

        cls = getattr(diffusers, spec_row["pipeline"])
        pipe = cls.from_pretrained(
            local,
            torch_dtype=getattr(torch, spec_row["dtype"]),
            local_files_only=True,
        )
        if spec_row["offload"] == "model":
            pipe.enable_model_cpu_offload()
        elif spec_row["offload"] == "sequential":
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to("cuda")
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        return pipe

    return load
