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
import preview

# Where a probe's weights land. Container disk, wiped with the worker — a
# probe never writes into /app/models, so it cannot disturb the baked
# production checkpoints it runs beside.
PROBE_CACHE = "/tmp/probe-models"

# How much of the job's window the FETCH may take before the probe gives up.
#
# The handler's ceiling covers the whole job; this covers the download alone,
# and it is smaller on purpose. A fetch allowed to run to the job ceiling
# leaves nothing for the load and the generation, so it would spend the entire
# rental and still have no clip — the most expensive way to learn nothing.
# Stopping here instead converts an impossible download into a MEASUREMENT:
# bytes landed, seconds spent, and the rate those imply for the rest.
DOWNLOAD_BUDGET_SECONDS = int(contract.PROBE_RUNTIME_CEILING_SECONDS * 0.55)

# Every candidate the owner authorised, pinned. The revision is not decoration:
# a repository name names a moving branch, and the licence recorded here is the
# licence AT THIS COMMIT — which is the only form of that claim that stays true.
#
# `dtype` is the precision to LOAD at, which is not the precision on disk.
# Wan ships fp32; loading bf16 halves it, and that is the difference between
# fitting this card and not.
# SAMPLING SETTINGS ARE CITED, NEVER CHOSEN HERE.
#
# Read from each publisher's own model card at the PINNED revision on
# 2026-08-29 by validation/probe_settings (a $0 registry read, re-runnable).
# A row carries `steps`/`guidance` only where the card states one; where it
# does not, the field is absent and the pipeline's own default stands, which
# the report says explicitly. A 13B distilled checkpoint sampled at an
# invented step count is not that model performing badly, it is the wrong
# experiment — so nothing here is a preference.
PROBE_MODELS: dict[str, dict] = {
    "ltx-13b": {
        "label": "LTX-Video 13B (distilled)",
        # The 13B build with a DIFFUSERS LAYOUT. Lightricks/LTX-Video also
        # carries 13B weights, but as root-level single files beside three
        # other 13B variants and four fp8 copies — pointing a snapshot at
        # that repository downloads roughly 200 GiB to use 27 of it, and
        # loading it needs the single-file path rather than from_pretrained.
        # This repo is one 13B checkpoint, loadable the ordinary way.
        "repo": "Lightricks/LTX-Video-0.9.8-13B-distilled",
        "revision": "7c64400e1861cc0d7b98d570a1926d5408ec60cd",
        "pipeline": "LTXImageToVideoPipeline",
        "licence": "other (LTX Open Weights)",
        "dtype": "bfloat16",
        "offload": "model",
        "width": 704,
        "height": 480,
        "frames": 97,
        "fps": 24,
        # Measured: transformer 24.29 + text_encoder 17.74 + vae 2.32.
        "download_gib": 44.36,
        # Cited: the card's FIRST pass runs at 30 steps, then hands latents to
        # a separate upsampler and a 10-step refine. This probe runs the base
        # pass only — one pipeline, ONIQ's own canvas — so 30 is the figure
        # that applies and the two-stage upscale is deliberately not measured.
        # It states no guidance scale, so the pipeline's default stands.
        "steps": 30,
        "sampling_source": "model card, base LTXConditionPipeline call",
        # THE VAE PATTERNS ARE TWO EXACT FILENAMES, not vae/*. This
        # repository nests a SECOND COMPLETE COPY of itself under vae/ —
        # vae/transformer/, vae/text_encoder/ — 42.03 GiB of it, and
        # huggingface_hub's fnmatch lets `*` cross a slash, so vae/* would
        # quietly pull the lot.
        "allow": [
            "model_index.json",
            "transformer/*",
            "text_encoder/*",
            "tokenizer/*",
            "scheduler/*",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ],
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
        # Cited: the card's i2v call sets guidance_scale=5.0 and states NO
        # step count, so the pipeline default stands and the report says so.
        "guidance": 5.0,
        "sampling_source": "model card, i2v example (no step count stated)",
        "allow": [
            "model_index.json", "transformer/*", "transformer_2/*",
            "text_encoder/*", "tokenizer/*", "scheduler/*", "vae/*",
            "image_encoder/*", "image_processor/*",
        ],
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
        # Cited: the card's i2v call, verbatim.
        "steps": 40,
        "guidance": 3.5,
        "sampling_source": "model card, i2v example",
        "allow": [
            "model_index.json", "transformer/*", "transformer_2/*",
            "text_encoder/*", "tokenizer/*", "scheduler/*", "vae/*",
            "image_encoder/*", "image_processor/*",
        ],
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
        # Cited: the card's i2v example, verbatim.
        "steps": 50,
        "guidance": 6.0,
        "sampling_source": "model card, i2v example",
        "allow": [
            "model_index.json", "transformer/*", "text_encoder/*",
            "tokenizer/*", "scheduler/*", "vae/*",
        ],
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
    "DOWNLOAD_TIMEOUT",
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


def cuda_identity(torch=None) -> dict:
    """WHICH card, and is this CUDA at all.

    The same proof production demands of every paid job: a CPU fallback is not
    success and the wrong card is not the benchmark. Reported here so the
    harness can apply exactly the check it applies to LTX rather than a weaker
    one — a benchmark measured on some other GPU would be worse than no
    benchmark, because it would look like an answer.
    """
    if torch is None:  # pragma: no cover - exercised through run()
        import torch
    if not torch.cuda.is_available():
        return {"device": "cpu"}
    free, total = torch.cuda.mem_get_info()
    return {
        "device": "cuda",
        "gpu_name": torch.cuda.get_device_name(0),
        "vram_total_mb": total // (1024 * 1024),
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


def dir_bytes(path: str) -> int:
    """Bytes actually on disk under a directory. Follows no symlinks, so a
    huggingface cache that hardlinks into a blob store is counted once."""
    total = 0
    seen: set = set()
    for root, _dirs, files in os.walk(path):
        for name in files:
            full = os.path.join(root, name)
            try:
                st = os.lstat(full)
            except OSError:
                continue
            if st.st_ino and st.st_ino in seen:
                continue
            seen.add(st.st_ino)
            total += st.st_size
    return total


def fetch_within_budget(fetch, measure, budget_s: int, *,
                        clock=time.monotonic, sleeper=time.sleep,
                        poll_s: float = 5.0, spawn=None):
    """Run the checkpoint download, and give up at the budget with numbers.

    huggingface_hub offers no whole-snapshot timeout, so the fetch runs on a
    worker thread and this polls the clock. On overrun the thread is
    ABANDONED rather than joined: the job is ending either way, the container
    is reclaimed with it, and waiting for a download that has already proved
    too slow would spend the rest of the window to change nothing.

    What comes back on overrun is the finding: how many bytes landed and how
    long they took. A candidate that cannot be fetched here is a real result
    about running this model on this worker, and it is only a result if the
    rate is measured rather than guessed.
    """
    import threading

    box: dict = {}

    def target():
        try:
            box["value"] = fetch()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
            box["error"] = exc

    if spawn is None:  # pragma: no cover - the real thread
        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        spawned = thread
    else:
        spawned = spawn(target)

    began = clock()
    while True:
        if "value" in box or "error" in box:
            break
        if clock() - began >= budget_s:
            landed = measure()
            elapsed = max(clock() - began, 1e-9)
            rate = landed / elapsed / (1024 * 1024)
            raise ProbeStop(
                "DOWNLOAD_TIMEOUT",
                f"{landed / 1024**3:.2f}GiB fetched in {elapsed:.0f}s "
                f"({rate:.1f} MiB/s) — the budget of {budget_s}s expired with "
                "the checkpoint incomplete, so no GPU time was spent on it",
            )
        sleeper(poll_s)
    if "error" in box:
        raise box["error"]
    return box["value"]


def probe_report(model_key: str, spec_row: dict, phases: Phases, *,
                 before: dict, peaks: dict, failure: str,
                 output_bytes: int = 0, detail: str = "",
                 identity: dict | None = None) -> dict:
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
        "steps": spec_row.get("steps") or "PIPELINE_DEFAULT",
        "guidance": spec_row.get("guidance") if spec_row.get("guidance") is not None
                    else "PIPELINE_DEFAULT",
        "sampling_source": spec_row.get("sampling_source") or "pipeline default",
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
        # vram_peak_mb alongside the byte-precise figure: the harness's
        # shared success check reads the megabyte field every paid job
        # reports, and the benchmark reads the bytes.
        row["vram_peak_mb"] = peaks["peak_allocated_bytes"] // (1024 * 1024)
    if identity:
        row.update(identity)
    if detail:
        row["detail"] = detail
    return row


def run(job: dict, input_path: str, output_path: str,
        load_pipeline=None, torch=None, fetch=None) -> dict:
    """One probe. Downloads, loads, conditions, generates, encodes, measures.

    THE DOWNLOAD IS ITS OWN PHASE. Owner directive 2026-08-29: "do not hide
    cold-start cost — Wan weight-loading time must be explicitly reported."
    Fetching 117.52 GiB and building a pipeline out of it are two different
    costs with two different fixes, and a single `model_load` number would
    report the network as the model.

    `fetch`, `load_pipeline` and `torch` are injectable for the CPU test rig,
    the same seam videogen uses: everything here except the CUDA pass itself
    is exercised without a GPU.
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

    if fetch is None:  # pragma: no cover - real path only
        fetch = _real_fetch(spec_row)
    if load_pipeline is None:  # pragma: no cover - real path only
        load_pipeline = _real_loader(spec_row)

    cache_dir = _cache_dir(spec_row)
    try:
        local = phases.time("download", lambda: fetch_within_budget(
            fetch,
            lambda: dir_bytes(cache_dir),
            DOWNLOAD_BUDGET_SECONDS,
        ))
    except ProbeStop:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ProbeStop("LOAD_FAILED", f"{type(exc).__name__}: {exc}") from exc
    disk["download_bytes"] = dir_bytes(cache_dir)

    try:
        pipe = phases.time("model_load", lambda: load_pipeline(local))
    except Exception as exc:  # noqa: BLE001
        raise ProbeStop(classify(exc), f"{type(exc).__name__}: {exc}") from exc

    try:
        image = phases.time("conditioning_load", lambda: _load_reference(input_path))
    except Exception as exc:  # noqa: BLE001
        raise ProbeStop("CONDITIONING_FAILURE", f"{type(exc).__name__}: {exc}") from exc

    try:
        result = phases.time("inference", lambda: pipe(
            image=image,
            prompt=job["params"]["prompt"],
            width=spec_row["width"],
            height=spec_row["height"],
            num_frames=spec_row["frames"],
            **_sampling(spec_row),
        ))
    except Exception as exc:  # noqa: BLE001
        raise ProbeStop(classify(exc), f"{type(exc).__name__}: {exc}") from exc

    frames = _frames_of(result)
    if not frames:
        raise ProbeStop("GENERATION_FAILURE", "the pipeline returned no frames")

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
        identity=cuda_identity(torch),
    )
    report.update(disk)
    # The clip goes to R2; a handful of downscaled frames come back here, so
    # the benchmark can be LOOKED at without a bucket credential. A probe is
    # only ever run by the harness, so this needs no flag — and a failure to
    # make a preview is never allowed to fail a probe that produced a video.
    report["preview_frames"] = preview.encode_frames(frames)
    return report


def _load_reference(path: str):  # pragma: no cover - thin PIL wrapper
    from PIL import Image

    return Image.open(path).convert("RGB")


def _frames_of(result):
    """The frames a diffusers video pipeline actually returned.

    Every video pipeline here answers with an output object whose `frames`
    is BATCHED — a list of clips, one per prompt. Treating that object, or
    the outer list, as the frame sequence is the failure videogen already
    learned: it encodes without error and writes a file nothing can play.
    A plain list passes through, which is what the CPU rig hands over.
    """
    frames = getattr(result, "frames", result)
    if frames is None:
        return []
    first = next(iter(frames), None) if len(frames) else None
    if isinstance(first, (list, tuple)):
        return list(first)
    try:  # a numpy batch: (batch, frames, h, w, c)
        import numpy as np

        if isinstance(frames, np.ndarray) and frames.ndim == 5:
            return list(frames[0])
    except ImportError:  # pragma: no cover - numpy is always present here
        pass
    return list(frames)


def _sampling(spec_row: dict) -> dict:
    """The sampling knobs for this candidate, and ONLY the ones its publisher
    states. A row that names no step count runs at its own pipeline's default
    and says so in the report — a guessed step count would make the benchmark
    a measurement of my guess."""
    knobs = {}
    if spec_row.get("steps"):
        knobs["num_inference_steps"] = spec_row["steps"]
    if spec_row.get("guidance") is not None:
        knobs["guidance_scale"] = spec_row["guidance"]
    return knobs


def _encode(frames, output_path: str, fps: int):  # pragma: no cover
    """h264/yuv420p, the same encoder production uses — a probe clip that
    only plays in one viewer would not be inspectable evidence."""
    import numpy as np
    import imageio.v2 as imageio

    writer = imageio.get_writer(
        output_path,
        fps=fps,
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


def _cache_dir(spec_row: dict) -> str:
    """Where this candidate's weights land. One directory per repo, so the
    bytes measured on a timeout belong to the candidate being timed."""
    return os.path.join(PROBE_CACHE, spec_row["repo"].replace("/", "--"))


def _real_fetch(spec_row: dict):  # pragma: no cover - needs the network
    """The download, separated from the load so each is timed on its own."""

    def fetch():
        from huggingface_hub import snapshot_download

        local_dir = _cache_dir(spec_row)
        os.makedirs(local_dir, exist_ok=True)
        return snapshot_download(
            spec_row["repo"],
            revision=spec_row["revision"],
            local_dir=local_dir,
            token=os.environ.get("HF_TOKEN") or None,
            # BOUNDED. Without this a snapshot takes whatever the repository
            # happens to contain, which for one of these candidates is four
            # 13B variants and four fp8 copies — 200 GiB fetched to use 27.
            allow_patterns=spec_row["allow"],
        )

    return fetch


def _real_loader(spec_row: dict):  # pragma: no cover - needs CUDA
    """Build the loader for one candidate. Every knob comes from the row."""

    def load(local):
        import torch
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
