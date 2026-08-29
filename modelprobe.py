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
import re
import time

import contract
import preview

# Where a probe's weights land. Container disk, wiped with the worker — a
# probe never writes into /app/models, so it cannot disturb the baked
# production checkpoints it runs beside.
PROBE_CACHE = "/tmp/probe-models"

# EVERY HUGGINGFACE CACHE POINTED SOMEWHERE THIS USER CAN WRITE.
#
# The first live probe failed in 7.4s with:
#
#   LOAD_FAILED: OSError: I/O error: I/O error: Permission denied (os error 13)
#
# after a 5s "download". Nothing to do with the model. The worker runs as
# uid 10001 with HOME=/home/oniq, and the image bakes its models as ROOT
# with that same HOME. The bake deletes ~/.cache/huggingface afterwards but
# `/home/oniq/.cache` itself survives, owned by root — so at job time the
# hub tries to create its cache inside a directory this user cannot write,
# and the Rust chunk-cache layer reports it as a doubled I/O error that
# looks nothing like a permissions fault.
#
# Set at MODULE level, so it is in place before anything in this process
# imports huggingface_hub and reads these into constants. Production is
# untouched: videogen loads baked checkpoints with local_files_only and
# never writes a hub cache at all.
_PROBE_HF_HOME = os.path.join(PROBE_CACHE, "hf")
os.environ.setdefault("HF_HOME", _PROBE_HF_HOME)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_PROBE_HF_HOME, "hub"))
os.environ.setdefault("HF_XET_CACHE", os.path.join(_PROBE_HF_HOME, "xet"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(PROBE_CACHE, "xdg"))

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
#
# SAMPLING SETTINGS ARE CITED, NEVER CHOSEN HERE.
#
# Read from each publisher's own model card at the PINNED revision on
# 2026-08-29 by validation/probe_settings (a $0 registry read, re-runnable).
# A row carries `steps`/`guidance` only where the card states one; where it
# does not, the field is absent and the pipeline's own default stands, which
# the report says explicitly. A 13B distilled checkpoint sampled at an
# invented step count is not that model performing badly, it is the wrong
# experiment — so nothing here is a preference.
#
# WHICH OFFLOAD, AND WHY IT IS NOT A PREFERENCE EITHER.
#
# enable_model_cpu_offload() keeps ONE pipeline component on the card at a
# time, so the peak is the largest component — the transformer. In bf16 that
# is params x 2 bytes, and against a 24 GB A5000 the arithmetic decides:
#
#   cogvideox-5b        9.31 GiB   fits, with room for activations
#   ltx-13b            24.21 GiB   does not fit
#   wan21-i2v-14b      26.08 GiB   does not fit
#   wan22 (per expert) 26.08 GiB   does not fit
#
# So the three large candidates get SEQUENTIAL offload, which moves
# submodules rather than whole components and therefore fits. That choice is
# made here, before any money moves, because an OOM caused by my picking
# model-level offload for a 26 GiB transformer would be a fact about my
# configuration wearing the costume of a fact about the model — exactly the
# "do not turn a technical failure into a quality judgment" the owner ruled
# out. Sequential offload is much slower, and that slowness is a REAL and
# reportable cost of running a 14B model on this card; it is not a reason to
# rent a bigger one.
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
        # ITS model_index.json DECLARES LTXConditionPipeline, not this class,
        # and that is checked rather than assumed: read at $0 on 2026-08-29,
        # and both classes take the SAME five components with the same names
        # (scheduler, vae, text_encoder, tokenizer, transformer) in diffusers
        # 0.35.2, so from_pretrained loads it here. What differs is the FLOW,
        # not the weights: the card drives a conditions-list call plus a
        # latent upsampler, and this probe makes the single-pass image call
        # every other candidate makes. That uniformity is the point of the
        # benchmark, and the difference belongs in the report rather than in
        # a quiet substitution.
        "pipeline": "LTXImageToVideoPipeline",
        "licence": "other (LTX Open Weights)",
        "dtype": "bfloat16",
        "offload": "sequential",
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
        "offload": "sequential",
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
        "offload": "sequential",
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
    "hunyuanvideo-1.5-i2v": {
        "label": "HunyuanVideo-1.5 480p I2V step-distilled (8 steps)",
        "repo": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled",
        # Pinned from the $0 model-bench read (gpu-validation run
        # 33269975707, 2026-08-29): the sha probe_settings resolved from the
        # live listing, at which COVERAGE reported every declared component
        # inside the allow list and the shipped configs matched the research
        # (use_meanflow true, scheduler shift 7.0, target_size 640).
        "revision": "854c04a4c8a53d990b418c7478f0802c0fc8c726",
        "pipeline": "HunyuanVideo15ImageToVideoPipeline",
        # NOT a permissive licence, and the restriction reaches OUTPUTS:
        # Tencent HunyuanVideo-1.5 Community License, LICENSE at
        # Tencent-Hunyuan/HunyuanVideo-1.5@60783e70 — Section 1.l excludes the
        # EU, UK and South Korea from the Territory, and Section 5.c forbids
        # using or displaying "Output or results" outside it. Recorded here so
        # the benchmark's decision weighs it; adopting the model in production
        # would be an owner business decision on top of any quality result.
        "licence": "Tencent HunyuanVideo-1.5 Community License (no EU/UK/KR, incl. outputs)",
        # Official default dtype. Tencent's CLI (--dtype) offers bf16 or fp32
        # only, and the diffusers docs page loads bf16; the CLI additionally
        # forces the VAE to fp16 internally, which the uniform diffusers
        # from_pretrained path does not reproduce — the whole pipeline runs
        # bf16 here, the same single-dtype convention as every other row.
        "dtype": "bfloat16",
        # 8.3B bf16 transformer is ~15.5 GiB resident under model-level
        # offload — before activations over ~12k-token variable-length
        # attention that this card would run on plain SDPA (the docs
        # recommend flash/sage for exactly that path). The margin is real on
        # paper and unmeasured in practice, and an OOM caused by that gamble
        # would be my configuration wearing the costume of a fact about the
        # model. Sequential offload is also the closest diffusers equivalent
        # of the path Tencent itself auto-enables on cards under 60 GB
        # (group offload, one block per group), and it is the mode three of
        # the four measured candidates already ran under.
        "offload": "sequential",
        # NO width/height ON PURPOSE, and the loader passes neither: this
        # pipeline accepts no such kwargs. It derives the canvas from the
        # reference image's aspect ratio against its trained 480p bucket
        # list (target_size 640, stride 16) and centre-crops to the closest
        # bucket. The output's real dimensions are read from the frames.
        "frames": 121,
        "fps": 24,
        # MEASURED by the same $0 read, from the listing's own byte counts
        # over exactly the allow patterns: transformer 15.52 + text_encoder
        # 13.17 + vae 2.35 + image_encoder 0.80 + text_encoder_2 0.41 +
        # tokenizers/configs. The community repo ships bf16 shards, which is
        # why this is half the search-snippet estimate the row landed with.
        "download_gib": 32.26,
        # Cited: the repo's own optimal-config table says "8 or 12
        # (recommended)" for this checkpoint, and Tencent's code defaults it
        # to 12 (PIPELINE_CONFIGS["480p_i2v_step_distilled"]: guidance 1.0,
        # flow shift 7.0, steps 12 — hyvideo/commons/__init__.py@60783e70).
        # The owner ordered 8 first. Guidance is deliberately NOT a row key:
        # the converted checkpoint ships its guider at scale 1.0 (CFG off,
        # negative prompt never encoded) and the pipeline's __call__ accepts
        # no guidance_scale at all.
        "steps": 8,
        "sampling_source": "official README optimal-config table + Tencent "
                           "PIPELINE_CONFIGS (8 or 12 steps; guidance and "
                           "shift ship in the checkpoint)",
        "allow": [
            "model_index.json", "transformer/*", "text_encoder/*",
            "tokenizer/*", "text_encoder_2/*", "tokenizer_2/*",
            "image_encoder/*", "feature_extractor/*", "scheduler/*",
            "vae/*", "guider/*",
        ],
    },
    "hunyuanvideo-1.5-i2v-12step": {
        # THE SAME CHECKPOINT at the other officially recommended step count.
        # Owner phase 15: dispatched only if the 8-step clip is technically
        # valid and visually promising, to answer whether quality improves.
        # A separate row rather than a dispatch knob because sampling is a
        # server-side constant the caller can pick but never set.
        "label": "HunyuanVideo-1.5 480p I2V step-distilled (12 steps)",
        "repo": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled",
        # Pinned from the $0 model-bench read (gpu-validation run
        # 33269975707, 2026-08-29): the sha probe_settings resolved from the
        # live listing, at which COVERAGE reported every declared component
        # inside the allow list and the shipped configs matched the research
        # (use_meanflow true, scheduler shift 7.0, target_size 640).
        "revision": "854c04a4c8a53d990b418c7478f0802c0fc8c726",
        "pipeline": "HunyuanVideo15ImageToVideoPipeline",
        # NOT a permissive licence, and the restriction reaches OUTPUTS:
        # Tencent HunyuanVideo-1.5 Community License, LICENSE at
        # Tencent-Hunyuan/HunyuanVideo-1.5@60783e70 — Section 1.l excludes the
        # EU, UK and South Korea from the Territory, and Section 5.c forbids
        # using or displaying "Output or results" outside it. Recorded here so
        # the benchmark's decision weighs it; adopting the model in production
        # would be an owner business decision on top of any quality result.
        "licence": "Tencent HunyuanVideo-1.5 Community License (no EU/UK/KR, incl. outputs)",
        # Official default dtype. Tencent's CLI (--dtype) offers bf16 or fp32
        # only, and the diffusers docs page loads bf16; the CLI additionally
        # forces the VAE to fp16 internally, which the uniform diffusers
        # from_pretrained path does not reproduce — the whole pipeline runs
        # bf16 here, the same single-dtype convention as every other row.
        "dtype": "bfloat16",
        # 8.3B bf16 transformer is ~15.5 GiB resident under model-level
        # offload — before activations over ~12k-token variable-length
        # attention that this card would run on plain SDPA (the docs
        # recommend flash/sage for exactly that path). The margin is real on
        # paper and unmeasured in practice, and an OOM caused by that gamble
        # would be my configuration wearing the costume of a fact about the
        # model. Sequential offload is also the closest diffusers equivalent
        # of the path Tencent itself auto-enables on cards under 60 GB
        # (group offload, one block per group), and it is the mode three of
        # the four measured candidates already ran under.
        "offload": "sequential",
        # NO width/height ON PURPOSE, and the loader passes neither: this
        # pipeline accepts no such kwargs. It derives the canvas from the
        # reference image's aspect ratio against its trained 480p bucket
        # list (target_size 640, stride 16) and centre-crops to the closest
        # bucket. The output's real dimensions are read from the frames.
        "frames": 121,
        "fps": 24,
        # MEASURED by the same $0 read, from the listing's own byte counts
        # over exactly the allow patterns: transformer 15.52 + text_encoder
        # 13.17 + vae 2.35 + image_encoder 0.80 + text_encoder_2 0.41 +
        # tokenizers/configs. The community repo ships bf16 shards, which is
        # why this is half the search-snippet estimate the row landed with.
        "download_gib": 32.26,
        "steps": 12,
        "sampling_source": "official README optimal-config table + Tencent "
                           "PIPELINE_CONFIGS (12 is Tencent's own default "
                           "for this checkpoint)",
        "allow": [
            "model_index.json", "transformer/*", "text_encoder/*",
            "tokenizer/*", "text_encoder_2/*", "tokenizer_2/*",
            "image_encoder/*", "feature_extractor/*", "scheduler/*",
            "vae/*", "guider/*",
        ],
    },
}

# HunyuanVideo-1.5 carried ARCHITECTURE_NOT_RESOLVED here from 2026-08-29
# until later the same day. Owner rule: probe only what resolves from
# authoritative material without guessing — and at first it did not: the
# tencent repository is a non-diffusers layout with eleven complete
# checkpoints sharing one transformer/ directory. What resolved it, from
# source rather than inference: diffusers 0.36.0+ ships
# HunyuanVideo15ImageToVideoPipeline (this image pins 0.38.0), and the
# hunyuanvideo-community org publishes per-variant diffusers-layout repos, so
# exactly the 480p I2V step-distilled checkpoint loads the ordinary
# from_pretrained way. The row's configuration is cited from Tencent's own
# code at Tencent-Hunyuan/HunyuanVideo-1.5@60783e70, not from a card summary.
NOT_EVALUATED: dict[str, str] = {}

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
    """A probe ended at a named stage. Carries the stage, never a verdict.

    `report` is everything that HAD been measured when it stopped, attached
    on the way out by run(). A failed probe is not a wasted rental: which
    stage broke, how long the download had run, how many bytes had landed and
    what the card was holding are the answer for that candidate, and throwing
    them away would leave a paid job reporting only the word that it failed.
    """

    def __init__(self, failure: str, detail: str, report: dict | None = None):
        super().__init__(f"{failure}: {detail}")
        self.failure = failure
        self.detail = detail
        self.report = report or {}


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


_SHA_REVISION = re.compile(r"^[0-9a-f]{40}$")


def spec(model_key: str) -> dict:
    """The server's row for this benchmark key. Never a caller-supplied repo.

    A row whose revision is not a pinned 40-hex commit is refused OUTRIGHT,
    before any bytes move: "main" names whatever the publisher pushes next,
    and a benchmark of moving bytes measures nothing anyone can cite. Rows
    land with an interim ref so the $0 registry read can resolve them, and
    this guard is what makes that interim state unspendable.
    """
    if model_key in NOT_EVALUATED:
        raise ProbeStop("MODEL_ERROR",
                        f"{model_key} is NOT_EVALUATED: {NOT_EVALUATED[model_key]}")
    if model_key not in PROBE_MODELS:
        raise ProbeStop("MODEL_ERROR", f"unknown probe model {model_key!r}")
    row = PROBE_MODELS[model_key]
    if not _SHA_REVISION.match(row.get("revision") or ""):
        raise ProbeStop(
            "MODEL_ERROR",
            f"{model_key} revision {row.get('revision')!r} is not a pinned "
            "commit sha — refusing to fetch moving bytes",
        )
    return row


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
    disk: dict = {}
    before: dict = {}

    if torch is None:  # pragma: no cover - real path only
        import torch

    try:
        return _measure(
            job, input_path, output_path, model_key, spec_row, phases,
            disk, before, load_pipeline, torch, fetch,
        )
    except ProbeStop as stop:
        # THE MEASUREMENTS SURVIVE THE FAILURE. Everything measured up to the
        # stage that broke is attached here and travels back with the error,
        # because "wan22 fetched 61 GiB in 990s and ran out of budget" is the
        # answer for that candidate, while "it failed" is not — and the GPU
        # was rented either way.
        stop.report = probe_report(
            model_key, spec_row, phases,
            before=before, peaks=cuda_peaks(torch),
            failure=stop.failure, detail=stop.detail,
            identity=cuda_identity(torch),
        )
        stop.report.update(disk)
        raise


def _measure(job, input_path, output_path, model_key, spec_row, phases,
             disk, before, load_pipeline, torch, fetch) -> dict:
    """The probe itself. Mutates `disk` and `before` in place so that a stop
    at any stage still carries whatever had been measured before it."""

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    before.update(cuda_snapshot(torch))

    # DISK FIRST. Cheapest possible refusal, and it distinguishes "this card
    # cannot hold the model" from "this worker had nowhere to put it" — two
    # findings that would otherwise both arrive as a failed probe.
    where = PROBE_CACHE if os.path.isdir(PROBE_CACHE) else "/tmp"
    free = free_disk_bytes(where)
    disk.update({"disk_free_bytes": free, "disk_total_bytes": total_disk_bytes(where)})
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

    # width/height travel ONLY for rows that carry them. HunyuanVideo-1.5's
    # pipeline accepts neither kwarg — it derives the canvas from the
    # reference image's aspect against its trained bucket list — and passing
    # them would TypeError a paid job at the last possible moment.
    call_kwargs = {
        "image": image,
        "prompt": job["params"]["prompt"],
        "num_frames": spec_row["frames"],
        **_sampling(spec_row),
    }
    if spec_row.get("width"):
        call_kwargs["width"] = spec_row["width"]
    if spec_row.get("height"):
        call_kwargs["height"] = spec_row["height"]
    try:
        result = phases.time("inference", lambda: pipe(**call_kwargs))
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
        # Belt and braces: the module-level assignment above is the one that
        # matters, but these directories must EXIST and be ours before the
        # hub or its chunk-cache tries to create them inside a root-owned
        # parent — which is precisely how the first probe died.
        for path in (os.environ["HF_HOME"], os.environ["HF_HUB_CACHE"],
                     os.environ["HF_XET_CACHE"], os.environ["XDG_CACHE_HOME"]):
            os.makedirs(path, exist_ok=True)

        from huggingface_hub import snapshot_download

        local_dir = _cache_dir(spec_row)
        os.makedirs(local_dir, exist_ok=True)
        return snapshot_download(
            spec_row["repo"],
            revision=spec_row["revision"],
            local_dir=local_dir,
            cache_dir=os.environ["HF_HUB_CACHE"],
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
