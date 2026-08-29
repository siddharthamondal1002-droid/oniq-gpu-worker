"""The nine benchmark candidates, resolved to repositories that ACTUALLY exist.

Owner brief 2026-08-29 ("VIDEO MODEL BENCHMARK — EXPANDED OPEN-SOURCE SET")
widened the field past LTX and flipped Wan from disabled to candidate. Before
a single GPU-second is spent, six questions have to be answered per model,
and every one of them is answerable for nothing:

  1. the exact checkpoint     2. the exact precision
  3. whether fp8/offload      4. the actual VRAM requirement
  5. whether an A5000 runs it 6. the minimum GPU that does

This module answers 1-3 by MEASURING the registry. `vram.py` turns its byte
counts into 4-6.

TWO RULES, both bought with earlier mistakes.

SEARCH, NEVER GUESS. hf_discover exists because the Dockerfile named an LTX
checkpoint that does not exist, and an anonymous 401 was read as "gated"
rather than "absent". Every candidate here is therefore a PREDICATE over the
publisher's real listing, not a repository id typed from memory. If nothing
matches, that is the finding — the predicate is never loosened until it hits
something.

MEASURE BYTES, NEVER PARAMETER COUNTS. A name is not a size. "A14B" and
"14B" are the same string length and, as this module shows, not the same
memory: Wan2.2's I2V-A14B is a two-expert mixture carrying `transformer/`
AND `transformer_2/`, so the weights on disk are roughly double what the
label suggests, while only one expert is resident at a time. That is exactly
the confusion the owner's "Do not assume A14B means the same architecture or
memory requirement as Wan2.1 14B" warns about, and the only defence is to
add up the actual files.

The token is used and never printed — same discipline as hf_auth.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

LIST_API = "https://huggingface.co/api/models"
INFO_API = "https://huggingface.co/api/models/{repo}?blobs=true"
RAW_API = "https://huggingface.co/{repo}/resolve/{revision}/{path}"

# Component directories a diffusers pipeline splits its weights across. Roles
# matter because offloading is per-component: what the GPU must hold at once
# is a function of WHICH of these is resident, never of the repository total.
ROLE_DIRS = (
    "transformer",
    "transformer_2",  # Wan2.2 MoE: the second expert. Absent elsewhere.
    "text_encoder",
    "text_encoder_2",
    "vae",
    "image_encoder",  # Wan I2V conditions on CLIP vision; LTX does not.
)

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pth", ".pt")

# torch_dtype as configs spell it, mapped to vram.py's vocabulary. Read rather
# than assumed: Wan ships fp32 transformers, so treating every checkpoint as
# bf16 overstates what a bf16 load actually costs by a factor of two.
SAFETENSOR_DTYPES = {
    "F64": "fp32", "F32": "fp32", "BF16": "bf16", "F16": "fp16",
    "F8_E4M3": "fp8", "F8_E5M2": "fp8", "I8": "int8", "U8": "int8",
}

DTYPE_NAMES = {
    "float32": "fp32", "torch.float32": "fp32", "float": "fp32",
    "bfloat16": "bf16", "torch.bfloat16": "bf16",
    "float16": "fp16", "torch.float16": "fp16", "half": "fp16",
    "float8_e4m3fn": "fp8", "torch.float8_e4m3fn": "fp8",
}


@dataclass(frozen=True)
class Candidate:
    """One row of the owner's benchmark, as a predicate over a real listing."""

    key: str  # this benchmark's row id
    decision: str  # the token the final recommendation must return
    label: str  # how the owner wrote it
    authors: tuple[str, ...]  # HF orgs to LIST — by publisher, never by name
    must: tuple[str, ...] = ()  # every fragment must appear in the repo id
    must_not: tuple[str, ...] = ()
    prefer_diffusers: bool = True
    target_shape: tuple[int, int] | None = None  # closest supported to 704x480
    # Which sub-checkpoint, where a repository ships several complete ones.
    # HunyuanVideo-1.5 keeps eleven under transformer/ — one per resolution
    # and task — and ONIQ needs exactly the 480p I2V one. Summing them
    # describes a machine nobody will ever build.
    variant: str | None = None
    # Which root-level file, where the variant is a FILE rather than a
    # directory. LTX ships its 13B base, fp8 and distilled weights this way,
    # so three of the owner's rows live inside one repository.
    checkpoint_file: str | None = None
    note: str = ""

    @property
    def author(self) -> str:
        return self.authors[0]

    def matches(self, repo_id: str) -> bool:
        low = repo_id.lower()
        if not any(low.startswith(a.lower() + "/") for a in self.authors):
            return False
        if any(frag.lower() not in low for frag in self.must):
            return False
        return all(frag.lower() not in low for frag in self.must_not)


# ONIQ's proven production shape. Every candidate is pointed at the closest
# size its publisher actually supports, because comparing a 480p model to a
# 720p one measures the resolution, not the model.
ONIQ_SHAPE = (704, 480)


CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        key="ltx-2b",
        decision="LTX_2B",
        label="LTX-Video 2B",
        authors=("Lightricks",),
        must=("ltx-video",),
        must_not=("13b", "gguf", "ic-lora", "control", "lora"),
        target_shape=ONIQ_SHAPE,
        note="the incumbent — every number here is the bar to beat, and the "
        "one row with a REAL measured VRAM peak to check the maths against",
    ),
    # THE THREE 13B ROWS SHARE ONE REPOSITORY AND DIFFER BY FILE. Lightricks
    # publishes no `LTX-Video-13B` repo at all: the 13B dev and fp8 weights are
    # root-level files inside Lightricks/LTX-Video (26.62 and 14.62 GiB), and
    # only the distilled build has a repository of its own. Measured, not
    # assumed — the first run collapsed all three onto one repo and reported
    # the same numbers three times.
    Candidate(
        key="ltx-13b",
        decision="LTX_13B",
        label="LTX-Video 13B",
        authors=("Lightricks",),
        must=("ltx-video",),
        must_not=("13b", "gguf", "ic-lora", "control", "lora"),
        checkpoint_file="ltxv-13b-0.9.8-dev.safetensors",
        target_shape=ONIQ_SHAPE,
        note="bf16 13B — a FILE in Lightricks/LTX-Video, not a repository",
    ),
    Candidate(
        key="ltx-13b-fp8",
        decision="LTX_13B",
        label="LTX-Video 13B FP8",
        authors=("Lightricks",),
        must=("ltx-video",),
        must_not=("13b", "gguf", "ic-lora", "control", "lora"),
        checkpoint_file="ltxv-13b-0.9.8-dev-fp8.safetensors",
        target_shape=ONIQ_SHAPE,
        note="the publisher's own fp8 build of the same weights",
    ),
    Candidate(
        key="ltx-13b-distilled",
        decision="LTX_13B",
        label="LTX-Video 13B distilled",
        authors=("Lightricks",),
        must=("ltx-video", "13b", "distilled"),
        must_not=("gguf", "ic-lora"),
        target_shape=ONIQ_SHAPE,
        note="fewer denoising steps — the one 13B build with its own repo",
    ),
    Candidate(
        key="wan21-i2v-14b-480p",
        decision="WAN2_1_I2V_14B",
        label="Wan2.1 I2V-14B-480P",
        authors=("Wan-AI",),
        must=("wan2.1", "i2v", "14b", "480p"),
        target_shape=(832, 480),
        note="OWNER-PRIORITISED: 480P is the closest official shape to "
        "ONIQ's 704x480 and short clips",
    ),
    Candidate(
        key="wan21-i2v-14b-720p",
        decision="WAN2_1_I2V_14B",
        label="Wan2.1 I2V-14B-720P",
        authors=("Wan-AI",),
        must=("wan2.1", "i2v", "14b", "720p"),
        target_shape=(1280, 720),
        note="carried for completeness; a bigger canvas costs VRAM and "
        "runtime that ONIQ's proven shape does not need",
    ),
    Candidate(
        key="wan22-i2v-a14b",
        decision="WAN2_2_I2V_A14B",
        label="Wan2.2 I2V-A14B",
        authors=("Wan-AI",),
        must=("wan2.2", "i2v", "a14b"),
        target_shape=(832, 480),
        note="SEPARATE CANDIDATE from Wan2.1 — mixture-of-experts, expect "
        "transformer/ AND transformer_2/",
    ),
    Candidate(
        key="hunyuanvideo-1.5-i2v",
        decision="HUNYUAN_VIDEO_1_5_I2V",
        label="HunyuanVideo-1.5 I2V",
        authors=("tencent",),
        must=("hunyuanvideo-1.5",),
        variant="480p_i2v",
        target_shape=(832, 480),
        note="ELEVEN complete checkpoints share transformer/, one per "
        "resolution and task. ONIQ needs 480p_i2v and only that one.",
    ),
    Candidate(
        key="cogvideox-5b-i2v",
        decision="COGVIDEOX_I2V",
        label="CogVideoX-5B-I2V",
        # THUDM published CogVideoX and later became zai-org. Both are listed
        # because the first run found nothing under THUDM alone, and "the
        # publisher renamed itself" is not the same finding as "the model does
        # not exist" — the distinction hf_discover was written to protect.
        authors=("THUDM", "zai-org"),
        must=("cogvideox", "5b", "i2v"),
        target_shape=(720, 480),
        note="lower-resource comparison candidate",
    ),
)

def _get(url: str, token, timeout: int = 60):
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _why(exc) -> str:
    """Why a fetch failed, in a form that distinguishes the cases that matter.

    401 anonymous means gated OR absent and cannot be told apart; that
    ambiguity is the whole reason hf_discover was written, so it is spelled
    out here rather than collapsed into "failed".
    """
    code = getattr(exc, "code", None)
    if code == 401:
        return "HTTP 401 (gated or absent — indistinguishable without a token)"
    if code == 403:
        return "HTTP 403 (terms not accepted for this account)"
    return f"HTTP {code}" if code else type(exc).__name__


def catalogue(author: str, token, get=_get) -> list:
    """Everything one publisher offers. The listing, not a guess."""
    query = urllib.parse.urlencode({"author": author, "limit": 1000})
    return get(f"{LIST_API}?{query}", token)


def repo_ids(listing) -> list[str]:
    out = []
    for entry in listing or []:
        rid = entry.get("id") or entry.get("modelId")
        if rid:
            out.append(rid)
    return sorted(set(out))


def _licence(info: dict) -> dict:
    """Licence as published. `license: other` means "read the card", so the
    card's own name/link fields are carried too — a bare "other" is not a
    licence finding, it is a pointer, and the owner's brief asks for the
    licence verified before production use."""
    card = info.get("cardData") or {}
    tags = info.get("tags") or []
    spdx = card.get("license") or next(
        (t.split(":", 1)[1] for t in tags if t.startswith("license:")), None
    )
    return {
        "licence": spdx,
        "licence_name": card.get("license_name"),
        "licence_link": card.get("license_link"),
    }


def measure(repo: str, token, get=_get) -> dict:
    """One repository, by the bytes it actually ships.

    Splits weights by ROLE, because the offloading question is per-component:
    with model-level offload the GPU holds one component at a time, so the
    largest single role — not the repository total — sets the floor.
    """
    row: dict = {"repo": repo}
    try:
        info = get(INFO_API.format(repo=repo), token)
    except Exception as exc:  # noqa: BLE001 - reported, never swallowed
        row.update(verdict="UNREADABLE", detail=_why(exc))
        return row

    row["revision"] = info.get("sha")
    row["gated"] = bool(info.get("gated"))
    row.update(_licence(info))

    sizes = {
        (s.get("rfilename") or ""): (s.get("size") or 0)
        for s in info.get("siblings") or []
    }
    row["is_diffusers"] = "model_index.json" in sizes

    roles: dict[str, int] = {}
    variants: dict[str, dict[str, int]] = {}
    role_files: dict[str, list] = {}
    nested = 0
    for path, size in sizes.items():
        if not path.endswith(WEIGHT_SUFFIXES):
            continue
        parts = path.split("/")
        if parts[0] not in ROLE_DIRS:
            continue
        if len(parts) == 2:
            roles[parts[0]] = roles.get(parts[0], 0) + size
            role_files.setdefault(parts[0], []).append((path, size))
        elif len(parts) == 3 and parts[1] not in ROLE_DIRS:
            # A named sub-checkpoint: transformer/480p_i2v/... Kept apart,
            # never added to its siblings.
            variants.setdefault(parts[0], {})
            variants[parts[0]][parts[1]] = variants[parts[0]].get(parts[1], 0) + size
        else:
            # A NESTED PIPELINE. LTX-Video-0.9.8-13B-distilled ships a whole
            # second copy of itself under vae/ — vae/transformer/...,
            # vae/text_encoder/... A prefix match reads that as a 44 GiB VAE,
            # which is how the first matrix came to project 750 GiB of decode
            # for a model that runs on one card. Counted and excluded.
            nested += size
    row["roles"] = roles
    row["role_variants"] = variants
    row["nested_bytes"] = nested
    # The raw per-file listing behind each role total. Printed so a wrong
    # sum is visible as a wrong sum rather than arriving as a confident GiB
    # figure nobody can check.
    row["role_files"] = {
        role: sorted(files, key=lambda f: -f[1]) for role, files in role_files.items()
    }

    # Root-level checkpoints. This is where LTX keeps its fp8 and distilled
    # variants: they are FILES inside the same repository, not repositories,
    # so a candidate list built only from repo names would miss two of the
    # four LTX rows the owner asked for.
    row["single_files"] = sorted(
        (
            {"name": path, "bytes": size}
            for path, size in sizes.items()
            if "/" not in path and path.endswith(WEIGHT_SUFFIXES)
        ),
        key=lambda f: -f["bytes"],
    )
    row["total_weight_bytes"] = sum(roles.values())
    row["config_paths"] = sorted(
        path for path in sizes if path.endswith("config.json")
    )
    return row


def header_dtype(repo: str, revision: str, path: str, token,
                 fetch=None) -> str | None:
    """The dtype actually on disk, read out of the safetensors header.

    Most diffusers component configs carry no `torch_dtype`, so the shipped
    precision cannot be read from JSON — and it is not a detail. Wan ships
    fp32 transformers; assuming bf16 on disk means a bf16 load is reported at
    twice its real size, which is the difference between one card and two.

    A safetensors file opens with a little-endian u64 header length followed
    by that many bytes of JSON naming every tensor's dtype. A Range request
    for the first 64 KiB is enough to read it, so this costs one partial
    fetch per component rather than a download. The dtype reported is the one
    the LARGEST tensor uses: headers routinely mix a few fp32 norms into an
    otherwise bf16 checkpoint, and the bulk is what sets the bytes.
    """
    if not path.endswith(".safetensors"):
        return None
    url = RAW_API.format(repo=repo, revision=revision or "main", path=path)
    raw = (fetch or _range_get)(url, token, 65536)
    if not raw or len(raw) < 8:
        return None
    length = int.from_bytes(raw[:8], "little")
    if length <= 0 or 8 + length > len(raw):
        return None
    try:
        header = json.loads(raw[8:8 + length].decode("utf-8"))
    except Exception:  # noqa: BLE001
        return None
    biggest, best = None, -1
    for name, meta in header.items():
        if name == "__metadata__" or not isinstance(meta, dict):
            continue
        shape = meta.get("shape") or []
        size = 1
        for dim in shape:
            size *= dim if isinstance(dim, int) else 1
        if size > best:
            biggest, best = meta.get("dtype"), size
    return SAFETENSOR_DTYPES.get(str(biggest or "").upper())


def _range_get(url: str, token, nbytes: int):
    request = urllib.request.Request(url)
    request.add_header("Range", f"bytes=0-{nbytes - 1}")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=60) as resp:
            return resp.read()
    except Exception:  # noqa: BLE001 — an unreadable header is not a failure
        return None


def fetch_config(repo: str, revision: str, path: str, token, get=_get) -> dict:
    """A component's config.json, for the shape arithmetic in vram.py.

    Read rather than assumed: hidden size, depth and the VAE's compression
    ratios decide the activation working set, and those are the numbers that
    separate "fits in 24 GB" from "does not".
    """
    url = RAW_API.format(repo=repo, revision=revision or "main", path=path)
    return get(url, token)


def variant_of(name: str) -> str:
    """Precision/variant of a single-file checkpoint, from its own name.

    LTX publishes `...-fp8.safetensors` and `...-distilled...safetensors`
    beside the base weights, which is the only reason three of the owner's
    four LTX rows can share one repository.
    """
    low = name.lower()
    if "fp8" in low:
        return "fp8"
    if "gguf" in low:
        return "gguf"
    if "distilled" in low:
        return "distilled"
    return "base"


def resolve(candidate: Candidate, ids: list[str]) -> list[str]:
    """Which real repositories satisfy this candidate's predicate."""
    return [rid for rid in ids if candidate.matches(rid)]


def preferred(candidate: Candidate, matches: list[str]) -> str | None:
    """The one repository to measure for this row.

    Diffusers-format repositories are preferred where the publisher ships
    both, because the worker's pipeline stack loads that layout — a repo
    ONIQ cannot load is not a candidate, whatever its quality.
    """
    if not matches:
        return None
    if candidate.prefer_diffusers:
        diffusers = [m for m in matches if m.lower().endswith("-diffusers")]
        if diffusers:
            return sorted(diffusers, key=len)[0]
    plain = [m for m in matches if not m.lower().endswith("-diffusers")]
    return sorted(plain or matches, key=len)[0]


@dataclass
class Row:
    """One measured benchmark row, ready for the VRAM and cost matrices."""

    candidate: Candidate
    repo: str | None = None
    alternates: list[str] = field(default_factory=list)
    measurement: dict = field(default_factory=dict)
    configs: dict = field(default_factory=dict)

    @property
    def status(self) -> str:
        if not self.repo:
            return "NOT-PUBLISHED"
        return self.measurement.get("verdict") or "MEASURED"
