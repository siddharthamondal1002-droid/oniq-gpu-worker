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


@dataclass(frozen=True)
class Candidate:
    """One row of the owner's benchmark, as a predicate over a real listing."""

    key: str  # this benchmark's row id
    decision: str  # the token the final recommendation must return
    label: str  # how the owner wrote it
    author: str  # HF org to LIST — the search is by publisher, not by name
    must: tuple[str, ...] = ()  # every fragment must appear in the repo id
    must_not: tuple[str, ...] = ()
    prefer_diffusers: bool = True
    target_shape: tuple[int, int] | None = None  # closest supported to 704x480
    note: str = ""

    def matches(self, repo_id: str) -> bool:
        low = repo_id.lower()
        if not low.startswith(self.author.lower() + "/"):
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
        author="Lightricks",
        must=("ltx-video",),
        must_not=("13b", "gguf", "ic-lora", "control"),
        target_shape=ONIQ_SHAPE,
        note="the incumbent — every number here is the bar to beat",
    ),
    Candidate(
        key="ltx-13b",
        decision="LTX_13B",
        label="LTX-Video 13B",
        author="Lightricks",
        must=("ltx-video", "13b"),
        must_not=("gguf",),
        target_shape=ONIQ_SHAPE,
        note="bf16 13B; single-file checkpoints live at the repo root",
    ),
    Candidate(
        key="ltx-13b-fp8",
        decision="LTX_13B",
        label="LTX-Video 13B FP8",
        author="Lightricks",
        must=("ltx-video", "13b"),
        must_not=("gguf",),
        target_shape=ONIQ_SHAPE,
        note="same repo as 13B — fp8 is a FILE, not a repository; "
        "resolved from the single-file listing",
    ),
    Candidate(
        key="ltx-13b-distilled",
        decision="LTX_13B",
        label="LTX-Video 13B distilled",
        author="Lightricks",
        must=("ltx-video", "13b"),
        must_not=("gguf",),
        target_shape=ONIQ_SHAPE,
        note="distilled = fewer steps; also a file-level variant",
    ),
    Candidate(
        key="wan21-i2v-14b-480p",
        decision="WAN2_1_I2V_14B",
        label="Wan2.1 I2V-14B-480P",
        author="Wan-AI",
        must=("wan2.1", "i2v", "14b", "480p"),
        target_shape=(832, 480),
        note="OWNER-PRIORITISED: 480P is the closest official shape to "
        "ONIQ's 704x480 and short clips",
    ),
    Candidate(
        key="wan21-i2v-14b-720p",
        decision="WAN2_1_I2V_14B",
        label="Wan2.1 I2V-14B-720P",
        author="Wan-AI",
        must=("wan2.1", "i2v", "14b", "720p"),
        target_shape=(1280, 720),
        note="carried for completeness; a bigger canvas costs VRAM and "
        "runtime that ONIQ's proven shape does not need",
    ),
    Candidate(
        key="wan22-i2v-a14b",
        decision="WAN2_2_I2V_A14B",
        label="Wan2.2 I2V-A14B",
        author="Wan-AI",
        must=("wan2.2", "i2v", "a14b"),
        target_shape=(832, 480),
        note="SEPARATE CANDIDATE from Wan2.1 — mixture-of-experts, expect "
        "transformer/ AND transformer_2/",
    ),
    Candidate(
        key="hunyuanvideo-1.5-i2v",
        decision="HUNYUAN_VIDEO_1_5_I2V",
        label="HunyuanVideo-1.5 I2V",
        author="tencent",
        must=("hunyuanvideo-1.5",),
        target_shape=(832, 480),
        note="evaluate the current open-weight implementation and its "
        "documented offloading configuration",
    ),
    Candidate(
        key="cogvideox-5b-i2v",
        decision="COGVIDEOX_I2V",
        label="CogVideoX-5B-I2V",
        author="THUDM",
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
    for role in ROLE_DIRS:
        total = sum(
            size
            for path, size in sizes.items()
            if path.startswith(role + "/") and path.endswith(WEIGHT_SUFFIXES)
        )
        if total:
            roles[role] = total
    row["roles"] = roles

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
    row["total_weight_bytes"] = sum(
        size for path, size in sizes.items() if path.endswith(WEIGHT_SUFFIXES)
    )
    row["config_paths"] = sorted(
        path for path in sizes if path.endswith("config.json")
    )
    return row


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
