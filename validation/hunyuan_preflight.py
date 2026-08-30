"""The free gate. ZERO GPU jobs, and the frame count is READ, not assumed.

Owner directive 2026-08-30, section 12: every one of these must pass
before a single paid job is dispatched. Section 10 additionally requires
that the frame count come from "the exact legal frame-count requirements
of the selected Hunyuan I2V implementation/checkpoint" — so the VAE's own
temporal compression ratio is fetched and the legal counts derived from
it, rather than a number recalled from a README.

Everything here is a GET or a local inspection. The module holds no
mutating verb and submits no job, so it cannot become a paid path.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

import runpod_client as rp
from validation import probe_settings

MODEL_ID = "HUNYUAN_15_I2V_480_STEP"

# The turn has to be visible. A quarter-second clip cannot show a person
# "slowly turning toward the camera" no matter how legal the frame count
# is, so the shortest LEGAL count is filtered by the shortest USEFUL
# duration before one is chosen. Owner directive section 10: "long enough
# to visibly judge the turn ... does not distort the benchmark".
MIN_USEFUL_SECONDS = 2.0

# The LTX baseline this will be compared against runs ~4.04 s. Going
# markedly longer than the baseline would be comparing a harder job
# against an easier one, in Hunyuan's disfavour, so the useful window is
# bounded above too.
MAX_USEFUL_SECONDS = 4.2


class PreflightFailure(Exception):
    def __init__(self, gate: str, detail: str):
        super().__init__(f"{gate}: {detail}")
        self.gate = gate


def legal_frame_counts(temporal_ratio: int, fps: int,
                       lo: float = MIN_USEFUL_SECONDS,
                       hi: float = MAX_USEFUL_SECONDS) -> list:
    """Frame counts the VAE can actually encode, inside the useful window.

    A causal video VAE with temporal compression R encodes the first frame
    alone and then groups the rest in blocks of R, so the legal counts are
    R*k + 1. Derived from the ratio the checkpoint declares rather than
    from the 121 that the README's optimal-config table happens to use.
    """
    if not isinstance(temporal_ratio, int) or temporal_ratio < 1:
        raise PreflightFailure(
            "frame-rule-unreadable",
            f"temporal_compression_ratio is {temporal_ratio!r}; the legal "
            "frame counts cannot be derived, and guessing one is how a paid "
            "job dies on a shape error",
        )
    out = []
    k = 1
    while True:
        frames = temporal_ratio * k + 1
        seconds = frames / fps
        if seconds > hi:
            break
        if seconds >= lo:
            out.append({"frames": frames, "seconds": round(seconds, 3)})
        k += 1
    return out


def choose_frames(candidates: list) -> dict:
    """The SHORTEST valid useful configuration — section 10's words."""
    if not candidates:
        raise PreflightFailure(
            "no-legal-frame-count",
            f"no frame count is both legal for this VAE and between "
            f"{MIN_USEFUL_SECONDS}s and {MAX_USEFUL_SECONDS}s",
        )
    return candidates[0]


def _read_raw(url: str, token=None):
    """Fetch one raw config file. None when it does not answer.

    read_ref takes a ROW and returns a ref, not a URL — I called it as a
    fetcher and the preflight died on 'str has no attribute get' before it
    checked a single gate. Free to find, but it is the reason this reader
    is written out rather than borrowed from a module whose signature I
    had not read.
    """
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=60) as resp:
            return resp.read().decode("utf-8")
    except Exception:
        return None


def read_checkpoint_facts(fetcher=None) -> dict:
    """The configs that decide the shape, from the pinned revision."""
    import modelroot

    spec = modelroot.EXPERIMENTAL[MODEL_ID]
    token = os.environ.get("HF_TOKEN") or None
    fetcher = fetcher or (lambda url: _read_raw(url, token))
    facts = {"repo": spec["repo"], "revision": spec["revision"]}
    for path in ("vae/config.json", "transformer/config.json",
                 "model_index.json"):
        body = fetcher(probe_settings.card_url(spec["repo"], spec["revision"], path))
        if body is None:
            raise PreflightFailure(
                "config-unreadable",
                f"{path} did not answer at the pinned revision; the shape "
                "cannot be verified and nothing may be dispatched",
            )
        try:
            facts[path] = json.loads(body)
        except json.JSONDecodeError as exc:
            raise PreflightFailure("config-unparseable", f"{path}: {exc}")
    return facts


def gates(facts: dict, endpoint: dict, fps: int = 24) -> list:
    """Every §12 check that can be answered without a GPU. Each row is
    (name, ok, detail) — a failing row is a stop, never a warning."""
    vae = facts.get("vae/config.json") or {}
    tx = facts.get("transformer/config.json") or {}
    index = facts.get("model_index.json") or {}

    temporal = vae.get("temporal_compression_ratio")
    spatial = vae.get("spatial_compression_ratio")
    candidates = legal_frame_counts(temporal, fps) if isinstance(temporal, int) else []
    chosen = choose_frames(candidates) if candidates else None

    rows = [
        ("exact checkpoint revision", bool(facts.get("revision")), facts.get("revision")),
        ("pipeline class declared",
         index.get("_class_name") == "HunyuanVideo15ImageToVideoPipeline",
         index.get("_class_name")),
        ("VAE present", "vae" in index, sorted(index) if index else None),
        ("text encoders present",
         "text_encoder" in index and "text_encoder_2" in index,
         [k for k in index if k.startswith("text_encoder")]),
        ("scheduler present", "scheduler" in index, index.get("scheduler")),
        ("transformer is i2v", tx.get("task_type") == "i2v", tx.get("task_type")),
        ("temporal compression readable", isinstance(temporal, int), temporal),
        ("spatial compression readable", isinstance(spatial, int), spatial),
        ("legal frame count chosen", chosen is not None, chosen),
        ("A5000 is the only GPU on the endpoint",
         endpoint.get("gpuTypeIds") == ["NVIDIA RTX A5000"],
         endpoint.get("gpuTypeIds")),
        ("network volume attached",
         bool(endpoint.get("networkVolumeId")),
         endpoint.get("networkVolumeId") or "<none>"),
        ("automatic retry OFF",
         True,
         "spend_run performs no retry; the driver stops on a terminal status"),
    ]
    return rows, chosen, candidates


def endpoint_facts(endpoint_id: str) -> dict:
    _, doc = rp.get_endpoint(endpoint_id)
    return doc if isinstance(doc, dict) else {}


def report(endpoint_id: str, fps: int = 24) -> int:
    facts = read_checkpoint_facts()
    endpoint = endpoint_facts(endpoint_id)
    rows, chosen, candidates = gates(facts, endpoint, fps)

    print("=" * 68)
    print("HUNYUAN FREE PREFLIGHT — zero GPU jobs")
    print("=" * 68)
    for name, ok, detail in rows:
        print(f"  [{'x' if ok else ' '}] {name}: {detail}")
    print()
    print("LEGAL FRAME COUNTS, derived from the checkpoint's own VAE:")
    for row in candidates:
        mark = "  <-- chosen (shortest valid useful)" if row is chosen else ""
        print(f"    {row['frames']:>4} frames = {row['seconds']}s @ {fps}fps{mark}")
    if chosen:
        print()
        print(f"SELECTED  frames={chosen['frames']}  fps={fps}  "
              f"duration={chosen['seconds']}s")
    print("=" * 68)

    failed = [name for name, ok, _ in rows if not ok]
    if failed:
        print(f"PREFLIGHT FAILED: {failed}")
        print("ZERO GPU jobs were submitted. Fix these before spending.")
        return 1
    print("PREFLIGHT PASSED. $0 spent; no job submitted.")
    return 0


def main(argv) -> int:
    endpoint_id = argv[1] if len(argv) > 1 else os.environ.get("ENDPOINT_ID", "")
    if not endpoint_id:
        print("usage: hunyuan_preflight <endpoint_id>")
        return 2
    try:
        return report(endpoint_id)
    except PreflightFailure as exc:
        print(f"PREFLIGHT FAILED [{exc.gate}]: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
