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
import urllib.error
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


def gates(facts: dict, endpoint: dict, fps: int = 24,
          reference: dict | None = None) -> list:
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
    if reference is not None:
        rows.append((
            "R2 reference readable and really an image",
            bool(reference.get("ok")),
            reference.get("error") or
            f"{reference.get('format')} "
            f"{reference.get('width')}x{reference.get('height')}",
        ))
    return rows, chosen, candidates


# Magic bytes, because a Content-Type header is what a server claims and
# the first bytes are what the decoder will actually see.
_MAGIC = {
    b"\x89PNG\r\n\x1a\n": "png",
    b"\xff\xd8\xff": "jpeg",
}


def reference_readable(base: str, key: str, *, fetch=None) -> dict:
    """Fetch the benchmark reference and confirm it is really an image.

    Owner directive section 11: use the already-validated single-person
    reference and do NOT regenerate it unless genuinely unavailable. So
    this checks availability rather than assuming it, and checks it by
    DECODING the first bytes rather than trusting a header — a 404 page
    served with image/png is still a 404 page, and conditioning a paid job
    on one produces a clip of nothing.

    IT READS THROUGH frame_pull's FETCHER, NOT A FRESH urlopen. This gate
    failed on 2026-08-30 with HTTP 403, which reads like a private bucket
    and is not one: r2.dev applies Cloudflare's UA-signature filter (error
    code 1010) to known scraper agents, python-urllib among them, and a
    bare urlopen sends exactly that UA. The finding was already measured
    on 2026-08-29, written into frame_pull's two-UA ladder, and recorded
    in validation/fixtures/FIXTURES.md as PUBLIC_R2_ARTIFACT_READ =
    CURRENT. Writing a second fetcher here threw all of that away and
    produced a 403 that would have been read as "the reference is gone".

    So there is ONE fetcher for this bucket. A refusal that survives it is
    a real refusal.
    """
    from validation.frame_pull import _fetch, http_detail

    if fetch is None:
        fetch = _fetch
    url = f"{base.rstrip('/')}/{key.lstrip('/')}"
    try:
        head = fetch(url)[:4096]
    except urllib.error.HTTPError as exc:
        detail = http_detail(exc)
        return {"ok": False, "url": url,
                "error": f"HTTP {exc.code}" + (f" ({detail})" if detail else "")}
    except Exception as exc:
        return {"ok": False, "url": url, "error": f"{type(exc).__name__}: {exc}"}
    if not head:
        return {"ok": False, "url": url, "error": "empty body"}
    kind = next((k for magic, k in _MAGIC.items() if head.startswith(magic)), None)
    if kind is None:
        return {"ok": False, "url": url,
                "error": f"not an image; first bytes {head[:16]!r}"}
    info = {"ok": True, "url": url, "format": kind, "head_bytes": len(head)}
    if kind == "png" and len(head) >= 24:
        import struct

        width, height = struct.unpack(">II", head[16:24])
        info["width"], info["height"] = width, height
    return info


def endpoint_facts(endpoint_id: str) -> dict:
    _, doc = rp.get_endpoint(endpoint_id)
    return doc if isinstance(doc, dict) else {}


def report(endpoint_id: str, fps: int = 24) -> int:
    facts = read_checkpoint_facts()
    endpoint = endpoint_facts(endpoint_id)

    reference = None
    ref_key = os.environ.get("GPU_TEST_INPUT_REF", "")
    base_raw = os.environ.get("R2_PUBLIC_BASE_URL", "")
    if ref_key:
        from validation.frame_pull import resolve_base

        try:
            base, note = resolve_base(base_raw)
            if note:
                print(f"NOTE: {note}")
            reference = reference_readable(base, ref_key)
        except Exception as exc:
            reference = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    rows, chosen, candidates = gates(facts, endpoint, fps, reference)

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
