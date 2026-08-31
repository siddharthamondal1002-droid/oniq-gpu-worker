"""WHICH SPATIAL LATENT UPSAMPLER, AT WHICH REVISION, UNDER WHICH LICENCE?

`ltx-upscaler.pin` needs one line — `<repo> <40-char sha>` — and it has stayed
empty because the three facts it encodes (the revision, the licence, and the
component config) are registry metadata that exists nowhere else. The build
container's egress proxy refuses huggingface.co (403 to CONNECT, confirmed for
huggingface.co, hf.co and cdn-lfs.huggingface.co), so they cannot be read from
there, and a plausible-looking sha typed by hand is a fabrication wearing the
costume of a fact.

A GitHub runner CAN reach the registry — `ltx-discover` and `model-bench`
already do, with the same HF_TOKEN. This module is the $0 read that resolves
the upsampler specifically, so the owner names a real revision instead of
copying one out of a browser.

WHY IT MEASURES TWO CANDIDATES RATHER THAN ONE. The diffusers LTX documentation
(fetched 2026-08-31 from raw.githubusercontent.com, which this environment CAN
reach) names both:

    Lightricks/ltxv-spatial-upscaler-0.9.7   paired with LTX-Video-0.9.7-dev
    a-r-r-o-w/LTX-0.9.8-Latent-Upsampler     paired with LTX-Video-0.9.8-13B-distilled

The first is Lightricks' own, so it falls inside the LTX Open Weights terms the
owner has accepted. The second is a personal namespace the docs point at with
upstream's own note "TODO: Update the checkpoint here once updated in LTX org",
so its right to redistribute derived weights is a separate question that its
LICENSE/NOTICE files have to answer. Measuring both, side by side, is what lets
that be decided on evidence.

It CHOOSES NOTHING. Like hf_discover, it lists and measures; naming the model
ONIQ ships is the owner's decision, and the licence is theirs to accept.

The token is used and never printed — the same discipline as hf_auth.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from validation.hf_discover import INFO_API, _get, _why

# The two the diffusers documentation names. Hardcoded rather than taken as a
# workflow input because GitHub caps workflow_dispatch at 10 inputs and
# test_no_workflow_exceeds_githubs_dispatch_input_cap enforces it — a new mode
# costs nothing, a new input costs one of ten.
CANDIDATES = (
    "Lightricks/ltxv-spatial-upscaler-0.9.7",
    "a-r-r-o-w/LTX-0.9.8-Latent-Upsampler",
)

RAW = "https://huggingface.co/{repo}/resolve/{revision}/config.json"

# Exactly what the Dockerfile's upscaler stage asserts field by field. Kept
# here so the read reports the SAME verdict the build would reach, rather than
# a looser one that lets a doomed pin through to a 25-minute build.
EXPECTED = {
    "_class_name": "LTXLatentUpsamplerModel",
    "dims": 3,
    "in_channels": 128,
    "mid_channels": 512,
    "num_blocks_per_stage": 4,
    "spatial_upsample": True,
    "temporal_upsample": False,
}

# The Dockerfile's own guard for this component: a spatial upsampler that
# weighs more than this is not a spatial upsampler.
SIZE_GUARD_BYTES = 1024**3

# A licence has to SHIP with the weights: the build refuses a bake whose
# snapshot carries no terms beside the bytes.
LICENCE_FILES = ("LICENSE", "LICENSE.md", "LICENSE.txt", "NOTICE", "NOTICE.md", "NOTICE.txt")


def _get_text(url: str, token, timeout: int = 60) -> str:
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def measure(repo: str, token, get=_get, get_text=_get_text) -> dict:
    """One candidate, against the gates the build would apply to it."""
    row: dict = {"repo": repo}
    try:
        info = get(INFO_API.format(repo=repo), token)
    except Exception as exc:
        row.update(verdict="UNREADABLE", detail=_why(exc))
        return row

    row["revision"] = info.get("sha")
    card = info.get("cardData") or {}
    row["licence"] = card.get("license") or next(
        (t.split(":", 1)[1] for t in (info.get("tags") or []) if t.startswith("license:")),
        None,
    )
    row["licence_name"] = card.get("license_name")
    row["licence_link"] = card.get("license_link")

    paths = {
        (s.get("rfilename") or ""): (s.get("size") or 0)
        for s in info.get("siblings") or []
    }
    row["root_files"] = sorted(p for p in paths if "/" not in p)
    row["licence_files"] = [p for p in row["root_files"] if p.upper() in
                            {f.upper() for f in LICENCE_FILES}]
    row["weight_bytes"] = sum(
        size for p, size in paths.items()
        if p.endswith((".safetensors", ".bin", ".pt", ".pth"))
    )
    row["within_size_guard"] = 0 < row["weight_bytes"] <= SIZE_GUARD_BYTES

    # The config decides whether this component is even the right SHAPE. Read
    # at the resolved revision, never at a branch name, so what is reported is
    # what a pin to that sha would actually bake.
    if row["revision"]:
        try:
            cfg = json.loads(
                get_text(RAW.format(repo=repo, revision=row["revision"]), token)
            )
            row["config"] = {k: cfg.get(k) for k in EXPECTED}
            row["config_mismatch"] = {
                k: cfg.get(k) for k, v in EXPECTED.items() if cfg.get(k) != v
            }
        except Exception as exc:
            row["config"] = None
            row["config_mismatch"] = {"config.json": _why(exc)}
    else:
        row["config"] = None
        row["config_mismatch"] = {"revision": "not resolved"}

    row["verdict"] = (
        "PINNABLE"
        if row["revision"] and not row["config_mismatch"]
        and row["licence_files"] and row["within_size_guard"]
        else "NOT PINNABLE"
    )
    return row


def report(token, get=_get, get_text=_get_text) -> tuple:
    if not token:
        print("BLOCKED: no credential, and an anonymous read cannot tell a "
              "gated repository from one that does not exist")
        return 2, []

    rows = [measure(repo, token, get, get_text) for repo in CANDIDATES]
    for row in rows:
        print("")
        print(f"  {row['repo']}")
        if row.get("verdict") == "UNREADABLE":
            print(f"    UNREADABLE — {row['detail']}")
            continue
        print(f"    revision      {row.get('revision')}")
        print(f"    licence tag   {row.get('licence')!r}")
        print(f"    license_name  {row.get('licence_name')!r}")
        print(f"    license_link  {row.get('licence_link')!r}")
        print(f"    licence files {row.get('licence_files') or 'NONE — the build refuses this'}")
        print(f"    root files    {row.get('root_files')}")
        print(f"    weight bytes  {row.get('weight_bytes')} "
              f"(guard {SIZE_GUARD_BYTES}, within={row.get('within_size_guard')})")
        print(f"    config        {row.get('config')}")
        if row.get("config_mismatch"):
            print(f"    MISMATCH      {row['config_mismatch']}")
        print(f"    VERDICT       {row['verdict']}")

    print("")
    print("A VERDICT OF PINNABLE IS NOT PERMISSION. It says the revision "
          "resolves, the config is the right shape, terms ship beside the "
          "weights and the size is sane. Whether those terms permit ONIQ to "
          "redistribute the weights inside a private commercial image is the "
          "owner's judgement, and reading the licence text is part of making "
          "it. To enable, put one line in ltx-upscaler.pin:")
    print("")
    for row in rows:
        if row.get("verdict") == "PINNABLE":
            print(f"    {row['repo']} {row['revision']}")
    print("")
    return (0 if any(r.get("verdict") == "PINNABLE" for r in rows) else 1), rows


def main(argv) -> int:
    import os

    from validation.hf_auth import TOKEN_VAR

    code, _ = report(os.environ.get(TOKEN_VAR))
    return code


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
