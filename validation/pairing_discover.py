"""WHICH LTX CHECKPOINT PAIRS WITH THE PINNED UPSAMPLER — AND FITS THE CARD?

The mirror of upscaler_discover. That one asked "which upsampler may we bake
beside the checkpoint we have"; run 33426496040 answered NONE, because the vae
ONIQ bakes (LTX-Video@8984fa25, a four-stage 0.9.0/2B-era encoder) is a
different network from the one the 0.9.7 upsampler was trained beside.

Owner directive 2026-08-31: fix the PAIRING by moving the checkpoint, not by
relaxing the gate. This module measures what that would cost before anything
is repointed, because two facts decide whether the directive is even
achievable and neither can be assumed:

  1. WHICH candidate's vae matches the upsampler's, on exactly the fields the
     Dockerfile's LATENT SPACE gate compares. A candidate that does not match
     is not a fix, it is the same failure with a different sha.
  2. WHETHER its transformer fits. ONIQ runs on ONE A5000 — 24 GiB, owner
     directive — and the 0.9.7 family includes 13B checkpoints. A 13B
     transformer at bf16 is ~26 GiB of weights alone: it would pair perfectly
     and never load. Measuring the bytes is what stops that being discovered
     on a rented card.

It CHOOSES NOTHING and repoints nothing. It reads the registry for $0 and
prints what each candidate would cost, so the repoint is made from measurement.

The token is used and never printed — the same discipline as hf_auth.
"""

from __future__ import annotations

import json

from validation.hf_discover import INFO_API, _get, _why
from validation.upscaler_discover import (
    LATENT_SPACE,
    RAW,
    VAE_PREFIX,
    _brief,
    _config_delta,
    _get_text,
    _licence_files,
)

# The upsampler the pin resolved and the owner accepted terms for. Written out
# and commented in ltx-upscaler.pin because the pairing failed;
# test_the_pinned_upsampler_matches_the_pin_file keeps these two in step.
UPSCALER_REPO = "Lightricks/ltxv-spatial-upscaler-0.9.7"
UPSCALER_REVISION = "c96c168c2bd8bbc82c9fe8259e5f89f8b2ea293f"

# What is baked today, so the report can show what would change.
CURRENT_REPO = "Lightricks/LTX-Video"
CURRENT_REVISION = "8984fa25007f376c1a299016d0957a37a2f797bb"

# The Dockerfile's own transformer guard. It is a 2B-CLASS guard: it exists so
# a 13B checkpoint cannot enter the image by accident. Deliberately choosing a
# larger one is an owner decision, not a workaround — so a candidate over this
# guard but under the card is reported as exactly that, rather than flattened
# into a refusal that hides the choice.
SIZE_GUARD_BYTES = 16 * 1024**3

# WHAT THE ENDPOINT MAY ACTUALLY RUN ON — owner directive 2026-08-30, recorded
# in validation/endpoint_gpus.py as WANTED and mirrored here with each card's
# capacity. Both are 48 GiB parts.
#
# THIS NUMBER WAS WRONG ONCE. It was first written as 24 GiB "A5000", carried
# over from the earlier retarget, which would have condemned every 13B
# candidate on a card that can hold one. The directive above superseded that,
# and test_the_cards_are_the_ones_the_endpoint_directive_names now ties this
# table to endpoint_gpus.WANTED so a future GPU change cannot leave it stale
# the same way. Each capacity is checkable against the live catalogue's
# memoryInGb, which is what gpu_matrix already matches on.
CARD_VRAM_GIB = {
    "NVIDIA RTX A6000": 48,
    "NVIDIA A40": 48,
}
# The SMALLEST approved card binds: a worker may be placed on either.
CARD_VRAM_BYTES = min(CARD_VRAM_GIB.values()) * 1024**3

# Every component the bake requires before a byte moves.
COMPONENTS = ("transformer", "vae", "text_encoder", "tokenizer", "scheduler")


def _paths(info) -> dict:
    return {(s.get("rfilename") or ""): (s.get("size") or 0)
            for s in info.get("siblings") or []}


def measure(repo: str, upstream_vae: dict, token, revision=None,
            get=_get, get_text=_get_text) -> dict:
    """One candidate checkpoint, against the gates the build applies."""
    row: dict = {"repo": repo, "asked_revision": revision}
    url = INFO_API.format(repo=repo)
    if revision:
        url = url.replace(f"{repo}?", f"{repo}/revision/{revision}?")
    try:
        info = get(url, token)
    except Exception as exc:
        row.update(verdict="UNREADABLE", detail=_why(exc))
        return row

    row["revision"] = info.get("sha")
    card = info.get("cardData") or {}
    row["licence"] = card.get("license")
    paths = _paths(info)
    row["licence_files"] = _licence_files(sorted(p for p in paths if "/" not in p))
    row["is_pipeline"] = "model_index.json" in paths
    row["missing_components"] = [
        c for c in COMPONENTS if not any(p.startswith(c + "/") for p in paths)]
    row["transformer_bytes"] = sum(
        size for p, size in paths.items()
        if p.startswith("transformer/") and p.endswith((".safetensors", ".bin")))
    row["pipeline_bytes"] = sum(
        size for p, size in paths.items()
        if p == "model_index.json" or any(p.startswith(c + "/") for c in COMPONENTS))
    row["within_size_guard"] = 0 < row["transformer_bytes"] <= SIZE_GUARD_BYTES
    row["fits_card"] = 0 < row["transformer_bytes"] < CARD_VRAM_BYTES

    # THE PAIRING, on exactly the Dockerfile's fields. Read at the resolved
    # sha, never at a branch, so what is reported is what a pin would bake.
    if not row["revision"] or "vae/config.json" not in paths:
        row["latent_delta"] = None
        row["pairs"] = "UNVERIFIED — no vae/config.json at this revision"
    else:
        try:
            theirs = json.loads(get_text(
                RAW.format(repo=repo, revision=row["revision"],
                           path=f"{VAE_PREFIX}/config.json"), token))
        except Exception as exc:
            row["latent_delta"] = None
            row["pairs"] = f"UNVERIFIED — {_why(exc)}"
        else:
            full = _config_delta(upstream_vae, theirs)
            gated = {k: v for k, v in full.items() if k in LATENT_SPACE}
            row["latent_delta"] = gated
            row["full_delta_fields"] = sorted(full)
            row["pairs"] = "PAIRS" if not gated else f"MISMATCH on {sorted(gated)}"

    if not row["is_pipeline"]:
        row["verdict"] = "NOT-A-PIPELINE"
    elif row["missing_components"]:
        row["verdict"] = f"INCOMPLETE — lacks {row['missing_components']}"
    elif not row["licence_files"]:
        row["verdict"] = "NO TERMS — the build refuses this"
    elif row["pairs"] != "PAIRS":
        row["verdict"] = f"DOES NOT PAIR — {row['pairs']}"
    elif not row["fits_card"]:
        row["verdict"] = (
            f"WILL NOT FIT — {row['transformer_bytes'] / 1024**3:.2f} GiB of "
            f"weights alone on a {CARD_VRAM_BYTES / 1024**3:.0f} GiB card")
    elif not row["within_size_guard"]:
        # NOT a refusal. It pairs and the card can hold it; what it exceeds is
        # a guard written to keep a 13B checkpoint from arriving by accident.
        # Choosing one on purpose is the owner's call, so this says so rather
        # than reading as "impossible".
        row["verdict"] = (
            f"PAIRS AND FITS, OVER TODAY'S GUARD — "
            f"{row['transformer_bytes'] / 1024**3:.2f} GiB transformer against "
            f"the {SIZE_GUARD_BYTES / 1024**3:.0f} GiB 2B-class guard. Raising "
            "that guard is an owner decision.")
    else:
        row["verdict"] = "CANDIDATE"
    return row


def candidates(token, get=_get) -> list:
    """Lightricks' own LTX repositories, plus the one baked today."""
    from validation.hf_discover import catalogue

    names = [m.get("id") or m.get("modelId") for m in catalogue(token, get)]
    names = [n for n in names if n and "upscaler" not in n and "upsampler" not in n.lower()]
    if CURRENT_REPO not in names:
        names.append(CURRENT_REPO)
    return sorted(set(names))


def report(token, get=_get, get_text=_get_text) -> tuple:
    if not token:
        print("BLOCKED: no credential, and an anonymous read cannot tell a "
              "gated repository from one that does not exist")
        return 2, []

    print(f"  PAIRING TARGET  {UPSCALER_REPO}@{UPSCALER_REVISION[:12]}")
    try:
        upstream_vae = json.loads(get_text(
            RAW.format(repo=UPSCALER_REPO, revision=UPSCALER_REVISION,
                       path=f"{VAE_PREFIX}/config.json"), token))
    except Exception as exc:
        print(f"  BLOCKED: cannot read the upsampler's own vae config — {_why(exc)}. "
              "Without it there is nothing to pair AGAINST, and a candidate "
              "reported as pairing would be reporting a comparison nobody made.")
        return 2, []

    rows = []
    # What is baked today, first, so the delta is legible.
    rows.append(measure(CURRENT_REPO, upstream_vae, token, CURRENT_REVISION,
                        get, get_text))
    for repo in candidates(token, get):
        rows.append(measure(repo, upstream_vae, token, None, get, get_text))

    for row in rows:
        print("")
        label = f"{row['repo']}"
        if row.get("asked_revision"):
            label += f"@{row['asked_revision'][:12]}  (BAKED TODAY)"
        print(f"  {label}")
        if row.get("verdict") == "UNREADABLE":
            print(f"    UNREADABLE — {row['detail']}")
            continue
        print(f"    revision      {row.get('revision')}")
        print(f"    licence       {row.get('licence')!r}  files {row.get('licence_files') or 'NONE'}")
        print(f"    components    {'complete' if not row['missing_components'] else 'MISSING ' + str(row['missing_components'])}")
        print(f"    transformer   {row['transformer_bytes'] / 1024**3:.2f} GiB "
              f"(2B-class guard {SIZE_GUARD_BYTES / 1024**3:.0f} GiB; "
              f"card {CARD_VRAM_BYTES / 1024**3:.0f} GiB — "
              f"{', '.join(sorted(CARD_VRAM_GIB))})")
        print(f"    pipeline      {row['pipeline_bytes'] / 1024**3:.2f} GiB to download")
        print(f"    PAIRS         {row['pairs']}")
        if row.get("latent_delta"):
            for key, (theirs_, ours) in sorted(row["latent_delta"].items()):
                print(f"      {key}")
                print(f"        upsampler's vae {_brief(theirs_)}")
                print(f"        this candidate  {_brief(ours)}")
        print(f"    VERDICT       {row['verdict']}")

    viable = [r for r in rows if r.get("verdict") == "CANDIDATE"
              or r.get("verdict", "").startswith("PAIRS AND FITS")]
    print("")
    if viable:
        print("  CANDIDATES THAT PAIR AND FIT — the repoint would be to one of:")
        for row in viable:
            note = "" if row["verdict"] == "CANDIDATE" else "  [needs the guard raised]"
            print(f"    {row['repo']} {row['revision']}  "
                  f"({row['transformer_bytes'] / 1024**3:.2f} GiB transformer){note}")
        print("")
        print("  A CANDIDATE IS NOT A DECISION. Repointing changes which weights "
              "EVERY production clip is generated by: output quality, VRAM "
              "headroom, image size and the measured per-clip economics all "
              "move together, and none of them is measured by this read.")
    else:
        print("  NOTHING PAIRS AND FITS. Every candidate either uses a "
              "different latent space from the pinned upsampler, ships no "
              "terms, or will not fit the card. Multi-scale cannot be enabled "
              "by repointing the checkpoint alone.")
    print("")
    return (0 if viable else 1), rows


def main(argv) -> int:
    import os

    from validation.hf_auth import TOKEN_VAR

    code, _ = report(os.environ.get(TOKEN_VAR))
    return code


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
