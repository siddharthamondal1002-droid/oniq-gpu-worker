"""Which LTX checkpoints ACTUALLY exist, measured with a working credential?

Run 7 settled something three earlier runs had wrong. Unauthenticated, the
registry answered 401 for Lightricks/LTX-Video-0.9.8-2B-distilled and that
was read here as "gated". It is not: Hugging Face answers 401 to anonymous
callers for gated AND non-existent repositories alike, precisely so that
existence cannot be probed without credentials. Presented with a valid
token the same path answers 404 — the repository does not exist.

So the Dockerfile has been naming a model that is not there, and the
fall-through that owner directive 2026-08-28 removed was the only reason
any image ever built at all.

This module does NOT pick a replacement. Choosing which checkpoint ONIQ
ships is the owner's decision — the same decision recorded in the
Dockerfile as "distilled 2B first per the owner's model decision" — and a
name invented here to make a build succeed is exactly the substitution the
directive forbids. It LISTS what the publisher actually offers, with each
candidate measured against the gates the build would apply, so the owner
can name the real one.

The token is used and never printed, the same discipline as hf_auth.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

from validation import image_size

DOCKERFILE = "Dockerfile"
LIST_API = "https://huggingface.co/api/models"
INFO_API = "https://huggingface.co/api/models/{repo}?blobs=true"

# Who publishes the model family the owner chose. Searching by author
# rather than by a guessed name is the point: a guessed name is how this
# went wrong in the first place.
AUTHOR = "Lightricks"
SEARCH = "LTX"


def _get(url: str, token, timeout: int = 60):
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _why(exc) -> str:
    code = getattr(exc, "code", None)
    return f"HTTP {code}" if code else type(exc).__name__


def catalogue(token, get=_get) -> list:
    """Every model this author publishes matching the family name."""
    query = urllib.parse.urlencode({"author": AUTHOR, "search": SEARCH, "limit": 100})
    return get(f"{LIST_API}?{query}", token)


def measure(repo: str, gates: dict, token, get=_get) -> dict:
    """One real repository, against the gates the build would apply."""
    row = {"repo": repo}
    try:
        info = get(INFO_API.format(repo=repo), token)
    except Exception as exc:
        row.update(verdict="UNREADABLE", detail=_why(exc))
        return row

    card = info.get("cardData") or {}
    row["licence"] = card.get("license") or next(
        (t.split(":", 1)[1] for t in (info.get("tags") or []) if t.startswith("license:")),
        None,
    )
    row["revision"] = info.get("sha")

    paths = {
        (s.get("rfilename") or ""): (s.get("size") or 0)
        for s in info.get("siblings") or []
    }
    row["is_diffusers"] = "model_index.json" in paths
    row["components"] = [
        c for c in gates["components"] if any(p.startswith(c + "/") for p in paths)
    ]
    row["missing_components"] = [
        c for c in gates["components"] if c not in row["components"]
    ]
    row["transformer_bytes"] = sum(
        size for p, size in paths.items()
        if p.startswith(gates["guard_prefix"]) and p.endswith(".safetensors")
    )
    row["download_bytes"] = sum(
        size for p, size in paths.items()
        if p == "model_index.json" or any(
            p.startswith(c + "/") for c in gates["components"]
        )
    )

    if not row["is_diffusers"]:
        row.update(verdict="NOT-A-PIPELINE", detail="no model_index.json")
    elif row["missing_components"]:
        row.update(verdict="INCOMPLETE",
                   detail=f"lacks {row['missing_components']}")
    elif not 0 < row["transformer_bytes"] <= gates["size_guard_bytes"]:
        row.update(
            verdict="OVER-GUARD",
            detail=(
                f"{row['transformer_bytes'] / 1024**3:.2f} GiB transformer fails "
                f"the {gates['size_guard_bytes'] / 1024**3:.0f} GiB 2B-class guard"
            ),
        )
    else:
        row.update(verdict="ELIGIBLE",
                   detail="passes every gate the build applies")
    return row


def report(text: str, token, get=_get) -> tuple:
    if not token:
        print("BLOCKED: no credential, and an anonymous listing cannot "
              "distinguish a gated repository from one that does not exist")
        return 2, []

    bake = image_size.parse_bakes(text)[0]
    gates = {
        "components": bake["components"],
        "guard_prefix": bake["guard_prefix"],
        "size_guard_bytes": bake["size_guard_bytes"],
    }
    named = bake["candidates"][0]
    print(f"THE DOCKERFILE NAMES: {named}")

    try:
        found = catalogue(token, get)
    except Exception as exc:
        print(f"BLOCKED: the catalogue could not be listed ({_why(exc)})")
        return 2, []

    ids = [m.get("id") or m.get("modelId") for m in found]
    ids = [i for i in ids if i]
    print(f"{AUTHOR} publishes {len(ids)} models matching {SEARCH!r}:")
    if named in ids:
        print(f"  the named model IS in the catalogue — the 404 was something else")
    else:
        print(f"  the named model is NOT in the catalogue — it does not exist")

    rows = []
    for repo in sorted(ids):
        row = measure(repo, gates, token, get)
        rows.append(row)
        size = row.get("transformer_bytes") or 0
        print(
            f"  {row['verdict']:14s} {repo}"
            + (f"  licence={row.get('licence')!r}" if row.get("licence") else "")
            + (f"  transformer={size / 1024**3:.2f}GiB" if size else "")
            + f"  — {row['detail']}"
        )

    eligible = [r for r in rows if r["verdict"] == "ELIGIBLE"]
    print("")
    if not eligible:
        print("NO ELIGIBLE CHECKPOINT: nothing this author publishes passes the "
              "gates as written. That is a decision for the owner, not a reason "
              "to relax a guard.")
        return 1, rows
    print(f"ELIGIBLE ({len(eligible)}), for the owner to choose between:")
    for row in eligible:
        print(
            f"  {row['repo']}  revision={row['revision']}  "
            f"licence={row.get('licence')!r}  "
            f"download={row['download_bytes'] / 1024**3:.2f}GiB"
        )
    print("")
    print("NOT CHOOSING ONE. Which checkpoint ONIQ ships is an owner decision, "
          "and a name picked here to make a build succeed is the substitution "
          "the directive forbids.")
    return 0, rows


def main(argv) -> int:
    import os

    from validation.hf_auth import TOKEN_VAR

    with open(DOCKERFILE, encoding="utf-8") as fh:
        code, _ = report(fh.read(), os.environ.get(TOKEN_VAR))
    return code


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
