"""Does the Qwen bake's LICENCE GATE still pass against the live registry?

The gate lives in the Dockerfile and refuses to download a checkpoint
whose Hugging Face metadata does not declare an allowed licence. That is
the right place for it — the image cannot be built without it — but it
only ever runs inside a long `media` build, so a licence change or a
withdrawn repo is discovered late, after the owner has already spent the
build.

This module answers the same question from metadata alone, in seconds,
downloading no weights and spending nothing.

Its parameters are PARSED FROM THE Dockerfile, never restated here. A
second hand-maintained copy of CANDIDATES is exactly the defect that put
storygen.py in three lists and none of the images (2026-08-27); the
Dockerfile stays the single source of truth and this module checks the
world against it.
"""

from __future__ import annotations

import json
import re
import urllib.request

DOCKERFILE = "Dockerfile"
API = "https://huggingface.co/api/models/{repo}?blobs=true"


class GateParseError(Exception):
    """The Dockerfile no longer states the gate in a readable form."""


def _story_block(text: str) -> str:
    """The Qwen bake, isolated by its DEST — the LTX bake declares the
    same names, so matching on CANDIDATES alone would read the wrong
    block and silently check the wrong models."""
    start = text.find('DEST = "/app/models/story"')
    if start == -1:
        raise GateParseError('no block declares DEST = "/app/models/story"')
    head = text.rfind("CANDIDATES = [", 0, start)
    if head == -1:
        raise GateParseError("the story block declares no CANDIDATES")
    end = text.find("\nEOF", start)
    return text[head : end if end != -1 else len(text)]


def _strings(block: str, name: str, opener: str, closer: str) -> list:
    """The quoted strings inside one bracketed literal.

    Deliberately NOT the ast literal evaluator: the repo's shell-surface
    scan bans that call by substring across the shipped modules, and the
    literal form trips it — including, on the first attempt, this very
    docstring naming it. A gate checker is the last place to argue with
    a control that keeps dynamic evaluation out, and quoted strings are
    all this ever needs to read.
    """
    match = re.search(rf"^{name}\s*=\s*\{opener}(.*?)\{closer}", block, re.M | re.S)
    if not match:
        raise GateParseError(f"{name} is not declared in a parseable form")
    found = re.findall(r"[\"']([^\"']+)[\"']", match.group(1))
    if not found:
        raise GateParseError(f"{name} declares no entries")
    return found


def parse_gate(text: str) -> dict:
    block = _story_block(text)
    guard = re.search(r"^SIZE_GUARD_BYTES\s*=\s*(\d+)\s*\*\s*1024\*\*3", block, re.M)
    if not guard:
        raise GateParseError("SIZE_GUARD_BYTES is not declared in a parseable form")
    return {
        "candidates": _strings(block, "CANDIDATES", "[", "]"),
        "allowed_licences": {
            x.lower() for x in _strings(block, "ALLOWED_LICENCES", "{", "}")
        },
        "size_guard_bytes": int(guard.group(1)) * 1024**3,
        "needed": tuple(_strings(block, "NEEDED", "(", ")")),
    }


def _default_fetch(repo: str) -> dict:
    with urllib.request.urlopen(API.format(repo=repo), timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def licence_of(info: dict):
    """Card data first, then a license: tag — the Dockerfile's order."""
    card = info.get("cardData") or {}
    licence = card.get("license")
    if licence is None:
        licence = next(
            (t.split(":", 1)[1] for t in (info.get("tags") or []) if t.startswith("license:")),
            None,
        )
    return licence


def survey(repo: str, gate: dict, fetch=_default_fetch) -> dict:
    """One candidate's verdict. Never downloads weights."""
    try:
        info = fetch(repo)
    except Exception as exc:
        return {"repo": repo, "verdict": "UNREACHABLE", "detail": type(exc).__name__}

    licence = licence_of(info)
    siblings = {s.get("rfilename"): (s.get("size") or 0) for s in info.get("siblings") or []}
    weight_bytes = sum(v for k, v in siblings.items() if str(k).endswith(".safetensors"))
    missing = [n for n in gate["needed"] if n not in siblings]

    result = {
        "repo": repo,
        "licence": licence,
        "weight_bytes": weight_bytes,
        "missing": missing,
    }
    if str(licence).lower() not in gate["allowed_licences"]:
        result["verdict"] = "REFUSE"
        result["detail"] = f"licence {licence!r} not in {sorted(gate['allowed_licences'])}"
    elif missing:
        result["verdict"] = "SKIP"
        result["detail"] = f"lacks {missing}"
    elif not 0 < weight_bytes <= gate["size_guard_bytes"]:
        result["verdict"] = "SKIP"
        result["detail"] = f"{weight_bytes} weight bytes fail the size guard"
    else:
        result["verdict"] = "BAKE"
        result["detail"] = "licence and shape both satisfy the gate"
    return result


def report(text: str, fetch=_default_fetch) -> tuple[int, list]:
    """Exit code and per-candidate rows. 0 when the bake would resolve.

    A REFUSE is never softened into a skip: the whole point of the gate
    is that a wrong licence stops the build, so a wrong licence stops
    this check too, even if a later candidate would bake.
    """
    gate = parse_gate(text)
    rows = [survey(repo, gate, fetch) for repo in gate["candidates"]]
    for row in rows:
        print(
            f"{row['verdict']:11s} {row['repo']}"
            + (f"  licence={row.get('licence')!r}" if "licence" in row else "")
            + (
                f"  weights={row['weight_bytes'] / 1024**3:.2f}GiB"
                if row.get("weight_bytes")
                else ""
            )
            + f"  — {row['detail']}"
        )
    if any(row["verdict"] == "REFUSE" for row in rows):
        print("GATE: a candidate's licence is NOT allowed — the media build would refuse")
        return 1, rows
    if not any(row["verdict"] == "BAKE" for row in rows):
        print("GATE: no candidate would bake — the media build would fail")
        return 1, rows
    print("GATE: the licence gate passes and at least one candidate would bake")
    return 0, rows


if __name__ == "__main__":
    import sys

    with open(DOCKERFILE, encoding="utf-8") as fh:
        code, _ = report(fh.read())
    sys.exit(code)
