"""How big is the `media` image, before anyone spends an hour building it?

Owner directive 2026-08-28: the image is to be built in CI and pushed to a
registry, then named by a template the API creates. That route has one
failure mode nobody can see until it has already cost a build — a GitHub
runner has a fixed disk, and this image bakes an LTX pipeline, an 8B causal
LLM and a torch+CUDA stack on top of each other.

So the size is surveyed from metadata first, for nothing, the same way the
licence gate is. Every parameter is PARSED FROM THE Dockerfile: which
candidates, which components, which file patterns each bake downloads. A
second hand-maintained copy of those patterns would drift from the build it
claims to predict, and a prediction that does not track the build is worse
than no prediction.

What this can and cannot answer:
- Model bytes are MEASURED, from the registry's own per-file sizes.
- The base stage (torch, CUDA libraries, dependencies) is NOT knowable from
  here. It is passed in, measured by the CI job that already builds that
  stage. Absent that figure the total is reported as a floor, never as a
  total — an unmeasured addend is never silently zero.
"""

from __future__ import annotations

import fnmatch
import json
import re
import urllib.request

from validation import qwen_assets

DOCKERFILE = "Dockerfile"
API = "https://huggingface.co/api/models/{repo}?blobs=true"

# What a GitHub-hosted ubuntu runner can actually hold. The pair matters:
# the build needs room for the layers AND the final image, and `docker
# build` keeps both. These are the documented defaults, not measurements of
# this account's runners, so the CI job re-reads the real figure with `df`
# and this is only the planning number.
RUNNER_ROOT_FREE_BYTES = 21 * 1024**3
RUNNER_MNT_FREE_BYTES = 65 * 1024**3


class SizeParseError(Exception):
    """The Dockerfile no longer states a bake in a readable form."""


def _block(text: str, dest: str) -> str:
    """One bake, isolated by its DEST — every bake declares CANDIDATES."""
    start = text.find(f'DEST = "{dest}"')
    if start == -1:
        raise SizeParseError(f"no block declares DEST = {dest!r}")
    head = text.rfind("CANDIDATES = [", 0, start)
    if head == -1:
        raise SizeParseError(f"the {dest} block declares no CANDIDATES")
    end = text.find("\nEOF", start)
    return text[head : end if end != -1 else len(text)]


def _balanced(text: str, open_at: int) -> str:
    """The contents of the bracket opening at open_at, nesting respected."""
    pairs = {"[": "]", "(": ")", "{": "}"}
    closer = pairs[text[open_at]]
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] in pairs:
            depth += 1
        elif text[i] == closer or text[i] in pairs.values():
            depth -= 1
            if depth == 0:
                return text[open_at + 1 : i]
    raise SizeParseError("an unbalanced bracket in the Dockerfile")


def _entries(body: str) -> list:
    """Split on commas at depth zero, so a tuple stays one entry."""
    out, depth, start = [], 0, 0
    for i, ch in enumerate(body):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            out.append(body[start:i])
            start = i + 1
    out.append(body[start:])
    return [e for e in out if e.strip()]


def _first_strings(block: str, name: str, opener: str, closer: str) -> list:
    """The FIRST quoted string of each ENTRY, not of each line.

    Two shapes have to read correctly and the difference is not cosmetic:
    CANDIDATES is a list of (repo, tag) PAIRS one per line — taking every
    quoted string would survey '#distilled' as a repository — while
    COMPONENTS is a flat tuple all on ONE line, where every entry is
    itself the string wanted. Reading first-per-LINE gets the first right
    and silently returns only 'transformer' for the second, which would
    have understated the LTX bake by its VAE and text encoder: several
    GiB missing from a projection whose whole job is to say whether the
    image fits.
    """
    match = re.search(rf"^{name}\s*=\s*\{opener}", block, re.M)
    if not match:
        raise SizeParseError(f"{name} is not declared in a parseable form")
    found = []
    for entry in _entries(_balanced(block, match.end() - 1)):
        quoted = re.findall(r"[\"']([^\"']+)[\"']", entry)
        if quoted:
            found.append(quoted[0])
    if not found:
        raise SizeParseError(f"{name} declares no entries")
    return found


def _patterns(block: str) -> list:
    """allow_patterns as the bake actually passes them to the downloader.

    Two forms appear: a plain list of globs, and the LTX form that builds
    per-component globs from COMPONENTS. Both are read from the source
    rather than restated, so a change to either tracks straight through.
    """
    at = block.find("allow_patterns=")
    if at == -1:
        at = block.find("allow_patterns =")
    if at == -1:
        raise SizeParseError("allow_patterns is not declared in a parseable form")
    # The whole argument expression, which may be a list PLUS a
    # comprehension over COMPONENTS. Reading to the first ] would take the
    # literal half and drop the generated half entirely.
    body = _entries(block[at:].split("=", 1)[1])[0]

    patterns = re.findall(r"[\"']([^\"']+)[\"']", body)
    comprehension = re.search(r"for\s+(\w+)\s+in\s+(\w+)", body)
    if comprehension:
        var, source = comprehension.groups()
        suffix = re.search(rf"{var}\s*\+\s*[\"']([^\"']*)[\"']", body)
        tail = suffix.group(1) if suffix else ""
        patterns = [p for p in patterns if p != tail]
        patterns += [c + tail for c in _first_strings(block, source, "(", ")")]
    if not patterns:
        raise SizeParseError("allow_patterns resolves to nothing")
    return patterns


def parse_bakes(text: str) -> list:
    """Every model bake in the Dockerfile, in build order."""
    bakes = []
    for dest in ("/app/models/ltx", "/app/models/story"):
        block = _block(text, dest)
        guard = re.search(r"^SIZE_GUARD_BYTES\s*=\s*(\d+)\s*\*\s*1024\*\*3", block, re.M)
        if not guard:
            raise SizeParseError(f"{dest} declares no parseable SIZE_GUARD_BYTES")
        bakes.append(
            {
                "dest": dest,
                "candidates": _first_strings(block, "CANDIDATES", "[", "]"),
                "patterns": _patterns(block),
                "size_guard_bytes": int(guard.group(1)) * 1024**3,
            }
        )
    return bakes


def parse_piper(text: str) -> str:
    """The voice tarball's URL, from the bake that fetches it."""
    match = re.search(r'URL\s*=\s*\(\s*"([^"]+)"\s*\n?\s*"([^"]*)"\s*\)', text)
    if match:
        return match.group(1) + match.group(2)
    match = re.search(r'^URL\s*=\s*"([^"]+)"', text, re.M)
    if not match:
        raise SizeParseError("the piper bake declares no parseable URL")
    return match.group(1)


def _default_fetch(repo: str) -> dict:
    with urllib.request.urlopen(API.format(repo=repo), timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _default_head(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as resp:
        return int(resp.headers.get("Content-Length") or 0)


def download_bytes(repo: str, patterns, fetch=_default_fetch) -> dict:
    """What this candidate would actually pull, by the registry's own sizes.

    Only files matching allow_patterns count. The repos carry multi-GB
    single-file checkpoints the bake deliberately does not fetch, so
    summing every sibling would overstate the image by more than the
    runner's whole disk.
    """
    try:
        info = fetch(repo)
    except Exception as exc:
        return {"repo": repo, "bytes": None, "detail": type(exc).__name__}
    total = 0
    files = 0
    for sibling in info.get("siblings") or []:
        name = sibling.get("rfilename") or ""
        if any(fnmatch.fnmatch(name, p) for p in patterns):
            total += sibling.get("size") or 0
            files += 1
    return {"repo": repo, "bytes": total, "files": files, "detail": "surveyed"}


def _gib(value) -> str:
    return "NOT MEASURED" if value is None else f"{value / 1024**3:.2f} GiB"


def report(text: str, base_image_bytes=None, fetch=_default_fetch, head=_default_head):
    """Projected media-image size. Exit 0 only when it plausibly fits.

    The projection is deliberately a FLOOR: model bytes land compressed in
    the registry and uncompressed on disk, and the build holds both at once.
    Reporting a floor as if it were the answer is how a build gets started
    that cannot finish, so the wording says floor everywhere it is one.
    """
    rows = []
    total = 0
    unknown = False

    for bake in parse_bakes(text):
        chosen = None
        for repo in bake["candidates"]:
            surveyed = download_bytes(repo, bake["patterns"], fetch)
            if surveyed["bytes"] is None:
                print(f"  UNREACHABLE {repo}: {surveyed['detail']}")
                continue
            if surveyed["bytes"] == 0:
                print(f"  EMPTY {repo}: no file matches the bake's patterns")
                continue
            chosen = surveyed
            break
        if chosen is None:
            print(f"BAKE {bake['dest']}: NOT MEASURED — no candidate could be surveyed")
            unknown = True
            rows.append({"dest": bake["dest"], "bytes": None})
            continue
        print(
            f"BAKE {bake['dest']}: {chosen['repo']} "
            f"{_gib(chosen['bytes'])} across {chosen['files']} files"
        )
        total += chosen["bytes"]
        rows.append({"dest": bake["dest"], "repo": chosen["repo"], "bytes": chosen["bytes"]})

    try:
        piper = head(parse_piper(text))
    except Exception as exc:
        print(f"BAKE piper: NOT MEASURED — {type(exc).__name__}")
        piper = None
        unknown = True
    else:
        print(f"BAKE piper: {_gib(piper)}")
        total += piper
    rows.append({"dest": "piper", "bytes": piper})

    print(f"MODEL BYTES: {_gib(total)}" + (" (INCOMPLETE — a bake is unmeasured)" if unknown else ""))
    print(f"BASE STAGE: {_gib(base_image_bytes)}")

    if base_image_bytes is None or unknown:
        print(
            "PROJECTED MEDIA IMAGE: NOT MEASURED — an addend is missing, and an "
            "unmeasured addend is not zero"
        )
        return 2, rows

    projected = base_image_bytes + total
    print(f"PROJECTED MEDIA IMAGE (FLOOR): {_gib(projected)}")
    print(
        f"RUNNER PLANNING BUDGET: root {_gib(RUNNER_ROOT_FREE_BYTES)}, "
        f"/mnt {_gib(RUNNER_MNT_FREE_BYTES)}"
    )
    # The build holds the layer cache and the assembled image at once, so
    # the disk has to carry the floor roughly twice over. One times the
    # floor fitting is not the question.
    if projected * 2 > RUNNER_MNT_FREE_BYTES:
        print(
            "VERDICT: DOES NOT FIT — twice the floor exceeds even /mnt. A hosted "
            "runner cannot build this image; the build needs a bigger disk."
        )
        return 1, rows
    if projected * 2 > RUNNER_ROOT_FREE_BYTES:
        print(
            "VERDICT: FITS ONLY ON /mnt — the docker data root must be moved off "
            "the root filesystem before the build, and free space reclaimed."
        )
        return 0, rows
    print("VERDICT: FITS as-is on the root filesystem.")
    return 0, rows


def main(argv) -> int:
    base = None
    if len(argv) > 1 and argv[1].strip():
        base = int(argv[1])
    with open(DOCKERFILE, encoding="utf-8") as fh:
        code, _ = report(fh.read(), base_image_bytes=base)
    return code


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
