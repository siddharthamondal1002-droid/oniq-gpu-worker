"""How big is the `media` image, before anyone spends an hour building it?

Owner directive 2026-08-28: the image is to be built in CI and pushed to a
registry, then named by a template the API creates. That route has one
failure mode nobody can see until it has already cost a build — a GitHub
runner has a fixed disk, and this image bakes an LTX pipeline, an 8B causal
LLM and a torch+CUDA stack on top of each other.

So the size is surveyed from metadata first, for nothing, the same way the
licence gate is. Every parameter is PARSED FROM THE Dockerfile: which
candidates, which components, which file patterns each bake downloads, and
which size guard each bake applies. A second hand-maintained copy of those
would drift from the build it claims to predict, and a prediction that does
not track the build is worse than no prediction.

IT MUST PICK THE CANDIDATE THE BUILD WOULD PICK. Run 50 measured what
happens otherwise: the first LTX candidate answered an HTTP error, and this
module fell to the next one and reported 44.36 GiB — the 13B repository,
which the Dockerfile's own 16GiB transformer guard REFUSES. It sized a
model the build would never bake and declared the image did not fit. So the
guards are applied here too: a candidate missing a required file, missing a
component, or over its guard is SKIPPED exactly as the build skips it.

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
import urllib.error
import urllib.request

DOCKERFILE = "Dockerfile"
API = "https://huggingface.co/api/models/{repo}?blobs=true"

# MEASURED on this account's runners, 2026-08-28 (image-publish run 3):
#
#   /dev/root  76887154688 total  62096482304 used  14773895168 avail
#
# and `df / /mnt` reported /dev/root for BOTH — so there is NO separate
# /mnt temp disk here, whatever the documented default says. An earlier
# version of this file planned against a 65 GiB /mnt that does not exist
# on these runners, and the publish workflow moved docker's data root onto
# it for nothing.
#
# 13.76 GiB free before anything is reclaimed. The preinstalled toolchains
# the publish workflow deletes are worth roughly 25 GB more, which is the
# only reason a build of this size is even arguable.
RUNNER_TOTAL_BYTES = 76887154688
RUNNER_FREE_BYTES = 14773895168
# What the reclaim step frees, NOT MEASURED as a total — the workflow
# prints df after it so a real number replaces this the first time a build
# gets that far. Deliberately conservative: overstating it would let this
# module bless a build that then dies at 90%.
RECLAIMABLE_BYTES = 20 * 1024**3
RUNNER_USABLE_BYTES = RUNNER_FREE_BYTES + RECLAIMABLE_BYTES


class SizeParseError(Exception):
    """The Dockerfile no longer states a bake in a readable form."""


# ------------------------------------------------------------ parsing


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
    have understated the LTX bake by its VAE and text encoder.
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


def _patterns(block: str) -> list:
    """allow_patterns as the bake actually passes them to the downloader."""
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


def _needed_files(block: str) -> list:
    """Files whose absence makes the bake skip a candidate.

    Two forms again: a NEEDED tuple iterated over, and a direct
    `if "x" not in paths` guard. Both are refusals, so both are read.
    """
    needed = []
    if re.search(r"^NEEDED\s*=\s*\(", block, re.M):
        needed += _first_strings(block, "NEEDED", "(", ")")
    needed += re.findall(r"[\"']([^\"']+)[\"']\s+not in paths", block)
    return sorted(set(needed))


def _guard_prefix(block: str) -> str:
    """Which files the size guard weighs.

    The LTX bake guards the TRANSFORMER only — that is what distinguishes a
    2B from a 13B — while the story bake guards every weight file. Taking
    the wrong one turns a passing candidate into a failing one or the
    reverse, so it is read rather than assumed.
    """
    match = re.search(r"p\.startswith\([\"']([^\"']+)[\"']\)", block)
    return match.group(1) if match else ""


def parse_bakes(text: str) -> list:
    """Every model bake in the Dockerfile, in build order."""
    bakes = []
    for dest in ("/app/models/ltx", "/app/models/story"):
        block = _block(text, dest)
        guard = re.search(r"^SIZE_GUARD_BYTES\s*=\s*(\d+)\s*\*\s*1024\*\*3", block, re.M)
        if not guard:
            raise SizeParseError(f"{dest} declares no parseable SIZE_GUARD_BYTES")
        components = []
        if re.search(r"^COMPONENTS\s*=\s*\(", block, re.M):
            components = _first_strings(block, "COMPONENTS", "(", ")")
        bakes.append(
            {
                "dest": dest,
                "candidates": _first_strings(block, "CANDIDATES", "[", "]"),
                "patterns": _patterns(block),
                "needed_files": _needed_files(block),
                "components": components,
                "guard_prefix": _guard_prefix(block),
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


# ------------------------------------------------------------ surveying


def _default_fetch(repo: str) -> dict:
    with urllib.request.urlopen(API.format(repo=repo), timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _default_head(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as resp:
        return int(resp.headers.get("Content-Length") or 0)


def _why(exc: Exception) -> str:
    """The HTTP CODE, not the exception's class name.

    'HTTPError' does not distinguish 404 (the repository is gone) from 401
    (it is gated and needs a token) from 429 (we asked too fast), and those
    lead to completely different actions. queue_probe learned this on
    2026-08-28 and this module repeated the mistake the same day; run 50
    reported a bare HTTPError for the owner's chosen LTX model and the
    reason had to be guessed at.
    """
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP {exc.code}"
    return type(exc).__name__


def survey(repo: str, bake: dict, fetch=_default_fetch) -> dict:
    """One candidate's verdict, by the same rules the build applies.

    BAKE means the build would download this one. SKIP means the build
    would reject it and try the next — so its bytes must never be counted.
    UNREACHABLE means the answer is unknown, which is not the same as SKIP.
    """
    try:
        info = fetch(repo)
    except Exception as exc:
        code = getattr(exc, "code", None)
        if code in (401, 403):
            # The build STOPS here (Dockerfile, auth_refused), so the
            # projection must too. Reporting a fall-through image would
            # describe a build that will never happen.
            return {"repo": repo, "verdict": "AUTH-REFUSED", "bytes": None,
                    "detail": _why(exc) + " — the build refuses to fall through"}
        return {"repo": repo, "verdict": "UNREACHABLE", "bytes": None, "detail": _why(exc)}

    paths = {
        (s.get("rfilename") or ""): (s.get("size") or 0)
        for s in info.get("siblings") or []
    }

    missing = [n for n in bake["needed_files"] if n not in paths]
    if missing:
        return {"repo": repo, "verdict": "SKIP", "bytes": None,
                "detail": f"lacks {missing}"}

    absent = [
        c for c in bake["components"]
        if not any(p.startswith(c + "/") for p in paths)
    ]
    if absent:
        return {"repo": repo, "verdict": "SKIP", "bytes": None,
                "detail": f"lacks component(s) {absent}"}

    guarded = sum(
        size for p, size in paths.items()
        if p.startswith(bake["guard_prefix"]) and p.endswith(".safetensors")
    )
    if not 0 < guarded <= bake["size_guard_bytes"]:
        return {
            "repo": repo,
            "verdict": "SKIP",
            "bytes": None,
            "guarded_bytes": guarded,
            "detail": (
                f"{guarded / 1024**3:.2f} GiB under {bake['guard_prefix'] or '(all weights)'} "
                f"fails the {bake['size_guard_bytes'] / 1024**3:.0f} GiB guard"
            ),
        }

    # Only files matching allow_patterns are downloaded. The repos also
    # carry multi-GB single-file checkpoints the bake never fetches;
    # summing every sibling would overstate the image by more than the
    # runner's whole disk.
    total = 0
    files = 0
    for name, size in paths.items():
        if any(fnmatch.fnmatch(name, p) for p in bake["patterns"]):
            total += size
            files += 1
    return {"repo": repo, "verdict": "BAKE", "bytes": total, "files": files,
            "guarded_bytes": guarded, "detail": "passes every gate the build applies"}


def _gib(value) -> str:
    return "NOT MEASURED" if value is None else f"{value / 1024**3:.2f} GiB"


def report(text: str, base_image_bytes=None, fetch=_default_fetch, head=_default_head):
    """Projected media-image size. Exit 0 only when it plausibly fits.

    The projection is deliberately a FLOOR: the build holds the layer cache
    and the assembled image at once. Reporting a floor as if it were the
    answer is how a build gets started that cannot finish, so the wording
    says floor everywhere it is one.
    """
    rows = []
    total = 0
    unknown = False

    for bake in parse_bakes(text):
        chosen = None
        blocked = False
        for repo in bake["candidates"]:
            surveyed = survey(repo, bake, fetch)
            print(f"  {surveyed['verdict']:12s} {repo}: {surveyed['detail']}")
            if surveyed["verdict"] == "AUTH-REFUSED":
                blocked = True
                break
            if surveyed["verdict"] == "BAKE":
                chosen = surveyed
                break
        if blocked:
            print(
                f"BAKE {bake['dest']}: BLOCKED — the first reachable answer was a "
                "refusal to authenticate, and the build stops rather than "
                "substituting a checkpoint nobody chose"
            )
            unknown = True
            rows.append({"dest": bake["dest"], "bytes": None, "blocked": True})
            continue
        if chosen is None:
            print(f"BAKE {bake['dest']}: NOT MEASURED — no candidate would bake")
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
        print(f"BAKE piper: NOT MEASURED — {_why(exc)}")
        piper = None
        unknown = True
    else:
        print(f"BAKE piper: {_gib(piper)}")
        total += piper
    rows.append({"dest": "piper", "bytes": piper})

    print(
        f"MODEL BYTES: {_gib(total)}"
        + (" (INCOMPLETE — a bake is unmeasured)" if unknown else "")
    )
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
        f"RUNNER (measured): {_gib(RUNNER_FREE_BYTES)} free of "
        f"{_gib(RUNNER_TOTAL_BYTES)}, one filesystem, no separate /mnt"
    )
    print(f"AFTER RECLAIM (estimated): {_gib(RUNNER_USABLE_BYTES)}")
    if projected > RUNNER_USABLE_BYTES:
        print(
            "VERDICT: DOES NOT FIT — the image alone exceeds the reclaimed disk. "
            "A hosted runner cannot build this; it needs a bigger disk."
        )
        return 1, rows
    if projected * 2 > RUNNER_USABLE_BYTES:
        print(
            "VERDICT: TIGHT — the image fits but not twice over, and a build "
            "holds layers and the assembled image at once. Reclaim first and "
            "watch the df the workflow prints; treat a failure here as "
            "expected, not surprising."
        )
        return 0, rows
    print("VERDICT: FITS with room for both copies.")
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
