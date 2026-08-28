"""Can the build reach the EXACT model the owner chose, and only that one?

Owner directive 2026-08-28 (option 1): keep the distilled LTX 2B. Do not
substitute Lightricks/LTX-Video. Do not edit the candidate list to route
around authentication. If authentication cannot be provided safely, STOP.

So this asks one question, for nothing, before any build is started: with
the credential this repository holds, does the intended checkpoint answer,
and is what answers actually the model that was chosen?

THE TOKEN IS NEVER PRINTED, and never returned. This module reads it from
the environment, puts it in one Authorization header, and reports only its
PRESENCE by variable NAME — the same discipline storage.py applies to the
R2 credentials, and the reason spend_run reports env var names rather than
values. A test asserts the value cannot reach stdout.

The intended repository is PARSED FROM THE Dockerfile — it is the first
candidate, the one the owner's model decision put first — so this cannot
drift into checking a different model than the build would download.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from validation import image_size

DOCKERFILE = "Dockerfile"
API = "https://huggingface.co/api/models/{repo}"

# NAME only. Nothing in this module ever holds the value beyond the single
# header it builds, and nothing ever prints it.
TOKEN_VAR = "HF_TOKEN"


class Blocked(Exception):
    """The intended model cannot be reached. Never softened into a skip."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def intended(text: str) -> dict:
    """The FIRST LTX candidate and the gates it must satisfy.

    First, not any: the candidate list is ordered by the owner's model
    decision, and this directive forbids falling past the head of it.
    """
    bake = image_size.parse_bakes(text)[0]
    return {
        "repo": bake["candidates"][0],
        "components": bake["components"],
        "guard_prefix": bake["guard_prefix"],
        "size_guard_bytes": bake["size_guard_bytes"],
        "needed_files": bake["needed_files"],
    }


def _fetch(repo: str, token, blobs: bool = True) -> dict:
    url = API.format(repo=repo) + ("?blobs=true" if blobs else "")
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _status(exc) -> int | None:
    return getattr(exc, "code", None)


def probe(text: str, token=None, fetch=_fetch) -> dict:
    """Reachability, identity and shape of the intended checkpoint.

    Raises Blocked rather than returning a soft verdict: this runs to
    decide whether a build may start, and 'probably' is not an answer a
    build gate may act on.
    """
    want = intended(text)
    repo = want["repo"]

    if not token:
        raise Blocked(
            "credential-absent",
            f"{TOKEN_VAR} is not set in this environment. The intended "
            f"checkpoint {repo} is gated (HTTP 401 measured 2026-08-28), so "
            "no build can proceed and no substitute will be chosen.",
        )

    try:
        info = fetch(repo, token)
    except Exception as exc:
        code = _status(exc)
        if code in (401, 403):
            raise Blocked(
                "credential-rejected",
                f"{repo} answered HTTP {code} WITH {TOKEN_VAR} presented. The "
                "token is set but does not grant access — the model's terms "
                "may not have been accepted on the account that issued it.",
            ) from None
        raise Blocked(
            "registry-unreachable",
            f"{repo} could not be read ({'HTTP ' + str(code) if code else type(exc).__name__}).",
        ) from None

    # IDENTITY. The API echoes the repository it answered for; a redirect or
    # a rename must not quietly become a different model.
    answered = info.get("id") or info.get("modelId")
    if answered != repo:
        raise Blocked(
            "identity-mismatch",
            f"asked for {repo}, the registry answered for {answered!r}",
        )

    revision = info.get("sha")
    if not revision:
        raise Blocked("revision-unknown", f"{repo} reports no commit sha to pin to")

    card = info.get("cardData") or {}
    licence = card.get("license")
    if licence is None:
        licence = next(
            (t.split(":", 1)[1] for t in (info.get("tags") or []) if t.startswith("license:")),
            None,
        )

    paths = {
        (s.get("rfilename") or ""): (s.get("size") or 0)
        for s in info.get("siblings") or []
    }
    missing = [n for n in want["needed_files"] if n not in paths]
    if missing:
        raise Blocked("not-a-snapshot", f"{repo} lacks {missing}")

    absent = [
        c for c in want["components"] if not any(p.startswith(c + "/") for p in paths)
    ]
    if absent:
        raise Blocked("incomplete-pipeline", f"{repo} lacks component(s) {absent}")

    # SHAPE. This is what makes it the 2B and not a 13B wearing the name.
    guarded = sum(
        size for p, size in paths.items()
        if p.startswith(want["guard_prefix"]) and p.endswith(".safetensors")
    )
    if not 0 < guarded <= want["size_guard_bytes"]:
        raise Blocked(
            "guard-failed",
            f"{guarded} bytes under {want['guard_prefix']} fails the "
            f"{want['size_guard_bytes']} byte guard — this is not the 2B class",
        )

    download_bytes = sum(
        size for p, size in paths.items()
        if p == "model_index.json" or any(
            p.startswith(c + "/") for c in want["components"]
        )
    )
    return {
        "repo": repo,
        "revision": revision,
        "licence": licence,
        "guarded_bytes": guarded,
        "download_bytes": download_bytes,
        "files": len(paths),
    }


def report(text: str, token=None, fetch=_fetch) -> tuple:
    print(f"CREDENTIAL: {TOKEN_VAR} is {'PRESENT' if token else 'ABSENT'} (name only)")
    try:
        found = probe(text, token, fetch)
    except Blocked as blocked:
        print(f"BLOCKED {blocked.code}: {blocked.detail}")
        print("NO SUBSTITUTE WILL BE CHOSEN — the directive forbids it")
        return 1, {"blocked": blocked.code}

    print(f"MODEL: {found['repo']}")
    print(f"REVISION: {found['revision']}")
    print(f"LICENCE: {found['licence']!r}")
    print(f"GUARDED BYTES: {found['guarded_bytes']} (inside the 2B-class guard)")
    print(f"DOWNLOAD BYTES: {found['download_bytes']} across {found['files']} listed files")
    print("REACHABLE: the intended checkpoint answers and is the model that was chosen")
    return 0, found


def main(argv) -> int:
    with open(DOCKERFILE, encoding="utf-8") as fh:
        code, _ = report(fh.read(), os.environ.get(TOKEN_VAR))
    return code


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
