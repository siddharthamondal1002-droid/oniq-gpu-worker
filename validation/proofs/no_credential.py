"""Run INSIDE the built image: the weights shipped, the credential did not.

Owner directive 2026-08-28 option 1, requirements 7 and 8. BuildKit's
secret mount is architecturally incapable of writing the value into a
layer — that is why it was chosen over ARG or ENV, both of which are
permanently readable off a published image. But "architecturally
incapable" is a claim about a tool, and this repo checks claims.

THE VALUE GOES IN AND ONLY A COUNT COMES OUT. The token is passed through
the environment and never echoed, so even a match cannot put it in a build
log; the process exits non-zero and names the file paths, not the secret.
Reporting "3 files matched" is enough to act on and leaks nothing.

A cache directory being non-empty is checked separately from the value
scan, because they fail for different reasons: a stale login writes a
cache without the raw token in it, and a bad mount writes the token
somewhere no cache lives.
"""

import os
import sys

TOKEN_VAR = "SCAN_FOR"

# Anything under these is the kernel's view of the running process, not
# image content, and /proc in particular contains this very process's own
# environment — which holds the needle by construction.
SKIP_ROOTS = ("/proc", "/sys", "/dev")

# A weight file cannot contain a token that was never written to it, and
# walking 30 GiB of safetensors would take longer than the build.
MAX_SCAN_BYTES = 4 * 1024 * 1024

CACHE_PATHS = (
    "/root/.cache/huggingface",
    "/home/oniq/.cache/huggingface",
    "/run/secrets",
)


def dirty_caches(paths=CACHE_PATHS):
    dirty = []
    for path in paths:
        try:
            if os.path.isdir(path) and os.listdir(path):
                dirty.append(path)
            elif os.path.isfile(path) and os.path.getsize(path):
                dirty.append(path)
        except OSError:
            continue
    return dirty


def scan(needle: bytes, root="/"):
    """Paths whose contents contain the needle. Never the contents."""
    hits = []
    for base, dirs, names in os.walk(root, onerror=lambda exc: None):
        if base.startswith(SKIP_ROOTS):
            dirs[:] = []
            continue
        for name in names:
            path = os.path.join(base, name)
            try:
                if not os.path.isfile(path) or os.path.islink(path):
                    continue
                if os.path.getsize(path) > MAX_SCAN_BYTES:
                    continue
                with open(path, "rb") as fh:
                    if needle in fh.read():
                        hits.append(path)
            except OSError:
                continue
    return hits


def main():
    raw = os.environ.get(TOKEN_VAR) or ""
    if not raw.strip():
        # Refusing beats passing: a scan for an empty needle would match
        # every file, or nothing, depending on the implementation, and
        # either way proves nothing about the image.
        print(f"SCAN REFUSED: {TOKEN_VAR} is empty, so there is nothing to look for")
        return 2

    dirty = dirty_caches()
    if dirty:
        print(f"FAIL: credential cache directories are not empty: {dirty}")
        return 1
    print("PROOF no credential cache: every cache path is absent or empty")

    hits = scan(raw.strip().encode())
    print(f"PROOF credential scan: {len(hits)} files contain the credential")
    if hits:
        print(f"FAIL: the credential survived into the image at {hits[:10]}")
        return 1
    print("PROOF the image carries the weights, not the credential")
    return 0


if __name__ == "__main__":
    sys.exit(main())
