"""Run INSIDE the built image: is this the thing that was supposed to ship?

Owner directive 2026-08-28 option 1 asks for six proofs before publishing —
the image starts, runs as uid 10001, and carries the LTX 2B, Qwen3-8B,
piper and the licence records. This is the part that has to run inside the
container, because a claim about an image made from outside it is a claim
about a tag.

It asserts the model by its LITERAL NAME. Every other guard in this repo
checks a shape — a size, a class, a component list — and a shape can be
satisfied by the wrong model. After run 51, where a gated checkpoint would
have been silently replaced by a different one that passed every shape
check, the name itself is checked too.

Deliberately dependency-free and stdlib-only: it runs in the shipped image,
which carries the worker's dependencies and must not need test tooling.
"""

import json
import os
import sys

# The exact identity the owner chose. Not derived from the Dockerfile here
# on purpose: this is the independent side of the check, and a value read
# from the same file that produced the image would agree with it by
# construction rather than by fact.
EXPECT_LTX = "Lightricks/LTX-Video-0.9.8-2B-distilled#distilled"
EXPECT_STORY_PREFIX = "Qwen/Qwen3-8B"
LTX_GUARD_BYTES = 16 * 1024**3
STORY_GUARD_BYTES = 20 * 1024**3

ROOT = "/app/models"

# Relative to ROOT, so the checks can be exercised against a fixture tree
# rather than only against a 40 GiB image that takes an hour to build.
REQUIRED_FILES = (
    "ltx/model_index.json",
    "story/config.json",
    "piper/en-us-ryan-high.onnx",
    "piper/en-us-ryan-high.onnx.json",
)


class ProofFailed(Exception):
    pass


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read().strip()


def weight_bytes(root):
    total = 0
    for base, _, names in os.walk(root):
        for name in names:
            if name.endswith(".safetensors"):
                total += os.path.getsize(os.path.join(base, name))
    return total


def check(report=print, root=ROOT):
    def at(*parts):
        return os.path.join(root, *parts)

    ltx = read(at("MODEL_ID"))
    if ltx != EXPECT_LTX:
        raise ProofFailed(f"MODEL_ID is {ltx!r}, expected {EXPECT_LTX!r}")
    report(f"PROOF ltx model: {ltx}")

    revision = read(at("LTX_REVISION"))
    if len(revision) < 7:
        raise ProofFailed(f"LTX_REVISION is {revision!r}, not a commit sha")
    report(f"PROOF ltx revision: {revision}")
    report("PROOF ltx licence: " + read(at("LTX_LICENCE")))

    story = read(at("STORY_MODEL_ID"))
    if not story.startswith(EXPECT_STORY_PREFIX):
        raise ProofFailed(f"STORY_MODEL_ID is {story!r}, expected {EXPECT_STORY_PREFIX}*")
    report(f"PROOF story model: {story}")

    for relative in REQUIRED_FILES:
        path = at(relative)
        if not os.path.exists(path):
            raise ProofFailed(f"missing baked asset {path}")
    report("PROOF piper voice: onnx and config both present")

    with open(at("ltx", "model_index.json"), encoding="utf-8") as fh:
        index = json.load(fh)
    if "LTX" not in str(index.get("_class_name") or ""):
        raise ProofFailed(f"ltx pipeline class is {index.get('_class_name')!r}")
    report(f"PROOF ltx pipeline class: {index['_class_name']}")

    with open(at("story", "config.json"), encoding="utf-8") as fh:
        config = json.load(fh)
    if "qwen3" not in str(config.get("model_type") or "").lower():
        raise ProofFailed(f"story model_type is {config.get('model_type')!r}")
    report(f"PROOF story model_type: {config['model_type']}")

    # The size guards again, against what actually landed. Metadata said
    # this before the download; bytes on disk say it after.
    ltx_bytes = weight_bytes(at("ltx", "transformer"))
    if not 0 < ltx_bytes <= LTX_GUARD_BYTES:
        raise ProofFailed(f"ltx transformer is {ltx_bytes} bytes")
    report(f"PROOF ltx transformer bytes: {ltx_bytes} (inside the 2B-class guard)")

    story_bytes = weight_bytes(at("story"))
    if not 0 < story_bytes <= STORY_GUARD_BYTES:
        raise ProofFailed(f"story weights are {story_bytes} bytes")
    report(f"PROOF story weight bytes: {story_bytes} (inside the 8B-class guard)")

    return {
        "ltx": ltx,
        "ltx_revision": revision,
        "story": story,
        "ltx_bytes": ltx_bytes,
        "story_bytes": story_bytes,
    }


def main():
    try:
        check()
    except (ProofFailed, OSError, ValueError) as exc:
        print(f"PROOF FAILED: {exc}")
        return 1
    print("PROOFS PASSED: the image carries the models that were chosen")
    return 0


if __name__ == "__main__":
    sys.exit(main())
