"""Stage ONE cache-resident checkpoint's weights into R2. $0, no GPU.

WHY THIS EXISTS. The 2026-09-01 owner decision moved the LTX text encoder
off the network volume onto container disk, because attaching a volume
permanently pins an endpoint to one datacenter. That fixed placement and
exposed a credential split the project had deliberately built:

    CI       has the HuggingFace token (BuildKit tmpfs, build-time only)
             and NO R2 credentials
    a worker has R2 credentials (RunPod secret store)
             and NO HuggingFace token

Nothing had both, so nothing could move a GATED checkpoint from HF into
the bucket. The checkpoint really is gated — the Dockerfile refuses to
build without a token and says so — and a standing directive forbids
substituting an ungated model to route around it.

OWNER DIRECTIVE 2026-09-01: amend the R2 rule. The three R2 variables
become GitHub secrets on the `gpu-spend` ENVIRONMENT, not repository
secrets, so reaching them still costs a reviewer's approval click. The
rule they amend is recorded in README.md and template_env.py and was
stated as gate 6; all of them are updated together rather than left to
contradict this file.

WHAT IT DOES, once, per checkpoint:

    download text_encoder/* from HF with the CI token
    tar it, sha256 it, write a manifest beside it
    upload both to weights/<family>/<directory>/<revision>.{tar,json}

The worker then reads it with credentials it already has, and needs no
HuggingFace token at all. See weights_r2 for the fetch half and for why
there is deliberately no fallback.

IT WRITES TO A PRODUCTION BUCKET, so it is token-gated like every other
mutation here, and it stages exactly ONE model per run. There is no batch
shape — a loop over a registry is how the wrong bytes get published under
a name something else will trust.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

TOKEN = "STAGE-WEIGHTS"

# The staged tarball plus the files it was made from have to coexist on
# the runner. 17.74 GiB twice is ~35.5 GiB against 13.76 GiB free on a
# stock hosted runner, so the workflow reclaims the preinstalled
# toolchains first — the same measured ~32 GiB reclaim image-publish
# depends on. This module refuses rather than discovering it at 90%.
NEED_HEADROOM_GIB = 4.0


class Refused(Exception):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def check_token(token: str) -> None:
    if (token or "").strip() != TOKEN:
        raise Refused(
            "token-wrong",
            f"this publishes weights other runs will trust; it needs "
            f"{TOKEN!r} exactly. Nothing was written.",
        )


def free_gib(path: str) -> float:
    return shutil.disk_usage(path).free / (1024 ** 3)


def headroom_gib(download_gib: float) -> float:
    """Slack on top of the two copies — PROPORTIONAL, with a floor and a cap.

    It was a flat 4 GiB, which is right for the 17.74 GiB encoder and absurd
    for anything small: a 1 MB component demanded the same 4 GiB of slack as
    a 17.74 GiB one. The in-image test rig caught it, because that container
    has 3.50 GiB free in /tmp and this dev machine has more — the local suite
    passed on a difference in environment rather than on the code being
    right.

    A quarter of the payload, floored at 0.1 GiB so a tiny component still
    has room for filesystem overhead and tar block padding, and capped at 4
    so a future 100 GiB component does not demand 25 GiB of slack. At 17.74
    GiB the quarter is 4.435, so the cap applies and the production number
    is exactly what it was: 2 x 17.74 + 4 = 39.48 GiB.
    """
    return min(NEED_HEADROOM_GIB, max(0.1, download_gib * 0.25))


def check_room(spec: dict, path: str) -> float:
    """Both copies have to fit: the download AND the tar made from it."""
    need = spec["download_gib"] * 2 + headroom_gib(spec["download_gib"])
    free = free_gib(path)
    if free < need:
        raise Refused(
            "disk-insufficient",
            f"{free:.2f} GiB free, need {need:.2f} GiB "
            f"({spec['download_gib']:.2f} downloaded plus the same again "
            f"tarred, plus {NEED_HEADROOM_GIB:.0f} headroom). Reclaim the "
            "runner's toolchains before this step, as image-publish does.",
        )
    return free


def stage(model_id: str, token: str, workdir: str, *, modelroot, weights_r2,
          downloader, uploader) -> dict:
    """Fetch one checkpoint from HF and publish it to R2.

    Every collaborator is injected: this module chooses WHAT to stage and
    refuses when it should not, and holds no credential handling of its
    own. That is also what lets the whole path be tested without a network.
    """
    check_token(token)
    spec = modelroot.spec_for(model_id)
    if spec is None:
        raise Refused(
            "model-unknown",
            f"{model_id!r} is not a known model; known ids are "
            + ", ".join(modelroot.known_ids()),
        )
    if modelroot.storage_class(model_id) != "cache":
        raise Refused(
            "not-cache-resident",
            f"{model_id!r} is volume-resident. Staging it to R2 would "
            "publish weights nothing reads, and the volume path hydrates "
            "deliberately before dispatch instead.",
        )

    free = check_room(spec, workdir)
    source = os.path.join(workdir, "source")
    os.makedirs(source, exist_ok=True)

    downloader(
        spec["repo"],
        revision=spec["revision"],
        local_dir=source,
        allow_patterns=spec["allow"],
    )

    record = weights_r2.stage(
        spec, source, os.path.join(workdir, "weights.tar"), uploader
    )
    # The source tree is the larger half and is of no further use; drop it
    # so a second staging in the same run is not refused for space the
    # first one is merely still holding.
    shutil.rmtree(source, ignore_errors=True)
    return {
        "model_id": model_id,
        "repo": spec["repo"],
        "revision": spec["revision"],
        "free_gib_before": round(free, 2),
        "key": weights_r2.object_key(spec),
        "manifest_key": weights_r2.manifest_key(spec),
        **record,
    }


def report(model_id: str, token: str) -> int:  # pragma: no cover - needs net
    import modelroot
    import storage
    import weights_r2
    from huggingface_hub import snapshot_download

    hf_token = os.environ.get("HF_TOKEN") or None
    if not hf_token:
        print(
            "REFUSED no-hf-token: the chosen checkpoint is gated and this "
            "step has no HF_TOKEN. Nothing was written."
        )
        return 1

    def download(repo, *, revision, local_dir, allow_patterns):
        return snapshot_download(
            repo,
            revision=revision,
            local_dir=local_dir,
            allow_patterns=allow_patterns,
            token=hf_token,
        )

    workdir = tempfile.mkdtemp(prefix="oniq-stage-")
    try:
        result = stage(
            model_id, token, workdir,
            modelroot=modelroot, weights_r2=weights_r2,
            downloader=download, uploader=storage.upload,
        )
    except Refused as refusal:
        print(f"REFUSED {refusal.code}: {refusal.detail}")
        print("NOTHING WAS WRITTEN")
        return 1
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print(json.dumps(result, indent=2, sort_keys=True))
    print()
    print(f"STAGED {result['model_id']} at revision {result['revision']}")
    print(f"  {result['file_count']} file(s), {result['tar_bytes']} bytes tarred")
    print(f"  sha256 {result['tar_sha256']}")
    print(f"  -> {result['key']}")
    print()
    print("The worker now reads this with the R2 credentials it already "
          "has, and needs no HuggingFace token. $0 — no GPU, no worker.")
    return 0


def main(argv) -> int:  # pragma: no cover - entry point
    if len(argv) < 3:
        print(f"usage: weights_stage <model_id> <{TOKEN}>")
        return 2
    return report(argv[1], argv[2])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))
