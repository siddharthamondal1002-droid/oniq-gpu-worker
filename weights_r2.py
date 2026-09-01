"""Model weights staged through R2, so a worker needs no HuggingFace token.

WHY THIS EXISTS — the defect it repairs, found 2026-09-01 before it cost
anything.

The 2026-09-01 owner decision (option B) moved the 17.74 GiB LTX text
encoder off the network volume and onto the worker's container disk,
fetched on a cold worker. That fixed placement — a volume permanently
pins an endpoint to one datacenter — and quietly created a credential
problem, because the checkpoint is GATED. The Dockerfile says so in its
own refusal:

    NO CREDENTIAL: /run/secrets/hf_token is absent or empty. The chosen
    checkpoint is gated, so there is nothing to do but stop.

At BUILD time the token arrives on a BuildKit tmpfs and a publish proof
asserts it does not survive into the image — correctly. So a worker has
no HF token, and an anonymous fetch of a gated repository is a 401
INSIDE a billed worker, after a ~40 GiB image pull, presenting as a new
mystery rather than as the missing credential it is.

OWNER DECISION 2026-09-01, option B over option A: stage the weights
through R2 rather than putting an HF token on the template. The worker
ALREADY holds R2 credentials — it writes every output through them — so
this adds no credential surface to a rented machine, which putting a
long-lived HuggingFace read token on every worker would.

THE REVISION IS IN THE KEY, and that is not decoration. The encoder and
the transformer share an embedding space; serving one checkpoint's
encoder beside another's transformer is a silent quality failure, not a
crash. A key that named only "text-encoder" would let a repoint read the
previous checkpoint's weights and produce plausible, wrong video.

THE DIGEST IS CHECKED, not the byte count alone. A truncated multipart
upload has the wrong length and would be caught by size; a corrupted one
need not be. The staging side records sha256 over the tarball and the
worker refuses on a mismatch rather than extracting it.

IT FAILS CLOSED. There is deliberately no fallback to HuggingFace when
R2 does not have the object: an unauthenticated fetch of a gated
repository is the exact failure this module exists to remove, and a
fallback would restore it at the worst possible moment.
"""

from __future__ import annotations

import hashlib
import json
import os
import tarfile

# 32 GiB. The job-input bound (contract.MAX_INPUT_BYTES, 16 MiB) is a
# DIFFERENT bound for a different thing and is deliberately left alone —
# a user-supplied input and a pinned model artefact have no business
# sharing a ceiling. This one sits above the 17.74 GiB encoder with room
# for a larger component later, and below anything that would fill a
# 201 GB container disk.
MAX_WEIGHTS_BYTES = 32 * 1024 ** 3

# Read in 8 MiB blocks: large enough that the digest is not syscall-bound,
# small enough that a 17.74 GiB file never lands in memory.
_CHUNK = 8 * 1024 * 1024


class WeightsUnavailable(Exception):
    """A named refusal. `code` is the whole diagnosis."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def object_key(spec: dict) -> str:
    """Where this exact checkpoint's component lives in the bucket.

    Derived, never typed: family, directory and REVISION all come from the
    registry entry, so the key cannot drift from the spec it serves.
    """
    for field in ("family", "directory", "revision"):
        if not spec.get(field):
            raise WeightsUnavailable(
                "spec-incomplete",
                f"the spec names no {field}, so no key can be derived from "
                "it. A guessed key would read some other checkpoint.",
            )
    return (
        f"weights/{spec['family']}/{spec['directory']}/"
        f"{spec['revision']}.tar"
    )


def manifest_key(spec: dict) -> str:
    return object_key(spec)[: -len(".tar")] + ".json"


def digest(path: str) -> str:
    """sha256 over a file, streamed."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(_CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def members_are_safe(tar: tarfile.TarFile) -> None:
    """Refuse a tarball that would write outside the destination.

    The archive is one this project staged, so this is not the usual
    hostile-input case — but an archive is a file, files get replaced, and
    a path traversal here writes into the image as uid 10001. Cheap to
    check, and the check is the difference between a trusted pipeline and
    a trusted-looking one.
    """
    for member in tar.getmembers():
        name = member.name
        if name.startswith("/") or os.path.isabs(name):
            raise WeightsUnavailable(
                "archive-unsafe", f"absolute path in archive: {name!r}"
            )
        if ".." in name.split("/"):
            raise WeightsUnavailable(
                "archive-unsafe", f"parent traversal in archive: {name!r}"
            )
        if member.issym() or member.islnk():
            raise WeightsUnavailable(
                "archive-unsafe", f"link member in archive: {name!r}"
            )


def stage(spec: dict, source_dir: str, tar_path: str, uploader) -> dict:
    """CI side: pack a downloaded component and put it in the bucket.

    `uploader(src_path, key) -> int` is storage.upload in production and a
    fake in tests; this module never builds an S3 client of its own, so it
    stays importable anywhere and carries no credential handling.
    """
    names = sorted(
        os.path.relpath(os.path.join(root, name), source_dir)
        for root, _, files in os.walk(source_dir)
        for name in files
    )
    if not names:
        raise WeightsUnavailable(
            "nothing-to-stage",
            f"{source_dir} holds no files. Uploading an empty archive would "
            "publish a checkpoint that unpacks to nothing.",
        )
    with tarfile.open(tar_path, "w") as tar:
        for name in names:
            tar.add(os.path.join(source_dir, name), arcname=name)

    sha = digest(tar_path)
    size = os.path.getsize(tar_path)
    if size > MAX_WEIGHTS_BYTES:
        raise WeightsUnavailable(
            "archive-too-large",
            f"{size} bytes exceeds the {MAX_WEIGHTS_BYTES} byte bound; the "
            "worker would refuse to read what this uploaded.",
        )
    record = {
        "repo": spec.get("repo"),
        "revision": spec["revision"],
        "files": names,
        "file_count": len(names),
        "tar_bytes": size,
        "tar_sha256": sha,
    }
    manifest_path = tar_path + ".json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, sort_keys=True)

    uploader(tar_path, object_key(spec))
    uploader(manifest_path, manifest_key(spec))
    return record


def fetch(spec: dict, dest_dir: str, work_dir: str, downloader) -> dict:
    """Worker side: pull the staged component and unpack it.

    `downloader(key, dest_path, max_bytes) -> int` is storage.download in
    production. The manifest is read FIRST so the digest to check against
    arrives before the bytes it describes — a digest fetched afterwards
    could be the one belonging to whatever actually downloaded.
    """
    manifest_path = os.path.join(work_dir, "manifest.json")
    try:
        downloader(manifest_key(spec), manifest_path, 1 * 1024 * 1024)
    except Exception as exc:
        raise WeightsUnavailable(
            "not-staged",
            f"no manifest at {manifest_key(spec)} ({type(exc).__name__}). "
            "This checkpoint has not been staged to R2; there is "
            "deliberately no fallback to HuggingFace, because an "
            "unauthenticated fetch of a gated repository is the failure "
            "this path exists to remove.",
        ) from exc
    with open(manifest_path, encoding="utf-8") as fh:
        record = json.load(fh)

    if record.get("revision") != spec["revision"]:
        raise WeightsUnavailable(
            "revision-mismatch",
            f"the staged manifest names revision {record.get('revision')!r}, "
            f"the registry pins {spec['revision']!r}. The encoder and the "
            "transformer share an embedding space; loading one checkpoint's "
            "encoder beside another's is a silent quality failure.",
        )

    tar_path = os.path.join(work_dir, "weights.tar")
    downloader(object_key(spec), tar_path, MAX_WEIGHTS_BYTES)

    got = digest(tar_path)
    if got != record.get("tar_sha256"):
        raise WeightsUnavailable(
            "digest-mismatch",
            f"downloaded archive is {got}, the manifest says "
            f"{record.get('tar_sha256')}. Refusing to unpack it.",
        )

    os.makedirs(dest_dir, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        members_are_safe(tar)
        tar.extractall(dest_dir, filter="data")
    os.remove(tar_path)
    return record
