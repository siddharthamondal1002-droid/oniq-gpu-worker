"""Put an experimental checkpoint on the persistent volume, once.

OWNER DIRECTIVE, 2026-08-30: model weights are DATA. This is the module
that makes that true — after it runs, changing frames, steps, revision or
offload mode is a configuration edit, not a 25 GiB image rebuild and a
two-hour cold pull.

Properties the directive requires, and how each is met:

  idempotent          READY returns immediately, downloading nothing
  lock protected      O_EXCL lock file; a second worker waits or refuses
  resumable           huggingface_hub resumes partial files by default
  manifest verified   every file's size recorded, digest over the manifest
  atomic completion   marker written to a temp name, then os.replace'd
  revision recorded   the pinned sha, re-checked by modelroot.resolve
  disk recorded       bytes actually consumed, measured after the fetch
  no GPU              nothing here imports torch
  never destructive   a valid existing model is never deleted

The marker is written LAST, so its presence means complete. Anything on
disk without it is an interrupted download, and modelroot refuses it as
MODEL_NOT_HYDRATED rather than loading half a checkpoint on a rented card.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time

import modelroot

MISSING = "MISSING"
DOWNLOADING = "DOWNLOADING"
READY = "READY"
CORRUPT = "CORRUPT"
DISK_INSUFFICIENT = "DISK_INSUFFICIENT"

LOCK_NAME = ".oniq-hydrating"
# A lock older than this is assumed to belong to a worker that died
# mid-download. Generous on purpose: 32 GiB took 40 s on the measured run,
# but a slow region could take far longer, and stealing a live lock would
# have two workers writing the same files.
LOCK_STALE_SECONDS = 3600

# Headroom over the declared download so a fetch cannot fill the volume
# and leave it unusable for the next model.
DISK_HEADROOM_GIB = 4.0


class HydrationRefused(Exception):
    def __init__(self, state: str, detail: str):
        super().__init__(f"{state}: {detail}")
        self.state = state
        self.detail = detail


def _free_gib(path: str) -> float:
    os.makedirs(path, exist_ok=True)
    return shutil.disk_usage(path).free / (1024 ** 3)


def manifest(path: str) -> dict:
    """Every regular file under `path`, relative name -> size.

    The marker itself and the lock are excluded: a manifest that recorded
    its own container could never verify, since the file's size is not
    known until after it is written.
    """
    files = {}
    for root, _dirs, names in os.walk(path):
        for name in names:
            if name in (modelroot.READY_MARKER, LOCK_NAME):
                continue
            full = os.path.join(root, name)
            if os.path.islink(full) or not os.path.isfile(full):
                continue
            files[os.path.relpath(full, path)] = os.path.getsize(full)
    return files


def manifest_digest(files: dict) -> str:
    """A digest over the manifest — names and sizes, in sorted order.

    Not a digest of 32 GiB of content: hashing the weights would take
    longer than fetching them and would be re-done on every verification.
    huggingface_hub already verifies each file's own hash on download; this
    fixes the SET of files, so a later addition, deletion or truncation
    changes the digest.
    """
    h = hashlib.sha256()
    for name in sorted(files):
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(str(files[name]).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def state(model_id: str) -> str:
    """Where this model stands, without changing anything."""
    if not modelroot.volume_mounted():
        return MISSING
    path = modelroot.model_dir(model_id)
    if not os.path.isdir(path):
        return MISSING
    if _lock_is_live(path):
        return DOWNLOADING
    marker = modelroot.read_marker(path)
    if marker is None:
        # Files with no marker: an interrupted fetch. Not CORRUPT — nothing
        # promised them — but not usable either.
        return MISSING if not manifest(path) else DOWNLOADING
    spec = modelroot.spec_for(model_id)
    if marker.get("revision") != spec["revision"]:
        return CORRUPT
    if modelroot.verify(path, marker):
        return CORRUPT
    return READY


def _lock_path(path: str) -> str:
    return os.path.join(path, LOCK_NAME)


def _lock_is_live(path: str) -> bool:
    try:
        age = time.time() - os.path.getmtime(_lock_path(path))
    except OSError:
        return False
    return age < LOCK_STALE_SECONDS


def _take_lock(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    lock = _lock_path(path)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        if _lock_is_live(path):
            raise HydrationRefused(
                DOWNLOADING,
                f"another worker holds {lock} and it is less than "
                f"{LOCK_STALE_SECONDS}s old. Two writers on one checkpoint "
                "produce a directory neither of them can verify.",
            )
        # Stale: the holder died. Reclaim rather than block forever.
        os.unlink(lock)
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w") as fh:
        json.dump({"pid": os.getpid(), "started": time.time()}, fh)


def _release_lock(path: str) -> None:
    try:
        os.unlink(_lock_path(path))
    except OSError:
        pass


def hydrate(model_id: str, downloader=None, token=None) -> dict:
    """Fetch one experimental model onto the volume. Idempotent."""
    spec = modelroot.spec_for(model_id)
    if spec is None:
        raise HydrationRefused(
            CORRUPT,
            f"{model_id!r} is not a known volume-resident model; known ids "
            "are " + ", ".join(modelroot.known_ids()),
        )
    # ONLY VOLUME MODELS NEED A VOLUME. A cache-resident model hydrates onto
    # the worker's own container disk, which is always present — requiring a
    # mount for it would refuse the very fetch that makes a cold worker
    # usable under the 2026-09-01 no-volume decision.
    if (modelroot.storage_class(model_id) == "volume"
            and not modelroot.volume_mounted()):
        raise HydrationRefused(
            MISSING,
            f"{modelroot.VOLUME_ROOT} is not mounted; there is nowhere to "
            "hydrate to. Attach the network volume to the endpoint first.",
        )

    path = modelroot.model_dir(model_id)
    current = state(model_id)
    if current == READY:
        marker = modelroot.read_marker(path) or {}
        return {
            "model_id": model_id,
            "state": READY,
            "path": path,
            "already_present": True,
            "revision": marker.get("revision"),
            "bytes": marker.get("bytes"),
        }

    need = spec["download_gib"] + DISK_HEADROOM_GIB
    # THE ROOT THIS MODEL ACTUALLY LANDS ON. Checking the volume's free
    # space before a fetch that goes to container disk would measure the
    # wrong filesystem — and on an endpoint with no volume it would measure
    # a path that does not exist.
    free = _free_gib(modelroot.root_for(model_id))
    if free < need:
        raise HydrationRefused(
            DISK_INSUFFICIENT,
            f"{free:.2f} GiB free, need {need:.2f} GiB "
            f"({spec['download_gib']:.2f} for the checkpoint plus "
            f"{DISK_HEADROOM_GIB:.0f} headroom). A fetch that fills the "
            "volume leaves it unusable for the next model too.",
        )

    _take_lock(path)
    started = time.monotonic()
    try:
        if downloader is None:  # pragma: no cover - network path
            from huggingface_hub import snapshot_download

            downloader = snapshot_download
        downloader(
            spec["repo"],
            revision=spec["revision"],
            local_dir=path,
            allow_patterns=spec["allow"],
            token=token or os.environ.get("HF_TOKEN") or None,
        )
        elapsed_ms = int((time.monotonic() - started) * 1000)

        files = manifest(path)
        if not files:
            raise HydrationRefused(
                CORRUPT,
                "the download reported success and left no files; refusing "
                "to write a READY marker over an empty directory",
            )
        total = sum(files.values())
        marker = {
            "model_id": model_id,
            "repo": spec["repo"],
            "revision": spec["revision"],
            "licence": spec["licence"],
            "files": files,
            "file_count": len(files),
            "bytes": total,
            "manifest_sha256": manifest_digest(files),
            "download_ms": elapsed_ms,
            "hydrated_at": time.time(),
        }
        # Atomic: a reader either sees no marker or sees a whole one.
        tmp = os.path.join(path, modelroot.READY_MARKER + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(marker, fh, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, os.path.join(path, modelroot.READY_MARKER))
    finally:
        _release_lock(path)

    return {
        "model_id": model_id,
        "state": READY,
        "path": path,
        "already_present": False,
        "revision": spec["revision"],
        "licence": spec["licence"],
        "bytes": marker["bytes"],
        "file_count": marker["file_count"],
        "manifest_sha256": marker["manifest_sha256"],
        "download_ms": marker["download_ms"],
        "disk_free_after_gib": round(_free_gib(modelroot.oniq_root()), 2),
    }
