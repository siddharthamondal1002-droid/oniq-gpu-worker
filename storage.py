"""R2 by reference — the worker's only storage surface.

The bucket is `oniq-gpu`, deliberately separate from `oniq-chat-media`:
a rented, internet-reachable GPU worker must never hold credentials to
users' chat media. Credentials arrive as exactly three env vars, set in
the RunPod endpoint's environment (never in this repo, never in GitHub
secrets). Until they exist every operation fails closed with
`storage-not-configured`, naming the missing VARIABLE NAMES and never
their values.
"""

from __future__ import annotations

import os

import contract

REQUIRED_VARS = ("R2_S3_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")
BUCKET = "oniq-gpu"


class StorageNotConfigured(Exception):
    code = "storage-not-configured"

    def __init__(self, missing):
        self.missing = tuple(missing)
        self.message = (
            "storage is not configured; missing env var(s): "
            + ", ".join(self.missing)
        )
        super().__init__(self.message)


class StorageError(Exception):
    """An R2 read or write that failed. `code` is r2-read-failed or
    r2-write-failed; the message never carries key material."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def missing_vars():
    return [name for name in REQUIRED_VARS if not os.environ.get(name)]


def require_configured() -> None:
    missing = missing_vars()
    if missing:
        raise StorageNotConfigured(missing)


def client():
    """Construct the R2 S3 client. Fails closed when unconfigured."""
    require_configured()
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_S3_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def download(key: str, dest_path: str, max_bytes: int = contract.MAX_INPUT_BYTES) -> int:
    """Fetch one object by reference, bounding its size BEFORE the body.

    Content-Length is checked via head_object first so an oversized object
    is refused without transferring it.
    """
    s3 = client()
    try:
        head = s3.head_object(Bucket=BUCKET, Key=key)
        size = int(head["ContentLength"])
    except Exception as exc:
        raise StorageError(
            "r2-read-failed", f"could not stat input object: {type(exc).__name__}"
        ) from exc
    if size > max_bytes:
        raise contract.ContractError(
            "input-too-large",
            f"input object is {size} bytes; the bound is {max_bytes}",
        )
    try:
        s3.download_file(BUCKET, key, dest_path)
    except Exception as exc:
        raise StorageError(
            "r2-read-failed", f"could not read input object: {type(exc).__name__}"
        ) from exc
    return size


def upload(src_path: str, key: str) -> int:
    s3 = client()
    try:
        size = os.path.getsize(src_path)
        s3.upload_file(src_path, BUCKET, key)
    except Exception as exc:
        raise StorageError(
            "r2-write-failed", f"could not write output object: {type(exc).__name__}"
        ) from exc
    return size
