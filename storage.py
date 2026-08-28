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
import re

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


# RunPod substitutes {{ RUNPOD_SECRET_<name> }} in a template's env at
# start-up. When the substitution does NOT happen - the secret was renamed,
# deleted, or serverless resolves references differently from pods - the
# container receives the reference TEXT. That text is a non-empty string,
# so a plain truthiness check calls it configured, the job is admitted, the
# GPU is paid for, and the upload fails at the very end against an endpoint
# URL of "{{ RUNPOD_SECRET_... }}". Owner directive 2026-08-28 put these
# variables in by reference, which is what makes this reachable.
UNRESOLVED = re.compile(r"\{\{.*\}\}", re.DOTALL)


def is_configured(value) -> bool:
    """A value the worker can actually use: present, and not a reference
    that nobody expanded."""
    if not value or not value.strip():
        return False
    return not UNRESOLVED.search(value)


def missing_vars():
    return [name for name in REQUIRED_VARS if not is_configured(os.environ.get(name))]


def require_configured() -> None:
    missing = missing_vars()
    if missing:
        raise StorageNotConfigured(missing)


def client():
    """Construct the R2 S3 client. Fails closed when unconfigured.

    Misconfiguration names itself: the first live job (2026-08-25,
    ea308ecd…-u1) died as an anonymous unexpected-exception/ValueError
    because boto3 refuses a scheme-less endpoint URL at construction and
    construction was unwrapped. The message names the VARIABLE, never
    its value.
    """
    require_configured()
    from urllib.parse import urlparse

    endpoint = os.environ["R2_S3_ENDPOINT"]
    parsed = urlparse(endpoint)
    if parsed.scheme != "https" or not parsed.netloc:
        raise StorageError(
            "r2-misconfigured",
            "R2_S3_ENDPOINT is not a valid https:// URL — set it to the "
            "full S3 API endpoint from the Cloudflare R2 dashboard, "
            "including the https:// scheme",
        )
    import boto3

    try:
        return boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            region_name="auto",
        )
    except Exception as exc:
        raise StorageError(
            "r2-misconfigured",
            f"could not construct the R2 client: {type(exc).__name__}",
        ) from exc


def _error_name(exc) -> str:
    """Class name, plus the S3 error code when the server sent one.

    "ClientError" alone left run #24 (2026-08-25) unable to say whether
    the input object was absent (404/NoSuchKey) or unreadable
    (AccessDenied). The server's error code is metadata, not a value —
    safe to print, and it makes the stop self-diagnosing.
    """
    code = None
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = (response.get("Error") or {}).get("Code")
    name = type(exc).__name__
    return f"{name}({code})" if code else name


def _probe_prefix(s3, key: str) -> str:
    """On a stat 404, list what actually exists near the requested key.

    Keys are owner-chosen object names, not secret material; twenty of
    them turn "not found" into "here is what IS there" — the difference
    between a fix and another blind retry (runs #24-#28, 2026-08-25/26).
    A failing list is itself the answer: the bucket is unreachable from
    this endpoint URL or unlisted by this token.
    """
    prefix = key.rsplit("/", 1)[0] + "/" if "/" in key else ""
    try:
        page = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, MaxKeys=20)
        keys = [obj.get("Key", "?") for obj in page.get("Contents", [])]
        if not keys and prefix:
            page = s3.list_objects_v2(Bucket=BUCKET, MaxKeys=20)
            keys = [obj.get("Key", "?") for obj in page.get("Contents", [])]
            return f"; nothing under '{prefix}', bucket root holds: {keys!r}"
        return f"; the bucket holds under '{prefix}': {keys!r}"
    except Exception as probe_exc:
        return f"; and listing the bucket failed too: {_error_name(probe_exc)}"


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
        detail = _error_name(exc)
        if "404" in detail or "NoSuchKey" in detail:
            detail += _probe_prefix(s3, key)
        raise StorageError(
            "r2-read-failed", f"could not stat input object: {detail}"
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
            "r2-read-failed", f"could not read input object: {_error_name(exc)}"
        ) from exc
    return size


def upload(src_path: str, key: str) -> int:
    s3 = client()
    try:
        size = os.path.getsize(src_path)
        s3.upload_file(src_path, BUCKET, key)
    except Exception as exc:
        raise StorageError(
            "r2-write-failed", f"could not write output object: {_error_name(exc)}"
        ) from exc
    return size
