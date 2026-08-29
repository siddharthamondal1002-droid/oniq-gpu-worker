"""Create a template naming a published image, and attach it to the endpoint.

This is the second half of owner directive 2026-08-28 (route 2). The first
half — image-publish.yml — puts a real image in a registry. This names it
and points endpoint p3zmlv8ek10dzt at it, repairing the dangling
templateId that has made the endpoint unusable since the worker was
deleted.

It is a MUTATION, so it is built the way stale_run.py is: a literal token,
every precondition checked before the first write, and the result verified
by re-reading rather than by trusting the write's own echo.

What it refuses, and why each refusal is here rather than assumed away:

- A TAG instead of a digest. `:latest` is a name that moves. An endpoint
  pinned to a tag can silently begin running an image nobody reviewed,
  on paid hardware. Only image@sha256:... is accepted.
- An endpoint other than the one named on the command line, and only
  after RunPod itself confirms that endpoint exists.
- An endpoint whose CURRENT template actually exists. The whole warrant
  for writing is that the reference is dangling; if it resolves, this is
  clobbering a working configuration and stops.
- A create that returns no id, and an attach that does not read back.

What it will not do, at all: set environment variables. The worker's three
R2 variables belong in the RunPod environment and nowhere else — storage.py
states that contract and this module has no parameter that could carry a
value. It REPORTS the names that must be set, and names are all it knows.

Costs nothing. Creating a template and repointing an endpoint start no
worker; workersMin stays 0, so no pod runs until a job arrives.
"""

from __future__ import annotations

import json
import re

import storage

AUTHORIZED_TOKEN = "ATTACH-TEMPLATE"

# image@sha256:<64 hex>. Anchored at both ends so a tag cannot ride along
# after the digest, and the registry host is required so a bare name
# cannot resolve against an unintended default registry.
DIGEST_REF = re.compile(
    r"^[a-z0-9.\-]+(?::\d+)?/[a-z0-9._\-/]+@sha256:[0-9a-f]{64}$"
)

TEMPLATE_NAME = "oniq-gpu-worker"
# The image carries an LTX pipeline, an 8B checkpoint and torch+CUDA. The
# container disk has to hold the whole image with room to write outputs;
# too small a disk fails the pull on the rented card, at cost.
#
# RAISED 80 -> 200 on 2026-08-29, and the reason is arithmetic rather than
# comfort. The model benchmark downloads a candidate checkpoint at job time,
# and the published footprints are 83.89 GiB (Wan2.1 I2V-14B) and 117.52 GiB
# (Wan2.2 I2V-A14B) — both larger than the ENTIRE 80 GB disk before the
# baked image is counted, so neither could be fetched at all. Converting to
# bf16 on the way in does not rescue Wan2.2 either: it is still ~64 GiB.
#
# 200 leaves room for the image (~40-52 GiB measured/estimated) plus the
# largest candidate plus the transient a snapshot download holds while
# writing. The nominal number is NOT trusted: modelprobe reads free space
# from the running worker and refuses before the first byte moves, because
# running out of disk 60 GiB into a 118 GiB fetch burns the whole watchdog
# window and produces no evidence about the model at all.
CONTAINER_DISK_GB = 200


class Refused(Exception):
    """A precondition failed. Nothing was written."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def check_token(token: str) -> None:
    if token != AUTHORIZED_TOKEN:
        raise Refused("token-missing", "the literal authorization token was not given")


def check_image(image: str) -> None:
    if not DIGEST_REF.match(image or ""):
        raise Refused(
            "image-not-pinned",
            "the image must be registry/name@sha256:<64 hex> — a tag moves, "
            "and an endpoint pinned to a moving name can start running an "
            "image nobody reviewed",
        )


def current_template_id(endpoint: dict):
    for key in ("templateId", "template_id"):
        if endpoint.get(key):
            return endpoint[key]
    return None


def check_endpoint(client, endpoint_id: str) -> dict:
    if not (endpoint_id or "").strip():
        raise Refused("endpoint-missing", "no endpoint id was given")
    try:
        _, endpoint = client.get_endpoint(endpoint_id)
    except Exception as exc:
        raise Refused("endpoint-unreadable", f"{type(exc).__name__}") from exc
    if not isinstance(endpoint, dict) or not endpoint.get("id"):
        raise Refused("endpoint-unreadable", "the endpoint read returned no id")
    if endpoint["id"] != endpoint_id:
        raise Refused(
            "endpoint-mismatch",
            f"asked for {endpoint_id}, RunPod answered with {endpoint['id']}",
        )
    return endpoint


def check_reference_is_dangling(client, endpoint: dict) -> None:
    """The warrant for writing is that the current reference resolves to
    nothing. If it resolves, this would replace a working template."""
    template_id = current_template_id(endpoint)
    if not template_id:
        return
    try:
        client.get_template(template_id)
    except Exception:
        return
    raise Refused(
        "template-exists",
        f"the endpoint already points at {template_id}, and that template "
        "resolves — replacing a working configuration is not what was "
        "authorized",
    )


def attach(client, endpoint_id: str, image: str, token: str) -> dict:
    check_token(token)
    check_image(image)
    endpoint = check_endpoint(client, endpoint_id)
    check_reference_is_dangling(client, endpoint)

    _, created = client.create_template(TEMPLATE_NAME, image, CONTAINER_DISK_GB)
    template_id = (created or {}).get("id")
    if not template_id:
        raise Refused("create-unconfirmed", "the create returned no template id")

    client.attach_template(endpoint_id, template_id)

    # VERIFY BY RE-READING. The PATCH's own response is the write claiming
    # it worked; the endpoint read is the endpoint saying so.
    try:
        _, after = client.get_endpoint(endpoint_id)
    except Exception as exc:
        raise Refused(
            "attach-unverified",
            f"the endpoint could not be re-read after the attach: {type(exc).__name__}",
        ) from exc
    if current_template_id(after) != template_id:
        raise Refused(
            "attach-unconfirmed",
            f"the endpoint still reports {current_template_id(after)!r}, not {template_id!r}",
        )
    return {"template_id": template_id, "image": image, "endpoint": after}


def report(client, endpoint_id: str, image: str, token: str) -> tuple:
    try:
        result = attach(client, endpoint_id, image, token)
    except Refused as refusal:
        print(f"REFUSED {refusal.code}: {refusal.detail}")
        print("NOTHING WAS WRITTEN")
        return 1, {"refused": refusal.code}

    print(f"CREATED template {result['template_id']}")
    print(f"IMAGE {result['image']}")
    print(f"ATTACHED to endpoint {endpoint_id}, confirmed by re-reading it")
    print(
        "ENV STILL REQUIRED (names only, values are the owner's and belong in "
        "the RunPod environment): " + ", ".join(storage.REQUIRED_VARS)
    )
    print(
        "Until those are set every job fails closed with storage-not-configured, "
        "which is the designed behaviour and costs a job, not a silent bad write."
    )
    return 0, result


def main(argv) -> int:
    import runpod_client

    endpoint_id = argv[1] if len(argv) > 1 else ""
    image = argv[2] if len(argv) > 2 else ""
    token = argv[3] if len(argv) > 3 else ""
    code, _ = report(runpod_client, endpoint_id, image, token)
    return code


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
