"""Point the endpoint's EXISTING template at a new image and a bigger disk.

Owner directive 2026-08-29: the worker needs ~200 GB of container disk,
because the benchmark downloads a checkpoint at job time and the two Wan
candidates are 83.89 and 117.52 GiB — each larger than the whole 80 GB disk
before the baked image is counted.

WHY NOT A NEW TEMPLATE. template_attach creates one, and that path was tried
first: RunPod answers `500 create template: save template: Template name
must be unique`, measured 2026-08-29. Working around it by inventing a
second name would also have thrown away the thing that makes the existing
template usable — its `env`, which holds the RunPod secret REFERENCES for
R2. A fresh template starts with none, so the endpoint would have run a
worker that could not write its output until a second write repaired it.

Updating in place has neither problem: the id the endpoint already points at
does not move, the env is untouched, and there is no window in which the
endpoint references something half-configured.

WHAT IT REFUSES, and each refusal is here because the alternative is worse:

- a tag instead of a digest. `:latest` is a name that moves, and an endpoint
  pinned to a moving name can silently begin running an image nobody
  reviewed, on paid hardware.
- a template the endpoint does not actually reference. Retargeting some
  other template would look like success and change nothing.
- a template whose CURRENT image cannot be read. Without it the previous
  configuration cannot be restored, so the change is not safe to make.
- a disk smaller than the one already set. Shrinking is a different act with
  a different failure mode (a pull that no longer fits).
- a write that does not read back with BOTH new values.

Costs nothing: no worker starts. workersMin stays 0.
"""

from __future__ import annotations

import re

AUTHORIZED_TOKEN = "RETARGET-TEMPLATE"

# image@sha256:<64 hex>, with a REAL registry host in front.
#
# Docker's own rule decides what counts as a host: the first path component
# is a registry only if it contains a dot or a colon, or is `localhost`.
# Without that, `owner/repo@sha256:...` is a valid reference — to Docker Hub,
# which is not the registry this image was pushed to. The endpoint would pull
# something else entirely, or nothing, on paid hardware.
DIGEST_REF = re.compile(
    r"^(?P<host>localhost(?::\d+)?|[a-z0-9\-]+(?:\.[a-z0-9\-]+)+(?::\d+)?)"
    r"/[a-z0-9._\-/]+@sha256:[0-9a-f]{64}$"
)


class Refused(Exception):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def check_token(token: str) -> None:
    if (token or "").strip() != AUTHORIZED_TOKEN:
        raise Refused(
            "token-missing",
            f"this repoints a live endpoint's worker; it needs "
            f"{AUTHORIZED_TOKEN!r} exactly",
        )


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


def read_current(client, endpoint_id: str, template_id: str) -> dict:
    """The endpoint really points at this template, and it really reads."""
    if not (endpoint_id or "").strip():
        raise Refused("endpoint-missing", "no endpoint id was given")
    if not (template_id or "").strip():
        raise Refused("template-missing", "no template id was given")
    try:
        _, endpoint = client.get_endpoint(endpoint_id)
    except Exception as exc:
        raise Refused("endpoint-unreadable", f"{type(exc).__name__}") from exc
    if not isinstance(endpoint, dict) or endpoint.get("id") != endpoint_id:
        raise Refused(
            "endpoint-unreadable",
            "the endpoint read did not answer with the id that was asked for",
        )
    referenced = current_template_id(endpoint)
    if referenced != template_id:
        raise Refused(
            "template-not-referenced",
            f"the endpoint is on {referenced!r}, not {template_id!r} — "
            "retargeting a template nothing points at would look like "
            "success and change nothing",
        )
    try:
        _, template = client.get_template(template_id)
    except Exception as exc:
        raise Refused("template-unreadable", f"{type(exc).__name__}") from exc
    if not isinstance(template, dict) or not template.get("imageName"):
        raise Refused(
            "template-unreadable",
            "the template reports no imageName, so the configuration being "
            "replaced cannot be recorded and the change is not safe to make",
        )
    return template


def retarget(client, endpoint_id: str, template_id: str, image: str,
             disk_gb: int, token: str) -> dict:
    check_token(token)
    check_image(image)
    before = read_current(client, endpoint_id, template_id)
    was_disk = before.get("containerDiskInGb")
    if isinstance(was_disk, int) and disk_gb < was_disk:
        raise Refused(
            "disk-would-shrink",
            f"{was_disk} GB -> {disk_gb} GB. Shrinking is a different act "
            "with a different failure mode (a pull that no longer fits), and "
            "it is not what this was authorized for",
        )

    client.retarget_template(template_id, image, disk_gb)

    # VERIFY BY RE-READING. The PATCH's own response is the write claiming it
    # worked; the template read is the template saying so.
    try:
        _, after = client.get_template(template_id)
    except Exception as exc:
        raise Refused(
            "retarget-unverified",
            f"the template could not be re-read after the write: "
            f"{type(exc).__name__}",
        ) from exc
    if after.get("imageName") != image:
        raise Refused(
            "retarget-unconfirmed",
            f"the template still reports {after.get('imageName')!r}",
        )
    if after.get("containerDiskInGb") != disk_gb:
        raise Refused(
            "retarget-unconfirmed",
            f"the disk reads back as {after.get('containerDiskInGb')!r}, "
            f"not {disk_gb}",
        )
    env = after.get("env")
    return {
        "template_id": template_id,
        "image": image,
        "container_disk_gb": disk_gb,
        "replaced": {
            "image": before.get("imageName"),
            "container_disk_gb": was_disk,
        },
        "env_keys": sorted(env.keys()) if isinstance(env, dict) else [],
    }


def report(client, endpoint_id: str, template_id: str, image: str,
           disk_gb: int, token: str) -> tuple:
    try:
        result = retarget(client, endpoint_id, template_id, image, disk_gb, token)
    except Refused as refusal:
        print(f"REFUSED {refusal.code}: {refusal.detail}")
        print("NOTHING WAS WRITTEN")
        return 1, {"refused": refusal.code}

    was = result["replaced"]
    print(f"RETARGETED template {result['template_id']}")
    print(f"  image  {was['image']}")
    print(f"      -> {result['image']}")
    print(f"  disk   {was['container_disk_gb']} GB -> {result['container_disk_gb']} GB")
    print("  confirmed by re-reading the template, not by the write's own echo")
    # NAMES ONLY. Whether the storage variables survived the write is the
    # question that decides whether the next job can upload anything at all.
    print(f"  env keys still present: {', '.join(result['env_keys']) or '(none)'}")
    print(
        "RESTORE, if needed: retarget the same template back to the image and "
        "disk printed above."
    )
    print(
        "NOMINAL ONLY: the disk figure is what was REQUESTED. The worker "
        "reports the disk it actually has before any download, and refuses "
        "on that rather than on this number."
    )
    return 0, result


def main(argv) -> int:
    import runpod_client

    endpoint_id = argv[1] if len(argv) > 1 else ""
    template_id = argv[2] if len(argv) > 2 else ""
    image = argv[3] if len(argv) > 3 else ""
    disk_gb = int(argv[4]) if len(argv) > 4 and argv[4] else 0
    token = argv[5] if len(argv) > 5 else ""
    code, _ = report(runpod_client, endpoint_id, template_id, image, disk_gb, token)
    return code


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
