"""Set the storage variables on the template, BY REFERENCE.

Owner directive 2026-08-28: the three R2 variables go onto template
`oniq-gpu-worker` so the worker can write its output. The values are not
credentials and never become credentials here - they are RunPod secret
references of the documented form `{{ RUNPOD_SECRET_<name> }}`, which
RunPod expands inside the container at start-up.

THE RULE HOLDS UNCHANGED FOR THIS MODULE: no R2 value ever enters this
repository, a log line, or this process. References are all it writes and
all it can write.

The WIDER rule was amended 2026-09-01 by owner directive, and this file is
not what changed. R2 credentials now also exist as GitHub secrets on the
`gpu-spend` ENVIRONMENT, reachable by exactly one job —
validation/weights_stage.py, which stages a GATED checkpoint into the
bucket because CI holds the HuggingFace token and a worker does not. The
amendment is recorded in README.md and in gate 6 of gpu-validation.yml. It
loosens nothing here: this module still handles no value, and the worker
still reads its credentials from RunPod's secret store, never from a job
input.

What this refuses to do matters more than what it does:

- it writes ONE field, `env`, so a stale read cannot rewrite the image;
- it refuses if the template already carries env keys it did not put
  there, rather than clobbering someone else's configuration;
- it refuses if the image drifts across the write;
- it verifies by re-reading over BOTH the REST and GraphQL views, because
  one path is an opinion;
- it needs a literal authorization token, like every other mutation here.

A note on the failure this cannot see: if RunPod does not expand the
reference - wrong secret name, or serverless resolving differently from
pods - the container receives the reference TEXT. `storage.is_configured`
rejects an unexpanded `{{ ... }}`, so that lands as a clean
storage-not-configured refusal before any GPU time is billed, rather than
as an upload failure at the end of a job that has already been paid for.
"""

from __future__ import annotations

import re

import storage
from validation.template_probe import _render, _rest_env_names

AUTHORIZED_TOKEN = "SET-TEMPLATE-ENV"

# The documented syntax, https://docs.runpod.io/pods/templates/secrets
REFERENCE = "{{{{ RUNPOD_SECRET_{name} }}}}"

# RunPod secret names as the console accepts them. Deliberately strict: a
# name with a brace or a space in it would produce a reference that never
# expands, and an unexpandable reference is the failure mode above.
SECRET_NAME = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class Refused(Exception):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def check_token(token: str) -> None:
    if (token or "").strip() != AUTHORIZED_TOKEN:
        raise Refused(
            "token-missing",
            f"this writes to a live template; it needs {AUTHORIZED_TOKEN!r} exactly",
        )


def parse_secret_names(raw: str) -> dict:
    """Map each required variable to the RunPod secret backing it.

    Given a comma-separated list, positionally against
    storage.REQUIRED_VARS. Given nothing, refuse - guessing a secret name
    produces a reference that silently fails to expand, which is worse
    than not writing at all.
    """
    names = [part.strip() for part in (raw or "").split(",") if part.strip()]
    if not names:
        raise Refused(
            "secret-names-missing",
            "name the RunPod secrets to reference; a guessed name expands to "
            "nothing and looks like success",
        )
    if len(names) != len(storage.REQUIRED_VARS):
        raise Refused(
            "secret-names-count",
            f"expected {len(storage.REQUIRED_VARS)} names for "
            f"{', '.join(storage.REQUIRED_VARS)}, got {len(names)}",
        )
    for name in names:
        if not SECRET_NAME.match(name):
            raise Refused(
                "secret-name-invalid",
                f"{name!r} is not a usable RunPod secret name",
            )
    return {
        var: REFERENCE.format(name=name)
        for var, name in zip(storage.REQUIRED_VARS, names)
    }


def read_template(client, template_id: str) -> dict:
    try:
        _, doc = client.get_template(template_id)
    except Exception as exc:
        raise Refused(
            "template-unreadable",
            f"the template could not be read: {type(exc).__name__}",
        ) from exc
    if not isinstance(doc, dict) or not doc.get("id"):
        raise Refused("template-missing", f"no template {template_id!r} came back")
    return doc


def check_no_foreign_env(names) -> None:
    """Refuse rather than clobber. `env` is sent whole, so anything on the
    template that this did not put there would be erased by the write."""
    if names is None:
        raise Refused(
            "env-unreadable",
            "the current env is UNKNOWN, and a whole-field write over an "
            "unknown value is how configuration disappears",
        )
    foreign = sorted(n for n in names if n not in storage.REQUIRED_VARS)
    if foreign:
        raise Refused(
            "env-not-empty",
            f"the template already carries {foreign}, which this write would "
            "erase; set these by hand or remove them first",
        )


def apply(client, template_id: str, secret_names: str, token: str) -> dict:
    check_token(token)
    references = parse_secret_names(secret_names)

    before = read_template(client, template_id)
    image_before = before.get("imageName")
    check_no_foreign_env(_rest_env_names(client, template_id))

    client.set_template_env(template_id, references)

    # VERIFY BY RE-READING, on both paths. The PATCH's own response is the
    # write claiming it worked.
    after = read_template(client, template_id)
    if after.get("imageName") != image_before:
        raise Refused(
            "image-drift",
            "the image changed across the write; the endpoint would run "
            "something these proofs never passed on",
        )

    rest = _rest_env_names(client, template_id)
    graph = client.template_env_names_graphql(template_id)
    for label, seen in (("REST", rest), ("GraphQL", graph)):
        if seen is None:
            raise Refused(
                "verify-unreadable",
                f"the {label} view could not be re-read after the write",
            )
        absent = [v for v in storage.REQUIRED_VARS if v not in seen]
        if absent:
            raise Refused(
                "verify-incomplete",
                f"the {label} view still lacks {', '.join(absent)} after the write",
            )
    return {
        "template_id": template_id,
        "image": image_before,
        "rest": rest,
        "graphql": graph,
    }


def report(client, template_id: str, secret_names: str, token: str) -> tuple:
    try:
        result = apply(client, template_id, secret_names, token)
    except Refused as refusal:
        print(f"REFUSED {refusal.code}: {refusal.detail}")
        print("NOTHING WAS WRITTEN")
        return 1, {"refused": refusal.code}

    print(f"SET on template {result['template_id']}: "
          f"{', '.join(storage.REQUIRED_VARS)} (names only, never values)")
    print(f"IMAGE unchanged across the write: {result['image']}")
    print(f"VERIFIED via REST   : {_render(result['rest'])}")
    print(f"VERIFIED via GraphQL: {_render(result['graphql'])}")
    print(
        "The values are RunPod secret references. If RunPod does not expand "
        "one, the worker refuses with storage-not-configured BEFORE any GPU "
        "time is billed - it does not run and fail at the upload."
    )
    return 0, result


def main(argv) -> int:
    import runpod_client

    template_id = argv[1] if len(argv) > 1 else ""
    secret_names = argv[2] if len(argv) > 2 else ""
    token = argv[3] if len(argv) > 3 else ""
    code, _ = report(runpod_client, template_id, secret_names, token)
    return code


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
