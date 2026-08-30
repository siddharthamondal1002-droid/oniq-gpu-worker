"""Point ONE endpoint at a template that ALREADY EXISTS. Token-gated.

Owner directive 2026-08-30: the owner deleted p3zmlv8ek10dzt and
ynysmj3dm92cwp and created 9gh6qbou1in8yb in the console. The new endpoint
came up referencing templateId ei22bjog46, which is not among the fifteen
templates the account reports — the same dangling reference that made
hhhdwtjw0y unusable, on a new id.

WHY THIS IS NOT template_attach. That module CREATES a template and then
points the endpoint at it, which is right when nothing usable exists. Here
something usable does: aqa3wkdf8g carries the digest-pinned image, 201 GB
of container disk, and the R2 environment the owner set by hand. Creating
a sixteenth template would produce one with the correct image and NO
environment, and every job against it would fail closed with
storage-not-configured — a fresh template is not a cheaper version of an
existing one, it is a different object with none of its history.

THE ONE REFUSAL THAT MATTERS: a template id that is not on the account.
The endpoint is broken right now precisely because it names a template
that is not there. Writing a second unverified id over the first would
look like a repair, read like a repair in the log, and leave the endpoint
exactly as unusable. So the target is proven to exist BEFORE the PATCH,
and the PATCH is proven to have landed by re-reading the endpoint.

Cheap by construction: templateId is the only field sent (see
runpod_client.attach_template), and every field that governs spend is
compared before and after.
"""

from __future__ import annotations

import json

TOKEN = "POINT-AT-TEMPLATE"

# Fields a PATCH must not move. Worker bounds and the GPU list decide what
# the endpoint costs; networkVolumeId decides which datacenter it is pinned
# to; executionTimeoutMs decides whether a long job is killed with the
# money spent. None of them is being asked to change here, so any movement
# is the API doing something that was not requested and the run says so
# rather than reporting a clean success.
GUARDED = (
    "gpuTypeIds",
    "workersMin",
    "workersMax",
    "workersStandby",
    "idleTimeout",
    "executionTimeoutMs",
    "networkVolumeId",
)


class Refused(Exception):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def check_token(token: str) -> None:
    if token != TOKEN:
        raise Refused(
            "token-wrong",
            f"the literal token {TOKEN} was not typed; nothing was written",
        )


def template_ids(client):
    """Every template id the account reports, or None when unreadable.

    None is UNKNOWN. It is never treated as 'the template is missing' and
    never as 'the template is present' — an unreadable account cannot
    prove either, so the caller refuses instead of guessing.
    """
    rows = client.list_templates_graphql()
    if rows is None:
        return None
    return {t.get("id") for t in rows if isinstance(t, dict) and t.get("id")}


def check_target_exists(client, template_id: str) -> None:
    ids = template_ids(client)
    if ids is None:
        raise Refused(
            "templates-unreadable",
            "the account's template list could not be read, so the target "
            "cannot be proven to exist. UNKNOWN is not a green light.",
        )
    if template_id not in ids:
        raise Refused(
            "target-template-missing",
            f"{template_id!r} is not among the {len(ids)} templates on the "
            "account. Pointing the endpoint at it would replace one dangling "
            "reference with another.",
        )


def current_template_id(endpoint: dict):
    for key in ("templateId", "template_id"):
        value = endpoint.get(key)
        if value:
            return value
    return None


def drift(before: dict, after: dict) -> dict:
    return {
        key: {"before": before.get(key), "after": after.get(key)}
        for key in GUARDED
        if before.get(key) != after.get(key)
    }


def apply(client, endpoint_id: str, template_id: str, token: str) -> dict:
    check_token(token)
    if not endpoint_id or not template_id:
        raise Refused(
            "identifiers-missing",
            f"endpoint={endpoint_id!r} template={template_id!r} — a blank id "
            "would make this write somewhere nobody named",
        )
    check_target_exists(client, template_id)

    _, before = client.get_endpoint(endpoint_id)
    was = current_template_id(before)
    if was == template_id:
        # Idempotent. Re-running must not restart a worker for nothing: a
        # templateId PATCH destroys the running worker, and on a 25 GiB
        # image that costs a fresh pull to change nothing.
        return {
            "endpoint": endpoint_id,
            "template_id": template_id,
            "template_id_before": was,
            "already_pointed": True,
            "guarded_fields_unchanged": list(GUARDED),
        }

    client.attach_template(endpoint_id, template_id)

    # VERIFY BY RE-READING. The PATCH's own response is the write claiming
    # it worked; the endpoint read is the endpoint saying so.
    _, after = client.get_endpoint(endpoint_id)
    landed = current_template_id(after)
    if landed != template_id:
        raise Refused(
            "not-applied",
            f"asked for {template_id!r}, the endpoint reports {landed!r}. The "
            "PATCH returned success and did not take effect.",
        )
    moved = drift(before, after)
    if moved:
        raise Refused(
            "collateral-change",
            f"the PATCH moved fields it was not asked to move: {moved}",
        )
    return {
        "endpoint": endpoint_id,
        "template_id": template_id,
        "template_id_before": was,
        "already_pointed": False,
        "guarded_fields_unchanged": list(GUARDED),
    }


def report(client, endpoint_id: str, template_id: str, token: str) -> tuple:
    try:
        result = apply(client, endpoint_id, template_id, token)
    except Refused as refusal:
        print(f"REFUSED {refusal.code}: {refusal.detail}")
        print("NOTHING WAS WRITTEN")
        return 1, {"refused": refusal.code}
    except Exception as exc:
        print(f"REFUSED unexpected: {type(exc).__name__}: {exc}")
        print("NOTHING IS CONFIRMED — read the endpoint before assuming a state")
        return 1, {"refused": type(exc).__name__}

    print(json.dumps(result, indent=2, sort_keys=True))
    if result["already_pointed"]:
        print(f"ALREADY POINTED: {endpoint_id} was on {template_id} — no PATCH "
              "sent, so no worker was restarted")
        return 0, result
    print(f"WAS {result['template_id_before']!r} -> NOW {template_id!r}, "
          "confirmed by re-reading the endpoint")
    print(f"UNCHANGED: {', '.join(GUARDED)}")
    print("A templateId change destroys the running worker; the next job "
          "starts a fresh one, which pulls the image before it runs.")
    return 0, result


def main(argv) -> int:
    import runpod_client

    endpoint_id = argv[1] if len(argv) > 1 else ""
    template_id = argv[2] if len(argv) > 2 else ""
    token = argv[3] if len(argv) > 3 else ""
    code, _ = report(runpod_client, endpoint_id, template_id, token)
    return code


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
