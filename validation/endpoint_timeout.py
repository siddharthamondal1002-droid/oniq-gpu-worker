"""Raise the execution ceiling on ONE endpoint. Owner-authorized, gated.

WHY THIS EXISTS
---------------
Every probe before Hunyuan loaded weights that were already baked into the
image. The Hunyuan probe is the first that downloads its checkpoint INSIDE
the paid job — 32.26 GiB, measured from the registry listing. The endpoint
ceiling is 600000 ms (10 minutes), which that download cannot fit into:
32.26 GiB in ten minutes needs about 440 Mbit/s sustained with nothing
left over for loading, inference, encoding or upload.

A job killed against the ceiling is the worst outcome available: the
worker is billed for the whole attempt and there is no artifact. So the
ceiling is raised BEFORE the job is dispatched rather than discovered by
one dying against it.

Owner authorization, 2026-08-30: raise it to 45 minutes. The worst case
is one job holding the card for that long, which at the live secure
rate stays inside the job cap. The rate itself is deliberately not
written down here: a price in a comment goes stale without anyone
noticing, and the admission gate refuses one on sight.

WHAT IT WILL NOT DO
-------------------
It sends executionTimeoutMs and nothing else. It cannot move a worker
bound, cannot touch a template, cannot submit a job. The workflow gate
requires the literal token, and the module carries no other mutating verb
so it cannot grow one quietly.

It also compares the endpoint document before and after. That is not
ceremony: the 2026-08-30 template retarget sent the field it meant to
change and silently dropped containerRegistryAuthId, leaving an endpoint
that could not pull its own image, and nothing the template printed showed
it. A write that is not read back is a write nobody checked.
"""

from __future__ import annotations

import json
import sys

# The fields a timeout change must NOT move. Compared by name so a
# provider that starts returning a new key does not silently pass.
GUARDED = (
    "templateId",
    "gpuTypeIds",
    "workersMin",
    "workersMax",
    "workersStandby",
    "idleTimeout",
    "networkVolumeId",
    "scalerType",
    "scalerValue",
)


# 45 minutes, per the owner's 2026-08-30 authorization. A constant rather
# than a dispatch input because GitHub caps workflow_dispatch at 25 inputs
# and a decision already taken does not need to be re-typed on every run.
DEFAULT_TIMEOUT_MS = 2_700_000


class Refused(Exception):
    """Raised when the endpoint did not end up the way it was asked to."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code


def drift(before: dict, after: dict) -> dict:
    """Guarded fields that changed. Empty dict means nothing else moved."""
    moved = {}
    for key in GUARDED:
        was, now = before.get(key), after.get(key)
        if was != now:
            moved[key] = {"before": was, "after": now}
    return moved


def apply(client, endpoint_id: str, timeout_ms: int) -> dict:
    before, after = client.set_execution_timeout(endpoint_id, timeout_ms)

    landed = after.get("executionTimeoutMs")
    if landed != timeout_ms:
        raise Refused(
            "timeout-not-applied",
            f"asked for {timeout_ms}, endpoint reports {landed!r}. The PATCH "
            "returned success, so this is the provider accepting a request "
            "and not honouring it — do not dispatch a job that depends on "
            "the new ceiling.",
        )

    moved = drift(before, after)
    if moved:
        raise Refused(
            "collateral-change",
            f"the PATCH moved fields it was not asked to move: {moved}. "
            "Restore them before dispatching anything; an endpoint whose "
            "worker bounds or template shifted underneath a probe will "
            "spend money answering the wrong question.",
        )

    return {
        "endpoint": endpoint_id,
        "execution_timeout_ms_before": before.get("executionTimeoutMs"),
        "execution_timeout_ms_after": landed,
        "guarded_fields_unchanged": list(GUARDED),
    }


def main(argv) -> int:
    import runpod_client as rp

    if len(argv) < 2 or not argv[1]:
        print("usage: endpoint_timeout <endpoint_id> [timeout_ms]")
        return 2
    endpoint_id = argv[1]
    timeout_ms = int(argv[2]) if len(argv) > 2 and argv[2] else DEFAULT_TIMEOUT_MS

    print(f"raising executionTimeoutMs on {endpoint_id} to {timeout_ms} ms "
          f"({timeout_ms / 60000:.0f} min)")
    try:
        result = apply(rp, endpoint_id, timeout_ms)
    except Refused as exc:
        print(f"REFUSED {exc}")
        return 1
    print(json.dumps(result, indent=2))
    print("$0 - a ceiling is not a charge. Nothing was dispatched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
