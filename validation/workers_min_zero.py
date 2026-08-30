"""workersMin -> 0 on ONE endpoint. A strict spend REDUCTION.

Owner decision 2026-08-30. The endpoint the owner created in the console
came up workersMin=1/workersMax=1 on an A5000, which holds the card
continuously whether or not a job is running, and which the admission gate
refuses outright (`endpoint-config-refused`: CI verifies an endpoint's
configuration, it never authors one). Asked which way to unblock it, the
owner chose to drop workersMin to zero rather than relax the guard.

WHY THIS MODULE EXISTS RATHER THAN A PATCH INSIDE THE DRIVER. The spend
driver must keep refusing a badly-shaped endpoint — that refusal is the
control. A driver that repaired what it was supposed to be checking would
be checking nothing. So the repair is a separate, deliberate, $0 dispatch,
and the driver's gate stays exactly as strict as it was.

It can only ever reduce spend: runpod_client.set_workers_min_zero takes no
value, so there is no argument by which this becomes a scale-up, and
workersMin is the only field in the PATCH body.
"""

from __future__ import annotations

import json

# Fields a PATCH must not move. workersMax is here too: the pair is what
# the admission gate reads, and a run that dropped the floor while quietly
# raising the ceiling would pass the gate having made things worse.
GUARDED = (
    "templateId",
    "gpuTypeIds",
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


def drift(before: dict, after: dict) -> dict:
    return {
        key: {"before": before.get(key), "after": after.get(key)}
        for key in GUARDED
        if before.get(key) != after.get(key)
    }


def apply(client, endpoint_id: str) -> dict:
    if not endpoint_id:
        raise Refused(
            "endpoint-missing",
            "a blank endpoint id would write somewhere nobody named",
        )
    _, before = client.get_endpoint(endpoint_id)
    was = before.get("workersMin")
    if was == 0:
        # Idempotent, and silent about it rather than sending a PATCH that
        # would restart a worker to change nothing.
        return {
            "endpoint": endpoint_id,
            "workers_min_before": 0,
            "workers_min_after": 0,
            "already_zero": True,
            "guarded_fields_unchanged": list(GUARDED),
        }

    status, raw = client.set_workers_min_zero(endpoint_id)
    if status not in (200, 201, 202):
        raise Refused(
            "patch-refused",
            f"PATCH /endpoints/{endpoint_id} -> {status} (body: {raw[:300]!r})",
        )

    # VERIFY BY RE-READING. The write's own response is the API claiming it
    # worked; the endpoint read is the endpoint saying so. The 2026-08-30
    # template retarget returned success and silently dropped a registry
    # credential, which cost hours and was invisible in everything the
    # write printed.
    _, after = client.get_endpoint(endpoint_id)
    landed = after.get("workersMin")
    if landed != 0:
        raise Refused(
            "not-applied",
            f"asked for workersMin=0, the endpoint reports {landed!r}. The "
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
        "workers_min_before": was,
        "workers_min_after": landed,
        "already_zero": False,
        "guarded_fields_unchanged": list(GUARDED),
    }


def report(client, endpoint_id: str) -> tuple:
    try:
        result = apply(client, endpoint_id)
    except Refused as refusal:
        print(f"REFUSED {refusal.code}: {refusal.detail}")
        print("NOTHING WAS WRITTEN")
        return 1, {"refused": refusal.code}
    except Exception as exc:
        print(f"REFUSED unexpected: {type(exc).__name__}: {exc}")
        print("NOTHING IS CONFIRMED — read the endpoint before assuming a state")
        return 1, {"refused": type(exc).__name__}

    print(json.dumps(result, indent=2, sort_keys=True))
    if result["already_zero"]:
        print("ALREADY ZERO: no PATCH sent, so no worker was restarted")
        return 0, result
    print(f"workersMin {result['workers_min_before']} -> 0, confirmed by "
          "re-reading the endpoint")
    print(f"UNCHANGED: {', '.join(GUARDED)}")
    print("SPEND REDUCTION: the GPU is now paid for only while a job is on "
          "it. A queued job starts a worker, which pulls the image first.")
    return 0, result


def main(argv) -> int:
    import runpod_client

    code, _ = report(runpod_client, argv[1] if len(argv) > 1 else "")
    return code


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
