"""Conform ONE endpoint to the worker bounds CI requires. Spend reduction.

validation/admission.check_endpoint_config refuses any endpoint that is
not min_workers=0/max_workers=1, and its docstring says why the driver
never fixes it: "CI verifies an endpoint's configuration; it never authors
one." That refusal is the control, and a driver that repaired what it is
supposed to be checking would be checking nothing.

So the repair is this: a separate, deliberate, $0 dispatch that writes
exactly those two literals and nothing else.

WHY BOTH FIELDS TOGETHER. They are one decision, not two. min=0 with
max=3 still permits three concurrent rentals; max=1 with min=1 still bills
a card continuously whether or not work exists. Sending them in one PATCH
also costs one worker restart instead of two, and on a 25 GiB image a
restart is a fresh pull.

It cannot raise anything: runpod_client.set_worker_bounds_min0_max1 takes
no values, so there is no argument by which a reduction becomes an
increase.
"""

from __future__ import annotations

import json

# Fields a PATCH must not move. networkVolumeId and dataCenterIds are here
# because moving either relocates where the endpoint may run — measured
# 2026-08-30, when a volume in a datacenter absent from RunPod's own
# dataCenterIds enum left an endpoint unable to place a single worker.
GUARDED = (
    "templateId",
    "gpuTypeIds",
    "idleTimeout",
    "executionTimeoutMs",
    "networkVolumeId",
    "dataCenterIds",
)

TARGET_MIN = 0
TARGET_MAX = 1


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
        raise Refused("endpoint-missing", "a blank endpoint id names nothing")

    _, before = client.get_endpoint(endpoint_id)
    was = (before.get("workersMin"), before.get("workersMax"))
    if was == (TARGET_MIN, TARGET_MAX):
        return {
            "endpoint": endpoint_id,
            "workers_min_before": was[0], "workers_max_before": was[1],
            "workers_min_after": was[0], "workers_max_after": was[1],
            "already_conformed": True,
            "guarded_fields_unchanged": list(GUARDED),
        }

    status, raw = client.set_worker_bounds_min0_max1(endpoint_id)
    if status not in (200, 201, 202):
        raise Refused(
            "patch-refused",
            f"PATCH /endpoints/{endpoint_id} -> {status} (body: {raw[:400]!r})",
        )

    _, after = client.get_endpoint(endpoint_id)
    landed = (after.get("workersMin"), after.get("workersMax"))
    if landed != (TARGET_MIN, TARGET_MAX):
        raise Refused(
            "not-applied",
            f"asked for min={TARGET_MIN}/max={TARGET_MAX}, the endpoint "
            f"reports min={landed[0]!r}/max={landed[1]!r}. The PATCH "
            "returned success and did not take effect.",
        )
    moved = drift(before, after)
    if moved:
        raise Refused(
            "collateral-change",
            f"the PATCH moved fields it was not asked to move: {moved}",
        )
    return {
        "endpoint": endpoint_id,
        "workers_min_before": was[0], "workers_max_before": was[1],
        "workers_min_after": landed[0], "workers_max_after": landed[1],
        "already_conformed": False,
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
    if result["already_conformed"]:
        print(f"ALREADY min={TARGET_MIN}/max={TARGET_MAX} — no PATCH sent, so "
              "no worker was restarted")
        return 0, result
    print(f"workersMin {result['workers_min_before']} -> {TARGET_MIN}, "
          f"workersMax {result['workers_max_before']} -> {TARGET_MAX}, "
          "confirmed by re-reading the endpoint")
    print(f"UNCHANGED: {', '.join(GUARDED)}")
    print("SPEND REDUCTION: the GPU is paid for only while a job is on it, "
          "and one job can no longer become three.")
    return 0, result


def main(argv) -> int:
    import runpod_client

    code, _ = report(runpod_client, argv[1] if len(argv) > 1 else "")
    return code


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
