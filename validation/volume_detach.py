"""Detach the network volume from ONE endpoint. Token-gated, reversible.

DIAGNOSTIC FIRST, REPAIR SECOND. Endpoint 9gh6qbou1in8yb has reported

    jobs    {inQueue: 1}
    workers {idle 0, initializing 0, ready 0, running 0, throttled 0,
             unhealthy 0}

for ninety minutes, across two readings, with workersMin=1 set. A minimum
of one means RunPod should hold a worker whether or not anything is
queued. It is holding none, and not even reporting one as throttled.

The same endpoint reported initializing=2 BEFORE the volume was attached.
The volume lives in US-MO-2; the endpoint carries no dataCenterIds at all,
and a PATCH setting both together is refused 400 by RunPod's schema.

Two hypotheses fit every fact so far:

  A. the volume pins the endpoint to a datacenter it is not configured to
     run in, so the scheduler has nowhere legal to place a worker;
  B. there is no A5000 capacity in that datacenter at all.

Detaching separates them, and nothing else available here does. If workers
appear, it is A and the fix is a placement one. If they do not, it is B
and no amount of endpoint configuration helps.

WHY THIS IS SAFE TO RUN. Detaching sends networkVolumeId="" and no
datacenter, so the pin leaves with the volume (see
runpod_client.attach_network_volume). The volume itself is untouched:
detach is not delete, the 50 GB and everything hydrated onto it survive,
and re-attaching is the same call with the id back in it. Storage keeps
billing either way, which is the point — nothing is lost by looking.
"""

from __future__ import annotations

import json

GUARDED = (
    "templateId",
    "gpuTypeIds",
    "workersMin",
    "workersMax",
    "workersStandby",
    "idleTimeout",
    "executionTimeoutMs",
)

TOKEN = "DETACH-VOLUME"


class Refused(Exception):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def apply(client, endpoint_id: str, token: str) -> dict:
    if token != TOKEN:
        raise Refused("token-wrong", f"the literal token {TOKEN} was not typed")
    if not endpoint_id:
        raise Refused("endpoint-missing", "a blank endpoint id names nothing")

    _, before = client.get_endpoint(endpoint_id)
    was = before.get("networkVolumeId") or ""
    if not was:
        return {
            "endpoint": endpoint_id,
            "network_volume_id_before": "",
            "network_volume_id_after": "",
            "already_detached": True,
            "guarded_fields_unchanged": list(GUARDED),
        }

    # Empty volume id, and therefore no datacenter: the pin leaves with it.
    before, after = client.attach_network_volume(endpoint_id, "")
    landed = after.get("networkVolumeId") or ""
    if landed:
        raise Refused(
            "detach-not-applied",
            f"asked to detach, the endpoint still reports {landed!r}. The "
            "PATCH returned success and did not take effect.",
        )
    moved = {
        key: {"before": before.get(key), "after": after.get(key)}
        for key in GUARDED
        if before.get(key) != after.get(key)
    }
    if moved:
        raise Refused(
            "collateral-change",
            f"the PATCH moved fields it was not asked to move: {moved}",
        )
    return {
        "endpoint": endpoint_id,
        "network_volume_id_before": was,
        "network_volume_id_after": "",
        "already_detached": False,
        "data_center_ids_after": after.get("dataCenterIds"),
        "guarded_fields_unchanged": list(GUARDED),
    }


def report(client, endpoint_id: str, token: str) -> tuple:
    try:
        result = apply(client, endpoint_id, token)
    except Refused as refusal:
        print(f"REFUSED {refusal.code}: {refusal.detail}")
        print("NOTHING WAS WRITTEN")
        return 1, {"refused": refusal.code}
    except Exception as exc:
        print(f"REFUSED unexpected: {type(exc).__name__}: {exc}")
        print("NOTHING IS CONFIRMED — read the endpoint before assuming a state")
        return 1, {"refused": type(exc).__name__}

    print(json.dumps(result, indent=2, sort_keys=True))
    if result["already_detached"]:
        print("ALREADY DETACHED: no PATCH sent")
        return 0, result
    print(f"DETACHED {result['network_volume_id_before']} — the VOLUME still "
          "exists and still holds whatever was hydrated onto it. Detach is "
          "not delete; re-attaching is the same call with the id back in it.")
    print("NOW READ THE WORKERS. If they start appearing, the volume's "
          "datacenter pin was the blocker. If they still do not, the "
          "datacenter has no A5000 for this endpoint and no configuration "
          "change will conjure one.")
    return 0, result


def main(argv) -> int:
    import runpod_client

    code, _ = report(
        runpod_client,
        argv[1] if len(argv) > 1 else "",
        argv[2] if len(argv) > 2 else "",
    )
    return code


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
