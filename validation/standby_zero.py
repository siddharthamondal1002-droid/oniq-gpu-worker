"""workersStandby -> 0 — owner directive 2026-08-26 (five-video battery,
Phase 2): NO JOB = NO GPU WORKER. A standby worker keeps warmth the
owner did not ask to pay for, and it makes zero-worker termination
unprovable — runs #31 and #33 both ended TERMINATION_UNKNOWN on exactly
this field.

The flow is verify -> patch (the literal zero) -> FRESH re-read ->
verify, and nothing here accepts a worker count as input: the only
mutation the harness owns is runpod_client.set_workers_standby_zero,
whose body is hard-coded, so this path can only ever scale DOWN. The
module exits nonzero unless a fresh read shows workersStandby 0 beside
workersMin 0 / workersMax 1. It never submits a job — the verification
is free.
"""

from __future__ import annotations

import json

from validation.spend_run import SpendStop, _show


def _endpoint_list(doc):
    return doc if isinstance(doc, list) else doc.get("endpoints", [])


def run(client) -> dict:
    raw, endpoints = client.get_endpoints()
    ep_list = _endpoint_list(endpoints)
    _show("endpoints (raw, redacted)", json.loads(raw))
    if len(ep_list) != 1:
        raise SpendStop(
            "endpoint-not-singular",
            f"{len(ep_list)} endpoint(s); need exactly one to patch safely",
        )
    endpoint = ep_list[0]
    endpoint_id = endpoint.get("id")
    if endpoint.get("workersMin") != 0 or endpoint.get("workersMax") != 1:
        raise SpendStop(
            "endpoint-config-drift",
            "workersMin/workersMax read back as "
            f"{endpoint.get('workersMin')}/{endpoint.get('workersMax')} — "
            "must be 0/1 before anything else; fixing that is an owner "
            "console action",
        )

    if endpoint.get("workersStandby") == 0:
        print("workersStandby already 0 — verified, nothing changed")
        return {"endpoint_id": endpoint_id, "workers_standby": 0, "changed": False}

    print(
        f"workersStandby reads {endpoint.get('workersStandby')!r} — "
        "patching to the literal 0 (spend reduction)"
    )
    status, body = client.set_workers_standby_zero(endpoint_id)
    print("patch status:", status)
    # The body is the diagnosis (both transports' answers, verbatim) —
    # print it in full BEFORE any stop can truncate it: run #35 lost the
    # GraphQL half to a 200-char slice and cost this exact round trip.
    print("patch response:", body)
    if status not in (200, 201):
        raise SpendStop(
            "standby-patch-refused",
            f"both transports refused (statuses/bodies printed above); "
            "set workersStandby to 0 in the RunPod console, then "
            "re-dispatch standby-zero to verify",
        )

    # Never trust the PATCH echo — only a fresh read counts.
    raw2, fresh = client.get_endpoints()
    match = [e for e in _endpoint_list(fresh) if e.get("id") == endpoint_id]
    _show("endpoints after patch (raw, redacted)", json.loads(raw2))
    if len(match) != 1 or match[0].get("workersStandby") != 0:
        raise SpendStop(
            "standby-not-zero",
            "the fresh read does not show workersStandby 0 — the patch "
            "did not take; not proceeding",
        )
    print("workersStandby = 0 — verified by fresh read")
    return {"endpoint_id": endpoint_id, "workers_standby": 0, "changed": True}


def main() -> int:
    import runpod_client as rp

    try:
        result = run(rp)
    except SpendStop as stop:
        print(f"STOP [{stop.code}]: {stop.message}")
        return 1
    print(
        "STANDBY-ZERO PASS — endpoint",
        result["endpoint_id"],
        "(changed)" if result["changed"] else "(already zero)",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
