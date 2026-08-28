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


def probe_only(client) -> None:
    """Print what the API says about standby. Writes NOTHING.

    Owner directive 2026-08-27: run the standby diagnostic, and stop
    re-attempting a mutation already proven unsupported. Both, which was
    impossible while the probes lived inside the patch-failure branch —
    reading the answer required attempting the write first. They are the
    same two read-only introspections; only the trigger is new.

    Unreadable prints as unreadable. An introspection that cannot run is
    not evidence that nothing exists, the same way the sweep reports None
    rather than 0.
    """
    probe = getattr(client, "standby_schema_probe", lambda: None)()
    if probe is None:
        print("graphql schema probe: unreadable (introspection blocked)")
    else:
        _show("graphql schema probe (names only)", probe)
    rest_probe = getattr(client, "rest_schema_probe", lambda: None)()
    if rest_probe is None:
        print("rest schema probe: unreadable")
    else:
        _show("rest openapi probe (names only)", rest_probe)


def read_only(client) -> dict:
    """The live standby facts, read and classified. No mutation."""
    raw, endpoints = client.get_endpoints()
    ep_list = _endpoint_list(endpoints)
    _show("endpoints (raw, redacted)", json.loads(raw))
    if len(ep_list) != 1:
        raise SpendStop(
            "endpoint-not-singular",
            f"{len(ep_list)} endpoint(s); need exactly one to read safely",
        )
    endpoint = ep_list[0]
    standby = endpoint.get("workersStandby")
    print("workersStandby READABLE :", repr(standby))
    print("workersMin / workersMax :", endpoint.get("workersMin"), "/", endpoint.get("workersMax"))
    probe_only(client)
    return {
        "endpoint_id": endpoint.get("id"),
        "workers_standby": standby,
        "workers_min": endpoint.get("workersMin"),
        "workers_max": endpoint.get("workersMax"),
    }


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
        # Before declaring the console the only path, read the schema
        # itself (read-only introspection): if any mutation is standby-
        # shaped, its name prints here and the next iteration uses it.
        probe_only(client)
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


def main(argv=None) -> int:
    import runpod_client as rp

    argv = list(argv or [])
    if argv[:1] == ["probe"]:
        # Diagnostic only. Nothing here can write, so it is safe to run
        # after the mutation has been proven unsupported.
        try:
            facts = read_only(rp)
        except SpendStop as stop:
            print(f"STOP [{stop.code}]: {stop.message}")
            return 1
        print("STANDBY PROBE COMPLETE — workersStandby =", repr(facts["workers_standby"]))
        return 0

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
    import sys

    raise SystemExit(main(sys.argv[1:]))
