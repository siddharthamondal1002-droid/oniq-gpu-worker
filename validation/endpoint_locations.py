"""Widen ONE endpoint back to every datacenter it may legally run in.

THE FAULT, read off the console's own Releases tab on 2026-09-01 because
no API surface exposes it:

    Release #3   locations   ALL -> EU-RO-1

Attaching a network volume NARROWS an endpoint's placement to the
volume's single datacenter. Detaching does NOT widen it back — release #7
the same morning removed the volume and left the pin exactly where it
was. The narrowing is ONE-WAY and it outlives the thing that caused it.

So the endpoint spends the rest of its life able to run in one
datacenter, and on the day that datacenter's approved VRAM tier runs dry
it has ZERO placement candidates. The signature is every worker bucket at
0 INCLUDING throttled — not "wants a card and cannot get one", which
reads throttled: 1 (measured on 7disu6my0mloco, 2026-08-30), but "has
nothing to try at all".

AND THAT IS WHY CREATING A NEW ENDPOINT LOOKED LIKE THE CURE. A new
endpoint is born at locations: ALL and finds capacity because it may look
everywhere. Attach storage to it and the pin returns; the next day it is
dead again. Daily endpoint recreation was never a RunPod requirement — it
was this field, re-narrowed each time.

WHAT THIS WRITES: `dataCenterIds`, set to every id the endpoint PATCH
schema accepts. That list is the schema's OWN documented default for the
field — the state a fresh endpoint starts in — so this restores rather
than invents. One field is sent and nothing else.

WHAT IT CANNOT PROVE, said plainly because a module that pretended
otherwise cost a run and a wrong diagnosis this same morning: NEITHER
REST ROUTE RETURNS THIS FIELD. Measured 2026-09-01 with
validation/endpoint_read.py across GET /endpoints/{endpointId} and GET
/endpoints — `dataCenterIds` and `locations` are ABSENT from both, not
null. A guard that refuses unless the endpoint echoes the value can never
pass, which is exactly the `datacenter-not-pinned` wall that blocked a
correct write earlier today. So this module verifies what it CAN — that
no other field moved — and names where the real proof lives: the
console's Releases tab, which records the change as a `locations` row.

STORAGE AND BREADTH ARE EXCLUSIVE, and the caller is told so rather than
left to find out. A worker placed in US-TX-1 cannot mount a volume that
lives in EU-RO-1. Widening an endpoint that still holds a network volume
would produce an endpoint whose workers may be placed where their own
storage is not, so this REFUSES while a volume is attached.

OWNER DECISION 2026-09-01, option B: no network volume, all datacenters.
Model weights ride the image and the worker's own container disk; the
endpoint keeps the widest placement pool RunPod offers. The 17.74 GiB
text encoder cannot go back into the image — run 33434875038 measured
that at 57.97 GiB and a hosted runner could not build it — so it lands on
container disk instead. Storage was traded for placement deliberately.
"""

from __future__ import annotations

import json

from validation.volume_setup import KNOWN_DATACENTERS

TOKEN = "WIDEN-LOCATIONS"

# Fields this PATCH must not move. Every one of them decides either what
# the endpoint costs or what it can run, and the 2026-08-30 template
# retarget is the standing reminder that a write can carry away a field it
# never named.
GUARDED = (
    "templateId",
    "gpuTypeIds",
    "gpuCount",
    "workersMin",
    "workersMax",
    "workersStandby",
    "idleTimeout",
    "executionTimeoutMs",
    "networkVolumeId",
    "scalerType",
    "scalerValue",
)


class Refused(Exception):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def check_token(token: str) -> None:
    if (token or "").strip() != TOKEN:
        raise Refused(
            "token-wrong",
            f"this changes where a production endpoint may run; it needs "
            f"{TOKEN!r} exactly. Nothing was written.",
        )


def check_no_volume(endpoint: dict) -> None:
    """A volume and a wide placement pool cannot both be true.

    Not a stylistic objection: a worker placed in one datacenter cannot
    mount storage that exists in another. Widening an endpoint that still
    holds a volume produces workers scheduled where their weights are
    not, which fails at read time inside a job that has already been paid
    for.
    """
    one = endpoint.get("networkVolumeId") or ""
    many = [v for v in (endpoint.get("networkVolumeIds") or []) if v]
    if one or many:
        raise Refused(
            "volume-still-attached",
            f"networkVolumeId={one!r} networkVolumeIds={many!r}. A worker "
            "placed in one datacenter cannot mount a volume in another, so "
            "widening now would schedule workers away from their own "
            "weights. Detach first (volume-detach), then widen.",
        )


def drift(before: dict, after: dict) -> dict:
    """Guarded fields that moved. Empty means nothing else changed."""
    moved = {}
    for key in GUARDED:
        was, now = before.get(key), after.get(key)
        if was != now:
            moved[key] = {"before": was, "after": now}
    return moved


def widen(client, endpoint_id: str, token: str) -> dict:
    check_token(token)
    if not (endpoint_id or "").strip():
        raise Refused("endpoint-missing", "no endpoint id was given")
    try:
        _, endpoint = client.get_endpoint(endpoint_id)
    except Exception as exc:
        raise Refused("endpoint-unreadable", f"{type(exc).__name__}") from exc
    if not isinstance(endpoint, dict) or endpoint.get("id") != endpoint_id:
        raise Refused(
            "endpoint-unreadable",
            "the read did not answer with the id that was asked for",
        )
    check_no_volume(endpoint)

    wanted = list(KNOWN_DATACENTERS)
    before, after = client.set_data_center_ids(endpoint_id, wanted)

    moved = drift(before, after)
    if moved:
        raise Refused(
            "collateral-change",
            f"the PATCH moved fields it was not asked to move: {moved}",
        )
    return {
        "endpoint": endpoint_id,
        "data_center_ids_written": wanted,
        "datacenter_count": len(wanted),
        "guarded_fields_unchanged": list(GUARDED),
        # Reported as unreadable rather than as a value, because that is
        # what it is. See the module docstring.
        "data_center_ids_readable": "dataCenterIds" in after,
        "locations_readable": "locations" in after,
        "network_volume_id": after.get("networkVolumeId") or "",
    }


def report(client, endpoint_id: str, token: str) -> int:
    try:
        result = widen(client, endpoint_id, token)
    except Refused as refusal:
        print(f"REFUSED {refusal.code}: {refusal.detail}")
        print("NOTHING WAS WRITTEN")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    print()
    print(f"WIDENED {endpoint_id} to all {result['datacenter_count']} "
          "datacenters the PATCH schema accepts — the field's own default, "
          "i.e. the state a new endpoint is born in.")
    print("UNCHANGED: " + ", ".join(GUARDED))
    print()
    print("NOT VERIFIABLE FROM HERE, and not claimed: neither REST route "
          "returns dataCenterIds or locations (measured 2026-09-01, both "
          "routes, keys ABSENT not null). Confirm in the console's Releases "
          "tab — the change appears there as a `locations` row going back "
          "to ALL. If it does not appear, the write did not land.")
    print("$0 - a placement pool is not a charge. Nothing was dispatched.")
    return 0


def main(argv) -> int:
    import runpod_client as rp

    if len(argv) < 3:
        print("usage: endpoint_locations <endpoint_id> <token>")
        return 2
    return report(rp, argv[1], argv[2])


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv))
