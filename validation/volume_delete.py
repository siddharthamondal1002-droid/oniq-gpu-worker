"""Delete ONE network volume. IRREVERSIBLE, and gated accordingly.

OWNER DIRECTIVE 2026-09-01: "remove any unnecessary billing." Under option
B — no network volume, every datacenter — the account's volumes mount
nowhere and bill for storage nothing reads. At the rate MEASURED from this
account's own /billing/networkvolumes line ($0.07/GB-month), that is
$3.50/month for 50 GB and $0.70 for 10.

WHY THIS IS SHAPED DIFFERENTLY FROM EVERY OTHER WRITER HERE. Detaching a
volume, retargeting a template, widening a placement pool — each is undone
by running the opposite operation. This one is not. A volume deleted is a
volume gone, along with anything hydrated onto it, and no read afterwards
can tell you what it held. So the gate is not "a literal token": it is a
literal token WITH THE VOLUME ID IN IT.

    volume_token = "DELETE-VOLUME:l99s0q5kd2"

One input carries both because workflow_dispatch is at GitHub's 25-input
cap, but the shape is better than two fields anyway: an operator cannot
authorize "a deletion" in the abstract and have the target come from
somewhere else. The id they typed is the id that dies.

WHAT IT REFUSES, each because the alternative destroys something:

  token-malformed     the literal is right but no id followed it, or the
                      reverse. A half-typed authorization is not one.
  volume-not-found    the id is not on the account. Deleting nothing while
                      reporting success would leave the real volume billing
                      and the operator believing it was handled.
  volume-attached     some endpoint still references it. That endpoint's
                      workers would lose their weights mid-flight, and the
                      endpoint would keep the datacenter pin the volume
                      gave it while losing the storage that justified it.
  volume-still-listed the delete returned success and the volume is still
                      there. Reported rather than retried: a destructive
                      call that did not take is a fact to look at, not one
                      to make again.

It deletes exactly ONE volume per run. There is deliberately no batch
shape — a loop over an account's storage is how the wrong thing goes.
"""

from __future__ import annotations

import json

LITERAL = "DELETE-VOLUME"

# Measured from this account's own billing line, not recalled. Same figure
# volume_setup states, imported so the two can never disagree about what a
# volume costs.
from validation.volume_setup import USD_PER_GB_MONTH


class Refused(Exception):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def parse_token(token: str) -> str:
    """The volume id the operator authorized, or raise.

    Both halves are required. `DELETE-VOLUME` alone authorizes nothing in
    particular, and a bare id authorizes nothing at all.
    """
    text = (token or "").strip()
    if not text.startswith(LITERAL + ":"):
        raise Refused(
            "token-malformed",
            f"expected {LITERAL}:<volume_id> exactly. A deletion has to name "
            "its own target — authorizing 'a deletion' and taking the id "
            "from elsewhere is how the wrong volume goes.",
        )
    volume_id = text[len(LITERAL) + 1:].strip()
    if not volume_id:
        raise Refused(
            "token-malformed",
            f"{LITERAL}: was typed with no volume id after it.",
        )
    return volume_id


def volumes(client):
    """Every network volume on the account, or raise. [] is a real answer;
    an unreadable account is NOT an empty one and must not delete on it."""
    try:
        _, doc = client._get_json(f"{client.REST_BASE}/networkvolumes")
    except Exception as exc:
        raise Refused(
            "account-unreadable",
            f"the volume list could not be read ({type(exc).__name__}). "
            "UNKNOWN is not 'no volumes'; nothing is deleted against a read "
            "that failed.",
        ) from exc
    return doc if isinstance(doc, list) else (doc or {}).get("networkVolumes") or []


def endpoints_holding(client, volume_id: str) -> list:
    """Endpoint ids that still reference this volume.

    BOTH FIELDS are checked. On this account the singular read empty while
    the plural still held a volume, on 2026-09-01 — checking one is not
    checking both, and a missed reference here costs a running endpoint its
    weights.
    """
    try:
        _, doc = client._get_json(f"{client.REST_BASE}/endpoints")
    except Exception as exc:
        raise Refused(
            "endpoints-unreadable",
            f"could not read the endpoints ({type(exc).__name__}), so "
            "whether anything still mounts this volume is UNKNOWN. Nothing "
            "is deleted on an unknown.",
        ) from exc
    rows = doc if isinstance(doc, list) else (doc or {}).get("endpoints") or []
    holding = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        refs = {row.get("networkVolumeId") or ""}
        refs.update(row.get("networkVolumeIds") or [])
        if volume_id in refs:
            holding.append(row.get("id"))
    return holding


def delete(client, token: str) -> dict:
    volume_id = parse_token(token)

    found = None
    for row in volumes(client):
        if isinstance(row, dict) and row.get("id") == volume_id:
            found = row
            break
    if found is None:
        raise Refused(
            "volume-not-found",
            f"{volume_id!r} is not on the account. Deleting nothing while "
            "reporting success would leave the real volume billing and the "
            "operator believing it was handled.",
        )

    holding = endpoints_holding(client, volume_id)
    if holding:
        raise Refused(
            "volume-attached",
            f"{volume_id!r} is still referenced by {holding}. Detach it "
            "first (volume-detach); deleting storage out from under a live "
            "endpoint takes its weights away mid-flight.",
        )

    size_gb = found.get("size")
    client.delete_network_volume(volume_id)

    # VERIFIED BY RE-READING, because the one call here that cannot be
    # undone is also the one whose success matters most to confirm.
    still = [r.get("id") for r in volumes(client) if isinstance(r, dict)]
    if volume_id in still:
        raise Refused(
            "volume-still-listed",
            f"the delete returned success and {volume_id!r} is still on the "
            "account. Reported rather than retried — a destructive call that "
            "did not take is a fact to look at, not one to make again.",
        )

    saved = round(size_gb * USD_PER_GB_MONTH, 2) if isinstance(
        size_gb, (int, float)) else None
    return {
        "volume_id": volume_id,
        "volume_name": found.get("name"),
        "size_gb": size_gb,
        "datacenter": found.get("dataCenterId"),
        "usd_per_gb_month_measured": USD_PER_GB_MONTH,
        "monthly_usd_saved": saved,
        "volumes_remaining": still,
    }


def report(client, token: str) -> int:
    try:
        result = delete(client, token)
    except Refused as refusal:
        print(f"REFUSED {refusal.code}: {refusal.detail}")
        print("NOTHING WAS DELETED")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    print()
    print(f"DELETED {result['volume_id']} ({result['volume_name']!r}, "
          f"{result['size_gb']} GB in {result['datacenter']}) — confirmed by "
          "re-reading the account, not by the call's own status.")
    if result["monthly_usd_saved"] is not None:
        print(f"BILLING REMOVED: ${result['monthly_usd_saved']}/month at the "
              f"${result['usd_per_gb_month_measured']}/GB-month rate measured "
              "from this account's own charge.")
    print(f"VOLUMES REMAINING: {result['volumes_remaining'] or 'none'}")
    print("IRREVERSIBLE. The data is gone; only the billing was the point.")
    return 0


def main(argv) -> int:
    import runpod_client as rp

    if len(argv) < 2:
        print(f"usage: volume_delete <{LITERAL}:volume_id>")
        return 2
    return report(rp, argv[1])


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv))
