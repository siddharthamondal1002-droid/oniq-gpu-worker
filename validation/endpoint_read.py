"""Read ONE endpoint from BOTH REST routes and print where they disagree.

$0. Every call is a GET; there is no write path in this module at all.

WHY THIS EXISTS
---------------
On 2026-09-01 the volume run refused with `datacenter-not-pinned`, and the
two facts it rested on both came from a single read of one route. Thirty
minutes later the OTHER route described the same endpoint differently:

  /endpoints/{id}   (read by the volume path, 05:00:55)
      networkVolumeId == the 50 GB 'oniq-models' volume, or the run would
      have refused `attach-not-applied` instead and never reached the
      datacenter check.

  /endpoints        (read by volume_probe, 05:30:24)
      networkVolumeId == 'vhxqqd8vhj' — a DIFFERENT, 10 GB volume, and
      `dataCenterIds` absent from the key list entirely.

One of those is not what the endpoint is. Which one decides whether a
50 GB volume is attached to production or orphaned, and whether the
datacenter refusal was a real finding or a field the API simply never
returns. Guessing between them would put a write on production founded
on the guess.

WHAT THE KEY LIST ALREADY SHOWS. The list route returns no `dataCenterIds`
key at all — not null, absent. A check that refuses on
`after.get("dataCenterIds")` being falsy therefore cannot pass on that
route no matter what the endpoint is actually pinned to. Whether the
detail route carries the field is exactly what this module reads, because
a guard that can never pass is not a guard, it is a wall.

NAMES AND IDS, NEVER VALUES. The listing discipline in this repo is that a
read has no reason to pull a secret across the wire. The endpoint document
is configuration, not env — env lives on the template — but the detail
route may carry more keys than the list, so anything whose NAME looks like
a credential is reported by type and length and never by value.
"""

from __future__ import annotations

import json
import re
import sys

SECRETISH = re.compile(
    r"(?i)(key|token|secret|password|passwd|auth|credential|bearer|env)"
)

# The fields the deployment actually turns on. Printed side by side so a
# disagreement between the routes is visible rather than inferred.
DECIDING = (
    "networkVolumeId",
    "networkVolumeIds",
    "dataCenterIds",
    "templateId",
    "executionTimeoutMs",
    "gpuTypeIds",
    "workersMin",
    "workersMax",
    "workersStandby",
    "idleTimeout",
)


def redact(key: str, value):
    """A value safe to print, or a description of one that is not."""
    if SECRETISH.search(key):
        return f"<redacted {type(value).__name__} len={len(str(value))}>"
    if isinstance(value, dict):
        return {k: redact(k, v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(key, v) for v in value]
    return value


def safe(doc):
    if not isinstance(doc, dict):
        return doc
    return {k: redact(k, v) for k, v in doc.items()}


def detail(client, endpoint_id: str):
    """GET /endpoints/{id} — the route the volume path reads."""
    try:
        _, doc = client._get_json(f"{client.REST_BASE}/endpoints/{endpoint_id}")
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return doc, None


def listed(client, endpoint_id: str):
    """The matching row from GET /endpoints — the route the probe reads."""
    try:
        _, doc = client._get_json(f"{client.REST_BASE}/endpoints")
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    rows = doc if isinstance(doc, list) else (doc or {}).get("endpoints") or []
    for row in rows:
        if isinstance(row, dict) and row.get("id") == endpoint_id:
            return row, None
    return None, f"{endpoint_id} not among the {len(rows)} endpoint(s) listed"


def volume(client, volume_id: str):
    """GET /networkvolumes/{id} — where a volume actually lives."""
    if not volume_id:
        return None, "no volume id"
    try:
        _, doc = client._get_json(
            f"{client.REST_BASE}/networkvolumes/{volume_id}"
        )
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return doc, None


def disagreements(a: dict, b: dict) -> dict:
    """Every key where the two documents differ, INCLUDING presence.

    Absence is a value here: a key one route returns and the other omits is
    the whole finding, and reporting it as `None == None` would hide it.
    """
    out = {}
    for key in sorted(set(a) | set(b)):
        in_a, in_b = key in a, key in b
        va, vb = a.get(key), b.get(key)
        if in_a == in_b and va == vb:
            continue
        out[key] = {
            "detail": redact(key, va) if in_a else "<ABSENT>",
            "list": redact(key, vb) if in_b else "<ABSENT>",
        }
    return out


def report(client, endpoint_id: str) -> int:
    d, d_err = detail(client, endpoint_id)
    l, l_err = listed(client, endpoint_id)

    print("=" * 68)
    print(f"ENDPOINT {endpoint_id} — read from BOTH routes, $0")
    print("=" * 68)
    print()
    print(f"GET /endpoints/{endpoint_id}")
    if d is None:
        print(f"  UNREADABLE: {d_err}")
    else:
        print(json.dumps(safe(d), indent=2, sort_keys=True))
    print()
    print("GET /endpoints  (the matching row)")
    if l is None:
        print(f"  UNREADABLE: {l_err}")
    else:
        print(json.dumps(safe(l), indent=2, sort_keys=True))
    print()

    print("-" * 68)
    print("THE FIELDS THE DEPLOYMENT TURNS ON")
    print(f"  {'field':<20} {'detail route':<34} list route")
    for key in DECIDING:
        dv = "<ABSENT>" if not isinstance(d, dict) or key not in d else repr(d[key])
        lv = "<ABSENT>" if not isinstance(l, dict) or key not in l else repr(l[key])
        print(f"  {key:<20} {dv[:33]:<34} {lv[:33]}")
    print()

    if isinstance(d, dict) and isinstance(l, dict):
        diff = disagreements(d, l)
        print("-" * 68)
        if diff:
            print(f"ROUTES DISAGREE on {len(diff)} key(s):")
            print(json.dumps(diff, indent=2, sort_keys=True))
        else:
            print("ROUTES AGREE on every key.")
        print()

    # WHERE EACH NAMED VOLUME ACTUALLY IS. A volume is datacenter-scoped, so
    # the id alone does not say whether the endpoint can run beside it.
    ids = []
    for doc in (d, l):
        if not isinstance(doc, dict):
            continue
        one = doc.get("networkVolumeId")
        if one and one not in ids:
            ids.append(one)
        for many in doc.get("networkVolumeIds") or []:
            if many and many not in ids:
                ids.append(many)
    print("-" * 68)
    if not ids:
        print("VOLUMES: neither route names a network volume on this endpoint.")
    for vid in ids:
        vdoc, verr = volume(client, vid)
        if vdoc is None:
            print(f"VOLUME {vid}: unreadable — {verr}")
        else:
            print(f"VOLUME {vid}: {json.dumps(safe(vdoc), sort_keys=True)}")
    print()
    print("=" * 68)
    print("$0 — every call above was a GET. Nothing was created or changed.")
    return 0


def main(argv) -> int:
    import runpod_client as rp

    if len(argv) < 2 or not argv[1]:
        print("usage: endpoint_read <endpoint_id>")
        return 2
    return report(rp, argv[1])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
