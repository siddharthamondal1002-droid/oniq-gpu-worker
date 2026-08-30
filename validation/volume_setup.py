"""Create ONE network volume and attach it to ONE endpoint. Token-gated.

Owner directive 2026-08-30, section 3: use the persistent/network-volume
mechanism, read its price first, and "use the smallest volume that safely
accommodates Hunyuan and future candidate hydration".

THE PRICE IS MEASURED, NOT ASSUMED. /billing/networkvolumes on this
account reports an hourly `amount` against `diskSpaceBilledGb`; amount x
720 / gb lands on $0.0699999975, i.e. $0.07 per GB-month. That figure came
off a real charge, which is why this module can state a cost instead of an
estimate.

THE RISK, stated plainly because it is the one that can strand the
endpoint: a network volume is datacenter-scoped, and an endpoint that
mounts one can only run where the volume is. If the A5000 is not available
in that datacenter, the endpoint has traded a slow cold pull for no GPU at
all. RunPod exposes no datacenter/GPU-availability document — no REST
path, introspection disabled, field suggestions disabled (all measured
2026-08-30) — so this cannot be verified in advance. It is therefore
DELIBERATELY reversible: attaching sends one field, the before/after
documents are both returned, and detaching is the same call with an empty
string.
"""

from __future__ import annotations

import json
import sys

# Measured from this account's own billing line, not recalled.
USD_PER_GB_MONTH = 0.07

# Hunyuan is 32.26 GiB measured. Plus the hydrator's 4 GiB headroom that is
# 36.26, and the account's billing line suggests a 40 GB floor. 50 GB is
# the smallest round size that clears all three with room for one more
# candidate's partial fetch, and it can be grown later through
# /networkvolumes/{id}/update without recreating it.
DEFAULT_SIZE_GB = 50


class Refused(Exception):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code


def monthly_cost(size_gb: int) -> float:
    return round(size_gb * USD_PER_GB_MONTH, 2)


def existing_volume(client, name: str):
    """A volume of this name already on the account, or None.

    Idempotence: re-running this must not leave two volumes billing for
    the same purpose.
    """
    try:
        _, doc = client._get_json(f"{client.REST_BASE}/networkvolumes")
    except Exception:
        return None
    rows = doc if isinstance(doc, list) else (doc or {}).get("networkVolumes") or []
    for row in rows:
        if isinstance(row, dict) and row.get("name") == name:
            return row
    return None


def apply(client, endpoint_id: str, name: str, size_gb: int,
          datacenter_id: str) -> dict:
    found = existing_volume(client, name)
    if found:
        volume = found
        created = False
    else:
        volume = client.create_network_volume(name, size_gb, datacenter_id)
        created = True

    volume_id = volume.get("id")
    if not volume_id:
        raise Refused(
            "volume-id-missing",
            f"the volume document carries no id: {volume!r}. Attaching "
            "nothing would leave the endpoint exactly as it was while the "
            "run reported success.",
        )

    before, after = client.attach_network_volume(endpoint_id, volume_id)
    landed = after.get("networkVolumeId")
    if landed != volume_id:
        raise Refused(
            "attach-not-applied",
            f"asked to attach {volume_id}, endpoint reports {landed!r}. The "
            "PATCH returned success and did not take effect; do not hydrate "
            "against a volume the endpoint will not mount.",
        )

    moved = {
        key: {"before": before.get(key), "after": after.get(key)}
        for key in ("templateId", "gpuTypeIds", "workersMin", "workersMax",
                    "workersStandby", "idleTimeout", "executionTimeoutMs")
        if before.get(key) != after.get(key)
    }
    if moved:
        raise Refused(
            "collateral-change",
            f"the PATCH moved fields it was not asked to move: {moved}",
        )

    return {
        "volume_id": volume_id,
        "volume_name": name,
        "size_gb": volume.get("size", size_gb),
        "datacenter": volume.get("dataCenterId", datacenter_id),
        "created_now": created,
        "endpoint": endpoint_id,
        "network_volume_id_before": before.get("networkVolumeId") or "",
        "network_volume_id_after": landed,
        "usd_per_gb_month_measured": USD_PER_GB_MONTH,
        "estimated_monthly_usd": monthly_cost(volume.get("size", size_gb)),
    }


def main(argv) -> int:
    import runpod_client as rp

    if len(argv) < 4:
        print("usage: volume_setup <endpoint_id> <name> <datacenter_id> [size_gb]")
        return 2
    endpoint_id, name, datacenter_id = argv[1], argv[2], argv[3]
    size_gb = int(argv[4]) if len(argv) > 4 else DEFAULT_SIZE_GB

    print(f"volume {name!r} {size_gb} GB in {datacenter_id} -> endpoint {endpoint_id}")
    print(f"measured rate ${USD_PER_GB_MONTH}/GB-month "
          f"=> ${monthly_cost(size_gb)}/month")
    try:
        result = apply(rp, endpoint_id, name, size_gb, datacenter_id)
    except Refused as exc:
        print(f"REFUSED {exc}")
        return 1
    print(json.dumps(result, indent=2))
    print()
    print("REVERSIBLE: detach with the same call and an empty volume id.")
    print("A volume is storage, not compute; no GPU job was submitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
