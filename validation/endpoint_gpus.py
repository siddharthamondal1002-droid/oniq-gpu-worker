"""Widen ONE endpoint's gpuTypeIds to the cards the owner has approved.

WHY THIS EXISTS. On 2026-08-30 endpoint 7disu6my0mloco sat with
`workers {idle 0, initializing 0, ready 0, running 0, throttled 1}` and
`gpuTypeIds ["NVIDIA RTX A6000"]`. Throttled means RunPod wants to place
a worker and cannot get the card. One card named and no fallback is a
single point of failure for the whole product: every story still, every
motion clip and every video goes through this endpoint, and none of them
run while the one approved card is unavailable.

OWNER DIRECTIVE 2026-08-30: the endpoint may use RTX A6000 and A40, and
nothing else. Both are 48 GB, so a model that fits today still fits — the
fallback changes where a job lands and roughly how long it takes, never
whether it can run at all. A 24 GB or 32 GB card would silently turn "a
bit slower" into "out of memory", which is why the approved set is these
two and why it lives here as a constant rather than as a dispatch input.

THE NAMES ARE RESOLVED, NEVER GUESSED. RunPod's canonical id is "NVIDIA
RTX A6000", while its display name is "A6000 48GB" — and the field takes
the former. A wrong string is accepted into the endpoint and then has
nowhere legal to schedule, which is precisely the shape of the US-MO-2
network-volume failure that cost ninety minutes the same day: an illegal
value in a field that reports success. So every card is looked up in the
LIVE catalogue before anything is written, and an unresolvable name
refuses the whole write rather than sending part of it.

This CANNOT increase the worker count — workersMax still bounds that, and
this module never touches it. It widens where one worker may be placed.
"""

from __future__ import annotations

import json

# The approved set, as canonical RunPod ids. A MODULE CONSTANT and not a
# workflow input, deliberately: which card the product runs on is a cost
# decision that belongs to the owner (CLAUDE.md, 2026-08-14), and a typed
# dispatch field is exactly how such a decision gets made by accident.
# test_workflow_gates asserts no gpu input reaches this.
WANTED = ("NVIDIA RTX A6000", "NVIDIA A40")

# Fields a gpuTypeIds PATCH must not move. networkVolumeId and
# dataCenterIds are here for the reason endpoint_bounds records: moving
# either relocates where the endpoint may run.
GUARDED = (
    "templateId",
    "workersMin",
    "workersMax",
    "idleTimeout",
    "executionTimeoutMs",
    "networkVolumeId",
    "dataCenterIds",
)


class Refused(Exception):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _norm(text) -> str:
    """Compare names without punctuation or case getting in the way."""
    return "".join(ch for ch in str(text or "").lower() if ch.isalnum())


def resolve(catalogue, wanted=WANTED) -> dict:
    """Map each approved name to the id RunPod actually uses.

    Exact `id` first, because that is the field gpuTypeIds carries. Only
    if that misses does it fall back to a normalized comparison against
    id and display_name, so "A40" still resolves whether the provider
    calls it "NVIDIA A40" or "A40 48GB".

    AMBIGUITY REFUSES. Two catalogue rows matching one name means the
    guess would decide which card the product runs on, and that is not a
    guess this module is allowed to make.
    """
    if not catalogue:
        raise Refused(
            "catalogue-unreadable",
            "the GPU catalogue came back empty — that is UNKNOWN, not "
            "'no cards exist', and nothing may be written against it",
        )
    by_id = {}
    for row in catalogue:
        if isinstance(row, dict) and row.get("id"):
            by_id[row["id"]] = row

    resolved, missing = {}, []
    for name in wanted:
        if name in by_id:
            resolved[name] = by_id[name]
            continue
        target = _norm(name)
        hits = [
            row for row in by_id.values()
            if _norm(row.get("id")) == target
            or _norm(row.get("display_name")) == target
        ]
        if len(hits) == 1:
            resolved[name] = hits[0]
        elif len(hits) > 1:
            raise Refused(
                "gpu-ambiguous",
                f"{name!r} matches {sorted(h.get('id') for h in hits)} — "
                "refusing to pick which card the product runs on",
            )
        else:
            missing.append(name)

    if missing:
        raise Refused(
            "gpu-unknown",
            f"the live catalogue does not list {missing}. Writing a name "
            "RunPod does not know puts a string in the field that the "
            "scheduler can never satisfy — the same failure as a "
            "datacenter absent from the dataCenterIds enum.",
        )
    return resolved


def apply(client, endpoint_id: str) -> dict:
    if not endpoint_id:
        raise Refused("endpoint-missing", "a blank endpoint id names nothing")

    _, catalogue = client.gpu_catalogue()
    resolved = resolve(catalogue)
    target = [resolved[name]["id"] for name in WANTED]

    _, before = client.get_endpoint(endpoint_id)
    was = before.get("gpuTypeIds") or []

    # A card outside the approved set means somebody chose it deliberately
    # and their choice is not recorded here. Overwriting it silently would
    # be this module deciding a cost question on its own.
    stray = [g for g in was if g not in target]
    if stray:
        raise Refused(
            "unapproved-card-present",
            f"the endpoint already names {stray}, which is outside the "
            f"owner-approved set {list(WANTED)}. Someone chose that; this "
            "module will not overwrite a decision it cannot read.",
        )

    if set(was) == set(target):
        return {
            "endpoint": endpoint_id,
            "gpus_before": was, "gpus_after": was,
            "already_conformed": True,
            "prices": {g: resolved_price(resolved, g) for g in target},
            "guarded_fields_unchanged": list(GUARDED),
        }

    status, raw = client.set_endpoint_gpu_types(endpoint_id, target)
    if status not in (200, 201, 202):
        raise Refused(
            "patch-refused",
            f"PATCH /endpoints/{endpoint_id} -> {status} (body: {raw[:800]!r})",
        )

    _, after = client.get_endpoint(endpoint_id)
    landed = after.get("gpuTypeIds") or []
    if set(landed) != set(target):
        raise Refused(
            "not-applied",
            f"asked for {target}, the endpoint reports {landed!r}. The "
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
        "gpus_before": was, "gpus_after": landed,
        "already_conformed": False,
        "prices": {g: resolved_price(resolved, g) for g in target},
        "guarded_fields_unchanged": list(GUARDED),
    }


def resolved_price(resolved: dict, gpu_id: str):
    """What the catalogue quotes for a card, so the log records what was
    authorised at the moment it was authorised. Quoted, never recalled —
    the same rule the live-price table keeps."""
    for row in resolved.values():
        if row.get("id") == gpu_id:
            return {
                "secure": row.get("secure_price"),
                "community": row.get("community_price"),
                "on_demand": row.get("on_demand_price"),
            }
    return None


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
    print("=== SUMMARY ===")
    if result["already_conformed"]:
        print(f"ALREADY {result['gpus_after']} — no PATCH sent, so no worker "
              "was restarted")
        return 0, result
    print(f"gpuTypeIds {result['gpus_before']} -> {result['gpus_after']}, "
          "confirmed by re-reading the endpoint")
    print(f"UNCHANGED: {', '.join(GUARDED)}")
    print("This widens WHERE one worker may be placed. workersMax still "
          "bounds HOW MANY, and this module never touches it.")
    return 0, result


def main(argv) -> int:
    import runpod_client

    code, _ = report(runpod_client, argv[1] if len(argv) > 1 else "")
    return code


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
