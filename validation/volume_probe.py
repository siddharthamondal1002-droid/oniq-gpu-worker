"""Read-only survey of RunPod network volumes. $0, no GPU, no writes.

Why this exists
---------------
Owner directive, 2026-08-30: move the baked model weights out of the
container image and onto a network volume. That is a spend decision the
owner has taken, but the RATE is not something to assume — and neither is
the constraint that actually decides whether the plan is possible at all.

A network volume is DATACENTER-SCOPED. An endpoint that mounts one can
only ever run in that one datacenter. The A5000 must therefore be
available THERE, or the migration trades a slow cold pull for an endpoint
that cannot get a card. That is the answer this probe exists to give, and
it is why the datacenter cross-check is not optional decoration.

Everything here is a GET. Nothing is created, patched or deleted; the
module deliberately has no write path, so it cannot become a spend mode
by mistake.
"""

from __future__ import annotations

import json
import re

import runpod_client as rp


def _spec():
    """The REST OpenAPI document, or None. Public — sent without auth."""
    for url in (f"{rp.REST_BASE}/openapi.json", "https://rest.runpod.io/openapi.json"):
        try:
            status, raw = rp._request(url, bearer=False)
        except rp.RunPodApiError:
            continue
        if status != 200 or not raw.lstrip().startswith("{"):
            continue
        try:
            return url, json.loads(raw), raw
        except json.JSONDecodeError:
            continue
    return None, None, ""


def _create_schema(doc: dict) -> dict | None:
    """What POST /networkvolumes actually accepts, from the spec itself.

    Read rather than assumed: run #34's 'not in input schema' refusal is
    the standing reminder that a field name guessed from documentation is
    a field name that gets rejected live.
    """
    try:
        schema = doc["paths"]["/networkvolumes"]["post"]["requestBody"][
            "content"]["application/json"]["schema"]
    except (KeyError, TypeError):
        return None
    ref = schema.get("$ref")
    if ref and ref.startswith("#/components/schemas/"):
        schema = doc.get("components", {}).get("schemas", {}).get(
            ref.rsplit("/", 1)[1], {})
    props = schema.get("properties")
    return {
        "properties": sorted(props) if isinstance(props, dict) else None,
        "required": sorted(schema.get("required") or []) or None,
    }


def existing_volumes():
    """Every network volume on the account. [] is a real answer; None is
    'could not read', and the two must never be conflated."""
    try:
        raw, doc = rp._get_json(f"{rp.REST_BASE}/networkvolumes")
    except rp.RunPodApiError as exc:
        return None, str(exc)
    rows = doc if isinstance(doc, list) else doc.get("networkVolumes") or []
    return [
        {
            "id": v.get("id"),
            "name": v.get("name"),
            "size_gb": v.get("size"),
            "datacenter": v.get("dataCenterId"),
        }
        for v in rows
    ], None


def volume_datacenters(doc: dict, raw: str) -> list[str] | None:
    """Datacenter ids the spec admits for a network volume, if it enumerates
    them. An enum is authoritative; a free-form string is not, and returns
    None rather than a guess."""
    try:
        schema = doc["components"]["schemas"]
    except (KeyError, TypeError):
        return None
    for name, body in schema.items():
        if "datacenter" not in name.lower():
            continue
        values = body.get("enum")
        if values:
            return sorted(str(v) for v in values)
    # Fall back to the create schema's own dataCenterId field.
    try:
        post = doc["paths"]["/networkvolumes"]["post"]["requestBody"][
            "content"]["application/json"]["schema"]
        ref = post.get("$ref")
        if ref:
            post = schema[ref.rsplit("/", 1)[1]]
        field = post.get("properties", {}).get("dataCenterId", {})
        ref = field.get("$ref")
        if ref:
            field = schema[ref.rsplit("/", 1)[1]]
        values = field.get("enum")
        if values:
            return sorted(str(v) for v in values)
    except (KeyError, TypeError):
        pass
    return None


def a5000_datacenters(gpu_id: str = "NVIDIA RTX A5000"):
    """Where the A5000 can actually be had, straight from the catalogue.

    Returned as (datacenter ids or None, raw availability note). None means
    the catalogue does not carry per-datacenter availability — which is a
    finding, not a licence to assume the card is everywhere.
    """
    try:
        parsed = rp.gpu_catalogue()
    except rp.RunPodApiError as exc:
        return None, f"catalogue unreadable: {exc}"
    rows = parsed.get("rest") if isinstance(parsed, dict) else parsed
    if not isinstance(rows, list):
        return None, "catalogue shape unrecognised"
    for g in rows:
        if not isinstance(g, dict):
            continue
        if gpu_id.lower() not in str(g.get("id", "")).lower() \
           and gpu_id.lower() not in str(g.get("displayName", "")).lower():
            continue
        for key in ("dataCenterIds", "dataCenters", "datacenterIds"):
            if isinstance(g.get(key), list) and g[key]:
                return sorted(str(d) for d in g[key]), f"from catalogue.{key}"
        return None, "catalogue row found, no per-datacenter field on it"
    return None, f"{gpu_id!r} not present in the catalogue"


def storage_rate(raw_spec: str):
    """Any per-GB storage price the API itself states. Deliberately returns
    None when the API is silent: a rate quoted from memory is exactly the
    kind of invented evidence this whole module refuses to produce."""
    hits = sorted(set(re.findall(r'"(\w*[Ss]torage\w*(?:Cost|Price|Rate)\w*)"',
                                 raw_spec)))
    return hits or None


def survey() -> dict:
    url, doc, raw = _spec()
    vols, vol_err = existing_volumes()
    dcs = volume_datacenters(doc, raw) if doc else None
    a5000_dcs, a5000_note = a5000_datacenters()
    overlap = None
    if dcs and a5000_dcs:
        overlap = sorted(set(dcs) & set(a5000_dcs))
    return {
        "spec_url": url,
        "create_schema": _create_schema(doc) if doc else None,
        "existing_volumes": vols,
        "existing_volumes_error": vol_err,
        "volume_datacenters": dcs,
        "a5000_datacenters": a5000_dcs,
        "a5000_datacenter_note": a5000_note,
        "datacenter_overlap": overlap,
        "storage_rate_fields_in_spec": storage_rate(raw),
    }


def report() -> int:
    s = survey()
    print(json.dumps(s, indent=2, sort_keys=True))
    print()
    print("=" * 68)
    if s["create_schema"] is None:
        print("NETWORK VOLUMES: the REST spec does not describe POST /networkvolumes.")
        print("  A volume cannot be created through this client until it does.")
    else:
        print(f"CREATE SCHEMA: required={s['create_schema']['required']}")
    if s["existing_volumes"] is None:
        print(f"EXISTING: unreadable - {s['existing_volumes_error']}")
    else:
        print(f"EXISTING: {len(s['existing_volumes'])} volume(s) on the account")
        for v in s["existing_volumes"]:
            print(f"  {v['id']}  {v['name']!r}  {v['size_gb']} GB  {v['datacenter']}")
    print()
    print("DATACENTER CONSTRAINT — the one that decides feasibility:")
    print(f"  volumes offered in : {s['volume_datacenters']}")
    print(f"  A5000 available in : {s['a5000_datacenters']}  ({s['a5000_datacenter_note']})")
    print(f"  overlap            : {s['datacenter_overlap']}")
    if s["datacenter_overlap"] == []:
        print("  VERDICT: NO OVERLAP. Mounting a volume would strand the endpoint")
        print("           in a datacenter with no A5000. Do not proceed.")
    elif s["datacenter_overlap"]:
        print("  VERDICT: feasible - pin the endpoint to one of the overlap ids.")
    else:
        print("  VERDICT: UNRESOLVED from the API. Neither side enumerates its")
        print("           datacenters here, so feasibility is not yet proven.")
    print()
    if s["storage_rate_fields_in_spec"]:
        print(f"RATE: spec carries {s['storage_rate_fields_in_spec']}")
    else:
        print("RATE: the API states no storage price. NOT INVENTED HERE -")
        print("      the owner is told to read it off the RunPod console.")
    print("=" * 68)
    print("$0 - every call above was a GET. Nothing was created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(report())
