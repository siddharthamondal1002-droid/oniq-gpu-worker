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


# One POST, one place. GraphQL sends READS over POST, and the gate in
# tests/test_workflow_gates.py asserts this is the only one and that it
# goes to GRAPHQL_URL — so a mutation cannot be slipped in beside it.
def _gql(query: str):
    """Send a GraphQL READ. Returns (data, error_text)."""
    status, raw = rp._request(rp.GRAPHQL_URL, method="POST",
                              body={"query": query})
    if status != 200:
        return None, f"HTTP {status} {raw[:300]}"
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError:
        return None, f"unparseable body {raw[:200]!r}"
    if doc.get("errors"):
        # A schema refusing a field NAMES the field that exists. Run 162
        # learned GpuAvailabilityInput has no "gpuTypeId" this way, which
        # is more than a null would ever have said.
        return None, "; ".join(e.get("message", "?") for e in doc["errors"])[:400]
    return doc.get("data"), None


def introspect(type_name: str):
    """Field names on one GraphQL type — when the server allows it.

    RunPod's Apollo server answers INTROSPECTION_DISABLED (measured
    2026-08-30, run 163), so this returns None WITH that reason rather
    than pretending the type does not exist. Kept, and kept honest about
    why, because the alternative — writing queries against remembered
    field names — is what the "no third-party claims where official
    documentation answers" rule exists to prevent, and a server's own
    error message is the closest thing to documentation available here.
    """
    data, err = _gql(
        '{ __type(name: "%s") { fields { name } inputFields { name } } }'
        % type_name
    )
    if err:
        return None, err
    if not data or not data.get("__type"):
        return None, f"{type_name} not in the schema"
    node = data["__type"]
    names = [f["name"] for f in (node.get("fields") or [])]
    names += [f["name"] for f in (node.get("inputFields") or [])]
    return sorted(names), None


def endpoint_locations():
    """Any location or datacenter the ENDPOINT document itself names.

    With introspection off this is the practical route to the feasibility
    question: a volume must be created in a datacenter the endpoint can
    actually run in, and the endpoint knows where it runs. Every key is
    reported, so a field this code does not anticipate stays visible
    instead of being silently dropped.
    """
    try:
        _, doc = rp._get_json(f"{rp.REST_BASE}/endpoints")
    except rp.RunPodApiError as exc:
        return None, str(exc)
    rows = doc if isinstance(doc, list) else doc.get("endpoints") or []
    out = []
    for e in rows:
        if not isinstance(e, dict):
            continue
        out.append({
            "id": e.get("id"),
            "keys": sorted(e),
            "locations": e.get("locations"),
            "dataCenterIds": e.get("dataCenterIds"),
            "networkVolumeId": e.get("networkVolumeId"),
            "gpuTypeIds": e.get("gpuTypeIds"),
        })
    return out, None


def derived_rate(billing):
    """The per-GB-month rate, DERIVED from a real charge on this account.

    /billing/networkvolumes reports an hourly `amount` against
    `diskSpaceBilledGb`. amount * 720 / gb lands on 0.06999999750 for the
    2026-08-30 line — $0.07/GB/month on a 720-hour month. That is a
    measured figure rather than one recalled from memory, which is the
    whole reason this probe refused to state a rate before now.

    Returns None when the payload is not the shape that reasoning
    assumes: a rate inferred from an unexpected document is worse than no
    rate, because the migration would be sized on it.
    """
    if not isinstance(billing, list) or not billing:
        return None
    row = billing[0]
    if not isinstance(row, dict):
        return None
    amount, gb = row.get("amount"), row.get("diskSpaceBilledGb")
    if not isinstance(amount, (int, float)) or not isinstance(gb, (int, float)):
        return None
    if gb <= 0:
        return None
    return {
        "usd_per_gb_month_720h": round(amount * 720 / gb, 6),
        "measured_from": row,
    }


def spec_paths(doc: dict) -> list[str] | None:
    """Every path the REST spec declares.

    Printed because two runs each spent a round trip discovering that a
    path guessed from documentation is not there — /datacenters is not a
    REST path at all. rest.runpod.io is unreachable from the dev
    container, so every guess costs a CI run; the document names its own
    paths, and reading them is one run instead of however many guesses.
    """
    paths = doc.get("paths") if isinstance(doc, dict) else None
    return sorted(paths) if isinstance(paths, dict) else None


def volume_billing():
    """What the account is ACTUALLY charged for network volumes.

    /billing/networkvolumes is declared by the spec, and a measured charge
    against the 10 GB volume already on the account beats any per-GB rate
    recalled from memory. Returns (rows, error).
    """
    try:
        _, doc = rp._get_json(f"{rp.REST_BASE}/billing/networkvolumes")
    except rp.RunPodApiError as exc:
        return None, str(exc)
    return doc, None


def datacenters():
    """Datacenter rows from GraphQL, with the field list read first.

    REST is not asked: its OpenAPI document declares no datacenters path,
    which run 162 established by printing the paths rather than guessing
    at two more.
    """
    fields, err = introspect("DataCenter")
    if not fields:
        return None, None, f"DataCenter introspection failed: {err}"
    wanted = [f for f in ("id", "name", "storageSupport", "listed",
                          "gpuAvailability") if f in fields]
    if "id" not in wanted:
        return None, None, f"DataCenter has no id field; has {fields}"
    selection = " ".join(f for f in wanted if f != "gpuAvailability")
    data, err = _gql("{ dataCenters { %s } }" % selection)
    if err or not data:
        return None, None, f"dataCenters query failed: {err}"
    rows = data.get("dataCenters") or []
    if not rows:
        return None, None, "dataCenters returned no rows"
    keys = sorted(set().union(*(set(r) for r in rows if isinstance(r, dict))))
    return rows, keys, f"GraphQL dataCenters (fields available: {fields})"


def _row_id(row: dict):
    for key in ("id", "dataCenterId", "name"):
        if row.get(key):
            return str(row[key])
    return None


def volume_capable(rows) -> list[str] | None:
    """Datacenter ids whose own row says they support network storage.

    None when no row carries such a field — an absent field is not a No,
    and must not be reported as one.
    """
    if not rows:
        return None
    keys = ("storageSupport", "storage", "networkStorageSupport",
            "supportsNetworkVolume", "networkVolumeSupport")
    found, saw_field = [], False
    for row in rows:
        if not isinstance(row, dict):
            continue
        for k in keys:
            if k in row:
                saw_field = True
                if row[k]:
                    ident = _row_id(row)
                    if ident:
                        found.append(ident)
                break
    return sorted(set(found)) if saw_field else None


def a5000_datacenters(rows, gpu_id: str = "NVIDIA RTX A5000"):
    """Where the A5000 is actually listed, from the datacenter rows.

    Returns (ids or None, note). None means the rows do not carry GPU
    availability — a finding to report, never a licence to assume the card
    is everywhere.
    """
    if not rows:
        return None, "no datacenter document to read"
    keys = ("gpuAvailability", "gpuTypes", "gpuTypeIds", "gpus")
    found, saw_field = [], False
    for row in rows:
        if not isinstance(row, dict):
            continue
        for k in keys:
            if k not in row:
                continue
            saw_field = True
            blob = json.dumps(row[k]).lower()
            if "a5000" in blob:
                ident = _row_id(row)
                if ident:
                    found.append(ident)
            break
    if not saw_field:
        return None, f"datacenter rows carry no GPU field (keys seen: {sorted(rows[0])})"
    return sorted(set(found)), "from the datacenter rows"


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
    rows, row_keys, dc_note = datacenters()
    dcs = volume_capable(rows)
    if dcs is None and doc:
        dcs = volume_datacenters(doc, raw)
    a5000_dcs, a5000_note = a5000_datacenters(rows)
    overlap = sorted(set(dcs) & set(a5000_dcs)) if dcs and a5000_dcs else None
    return {
        "spec_url": url,
        "create_schema": _create_schema(doc) if doc else None,
        "existing_volumes": vols,
        "existing_volumes_error": vol_err,
        "datacenter_row_keys": row_keys,
        "datacenter_source": dc_note,
        "rest_paths": spec_paths(doc) if doc else None,
        "datacenter_count": len(rows) if rows else None,
        "volume_datacenters": dcs,
        "a5000_datacenters": a5000_dcs,
        "a5000_datacenter_note": a5000_note,
        "datacenter_overlap": overlap,
        "storage_rate_fields_in_spec": storage_rate(raw),
        "volume_billing": volume_billing()[0],
        "volume_billing_error": volume_billing()[1],
        "introspection_note": introspect("GpuAvailabilityInput")[1],
        "endpoint_locations": endpoint_locations()[0],
        "derived_rate": derived_rate(volume_billing()[0]),
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
    print(f"  datacenter rows    : {s['datacenter_count']}, keys={s['datacenter_row_keys']}")
    print(f"  datacenter source  : {s['datacenter_source']}")
    print(f"  REST paths declared: {s['rest_paths']}")
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
    print(f"BILLING (/billing/networkvolumes): {s['volume_billing'] if s['volume_billing'] is not None else s['volume_billing_error']}")
    print(f"introspection : {s['introspection_note']}")
    print(f"ENDPOINTS    : {s['endpoint_locations']}")
    print(f"DERIVED RATE : {s['derived_rate']}")
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
