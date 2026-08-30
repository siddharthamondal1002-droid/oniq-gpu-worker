"""RunPod harness client — CI-side, never shipped in the worker image.

DISCOVER FIRST. Every fetch here returns (raw_text, parsed) so the parser
can be judged against the provider's actual bytes: these payload shapes
were originally written without ever reaching RunPod, and a wrong field
name does not raise — it yields a null price, which reads as "no
capacity", which reads as "the target GPU is unavailable". The read-only
discover mode exists to correct this file for free and should be expected
to find at least one error until the recorded fixtures say otherwise.

Rules built in:
- stdlib only (urllib), so the harness needs no extra installs;
- the API key is read from RUNPOD_API_KEY and never printed;
- discover is READ-ONLY: GETs and one GraphQL query, nothing else;
- there is no create-endpoint and no create-pod function at all — CI
  verifies an endpoint's configuration, it never authors one;
- the orphan sweep returns None when it cannot reach the API: "cannot
  confirm terminated" and "confirmed terminated" are never the same value.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

GRAPHQL_URL = "https://api.runpod.io/graphql"
REST_BASE = "https://rest.runpod.io/v1"
SERVERLESS_BASE = "https://api.runpod.ai/v2"

GPU_CATALOGUE_QUERY = """
query GpuTypes {
  gpuTypes {
    id
    displayName
    memoryInGb
    secureCloud
    communityCloud
    securePrice
    communityPrice
    lowestPrice(input: {gpuCount: 1}) {
      uninterruptablePrice
      minimumBidPrice
    }
  }
}
"""


class RunPodApiError(Exception):
    pass


def _api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise RunPodApiError("RUNPOD_API_KEY is not set in this environment")
    return key


def _request(url: str, *, method: str = "GET", body=None, timeout: int = 30, bearer: bool = True):
    payload = None
    # Cloudflare fronts api.runpod.io and bans urllib's default agent
    # signature outright (error code 1010, measured 2026-08-25) — the
    # same key succeeded from curl. Identify honestly, but as a real
    # client.
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "oniq-gpu-validation/1.0 (github-actions)",
    }
    if bearer:
        headers["Authorization"] = f"Bearer {_api_key()}"
    if body is not None:
        payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, raw
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return exc.code, raw
    except Exception as exc:
        raise RunPodApiError(f"request failed: {type(exc).__name__}") from exc


def _get_json(url: str):
    status, raw = _request(url)
    if status != 200:
        raise RunPodApiError(f"GET {url.split('?')[0]} -> {status}")
    return raw, json.loads(raw)


# ---------------------------------------------------------------- catalogue


def gpu_catalogue():
    """The GPU catalogue. Returns (raw_text, parsed_list).

    Three auth/transport attempts, because RunPod keys differ in what
    they may call (measured 2026-08-25: a key that lists pods over REST
    got 403 from GraphQL — restricted keys can lack GraphQL permission):
    GraphQL with Bearer auth, GraphQL with the api_key query parameter,
    then REST /gputypes. Whichever source answers, its raw bytes come
    back verbatim and the parser is judged against them. Error text
    carries response-body snippets (bodies never contain the key); the
    query-parameter URL is never printed anywhere.
    """
    body = {"query": GPU_CATALOGUE_QUERY}
    status, raw = _request(GRAPHQL_URL, method="POST", body=body)
    if status in (401, 403):
        status, raw = _request(
            f"{GRAPHQL_URL}?api_key={_api_key()}",
            method="POST",
            body=body,
            bearer=False,
        )
    if status == 200:
        doc = json.loads(raw)
        if doc.get("errors"):
            raise RunPodApiError(
                "graphql gpuTypes returned errors: "
                + "; ".join(e.get("message", "?") for e in doc["errors"])
            )
        return raw, [parse_gpu_type(g) for g in doc["data"]["gpuTypes"]]

    graphql_status, graphql_raw = status, raw
    rest_status, rest_raw = _request(f"{REST_BASE}/gputypes")
    if rest_status == 200:
        doc = json.loads(rest_raw)
        if isinstance(doc, list):
            entries = doc
        else:
            entries = doc.get("gpuTypes") or doc.get("data") or []
        return rest_raw, [parse_gpu_type(g) for g in entries]

    raise RunPodApiError(
        "gpu catalogue unavailable — "
        f"graphql -> {graphql_status} (body: {graphql_raw[:200]!r}); "
        f"rest /gputypes -> {rest_status} (body: {rest_raw[:200]!r}). "
        "Read the body snippets: a Cloudflare 'error code: 1010' is a "
        "client-signature block, while a RunPod auth error means the key "
        "lacks GraphQL permission (an owner toggle in the RunPod console)."
    )


def parse_gpu_type(g: dict) -> dict:
    """Normalize one gpuTypes entry. A missing price parses as None —
    which admission reads as NO CAPACITY, never as free."""
    lowest = g.get("lowestPrice") or {}
    return {
        "id": g.get("id"),
        "display_name": g.get("displayName"),
        "memory_gb": g.get("memoryInGb"),
        "secure_cloud": bool(g.get("secureCloud")),
        # Tri-state, deliberately: True/False only when the provider sent
        # an explicit boolean; None when the field is missing or another
        # type. Admission's community/secure-only branching keys on this
        # (owner directive 2026-08-26), and a missing field must reject
        # conservatively — never impersonate an explicit False.
        "community_cloud": (
            g.get("communityCloud")
            if isinstance(g.get("communityCloud"), bool)
            else None
        ),
        "secure_price": g.get("securePrice"),
        "community_price": g.get("communityPrice"),
        "on_demand_price": lowest.get("uninterruptablePrice"),
        "spot_price": lowest.get("minimumBidPrice"),
    }


def find_gpu(parsed_catalogue, gpu_id: str):
    """Match on the `id` field, which carries the canonical full name
    ("NVIDIA GeForce RTX 3090") and is what endpoint gpuTypeIds use.
    `displayName` is the short marketing name ("RTX 3090") — the parser
    originally assumed the opposite, and the 2026-08-25 raw payload in
    tests/fixtures is the regression evidence."""
    for gpu in parsed_catalogue:
        if gpu["id"] == gpu_id:
            return gpu
    return None


# ---------------------------------------------------------------- REST


def get_pods():
    return _get_json(f"{REST_BASE}/pods")


def get_endpoints():
    return _get_json(f"{REST_BASE}/endpoints")


def get_endpoint(endpoint_id: str):
    return _get_json(f"{REST_BASE}/endpoints/{endpoint_id}")


def get_template(template_id: str):
    return _get_json(f"{REST_BASE}/templates/{template_id}")


def set_workers_standby_zero(endpoint_id: str):
    """The harness's ONLY endpoint mutation — owner directive 2026-08-26
    (five-video battery, Phase 2: NO JOB = NO GPU WORKER). Sets
    workersStandby to the literal 0, a strict spend REDUCTION. There is
    deliberately no value parameter: this function cannot scale anything
    up, and no other field is ever named in either transport's body, so
    nothing else about the endpoint can change. Callers must re-read the
    endpoint afterwards — no write echo is trusted.

    Two transports, both measured 2026-08-26: REST PATCH first (run #34
    answered 400 'workersStandby not in input schema' — the route exists
    but does not carry this field), then a GraphQL saveEndpoint with the
    minimal partial input {id, workersStandby: 0} — the same partial-
    input shape RunPod's own SDK uses for update_endpoint_template. A
    failure returns both transports' statuses and bodies verbatim, so
    the next refusal diagnoses itself."""
    rest_status, rest_raw = _request(
        f"{REST_BASE}/endpoints/{endpoint_id}",
        method="PATCH",
        body={"workersStandby": 0},
    )
    if rest_status in (200, 201):
        return rest_status, rest_raw
    body = {
        "query": (
            "mutation SetStandbyZero($id: String!) { "
            "saveEndpoint(input: {id: $id, workersStandby: 0}) "
            "{ id workersStandby } }"
        ),
        "variables": {"id": endpoint_id},
    }
    g_status, g_raw = _request(GRAPHQL_URL, method="POST", body=body)
    if g_status == 200:
        try:
            doc = json.loads(g_raw)
        except json.JSONDecodeError:
            doc = {}
        if not doc.get("errors") and (doc.get("data") or {}).get("saveEndpoint"):
            return 200, g_raw
    return g_status, (
        f"rest patch -> {rest_status} (body: {rest_raw[:200]!r}); "
        f"graphql saveEndpoint -> {g_status} (body: {g_raw[:300]!r})"
    )


def template_env_names_graphql(template_id: str):
    """Env var NAMES on a template, via GraphQL (values are fetched by
    the API but only names ever leave this function). Returns a set, or
    None when the answer is unknown — never an empty set for 'could not
    look'."""
    query = 'query { myself { podTemplates { id env { key value } } } }'
    try:
        status, raw = _request(GRAPHQL_URL, method="POST", body={"query": query})
        if status != 200:
            return None
        doc = json.loads(raw)
        for tpl in ((doc.get("data") or {}).get("myself") or {}).get("podTemplates") or []:
            if tpl.get("id") == template_id:
                return {e.get("key") for e in tpl.get("env") or [] if isinstance(e, dict)}
        return None
    except (RunPodApiError, json.JSONDecodeError):
        return None


def list_templates_graphql():
    """Every template on the account: id, name, image. NAMES ONLY.

    Deliberately does NOT select `env`. The existing
    template_env_names_graphql asks for env because it must report which
    KEYS are set, and it strips the values on the way out — but a listing
    has no reason to pull secret values across the wire at all, so it does
    not ask for them. The narrower query is the safer one.

    Returns a list, or None when the answer is unknown — never [] for
    "could not look", which would read as "the account has no templates".
    """
    # containerDiskInGb / volumeInGb joined the selection 2026-08-29: a job
    # cannot download a checkpoint larger than the disk it has, so these two
    # numbers decide which candidate models are probeable on this endpoint at
    # all. Still NAMES ONLY — no env, no secret values.
    query = (
        "query { myself { podTemplates { id name imageName "
        "containerDiskInGb volumeInGb volumeMountPath } } }"
    )
    try:
        status, raw = _request(GRAPHQL_URL, method="POST", body={"query": query})
        if status != 200:
            return None
        doc = json.loads(raw)
        templates = ((doc.get("data") or {}).get("myself") or {}).get("podTemplates")
        if not isinstance(templates, list):
            return None
        return [
            {
                "id": t.get("id"),
                "name": t.get("name"),
                "imageName": t.get("imageName"),
                "containerDiskInGb": t.get("containerDiskInGb"),
                "volumeInGb": t.get("volumeInGb"),
                "volumeMountPath": t.get("volumeMountPath"),
            }
            for t in templates
            if isinstance(t, dict)
        ]
    except (RunPodApiError, json.JSONDecodeError):
        return None


def rest_template_surface():
    """Does the REST API expose a way to CREATE a template and ATTACH it?

    Read-only: this reads the public OpenAPI document and reports which
    template paths and verbs exist. Asking the spec is cheaper and safer
    than probing with a real POST, and it is the same technique
    rest_schema_probe used to settle the standby question — where the
    answer turned out to be that the field simply is not in the schema.
    """
    for url in (f"{REST_BASE}/openapi.json", "https://rest.runpod.io/openapi.json"):
        try:
            status, raw = _request(url, bearer=False)
        except RunPodApiError:
            continue
        if status != 200 or not raw.lstrip().startswith("{"):
            continue
        try:
            doc = json.loads(raw)
        except json.JSONDecodeError:
            continue
        paths = doc.get("paths") or {}
        found = {}
        for path, spec in paths.items():
            if "template" in path.lower():
                found[path] = sorted(
                    v.upper() for v in spec if v.lower() in
                    ("get", "post", "patch", "put", "delete")
                )
        endpoint_patch = sorted(
            (paths.get("/endpoints/{endpointId}") or {}).keys()
        )

        # WHAT CREATE ACTUALLY ACCEPTS. A path existing is not the same as a
        # path that can do the job: if the create body only takes an
        # imageName, then a template cannot be built from a repository here
        # and no amount of POSTing will produce one.
        def _props(node):
            try:
                schema = node["requestBody"]["content"]["application/json"]["schema"]
            except (KeyError, TypeError):
                return None
            ref = schema.get("$ref") if isinstance(schema, dict) else None
            if ref:
                name = str(ref).rsplit("/", 1)[-1]
                schema = ((doc.get("components") or {}).get("schemas") or {}).get(name)
            if not isinstance(schema, dict):
                return None
            return {
                "required": schema.get("required"),
                "properties": sorted((schema.get("properties") or {}).keys()),
            }

        create = _props((paths.get("/templates") or {}).get("post") or {})
        patch_ep = _props((paths.get("/endpoints/{endpointId}") or {}).get("patch") or {})

        # EVERY path, not just the ones named "template". A create body that
        # only takes an imageName means a template cannot be built from a
        # repository AT THAT PATH; it does not yet mean the API has no build
        # route at all. Reporting "impossible" off a keyword-filtered scan
        # would be enumerating failures rather than searching, so dump the
        # whole surface and let the absence be measured.
        every = {}
        for path, spec in paths.items():
            if not isinstance(spec, dict):
                continue
            every[path] = sorted(
                v.upper() for v in spec
                if v.lower() in ("get", "post", "patch", "put", "delete")
            )
        wanted = ("build", "github", "git", "repo", "registry", "source", "image")
        build_like = sorted(
            path for path in every if any(w in path.lower() for w in wanted)
        )
        return {
            "template_paths": found,
            "endpoint_verbs": endpoint_patch,
            "template_create_body": create,
            "endpoint_patch_body": patch_ep,
            "all_paths": every,
            "build_like_paths": build_like,
        }
    return None


def standby_schema_probe():
    """Read-only GraphQL introspection: which mutations exist, and which
    fields EndpointInput really carries. Run #36 proved workersStandby is
    not an EndpointInput field; this answers whether ANY mutation is
    standby-shaped before concluding the console is the only path.
    Names only — nothing here mutates. Returns None when unreadable."""
    query = (
        'query StandbyProbe { mutation: __type(name: "Mutation") '
        "{ fields { name } } input: __type(name: \"EndpointInput\") "
        "{ inputFields { name } } }"
    )
    status, raw = _request(GRAPHQL_URL, method="POST", body={"query": query})
    if status != 200:
        return None
    try:
        data = json.loads(raw).get("data") or {}
    except json.JSONDecodeError:
        return None
    mutations = [
        f.get("name") for f in ((data.get("mutation") or {}).get("fields") or [])
    ]
    return {
        "endpoint_input_fields": sorted(
            f.get("name") for f in ((data.get("input") or {}).get("inputFields") or [])
        ),
        "standby_shaped_mutations": sorted(
            m for m in mutations if m and "standby" in m.lower()
        ),
        "endpoint_shaped_mutations": sorted(
            m for m in mutations if m and "endpoint" in m.lower()
        ),
        "worker_shaped_mutations": sorted(
            m for m in mutations if m and "worker" in m.lower()
        ),
    }


def rest_schema_probe():
    """Read-only: the REST API's own OpenAPI document, reduced to what
    the endpoint PATCH actually accepts — the exact schema run #34's
    'not in input schema' refusal was validated against — plus every
    standby-shaped key name anywhere in the spec. Sent without auth
    (the spec is public); nothing mutates. Returns None when
    unreadable."""
    import re as _re

    for url in (f"{REST_BASE}/openapi.json", "https://rest.runpod.io/openapi.json"):
        try:
            status, raw = _request(url, bearer=False)
        except RunPodApiError:
            continue
        if status != 200 or not raw.lstrip().startswith("{"):
            continue
        try:
            doc = json.loads(raw)
        except json.JSONDecodeError:
            continue
        patch_props = None
        try:
            schema = doc["paths"]["/endpoints/{endpointId}"]["patch"][
                "requestBody"]["content"]["application/json"]["schema"]
            ref = schema.get("$ref")
            if ref and ref.startswith("#/components/schemas/"):
                schema = doc["components"]["schemas"][ref.rsplit("/", 1)[1]]
            props = schema.get("properties")
            patch_props = sorted(props) if isinstance(props, dict) else None
        except (KeyError, TypeError):
            patch_props = None
        return {
            "spec_url": url,
            "patch_endpoint_properties": patch_props,
            # NOT "…_keys": the spend_run redactor blanks any key whose
            # name CONTAINS "KEY", so a field called standby_shaped_keys
            # printed as "<redacted>" — hiding the one answer this probe
            # exists to give (measured 2026-08-27, run #32). The redactor
            # is right to over-redact; the diagnostic is what must be
            # named so it cannot collide.
            "standby_shaped_names": sorted(
                set(_re.findall(r'"(\w*[Ss]tandby\w*)"', raw))
            ),
        }
    return None


# Worker fields, most-specific first. Introspection would be the honest
# way to pick one, but RunPod's Apollo server answers
# INTROSPECTION_DISABLED (measured 2026-08-30), so the schema is probed by
# ASKING: each selection is tried in turn and a rejection names the field
# it did not like. The last entry is the one every GraphQL server can
# answer, so the walk always terminates on something real.
# MEASURED 2026-08-30, runs 166-167: RunPod's public API does not expose
# per-worker identity for a serverless endpoint, and there is no route to
# it left to try.
#
#   myself.endpoints        -> Cannot query field "workers" on type "Endpoint"
#   myself.serverlessEndpoints -> Cannot query field on type "User"
#   myself.endpoints.machines  -> Cannot query field "machines" on type "Endpoint"
#   myself.pods             -> 200, but a serverless endpoint is not a pod
#   __type / __schema       -> INTROSPECTION_DISABLED (Apollo, in production)
#   a deliberately invalid field -> GRAPHQL_VALIDATION_FAILED with NO
#                                  "Did you mean" suggestions
#
# The last line is what closes it: with introspection off AND suggestions
# off, the schema cannot be learned from the server, so any further
# attempt would be a guess dressed as a probe. The counts
# ({"initializing": 1}) therefore remain the only machine-readable signal,
# and they cannot distinguish a worker downloading steadily from one being
# recreated every few minutes. That distinction lives in the endpoint's
# System log in the RunPod console, which is an OWNER read.
#
# Kept as one call rather than four: it costs one GET, it records the
# refusal in the report where the next person will look, and if RunPod
# ever adds the field it starts answering without anyone rediscovering
# this list.
_WORKER_QUERY = "{ myself { endpoints { id workers { id status } } } }"

WORKER_DETAIL_UNAVAILABLE = (
    "RunPod exposes no per-worker identity for serverless endpoints; "
    "introspection and field suggestions are both disabled. The System "
    "log in the console is the only source (owner read)."
)


def worker_detail_graphql(endpoint_id: str):
    """Per-worker rows for ONE endpoint, or (None, note). Read-only.

    Would answer what the health counts cannot: whether a worker that has
    reported "initializing" for an hour is downloading or restarting. See
    the comment above for why it currently cannot.
    """
    status, raw = _request(GRAPHQL_URL, method="POST",
                           body={"query": _WORKER_QUERY})
    if status != 200:
        return None, f"{WORKER_DETAIL_UNAVAILABLE} (HTTP {status})"
    doc = json.loads(raw)
    if doc.get("errors"):
        return None, WORKER_DETAIL_UNAVAILABLE
    endpoints = ((doc.get("data") or {}).get("myself") or {}).get("endpoints") or []
    for ep in endpoints:
        if ep.get("id") == endpoint_id:
            return ep.get("workers") or [], "myself.endpoints.workers"
    return None, f"endpoint {endpoint_id} not in myself.endpoints"


def parse_endpoint(doc: dict) -> dict:
    return {
        "id": doc.get("id"),
        "name": doc.get("name"),
        "min_workers": doc.get("workersMin"),
        "max_workers": doc.get("workersMax"),
        "gpu_type_ids": doc.get("gpuTypeIds"),
        "idle_timeout": doc.get("idleTimeout"),
    }


# ---------------------------------------------------------------- serverless


def submit_job(endpoint_id: str, job_input: dict, policy: dict | None = None):
    """POST /run (async). Mutating — the spend path only.

    `policy` is a PER-JOB override and is sent only when a caller passes one.
    The live endpoint carries executionTimeoutMs 600000 (read 2026-08-29),
    which is right for production — a paid job whose checkpoint is already in
    the image has no business running ten minutes — and far too short for a
    benchmark that downloads a 44-118 GiB checkpoint before it starts.

    Raising the ENDPOINT's timeout would change the spend bound of every
    production job, which owner directive 2026-08-29 forbids. A per-job
    policy changes one job. That is the whole reason it is here rather than a
    PATCH, and it is the same fence `model` sits behind: only the probe path
    ever passes one, and the contract admits it on no production op.

    This field is NOT in rest.runpod.io's OpenAPI document, because /run
    lives on the serverless host, which publishes none — so it is unverified
    by schema and verified by outcome instead. That is safe here: an ignored
    field leaves the 600s default in place, and a rejected body fails the
    submit before a worker starts, which costs nothing.
    """
    body: dict = {"input": job_input}
    if policy:
        body["policy"] = policy
    status, raw = _request(
        f"{SERVERLESS_BASE}/{endpoint_id}/run",
        method="POST",
        body=body,
    )
    if status != 200:
        raise RunPodApiError(f"submit_job -> {status}")
    return raw, json.loads(raw)


def job_status(endpoint_id: str, job_id: str):
    return _get_json(f"{SERVERLESS_BASE}/{endpoint_id}/status/{job_id}")


def cancel_job(endpoint_id: str, job_id: str):
    status, raw = _request(
        f"{SERVERLESS_BASE}/{endpoint_id}/cancel/{job_id}", method="POST"
    )
    return status, raw


def endpoint_billing():
    """What every endpoint on the account has actually accrued. Read-only.

    Owner directive 2026-08-30: endpoint ynysmj3dm92cwp appeared on the
    account holding workersMin=1 and workersStandby=2 on A5000s, having
    never run a job. The owner chose to leave it and be told the cost, so
    the cost is READ rather than estimated from a per-hour rate and a
    guess at how long it has been up.
    """
    try:
        raw, doc = _get_json(f"{REST_BASE}/billing/endpoints")
    except RunPodApiError as exc:
        return None, str(exc)
    return doc, None


def set_execution_timeout(endpoint_id: str, timeout_ms: int):
    """PATCH executionTimeoutMs on ONE endpoint. Nothing else is sent.

    Owner authorization 2026-08-30: raise the ceiling to 45 minutes so the
    Hunyuan probe's 32.26 GiB checkpoint download can finish inside the
    job. The worst case is one job holding the card for 45 minutes; at the
    LIVE secure rate the discovery step reads, that stays inside the
    job cap. The rate is not written here — a price copied into a
    comment is a price that goes stale silently, which is why the
    admission gate refuses one.

    Read-before and read-after are not ceremony. The 2026-08-30 template
    retarget sent a field it meant to change and silently dropped
    containerRegistryAuthId, which the endpoint then could not pull with;
    that cost hours and was invisible in everything the template printed.
    So this returns both documents and the caller compares them.
    """
    if not isinstance(timeout_ms, int) or timeout_ms <= 0:
        raise RunPodApiError(f"refusing a non-positive timeout: {timeout_ms!r}")
    _, before = get_endpoint(endpoint_id)
    status, raw = _request(
        f"{REST_BASE}/endpoints/{endpoint_id}",
        method="PATCH",
        body={"executionTimeoutMs": timeout_ms},
    )
    if status not in (200, 201, 202):
        raise RunPodApiError(
            f"PATCH /endpoints/{endpoint_id} -> {status} (body: {raw[:300]!r})"
        )
    _, after = get_endpoint(endpoint_id)
    return before, after


def endpoint_health(endpoint_id: str):
    return _get_json(f"{SERVERLESS_BASE}/{endpoint_id}/health")


def purge_queue(endpoint_id: str):
    status, raw = _request(
        f"{SERVERLESS_BASE}/{endpoint_id}/purge-queue", method="POST"
    )
    return status, raw


# ---------------------------------------------------------------- sweep


def sweep_orphans():
    """Count anything that could still be billing: pods + endpoint workers.

    Returns a dict of counts, or None when the API cannot be reached —
    None is 'cannot confirm', which must never be converted to 0.
    """
    try:
        _, pods = get_pods()
        _, endpoints = get_endpoints()
    except (RunPodApiError, json.JSONDecodeError):
        return None
    pod_list = pods if isinstance(pods, list) else pods.get("pods", [])
    ep_list = (
        endpoints if isinstance(endpoints, list) else endpoints.get("endpoints", [])
    )
    workers = 0
    for ep in ep_list:
        parsed = parse_endpoint(ep)
        min_w = parsed["min_workers"]
        if isinstance(min_w, int):
            workers += min_w
    return {"pods": len(pod_list), "endpoint_min_workers": workers}


# ---------------------------------------------------------------- discover


def discover(out_path=None) -> dict:
    """READ-ONLY inventory: catalogue, target GPU, pods, endpoints.

    Prints RAW provider responses first and the parsed view second, so a
    parser error is visible as a disagreement between the two blocks.
    Never calls a mutating endpoint.
    """
    report = {}

    raw_cat, catalogue = gpu_catalogue()
    report["gpu_catalogue_raw"] = json.loads(raw_cat)
    report["gpu_catalogue_parsed"] = catalogue
    report["target_gpu_parsed"] = find_gpu(catalogue, "NVIDIA RTX A5000")

    raw_pods, pods = get_pods()
    report["pods_raw"] = json.loads(raw_pods)

    raw_eps, endpoints = get_endpoints()
    report["endpoints_raw"] = json.loads(raw_eps)
    ep_list = (
        endpoints if isinstance(endpoints, list) else endpoints.get("endpoints", [])
    )
    report["endpoints_parsed"] = [parse_endpoint(e) for e in ep_list]

    print("=== RAW gpuTypes (provider bytes, verbatim) ===")
    print(raw_cat)
    print("=== RAW pods ===")
    print(raw_pods)
    print("=== RAW endpoints ===")
    print(raw_eps)
    print("=== PARSED target GPU ===")
    print(json.dumps(report["target_gpu_parsed"], indent=1))
    print("=== PARSED endpoints ===")
    print(json.dumps(report["endpoints_parsed"], indent=1))

    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
    return report


def create_template(name: str, image_name: str, container_disk_gb: int):
    """Create a serverless template naming an EXISTING image.

    Deliberately minimal. POST /templates accepts fourteen properties
    (measured 2026-08-28) and this sends five: anything unsent keeps
    RunPod's default, and every field named here is a field that could be
    got wrong. In particular it sends NO `env`: the worker's three R2
    variables live in the RunPod environment and nowhere else — storage.py
    states that as the contract — so this function has no parameter that
    could carry a credential, and cannot leak one it never receives.
    """
    body = {
        "name": name,
        "imageName": image_name,
        "isServerless": True,
        "isPublic": False,
        "containerDiskInGb": container_disk_gb,
    }
    status, raw = _request(f"{REST_BASE}/templates", method="POST", body=body)
    if status not in (200, 201):
        raise RunPodApiError(f"POST /templates -> {status}: {raw[:200]}")
    return raw, json.loads(raw)


def attach_template(endpoint_id: str, template_id: str):
    """Point an endpoint at a template. templateId is the ONLY field sent.

    PATCH /endpoints/{id} also accepts workersMax, workersMin, gpuTypeIds,
    idleTimeout and executionTimeoutMs. Sending any of them — even at what
    is believed to be the current value — would let a stale read silently
    rewrite the endpoint's spend bounds. One field goes in the body, so
    nothing else can change. Callers must re-read; no write echo is
    trusted.
    """
    status, raw = _request(
        f"{REST_BASE}/endpoints/{endpoint_id}",
        method="PATCH",
        body={"templateId": template_id},
    )
    if status not in (200, 201):
        raise RunPodApiError(f"PATCH /endpoints/{endpoint_id} -> {status}: {raw[:200]}")
    return raw, json.loads(raw) if raw.strip().startswith("{") else {}


def retarget_template(template_id: str, image_name: str, container_disk_gb: int,
                      container_registry_auth_id: str | None = None):
    """Point an EXISTING template at a new image and disk. Two fields, named.

    The one-field rule that governs set_template_env and attach_template is
    about a stale read silently rewriting something nobody meant to touch.
    Here the image and the disk are exactly what is meant to change — owner
    directive 2026-08-29, 80 GB to 200 GB — and nothing else is sent, so
    `name`, `env` and `dockerStartCmd` cannot move.

    This exists because creating a second template is not possible: RunPod
    answers 500 "Template name must be unique", measured on 2026-08-29. It is
    also the better shape. Updating in place keeps the template id the
    endpoint already points at, and keeps the env holding the R2 secret
    REFERENCES — so the worker does not lose its storage configuration on the
    way to a bigger disk, and there is no window in which the endpoint runs a
    template that cannot write its output.
    """
    # THE REGISTRY CREDENTIAL RIDES ALONG, when the caller read one off the
    # template first. A narrow PATCH should leave unnamed fields alone, and
    # this one names only what it means to change — but "should" is the word
    # that cost 2026-08-30: after a retarget, workers went back to pulling
    # ghcr.io anonymously and hitting toomanyrequests, which is what an
    # absent credential looks like from the outside. Sending the id back
    # explicitly makes preservation something the request states rather than
    # something the provider is trusted to infer, and template_retarget then
    # re-reads it to confirm.
    body = {"imageName": image_name, "containerDiskInGb": container_disk_gb}
    if container_registry_auth_id:
        body["containerRegistryAuthId"] = container_registry_auth_id
    status, raw = _request(
        f"{REST_BASE}/templates/{template_id}",
        method="PATCH",
        body=body,
    )
    if status not in (200, 201):
        raise RunPodApiError(f"PATCH /templates/{template_id} -> {status}: {raw[:200]}")
    return raw, json.loads(raw) if raw.strip().startswith("{") else {}


def set_template_env(template_id: str, env: dict):
    """Set a template's env. `env` is the ONLY field sent.

    The same discipline as attach_template: PATCH /templates/{id} also
    accepts imageName, containerDiskInGb, name and dockerStartCmd, and
    sending any of them - even at what is believed to be the current
    value - would let a stale read silently rewrite the image this
    endpoint runs. One field goes in the body. Callers must re-read.

    Values here are RunPod secret REFERENCES, not credentials, and are
    never logged by this function or its callers.
    """
    status, raw = _request(
        f"{REST_BASE}/templates/{template_id}",
        method="PATCH",
        body={"env": env},
    )
    if status not in (200, 201):
        raise RunPodApiError(f"PATCH /templates/{template_id} -> {status}: {raw[:200]}")
    return raw, json.loads(raw) if raw.strip().startswith("{") else {}


def main(argv) -> int:
    if len(argv) >= 2 and argv[1] == "discover":
        out = None
        if "--json" in argv:
            out = argv[argv.index("--json") + 1]
        discover(out)
        return 0
    print("usage: runpod_client.py discover [--json out.json]")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
