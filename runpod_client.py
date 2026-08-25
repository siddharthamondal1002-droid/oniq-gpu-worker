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
    headers = {"Content-Type": "application/json"}
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
        "A key that works on REST but not GraphQL lacks GraphQL "
        "permission — an owner toggle on the API key in the RunPod "
        "console."
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
        "community_cloud": bool(g.get("communityCloud")),
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


def submit_job(endpoint_id: str, job_input: dict):
    """POST /run (async). Mutating — the spend path only."""
    status, raw = _request(
        f"{SERVERLESS_BASE}/{endpoint_id}/run",
        method="POST",
        body={"input": job_input},
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
    report["target_gpu_parsed"] = find_gpu(catalogue, "NVIDIA GeForce RTX 3090")

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
