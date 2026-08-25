"""Phases 9-19 driver for gpu-validation.yml — every decision testable.

The workflow used to hold this logic in YAML heredocs, which cannot be
tested. Everything that decides whether money may move now lives here,
with the transport injected so the gates run against fakes offline.

Discipline carried over from the ledger and the superloop, encoded:

- nothing here defaults a price, and stale figures are never reused —
  every stage re-queries before it may act;
- credential VALUES never reach a log: raw provider payloads are printed
  only through redact(), which blanks any key that looks secret-bearing
  (presence may be verified; values must never appear);
- UNKNOWN termination is never converted to success;
- the spend gate is two-factor (the literal SPEND input and the
  gpu-spend environment approval), and preflight additionally verifies
  the environment HAS a required-reviewer rule — GitHub auto-creates an
  unprotected environment on first reference, which would have turned
  the approval gate into a no-op;
- every stop is a typed SpendStop with a stable code, and the driver
  never auto-recovers around a financial failure.
"""

from __future__ import annotations

import json
import os
import statistics
import time
import urllib.request
from decimal import ROUND_UP, Decimal

from validation import admission

R2_ENV_REQUIRED = ("R2_S3_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")
_REDACT_MARKERS = ("KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL", "AUTHORIZATION")

TERMINATION_CONFIRMED = "CONFIRMED_TERMINATED"
TERMINATION_UNKNOWN = "TERMINATION_UNKNOWN"


class SpendStop(Exception):
    """A gate refused. code is stable; message never carries a secret."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


# ------------------------------------------------------------------ redact


def redact(obj):
    """Deep-copy with every secret-shaped key's value blanked."""
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            upper = str(key).upper()
            if any(marker in upper for marker in _REDACT_MARKERS):
                out[key] = "<redacted>"
            else:
                out[key] = redact(value)
        return out
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    return obj


def _show(label: str, obj) -> None:
    print(f"=== {label} ===")
    print(json.dumps(redact(obj), indent=1, default=str))


# ------------------------------------------------- environment protection


def _default_env_fetch(url: str, token: str):
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, {}
    except Exception:
        return 0, {}


def check_environment_protection(fetch=_default_env_fetch) -> int:
    """The gpu-spend environment must exist AND carry a required-reviewer
    rule. Referencing a missing environment silently creates an
    UNPROTECTED one, so 'the job waited for approval' cannot be assumed —
    it must be read back from the API. Returns the reviewer-rule count."""
    repo = os.environ.get("GITHUB_REPOSITORY")
    token = os.environ.get("GITHUB_TOKEN")
    if not repo or not token:
        raise SpendStop(
            "environment-unverifiable",
            "GITHUB_REPOSITORY/GITHUB_TOKEN absent; cannot verify gpu-spend "
            "protection, so spending must not proceed",
        )
    status, doc = fetch(
        f"https://api.github.com/repos/{repo}/environments/gpu-spend", token
    )
    if status == 404:
        raise SpendStop(
            "environment-missing",
            "the gpu-spend environment does not exist — creating it with a "
            "required reviewer is an owner action",
        )
    if status != 200:
        raise SpendStop(
            "environment-unverifiable",
            f"environments API answered {status}; unverified protection is "
            "not protection",
        )
    rules = [
        r
        for r in (doc.get("protection_rules") or [])
        if r.get("type") == "required_reviewers"
    ]
    if not rules:
        raise SpendStop(
            "environment-unprotected",
            "gpu-spend exists but has NO required reviewer — the approval "
            "gate would be a no-op; configuring a reviewer is an owner action",
        )
    return len(rules)


# --------------------------------------------------------------- preflight


def preflight(
    client,
    *,
    endpoint_id: str = "",
    input_ref: str = "",
    output_prefix: str = "",
    env_fetch=_default_env_fetch,
) -> dict:
    """Phases 9-10: every free verification, in a fixed order, no spend.

    Returns the facts later stages must re-verify (never merely reuse).
    """
    # 1. Authentication + nothing quietly running.
    raw_pods, pods = client.get_pods()
    pod_list = pods if isinstance(pods, list) else pods.get("pods", [])
    _show("pods (raw, redacted)", json.loads(raw_pods))
    if len(pod_list) != 0:
        raise SpendStop("unexpected-pods", f"{len(pod_list)} pod(s) exist; expected 0")

    # 2. Exactly one endpoint (or the one explicitly named).
    raw_eps, endpoints = client.get_endpoints()
    ep_list = endpoints if isinstance(endpoints, list) else endpoints.get("endpoints", [])
    _show("endpoints (raw, redacted)", json.loads(raw_eps))
    if endpoint_id:
        matches = [e for e in ep_list if e.get("id") == endpoint_id]
    else:
        matches = ep_list
    if len(matches) != 1:
        raise SpendStop(
            "endpoint-not-singular",
            f"{len(matches)} candidate endpoint(s); need exactly one "
            "(create it min_workers=0/max_workers=1 — an owner action)",
        )
    endpoint = matches[0]
    parsed = client.parse_endpoint(endpoint)
    if parsed["min_workers"] is None or parsed["max_workers"] is None:
        raise SpendStop(
            "endpoint-fields-unparsed",
            "worker bounds did not parse from the endpoint payload — parser "
            "vs raw mismatch; correct parse_endpoint against the raw above",
        )
    admission.check_endpoint_config(parsed["min_workers"], parsed["max_workers"])

    # 3. The endpoint is the 3090, by id.
    gpu_ids = parsed.get("gpu_type_ids") or []
    if admission.TARGET_GPU not in gpu_ids:
        raise SpendStop(
            "endpoint-not-3090",
            f"endpoint gpuTypeIds {gpu_ids} does not include the target",
        )

    # 4. R2 env NAMES present on the endpoint (values never printed).
    env_obj = endpoint.get("env") or {}
    env_names = set(env_obj) if isinstance(env_obj, dict) else {
        e.get("key") for e in env_obj if isinstance(e, dict)
    }
    missing = [name for name in R2_ENV_REQUIRED if name not in env_names]
    if missing:
        raise SpendStop(
            "r2-env-missing",
            "endpoint environment lacks: " + ", ".join(missing),
        )

    # 5. Test references exist (object keys, not credentials).
    if not input_ref or not output_prefix:
        raise SpendStop(
            "test-refs-missing",
            "GPU_TEST_INPUT_REF and GPU_TEST_OUTPUT_PREFIX must be set",
        )

    # 6. Live price, quoted now — never the previous run's number.
    _, catalogue = client.gpu_catalogue()
    target = admission.require_available(catalogue)
    reservation = admission.admit(
        gpu_name=target["id"],
        vram_gb=target["memory_gb"],
        runtime_seconds=admission.RUNTIME_CEILING_SECONDS,
        price_per_hour=target["secure_price"],
    )

    # 7. The approval gate is real, not auto-created-and-empty.
    reviewer_rules = check_environment_protection(env_fetch)

    facts = {
        "endpoint_id": parsed["id"],
        "gpu_id": target["id"],
        "vram_gb": target["memory_gb"],
        "live_price_per_hour": str(target["secure_price"]),
        "runtime_ceiling_s": admission.RUNTIME_CEILING_SECONDS,
        "reservation_usd": str(reservation.reserved_usd),
        "headroom_usd": str(admission.JOB_CAP_USD - reservation.reserved_usd),
        "reviewer_rules": reviewer_rules,
        "input_ref": input_ref,
        "output_prefix": output_prefix,
    }
    _show("preflight facts", facts)
    return facts


# ------------------------------------------------------------ one real job


def requote(client) -> dict:
    """Phase 12: immediately before provisioning, quote again."""
    _, catalogue = client.gpu_catalogue()
    target = admission.require_available(catalogue)
    reservation = admission.admit(
        gpu_name=target["id"],
        vram_gb=target["memory_gb"],
        runtime_seconds=admission.RUNTIME_CEILING_SECONDS,
        price_per_hour=target["secure_price"],
    )
    return {"price": Decimal(str(target["secure_price"])), "reservation": reservation.reserved_usd}


def submit_and_wait(
    client,
    endpoint_id: str,
    job_input: dict,
    *,
    poll_s: int = 5,
    sleep=time.sleep,
    clock=time.monotonic,
) -> dict:
    _, submitted = client.submit_job(endpoint_id, job_input)
    job_id = submitted.get("id")
    if not job_id:
        raise SpendStop("submit-unparsed", "job id missing from submit response")
    deadline = clock() + admission.RUNTIME_CEILING_SECONDS
    while clock() < deadline:
        _, status = client.job_status(endpoint_id, job_id)
        if status.get("status") in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            status["_job_id"] = job_id
            return status
        sleep(poll_s)
    client.cancel_job(endpoint_id, job_id)
    raise SpendStop("job-deadline", f"job {job_id} exceeded the ceiling; cancelled")


def verify_gpu_success(output) -> None:
    """Phase 14: HTTP 200 alone is insufficient, and so is each of these
    alone — all must hold."""
    if not isinstance(output, dict) or output.get("ok") is not True:
        raise SpendStop("job-not-ok", f"worker did not report ok; code={None if not isinstance(output, dict) else output.get('code')}")
    if output.get("device") != "cuda":
        raise SpendStop("not-cuda", "device is not cuda — CPU fallback is not success")
    if "3090" not in str(output.get("gpu_name") or ""):
        raise SpendStop("wrong-gpu", "gpu_name does not identify an RTX 3090")
    if output.get("vram_peak_mb") is None:
        raise SpendStop("no-vram-peak", "peak VRAM was not measured")
    if not output.get("output_bytes"):
        raise SpendStop("no-artifact", "no output artifact was written")
    unexpected = set(output) - set(_allowed_output_keys())
    if unexpected:
        raise SpendStop("schema-violation", f"unwhitelisted keys: {sorted(unexpected)}")


def _allowed_output_keys():
    import contract

    return contract.OUTPUT_WHITELIST


def confirm_termination(
    client,
    endpoint_id: str,
    *,
    wait_s: int = 180,
    poll_s: int = 5,
    sleep=time.sleep,
    clock=time.monotonic,
) -> str:
    """Phase 16. Only an API answer showing zero workers confirms; an
    unreachable API or unparseable shape is UNKNOWN, and UNKNOWN is never
    converted to success by anyone downstream."""
    deadline = clock() + wait_s
    last = None
    while clock() < deadline:
        try:
            raw, health = client.endpoint_health(endpoint_id)
        except Exception:
            sleep(poll_s)
            continue
        last = health
        workers = health.get("workers")
        if isinstance(workers, dict) and workers:
            counts = [v for v in workers.values() if isinstance(v, int)]
            if counts and sum(counts) == 0:
                _show("termination health", health)
                return TERMINATION_CONFIRMED
        sleep(poll_s)
    if last is not None:
        _show("last health before UNKNOWN", last)
    return TERMINATION_UNKNOWN


def actual_cost_usd(execution_ms, price_per_hour: Decimal) -> Decimal:
    """Computed from the provider's executionTime at the live price,
    rounded UP to the cent like every figure that gets compared against
    a ceiling. The invoice remains the final authority — this is the
    in-run bound, not a substitute for Phase 15's billing record."""
    if execution_ms is None:
        raise SpendStop("billing-unavailable", "executionTime missing from status")
    seconds = Decimal(int(execution_ms)) / Decimal(1000)
    exact = Decimal(str(price_per_hour)) * seconds / Decimal(3600)
    return exact.quantize(Decimal("0.01"), rounding=ROUND_UP)


def one_job(client, facts: dict, *, output_key: str, sleep=time.sleep, clock=time.monotonic) -> dict:
    """Phases 12-16 for a single job. Fail-closed at every boundary."""
    quote = requote(client)
    status = submit_and_wait(
        client,
        facts["endpoint_id"],
        {
            "op": "image_preprocess",
            "input_key": facts["input_ref"],
            "output_key": output_key,
        },
        sleep=sleep,
        clock=clock,
    )
    _show("job status (raw, redacted)", status)
    if status.get("status") != "COMPLETED":
        raise SpendStop("job-failed", f"terminal status {status.get('status')}")
    verify_gpu_success(status.get("output"))
    cost = actual_cost_usd(status.get("executionTime"), quote["price"])
    if cost > quote["reservation"]:
        raise SpendStop(
            "actual-over-reservation",
            f"computed ${cost} exceeds reservation ${quote['reservation']} — "
            "failing closed, not rewriting the reservation",
        )
    termination = confirm_termination(client, facts["endpoint_id"], sleep=sleep, clock=clock)
    row = {
        "job_id": status.get("_job_id"),
        "gpu_name": status["output"].get("gpu_name"),
        "vram_peak_mb": status["output"].get("vram_peak_mb"),
        "delay_ms": status.get("delayTime"),
        "execution_ms": status.get("executionTime"),
        "cost_usd": str(cost),
        "reservation_usd": str(quote["reservation"]),
        "output_key": output_key,
        "termination": termination,
    }
    _show("job row", row)
    if termination != TERMINATION_CONFIRMED:
        raise SpendStop(
            "termination-unknown",
            "worker termination could not be confirmed; not continuing",
        )
    return row


# --------------------------------------------------------- failure battery


def failure_battery(client, facts: dict, *, sleep=time.sleep, clock=time.monotonic) -> list:
    """Phase 17, the externally forceable cases. Each must surface its
    failure AND leave zero workers. Cases that need a bucket-side fixture
    this harness cannot create are reported, not faked."""
    cases = [
        (
            "r2-read-failure",
            {
                "op": "image_preprocess",
                "input_key": f"{facts['output_prefix']}/does-not-exist.bin",
                "output_key": f"{facts['output_prefix']}/never-written.jpeg",
            },
            ("r2-read-failed",),
        ),
        (
            "contract-refusal-op",
            {
                "op": "train_model",
                "input_key": facts["input_ref"],
                "output_key": f"{facts['output_prefix']}/never-written.jpeg",
            },
            ("op-not-allowed",),
        ),
        (
            "contract-refusal-params",
            {
                "op": "image_preprocess",
                "input_key": facts["input_ref"],
                "output_key": f"{facts['output_prefix']}/never-written.jpeg",
                "params": {"target_max_dim": 999999},
            },
            ("invalid-input",),
        ),
    ]
    rows = []
    for name, payload, expected_codes in cases:
        status = submit_and_wait(client, facts["endpoint_id"], payload, sleep=sleep, clock=clock)
        output = status.get("output") or {}
        surfaced = (
            status.get("status") == "COMPLETED"
            and output.get("ok") is False
            and output.get("code") in expected_codes
        )
        termination = confirm_termination(client, facts["endpoint_id"], sleep=sleep, clock=clock)
        row = {
            "case": name,
            "surfaced": surfaced,
            "code": output.get("code"),
            "termination": termination,
        }
        _show("failure case", row)
        if not surfaced:
            raise SpendStop("failure-not-surfaced", f"case {name} did not fail loudly")
        if termination != TERMINATION_CONFIRMED:
            raise SpendStop("termination-unknown", f"case {name}: termination unconfirmed")
        rows.append(row)
    print(
        "NOT FORCEABLE FROM THIS HARNESS (no bucket write access, by design): "
        "a mid-inference exception on a corrupt-but-existing object, a true "
        "R2 write denial, and a real timeout. Reported, not faked."
    )
    return rows


# ---------------------------------------------------------------- battery


def battery(client, facts: dict, n: int, *, sleep=time.sleep, clock=time.monotonic) -> list:
    rows = []
    for i in range(n):
        row = one_job(
            client,
            facts,
            output_key=f"{facts['output_prefix']}/battery-{n}-{i}.jpeg",
            sleep=sleep,
            clock=clock,
        )
        rows.append(row)
    return rows


def economics(rows: list) -> dict:
    """Phase 20 arithmetic over REAL rows. Refuses an empty set rather
    than reporting zeros that look like measurements."""
    if not rows:
        raise SpendStop("no-data", "economics need at least one real job row")
    costs = [Decimal(r["cost_usd"]) for r in rows]
    execs = [int(r["execution_ms"]) for r in rows if r.get("execution_ms") is not None]
    delays = [int(r["delay_ms"]) for r in rows if r.get("delay_ms") is not None]
    total = sum(costs)
    avg = (total / len(costs)).quantize(Decimal("0.0001"), rounding=ROUND_UP)
    out = {
        "jobs": len(rows),
        "total_cost_usd": str(total),
        "avg_cost_usd": str(avg),
        "median_cost_usd": str(statistics.median(costs)),
        "max_cost_usd": str(max(costs)),
        "cost_per_1000_usd": str((avg * 1000).quantize(Decimal("0.01"), rounding=ROUND_UP)),
        "avg_execution_ms": int(statistics.mean(execs)) if execs else None,
        "p95_execution_ms": int(sorted(execs)[max(0, int(len(execs) * 0.95) - 1)]) if execs else None,
        "avg_startup_overhead_ms": int(statistics.mean(delays)) if delays else None,
    }
    if execs and delays:
        busy = sum(execs)
        wall = busy + sum(delays)
        out["gpu_utilization"] = round(busy / wall, 3) if wall else None
    return out


# ------------------------------------------------------------------- main


def _inputs_from_env() -> dict:
    return {
        "endpoint_id": os.environ.get("ENDPOINT_ID", ""),
        "input_ref": os.environ.get("GPU_TEST_INPUT_REF", ""),
        "output_prefix": os.environ.get("GPU_TEST_OUTPUT_PREFIX", ""),
    }


def main(argv) -> int:
    import runpod_client as rp

    if len(argv) < 2 or argv[1] not in ("preflight", "run"):
        print("usage: python -m validation.spend_run preflight|run")
        return 2
    params = _inputs_from_env()
    try:
        facts = preflight(rp, **params)
        if argv[1] == "preflight":
            print("PREFLIGHT PASS — every free gate holds; spending remains "
                  "gated on SPEND + gpu-spend approval")
            return 0

        through = int(os.environ.get("THROUGH_PHASE", "16"))
        rows = [one_job(rp, facts, output_key=f"{facts['output_prefix']}/job-1.jpeg")]
        print("PHASE 13-16 PASS — one real job, verified and terminated")
        if through >= 17:
            failure_battery(rp, facts)
            print("PHASE 17 PASS — forceable failure cases surfaced, 0 orphans")
        if through >= 18:
            rows += battery(rp, facts, 5)
            print("PHASE 18 PASS — five-job battery")
        if through >= 19:
            rows += battery(rp, facts, 20)
            print("PHASE 19 PASS — twenty-job battery")
        _show("economics (real rows only)", economics(rows))
        sweep = rp.sweep_orphans()
        if sweep is None or sweep.get("pods") != 0 or sweep.get("endpoint_min_workers") != 0:
            raise SpendStop("orphan-alarm", f"final sweep not clean: {sweep}")
        print("ZERO-IDLE CONFIRMED — pods 0, endpoint min workers 0")
        return 0
    except SpendStop as stop:
        print(f"STOP [{stop.code}]: {stop.message}")
        return 1
    except admission.AdmissionRefused as refused:
        print(f"STOP [{refused.code}]: {refused.message}")
        return 1
    except admission.UnavailableGpu as unavailable:
        print(f"STOP [gpu-unavailable]: {unavailable}")
        return 1


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv))
