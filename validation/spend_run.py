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
  gpu-spend environment approval). Owner directive 2026-08-25: the
  checks run INSIDE the spend job (no separate blocking preflight job),
  so on the run path the gate is verified as EVIDENCE — this run's own
  recorded approval — because GitHub waves a job straight through an
  unprotected environment; a readable, empty approvals list means the
  mandated pause never happened and provisioning is refused. The
  standalone advisory preflight still reads the reviewer rule back;
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


def _env_names(env_obj) -> set:
    if isinstance(env_obj, dict):
        return set(env_obj)
    if isinstance(env_obj, list):
        return {e.get("key") for e in env_obj if isinstance(e, dict)}
    return set()


class SpendStop(Exception):
    """A gate refused. code is stable; message never carries a secret."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


# ------------------------------------------------------------------ redact


def redact(obj):
    """Deep-copy with every secret-shaped key's value blanked. Handles
    both {NAME: value} maps and RunPod's [{key: NAME, value: ...}] pair
    form — in pair form the secret hides under a field literally named
    'value', which name-based blanking alone would leak."""
    if isinstance(obj, dict):
        if "key" in obj and "value" in obj:
            pair = dict(obj)
            if any(m in str(pair.get("key")).upper() for m in _REDACT_MARKERS):
                pair["value"] = "<redacted>"
            return pair
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
        # The error body is the diagnosis (GitHub says WHY: missing token
        # permission vs. plan limitation) — losing it cost a debugging
        # round on 2026-08-25, the same way the Cloudflare 1010 body did.
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
            return exc.code, json.loads(body)
        except Exception:
            return exc.code, {"raw": body[:300]}
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
        detail = ""
        if isinstance(doc, dict):
            detail = doc.get("message") or doc.get("raw") or ""
        hint = (
            " — 403 here means either the workflow token lacks "
            "'deployments: read' permission, or environments are not "
            "available on this repository's plan (private repo)"
            if status == 403
            else ""
        )
        raise SpendStop(
            "environment-unverifiable",
            f"environments API answered {status}"
            + (f' saying "{detail}"' if detail else "")
            + f"{hint}; unverified protection is not protection",
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


def check_run_approval(fetch=_default_env_fetch) -> str:
    """Owner directive 2026-08-25: the spend job itself carries every
    check — no separate preflight job — but it must still have PAUSED for
    the gpu-spend required reviewer before provisioning. Direct evidence
    first: this run's own recorded approvals (who clicked). GitHub waves
    a job straight through an unprotected environment, so a readable,
    empty approvals list means the mandated pause never happened — stop
    before provisioning. Only when the approvals API is unreadable does
    the environment's reviewer rule, read back at job start, stand in:
    rule present while this job runs implies the pause occurred.

    Owner directive 2026-08-25 (second, same day): required reviewers
    are NOT offered on this private repository's GitHub plan — the
    Deployment protection rules section does not render at all — so for
    the dispatch-authorized single job the owner's own authenticated
    workflow_dispatch carrying the literal SPEND input IS the approval.
    The workflow must declare that explicitly via
    APPROVAL_MODE=owner-dispatch; any other value keeps the evidence
    requirement, so restoring the reviewer gate later is deleting one
    line of YAML."""
    if os.environ.get("APPROVAL_MODE") == "owner-dispatch":
        print(
            "approval mode: owner-dispatch — owner directive 2026-08-25: "
            "required reviewers are unavailable on this private repo's "
            "plan, and the owner's authenticated SPEND dispatch is the "
            "recorded approval for the single authorized job"
        )
        return "owner-dispatch"
    repo = os.environ.get("GITHUB_REPOSITORY")
    token = os.environ.get("GITHUB_TOKEN")
    run_id = os.environ.get("GITHUB_RUN_ID")
    if not repo or not token or not run_id:
        raise SpendStop(
            "approval-unverifiable",
            "GITHUB_REPOSITORY/GITHUB_TOKEN/GITHUB_RUN_ID absent; cannot "
            "verify the gpu-spend approval happened, so provisioning must "
            "not proceed",
        )
    status, doc = fetch(
        f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/approvals",
        token,
    )
    if status == 200 and isinstance(doc, list):
        for approval in doc:
            if isinstance(approval, dict) and approval.get("state") == "approved":
                user = (approval.get("user") or {}).get("login") or "unknown"
                return user
        raise SpendStop(
            "approval-not-recorded",
            "this run has NO recorded gpu-spend approval — GitHub never "
            "paused it, which means the environment carries no required "
            "reviewer; the owner-mandated approval cannot have happened, so "
            "provisioning is refused. Fix: gpu-spend -> Required reviewers "
            "-> Save protection rules, then dispatch again",
        )
    check_environment_protection(fetch)
    return "reviewer-rule-verified"


# --------------------------------------------------------------- preflight


def preflight(
    client,
    *,
    endpoint_id: str = "",
    input_ref: str = "",
    output_prefix: str = "",
    env_fetch=_default_env_fetch,
    approval_evidence: bool = False,
) -> dict:
    """Phases 9-10: every free verification, in a fixed order, no spend.

    approval_evidence=True is the run path (owner directive 2026-08-25:
    the checks live inside the spend job, behind the gpu-spend pause):
    instead of reading the environment's configuration, require evidence
    that THIS run was paused and approved. False is the standalone
    advisory preflight, which reads the configuration back.

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

    # 3. The endpoint is the target card, by id.
    gpu_ids = parsed.get("gpu_type_ids") or []
    if admission.TARGET_GPU not in gpu_ids:
        raise SpendStop(
            "endpoint-not-target",
            f"endpoint gpuTypeIds {gpu_ids} does not include the target "
            f"{admission.TARGET_GPU}",
        )
    extras = [g for g in gpu_ids if g != admission.TARGET_GPU]
    if extras:
        raise SpendStop(
            "endpoint-gpu-list-not-exclusive",
            f"endpoint can also allocate {extras} — the scheduler may hand "
            f"the job a non-target card, which fails the success gate AFTER "
            f"paying for the boot; restrict the endpoint to "
            f"{admission.TARGET_GPU} only",
        )

    # 4. R2 env NAMES present on the endpoint OR its template (values
    # never printed) — RunPod may store env on either object.
    env_names = _env_names(endpoint.get("env"))
    template_id = endpoint.get("templateId")
    if not (env_names >= set(R2_ENV_REQUIRED)):
        # The endpoints LIST is often a summary; the single GET may
        # carry the env the list omits.
        try:
            _, full = client.get_endpoint(parsed["id"])
            env_names |= _env_names((full or {}).get("env"))
            _show("endpoint (single GET, redacted)", full)
        except Exception as exc:
            print("single-endpoint fetch failed:", exc)
    if not (env_names >= set(R2_ENV_REQUIRED)) and template_id:
        try:
            _, template = client.get_template(template_id)
            _show("template (raw, redacted)", template)
            env_names |= _env_names(template.get("env"))
        except Exception as exc:
            print("template REST fetch failed:", exc)
        if not (env_names >= set(R2_ENV_REQUIRED)):
            graphql_fn = getattr(client, "template_env_names_graphql", None)
            graphql_names = None
            if callable(graphql_fn):
                try:
                    graphql_names = graphql_fn(template_id)
                except Exception as exc:
                    print("template GraphQL fetch failed:", type(exc).__name__)
            if graphql_names is None:
                print("template GraphQL env read: unknown")
            else:
                print("template GraphQL env names:", sorted(graphql_names))
                env_names |= graphql_names
    missing = [name for name in R2_ENV_REQUIRED if name not in env_names]
    if missing:
        # Distinguish a POSITIVE miss (an env set is visible and lacks the
        # names) from an UNREADABLE env (no API view exposes serverless
        # env at all — measured 2026-08-25: list and single GET carry no
        # env field, REST /templates 404s, GraphQL template read unknown).
        # Blocking forever on an unreadable signal is as wrong as passing
        # blind: when unreadable, proceed LOUDLY — the worker itself fails
        # closed at job time with storage-not-configured naming the
        # missing variables, bounded by the one-job reservation.
        if env_names:
            raise SpendStop(
                "r2-env-missing",
                "endpoint environment lacks: " + ", ".join(missing),
            )
        print(
            "WARNING [r2-env-unverifiable]: no API view exposes the "
            "endpoint's env; could not verify "
            + ", ".join(missing)
            + ". The worker fails closed with storage-not-configured at "
            "job time if they are absent."
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

    # 7. The approval gate is real, not auto-created-and-empty. On the
    # run path this means evidence THIS run paused and was approved; on
    # the advisory path it means the reviewer rule reads back present.
    if approval_evidence:
        approved_by = check_run_approval(env_fetch)
        reviewer_rules = f"approved:{approved_by}"
    else:
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
    watch_s: int | None = None,
    sleep=time.sleep,
    clock=time.monotonic,
) -> dict:
    _, submitted = client.submit_job(endpoint_id, job_input)
    job_id = submitted.get("id")
    if not job_id:
        raise SpendStop("submit-unparsed", "job id missing from submit response")
    # watch_s is the driver's WALL-CLOCK watch on the job, queue and cold
    # boot included; execution itself stays bounded by the contract's
    # runtime ceiling and the endpoint's executionTimeout. The default
    # watch equals the ceiling; a media job passes a wider watch because
    # a first pull of the model-baked image is minutes of delayTime.
    if watch_s is None:
        watch_s = admission.RUNTIME_CEILING_SECONDS
    deadline = clock() + watch_s
    while clock() < deadline:
        _, status = client.job_status(endpoint_id, job_id)
        if status.get("status") in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            status["_job_id"] = job_id
            return status
        sleep(poll_s)
    client.cancel_job(endpoint_id, job_id)
    raise SpendStop("job-deadline", f"job {job_id} exceeded the ceiling; cancelled")


# The one motion prompt of the first media experiment — a server
# constant, per the owner's Phase 5 spec; the workflow exposes no prompt
# input, so a dispatch cannot vary it.
VIDEO_PROMPT = (
    "A cinematic close-up. The subject slowly turns toward the camera, "
    "blinks naturally, and makes a subtle facial expression while the "
    "camera gently pushes forward. Natural movement, stable identity, "
    "realistic motion, consistent lighting."
)

# The audio canary narration — a module constant like VIDEO_PROMPT: the
# dispatch chooses op=audio_mux, never the text. Sized well inside the
# 4.04s video so the canary measures the happy path, not the gate.
AUDIO_NARRATION = "The character turns to face the light."

# The five-scene battery — owner directive 2026-08-26 (Phase 6 of the
# five-video superloop), verbatim and server-side like VIDEO_PROMPT:
# the dispatch chooses only through_phase=18, never the text. Order and
# wording are frozen; prompts are NOT optimized after seeing results,
# because the purpose is measurement.
VIDEO_BATTERY = (
    (
        "intro",
        "A cinematic medium shot. The character slowly turns toward the "
        "camera, blinks naturally and gives a subtle confident smile. "
        "Gentle camera push forward, realistic movement, stable identity, "
        "natural lighting.",
    ),
    (
        "walk",
        "The character slowly walks forward through the scene while the "
        "camera tracks backward smoothly. Natural body movement, realistic "
        "footsteps, stable appearance, cinematic lighting.",
    ),
    (
        "react",
        "The character looks toward something off camera, pauses, and "
        "gradually shows surprise and concern. Subtle facial movement, "
        "natural blinking, stable identity, cinematic close-up.",
    ),
    (
        "environment",
        "The character stands still while the surrounding environment "
        "moves naturally: subtle wind, moving background elements and "
        "changing light. The camera slowly pans sideways. Cinematic "
        "realism.",
    ),
    (
        "hero",
        "The character looks directly toward the camera and slowly moves "
        "forward. The camera gently pushes in while the character "
        "maintains consistent appearance and natural expression. Cinematic "
        "final-shot composition.",
    ),
)


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


def verify_audio_success(output) -> None:
    """The audio_mux success proof. This workload is CPU by design — it
    speaks and muxes on the already-rented worker — so the CUDA/VRAM
    checks do not apply; what must hold instead is the measured audio
    evidence: a real voice track of the right length in a real artifact,
    with nothing off the whitelist."""
    if not isinstance(output, dict) or output.get("ok") is not True:
        raise SpendStop(
            "job-not-ok",
            f"worker did not report ok; code={None if not isinstance(output, dict) else output.get('code')}",
        )
    if output.get("has_audio") is not True:
        raise SpendStop("no-audio-stream", "output carries no audio stream")
    if not output.get("narration_seconds"):
        raise SpendStop("no-narration", "narration duration was not measured")
    if not output.get("audio_sample_rate"):
        raise SpendStop("no-sample-rate", "audio sample rate was not measured")
    peak = output.get("audio_peak_dbfs")
    if peak is None or peak <= -60:
        raise SpendStop(
            "audio-silent", f"audio peaks at {peak} dBFS — silence with extra steps"
        )
    audio_s = output.get("audio_seconds") or 0
    video_s = output.get("video_seconds") or 0
    if not video_s or abs(audio_s - video_s) > 0.25:
        raise SpendStop(
            "audio-drift", f"audio {audio_s}s vs video {video_s}s exceeds 0.25s"
        )
    if output.get("tts_ms") is None or output.get("mux_ms") is None:
        raise SpendStop("no-timings", "tts/mux timings were not measured")
    if not output.get("output_bytes"):
        raise SpendStop("no-artifact", "no output artifact was written")
    unexpected = set(output) - set(_allowed_output_keys())
    if unexpected:
        raise SpendStop("schema-violation", f"unwhitelisted keys: {sorted(unexpected)}")


def verify_video_success(output: dict) -> None:
    """Phase 9 of the media loop, ON TOP of verify_gpu_success: a video
    job succeeds only when the model demonstrably loaded, CUDA inference
    demonstrably ran, real frames exist and a non-zero artifact was
    encoded. A completed status proves none of that by itself."""
    model = str(output.get("model") or "")
    if not model or model == "missing":
        raise SpendStop("model-unproven", "worker did not report the loaded model")
    if not output.get("model_load_ms"):
        raise SpendStop("model-unproven", "model load time was not measured")
    if not output.get("inference_ms"):
        raise SpendStop("no-inference", "CUDA inference time was not measured")
    if not output.get("frames"):
        raise SpendStop("no-frames", "no video frames were generated")
    if not output.get("video_seconds"):
        raise SpendStop("no-frames", "generated video has zero duration")
    if output.get("encode_ms") is None:
        raise SpendStop("no-encode", "video encode time was not measured")
    # Watermark evidence (monetization resolution loop, 2026-08-27): a
    # worker built from the watermark-capable contract reports whether the
    # mark was burned. Absent means the OLD image is still serving — the
    # canary report reads that as "image not yet rebuilt", never as clean.
    if "watermarked" in output and not isinstance(output["watermarked"], bool):
        raise SpendStop(
            "watermark-evidence-invalid", "watermarked must be a boolean when reported"
        )


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
) -> dict:
    """Phase 16/17. Owner directive 2026-08-26 (production launch,
    Phase 6): the production financial rule concerns ACTIVE COMPUTE —
    running, initializing, throttled and unhealthy workers must all read
    zero from the API. workersStandby is not settable by any reachable
    API (three-surface proof, ledger §16p) and is provider-managed pool
    warmth: when idle/ready workers remain, termination is still
    CONFIRMED but the standby count is RECORDED as
    STANDBY_PROVIDER_MANAGED — the total worker count is never claimed
    to be zero. An unreachable API or unparseable shape stays UNKNOWN,
    and UNKNOWN is never converted to success by anyone downstream.

    Returns {"status", "standby", "active"}; standby/active are None
    when UNKNOWN."""
    active_keys = ("running", "initializing", "throttled", "unhealthy")
    standby_keys = ("idle", "ready")
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
            active = {
                k: workers.get(k)
                for k in active_keys
                if isinstance(workers.get(k), int)
            }
            # running and initializing are the owner-named pair and must
            # be MEASURED zeros; throttled/unhealthy count when reported.
            if (
                "running" in active
                and "initializing" in active
                and sum(active.values()) == 0
            ):
                # idle and ready overlap in RunPod's health (a ready
                # worker is counted in both), so the pool size is the
                # max of the reported figures, not their sum.
                standby_counts = [
                    workers.get(k) for k in standby_keys
                    if isinstance(workers.get(k), int)
                ]
                standby = max(standby_counts, default=0)
                _show("termination health (active compute zero)", health)
                if standby:
                    print(
                        f"STANDBY_PROVIDER_MANAGED: {standby} worker(s) — "
                        "recorded, never claimed as zero (owner directive "
                        "2026-08-26, production Phase 6; §16p: no API can "
                        "set workersStandby)"
                    )
                return {
                    "status": TERMINATION_CONFIRMED,
                    "standby": standby,
                    "active": 0,
                }
        sleep(poll_s)
    if last is not None:
        _show("last health before UNKNOWN", last)
    return {"status": TERMINATION_UNKNOWN, "standby": None, "active": None}


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


def one_job(
    client,
    facts: dict,
    *,
    output_key: str,
    op: str = "image_preprocess",
    prompt: str | None = None,
    sleep=time.sleep,
    clock=time.monotonic,
) -> dict:
    """Phases 12-16 for a single job. Fail-closed at every boundary.

    `prompt` may only ever be one of this module's own constants
    (VIDEO_PROMPT or a VIDEO_BATTERY scene) — no caller input reaches it,
    because the workflow exposes no prompt field at all."""
    quote = requote(client)
    payload = {
        "op": op,
        "input_key": facts["input_ref"],
        "output_key": output_key,
    }
    watch_s = None
    if op == "video_generate":
        payload["params"] = {"prompt": prompt or VIDEO_PROMPT}
        # queue + first pull of the model-baked image can be many minutes
        # of delayTime before bounded execution even starts.
        watch_s = admission.RUNTIME_CEILING_SECONDS + 900
    if op == "audio_mux":
        # The canary narration is a module constant, same discipline as
        # VIDEO_PROMPT: the dispatch never chooses the text. The input is
        # an EXISTING video artifact — nothing is generated to test audio.
        payload["params"] = {"narration": AUDIO_NARRATION}
        watch_s = admission.RUNTIME_CEILING_SECONDS + 900
    status = submit_and_wait(
        client,
        facts["endpoint_id"],
        payload,
        watch_s=watch_s,
        sleep=sleep,
        clock=clock,
    )
    _show("job status (raw, redacted)", status)
    if status.get("status") != "COMPLETED":
        raise SpendStop("job-failed", f"terminal status {status.get('status')}")
    if op == "audio_mux":
        verify_audio_success(status.get("output"))
    else:
        verify_gpu_success(status.get("output"))
    if op == "video_generate":
        verify_video_success(status["output"])
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
        "op": op,
        "gpu_name": status["output"].get("gpu_name"),
        "vram_peak_mb": status["output"].get("vram_peak_mb"),
        "delay_ms": status.get("delayTime"),
        "execution_ms": status.get("executionTime"),
        "cost_usd": str(cost),
        "reservation_usd": str(quote["reservation"]),
        "output_key": output_key,
        "termination": termination["status"],
        "active_workers": termination["active"],
        "standby_provider_managed": termination["standby"],
    }
    if op == "video_generate":
        out = status["output"]
        video_seconds = Decimal(str(out.get("video_seconds")))
        row.update(
            {
                "vram_total_mb": out.get("vram_total_mb"),
                "model": out.get("model"),
                "model_load_ms": out.get("model_load_ms"),
                "inference_ms": out.get("inference_ms"),
                "encode_ms": out.get("encode_ms"),
                "frames": out.get("frames"),
                "fps": out.get("fps"),
                "resolution": f"{out.get('width')}x{out.get('height')}",
                "video_seconds": str(video_seconds),
                "output_bytes": out.get("output_bytes"),
                "cost_per_generated_second_usd": str(
                    (cost / video_seconds).quantize(Decimal("0.0001"), rounding=ROUND_UP)
                ),
                "cost_per_generated_minute_usd": str(
                    (cost * 60 / video_seconds).quantize(Decimal("0.01"), rounding=ROUND_UP)
                ),
            }
        )
    if op == "audio_mux":
        out = status["output"]
        row.update(
            {
                "narration_seconds": out.get("narration_seconds"),
                "audio_seconds": out.get("audio_seconds"),
                "video_seconds": out.get("video_seconds"),
                "audio_sample_rate": out.get("audio_sample_rate"),
                "audio_peak_dbfs": out.get("audio_peak_dbfs"),
                "audio_gain_db": out.get("audio_gain_db"),
                "tts_ms": out.get("tts_ms"),
                "mux_ms": out.get("mux_ms"),
                "output_bytes": out.get("output_bytes"),
            }
        )
    _show("job row", row)
    if termination["status"] != TERMINATION_CONFIRMED:
        raise SpendStop(
            "termination-unknown",
            "active compute could not be confirmed zero; not continuing",
        )
    return row


# ----------------------------------------------------- five-scene battery


def check_standby_zero(client) -> None:
    """Owner directive 2026-08-26 (production launch, Phase 6),
    superseding the same-day battery gate: workersStandby is not
    settable by any reachable API (three-surface proof, ledger §16p)
    and the production financial rule concerns ACTIVE COMPUTE, so a
    non-zero standby is RECORDED as STANDBY_PROVIDER_MANAGED and never
    blocks the run — and never lets anyone claim the total worker count
    is zero. Only endpoint ambiguity still stops here."""
    _, endpoints = client.get_endpoints()
    ep_list = endpoints if isinstance(endpoints, list) else endpoints.get("endpoints", [])
    if len(ep_list) != 1:
        raise SpendStop(
            "endpoint-not-singular", f"{len(ep_list)} endpoint(s); need exactly one"
        )
    standby = ep_list[0].get("workersStandby")
    if standby != 0:
        print(
            f"STANDBY_PROVIDER_MANAGED: workersStandby reads {standby!r} — "
            "recorded per owner directive 2026-08-26 (production Phase 6); "
            "termination is judged on active compute, and the total worker "
            "count is never claimed to be zero"
        )


def video_battery(client, facts: dict, *, sleep=time.sleep, clock=time.monotonic) -> list:
    """The owner's five-scene battery: exactly five video jobs, strictly
    sequential, each with its own requote/admission, verification,
    billing reconciliation and termination confirmation — one_job raises
    on ANY failure or UNKNOWN termination, which stops the battery cold
    with no retry and no next submission (Phase 7)."""
    check_standby_zero(client)
    rows = []
    for index, (slug, prompt) in enumerate(VIDEO_BATTERY, start=1):
        print(f"--- scene {index}/5 [{slug}] ---")
        row = one_job(
            client,
            facts,
            output_key=f"{facts['output_prefix']}/battery-{index}-{slug}.mp4",
            op="video_generate",
            prompt=prompt,
            sleep=sleep,
            clock=clock,
        )
        row["scene"] = slug
        rows.append(row)
        print(f"scene {index}/5 [{slug}] PASS — terminated, ${row['cost_usd']}")
    return rows


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
            "termination": termination["status"],
            "standby_provider_managed": termination["standby"],
        }
        _show("failure case", row)
        if not surfaced:
            raise SpendStop("failure-not-surfaced", f"case {name} did not fail loudly")
        if termination["status"] != TERMINATION_CONFIRMED:
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
        facts = preflight(rp, **params, approval_evidence=(argv[1] == "run"))
        if argv[1] == "preflight":
            print("PREFLIGHT PASS — every free gate holds; spending remains "
                  "gated on SPEND + gpu-spend approval")
            return 0

        through = int(os.environ.get("THROUGH_PHASE", "16"))
        op = os.environ.get("OP", "image_preprocess")
        if op not in ("image_preprocess", "video_generate", "audio_mux"):
            raise SpendStop("op-not-allowed", f"unknown OP {op!r}")
        if op == "audio_mux":
            # The audio canary is ONE job by definition: narration muxed
            # onto an EXISTING video named by test_input_key. No battery
            # shape exists for it, deliberately.
            if through != 16:
                raise SpendStop(
                    "audio-through-phase",
                    "audio_mux supports through_phase 16 (one canary) only",
                )
            rows = [
                one_job(
                    rp,
                    facts,
                    output_key=f"{facts['output_prefix']}/final-001.mp4",
                    op=op,
                )
            ]
            print("PHASE 13-16 PASS — one real audio job, verified and terminated")
        elif op == "video_generate":
            # Video knows exactly two shapes (owner directives 2026-08-26):
            # 16 = the single job; 18 = the five-scene battery — EXACTLY
            # five, never 1+5, never twenty. Anything else refuses.
            if through == 16:
                rows = [
                    one_job(
                        rp,
                        facts,
                        output_key=f"{facts['output_prefix']}/ltx-001.mp4",
                        op=op,
                    )
                ]
                print("PHASE 13-16 PASS — one real job, verified and terminated")
            elif through == 18:
                rows = video_battery(rp, facts)
                print(
                    "PHASE 18 PASS — five-scene battery, each job verified "
                    "and terminated"
                )
            else:
                raise SpendStop(
                    "video-through-phase",
                    "video_generate supports through_phase 16 (one job) or "
                    "18 (the five-scene battery) — nothing else",
                )
        else:
            rows = [
                one_job(
                    rp,
                    facts,
                    output_key=f"{facts['output_prefix']}/job-1.jpeg",
                    op=op,
                )
            ]
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
