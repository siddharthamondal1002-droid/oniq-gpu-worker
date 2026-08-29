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

from validation import admission, frame_pull

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
        #
        # 2026-08-28: the body says "Resource not accessible by
        # integration" and stops there. The HEADER says which permission
        # would have worked, so keep it too — guessing the permission
        # from the body is what cost this loop a cycle. The key name
        # deliberately contains no redaction marker (KEY/SECRET/TOKEN/
        # PASSWORD/CREDENTIAL/AUTHORIZATION); naming it *_key would have
        # printed the answer as <redacted>, which already happened once.
        accepted = ""
        try:
            accepted = exc.headers.get("X-Accepted-GitHub-Permissions") or ""
        except Exception:
            accepted = ""
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
            doc = json.loads(body)
        except Exception:
            doc = {"raw": body[:300]}
        if isinstance(doc, dict) and accepted:
            doc["accepted_github_permissions"] = accepted
        return exc.code, doc
    except Exception:
        return 0, {}


def check_environment_protection(fetch=_default_env_fetch) -> int:
    """The gpu-spend environment must exist AND carry a required-reviewer
    rule. Referencing a missing environment silently creates an
    UNPROTECTED one, so 'the job waited for approval' cannot be assumed —
    it must be read back from the API. Returns the reviewer-rule count.

    Note (2026-08-28): clearing a 403 here does NOT make this gate pass.
    PASS requires a required_reviewers rule, and owner directive
    2026-08-25 recorded that required reviewers are not offered on this
    private repository's plan. A readable API therefore turns
    environment-unverifiable into environment-unprotected — an honest
    verified answer rather than a blind spot, which is the whole point,
    but not a green light. The spend path does not come through here at
    all: it uses check_run_approval with APPROVAL_MODE=owner-dispatch."""
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
        accepted = ""
        if isinstance(doc, dict):
            accepted = doc.get("accepted_github_permissions") or ""
        hint = ""
        if status == 403:
            # Until 2026-08-28 this hint named 'deployments: read' as the
            # likely cause. The advisory job has granted exactly that
            # since run 26 and still answers 403, so the guess was wrong
            # and sent a whole loop after a permission already in place.
            # GitHub answers the question itself in the response header;
            # report what it said instead of guessing a second time.
            hint = (
                " — GitHub's X-Accepted-GitHub-Permissions header says: "
                + (accepted or "<header absent>")
                + ". 'deployments: read' is ALREADY granted on this job, so "
                "it is not the cause. Grant exactly the permission named "
                "above, read-only. An absent header means environments are "
                "not readable on this repository's plan at all"
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
    policy: dict | None = None,
    sleep=time.sleep,
    clock=time.monotonic,
) -> dict:
    _, submitted = client.submit_job(endpoint_id, job_input, policy=policy)
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

# The five-shot ACTION BATTERY — owner directive 2026-08-29, replacing
# the 2026-08-26 five-scene battery outright. Each shot is an ACTION
# CONTRACT: a structured statement of who moves, from what state to what
# state, what the camera does and what the environment does. The
# contract is the SOURCE OF TRUTH and stays separate from the LTX
# prompt — the prompt is DERIVED by compile_motion_prompt() and never
# hand-edited per shot, so a quality verdict on a clip always traces
# back to a named field rather than to prompt wording nobody recorded.
# Server-side like VIDEO_PROMPT: the dispatch chooses only
# through_phase=18, never the text. Every output name is unique per
# shot, and none is a calibration fixture basename (measured 2026-08-29
# — see the rename notes at the canary output keys in main()).
ACTION_BATTERY = (
    # Owner directive 2026-08-29 (multi-reference conditioning): each
    # shot names the REFERENCE it actually needs ("plate": a reference
    # id, mapped to a server-derived key — never a path) and carries the
    # owner's shot prompt VERBATIM. The action contract stays separate
    # from the prompt: the contract is what the footage is judged
    # against, the prompt is what the model is asked.
    {
        "slug": "maya-turns",
        "output": "shot-001-maya-turns.mp4",
        "plate": "a",
        "prompt": (
            "Maya stands on the abandoned railway platform beside the "
            "small girl holding a red balloon. Maya slowly turns her "
            "head and upper body toward the distant railway while the "
            "girl remains beside her. Their faces, clothing, body "
            "proportions, red balloon, and surrounding platform remain "
            "visually consistent throughout the shot. Gentle cinematic "
            "push-in."
        ),
        "contract": {
            "subject": "Maya",
            "start_state": "standing on the platform beside the girl",
            "action": "slowly turns her head and upper body toward the "
                      "distant railway",
            "end_state": "facing toward the railway",
            "camera_action": "gentle push-in (secondary to the turn)",
            "environment_action": "the girl remains beside her",
            "required_motion": "head_and_body_rotation",
        },
    },
    {
        "slug": "maya-walks",
        "output": "shot-002-maya-walks.mp4",
        "plate": "a",
        "prompt": (
            "Maya slowly walks along the abandoned railway platform "
            "while the small girl holding the red balloon remains "
            "nearby. Maya takes visible, natural steps forward. Her "
            "face, hair, clothing, body proportions and the girl's "
            "appearance remain consistent throughout the shot. The red "
            "balloon moves naturally with the girl. Gentle cinematic "
            "tracking shot."
        ),
        "contract": {
            "subject": "Maya",
            "start_state": "standing on the platform near the girl",
            "action": "walks along the platform with visible, natural "
                      "steps",
            "end_state": "several steps further along the platform",
            "camera_action": "gentle tracking shot",
            "environment_action": "the balloon moves naturally with the "
                                  "girl",
            "required_motion": "walking_legs_and_body",
        },
    },
    {
        "slug": "train-approaches",
        "output": "shot-003-train-approaches.mp4",
        "plate": "b",
        "prompt": (
            "A black train slowly approaches the abandoned railway "
            "platform from the distance. The train visibly changes "
            "position and becomes progressively closer. Its body, "
            "windows, headlights and structure remain consistent "
            "throughout the shot. The railway environment remains "
            "stable. Cinematic slow forward movement."
        ),
        "contract": {
            "subject": "the black train",
            "start_state": "distant on the railway line",
            "action": "visibly changes position and becomes "
                      "progressively closer to the platform",
            "end_state": "noticeably closer and larger in frame",
            "camera_action": "static; camera movement must not be the "
                             "only source of apparent motion",
            "environment_action": "the railway environment remains "
                                  "stable",
            "required_motion": "train_translation",
        },
    },
    {
        "slug": "train-door-opens",
        "output": "shot-004-train-door-opens.mp4",
        "plate": "b",
        "prompt": (
            "The black train is stopped at the abandoned railway "
            "platform. A clearly visible train door begins closed and "
            "then physically opens. The same train, doorway, windows "
            "and surrounding platform remain consistent throughout the "
            "shot. The door movement is continuous and clearly visible."
        ),
        "contract": {
            "subject": "the train door",
            "start_state": "closed, on the train stopped at the "
                           "platform",
            "action": "physically opens in one continuous visible "
                      "movement",
            "end_state": "open",
            "camera_action": "static",
            "environment_action": "train and platform remain "
                                  "consistent",
            "required_motion": "door_slide",
        },
    },
    {
        "slug": "maya-interacts",
        "output": "shot-005-maya-interacts.mp4",
        "plate": "a",
        "prompt": (
            "The small girl holding the red balloon stands beside Maya "
            "on the abandoned railway platform. The girl slowly raises "
            "one hand and points toward the darkness behind Maya. Maya "
            "notices the gesture and turns slightly toward the girl. "
            "Both characters remain visually consistent throughout the "
            "shot. The red balloon remains visible and attached to the "
            "girl's hand."
        ),
        "contract": {
            "subject": "the girl and Maya",
            "start_state": "standing near each other on the platform",
            "action": "the girl raises one hand and points; Maya "
                      "notices and turns slightly toward her",
            "end_state": "girl's hand raised, Maya turned toward her",
            "camera_action": "static to gentle push-in",
            "environment_action": "the balloon stays attached to the "
                                  "girl's hand",
            "required_motion": "arm_raise_and_reaction",
        },
    },
)


def compile_motion_prompt(shot_contract: dict) -> str:
    """One LTX motion prompt, DERIVED from an action contract.

    Pure text assembly — no I/O, no state, no defaults. The contract
    stays the source of truth: change a field and the prompt, the log
    line and the recorded row all change together, which is the whole
    reason the prompt is compiled rather than written five times by
    hand and drifted five separate ways."""
    return (
        f"{shot_contract['subject']}, {shot_contract['start_state']}, "
        f"{shot_contract['action']}; ends {shot_contract['end_state']}. "
        f"Camera: {shot_contract['camera_action']}. "
        f"Environment: {shot_contract['environment_action']}. "
        "Stable identity, consistent scene, realistic motion."
    )


# ONIQ's own image engine (fully in-house directive, 2026-08-27). Same
# rule as VIDEO_PROMPT: the dispatch never chooses the text.
IMAGE_PROMPT = (
    "A quiet street at night after rain, a single lamp overhead, wet "
    "asphalt reflecting the light. Cinematic, photographic, no text."
)

# THE MULTI-REFERENCE CONDITIONING PLATES — owner directive 2026-08-29.
#
# Two generated plates, plate-001 and plate-002, PROVED that this model
# cannot hold Maya + girl + balloon + train in one text-to-image frame:
# the first kept early clauses and dropped the rest (plus an invented
# man), the second kept one subject and the weather and dropped both the
# girl and the train — and put the lone figure ON the tracks despite an
# explicit "no figures on the tracks", which is negation-blindness
# measured twice. Both PLATE_INVALID, $0.02 of measured evidence.
#
# So the architecture changed instead of the wish: STOP asking one image
# to describe the whole movie. Each shot names the reference it actually
# needs — PLATE_A carries the characters, PLATE_B carries the train —
# and each plate prompt stays inside the two-to-three-element adherence
# budget the failures measured. POSITIVE DESCRIPTIONS ONLY: exclusions
# demonstrably backfire on this model, so there are none.
# PLATE A, MEASURED 2026-08-29 — GATE FAILED, prompt left as the owner
# specified it. Job b1971009-4b48-46b2-b7b1-5e3905d56d7e-u1, A5000,
# 489,326 bytes at validation/out/plate-a.png, $0.01. Retrieved and
# looked at; the byte count matches output_bytes exactly, so these are
# the pixels LTX produced.
#
#   Maya clearly identifiable ........ PASS
#   girl clearly identifiable ........ FAIL  no face, no legs; the lower
#                                            body is translucent and
#                                            dissolves into the sleepers
#   girl properly positioned ......... FAIL  beside Maya, but both stand
#                                            in the track bed, not on the
#                                            platform, and she has no
#                                            ground contact
#   red balloon unmistakable ......... FAIL  no balloon exists anywhere
#   no significant deformation ....... FAIL  (the girl, above)
#   no unwanted character ............ PASS  exactly two figures
#
# The balloon did not come out faint, it came out as something else.
# Saturated red totals 42 px in a 34x9 box (0.01% of frame); loosening
# the threshold grows it to a 66x10 streak ~10 px thick, aspect 6.6. A
# balloon is a compact blob with aspect near 1; every threshold measures
# a CORD. The model kept the hand-holds-string relation, dropped the
# object on the end of it, and resolved the leftover woman + cord +
# small figure into the likeliest scene that fits: walking a dog. The
# second figure even wears a harness.
#
# So the split into two plates did not go far enough. This prompt still
# carries two characters plus a prop plus staging plus weather, which is
# past the ~2-3 element adherence ceiling measured on plate-001 and
# plate-002. Splitting the MOVIE across plates fixed the plate count; it
# did not reduce what any one plate is asked to hold. Fixing that is a
# spec change and belongs to the owner, not to this file.
PLATE_A_PROMPT = (
    "Maya, a young adult woman in a dark coat, stands on an abandoned "
    "railway station platform in the rain. A small girl stands a few "
    "steps beside her on the platform, holding a bright red balloon on "
    "a string. Both characters are fully visible head to toe, with the "
    "empty platform and misty air around them. Rainy, eerie, cinematic, "
    "photographic realism."
)

PLATE_B_PROMPT = (
    "A black passenger train stands on the tracks beside an abandoned "
    "railway station platform in the rain. The train is large and "
    "clearly visible, its dark body, windows and closed doors facing "
    "the platform, the track stretching away behind it. Mist, wet "
    "surfaces, eerie cinematic atmosphere, photographic realism."
)

# Where each plate lives, relative to the run's output prefix. The
# battery derives every input_key from THESE — a reference ID in the
# shot maps to a server-derived key, and no caller-supplied path exists
# anywhere on this route (the same fence inHouseMotion holds).
PLATE_KEYS = {"a": "plate-a.png", "b": "plate-b.png"}

# THE BENCHMARK'S CONTROLLED REFERENCE — owner directive 2026-08-29.
#
# Deliberately the opposite of every plate that came before it. plate-001,
# plate-002 and plate-a each failed because they asked one image model to
# stage a whole scene — two characters, a prop, a train, weather, staging —
# and the measured adherence ceiling is two or three elements. Those failures
# are not this benchmark's problem to re-run: they are the reason this
# reference has ONE subject and nothing else.
#
# The purpose here is NOT to test image adherence. It is to give five video
# models the same starting frame so their MOTION and IDENTITY can be
# compared. A reference the image engine can draw reliably is therefore a
# design requirement, not a compromise — anything harder makes the reference
# itself the variable.
#
# One adult, plain background, upper body, facing camera. No props, no second
# person, no environment, no weather. Positive description only.
PROBE_REFERENCE_PROMPT = (
    "A photographic portrait of one adult woman standing against a plain "
    "light grey studio background. She faces the camera. Her head, "
    "shoulders and upper body are clearly visible and well lit. Sharp "
    "focus, natural skin tones, simple and clean."
)
PROBE_REFERENCE_KEY = "probe-reference.png"

# THE COMMON TEST. Same conceptual action for every candidate, so what is
# compared is temporal and identity capability rather than prompt complexity.
# Camera movement alone does not count and the sentence says so: the required
# motion is the SUBJECT's head and upper body, and the camera is pinned still
# precisely to remove the cheapest way for a model to look alive.
PROBE_ACTION_PROMPT = (
    "The woman slowly turns her head and upper body toward the camera. "
    "Her face, hair, clothing and body proportions stay the same "
    "throughout. The background stays plain and still. The camera does "
    "not move."
)


def verify_image_success(output: dict) -> None:
    """The in-house image engine's proof, ON TOP of verify_gpu_success.

    A still is not a video, so the video evidence does not apply — what
    must hold is that ONIQ's OWN model demonstrably loaded, CUDA
    inference demonstrably ran, and a real sized artifact at the video
    canvas came out. An op that quietly returned a placeholder would fail
    every one of these.
    """
    model = str(output.get("model") or "")
    if not model or model == "missing":
        raise SpendStop("model-unproven", "worker did not report the loaded model")
    if not output.get("model_load_ms"):
        raise SpendStop("model-unproven", "model load time was not measured")
    if not output.get("inference_ms"):
        raise SpendStop("no-inference", "CUDA inference time was not measured")
    if not output.get("output_bytes"):
        raise SpendStop("no-artifact", "no still was written")
    if output.get("format") != contract_image_format():
        raise SpendStop(
            "wrong-format",
            f"still is {output.get('format')!r}, not the contract format",
        )
    if (output.get("width"), output.get("height")) != contract_video_canvas():
        raise SpendStop(
            "wrong-canvas",
            "the still does not match the video canvas it must condition",
        )


def contract_image_format() -> str:
    import contract

    return contract.IMAGE_GEN_FORMAT


def contract_video_canvas() -> tuple:
    import contract

    return (contract.VIDEO_WIDTH, contract.VIDEO_HEIGHT)


def contract_probe_ceiling_ms() -> int:
    """The benchmark's window, in the milliseconds RunPod's policy wants.

    ONE source. If the worker's deadline and the provider's job policy came
    from two numbers, whichever was smaller would kill the job and the other
    would be a comment — and the measurement would be lost to a disagreement
    nobody wrote down.
    """
    import contract

    return contract.PROBE_RUNTIME_CEILING_SECONDS * 1000


def verify_gpu_success(output) -> None:
    """Phase 14: HTTP 200 alone is insufficient, and so is each of these
    alone — all must hold."""
    if not isinstance(output, dict) or output.get("ok") is not True:
        raise SpendStop("job-not-ok", f"worker did not report ok; code={None if not isinstance(output, dict) else output.get('code')}")
    if output.get("device") != "cuda":
        raise SpendStop("not-cuda", "device is not cuda — CPU fallback is not success")
    if output.get("gpu_name") != admission.TARGET_GPU:
        raise SpendStop(
            "wrong-gpu", f"gpu_name is not the owner-settled card ({admission.TARGET_GPU})"
        )
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
    input_key: str | None = None,
    model: str | None = None,
    sleep=time.sleep,
    clock=time.monotonic,
) -> dict:
    """Phases 12-16 for a single job. Fail-closed at every boundary.

    `prompt` may only ever come from this module's own constants
    (VIDEO_PROMPT, PLATE_A_PROMPT/PLATE_B_PROMPT, or an ACTION_BATTERY
    shot's stored prompt) — no caller input reaches it, because the
    workflow exposes no prompt field at all."""
    quote = requote(client)
    payload = {
        "op": op,
        # Per-shot conditioning (owner directive 2026-08-29): a shot that
        # names its own reference passes it here; everything else keeps
        # the run-wide test input. Both are server-derived — no caller
        # path reaches this field.
        "input_key": input_key or facts["input_ref"],
        "output_key": output_key,
    }
    if model is not None:
        # Only model_probe carries this, and the worker's contract admits it
        # on no other op — a benchmark ROW id, never a repository or a path.
        payload["model"] = model
    watch_s = None
    policy = None
    if op == "model_probe":
        # PER-JOB, probe only. The endpoint's own executionTimeoutMs is 600000
        # and stays there: raising it would change the spend bound of every
        # production job. This raises it for THIS job, to the probe ceiling the
        # contract sets — the same window the worker's own deadline uses, so
        # the two agree instead of one killing the other mid-measurement.
        policy = {"executionTimeout": contract_probe_ceiling_ms()}
        watch_s = contract_probe_ceiling_ms() // 1000 + 900
    if op == "image_generate":
        # Text-only by contract: sending an input_key is refused by the
        # worker, so the harness must not send one either.
        payload.pop("input_key")
        payload["params"] = {"prompt": prompt or IMAGE_PROMPT}
        watch_s = admission.RUNTIME_CEILING_SECONDS + 900
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
        policy=policy,
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
    if op == "image_generate":
        verify_image_success(status["output"])
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
    if op == "model_probe":
        # EVERY column the benchmark compares on, surfaced in the run log
        # rather than left inside a raw payload nobody reads. Cost stays out:
        # the worker does not know the live rate, and it is attached below
        # from this run's own quote, labelled as an estimate from measured
        # runtime unless the provider states billing itself.
        out = status["output"]
        row.update(
            {
                "label": out.get("label"),
                "repo": out.get("repo"),
                "revision": out.get("revision"),
                "licence": out.get("licence"),
                "dtype": out.get("dtype"),
                "offload": out.get("offload"),
                "failure": out.get("failure"),
                "resolution": f"{out.get('width')}x{out.get('height')}",
                "frames": out.get("frames"),
                "fps": out.get("fps"),
                "download_ms": out.get("download_ms"),
                "model_load_ms": out.get("model_load_ms"),
                "conditioning_load_ms": out.get("conditioning_load_ms"),
                "inference_ms": out.get("inference_ms"),
                "encode_ms": out.get("encode_ms"),
                "total_wall_ms": out.get("total_wall_ms"),
                "vram_total_bytes": out.get("vram_total_bytes"),
                "peak_allocated_bytes": out.get("peak_allocated_bytes"),
                "peak_reserved_bytes": out.get("peak_reserved_bytes"),
                "disk_total_bytes": out.get("disk_total_bytes"),
                "disk_free_bytes": out.get("disk_free_bytes"),
                "output_bytes": out.get("output_bytes"),
                # The rate is this run's LIVE quote; the product of it and a
                # measured runtime is an estimate and says so.
                "cost_basis": "ESTIMATED FROM MEASURED RUNTIME",
                "live_rate_usd_per_hour": str(quote["price"]),
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


# ----------------------------------------------- five-shot action battery


def record_standby_state(client) -> None:
    """Record workersStandby. It is OBSERVABILITY, not a gate.

    Owner directive 2026-08-27 removes workersStandby == 0 as a blocker
    outright and names what replaces it:

        workersMin == 0 AND workersMax == 1
        AND active GPU pods == 0 AND orphan pods == 0
        AND the GPU allow-list is the target card only

    which is the state that actually means "no idle GPU worker is running
    before the canary". preflight() already enforces every one of those —
    unexpected-pods, check_endpoint_config, endpoint-not-target and
    endpoint-gpu-list-not-exclusive — and the sweep covers orphans, so
    this function's job is to make the provider's number visible, never to
    stop on it.

    It was called check_standby_zero until 2026-08-27, which is the
    misleading part worth naming: the code has never blocked on standby,
    but a reader grepping for the gate found a function whose name
    promised one. Four cycles were spent looking for a control that no
    longer gated anything.

    Superseded directive, 2026-08-26 (production launch, Phase 6):
    workersStandby is not settable by any reachable API (three-surface
    proof, ledger §16p) and the production financial rule concerns ACTIVE
    COMPUTE. Recording it also stops anyone claiming the TOTAL worker
    count is zero. Only endpoint ambiguity still stops here."""
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


def require_reference(facts: dict, key: str, *, fetch=None) -> None:
    """The probe's conditioning image must EXIST before a GPU is rented.

    Run 72 paid for a job whose input had already been deleted from the
    bucket, and the failure looked identical to a model problem. A probe is
    worse: the worker would download tens of gigabytes of weights, load them,
    and only then discover there is nothing to condition on — the entire
    watchdog window spent to learn something a free HTTP read knew.

    Same check require_plates makes, for the one image the benchmark shares.
    """
    base = os.environ.get("R2_PUBLIC_BASE_URL", "")
    if not base:
        raise SpendStop(
            "reference-unverifiable",
            "R2_PUBLIC_BASE_URL is unset, so the probe reference cannot be "
            "confirmed to exist before the GPU is rented",
        )
    getter = fetch or frame_pull._fetch
    try:
        data = getter(frame_pull.public_url(base, key))
    except Exception as exc:  # noqa: BLE001
        raise SpendStop(
            "reference-missing",
            f"{key} could not be read: {type(exc).__name__}: {exc}",
        ) from exc
    if not data or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise SpendStop(
            "reference-missing",
            f"{key} is not a PNG — the reference must be drawn and "
            "inspected before any candidate is probed",
        )
    print(f"reference verified: {key} ({len(data)} bytes)")


def require_plates(facts: dict, *, fetch=None) -> dict:
    """Both conditioning plates, PROVED to exist before a rupee moves.

    The five-shot battery of 2026-08-29 (run 72) submitted a paid job
    whose input_key had been deleted from the bucket; the worker refused
    it in 202ms of billed execution. That was ONE job. This battery
    would repeat the mistake five times, so the plates are fetched over
    the public read base and proved by their own PNG magic BEFORE the
    first submission. Fail-closed: no base means no proof, and no proof
    means no battery — a SpendStop, never a shrug.

    Returns {reference_id: full_key} for the battery to condition on.
    """
    base = os.environ.get("R2_PUBLIC_BASE_URL", "")
    try:
        normalised = frame_pull.normalise_base(base)
    except frame_pull.FramePullError as exc:
        raise SpendStop(
            "plate-unverifiable",
            f"cannot prove the conditioning plates exist ({exc.code}); "
            "set R2_PUBLIC_BASE_URL (public read base) for the run",
        )
    fetcher = fetch or frame_pull._fetch
    keys = {}
    for ref, name in sorted(PLATE_KEYS.items()):
        key = f"{facts['output_prefix']}/{name}"
        try:
            data = fetcher(frame_pull.public_url(normalised, key))
            frame_pull.verify_png(data)
        except Exception as exc:  # noqa: BLE001 — every shape is a refusal
            raise SpendStop(
                "plate-missing",
                f"conditioning plate {ref.upper()} ({key}) did not verify: "
                f"{type(exc).__name__} — generate it before the battery",
            )
        keys[ref] = key
        print(f"plate {ref.upper()} verified: {key} ({len(data)} bytes)")
    return keys


def video_battery(client, facts: dict, *, sleep=time.sleep, clock=time.monotonic, fetch=None) -> list:
    """The owner's five-shot ACTION BATTERY (directive 2026-08-29):
    exactly five video jobs, strictly sequential, each with its own
    requote/admission, verification, billing reconciliation and
    termination confirmation — one_job raises on ANY failure or UNKNOWN
    termination, which stops the battery cold with no retry and no next
    submission (Phase 7). The intent is printed BEFORE each submission
    so the log carries what the shot was supposed to do next to what it
    measurably did — that adjacency is what makes a frame-pull verdict
    arguable from the log alone."""
    plate_keys = require_plates(facts, fetch=fetch)
    record_standby_state(client)
    rows = []
    for index, shot in enumerate(ACTION_BATTERY, start=1):
        slug = shot["slug"]
        shot_contract = shot["contract"]
        plate_key = plate_keys[shot["plate"]]
        print(f"--- shot {index}/5 [{slug}] ---")
        print(f"    conditioned on: plate {shot['plate'].upper()} = {plate_key}")
        print(f"    action: {shot_contract['action']}")
        print(f"    required_motion: {shot_contract['required_motion']}")
        row = one_job(
            client,
            facts,
            output_key=f"{facts['output_prefix']}/{shot['output']}",
            op="video_generate",
            prompt=shot["prompt"],
            input_key=plate_key,
            sleep=sleep,
            clock=clock,
        )
        row["scene"] = slug
        row["contract"] = shot_contract
        row["plate"] = plate_key
        rows.append(row)
        print(f"shot {index}/5 [{slug}] PASS — terminated, ${row['cost_usd']}")
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
        if op not in (
            "image_preprocess",
            "image_generate",
            "video_generate",
            "audio_mux",
            "model_probe",
        ):
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
                    # never final-001: a calibration fixture's basename — a rerun would overwrite it (measured 2026-08-29)
                    output_key=f"{facts['output_prefix']}/audio-final-001.mp4",
                    op=op,
                )
            ]
            print("PHASE 13-16 PASS — one real audio job, verified and terminated")
        elif op == "image_generate":
            # ONE still from ONIQ's own image engine: the conditioning
            # PLATE the action battery animates (owner directive
            # 2026-08-29). Like the audio canary this has exactly one
            # shape: no battery exists for it.
            if through != 16:
                raise SpendStop(
                    "image-through-phase",
                    "image_generate supports through_phase 16 (one still) only",
                )
            # WHICH plate: "a" (characters + balloon) or "b" (train).
            # Owner directive 2026-08-29, multi-reference conditioning —
            # one dispatch draws ONE plate, and the choice maps to a
            # module prompt and a fixed key. plate-001/plate-002 are
            # PLATE_INVALID evidence and are never written again.
            which = os.environ.get("PLATE", "a").strip().lower()
            if which == "ref":
                # The benchmark's controlled reference: ONE subject, drawn
                # once, and all five candidates condition on it.
                key, prompt_text, label = (
                    PROBE_REFERENCE_KEY, PROBE_REFERENCE_PROMPT, "probe reference"
                )
            elif which in PLATE_KEYS:
                key = PLATE_KEYS[which]
                prompt_text = PLATE_A_PROMPT if which == "a" else PLATE_B_PROMPT
                label = f"plate {which.upper()}"
            else:
                raise SpendStop(
                    "plate-unknown",
                    f"PLATE={which!r} — the references are 'a' "
                    "(characters + balloon), 'b' (train) and 'ref' (the "
                    "benchmark's single-subject reference)",
                )
            rows = [
                one_job(
                    rp,
                    facts,
                    output_key=f"{facts['output_prefix']}/{key}",
                    op=op,
                    prompt=prompt_text,
                )
            ]
            print(f"PHASE 13-16 PASS — {label} drawn, verified and terminated")
        elif op == "model_probe":
            # ONE candidate, ONE clip, on the A5000 — owner directive
            # 2026-08-29. Each dispatch names one benchmark row; there is no
            # battery shape, deliberately, so a single bad assumption cannot
            # spend five times over before anyone reads a result.
            import modelprobe

            if through != 16:
                raise SpendStop(
                    "probe-through-phase",
                    "model_probe supports through_phase 16 (one candidate) only",
                )
            candidate = os.environ.get("PROBE_MODEL", "").strip()
            if candidate in modelprobe.NOT_EVALUATED:
                raise SpendStop(
                    "probe-not-evaluated",
                    f"{candidate} is NOT_EVALUATED: "
                    f"{modelprobe.NOT_EVALUATED[candidate]}",
                )
            if candidate not in modelprobe.PROBE_MODELS:
                raise SpendStop(
                    "probe-unknown",
                    f"PROBE_MODEL={candidate!r} — authorised rows are "
                    + ", ".join(sorted(modelprobe.PROBE_MODELS)),
                )
            # THE REFERENCE MUST EXIST BEFORE THE GPU IS RENTED. Run 72 paid
            # for a job whose input had been deleted; the same check that
            # guards the battery guards this.
            reference = f"{facts['output_prefix']}/{PROBE_REFERENCE_KEY}"
            require_reference(facts, reference)
            row = modelprobe.PROBE_MODELS[candidate]
            print(f"probing {row['label']} — {row['repo']} @ {row['revision']}")
            print(f"  conditioned on: {reference}")
            print(f"  shape: {row['width']}x{row['height']}x{row['frames']} "
                  f"@ {row['fps']}fps  ({row['frames'] / row['fps']:.2f}s)")
            print(f"  published weights: {row['download_gib']:.2f} GiB, "
                  f"loading {row['dtype']} with {row['offload']} offload")
            rows = [
                one_job(
                    rp,
                    facts,
                    output_key=f"{facts['output_prefix']}/probe-{candidate}.mp4",
                    op=op,
                    prompt=PROBE_ACTION_PROMPT,
                    input_key=reference,
                    model=candidate,
                )
            ]
            print(f"PHASE 13-16 PASS — {row['label']} probed and terminated")
        elif op == "video_generate":
            # Video knows exactly two shapes (owner directives 2026-08-26
            # and 2026-08-29): 16 = the single canary; 18 = the five-shot
            # action battery — EXACTLY five, never 1+5, never twenty.
            # Anything else refuses.
            if through == 16:
                rows = [
                    one_job(
                        rp,
                        facts,
                        # never ltx-001: a calibration fixture's basename — a rerun would overwrite it (measured 2026-08-29)
                        output_key=f"{facts['output_prefix']}/ltx-canary-001.mp4",
                        op=op,
                    )
                ]
                print("PHASE 13-16 PASS — one real job, verified and terminated")
            elif through == 18:
                rows = video_battery(rp, facts)
                print(
                    "PHASE 18 PASS — five-shot action battery, each job "
                    "verified and terminated"
                )
            else:
                raise SpendStop(
                    "video-through-phase",
                    "video_generate supports through_phase 16 (one job) or "
                    "18 (the five-shot action battery) — nothing else",
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
        # WHAT THIS RUN GENERATED, named for the frame pull that follows.
        #
        # Owner directive 2026-08-29: LTX quality is judged from actual
        # frames, so a run has to say which objects it wrote. Purely
        # additive — nothing below reads this, no gate consults it, and a
        # filesystem error here cannot refuse work that already
        # succeeded. It carries keys and sizes, never a price and never a
        # credential.
        try:
            manifest = frame_pull.write_manifest(rows)
            print(f"artifact manifest: {len(manifest['clips'])} clip(s) named")
        except OSError as exc:
            print(f"artifact manifest not written ({type(exc).__name__})")
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
