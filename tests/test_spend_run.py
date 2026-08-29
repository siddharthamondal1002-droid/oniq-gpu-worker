import json
from decimal import Decimal

import pytest

from validation import admission, spend_run


# ------------------------------------------------------------------ fakes


def _gpu_types_raw(price=0.5, lowest=0.22):
    return json.dumps(
        {
            "data": {
                "gpuTypes": [
                    {
                        "id": "NVIDIA RTX A5000",
                        "displayName": "RTX A5000",
                        "memoryInGb": 24,
                        "secureCloud": True,
                        "communityCloud": True,
                        "securePrice": price,
                        "communityPrice": 0.22,
                        "lowestPrice": {
                            "uninterruptablePrice": lowest,
                            "minimumBidPrice": lowest,
                        },
                    }
                ]
            }
        }
    )


def _endpoint(env=None, gpus=None, mn=0, mx=1, standby=0):
    return {
        "id": "ep-123",
        "name": "oniq-gpu",
        "workersMin": mn,
        "workersMax": mx,
        "workersStandby": standby,
        "gpuTypeIds": gpus or ["NVIDIA RTX A5000"],
        "idleTimeout": 5,
        "env": env
        if env is not None
        else {
            "R2_S3_ENDPOINT": "https://example.r2.dev",
            "R2_ACCESS_KEY_ID": "AKIDVALUE-SHOULD-HIDE",
            "R2_SECRET_ACCESS_KEY": "SUPERSECRETVALUE",
        },
    }


class FakeClient:
    def __init__(self, *, pods=None, endpoints=None, price=0.5,
                 job_statuses=None, health_seq=None):
        import runpod_client as rp

        self.parse_endpoint = rp.parse_endpoint
        self._pods = pods if pods is not None else []
        self._endpoints = endpoints if endpoints is not None else [_endpoint()]
        self._price = price
        self._job_statuses = list(job_statuses or [])
        self._health_seq = list(health_seq or [])
        self.submitted = []
        self.cancelled = []

    def get_pods(self):
        return json.dumps(self._pods), self._pods

    def get_endpoints(self):
        return json.dumps(self._endpoints), self._endpoints

    def gpu_catalogue(self):
        import runpod_client as rp

        raw = _gpu_types_raw(price=self._price)
        doc = json.loads(raw)
        return raw, [rp.parse_gpu_type(g) for g in doc["data"]["gpuTypes"]]

    def submit_job(self, endpoint_id, job_input):
        self.submitted.append((endpoint_id, job_input))
        return "{}", {"id": f"job-{len(self.submitted)}"}

    def job_status(self, endpoint_id, job_id):
        status = self._job_statuses.pop(0)
        return json.dumps(status), dict(status)

    def cancel_job(self, endpoint_id, job_id):
        self.cancelled.append(job_id)
        return 200, "{}"

    def endpoint_health(self, endpoint_id):
        health = self._health_seq.pop(0) if self._health_seq else {
            "workers": {"idle": 0, "initializing": 0, "ready": 0,
                        "running": 0, "throttled": 0, "unhealthy": 0}
        }
        return json.dumps(health), dict(health)

    def sweep_orphans(self):
        return {"pods": 0, "endpoint_min_workers": 0}


class FakeTime:
    def __init__(self):
        self.now = 0.0

    def clock(self):
        return self.now

    def sleep(self, s):
        self.now += s


def _env_ok(url, token):
    return 200, {"protection_rules": [{"type": "required_reviewers"}]}


@pytest.fixture(autouse=True)
def _gh_env(monkeypatch):
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_TOKEN", "gh-token-value")
    monkeypatch.setenv("GITHUB_RUN_ID", "12345")


def _approved(url, token):
    if url.endswith("/approvals"):
        return 200, [{"state": "approved", "user": {"login": "the-owner"}}]
    return _env_ok(url, token)


def _preflight(client, **kw):
    kw.setdefault("input_ref", "in/test.png")
    kw.setdefault("output_prefix", "out/validation")
    kw.setdefault("env_fetch", _env_ok)
    return spend_run.preflight(client, **kw)


# ----------------------------------------------------------------- redact


def test_redact_blanks_secret_shaped_keys_recursively():
    doc = {
        "env": {"R2_SECRET_ACCESS_KEY": "SUPERSECRET", "R2_S3_ENDPOINT": "https://x"},
        "list": [{"apiKey": "abc", "name": "ok"}],
    }
    red = spend_run.redact(doc)
    assert red["env"]["R2_SECRET_ACCESS_KEY"] == "<redacted>"
    assert red["env"]["R2_S3_ENDPOINT"] == "https://x"
    assert red["list"][0]["apiKey"] == "<redacted>"
    assert red["list"][0]["name"] == "ok"


# --------------------------------------------------- environment protection


def test_environment_missing_is_a_stop():
    with pytest.raises(spend_run.SpendStop) as exc:
        spend_run.check_environment_protection(lambda u, t: (404, {}))
    assert exc.value.code == "environment-missing"


def test_environment_without_reviewers_is_a_stop():
    with pytest.raises(spend_run.SpendStop) as exc:
        spend_run.check_environment_protection(
            lambda u, t: (200, {"protection_rules": [{"type": "wait_timer"}]})
        )
    assert exc.value.code == "environment-unprotected"


def test_environment_unverifiable_is_a_stop_not_a_pass():
    with pytest.raises(spend_run.SpendStop) as exc:
        spend_run.check_environment_protection(lambda u, t: (403, {}))
    assert exc.value.code == "environment-unverifiable"


def test_environment_unverifiable_carries_the_servers_own_words():
    # Run #16 (2026-08-25) answered a bare "403" and the cause — token
    # permission vs. plan limitation — was undecidable from the log. The
    # STOP must quote the API's message so the next 403 diagnoses itself.
    with pytest.raises(spend_run.SpendStop) as exc:
        spend_run.check_environment_protection(
            lambda u, t: (403, {"message": "Resource not accessible by integration"})
        )
    assert "Resource not accessible by integration" in exc.value.message


def test_the_403_hint_no_longer_blames_a_permission_that_is_granted():
    # The old hint guessed "the workflow token lacks 'deployments: read'".
    # The advisory job has granted it since run 26 and still 403s, so the
    # guess was false and cost a loop. The message must say so rather
    # than repeat it.
    with pytest.raises(spend_run.SpendStop) as exc:
        spend_run.check_environment_protection(
            lambda u, t: (403, {"message": "Resource not accessible by integration"})
        )
    assert "is not the cause" in exc.value.message
    assert "ALREADY granted" in exc.value.message


def test_the_403_reports_the_permission_github_itself_named():
    # GitHub answers "which permission?" in X-Accepted-GitHub-Permissions.
    # Guessing it from the body is what went wrong; surface the header.
    with pytest.raises(spend_run.SpendStop) as exc:
        spend_run.check_environment_protection(
            lambda u, t: (
                403,
                {
                    "message": "Resource not accessible by integration",
                    "accepted_github_permissions": "actions=read",
                },
            )
        )
    assert "actions=read" in exc.value.message
    assert exc.value.code == "environment-unverifiable"


def test_a_missing_accepted_permissions_header_is_named_not_faked():
    with pytest.raises(spend_run.SpendStop) as exc:
        spend_run.check_environment_protection(
            lambda u, t: (403, {"message": "nope"})
        )
    assert "<header absent>" in exc.value.message


def test_the_accepted_permissions_key_survives_the_redactor():
    # standby_shaped_keys was eaten by _REDACT_MARKERS because it
    # contained "KEY", and three cycles went to a question whose answer
    # printed as <redacted>. This diagnostic must not repeat that.
    shown = spend_run.redact({"accepted_github_permissions": "actions=read"})
    assert shown["accepted_github_permissions"] == "actions=read"


def test_clearing_the_403_still_cannot_pass_without_a_reviewer_rule():
    # The point of the permission fix is unverifiable -> verified, NOT
    # unverifiable -> pass. A readable API with no required_reviewers
    # rule must still stop.
    with pytest.raises(spend_run.SpendStop) as exc:
        spend_run.check_environment_protection(lambda u, t: (200, {"protection_rules": []}))
    assert exc.value.code == "environment-unprotected"


def test_run_approval_passes_on_a_recorded_approval():
    assert spend_run.check_run_approval(_approved) == "the-owner"


def test_owner_dispatch_mode_is_the_recorded_approval(monkeypatch):
    # Owner directive 2026-08-25: required reviewers do not render on
    # this private repo's plan; the owner's authenticated SPEND dispatch
    # is the approval — declared explicitly, never inferred, no API call.
    monkeypatch.setenv("APPROVAL_MODE", "owner-dispatch")

    def explode(url, token):
        raise AssertionError("owner-dispatch mode must not call any API")

    assert spend_run.check_run_approval(explode) == "owner-dispatch"


def test_any_other_approval_mode_keeps_the_evidence_requirement(monkeypatch):
    monkeypatch.setenv("APPROVAL_MODE", "reviewer")

    def fetch(url, token):
        if url.endswith("/approvals"):
            return 200, []
        return _env_ok(url, token)

    with pytest.raises(spend_run.SpendStop) as exc:
        spend_run.check_run_approval(fetch)
    assert exc.value.code == "approval-not-recorded"


def test_run_approval_refuses_a_readable_empty_approvals_list():
    # GitHub waves a job straight through an unprotected environment, so
    # a readable-but-empty approvals list means the mandated pause never
    # happened — even if a reviewer rule was added after dispatch.
    def fetch(url, token):
        if url.endswith("/approvals"):
            return 200, []
        return _env_ok(url, token)

    with pytest.raises(spend_run.SpendStop) as exc:
        spend_run.check_run_approval(fetch)
    assert exc.value.code == "approval-not-recorded"


def test_run_approval_falls_back_to_the_reviewer_rule_when_unreadable():
    def fetch(url, token):
        if url.endswith("/approvals"):
            return 403, {"message": "Resource not accessible by integration"}
        return _env_ok(url, token)

    assert spend_run.check_run_approval(fetch) == "reviewer-rule-verified"


def test_run_approval_unreadable_and_unprotected_is_a_stop():
    def fetch(url, token):
        if url.endswith("/approvals"):
            return 403, {}
        return 200, {"protection_rules": []}

    with pytest.raises(spend_run.SpendStop) as exc:
        spend_run.check_run_approval(fetch)
    assert exc.value.code == "environment-unprotected"


def test_run_approval_requires_the_run_id(monkeypatch):
    monkeypatch.delenv("GITHUB_RUN_ID")
    with pytest.raises(spend_run.SpendStop) as exc:
        spend_run.check_run_approval(_approved)
    assert exc.value.code == "approval-unverifiable"


def test_preflight_on_the_run_path_uses_approval_evidence():
    client = FakeClient()
    facts = spend_run.preflight(
        client,
        input_ref="in/test.png",
        output_prefix="out/validation",
        env_fetch=_approved,
        approval_evidence=True,
    )
    assert facts["reviewer_rules"] == "approved:the-owner"


def test_preflight_on_the_run_path_stops_without_an_approval():
    def fetch(url, token):
        if url.endswith("/approvals"):
            return 200, []
        return _env_ok(url, token)

    client = FakeClient()
    with pytest.raises(spend_run.SpendStop) as exc:
        spend_run.preflight(
            client,
            input_ref="in/test.png",
            output_prefix="out/validation",
            env_fetch=fetch,
            approval_evidence=True,
        )
    assert exc.value.code == "approval-not-recorded"


def test_default_env_fetch_keeps_the_error_body(monkeypatch):
    import io
    import urllib.error
    import urllib.request

    def fake_urlopen(req, timeout=0):
        raise urllib.error.HTTPError(
            req.full_url, 403, "Forbidden", {},
            io.BytesIO(b'{"message": "Resource not accessible by integration"}'),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    status, doc = spend_run._default_env_fetch("https://api.github.com/x", "tok")
    assert status == 403
    assert doc["message"] == "Resource not accessible by integration"


def test_environment_with_reviewers_passes():
    assert spend_run.check_environment_protection(_env_ok) == 1


def test_missing_token_is_unverifiable(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN")
    with pytest.raises(spend_run.SpendStop) as exc:
        spend_run.check_environment_protection(_env_ok)
    assert exc.value.code == "environment-unverifiable"


# --------------------------------------------------------------- preflight


def test_preflight_happy_path_reserves_thirteen_cents():
    facts = _preflight(FakeClient())
    assert facts["endpoint_id"] == "ep-123"
    assert facts["reservation_usd"] == "0.13"
    assert facts["headroom_usd"] == "0.37"
    assert facts["reviewer_rules"] == 1


def test_preflight_refuses_existing_pods():
    with pytest.raises(spend_run.SpendStop) as exc:
        _preflight(FakeClient(pods=[{"id": "p1"}]))
    assert exc.value.code == "unexpected-pods"


def test_preflight_refuses_zero_endpoints():
    with pytest.raises(spend_run.SpendStop) as exc:
        _preflight(FakeClient(endpoints=[]))
    assert exc.value.code == "endpoint-not-singular"


def test_preflight_refuses_ambiguous_endpoints_without_id():
    two = [_endpoint(), dict(_endpoint(), id="ep-456")]
    with pytest.raises(spend_run.SpendStop) as exc:
        _preflight(FakeClient(endpoints=two))
    assert exc.value.code == "endpoint-not-singular"
    facts = _preflight(FakeClient(endpoints=two), endpoint_id="ep-456")
    assert facts["endpoint_id"] == "ep-456"


def test_preflight_refuses_unparsed_endpoint_fields():
    weird = {"id": "ep-1", "minWorkers": 0, "maxWorkers": 1,
             "gpuTypeIds": ["NVIDIA RTX A5000"], "env": _endpoint()["env"]}
    with pytest.raises(spend_run.SpendStop) as exc:
        _preflight(FakeClient(endpoints=[weird]))
    assert exc.value.code == "endpoint-fields-unparsed"


def test_preflight_refuses_non_target_endpoint():
    with pytest.raises(spend_run.SpendStop) as exc:
        _preflight(FakeClient(endpoints=[_endpoint(gpus=["NVIDIA GeForce RTX 3090"])]))
    assert exc.value.code == "endpoint-not-target"


def test_preflight_refuses_bad_worker_bounds():
    with pytest.raises(admission.AdmissionRefused):
        _preflight(FakeClient(endpoints=[_endpoint(mn=1, mx=1)]))


def test_preflight_names_missing_r2_vars_only():
    env = {"R2_S3_ENDPOINT": "https://x"}
    with pytest.raises(spend_run.SpendStop) as exc:
        _preflight(FakeClient(endpoints=[_endpoint(env=env)]))
    assert exc.value.code == "r2-env-missing"
    assert "R2_ACCESS_KEY_ID" in exc.value.message
    assert "R2_S3_ENDPOINT" not in exc.value.message


def test_preflight_never_prints_endpoint_secret_values(capsys):
    _preflight(FakeClient())
    printed = capsys.readouterr().out
    assert "SUPERSECRETVALUE" not in printed
    assert "AKIDVALUE-SHOULD-HIDE" not in printed
    assert "R2_SECRET_ACCESS_KEY" in printed  # the NAME is verified


def test_preflight_requires_test_refs():
    with pytest.raises(spend_run.SpendStop) as exc:
        spend_run.preflight(FakeClient(), input_ref="", output_prefix="",
                            env_fetch=_env_ok)
    assert exc.value.code == "test-refs-missing"


def test_preflight_stops_when_price_over_cap():
    with pytest.raises(admission.AdmissionRefused) as exc:
        _preflight(FakeClient(price=2.04))
    assert exc.value.code == "over-job-cap"


# ---------------------------------------------------------------- one job


def _good_status(execution_ms=8000):
    return {
        "status": "COMPLETED",
        "delayTime": 4000,
        "executionTime": execution_ms,
        "output": {
            "ok": True,
            "op": "image_preprocess",
            "output_key": "out/validation/job-1.jpeg",
            "device": "cuda",
            "gpu_name": "NVIDIA RTX A5000",
            "vram_total_mb": 24576,
            "vram_peak_mb": 812,
            "output_bytes": 51234,
            "width": 1024,
            "height": 768,
            "format": "jpeg",
            "duration_ms": 7500,
            "cleanup_ok": True,
        },
    }


def _run_one(client):
    ft = FakeTime()
    facts = _preflight(client)
    return spend_run.one_job(
        client, facts, output_key="out/validation/job-1.jpeg",
        sleep=ft.sleep, clock=ft.clock,
    )


def test_one_job_happy_path_row():
    client = FakeClient(job_statuses=[_good_status()])
    row = _run_one(client)
    assert row["termination"] == spend_run.TERMINATION_CONFIRMED
    assert row["cost_usd"] == "0.01"  # 8s at 0.5/h, ceiled to the cent
    assert row["gpu_name"] == "NVIDIA RTX A5000"
    assert client.submitted[0][1]["op"] == "image_preprocess"


def test_one_job_refuses_cpu_device():
    status = _good_status()
    status["output"]["device"] = "cpu"
    with pytest.raises(spend_run.SpendStop) as exc:
        _run_one(FakeClient(job_statuses=[status]))
    assert exc.value.code == "not-cuda"


def test_one_job_refuses_wrong_gpu():
    status = _good_status()
    status["output"]["gpu_name"] = "NVIDIA RTX 4090"
    with pytest.raises(spend_run.SpendStop) as exc:
        _run_one(FakeClient(job_statuses=[status]))
    assert exc.value.code == "wrong-gpu"


def test_gpu_verify_tracks_the_owner_settled_target():
    # Regression, 2026-08-27 canary: verify_gpu_success carried a
    # hardcoded "3090" after the owner retargeted admission to the
    # A5000, so a generation that SUCCEEDED on the settled card was
    # stopped as wrong-gpu. The verify must follow admission.TARGET_GPU
    # — the retired card refuses, and only the exact settled name passes.
    retired = _good_status()
    retired["output"]["gpu_name"] = "NVIDIA GeForce RTX 3090"
    with pytest.raises(spend_run.SpendStop) as exc:
        _run_one(FakeClient(job_statuses=[retired]))
    assert exc.value.code == "wrong-gpu"
    assert admission.TARGET_GPU in exc.value.message

    good = _good_status()
    assert good["output"]["gpu_name"] == admission.TARGET_GPU
    spend_run.verify_gpu_success(good["output"])


def test_one_job_requires_vram_peak_and_artifact():
    status = _good_status()
    status["output"]["vram_peak_mb"] = None
    with pytest.raises(spend_run.SpendStop) as exc:
        _run_one(FakeClient(job_statuses=[status]))
    assert exc.value.code == "no-vram-peak"


def test_one_job_rejects_unwhitelisted_schema():
    status = _good_status()
    status["output"]["internal_path"] = "/tmp/x"
    with pytest.raises(spend_run.SpendStop) as exc:
        _run_one(FakeClient(job_statuses=[status]))
    assert exc.value.code == "schema-violation"


def test_one_job_billing_unavailable_is_a_stop():
    status = _good_status()
    status["executionTime"] = None
    with pytest.raises(spend_run.SpendStop) as exc:
        _run_one(FakeClient(job_statuses=[status]))
    assert exc.value.code == "billing-unavailable"


def test_one_job_fails_closed_when_actual_exceeds_reservation():
    # 1000s at $0.5/h computes to $0.14 > the $0.13 reservation.
    with pytest.raises(spend_run.SpendStop) as exc:
        _run_one(FakeClient(job_statuses=[_good_status(execution_ms=1_000_000)]))
    assert exc.value.code == "actual-over-reservation"


def test_one_job_failed_status_is_a_stop():
    with pytest.raises(spend_run.SpendStop) as exc:
        _run_one(FakeClient(job_statuses=[{"status": "FAILED", "output": {}}]))
    assert exc.value.code == "job-failed"


def test_one_job_unknown_termination_is_a_stop():
    stuck = [{"workers": {"idle": 0, "running": 1, "initializing": 0}}] * 200
    client = FakeClient(job_statuses=[_good_status()], health_seq=stuck)
    with pytest.raises(spend_run.SpendStop) as exc:
        _run_one(client)
    assert exc.value.code == "termination-unknown"


def test_submit_deadline_cancels_and_stops():
    ft = FakeTime()
    pending = [{"status": "IN_PROGRESS"}] * 1000
    client = FakeClient(job_statuses=pending)
    with pytest.raises(spend_run.SpendStop) as exc:
        spend_run.submit_and_wait(client, "ep-123", {"op": "image_preprocess"},
                                  sleep=ft.sleep, clock=ft.clock)
    assert exc.value.code == "job-deadline"
    assert client.cancelled == ["job-1"]


# On $0.50/h, 180s of execution computes to $0.025 -> ceiled to $0.03.
def _good_video_status(execution_ms=180_000):
    return {
        "status": "COMPLETED",
        "delayTime": 300_000,  # first pull of a model-baked image
        "executionTime": execution_ms,
        "output": {
            "ok": True,
            "op": "video_generate",
            "output_key": "out/validation/ltx-canary-001.mp4",
            "device": "cuda",
            "gpu_name": "NVIDIA RTX A5000",
            "vram_total_mb": 24576,
            "vram_peak_mb": 9000,
            "model": "Lightricks/LTX-Video-0.9.7-distilled#distilled",
            "model_load_ms": 41_000,
            "inference_ms": 95_000,
            "encode_ms": 3_500,
            "frames": 97,
            "fps": 24,
            "video_seconds": 4.04,
            "width": 704,
            "height": 480,
            "format": "mp4",
            "output_bytes": 2_400_000,
            "duration_ms": 160_000,
            "cleanup_ok": True,
        },
    }


def _run_one_video(client):
    ft = FakeTime()
    facts = _preflight(client)
    return spend_run.one_job(
        client, facts, output_key="out/validation/ltx-canary-001.mp4",
        op="video_generate", sleep=ft.sleep, clock=ft.clock,
    )


def test_one_video_job_submits_the_server_prompt_only():
    client = FakeClient(job_statuses=[_good_video_status()])
    _run_one_video(client)
    payload = client.submitted[0][1]
    assert payload["op"] == "video_generate"
    assert payload["params"] == {"prompt": spend_run.VIDEO_PROMPT}


def test_one_video_job_row_carries_measured_economics():
    client = FakeClient(job_statuses=[_good_video_status()])
    row = _run_one_video(client)
    assert row["op"] == "video_generate"
    assert row["model"].endswith("#distilled")
    assert row["resolution"] == "704x480"
    assert row["frames"] == 97
    assert row["video_seconds"] == "4.04"
    assert row["cost_usd"] == "0.03"
    # $0.03 / 4.04s and *60, both ceiled — never rounded down.
    assert row["cost_per_generated_second_usd"] == "0.0075"
    assert row["cost_per_generated_minute_usd"] == "0.45"
    assert row["termination"] == spend_run.TERMINATION_CONFIRMED


def test_video_watch_survives_a_long_cold_pull():
    # 200 polls x 5s = 1000s of queue: past the 900s default watch, well
    # inside the video watch (ceiling + 900). The 900s runtime ceiling
    # still binds EXECUTION; the watch is wall-clock around it.
    long_queue = [{"status": "IN_QUEUE"}] * 200 + [_good_video_status()]
    client = FakeClient(job_statuses=long_queue)
    row = _run_one_video(client)
    assert row["cost_usd"] == "0.03"
    assert client.cancelled == []


def test_verify_video_success_passes_on_full_evidence():
    spend_run.verify_video_success(_good_video_status()["output"])


@pytest.mark.parametrize(
    "patch,code",
    [
        ({"model": "missing"}, "model-unproven"),
        ({"model": ""}, "model-unproven"),
        ({"model_load_ms": 0}, "model-unproven"),
        ({"inference_ms": None}, "no-inference"),
        ({"frames": 0}, "no-frames"),
        ({"video_seconds": 0}, "no-frames"),
        ({"encode_ms": None}, "no-encode"),
    ],
)
def test_verify_video_success_stops_on_each_missing_proof(patch, code):
    output = _good_video_status()["output"]
    output.update(patch)
    with pytest.raises(spend_run.SpendStop) as exc:
        spend_run.verify_video_success(output)
    assert exc.value.code == code


def test_verify_video_success_accepts_a_zero_ms_encode():
    # 0 is a measured value; only an ABSENT encode time is a stop.
    output = _good_video_status()["output"]
    output["encode_ms"] = 0
    spend_run.verify_video_success(output)


def test_one_video_job_stops_when_the_worker_proves_no_model():
    status = _good_video_status()
    status["output"]["model"] = "missing"
    client = FakeClient(job_statuses=[status])
    with pytest.raises(spend_run.SpendStop) as exc:
        _run_one_video(client)
    assert exc.value.code == "model-unproven"


# ----------------------------------------------- five-shot action battery
# Owner directive 2026-08-29: the ACTION BATTERY replaces the five-scene
# VIDEO_BATTERY. Each shot is an action CONTRACT — the source of truth,
# kept separate from the LTX prompt, which is DERIVED — and every output
# name is unique so no two shots (and no shot and any fixture) can
# collide in the bucket.


def test_action_battery_holds_five_complete_unique_shots():
    assert not hasattr(spend_run, "VIDEO_BATTERY")  # replaced, not aliased
    assert len(spend_run.ACTION_BATTERY) == 5
    slugs = [shot["slug"] for shot in spend_run.ACTION_BATTERY]
    outputs = [shot["output"] for shot in spend_run.ACTION_BATTERY]
    assert len(set(slugs)) == 5, slugs
    assert len(set(outputs)) == 5, outputs
    required = {
        "subject", "start_state", "action", "end_state",
        "camera_action", "environment_action", "required_motion",
    }
    for shot in spend_run.ACTION_BATTERY:
        assert set(shot["contract"]) == required, shot["slug"]
        for field, value in shot["contract"].items():
            assert isinstance(value, str) and value.strip(), (
                f"{shot['slug']}.{field} is empty"
            )


def test_compiled_prompts_are_bounded_and_carry_the_contract():
    import contract

    for shot in spend_run.ACTION_BATTERY:
        prompt = spend_run.compile_motion_prompt(shot["contract"])
        # Under the worker contract's ceiling, with room to spare — a
        # contract edit must not silently push a prompt over the wall.
        assert 0 < len(prompt) < contract.MAX_PROMPT_CHARS
        # The prompt is DERIVED: the subject and the action must survive
        # compilation verbatim, or the clip cannot be judged against the
        # contract that requested it.
        assert shot["contract"]["subject"] in prompt
        assert shot["contract"]["action"] in prompt
        contract.validate_job(
            {
                "op": "video_generate",
                "input_key": "in/a.jpg",
                "output_key": f"out/{shot['output']}",
                "params": {"prompt": prompt},
            }
        )


def test_no_spend_run_output_can_hit_a_calibration_fixture_basename():
    # MEASURED 2026-08-29: the three immutable calibration fixtures live
    # at validation/out/ltx-001.mp4, validation/audio-canary/final-001.mp4
    # and validation/video-test/ltx-001.mp4 — EXACTLY the keys the old
    # canaries wrote, so one ordinary re-run would have silently
    # overwritten the only real calibration data ONIQ has. Every output
    # key in spend_run is born as an f-string over output_prefix, so a
    # forbidden basename would have to appear in the module source as
    # '/<name>' — assert it never does.
    with open(spend_run.__file__, encoding="utf-8") as handle:
        source = handle.read()
    assert "/ltx-001.mp4" not in source
    assert "/final-001.mp4" not in source
    # And the battery's names, built exactly as video_battery builds them.
    forbidden = {"ltx-001.mp4", "final-001.mp4"}
    built = {
        f"out/validation/{shot['output']}".rsplit("/", 1)[1]
        for shot in spend_run.ACTION_BATTERY
    }
    assert not (built & forbidden), built & forbidden


def test_video_battery_runs_exactly_five_shots_in_order(capsys):
    client = FakeClient(job_statuses=[_good_video_status() for _ in range(5)])
    ft = FakeTime()
    facts = _preflight(client)
    rows = spend_run.video_battery(client, facts, sleep=ft.sleep, clock=ft.clock)
    assert len(rows) == 5
    assert len(client.submitted) == 5
    sent_prompts = [payload["params"]["prompt"] for _, payload in client.submitted]
    assert sent_prompts == [
        spend_run.compile_motion_prompt(shot["contract"])
        for shot in spend_run.ACTION_BATTERY
    ]
    keys = [payload["output_key"] for _, payload in client.submitted]
    assert keys == [
        "out/validation/shot-001-maya-turns.mp4",
        "out/validation/shot-002-maya-walks.mp4",
        "out/validation/shot-003-train-approaches.mp4",
        "out/validation/shot-004-train-door-opens.mp4",
        "out/validation/shot-005-maya-interacts.mp4",
    ]
    assert [r["scene"] for r in rows] == [
        "maya-turns", "maya-walks", "train-approaches",
        "train-door-opens", "maya-interacts",
    ]
    # The contract rides in the row, so the report can put the verdict
    # next to the intent without a second lookup.
    assert [r["contract"] for r in rows] == [
        shot["contract"] for shot in spend_run.ACTION_BATTERY
    ]
    assert all(r["termination"] == spend_run.TERMINATION_CONFIRMED for r in rows)
    assert all(r["vram_total_mb"] == 24576 for r in rows)
    # The log carries the intent next to the result: slug, action and
    # required_motion are printed BEFORE each submission.
    out = capsys.readouterr().out
    for shot in spend_run.ACTION_BATTERY:
        assert shot["slug"] in out
        assert shot["contract"]["action"] in out
        assert shot["contract"]["required_motion"] in out


def test_video_battery_records_provider_managed_standby_and_proceeds(capsys):
    # Owner directive 2026-08-26 (production Phase 6), superseding the
    # same-day blocking gate: standby is provider-managed and not
    # settable by any reachable API — RECORD it and run on active
    # compute; never claim the total worker count is zero.
    client = FakeClient(endpoints=[_endpoint(standby=1)],
                        job_statuses=[_good_video_status() for _ in range(5)])
    ft = FakeTime()
    facts = _preflight(client)
    rows = spend_run.video_battery(client, facts, sleep=ft.sleep, clock=ft.clock)
    assert len(rows) == 5
    assert "STANDBY_PROVIDER_MANAGED" in capsys.readouterr().out


def test_video_battery_stops_midway_with_no_retry_and_no_next_job():
    statuses = [
        _good_video_status(),
        _good_video_status(),
        {"status": "FAILED", "output": {}},
    ]
    client = FakeClient(job_statuses=statuses)
    ft = FakeTime()
    facts = _preflight(client)
    with pytest.raises(spend_run.SpendStop) as exc:
        spend_run.video_battery(client, facts, sleep=ft.sleep, clock=ft.clock)
    assert exc.value.code == "job-failed"
    # Scene 3 failed: it was submitted once (no retry) and scenes 4-5
    # were never submitted.
    assert len(client.submitted) == 3


def test_video_battery_stops_on_unknown_termination_midway():
    stuck = [{"workers": {"idle": 0, "running": 1, "initializing": 0}}] * 200
    client = FakeClient(job_statuses=[_good_video_status()], health_seq=stuck)
    ft = FakeTime()
    facts = _preflight(client)
    with pytest.raises(spend_run.SpendStop) as exc:
        spend_run.video_battery(client, facts, sleep=ft.sleep, clock=ft.clock)
    assert exc.value.code == "termination-unknown"
    assert len(client.submitted) == 1


def test_one_job_uses_the_compiled_shot_prompt_when_given():
    client = FakeClient(job_statuses=[_good_video_status()])
    ft = FakeTime()
    facts = _preflight(client)
    shot = spend_run.ACTION_BATTERY[1]
    prompt = spend_run.compile_motion_prompt(shot["contract"])
    spend_run.one_job(
        client, facts, output_key=f"out/validation/{shot['output']}",
        op="video_generate", prompt=prompt,
        sleep=ft.sleep, clock=ft.clock,
    )
    assert client.submitted[0][1]["params"]["prompt"] == prompt


# ------------------------------------------------------------- termination


def test_confirm_termination_requires_a_parsed_zero():
    ft = FakeTime()
    client = FakeClient(health_seq=[{"unexpected": "shape"}] * 200)
    out = spend_run.confirm_termination(client, "ep-123", sleep=ft.sleep, clock=ft.clock)
    assert out["status"] == spend_run.TERMINATION_UNKNOWN
    assert out["standby"] is None


def test_confirm_termination_confirms_on_zero_active_compute():
    ft = FakeTime()
    client = FakeClient(
        health_seq=[{"workers": {"idle": 0, "running": 1, "initializing": 0}},
                    {"workers": {"idle": 0, "running": 0, "initializing": 0}}]
    )
    out = spend_run.confirm_termination(client, "ep-123", sleep=ft.sleep, clock=ft.clock)
    assert out["status"] == spend_run.TERMINATION_CONFIRMED
    assert out["standby"] == 0


def test_confirm_termination_records_provider_managed_standby(capsys):
    # Owner directive 2026-08-26 (production Phase 6): active compute
    # zero CONFIRMS; the provider-managed standby pool is RECORDED —
    # idle and ready overlap, so the pool is their max, never a sum.
    ft = FakeTime()
    client = FakeClient(
        health_seq=[{"workers": {"idle": 1, "ready": 1, "running": 0,
                                 "initializing": 0, "throttled": 0,
                                 "unhealthy": 0}}]
    )
    out = spend_run.confirm_termination(client, "ep-123", sleep=ft.sleep, clock=ft.clock)
    assert out["status"] == spend_run.TERMINATION_CONFIRMED
    assert out["standby"] == 1
    assert "STANDBY_PROVIDER_MANAGED" in capsys.readouterr().out


def test_confirm_termination_needs_the_owner_named_pair_measured():
    # running alone is not enough — initializing must also be a MEASURED
    # zero; a payload that omits it never confirms.
    ft = FakeTime()
    client = FakeClient(health_seq=[{"workers": {"running": 0}}] * 200)
    out = spend_run.confirm_termination(client, "ep-123", sleep=ft.sleep, clock=ft.clock)
    assert out["status"] == spend_run.TERMINATION_UNKNOWN


# --------------------------------------------------------- failure battery


def _refusal_status(code):
    return {
        "status": "COMPLETED",
        "executionTime": 1200,
        "delayTime": 3000,
        "output": {"ok": False, "code": code, "error": "surfaced"},
    }


def test_failure_battery_requires_each_case_to_surface():
    client = FakeClient(job_statuses=[
        _refusal_status("r2-read-failed"),
        _refusal_status("op-not-allowed"),
        _refusal_status("invalid-input"),
    ])
    ft = FakeTime()
    facts = _preflight(client)
    rows = spend_run.failure_battery(client, facts, sleep=ft.sleep, clock=ft.clock)
    assert [r["case"] for r in rows] == [
        "r2-read-failure", "contract-refusal-op", "contract-refusal-params",
    ]
    assert all(r["surfaced"] for r in rows)


def test_failure_battery_stops_on_a_silent_failure():
    client = FakeClient(job_statuses=[_refusal_status("unexpected-exception")])
    ft = FakeTime()
    facts = _preflight(client)
    with pytest.raises(spend_run.SpendStop) as exc:
        spend_run.failure_battery(client, facts, sleep=ft.sleep, clock=ft.clock)
    assert exc.value.code == "failure-not-surfaced"


# --------------------------------------------------------------- economics


def test_economics_over_real_rows():
    rows = [
        {"cost_usd": "0.01", "execution_ms": 8000, "delay_ms": 4000},
        {"cost_usd": "0.02", "execution_ms": 9000, "delay_ms": 5000},
    ]
    out = spend_run.economics(rows)
    assert out["jobs"] == 2
    assert out["total_cost_usd"] == "0.03"
    assert Decimal(out["cost_per_1000_usd"]) >= Decimal("15.00")
    assert 0 < out["gpu_utilization"] < 1


def test_economics_refuses_zero_rows():
    with pytest.raises(spend_run.SpendStop) as exc:
        spend_run.economics([])
    assert exc.value.code == "no-data"


def test_actual_cost_rounds_up():
    assert spend_run.actual_cost_usd(1000, Decimal("0.5")) == Decimal("0.01")
    assert spend_run.actual_cost_usd(80_000, Decimal("0.5")) == Decimal("0.02")


def test_preflight_refuses_a_non_exclusive_gpu_list():
    # Measured 2026-08-25: the created endpoint listed A5000 and L4
    # beside the 3090 — a scheduler could allocate the wrong card and
    # bill its boot before Phase 14 refuses it.
    mixed = _endpoint(gpus=["NVIDIA RTX A5000", "NVIDIA L4",
                            "NVIDIA GeForce RTX 3090"])
    with pytest.raises(spend_run.SpendStop) as exc:
        _preflight(FakeClient(endpoints=[mixed]))
    assert exc.value.code == "endpoint-gpu-list-not-exclusive"
    assert "NVIDIA L4" in exc.value.message


def test_preflight_finds_r2_env_on_the_template():
    # RunPod may store env vars on the template rather than the endpoint
    # object; the check must look in both before declaring them missing.
    ep = _endpoint(env={})
    ep["templateId"] = "tpl-1"
    client = FakeClient(endpoints=[ep])
    client.get_template = lambda tid: (
        "{}",
        {"id": tid, "env": {
            "R2_S3_ENDPOINT": "https://x",
            "R2_ACCESS_KEY_ID": "id",
            "R2_SECRET_ACCESS_KEY": "sec",
        }},
    )
    facts = _preflight(client)
    assert facts["endpoint_id"] == "ep-123"


def test_preflight_stops_on_a_positive_env_miss():
    # An env set IS visible and lacks the names: hard stop.
    ep = _endpoint(env={"OTHER_VAR": "x"})
    ep["templateId"] = "tpl-1"
    client = FakeClient(endpoints=[ep])
    client.get_endpoint = lambda eid: ("{}", {"id": eid})
    client.get_template = lambda tid: ("{}", {"id": tid, "env": {}})
    client.template_env_names_graphql = lambda tid: None
    with pytest.raises(spend_run.SpendStop) as exc:
        _preflight(client)
    assert exc.value.code == "r2-env-missing"


def test_preflight_warns_and_passes_when_env_is_unreadable(capsys):
    # No API view exposes env at all: proceed loudly; the worker's own
    # storage-not-configured fail-closed is the runtime verifier.
    ep = _endpoint(env={})
    ep["templateId"] = "tpl-1"
    client = FakeClient(endpoints=[ep])
    client.get_endpoint = lambda eid: ("{}", {"id": eid})
    def _no_rest(tid):
        raise RuntimeError("404")
    client.get_template = _no_rest
    client.template_env_names_graphql = lambda tid: None
    facts = _preflight(client)
    assert facts["endpoint_id"] == "ep-123"
    assert "r2-env-unverifiable" in capsys.readouterr().out


def test_redact_blanks_pair_form_secret_values():
    pair = {"key": "R2_SECRET_ACCESS_KEY", "value": "LEAKME"}
    red = spend_run.redact({"env": [pair, {"key": "HOME", "value": "/x"}]})
    assert red["env"][0]["value"] == "<redacted>"
    assert red["env"][1]["value"] == "/x"


def test_preflight_reads_env_from_single_endpoint_get():
    ep = _endpoint(env={})
    client = FakeClient(endpoints=[ep])
    client.get_endpoint = lambda eid: ("{}", {"id": eid, "env": {
        "R2_S3_ENDPOINT": "https://x", "R2_ACCESS_KEY_ID": "i",
        "R2_SECRET_ACCESS_KEY": "s"}})
    facts = _preflight(client)
    assert facts["endpoint_id"] == "ep-123"


def test_preflight_reads_env_names_from_graphql_template():
    ep = _endpoint(env={})
    ep["templateId"] = "tpl-1"
    client = FakeClient(endpoints=[ep])
    client.get_endpoint = lambda eid: ("{}", {"id": eid})
    def _no_rest_template(tid):
        raise RuntimeError("404")
    client.get_template = _no_rest_template
    client.template_env_names_graphql = lambda tid: {
        "R2_S3_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"}
    facts = _preflight(client)
    assert facts["endpoint_id"] == "ep-123"


# ---------------------------------------------------------- audio canary


def _good_audio_status(execution_ms=9000):
    return {
        "status": "COMPLETED",
        "delayTime": 3000,
        "executionTime": execution_ms,
        "output": {
            "ok": True,
            "op": "audio_mux",
            "output_key": "out/validation/audio-final-001.mp4",
            "has_audio": True,
            "narration_seconds": 1.71,
            "audio_seconds": 4.042,
            "video_seconds": 4.042,
            "audio_sample_rate": 22050,
            "audio_peak_dbfs": -3.8,
            "audio_gain_db": -3.7,
            "tts_ms": 1731,
            "mux_ms": 45,
            "output_bytes": 991_234,
            "format": "mp4",
            "duration_ms": 8200,
            "cleanup_ok": True,
        },
    }


def _run_one_audio(client):
    ft = FakeTime()
    facts = _preflight(client)
    return spend_run.one_job(
        client, facts, output_key="out/validation/audio-final-001.mp4",
        op="audio_mux", sleep=ft.sleep, clock=ft.clock,
    )


def test_one_audio_job_submits_the_module_narration_only():
    client = FakeClient(job_statuses=[_good_audio_status()])
    _run_one_audio(client)
    payload = client.submitted[0][1]
    assert payload["op"] == "audio_mux"
    assert payload["params"] == {"narration": spend_run.AUDIO_NARRATION}


def test_one_audio_job_row_carries_measured_audio_evidence():
    client = FakeClient(job_statuses=[_good_audio_status()])
    row = _run_one_audio(client)
    assert row["op"] == "audio_mux"
    assert row["narration_seconds"] == 1.71
    assert row["audio_sample_rate"] == 22050
    assert row["tts_ms"] == 1731 and row["mux_ms"] == 45
    assert row["termination"] == spend_run.TERMINATION_CONFIRMED


def test_audio_job_needs_no_cuda_evidence_but_all_audio_evidence():
    # CPU by design: the cuda/vram checks must NOT gate this op...
    status = _good_audio_status()
    assert "device" not in status["output"]
    client = FakeClient(job_statuses=[status])
    row = _run_one_audio(client)
    assert row["gpu_name"] is None
    # ...and the audio evidence must. Silence with extra steps refuses.
    silent = _good_audio_status()
    silent["output"]["audio_peak_dbfs"] = -71.0
    with pytest.raises(spend_run.SpendStop) as exc:
        _run_one_audio(FakeClient(job_statuses=[silent]))
    assert exc.value.code == "audio-silent"


def test_audio_job_refuses_a_drifted_mux():
    status = _good_audio_status()
    status["output"]["audio_seconds"] = 3.0
    with pytest.raises(spend_run.SpendStop) as exc:
        _run_one_audio(FakeClient(job_statuses=[status]))
    assert exc.value.code == "audio-drift"


def test_audio_job_refuses_a_missing_voice_track():
    status = _good_audio_status()
    status["output"]["has_audio"] = False
    with pytest.raises(spend_run.SpendStop) as exc:
        _run_one_audio(FakeClient(job_statuses=[status]))
    assert exc.value.code == "no-audio-stream"


def test_audio_job_rejects_unwhitelisted_schema():
    status = _good_audio_status()
    status["output"]["voice_model_path"] = "/app/models/piper"
    with pytest.raises(spend_run.SpendStop) as exc:
        _run_one_audio(FakeClient(job_statuses=[status]))
    assert exc.value.code == "schema-violation"


def test_video_watermark_evidence_must_be_boolean_when_reported():
    # Old images report nothing (fine — the canary reads that as "image
    # not rebuilt"); a new image must report a real boolean, never a
    # truthy string that could fake either state.
    good = dict(_good_video_status()["output"])
    good["watermarked"] = True
    spend_run.verify_video_success(good)
    bad = dict(_good_video_status()["output"])
    bad["watermarked"] = "true"
    with pytest.raises(spend_run.SpendStop) as err:
        spend_run.verify_video_success(bad)
    assert err.value.code == "watermark-evidence-invalid"


# ------------------------------------- the in-house image engine's canary
# ONIQ's own image engine (fully in-house directive, 2026-08-27). The
# harness must dispatch it TEXT-ONLY and prove it drew something real.


def _good_image_status(execution_ms=30_000):
    return {
        "status": "COMPLETED",
        "delayTime": 5_000,
        "executionTime": execution_ms,
        "output": {
            "ok": True,
            "op": "image_generate",
            "output_key": "out/validation/plate-001.png",
            "device": "cuda",
            "gpu_name": "NVIDIA RTX A5000",
            "vram_total_mb": 24576,
            "vram_peak_mb": 9000,
            "model": "Lightricks/LTX-Video-0.9.7-distilled#distilled",
            "model_load_ms": 14_000,
            "inference_ms": 6_000,
            "encode_ms": 90,
            "width": 704,
            "height": 480,
            "format": "png",
            "output_bytes": 410_000,
            "duration_ms": 21_000,
            "cleanup_ok": True,
        },
    }


def _run_one_image(client):
    ft = FakeTime()
    facts = _preflight(client)
    return spend_run.one_job(
        client, facts, output_key="out/validation/plate-001.png",
        op="image_generate", sleep=ft.sleep, clock=ft.clock,
    )


def test_image_canary_is_dispatched_text_only():
    # The contract refuses an input_key on image_generate, so the harness
    # must not send one — and the prompt is a module constant, never a
    # dispatch input.
    client = FakeClient(job_statuses=[_good_image_status()])
    _run_one_image(client)
    payload = client.submitted[0][1]
    assert payload["op"] == "image_generate"
    assert "input_key" not in payload
    assert payload["params"] == {"prompt": spend_run.IMAGE_PROMPT}


def test_image_canary_verifies_a_real_measured_still():
    client = FakeClient(job_statuses=[_good_image_status()])
    row = _run_one_image(client)
    assert row["op"] == "image_generate"
    assert row["termination"] == spend_run.TERMINATION_CONFIRMED


def test_image_verify_refuses_an_unproven_or_wrong_still():
    good = _good_image_status()["output"]
    spend_run.verify_image_success(good)

    for field, bad, code in (
        ("model", "missing", "model-unproven"),
        ("model_load_ms", 0, "model-unproven"),
        ("inference_ms", 0, "no-inference"),
        ("output_bytes", 0, "no-artifact"),
        ("format", "jpeg", "wrong-format"),
        ("width", 512, "wrong-canvas"),
    ):
        broken = dict(good)
        broken[field] = bad
        with pytest.raises(spend_run.SpendStop) as exc:
            spend_run.verify_image_success(broken)
        assert exc.value.code == code


def test_image_canary_has_exactly_one_shape(monkeypatch):
    # Like the audio canary: no battery shape exists for it. A habitual
    # through_phase=18 must refuse BEFORE provisioning, not fan out into
    # five paid stills. preflight is stubbed so the guard itself is what
    # this exercises; a passing guard never reaches a submit.
    submitted = []
    monkeypatch.setattr(
        spend_run, "preflight",
        lambda *a, **k: {"endpoint_id": "ep-123", "input_ref": "in/test.png",
                         "output_prefix": "out/validation"},
    )
    monkeypatch.setattr(
        spend_run, "one_job",
        lambda *a, **k: submitted.append(k) or {},
    )
    monkeypatch.setattr(spend_run, "_inputs_from_env", lambda: {})
    monkeypatch.setenv("OP", "image_generate")
    monkeypatch.setenv("THROUGH_PHASE", "18")
    monkeypatch.setenv("APPROVAL_MODE", "owner-dispatch")

    assert spend_run.main(["spend_run", "run"]) == 1
    assert submitted == []  # nothing was provisioned


def test_image_canary_runs_one_still_at_phase_16(monkeypatch):
    calls = []
    monkeypatch.setattr(
        spend_run, "preflight",
        lambda *a, **k: {"endpoint_id": "ep-123", "input_ref": "in/test.png",
                         "output_prefix": "out/validation"},
    )
    monkeypatch.setattr(
        spend_run, "one_job",
        lambda *a, **k: (calls.append(k), _good_image_status()["output"])[1],
    )
    monkeypatch.setattr(spend_run, "_inputs_from_env", lambda: {})
    monkeypatch.setattr(spend_run, "economics", lambda rows: {})
    monkeypatch.setattr(spend_run, "confirm_termination_or_stop",
                        lambda *a, **k: None, raising=False)
    monkeypatch.setenv("OP", "image_generate")
    monkeypatch.setenv("THROUGH_PHASE", "16")
    monkeypatch.setenv("APPROVAL_MODE", "owner-dispatch")

    spend_run.main(["spend_run", "run"])
    assert len(calls) == 1
    assert calls[0]["op"] == "image_generate"
    # The still IS the action battery's conditioning plate (owner
    # directive 2026-08-29): main must dispatch PLATE_PROMPT — never
    # IMAGE_PROMPT — and name the object plate-002: plate-001 is the
    # PLATE_INVALID evidence and must never be overwritten.
    assert calls[0]["prompt"] == spend_run.PLATE_PROMPT
    assert calls[0]["output_key"] == (
        "out/validation/plate-002." + spend_run.contract_image_format()
    )


# ============================================ the no-idle-worker invariant
#
# Owner directive 2026-08-27: workersStandby == 0 is REMOVED as a hard
# blocker and replaced with the state that actually matters —
#
#     workersMin == 0 AND workersMax == 1
#     AND active GPU pods == 0 AND orphan pods == 0
#     AND the GPU allow-list is the target card only
#
# The harness already enforced all of it and never blocked on standby, but
# nothing PINNED that, and the function doing the recording was called
# check_standby_zero — a name that promised a gate the code did not have.
# These tests make the directive enforceable in both directions: each real
# condition blocks, and standby alone does not.


def _inv_endpoint(**over):
    base = {
        "id": "p3zmlv8ek10dzt",
        "workersMin": 0,
        "workersMax": 1,
        "workersStandby": 1,  # the value the owner made informational
        "gpuTypeIds": [admission.TARGET_GPU],
        "templateId": "hhhdwtjw0y",
    }
    base.update(over)
    return base


class _InvariantClient:
    """Minimal preflight client; every knob the invariant names."""

    def __init__(self, *, pods=(), endpoint=None):
        self._pods = list(pods)
        self._endpoint = endpoint if endpoint is not None else _inv_endpoint()

    def get_pods(self):
        return json.dumps(self._pods), self._pods

    def get_endpoints(self):
        doc = [self._endpoint]
        return json.dumps(doc), doc

    @staticmethod
    def parse_endpoint(doc):
        import runpod_client

        return runpod_client.parse_endpoint(doc)


def _stop_code(client):
    """Run preflight far enough to see which gate fires, if any."""
    try:
        spend_run.preflight(client, endpoint_id="")
    except spend_run.SpendStop as stop:
        return stop.code
    except admission.AdmissionRefused as refused:
        # AdmissionRefused carries .code; args[0] is the human message.
        return refused.code
    except Exception as exc:  # a later stage we do not model here
        return f"reached-later-stage:{type(exc).__name__}"
    return None


#: every gate the new invariant owns. Standby is deliberately not here.
INVARIANT_STOPS = {
    "unexpected-pods",
    "endpoint-config-refused",
    "endpoint-not-target",
    "endpoint-gpu-list-not-exclusive",
    "endpoint-not-singular",
}


def test_standby_one_alone_does_NOT_block():
    """The directive's whole point: a standby of 1, with every real
    condition satisfied, passes every gate the invariant owns.

    Preflight continues past them into stages this fixture does not model
    (test refs, live quote), so the assertion is that no INVARIANT gate
    fired — not that preflight ran to completion.
    """
    code = _stop_code(_InvariantClient(endpoint=_inv_endpoint(workersStandby=1)))
    assert code not in INVARIANT_STOPS, code


def test_a_running_pod_blocks():
    code = _stop_code(_InvariantClient(pods=[{"id": "pod-1"}]))
    assert code == "unexpected-pods"


def test_min_workers_above_zero_blocks():
    code = _stop_code(_InvariantClient(endpoint=_inv_endpoint(workersMin=1)))
    assert code == "endpoint-config-refused"


def test_max_workers_above_one_blocks():
    code = _stop_code(_InvariantClient(endpoint=_inv_endpoint(workersMax=2)))
    assert code == "endpoint-config-refused"


def test_a_second_gpu_on_the_allow_list_blocks():
    code = _stop_code(
        _InvariantClient(
            endpoint=_inv_endpoint(gpuTypeIds=[admission.TARGET_GPU, "NVIDIA L4"])
        )
    )
    assert code == "endpoint-gpu-list-not-exclusive"


def test_the_wrong_card_blocks():
    code = _stop_code(_InvariantClient(endpoint=_inv_endpoint(gpuTypeIds=["NVIDIA L4"])))
    assert code == "endpoint-not-target"


def test_the_recorder_records_and_never_raises():
    """It is observability. It must print the provider's number and let
    the run continue — that is the difference between this and a gate."""
    client = _InvariantClient(endpoint=_inv_endpoint(workersStandby=3))
    spend_run.record_standby_state(client)  # must not raise


def test_no_function_name_still_promises_a_standby_gate():
    """The name check_standby_zero cost four cycles: it advertised a gate
    the code did not have. A future 'check_standby_*' would do it again."""
    import ast

    tree = ast.parse(open(spend_run.__file__, encoding="utf-8").read())
    promises = [
        fn.name
        for fn in ast.walk(tree)
        if isinstance(fn, ast.FunctionDef)
        and "standby" in fn.name.lower()
        and fn.name.lower().startswith(("check_", "require_", "assert_"))
    ]
    assert not promises, (
        f"{promises} name a standby GATE, but standby is informational — "
        "rename to record_/read_ so the name matches the behaviour"
    )


def test_plate_prompt_carries_the_owner_ordering_and_exclusions():
    # Owner directive 2026-08-29 (one authorized replacement): the critical
    # elements come FIRST — plate-001 measured that this model keeps early
    # clauses and drops late ones — exactly two people are declared, and the
    # exclusions are explicit. This is the fix for a measured adherence
    # failure, so it is pinned like one.
    p = spend_run.PLATE_PROMPT
    train, balloon, maya = p.index("train"), p.index("red balloon"), p.index("Maya")
    assert train < maya and balloon < maya, "critical elements precede the cast"
    assert "Exactly two people" in p
    assert p.count("train") >= 2 and p.count("balloon") >= 3
    assert "away from the tracks" in p
    for exclusion in ("No other people", "no figures on the tracks",
                      "no deformed people", "no extra limbs", "no text"):
        assert exclusion in p, exclusion
    assert len(p) < 1000
