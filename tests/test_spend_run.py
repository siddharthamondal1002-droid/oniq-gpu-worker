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
                        "id": "NVIDIA GeForce RTX 3090",
                        "displayName": "RTX 3090",
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


def _endpoint(env=None, gpus=None, mn=0, mx=1):
    return {
        "id": "ep-123",
        "name": "oniq-gpu",
        "workersMin": mn,
        "workersMax": mx,
        "gpuTypeIds": gpus or ["NVIDIA GeForce RTX 3090"],
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
        health = self._health_seq.pop(0) if self._health_seq else {"workers": {"idle": 0, "running": 0}}
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
             "gpuTypeIds": ["NVIDIA GeForce RTX 3090"], "env": _endpoint()["env"]}
    with pytest.raises(spend_run.SpendStop) as exc:
        _preflight(FakeClient(endpoints=[weird]))
    assert exc.value.code == "endpoint-fields-unparsed"


def test_preflight_refuses_non_3090_endpoint():
    with pytest.raises(spend_run.SpendStop) as exc:
        _preflight(FakeClient(endpoints=[_endpoint(gpus=["NVIDIA RTX A5000"])]))
    assert exc.value.code == "endpoint-not-3090"


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
            "gpu_name": "NVIDIA GeForce RTX 3090",
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
    assert row["gpu_name"] == "NVIDIA GeForce RTX 3090"
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
    stuck = [{"workers": {"idle": 0, "running": 1}}] * 200
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


# ------------------------------------------------------------- termination


def test_confirm_termination_requires_a_parsed_zero():
    ft = FakeTime()
    client = FakeClient(health_seq=[{"unexpected": "shape"}] * 200)
    out = spend_run.confirm_termination(client, "ep-123", sleep=ft.sleep, clock=ft.clock)
    assert out == spend_run.TERMINATION_UNKNOWN


def test_confirm_termination_confirms_on_zero_workers():
    ft = FakeTime()
    client = FakeClient(
        health_seq=[{"workers": {"idle": 0, "running": 1}},
                    {"workers": {"idle": 0, "running": 0, "initializing": 0}}]
    )
    out = spend_run.confirm_termination(client, "ep-123", sleep=ft.sleep, clock=ft.clock)
    assert out == spend_run.TERMINATION_CONFIRMED


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


def test_preflight_still_stops_when_neither_carries_r2():
    ep = _endpoint(env={})
    ep["templateId"] = "tpl-1"
    client = FakeClient(endpoints=[ep])
    client.get_template = lambda tid: ("{}", {"id": tid, "env": {}})
    with pytest.raises(spend_run.SpendStop) as exc:
        _preflight(client)
    assert exc.value.code == "r2-env-missing"
