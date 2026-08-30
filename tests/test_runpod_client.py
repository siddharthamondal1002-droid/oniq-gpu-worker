import json
import os

import pytest

import runpod_client as rp

# RunPod's actual bytes, captured 2026-08-25 through the read-only
# discovery channel. The parser is judged against these, per the rule
# that it is not trusted until RAW and PARSED agree.
with open(
    os.path.join(os.path.dirname(__file__), "fixtures", "gpu_types_actual.json"),
    encoding="utf-8",
) as _fh:
    GPU_TYPES_FIXTURE = json.load(_fh)

_ENTRIES = {g["id"]: g for g in GPU_TYPES_FIXTURE["data"]["gpuTypes"]}


def test_parse_the_a5000_trap_list_price_is_not_capacity():
    # The A5000's real state: a securePrice of 0.27 in the catalogue,
    # while lowestPrice is null for both on-demand and spot. The list
    # price must parse through AND the capacity price must stay None.
    parsed = rp.parse_gpu_type(_ENTRIES["NVIDIA RTX A5000"])
    assert parsed["secure_price"] == 0.27
    assert parsed["on_demand_price"] is None
    assert parsed["spot_price"] is None
    assert parsed["secure_cloud"] is True


def test_parse_the_3090_real_entry():
    parsed = rp.parse_gpu_type(_ENTRIES["NVIDIA GeForce RTX 3090"])
    assert parsed["id"] == "NVIDIA GeForce RTX 3090"
    assert parsed["display_name"] == "RTX 3090"
    assert parsed["memory_gb"] == 24
    assert parsed["secure_price"] == 0.5
    assert parsed["community_price"] == 0.22
    assert parsed["on_demand_price"] == 0.22


def test_regression_id_carries_the_full_name_not_display_name():
    # The original parser matched on displayName and would have read the
    # 3090 as permanently unavailable — the exact predicted failure mode
    # (a wrong field reads as "no capacity", which reads as "the 3090 is
    # unavailable").
    parsed = [rp.parse_gpu_type(g) for g in GPU_TYPES_FIXTURE["data"]["gpuTypes"]]
    hit = rp.find_gpu(parsed, "NVIDIA GeForce RTX 3090")
    assert hit is not None
    assert hit["display_name"] == "RTX 3090"


def test_parse_missing_lowest_price_object():
    entry = dict(_ENTRIES["NVIDIA GeForce RTX 3090"], lowestPrice=None)
    parsed = rp.parse_gpu_type(entry)
    assert parsed["on_demand_price"] is None


def test_find_gpu():
    parsed = [rp.parse_gpu_type(g) for g in GPU_TYPES_FIXTURE["data"]["gpuTypes"]]
    assert rp.find_gpu(parsed, "NVIDIA GeForce RTX 3090")["memory_gb"] == 24
    assert rp.find_gpu(parsed, "NVIDIA B200") is None


def test_missing_api_key_names_the_variable_only(monkeypatch):
    with pytest.raises(rp.RunPodApiError) as exc:
        rp._api_key()
    assert "RUNPOD_API_KEY" in str(exc.value)


def test_gpu_catalogue_returns_raw_and_parsed(monkeypatch):
    raw = json.dumps(GPU_TYPES_FIXTURE)
    monkeypatch.setattr(rp, "_request", lambda *a, **k: (200, raw))
    got_raw, parsed = rp.gpu_catalogue()
    assert got_raw == raw  # provider bytes, verbatim
    assert parsed[0]["id"] == "NVIDIA GeForce RTX 3090"
    assert parsed[0]["display_name"] == "RTX 3090"


def test_graphql_errors_surface_as_api_error(monkeypatch):
    raw = json.dumps({"errors": [{"message": "Cannot query field X"}]})
    monkeypatch.setattr(rp, "_request", lambda *a, **k: (200, raw))
    with pytest.raises(rp.RunPodApiError) as exc:
        rp.gpu_catalogue()
    assert "Cannot query field X" in str(exc.value)


class _RecordingTransport:
    """Scripted responses keyed by call order; records every request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, *, method="GET", body=None, timeout=30, bearer=True):
        self.calls.append({"url": url, "method": method, "bearer": bearer})
        return self.responses.pop(0)


def test_catalogue_falls_back_to_param_auth_on_403(monkeypatch):
    # Measured 2026-08-25: Bearer -> 403 while the key is valid.
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_FAKE_KEY")
    transport = _RecordingTransport(
        [(403, "forbidden"), (200, json.dumps(GPU_TYPES_FIXTURE))]
    )
    monkeypatch.setattr(rp, "_request", transport)
    _, parsed = rp.gpu_catalogue()
    assert parsed[0]["id"] == "NVIDIA GeForce RTX 3090"
    assert transport.calls[1]["bearer"] is False
    assert transport.calls[1]["url"].startswith(rp.GRAPHQL_URL + "?api_key=")


def test_catalogue_falls_back_to_rest_gputypes(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_FAKE_KEY")
    rest_body = json.dumps(GPU_TYPES_FIXTURE["data"]["gpuTypes"])
    transport = _RecordingTransport(
        [(403, "forbidden"), (403, "forbidden"), (200, rest_body)]
    )
    monkeypatch.setattr(rp, "_request", transport)
    raw, parsed = rp.gpu_catalogue()
    assert raw == rest_body  # the answering source's bytes, verbatim
    assert transport.calls[2]["url"] == f"{rp.REST_BASE}/gputypes"
    assert parsed[0]["display_name"] == "RTX 3090"


def test_catalogue_total_failure_names_the_permission_fix(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_FAKE_KEY")
    transport = _RecordingTransport(
        [(403, "gql-no"), (403, "gql-no"), (404, "no such route")]
    )
    monkeypatch.setattr(rp, "_request", transport)
    with pytest.raises(rp.RunPodApiError) as exc:
        rp.gpu_catalogue()
    text = str(exc.value)
    assert "403" in text and "404" in text
    assert "GraphQL permission" in text
    assert "rpa_FAKE_KEY" not in text  # the key never reaches an error


def test_sweep_reports_none_when_api_unreachable(monkeypatch):
    def unreachable():
        raise rp.RunPodApiError("request failed: URLError")

    monkeypatch.setattr(rp, "get_pods", unreachable)
    assert rp.sweep_orphans() is None  # None, never 0


def test_sweep_zero_when_account_is_empty(monkeypatch):
    monkeypatch.setattr(rp, "get_pods", lambda: ("[]", []))
    monkeypatch.setattr(rp, "get_endpoints", lambda: ("[]", []))
    assert rp.sweep_orphans() == {
        "pods": 0,
        "endpoint_min_workers": 0,
        # Owner-accepted always-on capacity is reported alongside the
        # alarm count (2026-08-30), never folded into it.
        "accepted_min_workers": 0,
        "accepted_endpoints": {},
    }


def test_sweep_counts_pods_and_min_workers(monkeypatch):
    pods = [{"id": "p1"}]
    eps = [{"id": "e1", "workersMin": 1, "workersMax": 1}]
    monkeypatch.setattr(rp, "get_pods", lambda: (json.dumps(pods), pods))
    monkeypatch.setattr(rp, "get_endpoints", lambda: (json.dumps(eps), eps))
    assert rp.sweep_orphans() == {
        "pods": 1,
        "endpoint_min_workers": 1,
        "accepted_min_workers": 0,
        "accepted_endpoints": {},
    }


def test_discover_is_read_only_and_never_prints_the_key(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_FAKE_KEY_VALUE")
    calls = []

    def recording_request(url, *, method="GET", body=None, timeout=30):
        calls.append((method, url))
        if url == rp.GRAPHQL_URL:
            return 200, json.dumps(GPU_TYPES_FIXTURE)
        return 200, "[]"

    monkeypatch.setattr(rp, "_request", recording_request)
    out_file = tmp_path / "d.json"
    report = rp.discover(str(out_file))

    for method, url in calls:
        assert (method, url) in (
            ("POST", rp.GRAPHQL_URL),  # the one query
            ("GET", f"{rp.REST_BASE}/pods"),
            ("GET", f"{rp.REST_BASE}/endpoints"),
        ), f"unexpected call: {method} {url}"
    assert not any("/run" in url or "purge" in url for _, url in calls)

    printed = capsys.readouterr().out
    assert "rpa_FAKE_KEY_VALUE" not in printed
    assert "RAW gpuTypes" in printed
    assert report["target_gpu_parsed"]["memory_gb"] == 24
    assert json.loads(out_file.read_text())["target_gpu_parsed"]


def test_parse_endpoint_reads_worker_bounds():
    parsed = rp.parse_endpoint(
        {"id": "e", "name": "n", "workersMin": 0, "workersMax": 1,
         "gpuTypeIds": ["NVIDIA GeForce RTX 3090"], "idleTimeout": 5}
    )
    assert parsed["min_workers"] == 0
    assert parsed["max_workers"] == 1


def test_cli_refuses_unknown_commands(capsys):
    assert rp.main(["runpod_client.py"]) == 2
    assert rp.main(["runpod_client.py", "provision"]) == 2


def test_standby_patch_sends_only_the_literal_zero(monkeypatch):
    # The harness's one mutation: PATCH, one field, hard-coded 0. A
    # value parameter here would be a scale-UP surface; refuse to grow one.
    import inspect

    import runpod_client as rp

    assert list(inspect.signature(rp.set_workers_standby_zero).parameters) == [
        "endpoint_id"
    ]
    seen = {}

    def transport(url, *, method="GET", body=None, **kw):
        seen["url"] = url
        seen["method"] = method
        seen["body"] = body
        return 200, "{}"

    monkeypatch.setattr(rp, "_request", transport)
    status, raw = rp.set_workers_standby_zero("ep-123")
    assert status == 200
    assert seen["method"] == "PATCH"
    assert seen["url"].endswith("/endpoints/ep-123")
    assert seen["body"] == {"workersStandby": 0}


def test_no_scale_up_surface_exists():
    import runpod_client as rp

    for forbidden in ("create_endpoint", "create_pod", "set_workers_min",
                      "set_workers_max", "set_workers_standby"):
        assert not hasattr(rp, forbidden)


def test_standby_falls_back_to_graphql_partial_save(monkeypatch):
    # Measured (run #34): REST PATCH exists but refuses the field. The
    # fallback is the minimal {id, workersStandby: 0} saveEndpoint —
    # naming NO other field, so nothing else about the endpoint can move.
    import runpod_client as rp

    calls = []

    def transport(url, *, method="GET", body=None, **kw):
        calls.append((url, method, body))
        if method == "PATCH":
            return 400, '{"error":"Extra input keys provided in request body"}'
        return 200, json.dumps(
            {"data": {"saveEndpoint": {"id": "ep-123", "workersStandby": 0}}}
        )

    monkeypatch.setattr(rp, "_request", transport)
    status, raw = rp.set_workers_standby_zero("ep-123")
    assert status == 200
    graphql_call = calls[1]
    assert graphql_call[1] == "POST"
    assert "workersStandby: 0" in graphql_call[2]["query"]
    assert graphql_call[2]["variables"] == {"id": "ep-123"}
    # The mutation names id and workersStandby and nothing else.
    assert "gpuIds" not in graphql_call[2]["query"]
    assert "templateId" not in graphql_call[2]["query"]


def test_standby_failure_carries_both_transport_bodies(monkeypatch):
    import runpod_client as rp

    def transport(url, *, method="GET", body=None, **kw):
        if method == "PATCH":
            return 400, '{"error":"no such key"}'
        return 200, json.dumps({"errors": [{"message": "field not allowed"}]})

    monkeypatch.setattr(rp, "_request", transport)
    status, raw = rp.set_workers_standby_zero("ep-123")
    assert "rest patch -> 400" in raw
    assert "field not allowed" in raw


def test_rest_schema_probe_reduces_the_openapi_spec(monkeypatch):
    import runpod_client as rp

    spec = {
        "paths": {
            "/endpoints/{endpointId}": {
                "patch": {
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/EndpointUpdate"}
                            }
                        }
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "EndpointUpdate": {
                    "properties": {"workersMin": {}, "workersMax": {}, "idleTimeout": {}}
                },
                "Endpoint": {"properties": {"workersStandby": {"readOnly": True}}},
            }
        },
    }

    def transport(url, *, method="GET", body=None, bearer=True, **kw):
        assert bearer is False  # the public spec gets no credential
        return 200, json.dumps(spec)

    monkeypatch.setattr(rp, "_request", transport)
    probe = rp.rest_schema_probe()
    assert probe["patch_endpoint_properties"] == ["idleTimeout", "workersMax", "workersMin"]
    assert probe["standby_shaped_names"] == ["workersStandby"]


def test_no_probe_field_name_is_swallowed_by_the_redactor():
    """The diagnostics must survive being printed.

    spend_run.redact blanks any key whose name CONTAINS a secret marker,
    and "KEY" is one of them. That is the right direction to err in — but
    it silently ate rest_schema_probe's "standby_shaped_keys" on run #32,
    which was the single field that answers whether workersStandby is
    settable at all. Three cycles were spent on that question while its
    answer was being printed as "<redacted>".

    So the probes' own field names are checked against the markers here.
    A future diagnostic named "…_key…" fails this test instead of
    disappearing at the moment someone needs to read it.
    """
    from validation.spend_run import _REDACT_MARKERS, redact

    # Every field name either probe can return, with a recognisable value.
    probe_fields = {
        "endpoint_input_fields": ["workersMax"],
        "standby_shaped_mutations": [],
        "endpoint_shaped_mutations": ["saveEndpoint"],
        "worker_shaped_mutations": [],
        "spec_url": "https://rest.runpod.io/openapi.json",
        "patch_endpoint_properties": ["gpuTypeIds"],
        "standby_shaped_names": ["workersStandby"],
    }

    swallowed = [
        name
        for name in probe_fields
        if any(marker in name.upper() for marker in _REDACT_MARKERS)
    ]
    assert not swallowed, (
        f"{swallowed} collide with a redaction marker and would print as "
        "'<redacted>' — rename the field, do not weaken the redactor"
    )

    # And prove it end to end: nothing is blanked on the way out.
    assert redact(probe_fields) == probe_fields


def test_the_redactor_still_hides_real_secrets():
    """Guards the guard above: the fix must not have relaxed redaction."""
    from validation.spend_run import redact

    assert redact({"RUNPOD_API_KEY": "live"})["RUNPOD_API_KEY"] == "<redacted>"
    assert redact({"R2_ACCESS_KEY_ID": "live"})["R2_ACCESS_KEY_ID"] == "<redacted>"
    assert redact({"R2_SECRET_ACCESS_KEY": "x"})["R2_SECRET_ACCESS_KEY"] == "<redacted>"
    pair = redact({"key": "RUNPOD_API_KEY", "value": "live"})
    assert pair["value"] == "<redacted>"


# ------------------------------------------- owner-accepted always-on


def _sweep_with(monkeypatch, endpoints, pods=()):
    import runpod_client as rp

    monkeypatch.setattr(rp, "get_pods", lambda: ("[]", list(pods)))
    monkeypatch.setattr(rp, "get_endpoints", lambda: ("[]", list(endpoints)))
    return rp.sweep_orphans()


def test_an_accepted_endpoint_is_reported_but_does_not_alarm(monkeypatch):
    """Owner directive 2026-08-30: the owner's own always-on endpoint stays
    running, and the owner is told what it costs.

    Before this, workersMin=1 on that endpoint made the sweep fire on every
    single run. A guard that is permanently red is a guard people learn to
    scroll past, and the day a real orphan appears it would be one more red
    line among many.
    """
    result = _sweep_with(monkeypatch, [
        {"id": "9gh6qbou1in8yb", "workersMin": 1, "workersMax": 2},
    ])
    assert result["endpoint_min_workers"] == 0, "accepted capacity must not alarm"
    assert result["accepted_min_workers"] == 1, "but it must still be reported"
    assert result["accepted_endpoints"] == {"9gh6qbou1in8yb": 1}


def test_any_OTHER_endpoint_still_takes_the_run_down(monkeypatch):
    # The whole point of naming the exception: everything else still alarms.
    result = _sweep_with(monkeypatch, [
        {"id": "9gh6qbou1in8yb", "workersMin": 1, "workersMax": 2},
        {"id": "some-new-endpoint", "workersMin": 1, "workersMax": 1},
    ])
    assert result["endpoint_min_workers"] == 1


def test_a_pod_still_alarms_even_with_only_accepted_endpoints(monkeypatch):
    result = _sweep_with(
        monkeypatch,
        [{"id": "9gh6qbou1in8yb", "workersMin": 1, "workersMax": 2}],
        pods=[{"id": "pod-1"}],
    )
    assert result["pods"] == 1


def test_the_accepted_list_carries_a_reason_for_every_entry():
    import runpod_client as rp

    assert rp.ACCEPTED_ALWAYS_ON, "an empty list would be a silent blanket pass"
    for endpoint_id, reason in rp.ACCEPTED_ALWAYS_ON.items():
        assert endpoint_id and isinstance(reason, str) and len(reason) > 20, (
            f"{endpoint_id} is accepted with no recorded reason"
        )
