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


def test_sweep_reports_none_when_api_unreachable(monkeypatch):
    def unreachable():
        raise rp.RunPodApiError("request failed: URLError")

    monkeypatch.setattr(rp, "get_pods", unreachable)
    assert rp.sweep_orphans() is None  # None, never 0


def test_sweep_zero_when_account_is_empty(monkeypatch):
    monkeypatch.setattr(rp, "get_pods", lambda: ("[]", []))
    monkeypatch.setattr(rp, "get_endpoints", lambda: ("[]", []))
    assert rp.sweep_orphans() == {"pods": 0, "endpoint_min_workers": 0}


def test_sweep_counts_pods_and_min_workers(monkeypatch):
    pods = [{"id": "p1"}]
    eps = [{"id": "e1", "workersMin": 1, "workersMax": 1}]
    monkeypatch.setattr(rp, "get_pods", lambda: (json.dumps(pods), pods))
    monkeypatch.setattr(rp, "get_endpoints", lambda: (json.dumps(eps), eps))
    assert rp.sweep_orphans() == {"pods": 1, "endpoint_min_workers": 1}


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
