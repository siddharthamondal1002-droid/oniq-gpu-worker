"""Detaching the volume: reversible, guarded, and diagnostic."""

import pytest

from validation import volume_detach as vd

ENDPOINT = "9gh6qbou1in8yb"
VOLUME = "j2e7do8hcl"


class FakeClient:
    def __init__(self, *, endpoint, after=None, raises=None):
        self._endpoint = dict(endpoint)
        self._after = after
        self._raises = raises
        self.calls = []

    def get_endpoint(self, endpoint_id):
        return "{}", dict(self._endpoint)

    def attach_network_volume(self, endpoint_id, volume_id, datacenter_id=None):
        if self._raises:
            raise self._raises
        self.calls.append((endpoint_id, volume_id, datacenter_id))
        return dict(self._endpoint), dict(self._after or {})


def _endpoint(**over):
    doc = {
        "id": ENDPOINT, "templateId": "aqa3wkdf8g",
        "gpuTypeIds": ["NVIDIA RTX A5000"], "workersMin": 1,
        "workersMax": 2, "workersStandby": 2, "idleTimeout": 5,
        "executionTimeoutMs": 2700000, "networkVolumeId": VOLUME,
    }
    doc.update(over)
    return doc


def test_detach_sends_an_empty_volume_and_no_datacenter():
    """The pin must leave WITH the volume. A detach that dropped the volume
    and left dataCenterIds behind would strand the endpoint exactly as the
    incomplete attach did."""
    client = FakeClient(endpoint=_endpoint(), after=_endpoint(networkVolumeId=""))
    code, result = vd.report(client, ENDPOINT, vd.TOKEN)
    assert code == 0
    assert client.calls == [(ENDPOINT, "", None)]
    assert result["network_volume_id_before"] == VOLUME


def test_the_literal_token_is_required():
    client = FakeClient(endpoint=_endpoint())
    for token in ("", "detach-volume", "CREATE-AND-ATTACH-VOLUME", "yes"):
        code, result = vd.report(client, ENDPOINT, token)
        assert code == 1, token
        assert result["refused"] == "token-wrong"
    assert client.calls == []


def test_an_already_detached_endpoint_sends_no_patch():
    client = FakeClient(endpoint=_endpoint(networkVolumeId=""))
    code, result = vd.report(client, ENDPOINT, vd.TOKEN)
    assert code == 0
    assert result["already_detached"] is True
    assert client.calls == []


def test_a_detach_that_did_not_land_is_refused():
    client = FakeClient(endpoint=_endpoint(), after=_endpoint())
    code, result = vd.report(client, ENDPOINT, vd.TOKEN)
    assert code == 1
    assert result["refused"] == "detach-not-applied"


@pytest.mark.parametrize("field,value", [
    ("workersMin", 4), ("workersMax", 8), ("workersStandby", 5),
    ("templateId", "other"), ("gpuTypeIds", ["NVIDIA A100"]),
    ("idleTimeout", 600), ("executionTimeoutMs", 600000),
])
def test_any_collateral_movement_takes_the_run_down(field, value):
    client = FakeClient(
        endpoint=_endpoint(),
        after=_endpoint(networkVolumeId="", **{field: value}),
    )
    code, result = vd.report(client, ENDPOINT, vd.TOKEN)
    assert code == 1
    assert result["refused"] == "collateral-change"


def test_this_module_cannot_delete_a_volume():
    """Detach is not delete. The 50 GB and everything hydrated onto it must
    survive, or the diagnostic costs more than the answer is worth.

    Checked by walking CALLS, not by grepping text. The module's own log
    line says "Detach is not delete" — a substring match trips on the
    sentence explaining the guarantee, which is the opposite of evidence.
    What matters is which client methods this code can actually reach.
    """
    import ast
    import inspect

    called = set()
    for node in ast.walk(ast.parse(inspect.getsource(vd))):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Attribute):
                called.add(fn.attr)
            elif isinstance(fn, ast.Name):
                called.add(fn.id)

    forbidden = {
        "delete_network_volume", "remove_network_volume",
        "create_network_volume", "purge_queue", "run_sync", "submit_job",
        "attach_template", "set_execution_timeout", "set_workers_min_zero",
        "set_workers_standby_zero", "retarget_template", "create_template",
    }
    assert not (called & forbidden), sorted(called & forbidden)

    # The ONE mutating client call it may make, named explicitly.
    assert "attach_network_volume" in called
    assert "get_endpoint" in called


def test_an_exception_mid_write_never_reports_success():
    client = FakeClient(endpoint=_endpoint(), raises=RuntimeError("boom"))
    code, result = vd.report(client, ENDPOINT, vd.TOKEN)
    assert code == 1
    assert result["refused"] == "RuntimeError"
