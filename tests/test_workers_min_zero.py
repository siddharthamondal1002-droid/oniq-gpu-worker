"""workersMin -> 0: a spend reduction that cannot become an increase.

Owner decision 2026-08-30. The guard these tests protect is not "did the
number change" — it is that this path has no reachable way to raise
spend, and that a write which did anything else is a stop rather than a
reported success.
"""

import pytest

from validation import workers_min_zero as wm

ENDPOINT = "9gh6qbou1in8yb"


class FakeClient:
    def __init__(self, *, endpoint, after=None, status=200, raw="{}",
                 raises=None):
        self._endpoint = dict(endpoint)
        self._after = after
        self._status, self._raw = status, raw
        self._raises = raises
        self.patched = []

    def get_endpoint(self, endpoint_id):
        if self.patched and self._after is not None:
            return "{}", dict(self._after)
        return "{}", dict(self._endpoint)

    def set_workers_min_zero(self, endpoint_id):
        if self._raises:
            raise self._raises
        self.patched.append(endpoint_id)
        return self._status, self._raw


def _endpoint(**over):
    doc = {
        "id": ENDPOINT,
        "templateId": "aqa3wkdf8g",
        "gpuTypeIds": ["NVIDIA RTX A5000"],
        "workersMin": 1,
        "workersMax": 1,
        "workersStandby": 1,
        "idleTimeout": 5,
        "executionTimeoutMs": 2700000,
        "networkVolumeId": "j2e7do8hcl",
    }
    doc.update(over)
    return doc


def test_the_happy_path_drops_the_floor_and_nothing_else():
    client = FakeClient(endpoint=_endpoint(),
                        after=_endpoint(workersMin=0))
    code, result = wm.report(client, ENDPOINT)
    assert code == 0
    assert client.patched == [ENDPOINT]
    assert result["workers_min_before"] == 1
    assert result["workers_min_after"] == 0


def test_re_running_against_a_zero_floor_sends_no_patch():
    """A PATCH restarts the worker. A no-op re-run must not cost a pull."""
    client = FakeClient(endpoint=_endpoint(workersMin=0))
    code, result = wm.report(client, ENDPOINT)
    assert code == 0
    assert result["already_zero"] is True
    assert client.patched == []


def test_a_blank_endpoint_is_refused():
    client = FakeClient(endpoint=_endpoint())
    code, result = wm.report(client, "")
    assert code == 1
    assert result["refused"] == "endpoint-missing"
    assert client.patched == []


def test_a_patch_that_did_not_land_is_a_refusal_not_a_success():
    client = FakeClient(endpoint=_endpoint(), after=_endpoint(workersMin=1))
    code, result = wm.report(client, ENDPOINT)
    assert code == 1
    assert result["refused"] == "not-applied"


def test_a_non_success_status_is_a_refusal():
    client = FakeClient(endpoint=_endpoint(), status=400, raw="bad request")
    code, result = wm.report(client, ENDPOINT)
    assert code == 1
    assert result["refused"] == "patch-refused"


def test_an_exception_mid_write_never_reports_success():
    client = FakeClient(endpoint=_endpoint(), raises=RuntimeError("boom"))
    code, result = wm.report(client, ENDPOINT)
    assert code == 1
    assert result["refused"] == "RuntimeError"


@pytest.mark.parametrize("field,value", [
    ("workersMax", 4),
    ("workersStandby", 3),
    ("templateId", "someothertemplate"),
    ("gpuTypeIds", ["NVIDIA A100"]),
    ("networkVolumeId", ""),
    ("executionTimeoutMs", 600000),
    ("idleTimeout", 600),
])
def test_any_collateral_movement_takes_the_run_down(field, value):
    """workersMax is guarded for a specific reason: dropping the floor
    while quietly raising the ceiling would satisfy the admission gate
    having made the spend worse, not better."""
    client = FakeClient(
        endpoint=_endpoint(),
        after=_endpoint(workersMin=0, **{field: value}),
    )
    code, result = wm.report(client, ENDPOINT)
    assert code == 1
    assert result["refused"] == "collateral-change"


def test_the_writer_takes_no_value_so_it_cannot_scale_anything_up():
    """The structural guarantee, not a behavioural one.

    A value parameter is all it would take for a stale read, a typo or a
    dispatch input to turn this spend REDUCTION into an increase. There
    is no such parameter, and 0 is a literal in the request body.
    """
    import inspect

    import runpod_client

    sig = inspect.signature(runpod_client.set_workers_min_zero)
    assert list(sig.parameters) == ["endpoint_id"], (
        "a second parameter is the only thing standing between this and a "
        "scale-up"
    )
    source = inspect.getsource(runpod_client.set_workers_min_zero)
    assert '"workersMin": 0' in source
    body = source.split('body={')[1].split('}')[0]
    assert body.strip().rstrip(",") == '"workersMin": 0', (
        f"only workersMin may be in the body; found {body!r}"
    )


def test_this_module_holds_no_other_mutating_verb():
    """The driver's admission gate must stay the thing that refuses a
    badly-shaped endpoint. This module repairs ONE field and must not grow
    into a general endpoint editor."""
    import inspect

    source = inspect.getsource(wm)
    for forbidden in ("attach_template", "attach_network_volume",
                      "set_execution_timeout", "retarget_template",
                      "create_template", "set_template_env",
                      "set_workers_standby_zero", "run_sync", "submit_job"):
        assert forbidden not in source, forbidden
