"""Conforming an endpoint to the bounds the admission gate requires."""

import pytest

from validation import admission, endpoint_bounds as eb

ENDPOINT = "7disu6my0mloco"


class FakeClient:
    def __init__(self, *, endpoint, after=None, status=200, raw="{}", raises=None):
        self._endpoint = dict(endpoint)
        self._after = after
        self._status, self._raw = status, raw
        self._raises = raises
        self.patched = []

    def get_endpoint(self, endpoint_id):
        if self.patched and self._after is not None:
            return "{}", dict(self._after)
        return "{}", dict(self._endpoint)

    def set_worker_bounds_min0_max1(self, endpoint_id):
        if self._raises:
            raise self._raises
        self.patched.append(endpoint_id)
        return self._status, self._raw


def _endpoint(**over):
    doc = {
        "id": ENDPOINT, "templateId": "xfmaf7n83n",
        "gpuTypeIds": ["NVIDIA RTX A6000"],
        "workersMin": 1, "workersMax": 3, "workersStandby": 3,
        "idleTimeout": 5, "executionTimeoutMs": 600000,
        "networkVolumeId": "", "dataCenterIds": None,
    }
    doc.update(over)
    return doc


def test_it_writes_exactly_the_shape_admission_demands():
    client = FakeClient(
        endpoint=_endpoint(),
        after=_endpoint(workersMin=0, workersMax=1),
    )
    code, result = eb.report(client, ENDPOINT)
    assert code == 0
    assert client.patched == [ENDPOINT]
    assert (result["workers_min_after"], result["workers_max_after"]) == (0, 1)
    # The whole point: the endpoint now passes the gate that refused it.
    admission.check_endpoint_config(result["workers_min_after"],
                                    result["workers_max_after"])


def test_the_target_matches_what_admission_actually_requires():
    """If the gate's requirement ever changes, this module must not keep
    writing the old shape and reporting success."""
    admission.check_endpoint_config(eb.TARGET_MIN, eb.TARGET_MAX)
    with pytest.raises(admission.AdmissionRefused):
        admission.check_endpoint_config(eb.TARGET_MIN, eb.TARGET_MAX + 1)
    with pytest.raises(admission.AdmissionRefused):
        admission.check_endpoint_config(eb.TARGET_MIN + 1, eb.TARGET_MAX)


def test_an_already_conformed_endpoint_sends_no_patch():
    """A PATCH restarts the worker; a no-op re-run must not cost a pull."""
    client = FakeClient(endpoint=_endpoint(workersMin=0, workersMax=1))
    code, result = eb.report(client, ENDPOINT)
    assert code == 0
    assert result["already_conformed"] is True
    assert client.patched == []


def test_a_blank_endpoint_is_refused():
    client = FakeClient(endpoint=_endpoint())
    code, result = eb.report(client, "")
    assert code == 1
    assert result["refused"] == "endpoint-missing"
    assert client.patched == []


def test_a_patch_that_did_not_land_is_refused():
    client = FakeClient(endpoint=_endpoint(), after=_endpoint(workersMin=0))
    code, result = eb.report(client, ENDPOINT)
    assert code == 1
    assert result["refused"] == "not-applied"


def test_a_non_success_status_is_refused():
    client = FakeClient(endpoint=_endpoint(), status=400, raw="bad request")
    code, result = eb.report(client, ENDPOINT)
    assert code == 1
    assert result["refused"] == "patch-refused"


@pytest.mark.parametrize("field,value", [
    ("templateId", "someothertemplate"),
    ("gpuTypeIds", ["NVIDIA RTX A5000"]),
    ("idleTimeout", 600),
    ("executionTimeoutMs", 60000),
    ("networkVolumeId", "somevolume"),
    ("dataCenterIds", ["US-KS-2"]),
])
def test_any_collateral_movement_takes_the_run_down(field, value):
    """networkVolumeId and dataCenterIds are guarded because moving either
    relocates where the endpoint may run — measured 2026-08-30, when a
    volume in a datacenter absent from RunPod's own enum left an endpoint
    unable to place a single worker."""
    client = FakeClient(
        endpoint=_endpoint(),
        after=_endpoint(workersMin=0, workersMax=1, **{field: value}),
    )
    code, result = eb.report(client, ENDPOINT)
    assert code == 1
    assert result["refused"] == "collateral-change"


def test_an_exception_mid_write_never_reports_success():
    client = FakeClient(endpoint=_endpoint(), raises=RuntimeError("boom"))
    code, result = eb.report(client, ENDPOINT)
    assert code == 1
    assert result["refused"] == "RuntimeError"


def test_the_writer_takes_no_values_so_it_cannot_scale_anything_up():
    """The structural guarantee. A value parameter is all it would take for
    a stale read or a dispatch input to turn this reduction into an
    increase; there is none, and both numbers are literals in the body."""
    import inspect

    import runpod_client

    sig = inspect.signature(runpod_client.set_worker_bounds_min0_max1)
    assert list(sig.parameters) == ["endpoint_id"]
    source = inspect.getsource(runpod_client.set_worker_bounds_min0_max1)
    body = source.split("body={")[1].split("}")[0]
    assert body.strip().rstrip(",") == '"workersMin": 0, "workersMax": 1', body


def test_one_patch_not_two():
    """Two PATCHes would restart the worker twice, and on a 25 GiB image
    each restart is a fresh pull."""
    client = FakeClient(
        endpoint=_endpoint(),
        after=_endpoint(workersMin=0, workersMax=1),
    )
    eb.apply(client, ENDPOINT)
    assert len(client.patched) == 1
