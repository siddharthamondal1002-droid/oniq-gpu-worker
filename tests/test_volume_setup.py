"""Attaching the model volume, and the half of it that was missing.

MEASURED 2026-08-30. The attach sent networkVolumeId alone, on a
one-field-per-PATCH principle that is right for most endpoint writes and
wrong for this one. A network volume is datacenter-scoped: an endpoint
holding a volume in US-MO-2 can only run in US-MO-2. With the volume set
and no dataCenterIds key at all, the endpoint reported

    jobs    {inQueue: 1}
    workers {idle 0, initializing 0, ready 0, running 0, throttled 0,
             unhealthy 0}

for fifty minutes. The same endpoint had shown initializing=2 before the
attach. Nothing billed, nothing scheduled, nothing to see — the attach
reported success and the queue simply never drained.

This module had no unit tests at all when that happened.
"""

import pytest

from validation import volume_setup as vs

ENDPOINT = "9gh6qbou1in8yb"
VOLUME = "j2e7do8hcl"
DC = "US-MO-2"


class FakeClient:
    REST_BASE = "https://rest.example/v1"

    def __init__(self, *, volumes=None, endpoint=None, after=None,
                 created=None, attach_raises=None):
        self._volumes = volumes if volumes is not None else []
        self._endpoint = dict(endpoint or {})
        self._after = after
        self._created = created
        self._attach_raises = attach_raises
        self.attached = []
        self.created_volumes = []

    def _get_json(self, url):
        return "[]", list(self._volumes)

    def create_network_volume(self, name, size_gb, datacenter_id):
        self.created_volumes.append((name, size_gb, datacenter_id))
        return dict(self._created or {})

    def get_endpoint(self, endpoint_id):
        return "{}", dict(self._endpoint)

    def attach_network_volume(self, endpoint_id, volume_id, datacenter_id=None):
        if self._attach_raises:
            raise self._attach_raises
        self.attached.append((endpoint_id, volume_id, datacenter_id))
        return dict(self._endpoint), dict(self._after or {})


def _endpoint(**over):
    doc = {
        "id": ENDPOINT,
        "templateId": "aqa3wkdf8g",
        "gpuTypeIds": ["NVIDIA RTX A5000"],
        "workersMin": 0,
        "workersMax": 1,
        "workersStandby": 1,
        "idleTimeout": 5,
        "executionTimeoutMs": 2700000,
        "networkVolumeId": "",
    }
    doc.update(over)
    return doc


def _volume(**over):
    doc = {"id": VOLUME, "name": vs.DEFAULT_NAME, "size": 50,
           "dataCenterId": DC}
    doc.update(over)
    return doc


def test_an_existing_volume_is_reused_and_never_recreated():
    """Two volumes billing for the same purpose is money for nothing."""
    client = FakeClient(
        volumes=[_volume()],
        endpoint=_endpoint(),
        after=_endpoint(networkVolumeId=VOLUME, dataCenterIds=[DC]),
    )
    result = vs.apply(client, ENDPOINT, vs.DEFAULT_NAME, 50, DC)
    assert client.created_volumes == [], "a volume of this name already exists"
    assert result["created_now"] is False
    assert result["volume_id"] == VOLUME


def test_the_attach_pins_the_endpoint_to_the_volumes_datacenter():
    """THE 2026-08-30 FAILURE. An attach that lands the volume and not the
    pin looks successful and schedules nothing."""
    client = FakeClient(
        volumes=[_volume()],
        endpoint=_endpoint(),
        after=_endpoint(networkVolumeId=VOLUME, dataCenterIds=[DC]),
    )
    result = vs.apply(client, ENDPOINT, vs.DEFAULT_NAME, 50, DC)
    assert client.attached == [(ENDPOINT, VOLUME, DC)], (
        "the datacenter must travel with the volume"
    )
    assert result["data_center_ids_after"] == [DC]


def test_the_datacenter_comes_from_the_volume_not_the_caller():
    """The only correct value is where the volume actually IS. A caller's
    guess could pin the endpoint away from its own storage."""
    client = FakeClient(
        volumes=[_volume(dataCenterId="US-KS-2")],
        endpoint=_endpoint(),
        after=_endpoint(networkVolumeId=VOLUME, dataCenterIds=["US-KS-2"]),
    )
    vs.apply(client, ENDPOINT, vs.DEFAULT_NAME, 50, DC)
    assert client.attached == [(ENDPOINT, VOLUME, "US-KS-2")]


def test_an_endpoint_that_did_not_take_the_pin_is_refused():
    """Success on the volume alone is exactly the state that hung."""
    client = FakeClient(
        volumes=[_volume()],
        endpoint=_endpoint(),
        after=_endpoint(networkVolumeId=VOLUME),  # no dataCenterIds
    )
    with pytest.raises(vs.Refused) as exc:
        vs.apply(client, ENDPOINT, vs.DEFAULT_NAME, 50, DC)
    assert exc.value.code == "datacenter-not-pinned"


def test_a_pin_to_the_wrong_datacenter_is_refused():
    client = FakeClient(
        volumes=[_volume()],
        endpoint=_endpoint(),
        after=_endpoint(networkVolumeId=VOLUME, dataCenterIds=["EU-RO-1"]),
    )
    with pytest.raises(vs.Refused) as exc:
        vs.apply(client, ENDPOINT, vs.DEFAULT_NAME, 50, DC)
    assert exc.value.code == "datacenter-not-pinned"


def test_a_volume_document_with_no_datacenter_is_refused():
    client = FakeClient(volumes=[_volume(dataCenterId=None)],
                        endpoint=_endpoint())
    # DEFAULT_DATACENTER still covers it, so blank BOTH to reach the refusal.
    with pytest.raises(vs.Refused) as exc:
        vs.apply(client, ENDPOINT, vs.DEFAULT_NAME, 50, "")
    assert exc.value.code == "volume-datacenter-unknown"


def test_a_volume_with_no_id_is_refused_before_anything_is_attached():
    client = FakeClient(volumes=[_volume(id=None)], endpoint=_endpoint())
    with pytest.raises(vs.Refused) as exc:
        vs.apply(client, ENDPOINT, vs.DEFAULT_NAME, 50, DC)
    assert exc.value.code == "volume-id-missing"
    assert client.attached == []


def test_an_attach_that_did_not_land_is_refused():
    client = FakeClient(
        volumes=[_volume()],
        endpoint=_endpoint(),
        after=_endpoint(networkVolumeId=""),
    )
    with pytest.raises(vs.Refused) as exc:
        vs.apply(client, ENDPOINT, vs.DEFAULT_NAME, 50, DC)
    assert exc.value.code == "attach-not-applied"


@pytest.mark.parametrize("field,value", [
    ("templateId", "someothertemplate"),
    ("gpuTypeIds", ["NVIDIA A100"]),
    ("workersMin", 2),
    ("workersMax", 5),
    ("workersStandby", 3),
    ("idleTimeout", 600),
    ("executionTimeoutMs", 600000),
])
def test_any_collateral_movement_takes_the_run_down(field, value):
    client = FakeClient(
        volumes=[_volume()],
        endpoint=_endpoint(),
        after=_endpoint(networkVolumeId=VOLUME, dataCenterIds=[DC],
                        **{field: value}),
    )
    with pytest.raises(vs.Refused) as exc:
        vs.apply(client, ENDPOINT, vs.DEFAULT_NAME, 50, DC)
    assert exc.value.code == "collateral-change"


def test_the_measured_rate_is_used_and_not_invented():
    assert vs.USD_PER_GB_MONTH == 0.07
    assert vs.monthly_cost(50) == 3.5
    assert vs.monthly_cost(100) == 7.0


def test_detaching_sends_no_datacenter_so_the_pin_goes_with_the_volume():
    """Reversibility is the whole reason the attach was made cheap. An
    empty volume id must not leave the endpoint pinned to a datacenter it
    no longer has storage in."""
    import inspect

    import runpod_client

    source = inspect.getsource(runpod_client.attach_network_volume)
    assert "if volume_id and datacenter_id:" in source, (
        "the datacenter must be conditional on there being a volume"
    )
