"""Attaching the model volume, and the half of it that was missing.

MEASURED 2026-08-30. The attach sent networkVolumeId alone, on a
one-field-per-PATCH principle that is right for most endpoint writes and
wrong for this one. A network volume is datacenter-scoped: an endpoint
holding a volume in one datacenter can only run in that datacenter. That
was measured against US-MO-2, which RunPod has since removed — taking the
account's only volume with it and leaving the endpoint's networkVolumeId
empty. The fixture below therefore uses a LIVE id; the lesson is about the
pinning, not about that datacenter. With the volume set
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
# A datacenter the endpoint schema actually accepts. It was US-MO-2 until
# 2026-09-01, when that id stopped existing; apply() now refuses an id the
# endpoint could never place a worker in, so the fixture has to name a real
# one. Any member of KNOWN_DATACENTERS would do — the tests are about the
# pinning behaviour, not this id.
DC = "US-KS-2"


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
    """A volume whose document names no datacenter cannot be pinned against.

    Reaching this refusal used to mean blanking the caller's id so the
    DEFAULT_DATACENTER fallback was exposed. There is no default any more —
    blanking it now stops earlier, at datacenter-required — so the caller
    passes a REAL id and the refusal comes from the volume document itself,
    which is the case this test was always about.
    """
    client = FakeClient(volumes=[_volume(dataCenterId=None)],
                        endpoint=_endpoint())
    with pytest.raises(vs.Refused) as exc:
        vs.apply(client, ENDPOINT, vs.DEFAULT_NAME, 50, DC)
    assert exc.value.code == "volume-datacenter-unknown"


def test_a_removed_datacenter_is_refused_before_any_create():
    """US-MO-2 was this module's default until RunPod removed it. A create
    against a dead id costs a run to discover; an endpoint pinned to a
    datacenter it cannot be placed in is worse, and is exactly how this
    endpoint was stranded."""
    client = FakeClient(volumes=[], endpoint=_endpoint())
    with pytest.raises(vs.Refused) as exc:
        vs.apply(client, ENDPOINT, vs.DEFAULT_NAME, 50, "US-MO-2")
    assert exc.value.code == "datacenter-unknown"
    assert client.attached == []
    assert "US-MO-2" not in vs.KNOWN_DATACENTERS


def test_the_default_datacenter_is_a_recorded_choice_not_a_guess():
    """This was None on purpose until the owner named one on 2026-09-01.

    No API says which datacenters sell volumes AND carry the approved cards,
    so an id picked HERE would have been a guess; one read off the console and
    recorded is a decision. What the constant must never be is unvalidated —
    a typo in it pins the endpoint to a datacenter that cannot place a worker,
    which is the US-MO-2 failure with a different name."""
    assert vs.DEFAULT_DATACENTER in vs.KNOWN_DATACENTERS

    client = FakeClient(volumes=[], endpoint=_endpoint())
    with pytest.raises(vs.Refused) as exc:
        vs.apply(client, ENDPOINT, vs.DEFAULT_NAME, 50, "")
    assert exc.value.code == "datacenter-required"
    assert client.attached == []


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


def test_a_volume_in_another_datacenter_is_refused_not_attached():
    """A volume reused BY NAME can sit somewhere else entirely, and attaching
    it pins the endpoint there — which is how US-MO-2 stranded this endpoint.

    The old code read `volume.get("dataCenterId") or datacenter_id`, so a
    mismatch was invisible: the caller's own id stood in for the answer. That
    fallback asserted the very thing that must be verified."""
    client = FakeClient(volumes=[_volume(dataCenterId="EU-RO-1")],
                        endpoint=_endpoint())
    with pytest.raises(vs.Refused) as exc:
        vs.apply(client, ENDPOINT, vs.DEFAULT_NAME, 50, DC)
    assert exc.value.code == "volume-datacenter-mismatch"
    assert client.attached == []
