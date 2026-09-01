"""Widening the placement pool back to every datacenter.

THE FAULT, from the console's Releases tab on 2026-09-01:

    Release #3   locations   ALL -> EU-RO-1
    Release #7   networkVolumeId removed   (locations NOT touched)

Attaching a volume narrows placement to one datacenter. Detaching does
not widen it back. The endpoint then dies the day that one datacenter's
approved VRAM tier runs dry — every worker bucket 0 INCLUDING throttled,
because there is nothing to try — and creating a fresh endpoint appears
to cure it only because a new endpoint is born at locations: ALL.
"""

import pytest

from validation import endpoint_locations as el
from validation.volume_setup import KNOWN_DATACENTERS

ENDPOINT = "mbnif6m7jbyf93"


def _endpoint(**over):
    base = {
        "id": ENDPOINT,
        "templateId": "aqa3wkdf8g",
        "gpuTypeIds": ["NVIDIA A40", "NVIDIA RTX A6000"],
        "gpuCount": 1,
        "workersMin": 0,
        "workersMax": 1,
        "workersStandby": 1,
        "idleTimeout": 60,
        "executionTimeoutMs": 2_700_000,
        "networkVolumeId": "",
        "networkVolumeIds": [],
        "scalerType": "QUEUE_DELAY",
        "scalerValue": 4,
    }
    base.update(over)
    return base


class FakeClient:
    def __init__(self, endpoint=None, after=None):
        self._endpoint = dict(endpoint or _endpoint())
        self._after = after
        self.written = []

    def get_endpoint(self, endpoint_id):
        return "{}", dict(self._endpoint)

    def set_data_center_ids(self, endpoint_id, ids):
        self.written.append((endpoint_id, list(ids)))
        after = dict(self._after) if self._after is not None else dict(self._endpoint)
        return dict(self._endpoint), after


def test_it_writes_every_datacenter_the_schema_accepts():
    """The list is the schema's own default — the state a NEW endpoint is
    born in. This restores rather than invents."""
    client = FakeClient()
    result = el.widen(client, ENDPOINT, el.TOKEN)
    assert client.written == [(ENDPOINT, list(KNOWN_DATACENTERS))]
    assert result["datacenter_count"] == 28


def test_the_list_cannot_drift_from_the_one_the_volume_path_validates():
    """Two lists of datacenter ids would eventually disagree, and the
    disagreement would only show up as an endpoint that cannot place a
    worker. There is one list."""
    import inspect

    source = inspect.getsource(el)
    assert "from validation.volume_setup import KNOWN_DATACENTERS" in source
    assert len(KNOWN_DATACENTERS) == 28


def test_the_wrong_token_writes_nothing():
    client = FakeClient()
    with pytest.raises(el.Refused) as exc:
        el.widen(client, ENDPOINT, "please")
    assert exc.value.code == "token-wrong"
    assert client.written == []


def test_it_refuses_while_a_volume_is_still_attached():
    """A worker placed in US-TX-1 cannot mount a volume in EU-RO-1.
    Widening an endpoint that still holds one schedules workers away from
    their own weights — which fails inside a job already being paid for."""
    client = FakeClient(_endpoint(networkVolumeId="l99s0q5kd2",
                                  networkVolumeIds=["l99s0q5kd2"]))
    with pytest.raises(el.Refused) as exc:
        el.widen(client, ENDPOINT, el.TOKEN)
    assert exc.value.code == "volume-still-attached"
    assert client.written == []


def test_the_plural_volume_field_alone_also_refuses():
    """The singular field read empty while the list still held a volume on
    this account earlier the same day. Checking one is not checking both."""
    client = FakeClient(_endpoint(networkVolumeId="",
                                  networkVolumeIds=["l99s0q5kd2"]))
    with pytest.raises(el.Refused) as exc:
        el.widen(client, ENDPOINT, el.TOKEN)
    assert exc.value.code == "volume-still-attached"
    assert client.written == []


def test_a_field_that_moved_without_being_asked_is_refused():
    """The 2026-08-30 template retarget sent the field it meant to change
    and silently dropped containerRegistryAuthId."""
    client = FakeClient(after=_endpoint(workersMin=1))
    with pytest.raises(el.Refused) as exc:
        el.widen(client, ENDPOINT, el.TOKEN)
    assert exc.value.code == "collateral-change"


def test_it_reports_the_field_as_unreadable_instead_of_claiming_success():
    """MEASURED 2026-09-01 on BOTH REST routes: dataCenterIds and locations
    are ABSENT from the endpoint document, not null. A module that treated
    a clean read as proof would be the `datacenter-not-pinned` wall again,
    which blocked a correct write earlier the same day."""
    client = FakeClient()
    result = el.widen(client, ENDPOINT, el.TOKEN)
    assert result["data_center_ids_readable"] is False
    assert result["locations_readable"] is False


def test_the_report_sends_the_reader_to_the_only_place_the_proof_exists(capsys):
    client = FakeClient()
    assert el.report(client, ENDPOINT, el.TOKEN) == 0
    text = capsys.readouterr().out
    assert "Releases" in text, "the console tab is the only record of this write"
    assert "$0" in text


def test_an_unreadable_endpoint_writes_nothing():
    class Dead(FakeClient):
        def get_endpoint(self, endpoint_id):
            raise RuntimeError("500")

    client = Dead()
    with pytest.raises(el.Refused) as exc:
        el.widen(client, ENDPOINT, el.TOKEN)
    assert exc.value.code == "endpoint-unreadable"
    assert client.written == []


def test_the_client_refuses_an_empty_datacenter_list():
    """An endpoint with nowhere to run is the fault this repairs, not a
    state the repair may create."""
    import runpod_client as rp

    with pytest.raises(rp.RunPodApiError):
        rp.set_data_center_ids(ENDPOINT, [])
