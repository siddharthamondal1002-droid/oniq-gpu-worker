"""Deleting a network volume — the one call here that cannot be undone.

OWNER DIRECTIVE 2026-09-01: "remove any unnecessary billing." Under option
B the account's volumes mount nowhere and bill for storage nothing reads.
Every other writer in this repository is reversible by running its
opposite; this one is not, which is why the token has to name its target.
"""

import pytest

from validation import volume_delete as vd

VOLUME = "l99s0q5kd2"
OTHER = "vhxqqd8vhj"


def _volume(vid=VOLUME, **over):
    row = {"id": vid, "name": "oniq-models", "size": 50,
           "dataCenterId": "EU-RO-1"}
    row.update(over)
    return row


class FakeClient:
    REST_BASE = "https://rest.example/v1"

    def __init__(self, volumes=None, endpoints=None, survives=False,
                 volumes_raise=False, endpoints_raise=False):
        self._volumes = list(volumes if volumes is not None else [_volume()])
        self._endpoints = list(endpoints or [])
        self._survives = survives
        self._volumes_raise = volumes_raise
        self._endpoints_raise = endpoints_raise
        self.deleted = []

    def _get_json(self, url):
        if url.endswith("/networkvolumes"):
            if self._volumes_raise:
                raise RuntimeError("503")
            return "", list(self._volumes)
        if url.endswith("/endpoints"):
            if self._endpoints_raise:
                raise RuntimeError("503")
            return "", list(self._endpoints)
        raise AssertionError(f"unexpected read: {url}")

    def delete_network_volume(self, volume_id):
        self.deleted.append(volume_id)
        if not self._survives:
            self._volumes = [v for v in self._volumes if v["id"] != volume_id]
        return 204, ""


def _token(vid=VOLUME):
    return f"{vd.LITERAL}:{vid}"


def test_the_token_must_carry_the_id_it_authorizes():
    """A deletion has to name its own target. Authorizing 'a deletion' and
    taking the id from somewhere else is how the wrong volume goes."""
    for bad in (vd.LITERAL, f"{vd.LITERAL}:", "", "delete", VOLUME,
                "DELETE-VOLUMES:x"):
        client = FakeClient()
        with pytest.raises(vd.Refused) as exc:
            vd.delete(client, bad)
        assert exc.value.code == "token-malformed", bad
        assert client.deleted == []


def test_it_deletes_the_named_volume_and_reports_the_billing_removed():
    client = FakeClient()
    result = vd.delete(client, _token())
    assert client.deleted == [VOLUME]
    assert result["size_gb"] == 50
    # 50 GB at the rate measured from this account's own charge.
    assert result["monthly_usd_saved"] == 3.5
    assert result["volumes_remaining"] == []


def test_the_rate_is_the_one_the_volume_path_measured():
    """Two copies of a price disagree eventually, and the disagreement shows
    up as a cost report nobody can reconcile."""
    import inspect

    from validation.volume_setup import USD_PER_GB_MONTH

    assert vd.USD_PER_GB_MONTH is USD_PER_GB_MONTH
    assert "from validation.volume_setup import USD_PER_GB_MONTH" in (
        inspect.getsource(vd))


def test_a_volume_an_endpoint_still_holds_is_refused():
    """Deleting storage out from under a live endpoint takes its weights
    away mid-flight."""
    client = FakeClient(endpoints=[{"id": "mbnif6m7jbyf93",
                                    "networkVolumeId": VOLUME}])
    with pytest.raises(vd.Refused) as exc:
        vd.delete(client, _token())
    assert exc.value.code == "volume-attached"
    assert "mbnif6m7jbyf93" in exc.value.detail
    assert client.deleted == []


def test_the_plural_reference_alone_also_refuses():
    """On this account the singular field read empty while the list still
    held a volume, on 2026-09-01. Checking one is not checking both."""
    client = FakeClient(endpoints=[{"id": "mbnif6m7jbyf93",
                                    "networkVolumeId": "",
                                    "networkVolumeIds": [VOLUME]}])
    with pytest.raises(vd.Refused) as exc:
        vd.delete(client, _token())
    assert exc.value.code == "volume-attached"
    assert client.deleted == []


def test_an_id_not_on_the_account_is_refused():
    """Deleting nothing while reporting success leaves the real volume
    billing and the operator believing it was handled."""
    client = FakeClient(volumes=[_volume(OTHER)])
    with pytest.raises(vd.Refused) as exc:
        vd.delete(client, _token(VOLUME))
    assert exc.value.code == "volume-not-found"
    assert client.deleted == []


def test_an_unreadable_account_deletes_nothing():
    """UNKNOWN is not 'no volumes', and it is certainly not permission."""
    client = FakeClient(volumes_raise=True)
    with pytest.raises(vd.Refused) as exc:
        vd.delete(client, _token())
    assert exc.value.code == "account-unreadable"
    assert client.deleted == []


def test_unreadable_endpoints_delete_nothing():
    """Whether anything still mounts this volume would be a guess."""
    client = FakeClient(endpoints_raise=True)
    with pytest.raises(vd.Refused) as exc:
        vd.delete(client, _token())
    assert exc.value.code == "endpoints-unreadable"
    assert client.deleted == []


def test_a_delete_that_did_not_take_is_reported_not_retried():
    client = FakeClient(survives=True)
    with pytest.raises(vd.Refused) as exc:
        vd.delete(client, _token())
    assert exc.value.code == "volume-still-listed"
    assert client.deleted == [VOLUME], "called once, not retried"


def test_only_one_volume_can_go_per_run():
    """A loop over an account's storage is how the wrong thing goes, so
    there is deliberately no batch shape to loop with."""
    import inspect

    source = inspect.getsource(vd)
    assert "for volume_id in" not in source
    assert "volume_ids" not in source


def test_the_client_refuses_an_empty_id():
    """A blank would address the collection, not one volume."""
    import runpod_client as rp

    with pytest.raises(rp.RunPodApiError):
        rp.delete_network_volume("")
