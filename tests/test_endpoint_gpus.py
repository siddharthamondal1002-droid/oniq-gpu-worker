"""Widening an endpoint's GPU set: resolved, approved, and one field only.

The failure this module exists to prevent is a string RunPod accepts and
then cannot schedule. That happened on 2026-08-30 with a network volume in
US-MO-2 — a datacenter absent from RunPod's own enum — and the endpoint
reported success while placing nothing for ninety minutes. A gpuTypeIds
value has the identical shape, so these tests hold the writer to resolving
every name against the live catalogue before it writes any of them.
"""

import pytest

from validation import endpoint_gpus as eg

ENDPOINT = "7disu6my0mloco"

A6000 = {"id": "NVIDIA RTX A6000", "display_name": "A6000 48GB",
         "memory_gb": 48, "secure_price": None, "community_price": 0.33,
         "on_demand_price": 0.33}
A40 = {"id": "NVIDIA A40", "display_name": "A40 48GB", "memory_gb": 48,
       "secure_price": 0.39, "community_price": None, "on_demand_price": 0.39}
A5000 = {"id": "NVIDIA RTX A5000", "display_name": "RTX A5000 24GB",
         "memory_gb": 24, "secure_price": None, "community_price": 0.16,
         "on_demand_price": 0.16}

CATALOGUE = [A5000, A6000, A40]


def _endpoint(**over):
    doc = {
        "id": ENDPOINT, "templateId": "xfmaf7n83n",
        "gpuTypeIds": ["NVIDIA RTX A6000"],
        "workersMin": 0, "workersMax": 1, "workersStandby": 3,
        "idleTimeout": 5, "executionTimeoutMs": 2700000,
        "networkVolumeId": "", "dataCenterIds": None,
    }
    doc.update(over)
    return doc


class FakeClient:
    def __init__(self, *, endpoint=None, after=None, catalogue=CATALOGUE,
                 status=200, raw="{}"):
        self._endpoint = endpoint if endpoint is not None else _endpoint()
        self._after = after
        self._catalogue = catalogue
        self._status, self._raw = status, raw
        self.patched = []

    def gpu_catalogue(self):
        return "{}", list(self._catalogue)

    def get_endpoint(self, endpoint_id):
        if self.patched and self._after is not None:
            return "{}", dict(self._after)
        return "{}", dict(self._endpoint)

    def set_endpoint_gpu_types(self, endpoint_id, gpu_type_ids):
        self.patched.append((endpoint_id, list(gpu_type_ids)))
        return self._status, self._raw

    def __getattr__(self, name):
        raise AssertionError(f"this module must not call {name!r}")


# ------------------------------------------------- the write it is for

def test_it_widens_to_exactly_the_approved_set():
    client = FakeClient(
        after=_endpoint(gpuTypeIds=["NVIDIA RTX A6000", "NVIDIA A40"]),
    )
    code, result = eg.report(client, ENDPOINT)
    assert code == 0
    assert client.patched == [(ENDPOINT, ["NVIDIA RTX A6000", "NVIDIA A40"])]
    assert set(result["gpus_after"]) == {"NVIDIA RTX A6000", "NVIDIA A40"}


def test_one_patch_only_and_it_carries_gpu_ids_alone():
    """Two PATCHes would restart the worker twice, and on a 201 GB image a
    restart is a fresh pull."""
    client = FakeClient(
        after=_endpoint(gpuTypeIds=["NVIDIA RTX A6000", "NVIDIA A40"]),
    )
    eg.report(client, ENDPOINT)
    assert len(client.patched) == 1


def test_an_already_widened_endpoint_is_not_patched_again(capsys):
    both = ["NVIDIA RTX A6000", "NVIDIA A40"]
    client = FakeClient(endpoint=_endpoint(gpuTypeIds=both))
    code, result = eg.report(client, ENDPOINT)
    assert code == 0
    assert client.patched == []
    assert result["already_conformed"] is True
    assert "no worker" in capsys.readouterr().out


# --------------------------------------- the US-MO-2 lesson, generalised

def test_a_card_the_catalogue_does_not_list_refuses_the_whole_write(capsys):
    """The point of the module. An unknown name must never reach the
    field — partially writing it would be worse than not writing at all."""
    client = FakeClient(catalogue=[A5000, A6000])       # no A40
    code, result = eg.report(client, ENDPOINT)
    out = capsys.readouterr().out
    assert code == 1
    assert result["refused"] == "gpu-unknown"
    assert client.patched == []
    assert "NOTHING WAS WRITTEN" in out
    assert "NVIDIA A40" in out


def test_an_empty_catalogue_is_UNKNOWN_and_never_becomes_a_write(capsys):
    client = FakeClient(catalogue=[])
    code, result = eg.report(client, ENDPOINT)
    assert code == 1
    assert result["refused"] == "catalogue-unreadable"
    assert client.patched == []
    assert "UNKNOWN" in capsys.readouterr().out


def test_an_exact_id_match_wins_and_is_unambiguous_by_definition():
    """gpuTypeIds carries the id, so a row whose id IS the wanted name
    settles it — a second row that merely looks similar cannot make the
    field's own key ambiguous."""
    lookalike = dict(A40, id="NVIDIA-A40-PCIe", display_name="NVIDIA A40")
    resolved = eg.resolve([A6000, A40, lookalike])
    assert resolved["NVIDIA A40"]["id"] == "NVIDIA A40"


def test_two_fuzzy_matches_and_no_exact_id_refuses_rather_than_picking_one():
    """Only the FALLBACK can be ambiguous, and there the module must not
    decide which card the product runs on."""
    one = dict(A40, id="NVIDIA-A40", display_name="A40 48GB")
    two = dict(A40, id="A40-SECOND", display_name="NVIDIA A40")
    with pytest.raises(eg.Refused) as caught:
        eg.resolve([A6000, one, two], wanted=("NVIDIA A40",))
    assert caught.value.code == "gpu-ambiguous"


def test_a_display_name_resolves_to_the_id_the_field_actually_takes():
    """gpuTypeIds carries the canonical id, not the marketing name. If
    RunPod ever renames the id, the display name still resolves."""
    renamed = dict(A40, id="NVIDIA-A40-48", display_name="NVIDIA A40")
    resolved = eg.resolve([A6000, renamed])
    assert resolved["NVIDIA A40"]["id"] == "NVIDIA-A40-48"


def test_resolution_happens_before_any_write():
    """An unresolvable card must not leave the endpoint half-widened."""
    client = FakeClient(catalogue=[A6000])
    with pytest.raises(eg.Refused):
        eg.apply(client, ENDPOINT)
    assert client.patched == []


# ------------------------------------------------- what it will not do

def test_it_refuses_to_overwrite_a_card_nobody_recorded(capsys):
    client = FakeClient(endpoint=_endpoint(gpuTypeIds=["NVIDIA H100 PCIe"]))
    code, result = eg.report(client, ENDPOINT)
    assert code == 1
    assert result["refused"] == "unapproved-card-present"
    assert client.patched == []
    assert "H100" in capsys.readouterr().out


def test_a_patch_that_moved_anything_else_is_a_refusal(capsys):
    client = FakeClient(
        after=_endpoint(gpuTypeIds=["NVIDIA RTX A6000", "NVIDIA A40"],
                        workersMax=3),
    )
    code, result = eg.report(client, ENDPOINT)
    assert code == 1
    assert result["refused"] == "collateral-change"
    assert "workersMax" in capsys.readouterr().out


def test_a_successful_status_that_did_not_land_is_a_refusal(capsys):
    """A 200 is the provider's echo, not the endpoint's state. The
    template retarget taught this the hard way."""
    client = FakeClient(after=_endpoint())          # unchanged
    code, result = eg.report(client, ENDPOINT)
    assert code == 1
    assert result["refused"] == "not-applied"
    assert client.patched                            # it did try


def test_a_refused_patch_names_the_body(capsys):
    client = FakeClient(status=400, raw='{"error":"enum"}')
    code, result = eg.report(client, ENDPOINT)
    assert code == 1
    assert result["refused"] == "patch-refused"
    assert "enum" in capsys.readouterr().out


def test_a_blank_endpoint_id_writes_nothing():
    client = FakeClient()
    code, result = eg.report(client, "")
    assert code == 1
    assert result["refused"] == "endpoint-missing"
    assert client.patched == []


def test_it_never_touches_the_worker_count():
    """Widening WHERE a worker may be placed must not change HOW MANY."""
    client = FakeClient(
        after=_endpoint(gpuTypeIds=["NVIDIA RTX A6000", "NVIDIA A40"]),
    )
    eg.report(client, ENDPOINT)
    _, sent = client.patched[0]
    assert sent == ["NVIDIA RTX A6000", "NVIDIA A40"]
    # The body carries gpuTypeIds and nothing else, and both worker-count
    # fields are guarded against drift on the read-back.
    assert "workersMin" in eg.GUARDED
    assert "workersMax" in eg.GUARDED


# ------------------------------------------------------- the owner's set

def test_the_approved_set_is_the_two_cards_the_owner_named():
    """Owner directive 2026-08-30. A third card cannot join by an edit
    that looks like a typo."""
    assert eg.WANTED == ("NVIDIA RTX A6000", "NVIDIA A40")


def test_the_approved_set_is_a_constant_not_a_parameter():
    """apply() takes an endpoint id and nothing else — there is no
    argument by which a dispatch could name a different card."""
    import inspect

    assert list(inspect.signature(eg.apply).parameters) == ["client", "endpoint_id"]


def test_the_price_of_what_was_authorised_is_recorded():
    """Quoted at the moment of the write, never recalled — so the log
    says what the owner's decision actually cost."""
    client = FakeClient(
        after=_endpoint(gpuTypeIds=["NVIDIA RTX A6000", "NVIDIA A40"]),
    )
    _, result = eg.report(client, ENDPOINT)
    assert result["prices"]["NVIDIA A40"]["secure"] == 0.39
    assert result["prices"]["NVIDIA RTX A6000"]["community"] == 0.33


def test_no_hardcoded_rate_lives_in_this_module():
    """The historical-price guard: a number written into the file is a
    number that goes stale silently."""
    import inspect
    import re

    source = inspect.getsource(eg)
    assert not re.search(r"\b0\.\d{2,}\b", source), "a rate is hardcoded here"
