"""standby_zero: verify -> patch the literal zero -> fresh-read verify.

The fake client records every call so the tests can prove the module
never patches when already zero, never trusts the PATCH echo, and
refuses ambiguous or drifted endpoints before touching anything.
"""

import json

import pytest

from validation import spend_run, standby_zero


def _ep(standby, mn=0, mx=1):
    return {
        "id": "ep-123",
        "name": "oniq-gpu-worker",
        "workersMin": mn,
        "workersMax": mx,
        "workersStandby": standby,
        "gpuTypeIds": ["NVIDIA GeForce RTX 3090"],
    }


class FakeClient:
    def __init__(self, reads, patch_status=200, patch_body="{}"):
        # `reads` is the sequence of endpoint lists successive
        # get_endpoints calls return — the module must re-read after a
        # patch rather than trusting its echo.
        self.reads = list(reads)
        self.patch_status = patch_status
        self.patch_body = patch_body
        self.patched = []

    def get_endpoints(self):
        doc = self.reads.pop(0)
        return json.dumps(doc), doc

    def set_workers_standby_zero(self, endpoint_id):
        self.patched.append(endpoint_id)
        return self.patch_status, self.patch_body


def test_already_zero_verifies_without_patching():
    client = FakeClient(reads=[[_ep(standby=0)]])
    result = standby_zero.run(client)
    assert result == {"endpoint_id": "ep-123", "workers_standby": 0, "changed": False}
    assert client.patched == []


def test_standby_one_patches_and_verifies_by_fresh_read():
    client = FakeClient(reads=[[_ep(standby=1)], [_ep(standby=0)]])
    result = standby_zero.run(client)
    assert result["changed"] is True
    assert client.patched == ["ep-123"]
    assert client.reads == []  # the fresh read actually happened


def test_patch_refusal_is_a_stop_and_the_full_body_is_printed(capsys):
    # Run #35 lost the informative half of the diagnosis to a truncating
    # slice in the STOP message — the body now prints in FULL first.
    client = FakeClient(
        reads=[[_ep(standby=1)]],
        patch_status=405,
        patch_body='{"error": "method not allowed"}' + "x" * 400,
    )
    with pytest.raises(spend_run.SpendStop) as exc:
        standby_zero.run(client)
    assert exc.value.code == "standby-patch-refused"
    out = capsys.readouterr().out
    assert "method not allowed" in out
    assert "x" * 400 in out  # nothing truncated


def test_fresh_read_still_nonzero_is_a_stop():
    # A 200 PATCH whose fresh read still shows 1 is a failed change, not
    # a success with a caveat.
    client = FakeClient(reads=[[_ep(standby=1)], [_ep(standby=1)]])
    with pytest.raises(spend_run.SpendStop) as exc:
        standby_zero.run(client)
    assert exc.value.code == "standby-not-zero"


def test_multiple_endpoints_refuse_to_patch_anything():
    client = FakeClient(reads=[[_ep(standby=1), dict(_ep(standby=1), id="ep-456")]])
    with pytest.raises(spend_run.SpendStop) as exc:
        standby_zero.run(client)
    assert exc.value.code == "endpoint-not-singular"
    assert client.patched == []


def test_min_max_drift_stops_before_any_patch():
    client = FakeClient(reads=[[_ep(standby=1, mn=1)]])
    with pytest.raises(spend_run.SpendStop) as exc:
        standby_zero.run(client)
    assert exc.value.code == "endpoint-config-drift"
    assert client.patched == []


def test_never_prints_secret_values(capsys):
    ep = _ep(standby=0)
    ep["env"] = {"R2_SECRET_ACCESS_KEY": "SUPERSECRETVALUE"}
    client = FakeClient(reads=[[ep]])
    standby_zero.run(client)
    out = capsys.readouterr().out
    assert "SUPERSECRETVALUE" not in out
    assert "R2_SECRET_ACCESS_KEY" in out  # the NAME may appear
