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


# ------------------------------------------------ the read-only probe path
#
# Owner directive 2026-08-27: run the standby diagnostic, and stop
# re-attempting a mutation already proven unsupported. Those two were in
# tension because the probes lived inside the patch-failure branch —
# reading the answer required attempting the write first. The probe path
# exists to break that tie, so the tests that matter are the ones proving
# it cannot write.


class _ProbeClient:
    """Records every call, and screams if a mutation is attempted."""

    def __init__(self, standby=1):
        self.calls = []
        self._standby = standby

    def get_endpoints(self):
        self.calls.append("get_endpoints")
        doc = [
            {
                "id": "p3zmlv8ek10dzt",
                "workersStandby": self._standby,
                "workersMin": 0,
                "workersMax": 1,
            }
        ]
        return json.dumps(doc), doc

    def standby_schema_probe(self):
        self.calls.append("standby_schema_probe")
        return {
            "endpoint_input_fields": ["gpuTypeIds", "workersMax", "workersMin"],
            "standby_shaped_mutations": [],
            "endpoint_shaped_mutations": ["saveEndpoint"],
            "worker_shaped_mutations": [],
        }

    def rest_schema_probe(self):
        self.calls.append("rest_schema_probe")
        return {
            "spec_url": "https://rest.runpod.io/openapi.json",
            "patch_endpoint_properties": ["gpuTypeIds", "workersMax", "workersMin"],
            "standby_shaped_names": ["workersStandby"],
        }

    def set_workers_standby_zero(self, endpoint_id):  # pragma: no cover
        raise AssertionError("the probe path attempted a MUTATION")


def test_the_probe_path_never_mutates():
    client = _ProbeClient()
    facts = standby_zero.read_only(client)
    assert facts["workers_standby"] == 1
    assert "set_workers_standby_zero" not in client.calls
    assert "standby_schema_probe" in client.calls
    assert "rest_schema_probe" in client.calls


def test_probe_argv_selects_the_read_only_path(capsys, monkeypatch):
    client = _ProbeClient()
    monkeypatch.setitem(__import__("sys").modules, "runpod_client", client)
    assert standby_zero.main(["probe"]) == 0
    out = capsys.readouterr().out
    # The answer the redaction bug was hiding must be visible here.
    assert "workersStandby" in out
    assert "<redacted>" not in out
    assert "standby_shaped_names" in out


def test_the_probe_reports_unreadable_rather_than_empty():
    """None is never []. An introspection that cannot run is not proof
    that no standby mutation exists — the sweep's rule, applied here."""

    class _Blind(_ProbeClient):
        def standby_schema_probe(self):
            return None

        def rest_schema_probe(self):
            return None

    client = _Blind()
    standby_zero.probe_only(client)  # must not raise
    assert "set_workers_standby_zero" not in client.calls


def test_the_mutating_path_is_still_the_only_writer():
    """The probe path must not have become a second way to write.

    Counted as CALL SITES, not as occurrences of the name: the module
    docstring also names the function, and a text count would have this
    test failing on a comment edit while still passing if someone added a
    real second call inside a string-free branch.
    """
    import ast

    tree = ast.parse(open(standby_zero.__file__, encoding="utf-8").read())
    call_sites = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "set_workers_standby_zero"
    ]
    assert len(call_sites) == 1, f"{len(call_sites)} call sites write standby"

    # ...and it is inside run(), never inside the probe path.
    writers = [
        fn.name
        for fn in ast.walk(tree)
        if isinstance(fn, ast.FunctionDef)
        and any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "set_workers_standby_zero"
            for n in ast.walk(fn)
        )
    ]
    assert writers == ["run"], writers


def test_the_read_reports_what_the_next_deployment_step_depends_on():
    """Until 2026-09-01 nothing in this repo could answer "what is this
    endpoint configured as" without going through a mode that PATCHES it, and
    three fields decide whether the next step can work at all:

      templateId          ONIQ's endpoint pointed at one that 404s while the
                          real oniq-gpu-worker template sat on the account
                          under another id.
      networkVolumeId     where a volume-resident model hydrates TO. The text
                          encoder left the image, so no volume means every
                          clip refuses.
      executionTimeoutMs  the ceiling a cold pull plus a 17.74 GiB hydrate
                          has to fit inside.
    """
    endpoint = {
        "id": "ep1", "workersStandby": 0, "workersMin": 0, "workersMax": 1,
        "templateId": "tpl1", "networkVolumeId": "vol1",
        "executionTimeoutMs": 900000, "gpuTypeIds": ["NVIDIA RTX A6000"],
    }

    class Client:
        def get_endpoints(self):
            return json.dumps({"endpoints": [endpoint]}), {"endpoints": [endpoint]}

    facts = standby_zero.read_only(Client())
    assert facts["template_id"] == "tpl1"
    assert facts["network_volume_id"] == "vol1"
    assert facts["execution_timeout_ms"] == 900000


def test_an_unattached_volume_reads_as_None_not_as_absent_key():
    """None is the answer that matters — it is the one that makes a hydrate
    impossible, so it must survive into the facts rather than vanishing."""
    endpoint = {"id": "ep1", "workersStandby": 0, "workersMin": 0,
                "workersMax": 1, "templateId": "tpl1"}

    class Client:
        def get_endpoints(self):
            return json.dumps({"endpoints": [endpoint]}), {"endpoints": [endpoint]}

    facts = standby_zero.read_only(Client())
    assert "network_volume_id" in facts
    assert facts["network_volume_id"] is None
