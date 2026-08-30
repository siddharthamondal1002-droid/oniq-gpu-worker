"""The narrowest mutation on this workflow: one endpoint, one field.

Owner authorization 2026-08-30 covered raising the execution ceiling so a
32.26 GiB in-job download can finish. It covered nothing else, and these
tests are what keep the module inside that.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation import endpoint_timeout as et  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE = {
    "id": "ep1",
    "executionTimeoutMs": 600000,
    "templateId": "tpl1",
    "gpuTypeIds": ["NVIDIA RTX A5000"],
    "workersMin": 0,
    "workersMax": 1,
    "workersStandby": 1,
    "idleTimeout": 1,
    "networkVolumeId": "",
    "scalerType": "QUEUE_DELAY",
    "scalerValue": 4,
}


class Client:
    def __init__(self, after):
        self.after = after
        self.calls = []

    def set_execution_timeout(self, endpoint_id, timeout_ms):
        self.calls.append((endpoint_id, timeout_ms))
        return dict(BASE), dict(self.after)


def test_the_happy_path_changes_the_ceiling_and_nothing_else():
    after = dict(BASE, executionTimeoutMs=2700000)
    client = Client(after)
    result = et.apply(client, "ep1", 2700000)
    assert client.calls == [("ep1", 2700000)]
    assert result["execution_timeout_ms_before"] == 600000
    assert result["execution_timeout_ms_after"] == 2700000


def test_a_patch_that_reports_success_but_did_not_apply_is_refused():
    # The provider accepting a request is not the provider honouring it.
    # Dispatching a 45-minute job against a ceiling still set to 10 spends
    # the money and produces nothing.
    client = Client(dict(BASE))
    with pytest.raises(et.Refused) as exc:
        et.apply(client, "ep1", 2700000)
    assert exc.value.code == "timeout-not-applied"


@pytest.mark.parametrize("field,value", [
    ("templateId", "someone-elses-template"),
    ("workersMin", 1),
    ("workersMax", 4),
    ("workersStandby", 3),
    ("networkVolumeId", "vol-123"),
    ("gpuTypeIds", ["NVIDIA A100"]),
])
def test_collateral_change_is_refused(field, value):
    # The 2026-08-30 template retarget sent the field it meant to change
    # and silently dropped containerRegistryAuthId; the endpoint then could
    # not pull its own image and nothing printed showed why. A write that
    # is not read back is a write nobody checked.
    after = dict(BASE, executionTimeoutMs=2700000, **{field: value})
    with pytest.raises(et.Refused) as exc:
        et.apply(Client(after), "ep1", 2700000)
    assert exc.value.code == "collateral-change"
    assert field in str(exc.value)


def test_every_spend_shaped_field_is_guarded():
    # Worker bounds are what turn a ceiling change into a bill. If one of
    # these ever leaves GUARDED, this test is the thing that notices.
    for field in ("workersMin", "workersMax", "workersStandby", "idleTimeout"):
        assert field in et.GUARDED


def test_the_module_carries_no_other_mutating_verb():
    with open(os.path.join(ROOT, "validation", "endpoint_timeout.py"),
              encoding="utf-8") as fh:
        source = fh.read()
    for forbidden in ("submit_job", "purge_queue", "cancel_job",
                      "create_template", "retarget_template",
                      "attach_template", "set_workers_standby_zero",
                      "set_template_env"):
        assert forbidden not in source, forbidden
    # One mutating client call, named once.
    assert source.count("set_execution_timeout") == 1


def test_the_workflow_job_needs_the_literal_token():
    import yaml
    with open(os.path.join(ROOT, ".github", "workflows",
                           "gpu-validation.yml"), encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    job = doc["jobs"]["endpoint_timeout"]
    condition = job["if"].strip()
    assert "inputs.mode == 'endpoint-timeout'" in condition
    assert "inputs.timeout_token == 'SET-EXECUTION-TIMEOUT'" in condition
    commands = " ".join(str(s.get("run", "")) for s in job["steps"])
    assert "validation.endpoint_timeout" in commands
    assert "inputs.endpoint_id" in commands
    for forbidden in ("SPEND", "spend_run", "submit", "workersStandby"):
        assert forbidden not in commands
