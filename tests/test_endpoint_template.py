"""Pointing an endpoint at an EXISTING template.

The endpoint 9gh6qbou1in8yb came out of the console referencing a template
that is not on the account. The failure these tests exist to prevent is
replacing that dangling reference with another one and reporting success.
"""

import pytest

from validation import endpoint_template as et

ENDPOINT = "9gh6qbou1in8yb"
GOOD = "aqa3wkdf8g"
DANGLING = "ei22bjog46"


class FakeClient:
    def __init__(self, *, templates, endpoint, after=None, patch_raises=None):
        self._templates = templates
        self._endpoint = dict(endpoint)
        self._after = after
        self._patch_raises = patch_raises
        self.patched = []
        self.reads = 0

    def list_templates_graphql(self):
        return self._templates

    def get_endpoint(self, endpoint_id):
        self.reads += 1
        if self.patched and self._after is not None:
            return "{}", dict(self._after)
        return "{}", dict(self._endpoint)

    def attach_template(self, endpoint_id, template_id):
        if self._patch_raises:
            raise self._patch_raises
        self.patched.append((endpoint_id, template_id))
        return "{}", {}


def _endpoint(**over):
    doc = {
        "id": ENDPOINT,
        "templateId": DANGLING,
        "gpuTypeIds": ["NVIDIA RTX A5000"],
        "workersMin": 1,
        "workersMax": 1,
        "workersStandby": 1,
        "idleTimeout": 5,
        "executionTimeoutMs": 600000,
        "networkVolumeId": "",
    }
    doc.update(over)
    return doc


def _templates(*ids):
    return [{"id": i, "name": i} for i in ids]


def test_the_happy_path_sends_only_the_template_id():
    client = FakeClient(
        templates=_templates(GOOD, "runpod-ubuntu"),
        endpoint=_endpoint(),
        after=_endpoint(templateId=GOOD),
    )
    code, result = et.report(client, ENDPOINT, GOOD, et.TOKEN)
    assert code == 0
    assert client.patched == [(ENDPOINT, GOOD)]
    assert result["template_id_before"] == DANGLING
    assert result["already_pointed"] is False


def test_a_template_not_on_the_account_is_refused_before_any_write():
    """The whole point. One dangling reference must not become another."""
    client = FakeClient(
        templates=_templates(GOOD),
        endpoint=_endpoint(),
    )
    code, result = et.report(client, ENDPOINT, "nosuchtemplate", et.TOKEN)
    assert code == 1
    assert result["refused"] == "target-template-missing"
    assert client.patched == [], "nothing may be written on a refusal"


def test_an_unreadable_template_list_refuses_rather_than_guessing():
    client = FakeClient(templates=None, endpoint=_endpoint())
    code, result = et.report(client, ENDPOINT, GOOD, et.TOKEN)
    assert code == 1
    assert result["refused"] == "templates-unreadable"
    assert client.patched == []


def test_the_literal_token_is_required():
    client = FakeClient(templates=_templates(GOOD), endpoint=_endpoint())
    for token in ("", "point-at-template", "ATTACH-TEMPLATE", "yes"):
        code, result = et.report(client, ENDPOINT, GOOD, token)
        assert code == 1, token
        assert result["refused"] == "token-wrong"
    assert client.patched == []


def test_a_blank_identifier_is_refused():
    client = FakeClient(templates=_templates(GOOD), endpoint=_endpoint())
    assert et.report(client, "", GOOD, et.TOKEN)[0] == 1
    assert et.report(client, ENDPOINT, "", et.TOKEN)[0] == 1
    assert client.patched == []


def test_a_patch_that_did_not_land_is_a_refusal_not_a_success():
    client = FakeClient(
        templates=_templates(GOOD),
        endpoint=_endpoint(),
        after=_endpoint(templateId=DANGLING),
    )
    code, result = et.report(client, ENDPOINT, GOOD, et.TOKEN)
    assert code == 1
    assert result["refused"] == "not-applied"


@pytest.mark.parametrize("field,value", [
    ("workersMin", 3),
    ("workersMax", 5),
    ("workersStandby", 2),
    ("gpuTypeIds", ["NVIDIA A100"]),
    ("idleTimeout", 600),
    ("executionTimeoutMs", 60000),
    ("networkVolumeId", "someothervolume"),
])
def test_any_collateral_movement_takes_the_run_down(field, value):
    """A PATCH that quietly rewrites a spend bound is the exact failure the
    2026-08-30 template retarget produced when it dropped
    containerRegistryAuthId — invisible in everything the write printed."""
    client = FakeClient(
        templates=_templates(GOOD),
        endpoint=_endpoint(),
        after=_endpoint(templateId=GOOD, **{field: value}),
    )
    code, result = et.report(client, ENDPOINT, GOOD, et.TOKEN)
    assert code == 1
    assert result["refused"] == "collateral-change"


def test_re_running_against_the_same_template_sends_no_patch():
    """Idempotence with teeth: a templateId PATCH destroys the running
    worker, so a no-op re-run must not cost a 25 GiB pull."""
    client = FakeClient(
        templates=_templates(GOOD),
        endpoint=_endpoint(templateId=GOOD),
    )
    code, result = et.report(client, ENDPOINT, GOOD, et.TOKEN)
    assert code == 0
    assert result["already_pointed"] is True
    assert client.patched == []


def test_an_api_failure_mid_write_never_reports_success():
    client = FakeClient(
        templates=_templates(GOOD),
        endpoint=_endpoint(),
        patch_raises=RuntimeError("PATCH -> 500"),
    )
    code, result = et.report(client, ENDPOINT, GOOD, et.TOKEN)
    assert code == 1
    assert result["refused"] == "RuntimeError"


def test_the_guarded_list_covers_every_spend_bearing_field():
    for field in ("workersMin", "workersMax", "workersStandby", "gpuTypeIds",
                  "executionTimeoutMs", "networkVolumeId", "idleTimeout"):
        assert field in et.GUARDED, field
