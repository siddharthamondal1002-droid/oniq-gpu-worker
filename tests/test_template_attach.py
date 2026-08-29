"""The attach mutation: every refusal, and nothing written when one fires.

This is the only code in the repo that creates a RunPod template and
repoints a paid endpoint. The tests are written the way stale_run's are —
each precondition gets a test that it refuses, and a test that the refusal
happened BEFORE any write.
"""

import pytest

from validation import template_attach as ta

GOOD = "ghcr.io/owner/oniq-gpu-worker@sha256:" + "a" * 64
ENDPOINT = "p3zmlv8ek10dzt"


class Client:
    def __init__(self, endpoint=None, templates=None, created=None, after=None):
        self._endpoint = endpoint if endpoint is not None else {"id": ENDPOINT, "templateId": "gone"}
        self._templates = templates or {}
        self._created = created if created is not None else {"id": "newtmpl"}
        self._after = after
        self.writes = []
        self.reads = 0

    def get_endpoint(self, endpoint_id):
        self.reads += 1
        if isinstance(self._endpoint, Exception):
            raise self._endpoint
        if self._after is not None and self.writes:
            return "{}", self._after
        return "{}", self._endpoint

    def get_template(self, template_id):
        if template_id not in self._templates:
            raise RuntimeError("404")
        return "{}", self._templates[template_id]

    def create_template(self, name, image_name, container_disk_gb):
        self.writes.append(("create", name, image_name, container_disk_gb))
        return "{}", self._created

    def attach_template(self, endpoint_id, template_id):
        self.writes.append(("attach", endpoint_id, template_id))
        return "{}", {}


def _attached(template_id="newtmpl"):
    return {"id": ENDPOINT, "templateId": template_id}


# ----------------------------------------------------------- refusals

def test_the_literal_token_is_required():
    client = Client()
    with pytest.raises(ta.Refused) as exc:
        ta.attach(client, ENDPOINT, GOOD, "attach-template")
    assert exc.value.code == "token-missing"
    assert client.writes == []


def test_no_token_at_all_refuses():
    client = Client()
    with pytest.raises(ta.Refused):
        ta.attach(client, ENDPOINT, GOOD, "")
    assert client.writes == []


@pytest.mark.parametrize(
    "image",
    [
        "ghcr.io/owner/oniq-gpu-worker:latest",
        "ghcr.io/owner/oniq-gpu-worker",
        "oniq-gpu-worker@sha256:" + "a" * 64,
        "ghcr.io/owner/w@sha256:" + "a" * 63,
        "ghcr.io/owner/w@sha256:" + "a" * 64 + ":latest",
        "ghcr.io/owner/w@sha256:" + "A" * 64,
        "",
    ],
)
def test_only_a_fully_qualified_digest_is_accepted(image):
    """A tag moves. An endpoint pinned to one can silently start running an
    image nobody reviewed, on hardware that bills by the second."""
    client = Client()
    with pytest.raises(ta.Refused) as exc:
        ta.attach(client, ENDPOINT, image, ta.AUTHORIZED_TOKEN)
    assert exc.value.code == "image-not-pinned"
    assert client.writes == []


def test_a_digest_reference_is_accepted():
    client = Client(after=_attached())
    result = ta.attach(client, ENDPOINT, GOOD, ta.AUTHORIZED_TOKEN)
    assert result["template_id"] == "newtmpl"


def test_an_unreadable_endpoint_refuses():
    client = Client(endpoint=RuntimeError("500"))
    with pytest.raises(ta.Refused) as exc:
        ta.attach(client, ENDPOINT, GOOD, ta.AUTHORIZED_TOKEN)
    assert exc.value.code == "endpoint-unreadable"
    assert client.writes == []


def test_an_endpoint_answering_with_a_different_id_refuses():
    client = Client(endpoint={"id": "someoneelse", "templateId": "gone"})
    with pytest.raises(ta.Refused) as exc:
        ta.attach(client, ENDPOINT, GOOD, ta.AUTHORIZED_TOKEN)
    assert exc.value.code == "endpoint-mismatch"
    assert client.writes == []


def test_a_template_that_still_resolves_is_not_clobbered():
    """The warrant for writing is a DANGLING reference. If it resolves,
    this would replace a working configuration."""
    client = Client(templates={"gone": {"id": "gone"}})
    with pytest.raises(ta.Refused) as exc:
        ta.attach(client, ENDPOINT, GOOD, ta.AUTHORIZED_TOKEN)
    assert exc.value.code == "template-exists"
    assert client.writes == []


def test_an_endpoint_with_no_template_at_all_may_be_attached():
    client = Client(endpoint={"id": ENDPOINT}, after=_attached())
    assert ta.attach(client, ENDPOINT, GOOD, ta.AUTHORIZED_TOKEN)["template_id"] == "newtmpl"


def test_a_create_returning_no_id_refuses_before_attaching():
    client = Client(created={})
    with pytest.raises(ta.Refused) as exc:
        ta.attach(client, ENDPOINT, GOOD, ta.AUTHORIZED_TOKEN)
    assert exc.value.code == "create-unconfirmed"
    assert [w[0] for w in client.writes] == ["create"]


def test_an_attach_that_does_not_read_back_is_unconfirmed():
    """The PATCH response is the write claiming it worked. The endpoint
    read is the endpoint saying so."""
    client = Client(after={"id": ENDPOINT, "templateId": "gone"})
    with pytest.raises(ta.Refused) as exc:
        ta.attach(client, ENDPOINT, GOOD, ta.AUTHORIZED_TOKEN)
    assert exc.value.code == "attach-unconfirmed"


# ----------------------------------------------------------- the write

def test_the_create_sends_the_digest_and_a_disk_but_never_env():
    client = Client(after=_attached())
    ta.attach(client, ENDPOINT, GOOD, ta.AUTHORIZED_TOKEN)
    kind, name, image, disk = client.writes[0]
    assert kind == "create"
    assert image == GOOD
    assert name == ta.TEMPLATE_NAME
    assert disk == ta.CONTAINER_DISK_GB
    # The benchmark's largest candidate is 117.52 GiB of weights and the
    # baked image already holds ~40-52 GiB. A disk that cannot hold both
    # cannot evaluate the model at all — which is what 80 GB meant.
    assert ta.CONTAINER_DISK_GB >= 160, (
        "too small to fetch Wan2.2 (117.52 GiB) beside the baked image"
    )


def test_create_template_has_no_parameter_that_could_carry_a_credential():
    """storage.py's contract: the three R2 variables live in the RunPod
    environment and nowhere else. The safest way to honour that is to have
    no parameter capable of carrying one."""
    import inspect

    import runpod_client

    params = set(inspect.signature(runpod_client.create_template).parameters)
    assert params == {"name", "image_name", "container_disk_gb"}


def test_the_attach_names_the_endpoint_and_the_new_template_only():
    client = Client(after=_attached())
    ta.attach(client, ENDPOINT, GOOD, ta.AUTHORIZED_TOKEN)
    assert client.writes[1] == ("attach", ENDPOINT, "newtmpl")


def test_the_endpoint_patch_body_carries_templateid_and_nothing_else():
    """PATCH also accepts workersMax, workersMin, gpuTypeIds and the
    timeouts. Sending any of them, even at a value just read, would let a
    stale read rewrite the endpoint's spend bounds."""
    import inspect

    import runpod_client

    source = inspect.getsource(runpod_client.attach_template)
    body = source.split("body=", 1)[1]
    for field in ("workersMax", "workersMin", "gpuTypeIds", "idleTimeout",
                  "executionTimeoutMs", "scalerValue"):
        assert field not in body


def test_the_report_names_the_env_vars_that_remain_the_owners(capsys):
    code, _ = ta.report(Client(after=_attached()), ENDPOINT, GOOD, ta.AUTHORIZED_TOKEN)
    out = capsys.readouterr().out
    assert code == 0
    import storage

    for name in storage.REQUIRED_VARS:
        assert name in out


def test_a_refusal_reports_that_nothing_was_written(capsys):
    code, _ = ta.report(Client(), ENDPOINT, GOOD, "wrong")
    out = capsys.readouterr().out
    assert code == 1
    assert "NOTHING WAS WRITTEN" in out
