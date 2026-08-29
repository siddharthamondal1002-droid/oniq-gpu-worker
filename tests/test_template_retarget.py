"""Retargeting the live template: every refusal, and nothing written when one fires.

This is the only code that changes the image a paid endpoint runs without
creating anything. The tests are written the way template_attach's are — each
precondition gets a test that it refuses, and a test that the refusal happened
BEFORE any write.
"""

import pytest

from validation import template_retarget as tr

GOOD = "ghcr.io/owner/oniq-gpu-worker@sha256:" + "a" * 64
OLD = "ghcr.io/owner/oniq-gpu-worker@sha256:" + "b" * 64
ENDPOINT = "p3zmlv8ek10dzt"
TEMPLATE = "aqa3wkdf8g"


class Client:
    def __init__(self, endpoint=None, template=None, after=None, patch=None):
        self._endpoint = (
            endpoint if endpoint is not None
            else {"id": ENDPOINT, "templateId": TEMPLATE}
        )
        self._template = (
            template if template is not None
            else {"id": TEMPLATE, "imageName": OLD, "containerDiskInGb": 80,
                  "env": {"R2_S3_ENDPOINT": "{{ RUNPOD_SECRET_R2_S3_ENDPOINT }}"}}
        )
        self._after = after
        self.writes = []

    def get_endpoint(self, endpoint_id):
        if isinstance(self._endpoint, Exception):
            raise self._endpoint
        return "{}", self._endpoint

    def get_template(self, template_id):
        if isinstance(self._template, Exception):
            raise self._template
        if self.writes and self._after is not None:
            return "{}", self._after
        return "{}", self._template

    def retarget_template(self, template_id, image_name, container_disk_gb):
        self.writes.append(("retarget", template_id, image_name, container_disk_gb))
        return "{}", {}


def _after(image=GOOD, disk=200, env=None):
    return {
        "id": TEMPLATE, "imageName": image, "containerDiskInGb": disk,
        "env": env if env is not None
        else {"R2_S3_ENDPOINT": "{{ RUNPOD_SECRET_R2_S3_ENDPOINT }}"},
    }


# ------------------------------------------------------------- refusals


def test_the_literal_token_is_required():
    client = Client()
    with pytest.raises(tr.Refused) as exc:
        tr.retarget(client, ENDPOINT, TEMPLATE, GOOD, 200, "RETARGET")
    assert exc.value.code == "token-missing"
    assert client.writes == []


@pytest.mark.parametrize("image", [
    "",
    "ghcr.io/owner/repo:latest",
    "ghcr.io/owner/repo",
    "owner/repo@sha256:" + "a" * 64,          # no registry host
    "ghcr.io/owner/repo@sha256:" + "a" * 63,  # short digest
    "ghcr.io/owner/repo@sha256:" + "a" * 64 + ":tag",
])
def test_only_a_fully_qualified_digest_is_accepted(image):
    """A moving name lets an endpoint silently start running an image nobody
    reviewed, on paid hardware."""
    client = Client()
    with pytest.raises(tr.Refused) as exc:
        tr.retarget(client, ENDPOINT, TEMPLATE, image, 200, tr.AUTHORIZED_TOKEN)
    assert exc.value.code == "image-not-pinned"
    assert client.writes == []


def test_a_missing_endpoint_or_template_id_refuses():
    for endpoint, template, code in (
        ("", TEMPLATE, "endpoint-missing"),
        (ENDPOINT, "", "template-missing"),
    ):
        client = Client()
        with pytest.raises(tr.Refused) as exc:
            tr.retarget(client, endpoint, template, GOOD, 200, tr.AUTHORIZED_TOKEN)
        assert exc.value.code == code
        assert client.writes == []


def test_an_unreadable_endpoint_refuses():
    client = Client(endpoint=RuntimeError("500"))
    with pytest.raises(tr.Refused) as exc:
        tr.retarget(client, ENDPOINT, TEMPLATE, GOOD, 200, tr.AUTHORIZED_TOKEN)
    assert exc.value.code == "endpoint-unreadable"
    assert client.writes == []


def test_a_template_the_endpoint_does_not_reference_refuses():
    """Retargeting a template nothing points at would look like success and
    change nothing."""
    client = Client(endpoint={"id": ENDPOINT, "templateId": "someoneelse"})
    with pytest.raises(tr.Refused) as exc:
        tr.retarget(client, ENDPOINT, TEMPLATE, GOOD, 200, tr.AUTHORIZED_TOKEN)
    assert exc.value.code == "template-not-referenced"
    assert "someoneelse" in exc.value.detail
    assert client.writes == []


def test_a_template_with_no_readable_image_refuses():
    """Without the current image the previous configuration cannot be
    restored, so the change is not safe to make."""
    client = Client(template={"id": TEMPLATE, "containerDiskInGb": 80})
    with pytest.raises(tr.Refused) as exc:
        tr.retarget(client, ENDPOINT, TEMPLATE, GOOD, 200, tr.AUTHORIZED_TOKEN)
    assert exc.value.code == "template-unreadable"
    assert client.writes == []


def test_a_disk_that_would_shrink_refuses():
    client = Client()
    with pytest.raises(tr.Refused) as exc:
        tr.retarget(client, ENDPOINT, TEMPLATE, GOOD, 40, tr.AUTHORIZED_TOKEN)
    assert exc.value.code == "disk-would-shrink"
    assert client.writes == []


def test_an_unverified_write_refuses_rather_than_claiming_success():
    class Broken(Client):
        def get_template(self, template_id):
            if self.writes:
                raise RuntimeError("502")
            return "{}", self._template

    client = Broken()
    with pytest.raises(tr.Refused) as exc:
        tr.retarget(client, ENDPOINT, TEMPLATE, GOOD, 200, tr.AUTHORIZED_TOKEN)
    assert exc.value.code == "retarget-unverified"


def test_a_write_that_did_not_take_is_unconfirmed():
    for after, where in ((_after(image=OLD), "image"), (_after(disk=80), "disk")):
        client = Client(after=after)
        with pytest.raises(tr.Refused) as exc:
            tr.retarget(client, ENDPOINT, TEMPLATE, GOOD, 200, tr.AUTHORIZED_TOKEN)
        assert exc.value.code == "retarget-unconfirmed", where


# --------------------------------------------------------- the write itself


def test_exactly_two_fields_are_sent_and_nothing_else():
    """`name`, `env` and `dockerStartCmd` are also accepted by this PATCH.
    Sending any of them would let a stale read rewrite something nobody meant
    to touch — that is the whole reason set_template_env sends one field."""
    import inspect

    import runpod_client

    src = inspect.getsource(runpod_client.retarget_template)
    assert 'body={"imageName": image_name, "containerDiskInGb": container_disk_gb}' in src
    assert '"env"' not in src.split("body=")[1].split("\n")[0]
    assert '"name"' not in src.split("body=")[1].split("\n")[0]


def test_an_authorized_retarget_writes_once_and_records_what_it_replaced():
    client = Client(after=_after())
    result = tr.retarget(client, ENDPOINT, TEMPLATE, GOOD, 200, tr.AUTHORIZED_TOKEN)
    assert client.writes == [("retarget", TEMPLATE, GOOD, 200)]
    assert result["replaced"] == {"image": OLD, "container_disk_gb": 80}
    assert result["container_disk_gb"] == 200


def test_the_storage_env_survives_the_write():
    """The reason this updates in place rather than creating a template: a
    fresh one starts with no env, so the endpoint would run a worker that
    cannot upload its output."""
    result = tr.retarget(Client(after=_after()), ENDPOINT, TEMPLATE, GOOD, 200,
                         tr.AUTHORIZED_TOKEN)
    assert result["env_keys"] == ["R2_S3_ENDPOINT"]


def test_an_env_wiped_by_the_write_is_visible_in_the_report(capsys):
    """Names only, and it is the question that decides whether the next job
    can upload anything at all."""
    code, _ = tr.report(Client(after=_after(env={})), ENDPOINT, TEMPLATE, GOOD,
                        200, tr.AUTHORIZED_TOKEN)
    assert code == 0
    assert "env keys still present: (none)" in capsys.readouterr().out


def test_the_report_prints_the_restore_card_and_calls_the_disk_nominal(capsys):
    code, _ = tr.report(Client(after=_after()), ENDPOINT, TEMPLATE, GOOD, 200,
                        tr.AUTHORIZED_TOKEN)
    out = capsys.readouterr().out
    assert code == 0
    assert OLD in out and GOOD in out
    assert "80 GB -> 200 GB" in out
    assert "NOMINAL ONLY" in out
    assert "RESTORE" in out


def test_a_refusal_reports_that_nothing_was_written(capsys):
    code, result = tr.report(Client(), ENDPOINT, TEMPLATE, "nope", 200,
                             tr.AUTHORIZED_TOKEN)
    out = capsys.readouterr().out
    assert code == 1
    assert result == {"refused": "image-not-pinned"}
    assert "NOTHING WAS WRITTEN" in out


def test_no_credential_can_travel_through_this_module():
    """The R2 values live in RunPod and nowhere else; this module has no
    parameter that could carry one."""
    import inspect

    params = set(inspect.signature(tr.retarget).parameters)
    assert params == {"client", "endpoint_id", "template_id", "image",
                      "disk_gb", "token"}
