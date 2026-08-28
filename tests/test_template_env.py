"""Setting the storage variables on the live template.

This is a WRITE against the object the paid endpoint runs, so the tests
are mostly about what it refuses. The one thing it must never do is turn
a credential into something this repository can see; the one thing it
must never break is the image the proofs were run against.
"""

import pytest

import storage
from validation import template_env as te

TEMPLATE = "aqa3wkdf8g"
IMAGE = "ghcr.io/o/w@sha256:" + "d" * 64
NAMES = "R2_S3_ENDPOINT,R2_ACCESS_KEY_ID,R2_SECRET_ACCESS_KEY"


class Client:
    def __init__(self, env=None, image=IMAGE, image_after=None, missing=False):
        self.env = dict(env or {})
        self.image = image
        self.image_after = image_after
        self.missing = missing
        self.writes = []
        self.reads = 0

    def get_template(self, template_id):
        self.reads += 1
        if self.missing:
            return 200, {}
        image = self.image
        if self.image_after is not None and self.writes:
            image = self.image_after
        return 200, {"id": template_id, "imageName": image, "env": dict(self.env)}

    def set_template_env(self, template_id, env):
        self.writes.append((template_id, dict(env)))
        self.env = dict(env)
        return "", {}

    def template_env_names_graphql(self, template_id):
        return set(self.env)


def _apply(client, names=NAMES, token=te.AUTHORIZED_TOKEN):
    return te.apply(client, TEMPLATE, names, token)


# ------------------------------------------------------ the happy write

def test_it_sets_every_required_variable_as_a_reference():
    client = Client()
    result = _apply(client)
    _, written = client.writes[0]
    assert set(written) == set(storage.REQUIRED_VARS)
    assert written["R2_ACCESS_KEY_ID"] == "{{ RUNPOD_SECRET_R2_ACCESS_KEY_ID }}"
    assert result["template_id"] == TEMPLATE


def test_the_reference_is_the_syntax_runpod_documents():
    """`{{ RUNPOD_SECRET_<name> }}` - braces, spaces and prefix exactly."""
    client = Client()
    _apply(client, names="a,b,c")
    _, written = client.writes[0]
    assert written["R2_S3_ENDPOINT"] == "{{ RUNPOD_SECRET_a }}"


def test_only_env_is_ever_sent():
    """PATCH /templates also takes imageName. A stale read that rewrote it
    would point the endpoint at an image no proof ever ran against."""
    client = Client()
    _apply(client)
    assert list(client.writes[0][1]) == list(storage.REQUIRED_VARS)


def test_a_rerun_over_its_own_keys_is_allowed():
    client = Client(env={v: "{{ RUNPOD_SECRET_x }}" for v in storage.REQUIRED_VARS})
    _apply(client)
    assert client.writes


# --------------------------------------------------------- the refusals

def test_without_the_literal_token_nothing_is_written():
    client = Client()
    with pytest.raises(te.Refused) as exc:
        _apply(client, token="please")
    assert exc.value.code == "token-missing"
    assert client.writes == []


def test_a_missing_secret_name_refuses_rather_than_guessing():
    """A guessed name yields a reference that expands to nothing, and an
    unexpanded reference looks exactly like a configured variable."""
    client = Client()
    with pytest.raises(te.Refused) as exc:
        _apply(client, names="")
    assert exc.value.code == "secret-names-missing"
    assert client.writes == []


def test_the_wrong_number_of_names_refuses():
    with pytest.raises(te.Refused) as exc:
        _apply(Client(), names="only,two")
    assert exc.value.code == "secret-names-count"


@pytest.mark.parametrize("bad", ["has space", "brace}", "{{x}}", "", "x" * 65])
def test_an_unusable_secret_name_refuses(bad):
    client = Client()
    with pytest.raises(te.Refused):
        _apply(client, names=f"{bad},b,c")
    assert client.writes == []


def test_foreign_env_is_refused_rather_than_clobbered():
    """env is written whole, so anything already there would be erased."""
    client = Client(env={"HF_TOKEN": "x"})
    with pytest.raises(te.Refused) as exc:
        _apply(client)
    assert exc.value.code == "env-not-empty"
    assert "HF_TOKEN" in exc.value.detail
    assert client.writes == []


def test_an_unreadable_env_refuses_rather_than_writing_over_UNKNOWN():
    class Blind(Client):
        def get_template(self, template_id):
            if self.reads == 1:
                self.reads += 1
                return 200, {"id": template_id, "imageName": IMAGE, "env": "???"}
            self.reads += 1
            return 200, {"id": template_id, "imageName": IMAGE, "env": {}}

    with pytest.raises(te.Refused) as exc:
        _apply(Blind())
    assert exc.value.code == "env-unreadable"


def test_a_template_that_is_not_there_refuses():
    client = Client(missing=True)
    with pytest.raises(te.Refused) as exc:
        _apply(client)
    assert exc.value.code == "template-missing"
    assert client.writes == []


def test_an_image_that_drifts_across_the_write_refuses():
    client = Client(image_after="ghcr.io/o/w:latest")
    with pytest.raises(te.Refused) as exc:
        _apply(client)
    assert exc.value.code == "image-drift"


def test_a_write_that_did_not_take_is_caught_by_the_reread():
    class Liar(Client):
        def set_template_env(self, template_id, env):
            self.writes.append((template_id, dict(env)))
            return "", {}          # claims success, changes nothing

    with pytest.raises(te.Refused) as exc:
        _apply(Liar())
    assert exc.value.code == "verify-incomplete"


def test_both_views_must_confirm_not_just_one():
    class HalfBlind(Client):
        def template_env_names_graphql(self, template_id):
            return set()

    with pytest.raises(te.Refused) as exc:
        _apply(HalfBlind())
    assert exc.value.code == "verify-incomplete"
    assert "GraphQL" in exc.value.detail


# ---------------------------------------------------------- the report

def test_the_report_never_prints_a_reference_value(capsys):
    code, _ = te.report(Client(), TEMPLATE, NAMES, te.AUTHORIZED_TOKEN)
    out = capsys.readouterr().out
    assert code == 0
    assert "RUNPOD_SECRET" not in out
    assert "R2_ACCESS_KEY_ID" in out


def test_a_refusal_says_nothing_was_written(capsys):
    code, facts = te.report(Client(), TEMPLATE, NAMES, "nope")
    out = capsys.readouterr().out
    assert code == 1
    assert "NOTHING WAS WRITTEN" in out
    assert facts["refused"] == "token-missing"
