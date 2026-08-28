"""The template probe: read-only, and it must not call a create no one asked for.

The endpoint's templateId is a dangling reference. The tempting fix is to POST
a new template - but a template names an image, and an image that does not
exist anywhere cannot be named. These tests hold the probe to reporting that
rather than acting on it.
"""

from validation import template_probe as tp

_UNSET = object()


class Client:
    """Records every call, so a mutation shows up as a failed assertion."""

    def __init__(self, templates, surface, env=None, rest=_UNSET):
        self._templates = templates
        self._surface = surface
        self._env = env
        self._rest = rest
        self.calls = []

    def list_templates_graphql(self):
        self.calls.append("list_templates_graphql")
        return self._templates

    def rest_template_surface(self):
        self.calls.append("rest_template_surface")
        return self._surface

    def template_env_names_graphql(self, template_id):
        self.calls.append("template_env_names_graphql")
        return self._env

    def get_template(self, template_id):
        self.calls.append("get_template")
        if self._rest is _UNSET:            # default: REST agrees with GraphQL
            if self._env is None:
                raise AssertionError("unreadable")
            return 200, {"env": {k: "value-that-must-never-be-printed" for k in self._env}}
        return 200, self._rest

    def __getattr__(self, name):
        raise AssertionError(f"the probe must not call {name!r}")


IMAGE_ONLY = {
    "template_paths": {"/templates": ["GET", "POST"]},
    "endpoint_verbs": ["delete", "get", "patch"],
    "template_create_body": {
        "required": ["name", "imageName"],
        "properties": ["env", "imageName", "isServerless", "name", "ports"],
    },
    "endpoint_patch_body": {"required": [], "properties": ["templateId"]},
    "all_paths": {"/templates": ["GET", "POST"]},
    "build_like_paths": [],
}


def test_a_missing_template_is_named_a_dangling_reference(capsys):
    client = Client([{"id": "other", "name": "stock", "imageName": "x"}], IMAGE_ONLY)
    tp.report(client, "hhhdwtjw0y")
    out = capsys.readouterr().out
    assert "MISSING: hhhdwtjw0y" in out
    assert "dangling reference" in out


def test_a_present_template_is_not_reported_missing(capsys):
    client = Client([{"id": "hhhdwtjw0y", "name": "oniq", "imageName": "x"}], IMAGE_ONLY)
    tp.report(client, "hhhdwtjw0y")
    out = capsys.readouterr().out
    assert "FOUND: hhhdwtjw0y" in out
    assert "MISSING" not in out


def test_an_image_only_create_body_is_reported_as_having_no_build_field(capsys):
    tp.report(Client([], IMAGE_ONLY), "hhhdwtjw0y")
    out = capsys.readouterr().out
    assert "NO BUILD FIELD" in out
    assert "BUILD FIELD PRESENT" not in out


def test_a_create_body_carrying_a_repository_is_reported_as_a_build_field(capsys):
    surface = dict(IMAGE_ONLY)
    surface["template_create_body"] = {
        "required": ["name"],
        "properties": ["githubRepo", "imageName", "name"],
    }
    tp.report(Client([], surface), "hhhdwtjw0y")
    out = capsys.readouterr().out
    assert "BUILD FIELD PRESENT" in out
    assert "githubRepo" in out


def test_unreadable_templates_are_unknown_never_none_exist(capsys):
    tp.report(Client(None, IMAGE_ONLY), "hhhdwtjw0y")
    out = capsys.readouterr().out
    assert "unreadable" in out
    assert "MISSING" not in out


def test_the_probe_only_reads(capsys):
    client = Client([], IMAGE_ONLY)
    tp.report(client, "hhhdwtjw0y")
    assert client.calls == ["list_templates_graphql", "rest_template_surface"]


def test_a_blank_id_refuses_rather_than_printing_a_trivially_true_missing(capsys):
    client = Client([{"id": "other", "name": "stock", "imageName": "x"}], IMAGE_ONLY)
    code, _ = tp.report(client, "")
    out = capsys.readouterr().out
    assert code == 2
    assert "MISSING" not in out
    assert "NO TEMPLATE ID GIVEN" in out
    assert client.calls == []


def test_whitespace_is_not_an_id_either(capsys):
    code, _ = tp.report(Client([], IMAGE_ONLY), "   ")
    assert code == 2
    assert "NO TEMPLATE ID GIVEN" in capsys.readouterr().out


# ------------------------------------------- the storage variables

R2 = {"R2_S3_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"}
HERE = [{"id": "aqa3wkdf8g", "name": "oniq-gpu-worker", "imageName": "x@sha256:1"}]


def test_every_required_variable_set_reads_as_ready(capsys):
    code, facts = tp.report(Client(HERE, IMAGE_ONLY, env=R2 | {"PYTHONUNBUFFERED"}), "aqa3wkdf8g")
    out = capsys.readouterr().out
    assert "STORAGE READY" in out
    assert code == 0
    assert facts["storage"] is True


def test_one_absent_variable_names_it_and_fails_the_probe(capsys):
    code, facts = tp.report(Client(HERE, IMAGE_ONLY, env=R2 - {"R2_SECRET_ACCESS_KEY"}), "aqa3wkdf8g")
    out = capsys.readouterr().out
    assert "STORAGE NOT READY" in out
    assert "R2_SECRET_ACCESS_KEY" in out
    assert "NOT LAUNCH READY" in out
    assert code == 1
    assert facts["storage"] is False


def test_an_unreadable_env_is_UNKNOWN_and_never_becomes_ready(capsys):
    """The standing rule: UNKNOWN is never converted into success. A job
    started against an unverified template is the one that fails after it
    has already been paid for."""
    code, facts = tp.report(Client(HERE, IMAGE_ONLY, env=None), "aqa3wkdf8g")
    out = capsys.readouterr().out
    assert "UNKNOWN" in out
    assert "STORAGE READY" not in out
    assert code == 1
    assert facts["storage"] is None


def test_the_env_is_read_by_the_names_only_query(capsys):
    """runpod_client has two template readers; only one of them asks the
    API for values, and it strips them. The probe must use that one."""
    client = Client(HERE, IMAGE_ONLY, env=R2)
    tp.report(client, "aqa3wkdf8g")
    assert "template_env_names_graphql" in client.calls


def test_a_template_that_is_not_there_is_not_asked_for_its_env(capsys):
    """No point querying the environment of something that does not exist,
    and no reason to pull secret material for a question nobody asked."""
    client = Client([{"id": "other", "name": "stock", "imageName": "x"}], IMAGE_ONLY)
    code, facts = tp.report(client, "aqa3wkdf8g")
    assert "template_env_names_graphql" not in client.calls
    assert facts["storage"] == "unchecked"


def test_the_probe_still_only_reads_when_it_checks_storage():
    client = Client(HERE, IMAGE_ONLY, env=R2)
    tp.report(client, "aqa3wkdf8g")
    assert client.calls == [
        "list_templates_graphql",
        "template_env_names_graphql",
        "get_template",
        "rest_template_surface",
    ]


def test_the_required_names_come_from_storage_not_a_copy():
    """A second hand-written list would drift from the one the worker
    actually enforces at runtime."""
    import storage

    assert set(storage.REQUIRED_VARS) == R2


def test_the_verdict_is_repeated_after_the_schema_dump(capsys):
    """The schema dump is two hundred lines. A reader tailing the log has
    to reach the answer, so it is printed again at the very end."""
    tp.report(Client(HERE, IMAGE_ONLY, env=R2 - {"R2_S3_ENDPOINT"}), "aqa3wkdf8g")
    out = capsys.readouterr().out
    head, _, tail = out.partition("=== SUMMARY ===")
    assert "STORAGE NOT READY" in head          # said once where it happens
    assert "STORAGE NOT READY" in tail          # and again where it is read
    assert "TEMPLATE aqa3wkdf8g: present" in tail


def test_the_summary_says_absent_without_saying_it_twice(capsys):
    tp.report(Client([{"id": "other", "name": "s", "imageName": "x"}], IMAGE_ONLY), "aqa3wkdf8g")
    _, _, tail = capsys.readouterr().out.partition("=== SUMMARY ===")
    assert "TEMPLATE aqa3wkdf8g: absent" in tail


def test_the_summary_carries_the_ready_verdict_too(capsys):
    tp.report(Client(HERE, IMAGE_ONLY, env=R2), "aqa3wkdf8g")
    _, _, tail = capsys.readouterr().out.partition("=== SUMMARY ===")
    assert "STORAGE READY" in tail


# ------------------------- one path is an opinion, two are a measurement

def test_the_env_is_read_over_BOTH_apis(capsys):
    client = Client(HERE, IMAGE_ONLY, env=R2)
    tp.report(client, "aqa3wkdf8g")
    out = capsys.readouterr().out
    assert "template_env_names_graphql" in client.calls
    assert "get_template" in client.calls
    assert "ENV via GraphQL" in out
    assert "ENV via REST" in out


def test_the_two_apis_disagreeing_is_itself_the_finding(capsys):
    """If one view says the variables are set and the other says they are
    not, reporting either number would be a guess wearing a measurement's
    clothes. Refuse, and name what each one saw."""
    client = Client(HERE, IMAGE_ONLY, env=set(), rest={"env": {n: "v" for n in R2}})
    code, facts = tp.report(client, "aqa3wkdf8g")
    out = capsys.readouterr().out
    assert "ENV DISAGREEMENT" in out
    assert "R2_S3_ENDPOINT" in out
    assert code == 1
    assert facts["storage"] is None


def test_rest_env_as_a_key_value_list_is_read_too(capsys):
    """RunPod returns env as a mapping in one place and a list of
    {key, value} in another. Both are the same fact."""
    client = Client(HERE, IMAGE_ONLY, env=R2,
                    rest={"env": [{"key": n, "value": "secret"} for n in R2]})
    code, _ = tp.report(client, "aqa3wkdf8g")
    assert code == 0
    assert "STORAGE READY" in capsys.readouterr().out


def test_a_template_with_no_env_key_is_none_set_not_unreadable(capsys):
    client = Client(HERE, IMAGE_ONLY, env=set(), rest={"name": "oniq"})
    code, facts = tp.report(client, "aqa3wkdf8g")
    assert facts["storage"] is False
    assert "STORAGE NOT READY" in capsys.readouterr().out


def test_no_environment_VALUE_is_ever_printed(capsys):
    """The REST template carries values; this probe carries names."""
    client = Client(HERE, IMAGE_ONLY, env=R2)
    tp.report(client, "aqa3wkdf8g")
    assert "value-that-must-never-be-printed" not in capsys.readouterr().out


def test_an_unreadable_rest_falls_back_to_graphql_rather_than_going_blind(capsys):
    class NoRest(Client):
        def get_template(self, template_id):
            raise RuntimeError("HTTP 404")

    tp.report(NoRest(HERE, IMAGE_ONLY, env=R2), "aqa3wkdf8g")
    out = capsys.readouterr().out
    assert "ENV via REST   : unreadable (UNKNOWN)" in out
    assert "STORAGE READY" in out
