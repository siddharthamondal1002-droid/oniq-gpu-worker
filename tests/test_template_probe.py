"""The template probe: read-only, and it must not call a create no one asked for.

The endpoint's templateId is a dangling reference. The tempting fix is to POST
a new template - but a template names an image, and an image that does not
exist anywhere cannot be named. These tests hold the probe to reporting that
rather than acting on it.
"""

from validation import template_probe as tp


class Client:
    """Records every call, so a mutation shows up as a failed assertion."""

    def __init__(self, templates, surface):
        self._templates = templates
        self._surface = surface
        self.calls = []

    def list_templates_graphql(self):
        self.calls.append("list_templates_graphql")
        return self._templates

    def rest_template_surface(self):
        self.calls.append("rest_template_surface")
        return self._surface

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
