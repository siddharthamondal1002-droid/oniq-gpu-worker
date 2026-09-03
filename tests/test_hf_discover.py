"""The catalogue probe: list what exists, measure it, choose nothing.

Run 7 proved the Dockerfile names a repository that does not exist, which
makes "find a replacement" the obvious next move and exactly the move the
owner's directive forbids an agent to make alone. These tests hold the
probe to listing and measuring only.
"""

import urllib.error

import pytest

from validation import hf_discover

GIB = 1024**3
SECRET = "hf_notarealtokenvalue"


def _dockerfile():
    with open("Dockerfile", encoding="utf-8") as fh:
        return fh.read()


def _model(transformer=4 * GIB, components=("transformer", "vae", "text_encoder",
                                            "tokenizer", "scheduler"),
           index=True, licence="other", sha="abc1234"):
    files = {}
    if index:
        files["model_index.json"] = 100
    for c in components:
        files[f"{c}/w.safetensors" if c == "transformer" else f"{c}/c.json"] = (
            transformer if c == "transformer" else 1024
        )
    files["single-file.safetensors"] = 40 * GIB
    return {
        "sha": sha,
        "cardData": {"license": licence},
        "siblings": [{"rfilename": n, "size": s} for n, s in files.items()],
    }


def _registry(catalogue_ids, models):
    def get(url, token, timeout=60):
        assert token, "the probe must present the credential"
        # Both URLs share the /api/models prefix — the listing is
        # /api/models?..., a single model is /api/models/<repo>?... — so
        # route on which follows, not on startswith.
        if "/api/models?" in url:
            return [{"id": i} for i in catalogue_ids]
        repo = url.split("/api/models/", 1)[1].split("?", 1)[0]
        value = models[repo]
        if isinstance(value, Exception):
            raise value
        return value

    return get


def test_no_credential_refuses_because_anonymous_cannot_tell_gated_from_gone(capsys):
    code, rows = hf_discover.report(_dockerfile(), None)
    out = capsys.readouterr().out
    assert code == 2
    assert rows == []
    assert "gated" in out and "does not exist" in out


def test_a_named_model_absent_from_the_catalogue_is_reported_as_not_existing(capsys):
    """The state this module was written for: the Dockerfile naming
    something the publisher does not publish."""
    other = "Lightricks/LTX-2.5"
    get = _registry([other], {other: _model()})
    hf_discover.report(_dockerfile(), SECRET, get)
    out = capsys.readouterr().out
    assert "does not exist" in out
    assert "Lightricks/LTX-Video-0.9.7-distilled" in out


def test_the_named_model_being_present_says_the_404_was_something_else(capsys):
    named = "Lightricks/LTX-Video-0.9.7-distilled"
    get = _registry([named], {named: _model()})
    hf_discover.report(_dockerfile(), SECRET, get)
    assert "the 404 was something else" in capsys.readouterr().out


def test_an_over_guard_candidate_is_reported_not_hidden(capsys):
    get = _registry(
        ["Lightricks/LTX-Video-0.9.7-distilled"],
        # OVER the raised 32 GiB guard. 24 GiB used to be over the old
        # 16 GiB one; the guard moved with the 0.9.7-distilled repoint, and
        # the property under test is the REPORTING of an over-guard
        # candidate, not any particular size.
        {"Lightricks/LTX-Video-0.9.7-distilled": _model(transformer=40 * GIB)},
    )
    code, rows = hf_discover.report(_dockerfile(), SECRET, get)
    out = capsys.readouterr().out
    assert rows[0]["verdict"] == "OVER-GUARD"
    assert "40.00 GiB transformer" in out
    assert code == 1


def test_a_non_diffusers_repo_is_reported(capsys):
    get = _registry(["Lightricks/other"], {"Lightricks/other": _model(index=False)})
    _, rows = hf_discover.report(_dockerfile(), SECRET, get)
    assert rows[0]["verdict"] == "NOT-A-PIPELINE"


def test_an_incomplete_pipeline_is_reported(capsys):
    get = _registry(
        ["Lightricks/partial"],
        {"Lightricks/partial": _model(components=("transformer", "vae"))},
    )
    _, rows = hf_discover.report(_dockerfile(), SECRET, get)
    assert rows[0]["verdict"] == "INCOMPLETE"
    assert "text_encoder" in rows[0]["missing_components"]


def test_an_unreadable_repo_reports_its_code(capsys):
    get = _registry(
        ["Lightricks/gone"],
        {"Lightricks/gone": urllib.error.HTTPError("u", 404, "no", {}, None)},
    )
    _, rows = hf_discover.report(_dockerfile(), SECRET, get)
    assert rows[0]["verdict"] == "UNREADABLE"
    assert rows[0]["detail"] == "HTTP 404"


def test_eligible_candidates_are_listed_with_revision_and_licence(capsys):
    get = _registry(
        ["Lightricks/a", "Lightricks/b"],
        {"Lightricks/a": _model(sha="aaa1111", licence="apache-2.0"),
         "Lightricks/b": _model(transformer=40 * GIB)},
    )
    code, _ = hf_discover.report(_dockerfile(), SECRET, get)
    out = capsys.readouterr().out
    assert code == 0
    assert "ELIGIBLE (1)" in out
    assert "aaa1111" in out
    assert "apache-2.0" in out


def test_it_refuses_to_choose(capsys):
    """The whole point. A name picked here to make a build succeed is the
    substitution owner directive 2026-08-28 forbids."""
    get = _registry(
        ["Lightricks/a", "Lightricks/b"],
        {"Lightricks/a": _model(), "Lightricks/b": _model()},
    )
    hf_discover.report(_dockerfile(), SECRET, get)
    out = capsys.readouterr().out
    assert "NOT CHOOSING ONE" in out


def test_it_never_writes_the_dockerfile():
    """A probe that edited the candidate list would be making the decision
    by another route."""
    import inspect

    source = inspect.getsource(hf_discover)
    assert "open(DOCKERFILE" not in source.replace('with open(DOCKERFILE, encoding="utf-8") as fh:', "")
    for verb in ('"w"', "'w'", "write(", "replace("):
        assert verb not in source, verb


def test_the_token_never_reaches_stdout(capsys):
    get = _registry(["Lightricks/a"], {"Lightricks/a": _model()})
    hf_discover.report(_dockerfile(), SECRET, get)
    assert SECRET not in capsys.readouterr().out


def test_the_gates_are_the_dockerfiles_own():
    from validation import image_size

    bake = image_size.parse_bakes(_dockerfile())[0]
    assert bake["guard_prefix"] == "transformer/"
    assert bake["size_guard_bytes"] == 32 * GIB
