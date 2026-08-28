"""The gated-model probe: the right model, or nothing — and never the token.

Owner directive 2026-08-28 option 1: keep the distilled LTX 2B, do not
substitute, do not edit the candidate list to route around authentication,
and STOP if authentication cannot be supplied safely. These tests hold the
probe to all four, plus the rule that a credential is reported by NAME.
"""

import urllib.error

import pytest

from validation import hf_auth

GIB = 1024**3
REPO = "Lightricks/LTX-Video"
REVISION = "8984fa25007f376c1a299016d0957a37a2f797bb"
SECRET = "hf_thisisnotarealtokenvalue"


def _dockerfile():
    with open("Dockerfile", encoding="utf-8") as fh:
        return fh.read()


def _info(repo=REPO, sha="deadbeef" * 5, licence="other", transformer=4 * GIB, extra=None):
    files = {
        "model_index.json": 100,
        "transformer/t.safetensors": transformer,
        "vae/v.safetensors": GIB,
        "text_encoder/e.safetensors": 9 * GIB,
        "tokenizer/tokenizer.json": 10,
        "scheduler/scheduler_config.json": 10,
        "ltx-video-single-file.safetensors": 40 * GIB,
    }
    files.update(extra or {})
    return {
        "id": repo,
        "sha": sha,
        "cardData": {"license": licence},
        "siblings": [{"rfilename": n, "size": s} for n, s in files.items()],
    }


def _ok(info=None):
    def fetch(repo, token, blobs=True):
        assert token, "the probe must present the credential"
        return info or _info()

    return fetch


# ------------------------------------------- the model it actually checks

def test_the_intended_model_is_the_head_of_the_real_candidate_list():
    """The owner's model decision, 2026-08-28 option 1. There is exactly
    one candidate now, so "the first" and "the only" coincide - which is
    the point: nothing to fall past."""
    assert hf_auth.intended(_dockerfile())["repo"] == REPO


def test_the_gates_come_from_the_dockerfile_not_from_here():
    want = hf_auth.intended(_dockerfile())
    assert want["guard_prefix"] == "transformer/"
    assert want["size_guard_bytes"] == 16 * GIB
    assert "text_encoder" in want["components"]


# ------------------------------------------------------------ refusals

def test_a_missing_credential_blocks_and_names_no_substitute():
    with pytest.raises(hf_auth.Blocked) as exc:
        hf_auth.probe(_dockerfile(), token=None, fetch=_ok())
    assert exc.value.code == "credential-absent"


@pytest.mark.parametrize("code", [401, 403])
def test_a_rejected_credential_blocks_rather_than_falling_through(code):
    def refused(repo, token, blobs=True):
        raise urllib.error.HTTPError("u", code, "no", {}, None)

    with pytest.raises(hf_auth.Blocked) as exc:
        hf_auth.probe(_dockerfile(), token=SECRET, fetch=refused)
    assert exc.value.code == "credential-rejected"


def test_a_registry_outage_blocks_too():
    def down(repo, token, blobs=True):
        raise TimeoutError("nope")

    with pytest.raises(hf_auth.Blocked) as exc:
        hf_auth.probe(_dockerfile(), token=SECRET, fetch=down)
    assert exc.value.code == "registry-unreachable"


def test_an_answer_for_a_different_repository_is_an_identity_mismatch():
    """A rename or a redirect must not quietly become a different model."""
    with pytest.raises(hf_auth.Blocked) as exc:
        hf_auth.probe(_dockerfile(), SECRET, _ok(_info(repo="Lightricks/LTX-2.5")))
    assert exc.value.code == "identity-mismatch"


def test_no_commit_sha_means_nothing_to_pin_to():
    info = _info()
    del info["sha"]
    with pytest.raises(hf_auth.Blocked) as exc:
        hf_auth.probe(_dockerfile(), SECRET, _ok(info))
    assert exc.value.code == "revision-unknown"


def test_a_thirteen_b_wearing_the_name_fails_the_shape_guard():
    with pytest.raises(hf_auth.Blocked) as exc:
        hf_auth.probe(_dockerfile(), SECRET, _ok(_info(transformer=26 * GIB)))
    assert exc.value.code == "guard-failed"


def test_a_missing_component_blocks():
    info = _info()
    info["siblings"] = [s for s in info["siblings"] if not s["rfilename"].startswith("vae/")]
    with pytest.raises(hf_auth.Blocked) as exc:
        hf_auth.probe(_dockerfile(), SECRET, _ok(info))
    assert exc.value.code == "incomplete-pipeline"


def test_a_missing_model_index_blocks():
    info = _info()
    info["siblings"] = [s for s in info["siblings"] if s["rfilename"] != "model_index.json"]
    with pytest.raises(hf_auth.Blocked) as exc:
        hf_auth.probe(_dockerfile(), SECRET, _ok(info))
    assert exc.value.code == "not-a-snapshot"


# ------------------------------------------------------------- success

def test_a_reachable_intended_model_returns_its_revision_and_licence():
    found = hf_auth.probe(_dockerfile(), SECRET, _ok())
    assert found["repo"] == REPO
    assert found["revision"] == "deadbeef" * 5
    assert found["licence"] == "other"


def test_the_download_estimate_excludes_the_single_file_checkpoint():
    """The bake fetches pipeline components only; the repo also carries a
    40 GiB single-file weight it never touches."""
    found = hf_auth.probe(_dockerfile(), SECRET, _ok())
    assert found["download_bytes"] == 14 * GIB + 120


# --------------------------------------------------------- the credential

def test_the_token_value_never_reaches_stdout_on_success(capsys):
    hf_auth.report(_dockerfile(), SECRET, _ok())
    out = capsys.readouterr().out
    assert SECRET not in out
    assert "PRESENT" in out
    assert hf_auth.TOKEN_VAR in out


def test_the_token_value_never_reaches_stdout_on_refusal(capsys):
    def refused(repo, token, blobs=True):
        raise urllib.error.HTTPError("u", 401, "no", {}, None)

    code, _ = hf_auth.report(_dockerfile(), SECRET, refused)
    out = capsys.readouterr().out
    assert code == 1
    assert SECRET not in out


def test_an_absent_credential_is_reported_by_name_and_blocks(capsys):
    code, _ = hf_auth.report(_dockerfile(), None, _ok())
    out = capsys.readouterr().out
    assert code == 1
    assert "ABSENT" in out
    assert "NO SUBSTITUTE" in out


def test_the_probe_never_returns_the_token():
    found = hf_auth.probe(_dockerfile(), SECRET, _ok())
    assert SECRET not in repr(found)


def test_the_module_holds_no_literal_credential():
    """Requirement 2: never hardcode the token."""
    import inspect

    source = inspect.getsource(hf_auth)
    assert "hf_" + "s" not in source.lower().replace("hf_auth", "")
    assert hf_auth.TOKEN_VAR == "HF_TOKEN"


# ------------------------------------ the bakes are heredoc'd python

def _bake_blocks():
    import re

    with open("Dockerfile", encoding="utf-8") as fh:
        text = fh.read()
    return re.findall(r"RUN[^\n]*python3 - <<'EOF'\n(.*?)\nEOF", text, re.S)


def test_every_bake_is_syntactically_valid_python():
    """Nothing else ever compiles these. A syntax error in a bake surfaces
    only part-way through a build that downloads tens of GiB first."""
    import ast

    blocks = _bake_blocks()
    assert len(blocks) == 3
    for block in blocks:
        ast.parse(block)


def test_the_ltx_bake_receives_the_credential_and_the_others_do_not():
    """Requirement 5: pass it only where required. The story and voice
    bakes fetch public assets; giving them the token would widen its reach
    for nothing."""
    with open("Dockerfile", encoding="utf-8") as fh:
        lines = [l for l in fh if l.startswith("RUN") and "python3 - <<" in l]
    assert len(lines) == 3
    assert "--mount=type=secret,id=hf_token" in lines[0]
    assert "secret" not in lines[1]
    assert "secret" not in lines[2]


def test_the_image_takes_no_build_argument_and_no_credential_env():
    """Requirements 2, 3 and 7. ARG and ENV both survive into the published
    image where `docker history` and `docker inspect` can read them; a
    secret mount does not."""
    with open("Dockerfile", encoding="utf-8") as fh:
        text = fh.read()
    for line in text.splitlines():
        stripped = line.strip()
        assert not stripped.startswith("ARG "), stripped
        if stripped.startswith("ENV "):
            for word in ("TOKEN", "SECRET", "KEY", "PASSWORD", "HF_"):
                assert word not in stripped.upper(), stripped


def test_the_ltx_bake_has_exactly_one_candidate():
    """Owner directive 2026-08-28 option 1: no substitute exists to fall
    through to. Guarding a fall-through is weaker than not having one."""
    from validation import image_size

    with open("Dockerfile", encoding="utf-8") as fh:
        bakes = image_size.parse_bakes(fh.read())
    assert bakes[0]["candidates"] == ["Lightricks/LTX-Video"]


def test_the_ltx_bake_pins_the_exact_revision_the_owner_named():
    """A repository name names a moving branch; a sha names bytes - and it
    fixes the LICENCE TERMS too, because terms at a commit cannot change."""
    block = _bake_blocks()[0]
    assert f'PINNED_REVISION = "{REVISION}"' in block


def test_the_ltx_bake_now_has_a_licence_gate_of_its_own():
    """It never did. LTX is not Apache, so the Qwen gate did not cover it,
    and the owner's acceptance needed somewhere to be recorded."""
    block = _bake_blocks()[0]
    assert "ALLOWED_LICENCES" in block
    assert '"other"' in block.split("ALLOWED_LICENCES", 1)[1][:60]
    assert "refusing" in block


def test_the_ltx_bake_ships_the_licence_text_with_the_weights():
    """Redistributing someone's model without their terms attached is the
    compliance failure this catches."""
    block = _bake_blocks()[0]
    assert '"LICENSE*"' in block
    assert '"NOTICE*"' in block
    assert "refusing to redistribute the weights without their terms" in block


def test_the_ltx_bake_pins_the_revision_it_surveyed():
    """A repository name names a moving branch; a sha names bytes."""
    block = _bake_blocks()[0]
    assert "revision=revision" in block
    assert "PINNED_REVISION" in block


def test_the_ltx_bake_records_revision_and_licence_into_the_image():
    block = _bake_blocks()[0]
    assert "/app/models/LTX_REVISION" in block
    assert "/app/models/LTX_LICENCE" in block


def test_the_bake_never_puts_the_token_in_the_environment():
    """huggingface_hub reads HF_TOKEN implicitly. An implicit credential is
    one that can travel somewhere unnoticed, so it is passed explicitly."""
    block = _bake_blocks()[0]
    assert "os.environ[" not in block
    assert "token=TOKEN" in block


def test_the_bake_clears_every_token_cache_home():
    """This stage runs as root while HOME is /home/oniq, so a cache written
    to either would otherwise ride into the published layer."""
    block = _bake_blocks()[0]
    assert "/root/.cache/huggingface" in block
    assert "/home/oniq/.cache/huggingface" in block


def test_the_story_licence_gate_is_still_mandatory():
    """Requirement 10. The Apache gate on the story model is untouched by
    the LTX work."""
    block = _bake_blocks()[1]
    assert "ALLOWED_LICENCES" in block
    assert "apache-2.0" in block
    assert "refusing to bake" in block


def test_the_size_guards_both_survive():
    """Requirement 13."""
    from validation import image_size

    with open("Dockerfile", encoding="utf-8") as fh:
        bakes = image_size.parse_bakes(fh.read())
    assert bakes[0]["size_guard_bytes"] == 16 * 1024**3
    assert bakes[1]["size_guard_bytes"] == 20 * 1024**3
