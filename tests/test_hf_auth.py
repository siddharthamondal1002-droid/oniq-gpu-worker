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
# Moved 2026-08-31 with the owner's checkpoint directive: the old
# checkpoint's vae is a different network from the pinned spatial
# upsampler's, so multi-scale could not be enabled against it.
REPO = "Lightricks/LTX-Video-0.9.7-distilled"
REVISION = "057509edea1493cae5e62e9d8f780ebda3fb4333"
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
    assert want["size_guard_bytes"] == 32 * GIB
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


def test_a_transformer_over_the_guard_fails_the_shape_guard():
    """The guard is RAISED, not removed, and this is what proves it.

    It used to read "a thirteen B wearing the name", with 26 GiB against a
    16 GiB guard. A 13B is now what ONIQ deliberately bakes (owner directive
    2026-08-31), so the size moved to one the 32 GiB guard still refuses —
    70.75 GiB, which is what run 33432424021 actually measured for
    Lightricks/LTX-2-Pre-Trained. A guard that admits everything is not a
    guard, so the refusal is pinned against a real repository rather than an
    invented number."""
    with pytest.raises(hf_auth.Blocked) as exc:
        hf_auth.probe(_dockerfile(), SECRET,
                      _ok(_info(transformer=int(70.75 * GIB))))
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


def _bakes_by_role():
    """The bakes keyed by WHAT THEY BAKE, never by position.

    These assertions used to index the list — [0] LTX, [1] story, [2] voice
    — and splitting the 16.05 GiB LTX layer into text-encoder passes on
    2026-08-30 silently re-pointed every one of them at a different block.
    The licence-gate test then read a pass and failed; had it read a block
    that happened to contain the string, it would have PASSED while
    checking nothing. A name cannot slide the way an index can.
    """
    roles = {}
    for block in _bake_blocks():
        if "LTX_TX_PASS" in block:
            # The TRANSFORMER passes, added 2026-08-31 when the checkpoint
            # moved to 0.9.7-distilled: at 24.29 GiB it can no longer ride in
            # the first layer. Named BEFORE the /app/models/ltx test below for
            # the reason this function's docstring gives — these write into
            # that same directory, so an unnamed pass would be filed as the
            # transformer bake, silently replacing it and taking the count.
            roles.setdefault("transformer_passes", []).append(block)
        elif "LTX_PASS" in block:
            roles.setdefault("ltx_passes", []).append(block)
        elif "ltx-upscaler.pin" in block:
            # BEFORE the /app/models/ltx test below, because the upscaler
            # lands INSIDE that directory and would otherwise be filed as the
            # transformer bake — silently replacing it, and taking the count
            # with it. Named first, for exactly the reason this function's
            # docstring gives: a name cannot slide the way an index can.
            roles["upscaler"] = block
        elif "/app/models/ltx" in block:
            roles["ltx"] = block
        elif "/app/models/piper" in block:
            roles["voice"] = block
        else:
            roles["story"] = block
    missing = {"ltx", "story", "voice"} - set(roles)
    assert not missing, f"Dockerfile no longer carries bake(s): {sorted(missing)}"
    return roles


def test_every_bake_is_syntactically_valid_python():
    """Nothing else ever compiles these. A syntax error in a bake surfaces
    only part-way through a build that downloads tens of GiB first."""
    import ast

    blocks = _bake_blocks()
    roles = _bakes_by_role()
    # Three named bakes, the optional upscaler, plus however many transformer
    # and text-encoder passes the layer split uses; every one compiled, none
    # skipped. The arithmetic is spelled out rather than hardcoded so that
    # adding a pass cannot quietly drop a block from this check.
    assert len(blocks) == (
        3
        + len(roles.get("transformer_passes", []))
        + len(roles.get("ltx_passes", []))
        + (1 if "upscaler" in roles else 0)
    )
    for block in blocks:
        ast.parse(block)


def test_the_ltx_bake_receives_the_credential_and_the_others_do_not():
    """Requirement 5: pass it only where required. The story and voice
    bakes fetch public assets; giving them the token would widen its reach
    for nothing."""
    with open("Dockerfile", encoding="utf-8") as fh:
        text = fh.read()
    import re

    # Pair each bake's RUN line with the block it opens, so the credential
    # is checked against WHAT IS BAKED rather than against a line number.
    pairs = re.findall(r"(RUN[^\n]*python3 - <<'EOF')\n(.*?)\nEOF", text, re.S)
    assert len(pairs) >= 3
    for run_line, block in pairs:
        needs_token = "LTX_PASS" in block or "/app/models/ltx" in block
        has_token = "--mount=type=secret,id=hf_token" in run_line
        assert has_token == needs_token, (
            f"credential reach is wrong for: {run_line}"
        )
    # The LTX family alone: the story and voice assets are public, and
    # widening the token's reach buys nothing.
    assert sum("secret" in run for run, _ in pairs) == sum(
        1 for _, b in pairs if "LTX_PASS" in b or "/app/models/ltx" in b
    )


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
    assert bakes[0]["candidates"] == ["Lightricks/LTX-Video-0.9.7-distilled"]


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
    block = _bakes_by_role()["ltx"]
    assert "/root/.cache/huggingface" in block
    assert "/home/oniq/.cache/huggingface" in block


def test_the_story_licence_gate_is_still_mandatory():
    """Requirement 10. The Apache gate on the story model is untouched by
    the LTX work."""
    block = _bakes_by_role()["story"]
    assert "ALLOWED_LICENCES" in block
    assert "apache-2.0" in block
    assert "refusing to bake" in block


def test_the_size_guards_both_survive():
    """Requirement 13."""
    from validation import image_size

    with open("Dockerfile", encoding="utf-8") as fh:
        bakes = image_size.parse_bakes(fh.read())
    assert bakes[0]["size_guard_bytes"] == 32 * 1024**3
    assert bakes[1]["size_guard_bytes"] == 20 * 1024**3


def test_the_transformer_passes_cover_every_file_exactly_once():
    """The split must be a PARTITION. A file assigned to two passes is wasted
    bandwidth; a file assigned to none is a broken pipeline that only shows up
    when a rented card tries to load it.

    This runs the Dockerfile's own grouping arithmetic — lifted from the bake,
    not reimplemented — over the shard layouts a repository can plausibly
    have.

    RETARGETED 2026-08-31 from the text-encoder passes to the transformer
    ones. The text encoder left the image entirely (owner directive: a hosted
    runner cannot build a 57.97 GiB image, and it is 17.74 GiB of that), and
    the transformer took its place as the component too large for one layer.
    Same algorithm, same partition property, same reason — so the test moved
    rather than being deleted with the code it was written for.
    """
    passes = _bakes_by_role()["transformer_passes"]
    assert len(passes) == 3, "the split is written as three passes"

    def group_of(running, size, total, n_passes):
        return min(int((running + size / 2) * n_passes / total) if total else 0,
                   n_passes - 1)

    # The arithmetic above must be the arithmetic in the image.
    for block in passes:
        assert "(running + size / 2) * PASSES / total" in block

    GB = 1024 ** 3
    layouts = {
        "four equal shards": [(f"transformer/m-{i}.safetensors", 2 * GB)
                              for i in range(4)],
        "one unsharded file": [("transformer/model.safetensors", 9 * GB)],
        "uneven shards + config": [("transformer/config.json", 1000),
                                   ("transformer/m-1.safetensors", 4 * GB),
                                   ("transformer/m-2.safetensors", 4 * GB),
                                   ("transformer/m-3.safetensors", 1 * GB)],
    }
    for label, files in layouts.items():
        files = sorted(files)
        total = sum(size for _, size in files)
        assigned, running = {}, 0
        for name, size in files:
            assigned.setdefault(group_of(running, size, total, 3), []).append(name)
            running += size
        flat = [n for names in assigned.values() for n in names]
        assert sorted(flat) == sorted(n for n, _ in files), label
        assert len(flat) == len(set(flat)), f"{label}: a file is in two passes"


def test_the_last_transformer_pass_refuses_an_incomplete_component():
    """A split download that lands part of a transformer must fail the BUILD,
    not a paid job. Only the final pass can know the component is whole.

    RETARGETED 2026-08-31 with the test above: the text encoder left the image
    and the transformer took its place as the split component. The check is
    guarded by `PASS == PASSES - 1` rather than living in a distinct last
    block, because the three passes share one body — so this asserts the GUARD
    exists, which is the thing that makes only the final pass verify."""
    passes = _bakes_by_role()["transformer_passes"]
    for block in passes:
        assert "transformer incomplete after all passes" in block
        assert "TRANSFORMER COMPLETE" in block
        # The guard is what stops pass 0 declaring a two-thirds download whole.
        assert "if PASS == PASSES - 1:" in block


def test_a_bake_block_only_verifies_what_it_downloads():
    """MEASURED THE EXPENSIVE WAY, run 33464498069: the first LTX block still
    weighed DEST/transformer on disk after the transformer had moved into its
    own split passes, so the build refused itself with "downloaded transformer
    is 0 bytes".

    A block that verifies a component it does not fetch fails for a reason
    that has nothing to do with the component. This checks the invariant
    directly: every component the first block walks on disk must be one
    FIRST_PASS actually downloads."""
    import re

    from validation.image_size import _first_strings

    block = _bakes_by_role()["ltx"]
    first_pass = set(_first_strings(block, "FIRST_PASS", "(", ")"))
    assert first_pass, "FIRST_PASS is not parseable"

    walked = set(re.findall(r'os\.path\.join\(DEST,\s*"([a-z_]+)"\)', block))
    # A loop over FIRST_PASS itself is fine — it cannot name a component the
    # pass does not fetch.
    stray = walked - first_pass
    assert not stray, (
        f"the first LTX bake verifies {sorted(stray)} on disk but FIRST_PASS "
        f"only downloads {sorted(first_pass)}"
    )


def test_the_on_disk_size_guard_lives_where_the_weights_land():
    """The metadata guard in survey() refuses an oversized model before a byte
    moves. The ON-DISK guard is the second half of that pair — a registry that
    under-reported a size could otherwise smuggle a bigger model past both —
    and it has to sit in the pass that actually writes the weights."""
    passes = _bakes_by_role()["transformer_passes"]
    for block in passes:
        assert "SIZE_GUARD_BYTES" in block
        assert "outside the" in block
    # And it is guarded to the final pass, like the completeness check: an
    # earlier pass holds only part of the weights and would refuse a model
    # that is fine.
    for block in passes:
        guarded = block.split("if PASS == PASSES - 1:", 1)
        assert len(guarded) == 2
        assert "SIZE_GUARD_BYTES" in guarded[1]
