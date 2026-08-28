"""The image-size projection: parsed from the Dockerfile, never restated.

The projection exists to answer one question before an hour of build time
is spent — does the media image fit on a runner. Three ways that answer
goes wrong, each with tests here: a parse that silently drops part of a
bake, a candidate sized that the build would REJECT, and an unmeasured
addend quietly counted as zero.
"""

import urllib.error

import pytest

from validation import image_size as isz


DOCKERFILE = """
FROM python:3.11-slim AS base
FROM base AS media
RUN python3 - <<'EOF'
CANDIDATES = [
    ("Vendor/ltx-2b", "#distilled"),
    ("Vendor/ltx-13b", "#distilled"),
    ("Vendor/ltx", ""),
]
DEST = "/app/models/ltx"
SIZE_GUARD_BYTES = 16 * 1024**3
COMPONENTS = ("transformer", "vae", "text_encoder", "tokenizer", "scheduler")
if "model_index.json" not in paths:
    raise RuntimeError("no model_index.json")
transformer_bytes = sum(
    size for p, size in paths.items()
    if p.startswith("transformer/") and p.endswith(".safetensors")
)
snapshot_download(
    repo,
    local_dir=DEST,
    allow_patterns=["model_index.json"] + [c + "/*" for c in COMPONENTS],
)
EOF
RUN python3 - <<'EOF'
CANDIDATES = [
    "Vendor/story-awq",
    "Vendor/story",
]
DEST = "/app/models/story"
SIZE_GUARD_BYTES = 20 * 1024**3
NEEDED = ("config.json", "tokenizer_config.json")
snapshot_download(
    repo,
    local_dir=DEST,
    allow_patterns=["*.json", "*.safetensors", "*.txt"],
)
EOF
RUN python3 - <<'EOF'
URL = ("https://example.invalid/voices/"
       "voice.tar.gz")
EOF
"""

GIB = 1024**3
LTX = 0
STORY = 1


def _repo(files):
    return {"siblings": [{"rfilename": n, "size": s} for n, s in files.items()]}


def _ltx(transformer_gib, extra=None):
    files = {
        "model_index.json": 100,
        "transformer/t.safetensors": int(transformer_gib * GIB),
        "vae/v.safetensors": GIB,
        "text_encoder/e.safetensors": 9 * GIB,
        "tokenizer/t.json": 10,
        "scheduler/s.json": 10,
        # The single-file checkpoint the bake deliberately never fetches.
        "ltx-video-full.safetensors": 40 * GIB,
    }
    files.update(extra or {})
    return _repo(files)


# ----------------------------------------------------------- parsing

def test_candidates_read_the_repo_not_the_tag():
    bakes = isz.parse_bakes(DOCKERFILE)
    assert bakes[LTX]["candidates"] == ["Vendor/ltx-2b", "Vendor/ltx-13b", "Vendor/ltx"]
    assert "#distilled" not in bakes[LTX]["candidates"]


def test_a_single_line_component_tuple_yields_every_component():
    """The first regression. COMPONENTS sits on ONE line; reading
    first-per-line returned only 'transformer' and dropped the VAE and
    text encoder."""
    assert isz.parse_bakes(DOCKERFILE)[LTX]["patterns"] == [
        "model_index.json",
        "transformer/*",
        "vae/*",
        "text_encoder/*",
        "tokenizer/*",
        "scheduler/*",
    ]


def test_a_plain_pattern_list_is_read_verbatim():
    assert isz.parse_bakes(DOCKERFILE)[STORY]["patterns"] == [
        "*.json", "*.safetensors", "*.txt",
    ]


def test_the_guard_scope_is_read_per_bake():
    """LTX guards the TRANSFORMER only — that is what separates a 2B from a
    13B — while the story bake guards every weight file."""
    bakes = isz.parse_bakes(DOCKERFILE)
    assert bakes[LTX]["guard_prefix"] == "transformer/"
    assert bakes[STORY]["guard_prefix"] == ""


def test_required_files_are_read_from_both_forms():
    bakes = isz.parse_bakes(DOCKERFILE)
    assert bakes[LTX]["needed_files"] == ["model_index.json"]
    assert bakes[STORY]["needed_files"] == ["config.json", "tokenizer_config.json"]


def test_a_split_url_literal_is_rejoined():
    assert isz.parse_piper(DOCKERFILE) == "https://example.invalid/voices/voice.tar.gz"


def test_a_dockerfile_missing_a_bake_raises_rather_than_guessing():
    with pytest.raises(isz.SizeParseError):
        isz.parse_bakes("FROM scratch\n")


# ----------------------------------------------------------- surveying

def test_only_files_matching_the_patterns_are_counted():
    bake = isz.parse_bakes(DOCKERFILE)[LTX]
    out = isz.survey("r", bake, lambda _: _ltx(4))
    assert out["verdict"] == "BAKE"
    # 4 + 1 + 9 GiB of components, and NOT the 40 GiB single-file weights.
    assert out["bytes"] == 14 * GIB + 120


def test_a_candidate_over_its_guard_is_skipped_and_never_sized():
    """The second regression, measured in run 50: the 13B repository was
    sized at 44 GiB and the image declared not to fit, when the build
    would have refused that candidate outright."""
    bake = isz.parse_bakes(DOCKERFILE)[LTX]
    out = isz.survey("r", bake, lambda _: _ltx(26))
    assert out["verdict"] == "SKIP"
    assert out["bytes"] is None
    assert "guard" in out["detail"]


def test_the_guard_weighs_only_the_prefix_it_names():
    """A 9 GiB text encoder must not push a 4 GiB transformer over a 16 GiB
    transformer guard."""
    bake = isz.parse_bakes(DOCKERFILE)[LTX]
    assert isz.survey("r", bake, lambda _: _ltx(15))["verdict"] == "BAKE"


def test_a_candidate_missing_a_required_file_is_skipped():
    bake = isz.parse_bakes(DOCKERFILE)[STORY]
    info = _repo({"config.json": 1, "w.safetensors": GIB})
    out = isz.survey("r", bake, lambda _: info)
    assert out["verdict"] == "SKIP"
    assert "tokenizer_config.json" in out["detail"]


def test_a_candidate_missing_a_component_is_skipped():
    bake = isz.parse_bakes(DOCKERFILE)[LTX]
    info = _ltx(4)
    info["siblings"] = [s for s in info["siblings"] if not s["rfilename"].startswith("vae/")]
    out = isz.survey("r", bake, lambda _: info)
    assert out["verdict"] == "SKIP"
    assert "vae" in out["detail"]


def test_an_unreachable_repo_reports_the_http_code_not_the_class_name():
    """'HTTPError' does not distinguish 404 (gone) from 401 (gated) from
    429 (too fast), and those lead to different actions. Run 50 reported a
    bare HTTPError for the owner's chosen model and the reason had to be
    guessed at."""
    def gone(_):
        raise urllib.error.HTTPError("u", 404, "Not Found", {}, None)

    bake = isz.parse_bakes(DOCKERFILE)[LTX]
    out = isz.survey("r", bake, gone)
    assert out["verdict"] == "UNREACHABLE"
    assert out["detail"] == "HTTP 404"
    assert out["bytes"] is None


def test_a_non_http_failure_still_reports_something_specific():
    def boom(_):
        raise TimeoutError("nope")

    bake = isz.parse_bakes(DOCKERFILE)[LTX]
    assert isz.survey("r", bake, boom)["detail"] == "TimeoutError"


def test_an_unreachable_first_candidate_falls_through_to_a_passing_one(capsys):
    calls = []

    def fetch(repo):
        calls.append(repo)
        if repo == "Vendor/ltx-2b":
            raise urllib.error.HTTPError("u", 429, "Too Many", {}, None)
        if repo == "Vendor/ltx-13b":
            return _ltx(26)
        return _ltx(4)

    code, rows = isz.report(DOCKERFILE, base_image_bytes=GIB, fetch=fetch, head=lambda u: 1)
    out = capsys.readouterr().out
    assert "HTTP 429" in out
    # The 13B is skipped by the guard, so the third candidate is chosen.
    assert rows[LTX]["repo"] == "Vendor/ltx"


# ----------------------------------------------------------- arithmetic

def test_an_unmeasured_base_stage_refuses_to_produce_a_total(capsys):
    code, _ = isz.report(
        DOCKERFILE, base_image_bytes=None,
        fetch=lambda r: _ltx(1, {"config.json": 1, "tokenizer_config.json": 1}),
        head=lambda u: 60 * 1024**2,
    )
    out = capsys.readouterr().out
    assert code == 2
    assert "PROJECTED MEDIA IMAGE: NOT MEASURED" in out
    assert "FITS" not in out


def test_a_bake_with_no_viable_candidate_refuses_to_produce_a_total(capsys):
    def gone(_):
        raise urllib.error.HTTPError("u", 404, "Not Found", {}, None)

    code, _ = isz.report(DOCKERFILE, base_image_bytes=GIB, fetch=gone, head=lambda u: 1)
    assert code == 2
    assert "NOT MEASURED" in capsys.readouterr().out


def _both(transformer_gib, weight_gib):
    def fetch(repo):
        if "story" in repo:
            return _repo({
                "config.json": 1, "tokenizer_config.json": 1,
                "w.safetensors": int(weight_gib * GIB),
            })
        return _ltx(transformer_gib)

    return fetch


def test_a_small_image_fits_as_is(capsys):
    def tiny(repo):
        if "story" in repo:
            return _repo({"config.json": 1, "tokenizer_config.json": 1,
                          "w.safetensors": GIB})
        return _repo({
            "model_index.json": 1, "transformer/t.safetensors": GIB,
            "vae/v.safetensors": 1, "text_encoder/e.safetensors": 1,
            "tokenizer/t.json": 1, "scheduler/s.json": 1,
        })

    code, _ = isz.report(DOCKERFILE, base_image_bytes=GIB, fetch=tiny, head=lambda u: 1)
    assert code == 0
    assert "VERDICT: FITS as-is" in capsys.readouterr().out


def test_a_large_image_needs_the_mount(capsys):
    code, _ = isz.report(DOCKERFILE, base_image_bytes=8 * GIB, fetch=_both(4, 6), head=lambda u: 1)
    assert code == 0
    assert "FITS ONLY ON /mnt" in capsys.readouterr().out


def test_an_image_too_big_for_the_mount_is_refused(capsys):
    code, _ = isz.report(DOCKERFILE, base_image_bytes=8 * GIB, fetch=_both(15, 19), head=lambda u: 1)
    assert code == 1
    assert "DOES NOT FIT" in capsys.readouterr().out


def test_the_budget_accounts_for_the_build_holding_two_copies():
    """One times the floor fitting is not the question: docker keeps the
    layer cache and the assembled image at the same time."""
    assert isz.RUNNER_MNT_FREE_BYTES > isz.RUNNER_ROOT_FREE_BYTES


# ------------------------------------- the substitution the build refuses

def _refuses(code):
    def fetch(_):
        raise urllib.error.HTTPError("u", code, "Unauthorized", {}, None)

    return fetch


@pytest.mark.parametrize("code", [401, 403])
def test_an_auth_refusal_stops_rather_than_falling_through(code, capsys):
    """Measured in run 51: the owner's chosen LTX candidate answers HTTP
    401 — gated, not gone. Falling through turns an INFRASTRUCTURE failure
    into a MODEL SUBSTITUTION, shipping a checkpoint nobody chose under
    the same image name."""
    bake = isz.parse_bakes(DOCKERFILE)[LTX]
    out = isz.survey("r", bake, _refuses(code))
    assert out["verdict"] == "AUTH-REFUSED"
    assert out["bytes"] is None


def test_a_blocked_bake_never_reaches_a_later_candidate():
    seen = []

    def fetch(repo):
        seen.append(repo)
        if repo == "Vendor/ltx-2b":
            raise urllib.error.HTTPError("u", 401, "Unauthorized", {}, None)
        return _ltx(4)

    isz.report(DOCKERFILE, base_image_bytes=GIB, fetch=fetch, head=lambda u: 1)
    assert "Vendor/ltx-13b" not in seen
    assert "Vendor/ltx" not in seen


def test_a_blocked_bake_refuses_to_produce_a_total(capsys):
    def fetch(repo):
        if repo == "Vendor/ltx-2b":
            raise urllib.error.HTTPError("u", 401, "Unauthorized", {}, None)
        return _ltx(4)

    code, _ = isz.report(DOCKERFILE, base_image_bytes=GIB, fetch=fetch, head=lambda u: 1)
    out = capsys.readouterr().out
    assert code == 2
    assert "BLOCKED" in out
    assert "PROJECTED MEDIA IMAGE: NOT MEASURED" in out
    assert "VERDICT" not in out


def test_a_404_still_falls_through_because_that_is_a_judgement(capsys):
    """A repository that does not exist is a fact about the model. Only a
    refusal to authenticate is the ambiguous case."""
    def fetch(repo):
        if repo == "Vendor/ltx-2b":
            raise urllib.error.HTTPError("u", 404, "Not Found", {}, None)
        if repo == "Vendor/ltx-13b":
            return _ltx(26)
        return _ltx(4)

    code, rows = isz.report(DOCKERFILE, base_image_bytes=GIB, fetch=fetch, head=lambda u: 1)
    assert rows[LTX]["repo"] == "Vendor/ltx"


def test_the_dockerfile_bakes_refuse_an_auth_error_in_both_blocks():
    """The projection and the build must agree, so the guard is asserted
    in the Dockerfile itself rather than only in this module."""
    with open("Dockerfile", encoding="utf-8") as fh:
        text = fh.read()
    assert text.count("def auth_refused(exc):") == 2
    assert text.count("AUTH REFUSED for") == 2
    assert text.count("status in (401, 403)") == 2
