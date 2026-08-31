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
SIZE_GUARD_BYTES = 32 * 1024**3
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
    """The second regression, measured in run 50: an over-guard repository was
    sized at 44 GiB and the image declared not to fit, when the build
    would have refused that candidate outright.

    The size here is 40 GiB rather than the original 26: a 13B is what ONIQ
    now deliberately bakes, so the guard moved 16 -> 32 GiB with it. What is
    under test is that an over-guard candidate is SKIPPED rather than sized,
    whatever the guard happens to be."""
    bake = isz.parse_bakes(DOCKERFILE)[LTX]
    out = isz.survey("r", bake, lambda _: _ltx(40))
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
            # Over the raised 32 GiB guard, so the fall-through still has
            # something to fall through past. 26 GiB was over the OLD guard;
            # a 13B is now what ONIQ bakes on purpose.
            return _ltx(40)
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


def test_a_small_image_fits_with_room_for_both_copies(capsys):
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
    assert "VERDICT: FITS with room for both copies" in capsys.readouterr().out


def test_an_image_that_fits_once_but_not_twice_is_reported_tight(capsys):
    """A build holds the layer cache and the assembled image at the same
    time, so fitting once is not the question."""
    code, _ = isz.report(DOCKERFILE, base_image_bytes=8 * GIB, fetch=_both(4, 6), head=lambda u: 1)
    assert code == 0
    assert "VERDICT: TIGHT" in capsys.readouterr().out


def test_an_image_too_big_for_the_mount_is_refused(capsys):
    code, _ = isz.report(DOCKERFILE, base_image_bytes=8 * GIB, fetch=_both(15, 19), head=lambda u: 1)
    assert code == 1
    assert "DOES NOT FIT" in capsys.readouterr().out


def test_the_runner_budget_is_a_measurement_plus_a_named_estimate():
    """Measured 2026-08-28: one filesystem, 13.76 GiB free, no separate
    /mnt. An earlier version planned against a 65 GiB /mnt that does not
    exist on these runners."""
    assert isz.RUNNER_FREE_BYTES == 14773895168
    assert isz.RUNNER_TOTAL_BYTES == 76887154688
    assert isz.RUNNER_USABLE_BYTES == isz.RUNNER_FREE_BYTES + isz.RECLAIMABLE_BYTES
    assert not hasattr(isz, "RUNNER_MNT_FREE_BYTES")


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
            # Over the raised 32 GiB guard, so the fall-through still has
            # something to fall through past. 26 GiB was over the OLD guard;
            # a 13B is now what ONIQ bakes on purpose.
            return _ltx(40)
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


def test_components_moved_into_split_passes_are_still_counted():
    """MEASURED 2026-08-31, run 33434036705: the projection reported the LTX
    bake at 2.32 GiB and printed "VERDICT: FITS" for a pipeline the registry
    measures at 44.36 GiB.

    `_block` isolates ONE block per DEST, so any component moved out of the
    main bake into a split layer stopped being counted. The text encoder had
    been invisible this way since the 2026-08-30 split; moving the 24.29 GiB
    transformer out on 2026-08-31 made the omission larger than the number
    being reported.

    This is not a harmless inaccuracy. This projection is what decides whether
    to start a build that takes a hosted runner the best part of an hour, and
    a FITS read off a number missing 42 GiB is exactly the wasted forty
    minutes it exists to prevent."""
    bake = isz.parse_bakes(DOCKERFILE)[LTX]
    counted = {p for p in bake["patterns"] if p.endswith("/*")}
    # Every component the bake declares must be reachable by some pattern, no
    # matter which layer fetches it.
    for component in bake["components"]:
        assert f"{component}/*" in counted, (component, sorted(counted))


def test_both_ways_a_split_pass_names_its_component_are_read():
    """A split pass can name its component two ways — a PREFIX constant, or an
    inline startswith. Reading only one form is what left the text encoder
    uncounted and produced a FITS verdict on an image 42 GiB larger than
    reported.

    Checked against a FIXTURE, not against the live Dockerfile. It was written
    against the Dockerfile and broke within the hour when the text-encoder
    passes were removed — a parser capability should not be tested by which
    forms a particular file happens to use today."""
    fixture = '''
CANDIDATES = ["a/b"]
DEST = "/app/models/thing"
allow_patterns=["model_index.json"]
EOF
DEST = "/app/models/thing"
PREFIX = "transformer/"
EOF
DEST = "/app/models/thing"
files = [s for s in sibs if s.rfilename.startswith("text_encoder/")]
EOF
'''
    prefixes = isz._split_prefixes(fixture, "/app/models/thing")
    assert prefixes == ["text_encoder/", "transformer/"]

    # And the live Dockerfile's own split really is picked up.
    text = open("Dockerfile", encoding="utf-8").read()
    assert "transformer/" in isz._split_prefixes(text, "/app/models/ltx")
