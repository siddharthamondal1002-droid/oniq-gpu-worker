"""The image-size projection: parsed from the Dockerfile, never restated.

The projection exists to answer one question before an hour of build time
is spent — does the media image fit on a runner. Every test here defends
the two ways that answer goes wrong: a parse that silently drops part of a
bake, and an unmeasured addend quietly counted as zero.
"""

import pytest

from validation import image_size as isz


DOCKERFILE = """
FROM python:3.11-slim AS base
FROM base AS media
RUN python3 - <<'EOF'
CANDIDATES = [
    ("Vendor/ltx-distilled", "#distilled"),
    ("Vendor/ltx", ""),
]
DEST = "/app/models/ltx"
SIZE_GUARD_BYTES = 16 * 1024**3
COMPONENTS = ("transformer", "vae", "text_encoder", "tokenizer", "scheduler")
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


def _repo(files):
    return {"siblings": [{"rfilename": n, "size": s} for n, s in files.items()]}


def test_candidates_read_the_repo_not_the_tag():
    bakes = isz.parse_bakes(DOCKERFILE)
    assert bakes[0]["candidates"] == ["Vendor/ltx-distilled", "Vendor/ltx"]
    assert "#distilled" not in bakes[0]["candidates"]


def test_a_single_line_component_tuple_yields_every_component():
    """The regression. COMPONENTS sits on ONE line; reading first-per-line
    returned only 'transformer' and dropped the VAE and text encoder."""
    patterns = isz.parse_bakes(DOCKERFILE)[0]["patterns"]
    assert patterns == [
        "model_index.json",
        "transformer/*",
        "vae/*",
        "text_encoder/*",
        "tokenizer/*",
        "scheduler/*",
    ]


def test_a_plain_pattern_list_is_read_verbatim():
    assert isz.parse_bakes(DOCKERFILE)[1]["patterns"] == ["*.json", "*.safetensors", "*.txt"]


def test_a_split_url_literal_is_rejoined():
    assert isz.parse_piper(DOCKERFILE) == "https://example.invalid/voices/voice.tar.gz"


def test_only_files_matching_the_patterns_are_counted():
    """The repos carry single-file checkpoints the bake never fetches;
    counting them would overstate the image by more than a runner's disk."""
    info = _repo({
        "model_index.json": 100,
        "transformer/x.safetensors": 4 * GIB,
        "vae/y.safetensors": GIB,
        "ltx-video-2b.safetensors": 40 * GIB,
    })
    out = isz.download_bytes("r", ["model_index.json", "transformer/*", "vae/*"], lambda _: info)
    assert out["bytes"] == 5 * GIB + 100
    assert out["files"] == 3


def test_an_unreachable_repo_is_none_not_zero():
    def boom(_):
        raise TimeoutError("nope")

    assert isz.download_bytes("r", ["*"], boom)["bytes"] is None


def test_an_unmeasured_base_stage_refuses_to_produce_a_total(capsys):
    code, _ = isz.report(
        DOCKERFILE,
        base_image_bytes=None,
        fetch=lambda r: _repo({"a.safetensors": GIB, "model_index.json": 1,
                               "transformer/t.safetensors": GIB}),
        head=lambda u: 60 * 1024**2,
    )
    out = capsys.readouterr().out
    assert code == 2
    assert "PROJECTED MEDIA IMAGE: NOT MEASURED" in out
    assert "FITS" not in out


def test_an_unreachable_bake_refuses_to_produce_a_total(capsys):
    def boom(_):
        raise TimeoutError("nope")

    code, _ = isz.report(DOCKERFILE, base_image_bytes=GIB, fetch=boom, head=lambda u: 1)
    out = capsys.readouterr().out
    assert code == 2
    assert "NOT MEASURED" in out


def test_a_small_image_fits_as_is(capsys):
    code, _ = isz.report(
        DOCKERFILE,
        base_image_bytes=GIB,
        fetch=lambda r: _repo({"model_index.json": 1, "transformer/t.safetensors": GIB,
                               "a.safetensors": GIB}),
        head=lambda u: 1,
    )
    assert code == 0
    assert "VERDICT: FITS as-is" in capsys.readouterr().out


def test_a_large_image_needs_the_mount(capsys):
    code, _ = isz.report(
        DOCKERFILE,
        base_image_bytes=8 * GIB,
        fetch=lambda r: _repo({"model_index.json": 1, "transformer/t.safetensors": 8 * GIB,
                               "a.safetensors": 8 * GIB}),
        head=lambda u: 1,
    )
    assert code == 0
    assert "FITS ONLY ON /mnt" in capsys.readouterr().out


def test_an_image_too_big_for_the_mount_is_refused(capsys):
    code, _ = isz.report(
        DOCKERFILE,
        base_image_bytes=10 * GIB,
        fetch=lambda r: _repo({"model_index.json": 1, "transformer/t.safetensors": 30 * GIB,
                               "a.safetensors": 30 * GIB}),
        head=lambda u: 1,
    )
    assert code == 1
    assert "DOES NOT FIT" in capsys.readouterr().out


def test_the_budget_accounts_for_the_build_holding_two_copies():
    """One times the floor fitting is not the question: docker keeps the
    layer cache and the assembled image at the same time."""
    assert isz.RUNNER_MNT_FREE_BYTES > isz.RUNNER_ROOT_FREE_BYTES


def test_a_dockerfile_missing_a_bake_raises_rather_than_guessing():
    with pytest.raises(isz.SizeParseError):
        isz.parse_bakes("FROM scratch\n")
