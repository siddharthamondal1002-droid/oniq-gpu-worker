import pytest

from validation import model_registry as mr


def fake_get(payloads):
    def get(url, token, timeout=60):
        for key, value in payloads.items():
            if key in url:
                if isinstance(value, Exception):
                    raise value
                return value
        raise AssertionError(f"unexpected url {url}")

    return get


class FakeHTTPError(Exception):
    def __init__(self, code):
        super().__init__(f"HTTP {code}")
        self.code = code


# ---------------------------------------------------------------- predicates


def test_candidate_matches_only_within_its_author():
    ltx = next(c for c in mr.CANDIDATES if c.key == "ltx-2b")
    assert ltx.matches("Lightricks/LTX-Video")
    # Someone else's mirror of the same name is NOT the publisher's checkpoint.
    assert not ltx.matches("someone-else/LTX-Video")


def test_ltx_2b_predicate_excludes_the_13b_rows():
    ltx2b = next(c for c in mr.CANDIDATES if c.key == "ltx-2b")
    assert ltx2b.matches("Lightricks/LTX-Video")
    assert not ltx2b.matches("Lightricks/LTX-Video-13B-Distilled")
    assert not ltx2b.matches("Lightricks/LTX-Video-13B-GGUF")


def test_wan21_and_wan22_never_match_each_other():
    """The brief requires Wan2.2 be treated as a separate candidate."""
    w21 = next(c for c in mr.CANDIDATES if c.key == "wan21-i2v-14b-480p")
    w22 = next(c for c in mr.CANDIDATES if c.key == "wan22-i2v-a14b")
    assert w21.matches("Wan-AI/Wan2.1-I2V-14B-480P")
    assert not w21.matches("Wan-AI/Wan2.2-I2V-A14B")
    assert w22.matches("Wan-AI/Wan2.2-I2V-A14B")
    assert not w22.matches("Wan-AI/Wan2.1-I2V-14B-480P")


def test_every_candidate_carries_a_decision_token():
    """The final answer must be one of the owner's tokens, so each row has to
    map to one — otherwise a winner could not be reported in his vocabulary."""
    allowed = {
        "LTX_2B",
        "LTX_13B",
        "WAN2_1_I2V_14B",
        "WAN2_2_I2V_A14B",
        "HUNYUAN_VIDEO_1_5_I2V",
        "COGVIDEOX_I2V",
        # Owner directive 2026-08-30 (Wan2.2 LoRA production integration).
        # The reference implementation runs I2V-A14B, whose two experts
        # measure 53.23 GiB EACH on this account's own registry read; the
        # publisher names the 5B as the variant that "can also run on
        # consumer-grade graphics cards like 4090". Measuring it is the only
        # way to answer whether the A5000 can carry this feature at all.
        "WAN2_2_TI2V_5B",
    }
    assert {c.decision for c in mr.CANDIDATES} <= allowed
    assert len(mr.CANDIDATES) == 10, (
        "nine from the 2026-08-29 brief, plus Wan2.2 TI2V-5B added "
        "2026-08-30 when the Wan2.2 directive landed"
    )


def test_resolve_and_preferred_favour_the_loadable_layout():
    ids = ["Wan-AI/Wan2.1-I2V-14B-480P", "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"]
    cand = next(c for c in mr.CANDIDATES if c.key == "wan21-i2v-14b-480p")
    assert mr.resolve(cand, ids) == ids
    assert mr.preferred(cand, ids).endswith("-Diffusers")


def test_preferred_is_none_when_nothing_matches():
    cand = next(c for c in mr.CANDIDATES if c.key == "wan22-i2v-a14b")
    assert mr.preferred(cand, []) is None


# ---------------------------------------------------------------- measurement


def test_measure_splits_weights_by_role_and_finds_the_second_expert():
    """Wan2.2 A14B ships two experts. Missing transformer_2 would halve the
    reported weights and make the model look deployable when it is not."""
    info = {
        "sha": "abc123",
        "cardData": {"license": "apache-2.0"},
        "siblings": [
            {"rfilename": "model_index.json", "size": 500},
            {"rfilename": "transformer/a.safetensors", "size": 1000},
            {"rfilename": "transformer_2/a.safetensors", "size": 1000},
            {"rfilename": "text_encoder/t.safetensors", "size": 300},
            {"rfilename": "vae/v.safetensors", "size": 100},
            {"rfilename": "image_encoder/i.safetensors", "size": 50},
            {"rfilename": "transformer/config.json", "size": 9},
        ],
    }
    row = mr.measure("Wan-AI/Wan2.2-I2V-A14B", "tok", get=fake_get({"api/models": info}))
    assert row["roles"] == {
        "transformer": 1000,
        "transformer_2": 1000,
        "text_encoder": 300,
        "vae": 100,
        "image_encoder": 50,
    }
    assert row["is_diffusers"] is True
    assert row["revision"] == "abc123"
    assert row["licence"] == "apache-2.0"
    # config.json is not a weight file and must not be counted as one.
    assert row["total_weight_bytes"] == 2450
    assert "transformer/config.json" in row["config_paths"]


def test_measure_surfaces_root_level_single_file_variants():
    """LTX keeps fp8 and distilled as FILES in one repo. A candidate list
    built only from repository names would silently drop two owner rows."""
    info = {
        "sha": "d00d",
        "cardData": {},
        "tags": ["license:other"],
        "siblings": [
            {"rfilename": "ltxv-13b-0.9.8-dev.safetensors", "size": 26_000},
            {"rfilename": "ltxv-13b-0.9.8-dev-fp8.safetensors", "size": 13_000},
            {"rfilename": "ltxv-13b-0.9.8-distilled.safetensors", "size": 26_000},
            {"rfilename": "README.md", "size": 10},
        ],
    }
    row = mr.measure("Lightricks/LTX-Video", "tok", get=fake_get({"api/models": info}))
    names = [f["name"] for f in row["single_files"]]
    assert "ltxv-13b-0.9.8-dev-fp8.safetensors" in names
    assert "README.md" not in names
    # Sorted largest first, so the base checkpoint leads.
    assert row["single_files"][0]["bytes"] == 26_000
    assert row["licence"] == "other", "licence falls back to the tag"


def test_variant_of_reads_precision_from_the_file_name():
    assert mr.variant_of("ltxv-13b-0.9.8-dev-fp8.safetensors") == "fp8"
    assert mr.variant_of("ltxv-13b-distilled.safetensors") == "distilled"
    assert mr.variant_of("ltxv-13b-dev.safetensors") == "base"
    assert mr.variant_of("ltxv-13b-Q4.gguf") == "gguf"


def test_unreadable_401_says_gated_or_absent_rather_than_guessing():
    """The mistake hf_discover was written for: 401 anonymous cannot tell a
    gated repo from one that does not exist, and must not claim either."""
    row = mr.measure(
        "Wan-AI/Nope", "tok", get=fake_get({"api/models": FakeHTTPError(401)})
    )
    assert row["verdict"] == "UNREADABLE"
    assert "gated or absent" in row["detail"]


def test_unreadable_403_is_reported_as_terms_not_accepted():
    row = mr.measure(
        "tencent/HunyuanVideo-1.5", "tok",
        get=fake_get({"api/models": FakeHTTPError(403)}),
    )
    assert row["verdict"] == "UNREADABLE"
    assert "terms" in row["detail"]


def test_licence_link_is_carried_because_other_is_a_pointer_not_a_licence():
    info = {
        "sha": "x",
        "cardData": {
            "license": "other",
            "license_name": "ltx-video-licence",
            "license_link": "LICENSE.md",
        },
        "siblings": [],
    }
    row = mr.measure("Lightricks/LTX-Video", "t", get=fake_get({"api/models": info}))
    assert row["licence_name"] == "ltx-video-licence"
    assert row["licence_link"] == "LICENSE.md"


def test_repo_ids_dedupes_and_sorts_the_listing():
    listing = [{"id": "b/x"}, {"modelId": "a/y"}, {"id": "b/x"}]
    assert mr.repo_ids(listing) == ["a/y", "b/x"]


def test_catalogue_asks_for_the_author_not_a_guessed_name():
    seen = {}

    def get(url, token, timeout=60):
        seen["url"] = url
        return []

    mr.catalogue("Wan-AI", "tok", get=get)
    assert "author=Wan-AI" in seen["url"]


def test_fetch_config_pins_the_measured_revision():
    """A config read off `main` can describe different weights than the ones
    measured. The revision travels."""
    seen = {}

    def get(url, token, timeout=60):
        seen["url"] = url
        return {"num_layers": 48}

    mr.fetch_config("Wan-AI/W", "deadbeef", "transformer/config.json", "t", get=get)
    assert "/resolve/deadbeef/transformer/config.json" in seen["url"]


def test_token_is_never_echoed_into_the_url():
    seen = {}

    def get(url, token, timeout=60):
        seen["url"] = url
        return []

    mr.catalogue("Lightricks", "SECRET-TOKEN", get=get)
    assert "SECRET-TOKEN" not in seen["url"]


# ------------------------------------------------ the dtype actually on disk


def _header(tensors):
    """A minimal safetensors prefix: u64 header length, then the JSON."""
    import json as _json
    import struct
    blob = _json.dumps(tensors).encode()
    return struct.pack("<Q", len(blob)) + blob


def test_header_dtype_reads_the_precision_from_the_file_itself():
    """Component configs mostly carry no torch_dtype, and the shipped
    precision is not a detail: Wan ships fp32 transformers, so assuming bf16
    reports a bf16 load at twice its real size."""
    blob = _header({"w": {"dtype": "F32", "shape": [4096, 4096]}})
    assert mr.header_dtype("r", "rev", "t/x.safetensors", None,
                           fetch=lambda *a: blob) == "fp32"


def test_header_dtype_follows_the_largest_tensor_not_the_first():
    """Checkpoints routinely mix a few fp32 norms into a bf16 model. The bulk
    is what sets the bytes."""
    blob = _header({
        "norm": {"dtype": "F32", "shape": [16]},
        "w": {"dtype": "BF16", "shape": [4096, 4096]},
    })
    assert mr.header_dtype("r", "rev", "t/x.safetensors", None,
                           fetch=lambda *a: blob) == "bf16"


def test_header_dtype_recognises_an_fp8_checkpoint():
    blob = _header({"w": {"dtype": "F8_E4M3", "shape": [4096, 4096]}})
    assert mr.header_dtype("r", "rev", "t/x.safetensors", None,
                           fetch=lambda *a: blob) == "fp8"


def test_header_dtype_declines_rather_than_guesses():
    """Every failure path returns None so the caller keeps its own default
    rather than inheriting a fabricated precision."""
    blob = _header({"w": {"dtype": "F32", "shape": [8]}})
    assert mr.header_dtype("r", "rev", "t/x.bin", None, fetch=lambda *a: blob) is None
    assert mr.header_dtype("r", "rev", "t/x.safetensors", None,
                           fetch=lambda *a: None) is None
    assert mr.header_dtype("r", "rev", "t/x.safetensors", None,
                           fetch=lambda *a: b"\x00" * 4) is None


def test_header_dtype_refuses_a_header_longer_than_it_fetched():
    """A truncated read must not be parsed as if it were complete."""
    import struct
    assert mr.header_dtype(
        "r", "rev", "t/x.safetensors", None,
        fetch=lambda *a: struct.pack("<Q", 10**9) + b"{}",
    ) is None


def test_header_dtype_survives_a_metadata_only_header():
    blob = _header({"__metadata__": {"format": "pt"}})
    assert mr.header_dtype("r", "rev", "t/x.safetensors", None,
                           fetch=lambda *a: blob) is None
