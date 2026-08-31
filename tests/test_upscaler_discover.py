"""The upsampler read, tested against fakes — it never touches the network here.

The module exists so that `ltx-upscaler.pin` can be filled from measurement
rather than from a value typed out of a browser. That makes two properties
load-bearing, and both are asserted below:

  - it applies the SAME gates the Dockerfile applies, so a PINNABLE verdict
    cannot promise a build that the build itself would then refuse;
  - it never prints the credential, the same discipline hf_auth holds.
"""

import json

import pytest

from validation import upscaler_discover as ud

SECRET = "hf_a_token_that_must_never_be_printed"

OFFICIAL = "Lightricks/ltxv-spatial-upscaler-0.9.7"
THIRD_PARTY = "a-r-r-o-w/LTX-0.9.8-Latent-Upsampler"

GOOD_INFO = {
    "sha": "1111111111111111111111111111111111111111",
    "cardData": {"license": "other", "license_name": "ltx-open-weights",
                 "license_link": "LICENSE.md"},
    "siblings": [
        {"rfilename": "config.json", "size": 400},
        {"rfilename": "LICENSE.md", "size": 9000},
        {"rfilename": "diffusion_pytorch_model.safetensors", "size": 505_000_000},
    ],
}
GOOD_CONFIG = json.dumps({"_class_name": ud.MODEL_CLASS, **ud.EXPECTED})


def _fakes(info=GOOD_INFO, config=GOOD_CONFIG, seen=None):
    def get(url, token, timeout=60):
        if seen is not None:
            seen.append((url, token))
        if isinstance(info, Exception):
            raise info
        return info

    def get_text(url, token, timeout=60):
        if seen is not None:
            seen.append((url, token))
        if isinstance(config, Exception):
            raise config
        return config

    return get, get_text


class TestTheGatesMatchTheBuild:
    def test_a_complete_candidate_is_pinnable(self):
        get, get_text = _fakes()
        row = ud.measure(OFFICIAL, SECRET, get, get_text)
        assert row["verdict"] == "PINNABLE"
        assert row["revision"] == GOOD_INFO["sha"]

    def test_the_expected_config_is_exactly_the_dockerfiles(self):
        # If the Dockerfile's gate ever changes, this read must change with it,
        # or a PINNABLE verdict starts promising a build that then refuses.
        docker = open("Dockerfile", encoding="utf-8").read()
        for key, value in ud.EXPECTED.items():
            assert f'"{key}"' in docker, key
        assert "SIZE_GUARD_BYTES = 1024**3" in docker

    def test_a_wrong_config_field_is_not_pinnable(self):
        bad = {"_class_name": ud.MODEL_CLASS, **ud.EXPECTED}
        bad["temporal_upsample"] = True  # would change the frame count
        get, get_text = _fakes(config=json.dumps(bad))
        row = ud.measure(OFFICIAL, SECRET, get, get_text)
        assert row["verdict"] == "NOT PINNABLE"
        assert row["config_mismatch"] == {"temporal_upsample": True}

    def test_a_repository_shipping_no_terms_is_not_pinnable(self):
        info = dict(GOOD_INFO)
        info["siblings"] = [s for s in GOOD_INFO["siblings"]
                            if s["rfilename"] != "LICENSE.md"]
        get, get_text = _fakes(info=info)
        row = ud.measure(THIRD_PARTY, SECRET, get, get_text)
        assert row["licence_files"] == []
        assert row["verdict"] == "NOT PINNABLE"

    def test_weights_above_the_guard_are_not_pinnable(self):
        info = dict(GOOD_INFO)
        info["siblings"] = [
            {"rfilename": "LICENSE.md", "size": 9000},
            {"rfilename": "model.safetensors", "size": ud.SIZE_GUARD_BYTES + 1},
        ]
        get, get_text = _fakes(info=info)
        row = ud.measure(OFFICIAL, SECRET, get, get_text)
        assert row["within_size_guard"] is False
        assert row["verdict"] == "NOT PINNABLE"

    def test_an_unreadable_repository_says_so_rather_than_guessing(self):
        get, get_text = _fakes(info=urllib_error(404))
        row = ud.measure(THIRD_PARTY, SECRET, get, get_text)
        assert row["verdict"] == "UNREADABLE"
        assert "404" in row["detail"]

    def test_the_config_is_read_at_the_resolved_sha_never_at_a_branch(self):
        seen: list = []
        get, get_text = _fakes(seen=seen)
        ud.measure(OFFICIAL, SECRET, get, get_text)
        config_url = [u for u, _ in seen if u.endswith("config.json")][0]
        assert GOOD_INFO["sha"] in config_url
        assert "/main/" not in config_url


class TestItRefusesAndLeaksNothing:
    def test_no_credential_is_refused_rather_than_read_anonymously(self, capsys):
        code, rows = ud.report(None)
        assert code == 2 and rows == []
        assert "BLOCKED" in capsys.readouterr().out

    def test_the_token_is_never_printed(self, capsys):
        get, get_text = _fakes()
        ud.report(SECRET, get, get_text)
        assert SECRET not in capsys.readouterr().out

    def test_it_measures_both_candidates_and_chooses_neither(self, capsys):
        get, get_text = _fakes()
        _, rows = ud.report(SECRET, get, get_text)
        assert [r["repo"] for r in rows] == list(ud.CANDIDATES)
        out = capsys.readouterr().out
        # It prints the line to paste, but says plainly that a verdict is not
        # permission — the licence judgement stays the owner's.
        assert "IS NOT PERMISSION" in out
        assert "owner's judgement" in out

    def test_both_documented_candidates_are_covered(self):
        assert OFFICIAL in ud.CANDIDATES and THIRD_PARTY in ud.CANDIDATES


def urllib_error(code):
    import urllib.error

    return urllib.error.HTTPError("u", code, "no", None, None)


class TestBothRepositoryShapes:
    """The first live run (2026-08-31) got BOTH of these wrong, which is the
    only reason they are pinned here: a resolver that mis-reads the registry
    is worse than none, because it condemns a good candidate quietly."""

    def test_a_licence_named_after_the_licence_is_found(self):
        # Lightricks ships LTX-Video-Open-Weights-License-0.X.txt. An exact
        # LICENSE/NOTICE list missed it and reported the repo unlicensed.
        assert ud._licence_files(
            [".gitattributes", "LTX-Video-Open-Weights-License-0.X.txt",
             "README.md", "model_index.json"]
        ) == ["LTX-Video-Open-Weights-License-0.X.txt"]

    def test_a_repository_with_genuinely_no_terms_still_reports_none(self):
        assert ud._licence_files(
            [".gitattributes", "config.json",
             "diffusion_pytorch_model.safetensors"]) == []

    def test_a_pipeline_repo_finds_the_component_config_in_its_subfolder(self):
        info = {
            "sha": "c" * 40,
            "cardData": {"license": "other"},
            "siblings": [
                {"rfilename": "model_index.json", "size": 300},
                {"rfilename": "LTX-Video-Open-Weights-License-0.X.txt", "size": 11000},
                {"rfilename": "latent_upsampler/config.json", "size": 231},
                {"rfilename": "latent_upsampler/diffusion_pytorch_model.safetensors",
                 "size": 505_009_832},
                {"rfilename": "vae/diffusion_pytorch_model.safetensors",
                 "size": 2_400_000_000},
            ],
        }
        get, get_text = _fakes(info=info)
        row = ud.measure(OFFICIAL, SECRET, get, get_text)
        assert row["is_pipeline"] is True
        assert row["config_path"] == "latent_upsampler/config.json"
        # The GUARD APPLIES TO THE COMPONENT, not the pipeline: counting the
        # vae's 2.4 GB against a 1 GiB component guard failed a repo that is
        # fine.
        assert row["weight_bytes"] == 505_009_832
        assert row["within_size_guard"] is True
        assert row["licence_files"] == ["LTX-Video-Open-Weights-License-0.X.txt"]
        assert row["verdict"] == "PINNABLE"

    def test_a_bare_component_repo_still_reads_the_root_config(self):
        get, get_text = _fakes()
        row = ud.measure(THIRD_PARTY, SECRET, get, get_text)
        assert row["is_pipeline"] is False
        assert row["config_path"] == "config.json"
        assert row["component_prefix"] == "(repository root)"

    def test_a_missing_config_says_where_it_looked(self):
        info = dict(GOOD_INFO)
        info["siblings"] = [{"rfilename": "model_index.json", "size": 300},
                            {"rfilename": "LICENSE", "size": 10}]
        get, get_text = _fakes(info=info)
        row = ud.measure(OFFICIAL, SECRET, get, get_text)
        assert row["config_path"] is None
        assert "searched" in row["config_mismatch"]["config.json"]


class TestAbsentIsNotWrong:
    """Lightricks' config declares only _class_name and leaves the rest to the
    class defaults. The build constructs the model and asserts the RESOLVED
    values, so this read must not condemn a config for being terse — that
    false negative is what blocked the only licence-clean candidate."""

    LIGHTRICKS_CONFIG = json.dumps({
        "_class_name": "LTXLatentUpsamplerModel",
        "_diffusers_version": "0.35.0.dev0",
    })

    def test_a_config_that_defaults_everything_is_still_pinnable(self):
        get, get_text = _fakes(config=self.LIGHTRICKS_CONFIG)
        row = ud.measure(OFFICIAL, SECRET, get, get_text)
        assert row["config_mismatch"] == {}
        assert row["defaulted"] == sorted(ud.EXPECTED)
        assert row["verdict"] == "PINNABLE"

    def test_a_present_but_wrong_field_is_still_caught(self):
        get, get_text = _fakes(config=json.dumps(
            {"_class_name": ud.MODEL_CLASS, "in_channels": 64}))
        row = ud.measure(OFFICIAL, SECRET, get, get_text)
        assert row["config_mismatch"] == {"in_channels": 64}
        assert row["verdict"] == "NOT PINNABLE"

    def test_the_wrong_class_is_caught_however_terse_the_config(self):
        get, get_text = _fakes(config=json.dumps({"_class_name": "AutoencoderKL"}))
        row = ud.measure(OFFICIAL, SECRET, get, get_text)
        assert row["config_mismatch"]["_class_name"] == "AutoencoderKL"
        assert row["verdict"] == "NOT PINNABLE"

    def test_the_build_asserts_the_constructed_model_not_the_json(self):
        # The read is only honest if the build really does resolve defaults.
        docker = open("Dockerfile", encoding="utf-8").read()
        assert "LTXLatentUpsamplerModel.from_config(cfg)" in docker
        assert "getattr(model.config, k, None)" in docker
        assert "SIZE_GUARD_BYTES = 1024**3" in docker


class TestUnknownIsNotFalse:
    """The cross-check's first run printed SAME LATENTS: False for a repo whose
    hash had not come back. Absent data is not negative data — the third time
    this module made that mistake, so it is pinned."""

    def _cross(self, baked_oid, candidate_oid):
        def get(url, token, timeout=60):
            return {"siblings": [{"rfilename": "vae/diffusion_pytorch_model.safetensors",
                                  "lfs": {"oid": baked_oid} if baked_oid else {}}]}
        rows = [{"repo": "cand", "blob_hashes":
                 {"vae/diffusion_pytorch_model.safetensors": candidate_oid}}]
        return ud.vae_crosscheck(rows, "tok", get)

    def test_a_missing_baked_hash_is_unknown_not_a_mismatch(self):
        out = self._cross(None, "a" * 64)
        assert out["matches"]["cand"].startswith("UNKNOWN")
        assert "UNRESOLVED" in out["note"]

    def test_two_real_and_equal_hashes_are_the_same(self):
        out = self._cross("a" * 64, "a" * 64)
        assert out["matches"]["cand"] == "SAME"

    def test_two_real_and_different_hashes_are_different(self):
        out = self._cross("a" * 64, "b" * 64)
        assert out["matches"]["cand"] == "DIFFERENT"
