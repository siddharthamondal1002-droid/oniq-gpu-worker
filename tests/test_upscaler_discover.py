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
    this module made that mistake, so it is pinned.

    It now reads the TREE endpoint rather than model-info's siblings list,
    because that is the only one huggingface_hub's own parsers say carries a
    Xet hash, and because siblings returned an lfs dict with no oid in it for
    the very file this comparison exists to check."""

    VAE = "vae/diffusion_pytorch_model.safetensors"

    def _cross(self, baked, candidate):
        """`baked` and `candidate` are (sha256, xetHash) pairs; None for absent."""

        def entry(pair):
            sha, xet = pair
            e = {"type": "file", "path": self.VAE, "size": 2_400_000_000,
                 "oid": "g" * 40}
            if sha:
                e["lfs"] = {"oid": sha, "size": 2_400_000_000, "pointerSize": 134}
            if xet:
                e["xetHash"] = xet
            return e

        def get(url, token, timeout=60):
            assert "/tree/" in url, url
            return [entry(baked if ud.BAKED_REPO in url else candidate)]

        rows = [{"repo": "cand", "revision": "d" * 40,
                 "blob_hashes": {self.VAE: candidate[0]}}]
        return ud.vae_crosscheck(rows, "tok", get)

    def test_a_missing_baked_hash_is_unknown_not_a_mismatch(self):
        out = self._cross((None, None), ("a" * 64, None))
        assert out["matches"]["cand"].startswith("UNKNOWN")
        assert "UNRESOLVED" in out["note"]

    def test_two_real_and_equal_hashes_are_the_same(self):
        out = self._cross(("a" * 64, None), ("a" * 64, None))
        assert out["matches"]["cand"] == "SAME (sha256)"

    def test_two_real_and_different_hashes_are_different(self):
        out = self._cross(("a" * 64, None), ("b" * 64, None))
        assert out["matches"]["cand"] == "DIFFERENT (sha256)"

    def test_the_raw_entry_is_kept_when_neither_hash_comes_back(self):
        # The absence that cost three runs left no record of what the registry
        # had actually returned, so the note in the pin guessed at a cause.
        out = self._cross((None, None), (None, None))
        raw = out["baked_vae"]["files"][self.VAE]["raw"]
        assert raw["type"] == "file" and raw["size"] == 2_400_000_000

    def test_an_unreadable_tree_says_so_rather_than_reporting_a_mismatch(self):
        def get(url, token, timeout=60):
            raise urllib_error(404)

        rows = [{"repo": "cand", "revision": "d" * 40,
                 "blob_hashes": {self.VAE: "a" * 64}}]
        out = ud.vae_crosscheck(rows, "tok", get)
        assert "404" in out["error"]
        assert "matches" not in out


class TestLikeIsComparedWithLike:
    """A sha256 and a Xet hash are different functions over the same bytes.
    Comparing one to the other answers DIFFERENT for two identical files — the
    same shape of false negative, one layer down."""

    def _files(self, sha=None, xet=None):
        return {"vae/x.safetensors": {"sha256": sha, "xet": xet, "raw": {}}}

    def test_a_xet_hash_settles_it_when_no_sha256_came_back(self):
        assert ud._same_content(self._files(xet="x" * 64),
                                self._files(xet="x" * 64)) == "SAME (xet)"
        assert ud._same_content(self._files(xet="x" * 64),
                                self._files(xet="y" * 64)) == "DIFFERENT (xet)"

    def test_a_sha256_is_never_compared_against_a_xet_hash(self):
        # One side has only a sha256, the other only a Xet hash. There is no
        # comparison to make, and inventing one would answer DIFFERENT.
        assert ud._same_content(self._files(sha="a" * 64),
                                self._files(xet="a" * 64)).startswith("UNKNOWN")

    def test_sha256_is_preferred_when_both_currencies_are_present(self):
        # Both are deterministic, but sha256 is the one the rest of this module
        # already prints and the one the pin quotes.
        out = ud._same_content(self._files(sha="a" * 64, xet="x" * 64),
                               self._files(sha="a" * 64, xet="y" * 64))
        assert out == "SAME (sha256)"


class TestOneReaderForTheHash:
    def test_both_paths_read_the_lfs_sha256_the_same_way(self):
        # measure() fell back from `oid` to `sha256`; the cross-check read only
        # `oid`. Two readings of one field is how a laxer path and a stricter
        # path drift apart, and the stricter one then reports an absence.
        assert ud._lfs_sha256({"lfs": {"oid": "a" * 64}}) == "a" * 64
        assert ud._lfs_sha256({"lfs": {"sha256": "b" * 64}}) == "b" * 64
        assert ud._lfs_sha256({"lfs": {}}) is None
        assert ud._lfs_sha256({}) is None

    def test_the_tree_endpoint_is_the_one_huggingface_hub_documents(self):
        # HfApi.list_repo_tree builds
        #   {endpoint}/api/{repo_type}s/{repo_id}/tree/{revision}{path}
        # and its RepoFile parser pops "xetHash". Model-info's RepoSibling does
        # not carry one, which is why this read exists at all.
        assert ud.TREE_API.startswith("https://huggingface.co/api/models/")
        assert "/tree/{revision}/{path}" in ud.TREE_API


class TestWhatADifferenceMeans:
    """DIFFERENT BYTES IS NOT DIFFERENT LATENT SPACE — and the run on
    2026-08-31 measured exactly that case, so the distinction is load-bearing
    rather than hypothetical. The baked vae hashed to 265ca87c… and the
    upscaler's to 3419989c… in both currencies. Whether that matters is a
    question about the CONFIG the upsample pipeline normalises with, not about
    the bytes."""

    VAE = "vae/diffusion_pytorch_model.safetensors"
    BAKED_CFG = {"_class_name": "AutoencoderKLLTXVideo", "_diffusers_version": "0.28.0",
                 "latent_channels": 128, "latents_mean": [0.1] * 128,
                 "latents_std": [1.0] * 128, "scaling_factor": 1.0}

    def _run(self, theirs_cfg, baked_sha="a" * 64, cand_sha="b" * 64):
        def get(url, token, timeout=60):
            sha = baked_sha if ud.BAKED_REPO in url else cand_sha
            entry = {"type": "file", "path": self.VAE, "size": 1_600_000_000,
                     "oid": "g" * 40, "xetHash": None}
            if sha:
                entry["lfs"] = {"oid": sha, "size": 1_600_000_000, "pointerSize": 134}
            return [entry]

        seen = []

        def get_text(url, token, timeout=60):
            seen.append(url)
            return json.dumps(self.BAKED_CFG if ud.BAKED_REPO in url else theirs_cfg)

        rows = [{"repo": "them/repo", "revision": "d" * 40,
                 "blob_hashes": {self.VAE: cand_sha}}]
        return ud.vae_crosscheck(rows, "tok", get, get_text), seen

    def test_identical_configs_make_a_byte_difference_a_packaging_difference(self):
        out, _ = self._run(dict(self.BAKED_CFG))
        assert out["matches"]["them/repo"] == "DIFFERENT (sha256)"
        assert out["config_delta"]["them/repo"] == {}

    def test_a_different_normalisation_is_named_field_by_field(self):
        # latents_mean and latents_std ARE the normalisation
        # LTXLatentUpsamplePipeline applies. A checkpoint that disagrees on
        # them does not share a latent space, whatever the weights hash to.
        theirs = dict(self.BAKED_CFG, latents_mean=[0.9] * 128)
        out, _ = self._run(theirs)
        assert list(out["config_delta"]["them/repo"]) == ["latents_mean"]

    def test_a_diffusers_version_bump_is_not_a_latent_space_change(self):
        theirs = dict(self.BAKED_CFG, _diffusers_version="0.38.0")
        out, _ = self._run(theirs)
        assert out["config_delta"]["them/repo"] == {}

    def test_the_config_is_not_fetched_when_the_bytes_already_agree(self):
        # Nothing to interpret, and a read that costs nothing still costs a
        # reader's attention when it prints an answer to a question nobody
        # asked.
        out, seen = self._run(dict(self.BAKED_CFG), cand_sha="a" * 64)
        assert out["matches"]["them/repo"] == "SAME (sha256)"
        assert out["config_delta"] == {} and seen == []

    def test_an_unreadable_config_says_so_rather_than_reporting_agreement(self):
        def get(url, token, timeout=60):
            sha = "a" * 64 if ud.BAKED_REPO in url else "b" * 64
            return [{"type": "file", "path": self.VAE, "size": 1, "oid": "g" * 40,
                     "lfs": {"oid": sha, "size": 1, "pointerSize": 134}}]

        def get_text(url, token, timeout=60):
            raise urllib_error(404)

        rows = [{"repo": "them/repo", "revision": "d" * 40,
                 "blob_hashes": {self.VAE: "b" * 64}}]
        out = ud.vae_crosscheck(rows, "tok", get, get_text)
        assert "404" in out["config_delta"]["them/repo"]["_error"]

    def test_a_long_vector_prints_as_its_head_and_its_length(self):
        printed = ud._brief([0.1] * 128)
        assert printed.endswith("128 values]") and len(printed) < 80
        assert ud._brief([1, 2]) == "[1, 2]"
        assert ud._brief(128) == "128"
