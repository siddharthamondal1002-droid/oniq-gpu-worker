"""The other side of the pairing gate, tested against fakes.

upscaler_discover asks which upsampler may be baked beside the checkpoint we
have. This asks which checkpoint may be baked beside the upsampler we pinned,
and it exists because the first question answered NONE.

Two properties are load-bearing and both are asserted:

  - it applies the SAME latent-space fields the Dockerfile compares, so a
    CANDIDATE verdict cannot promise a build that then refuses;
  - it refuses a candidate that pairs perfectly and cannot fit the card,
    because "loads" and "pairs" are different questions and only one of them
    is about bytes.
"""

import json

import pytest

from validation import pairing_discover as pd
from validation import upscaler_discover as ud

SECRET = "hf_a_token_that_must_never_be_printed"

# The upsampler's own vae: five-stage encoder, timestep-conditioned decoder.
UPSTREAM_VAE = {
    "_class_name": "AutoencoderKLLTXVideo",
    "latent_channels": 128,
    "latents_mean": [0.1] * 128,
    "latents_std": [1.0] * 128,
    "block_out_channels": [128, 256, 512, 1024, 2048],
    "layers_per_block": [4, 6, 6, 2, 2],
    "spatio_temporal_scaling": [True, True, True, True],
    "down_block_types": ["LTXVideo095DownBlock3D"] * 4,
    "timestep_conditioning": True,
}
# What ONIQ bakes today: four stages, no timestep conditioning.
BAKED_VAE = dict(
    UPSTREAM_VAE,
    block_out_channels=[128, 256, 512, 512],
    layers_per_block=[4, 3, 3, 3, 3],
    spatio_temporal_scaling=[True, True, True, False],
)
BAKED_VAE.pop("down_block_types")

GIB = 1024**3


def _info(transformer_bytes, vae_config=True, licence=True, pipeline=True,
          components=pd.COMPONENTS, sha="a" * 40):
    siblings = [{"rfilename": f"{c}/config.json", "size": 400} for c in components]
    siblings.append({"rfilename": "transformer/diffusion_pytorch_model.safetensors",
                     "size": transformer_bytes})
    siblings.append({"rfilename": "vae/diffusion_pytorch_model.safetensors",
                     "size": int(2.4 * GIB)})
    if pipeline:
        siblings.append({"rfilename": "model_index.json", "size": 300})
    if licence:
        siblings.append({"rfilename": "LTX-Video-Open-Weights-License-0.X.txt",
                         "size": 11000})
    if not vae_config:
        siblings = [s for s in siblings if s["rfilename"] != "vae/config.json"]
    return {"sha": sha, "cardData": {"license": "other"}, "siblings": siblings}


CATALOGUE = [{"id": "Lightricks/LTX-Video"},
             {"id": "Lightricks/LTX-Video-0.9.7-dev"},
             {"id": "Lightricks/ltxv-spatial-upscaler-0.9.7"}]


def _fakes(info, vae):
    def get(url, token, timeout=60):
        if isinstance(info, Exception):
            raise info
        # The LIST endpoint answers with an array; the INFO one with an
        # object. Routing on that here keeps candidates() honest — it really
        # does iterate a list in production.
        if url.startswith("https://huggingface.co/api/models?"):
            return CATALOGUE
        return info

    def get_text(url, token, timeout=60):
        return json.dumps(vae)

    return get, get_text


class TestTheGateIsTheBuilds:
    def test_the_fields_compared_are_the_dockerfiles(self):
        # Imported, never re-typed: two copies of this tuple is how a read and
        # a build reach different verdicts.
        assert pd.LATENT_SPACE is ud.LATENT_SPACE
        docker = open("Dockerfile", encoding="utf-8").read()
        block = docker.split("LATENT_SPACE = (", 1)[1].split(")", 1)[0]
        assert sorted(f.strip().strip('",') for f in block.split()
                      if f.strip(' ,"')) == sorted(pd.LATENT_SPACE)

    def test_a_matching_vae_within_the_guard_is_a_candidate(self):
        get, get_text = _fakes(_info(int(9 * GIB)), UPSTREAM_VAE)
        row = pd.measure("them/ltx", UPSTREAM_VAE, SECRET, None, get, get_text)
        assert row["pairs"] == "PAIRS"
        assert row["verdict"] == "CANDIDATE"

    def test_todays_baked_vae_is_reported_as_the_mismatch_it_is(self):
        # The exact failure that started this: it pairs on nothing that
        # matters and would be baked happily by every other check.
        get, get_text = _fakes(_info(int(9 * GIB)), BAKED_VAE)
        row = pd.measure(pd.CURRENT_REPO, UPSTREAM_VAE, SECRET,
                         pd.CURRENT_REVISION, get, get_text)
        assert row["verdict"].startswith("DOES NOT PAIR")
        assert sorted(row["latent_delta"]) == [
            "block_out_channels", "down_block_types", "layers_per_block",
            "spatio_temporal_scaling"]


class TestPairingIsNotFitting:
    def test_weights_larger_than_the_card_are_refused_even_though_they_pair(self):
        # 60 GiB of weights alone on a 48 GiB card. It would pair perfectly and
        # never load — measured here rather than on a rented A6000.
        get, get_text = _fakes(_info(int(60 * GIB)), UPSTREAM_VAE)
        row = pd.measure("them/ltx-huge", UPSTREAM_VAE, SECRET, None, get, get_text)
        assert row["pairs"] == "PAIRS"
        assert row["fits_card"] is False
        assert row["verdict"].startswith("WILL NOT FIT")

    def test_over_the_guard_but_under_the_card_is_a_decision_not_a_refusal(self):
        # 40 GiB against a 32 GiB guard on a 48 GiB card. The guard exists to
        # stop an oversized checkpoint arriving BY ACCIDENT; choosing one on
        # purpose is the owner's call, so this must not read as "impossible".
        #
        # The size here moved from 26 GiB when the guard was raised 16 -> 32
        # for the 0.9.7-distilled repoint. The PROPERTY under test is the
        # three-way distinction — inside the guard, between guard and card,
        # over the card — not any particular number.
        get, get_text = _fakes(_info(int(40 * GIB)), UPSTREAM_VAE)
        row = pd.measure("them/ltx-huge-ish", UPSTREAM_VAE, SECRET, None, get, get_text)
        assert row["within_size_guard"] is False and row["fits_card"] is True
        assert row["verdict"].startswith("PAIRS AND FITS, OVER TODAY'S GUARD")
        assert "owner decision" in row["verdict"]

    def test_the_cards_are_the_ones_the_endpoint_directive_names(self):
        # THIS NUMBER WAS WRONG ONCE — written as a 24 GiB A5000, carried over
        # from the earlier retarget, which would have condemned every 13B
        # candidate on a card that can hold one. Tying the table to the
        # directive's own list is what stops it going stale again.
        from validation import endpoint_gpus

        assert set(pd.CARD_VRAM_GIB) == set(endpoint_gpus.WANTED)
        assert pd.CARD_VRAM_BYTES == 48 * GIB


class TestItRefusesRatherThanGuessing:
    def test_no_credential_is_refused(self, capsys):
        code, rows = pd.report(None)
        assert code == 2 and rows == []
        assert "BLOCKED" in capsys.readouterr().out

    def test_an_unreadable_upsampler_config_blocks_instead_of_pairing(self, capsys):
        def get(url, token, timeout=60):
            return _info(int(9 * GIB))

        def get_text(url, token, timeout=60):
            raise _http(404)

        code, rows = pd.report(SECRET, get, get_text)
        assert code == 2 and rows == []
        out = capsys.readouterr().out
        assert "nothing to pair AGAINST" in out

    def test_a_missing_vae_config_is_unverified_not_paired(self):
        get, get_text = _fakes(_info(int(9 * GIB), vae_config=False), UPSTREAM_VAE)
        row = pd.measure("them/ltx", UPSTREAM_VAE, SECRET, None, get, get_text)
        assert row["pairs"].startswith("UNVERIFIED")
        assert row["verdict"].startswith("DOES NOT PAIR")

    def test_a_repository_shipping_no_terms_is_refused(self):
        get, get_text = _fakes(_info(int(9 * GIB), licence=False), UPSTREAM_VAE)
        row = pd.measure("them/ltx", UPSTREAM_VAE, SECRET, None, get, get_text)
        assert row["verdict"].startswith("NO TERMS")

    def test_an_incomplete_pipeline_is_refused(self):
        get, get_text = _fakes(
            _info(int(9 * GIB), components=("transformer", "vae")), UPSTREAM_VAE)
        row = pd.measure("them/ltx", UPSTREAM_VAE, SECRET, None, get, get_text)
        assert "INCOMPLETE" in row["verdict"]
        assert "text_encoder" in row["verdict"]

    def test_the_token_is_never_printed(self, capsys):
        get, get_text = _fakes(_info(int(9 * GIB)), UPSTREAM_VAE)
        pd.report(SECRET, get, get_text)
        assert SECRET not in capsys.readouterr().out


class TestItStaysInStepWithThePin:
    def test_the_pinned_upsampler_matches_the_pin_file(self):
        # Two places naming one artifact is how they drift. The pin is LIVE
        # again — the checkpoint moved to meet it on 2026-08-31 — so this
        # reads the declaration rather than the prose.
        pin = open("ltx-upscaler.pin", encoding="utf-8").read()
        declared = [l.split("#", 1)[0].strip() for l in pin.splitlines()
                    if l.split("#", 1)[0].strip()]
        assert declared == [f"{pd.UPSCALER_REPO} {pd.UPSCALER_REVISION}"]

    def test_the_current_checkpoint_matches_the_dockerfiles_pin(self):
        docker = open("Dockerfile", encoding="utf-8").read()
        assert f'PINNED_REVISION = "{pd.CURRENT_REVISION}"' in docker
        assert f'("{pd.CURRENT_REPO}", "")' in docker

    def test_the_transformer_guard_matches_the_dockerfiles(self):
        docker = open("Dockerfile", encoding="utf-8").read()
        assert f"SIZE_GUARD_BYTES = {pd.SIZE_GUARD_BYTES // 1024**3} * 1024**3" in docker


def _http(code):
    import urllib.error

    return urllib.error.HTTPError("u", code, "no", None, None)


class TestTheCatalogueIsNotTheUpsampler:
    def test_the_upsampler_repo_is_not_offered_as_a_checkpoint(self):
        # It appears in the same author listing and is not a pipeline; leaving
        # it in would print one confusing NOT-A-PIPELINE row every run.
        def get(url, token, timeout=60):
            return CATALOGUE

        names = pd.candidates(SECRET, get)
        assert "Lightricks/ltxv-spatial-upscaler-0.9.7" not in names
        assert pd.CURRENT_REPO in names

    def test_the_baked_checkpoint_is_listed_even_if_the_search_misses_it(self):
        def get(url, token, timeout=60):
            return [{"id": "Lightricks/LTX-Video-0.9.7-dev"}]

        assert pd.CURRENT_REPO in pd.candidates(SECRET, get)
