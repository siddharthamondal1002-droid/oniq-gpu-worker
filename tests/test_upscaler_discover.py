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
GOOD_CONFIG = json.dumps(dict(ud.EXPECTED))


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
        bad = dict(ud.EXPECTED)
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
