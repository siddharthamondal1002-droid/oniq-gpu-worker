"""The $0 read of publisher-stated sampling settings.

The point of this module is that it does NOT decide anything: it prints
citations and a human writes the number down. These tests hold that line.
"""

import modelprobe
from validation import probe_settings as ps


def test_it_reads_the_pinned_revision_not_a_branch():
    """A model card on main can describe a checkpoint the probe is not
    running. The citation has to come from the commit being probed."""
    row = modelprobe.PROBE_MODELS["cogvideox-i2v"]
    url = ps.card_url(row["repo"], row["revision"], "README.md")
    assert row["revision"] in url
    assert "/main/" not in url


def test_an_unreadable_card_is_reported_rather_than_assumed(capsys):
    ps.report(fetcher=lambda url: None)
    out = capsys.readouterr().out
    assert "UNREADABLE (no citation available)" in out
    assert "PIPELINE DEFAULT" in out


def test_a_card_with_no_setting_says_so_rather_than_going_quiet(capsys):
    ps.report(fetcher=lambda url: "# A model\n\nJust prose.\n")
    out = capsys.readouterr().out
    assert "states no step or guidance setting" in out


def test_the_citation_lines_are_printed_verbatim(capsys):
    ps.report(fetcher=lambda url: "pipe(num_inference_steps=40, guidance_scale=3.5)")
    out = capsys.readouterr().out
    assert "num_inference_steps=40, guidance_scale=3.5" in out


def test_it_reads_every_candidate_and_skips_the_unevaluated_one(capsys):
    ps.report(fetcher=lambda url: "")
    out = capsys.readouterr().out
    for key in modelprobe.PROBE_MODELS:
        assert key in out
    for key, reason in modelprobe.NOT_EVALUATED.items():
        assert f"{key}: NOT_EVALUATED ({reason})" in out


def test_it_never_writes_a_setting_back_into_the_model_table():
    """Reading is not deciding. If this module could mutate a row, a bad
    parse would silently become the experiment."""
    before = {k: dict(v) for k, v in modelprobe.PROBE_MODELS.items()}
    ps.report(fetcher=lambda url: "num_inference_steps=999")
    assert {k: dict(v) for k, v in modelprobe.PROBE_MODELS.items()} == before


def test_the_scheduler_reader_ignores_a_body_that_is_not_a_config():
    assert ps.scheduler_defaults("<!doctype html>") == {}
    assert ps.scheduler_defaults("[1,2,3]") == {}
