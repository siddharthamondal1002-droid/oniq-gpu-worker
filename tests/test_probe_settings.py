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


# ------------------------- does the allow list cover what the pipeline needs


def test_component_folders_are_read_from_the_index_not_guessed():
    index = ('{"_class_name":"WanImageToVideoPipeline","_diffusers_version":"0.35.2",'
             '"vae":["diffusers","AutoencoderKLWan"],'
             '"transformer":["diffusers","WanTransformer3DModel"],'
             '"boundary_ratio":0.9}')
    assert ps.declared_components(index) == {
        "vae": "AutoencoderKLWan", "transformer": "WanTransformer3DModel",
    }


def test_a_body_that_is_not_an_index_declares_nothing():
    assert ps.declared_components("<!doctype html>") == {}
    assert ps.declared_components("[1,2]") == {}


def test_coverage_asks_only_whether_a_folder_is_reachable():
    """The narrow question. Whether a pattern fetches too MUCH is the LTX vae
    case and is answered by the download total; this answers the other one —
    a subfolder the pipeline needs that nothing would fetch."""
    assert ps.covered_by(["transformer/*", "vae/config.json"], "vae")
    assert ps.covered_by(["transformer/*"], "transformer")
    assert not ps.covered_by(["transformer/*"], "image_encoder")


def test_a_missing_component_is_shouted_not_mentioned(capsys):
    """It is not discovered until from_pretrained runs, which is after the
    whole download has been paid for on a rented GPU."""
    index = ('{"_class_name":"X","needed_thing":["diffusers","Y"]}')
    ps.report(fetcher=lambda url: index if url.endswith("model_index.json") else "")
    out = capsys.readouterr().out
    assert "MISSING FROM allow: needed_thing" in out


def test_an_unreadable_index_says_the_check_did_not_run(capsys):
    def fetcher(url):
        return None if url.endswith("model_index.json") else ""

    ps.report(fetcher=fetcher)
    assert "coverage NOT checked" in capsys.readouterr().out


def test_a_body_that_is_not_an_index_reads_as_unknown_rather_than_crashing():
    """A 404 page or a redirect is a real outcome. The reader must survive it
    — it runs before every probe and a crash here blocks the benchmark."""
    ps.report(fetcher=lambda url: "<!doctype html>404")
