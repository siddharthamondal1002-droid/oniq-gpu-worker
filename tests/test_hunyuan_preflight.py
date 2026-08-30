"""The free gate, and the frame count it derives.

Owner directive 2026-08-30 section 10: the frame count must come from "the
exact legal frame-count requirements of the selected Hunyuan I2V
implementation/checkpoint", not from a README. Section 12: zero GPU jobs.
"""

import os
import sys
import urllib.error

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation import hunyuan_preflight as hp  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_legal_counts_follow_the_vae_rule_not_a_readme():
    # A causal video VAE with temporal compression R encodes frame 0 alone
    # and the rest in blocks of R: legal counts are R*k+1.
    for row in hp.legal_frame_counts(4, 24):
        assert (row["frames"] - 1) % 4 == 0
    for row in hp.legal_frame_counts(8, 24):
        assert (row["frames"] - 1) % 8 == 0


def test_the_chosen_count_is_the_shortest_useful_one():
    chosen = hp.choose_frames(hp.legal_frame_counts(4, 24))
    assert chosen["frames"] == 49
    assert chosen["seconds"] >= hp.MIN_USEFUL_SECONDS


def test_counts_below_the_useful_floor_are_excluded():
    # 5 frames is legal for ratio 4 and cannot show a person slowly
    # turning. Legality is necessary, not sufficient.
    frames = [r["frames"] for r in hp.legal_frame_counts(4, 24)]
    assert 5 not in frames and 9 not in frames


def test_counts_above_the_baseline_window_are_excluded():
    # Going markedly longer than the ~4.04s LTX baseline would compare a
    # harder job against an easier one, in Hunyuan's disfavour.
    for row in hp.legal_frame_counts(4, 24):
        assert row["seconds"] <= hp.MAX_USEFUL_SECONDS


def test_an_unreadable_ratio_refuses_rather_than_assuming_four():
    for bad in (None, "4", 0, -1):
        with pytest.raises(hp.PreflightFailure) as exc:
            hp.legal_frame_counts(bad, 24)
        assert exc.value.gate == "frame-rule-unreadable"


def test_gates_fail_when_the_endpoint_has_no_volume():
    facts = {
        "vae/config.json": {"temporal_compression_ratio": 4,
                            "spatial_compression_ratio": 16},
        "transformer/config.json": {"task_type": "i2v"},
        "model_index.json": {"_class_name": "HunyuanVideo15ImageToVideoPipeline",
                             "vae": [], "text_encoder": [], "text_encoder_2": [],
                             "scheduler": []},
        "revision": "a" * 40,
    }
    rows, chosen, _ = hp.gates(facts, {"gpuTypeIds": ["NVIDIA RTX A5000"],
                                       "networkVolumeId": ""})
    failed = [name for name, ok, _ in rows if not ok]
    assert "network volume attached" in failed
    assert chosen["frames"] == 49


def test_gates_pass_on_a_fully_prepared_endpoint():
    facts = {
        "vae/config.json": {"temporal_compression_ratio": 4,
                            "spatial_compression_ratio": 16},
        "transformer/config.json": {"task_type": "i2v"},
        "model_index.json": {"_class_name": "HunyuanVideo15ImageToVideoPipeline",
                             "vae": [], "text_encoder": [], "text_encoder_2": [],
                             "scheduler": []},
        "revision": "a" * 40,
    }
    rows, _, _ = hp.gates(facts, {"gpuTypeIds": ["NVIDIA RTX A5000"],
                                  "networkVolumeId": "vol-123"})
    assert [name for name, ok, _ in rows if not ok] == []


def test_a_non_a5000_endpoint_fails_the_gate():
    facts = {
        "vae/config.json": {"temporal_compression_ratio": 4,
                            "spatial_compression_ratio": 16},
        "transformer/config.json": {"task_type": "i2v"},
        "model_index.json": {"_class_name": "HunyuanVideo15ImageToVideoPipeline",
                             "vae": [], "text_encoder": [], "text_encoder_2": [],
                             "scheduler": []},
        "revision": "a" * 40,
    }
    rows, _, _ = hp.gates(facts, {"gpuTypeIds": ["NVIDIA RTX A5000", "NVIDIA A100"],
                                  "networkVolumeId": "vol-123"})
    assert "A5000 is the only GPU on the endpoint" in [
        name for name, ok, _ in rows if not ok
    ]


def test_the_module_submits_no_job():
    with open(os.path.join(ROOT, "validation", "hunyuan_preflight.py"),
              encoding="utf-8") as fh:
        source = fh.read()
    for forbidden in ("submit_job", "purge_queue", "cancel_job",
                      "retarget_template", "set_template_env",
                      "create_template", "set_execution_timeout"):
        assert forbidden not in source, forbidden


# ------------------------------------- the reference is really an image


class _Resp:
    def __init__(self, body, status=200):
        self._body, self.status = body, status
        # frame_pull's fetcher reads Content-Length before the body, to
        # refuse an oversized artifact before it lands in a runner's
        # memory. A fake without headers would not exercise that path.
        self.headers = {"Content-Length": str(len(body))}

    def read(self, n=None):
        return self._body[:n] if n else self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


PNG_HEAD = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8 + \
    (704).to_bytes(4, "big") + (480).to_bytes(4, "big") + b"\x00" * 64


def test_a_real_png_reference_passes_and_reports_its_size():
    info = hp.reference_readable("https://pub-x.r2.dev", "a/ref.png",
                                 fetch=lambda url: PNG_HEAD)
    assert info["ok"] is True
    assert info["format"] == "png"
    assert (info["width"], info["height"]) == (704, 480)


def test_an_html_error_page_served_as_200_is_refused():
    """The failure this check exists for.

    A 404 page served with image/png is still a 404 page. Conditioning a
    paid job on one produces a clip of nothing, and the money is spent
    before anyone looks. Headers are what a server claims; magic bytes are
    what the decoder will actually see.
    """
    info = hp.reference_readable("https://pub-x.r2.dev", "a/missing.png",
                                 fetch=lambda url: b"<!DOCTYPE html><html>404")
    assert info["ok"] is False
    assert "not an image" in info["error"]


def test_an_unreachable_reference_is_a_failed_gate_not_a_crash():
    def boom(url):
        raise OSError("connection refused")

    info = hp.reference_readable("https://pub-x.r2.dev", "a/ref.png", fetch=boom)
    assert info["ok"] is False
    assert "OSError" in info["error"]


def test_an_empty_body_is_refused_rather_than_read_as_an_image():
    info = hp.reference_readable("https://pub-x.r2.dev", "a/ref.png",
                                 fetch=lambda url: b"")
    assert info["ok"] is False
    assert "empty" in info["error"]


def _through_frame_pull(opener):
    """reference_readable's real reader, with ONE seam: the socket.

    The ladder under test lives in frame_pull._fetch, so the fake is
    injected where frame_pull itself takes one rather than by patching a
    module attribute — _fetch binds its opener as a default at definition
    time, so a monkeypatched attribute would never be consulted and the
    test would silently exercise nothing.
    """
    from validation import frame_pull

    return lambda url: frame_pull._fetch(url, opener=opener)


def test_the_reference_is_read_through_frame_pulls_two_UA_ladder():
    """MEASURED 2026-08-30: this gate failed with HTTP 403 and the run was
    one step from being reported as "the reference is gone".

    It was not gone. r2.dev applies Cloudflare's UA-signature filter (error
    code 1010) to known scraper agents, python-urllib among them, and a
    bare urlopen sends exactly that UA. frame_pull had measured this on
    2026-08-29, built the browser-UA retry, and FIXTURES.md records the
    posture as PUBLIC_R2_ARTIFACT_READ = CURRENT. A second fetcher written
    here threw all of that away.

    So: a 403 under the honest UA must still resolve.
    """
    from validation import frame_pull

    seen = []

    def opener(url, ua, timeout=120):
        seen.append(ua)
        if ua == frame_pull.UA_PRIMARY:
            raise urllib.error.HTTPError(url, 403, "Forbidden", {}, None)
        return _Resp(PNG_HEAD)

    info = hp.reference_readable("https://pub-x.r2.dev", "a/ref.png",
                                 fetch=_through_frame_pull(opener))
    assert info["ok"] is True, info
    assert seen == [frame_pull.UA_PRIMARY, frame_pull.UA_BROWSER]


def test_a_403_that_survives_both_UAs_is_still_a_failed_gate():
    """The retry must not become a blanket pass. A bucket that really is
    private refuses both identities, and that is a stop."""
    def opener(url, ua, timeout=120):
        raise urllib.error.HTTPError(url, 403, "Forbidden", {}, None)

    info = hp.reference_readable("https://pub-x.r2.dev", "a/ref.png",
                                 fetch=_through_frame_pull(opener))
    assert info["ok"] is False
    assert "403" in info["error"]


def test_a_404_is_not_retried_under_a_second_identity():
    """Only the UA-ban shape earns the second identity. A missing object
    answers every UA identically, and retrying would blur the diagnosis."""
    from validation import frame_pull

    seen = []

    def opener(url, ua, timeout=120):
        seen.append(ua)
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    info = hp.reference_readable("https://pub-x.r2.dev", "a/gone.png",
                                 fetch=_through_frame_pull(opener))
    assert info["ok"] is False
    assert "404" in info["error"]
    assert seen == [frame_pull.UA_PRIMARY], "a 404 must not be retried"


def test_the_default_reader_IS_frame_pulls_and_not_a_bare_urlopen():
    """The wiring claim itself, held in place.

    Every test above injects a fetcher, so none of them would notice this
    module quietly growing a second reader again. This one reads the
    source: reference_readable must reach for frame_pull's _fetch and must
    not call urlopen itself.
    """
    import ast
    import inspect
    import textwrap

    source = inspect.getsource(hp.reference_readable)
    assert "from validation.frame_pull import _fetch" in source

    # The DOCSTRING explains the regression and therefore says "urlopen".
    # Matching on raw source would trip on the explanation rather than on
    # the code, so the body is checked with the docstring removed.
    tree = ast.parse(textwrap.dedent(source))
    fn = tree.body[0]
    if (fn.body and isinstance(fn.body[0], ast.Expr)
            and isinstance(fn.body[0].value, ast.Constant)
            and isinstance(fn.body[0].value.value, str)):
        fn.body = fn.body[1:]
    body = ast.unparse(fn)
    assert "urlopen" not in body, (
        "a bare urlopen sends the python-urllib UA that r2.dev's 1010 "
        "filter refuses — this is exactly the 2026-08-30 regression"
    )


def test_an_unreadable_reference_fails_the_gate_list():
    facts = {
        "vae/config.json": {"temporal_compression_ratio": 4,
                            "spatial_compression_ratio": 16},
        "transformer/config.json": {"task_type": "i2v"},
        "model_index.json": {"_class_name": "HunyuanVideo15ImageToVideoPipeline",
                             "vae": [], "text_encoder": [], "text_encoder_2": [],
                             "scheduler": []},
        "revision": "a" * 40,
    }
    endpoint = {"gpuTypeIds": ["NVIDIA RTX A5000"], "networkVolumeId": "v1"}
    rows, _, _ = hp.gates(facts, endpoint, 24, {"ok": False, "error": "HTTP 404"})
    failed = [name for name, ok, _ in rows if not ok]
    assert any("reference" in name for name in failed)


def test_no_reference_information_means_no_reference_gate():
    # Absent evidence must not be reported as a pass.
    facts = {
        "vae/config.json": {"temporal_compression_ratio": 4,
                            "spatial_compression_ratio": 16},
        "transformer/config.json": {"task_type": "i2v"},
        "model_index.json": {"_class_name": "HunyuanVideo15ImageToVideoPipeline",
                             "vae": [], "text_encoder": [], "text_encoder_2": [],
                             "scheduler": []},
        "revision": "a" * 40,
    }
    endpoint = {"gpuTypeIds": ["NVIDIA RTX A5000"], "networkVolumeId": "v1"}
    rows, _, _ = hp.gates(facts, endpoint, 24, None)
    assert not any("reference" in name for name, _, _ in rows)
