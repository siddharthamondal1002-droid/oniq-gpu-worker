"""The seam that decides where weights are read from.

Owner directive 2026-08-30: the weights move off the image and onto a
network volume. These tests pin the property that makes that move safe to
do incrementally — per-component resolution with the image as fallback —
and the property that keeps it inside the owner's standing constraint:
nothing a caller sends can steer it.
"""

import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modelroot  # noqa: E402


def _reload(monkeypatch, volume_root):
    monkeypatch.setenv("MODEL_ROOT", str(volume_root))
    return importlib.reload(modelroot)


def test_an_absent_volume_falls_back_to_the_baked_image(monkeypatch, tmp_path):
    mr = _reload(monkeypatch, tmp_path / "nothing-here")
    assert mr.resolve("ltx") == "/app/models/ltx"
    assert mr.resolve_file("MODEL_ID") == "/app/models/MODEL_ID"


def test_an_empty_volume_falls_back_too(monkeypatch, tmp_path):
    # The state EVERY worker is in before hydration finishes. Failing
    # closed here would break a baked image the moment a blank volume was
    # attached to it.
    (tmp_path / "ltx").mkdir()
    mr = _reload(monkeypatch, tmp_path)
    assert mr.resolve("ltx") == "/app/models/ltx"


def test_a_populated_volume_wins(monkeypatch, tmp_path):
    component = tmp_path / "ltx"
    component.mkdir()
    (component / "model_index.json").write_text("{}")
    mr = _reload(monkeypatch, tmp_path)
    assert mr.resolve("ltx") == str(component)


def test_components_resolve_independently(monkeypatch, tmp_path):
    # The migration state that actually happens: one component hydrated,
    # the others still baked. A global "is the volume ready" flag would
    # get this wrong in both directions.
    ltx = tmp_path / "ltx"
    ltx.mkdir()
    (ltx / "model_index.json").write_text("{}")
    mr = _reload(monkeypatch, tmp_path)
    assert mr.resolve("ltx") == str(ltx)
    assert mr.resolve("story") == "/app/models/story"
    assert mr.resolve("piper") == "/app/models/piper"


def test_resolution_is_live_not_frozen_at_import(monkeypatch, tmp_path):
    # A warm worker can be hydrated mid-life: the volume is empty when the
    # process starts and populated by the time the next job arrives. If
    # resolution were cached at import, that worker would keep reading a
    # path that, on a weightless image, does not exist.
    mr = _reload(monkeypatch, tmp_path)
    assert mr.resolve("story") == "/app/models/story"
    story = tmp_path / "story"
    story.mkdir()
    (story / "config.json").write_text("{}")
    assert mr.resolve("story") == str(story)


def test_a_marker_file_resolves_by_file_not_by_directory(monkeypatch, tmp_path):
    (tmp_path / "MODEL_ID").write_text("Lightricks/LTX-Video#distilled")
    mr = _reload(monkeypatch, tmp_path)
    assert mr.resolve_file("MODEL_ID") == str(tmp_path / "MODEL_ID")
    # An absent marker still falls back rather than inventing one.
    assert mr.resolve_file("STORY_MODEL_ID") == "/app/models/STORY_MODEL_ID"


def test_nothing_a_caller_sends_can_steer_the_root():
    """Owner constraint: a user may not specify GPU, provider, endpoint,
    model, checkpoint, runtime, budget, precision or offloading strategy.
    Where weights load from belongs on that list, so the module reads the
    environment the TEMPLATE sets and takes no request-shaped input."""
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           os.pardir, "modelroot.py"), encoding="utf-8") as fh:
        source = fh.read()
    for forbidden in ("job", "request", "payload", "job_input", "contract"):
        assert f"import {forbidden}" not in source
    # One env var, read once, named explicitly.
    assert source.count("os.environ") == 1
    assert 'os.environ.get("MODEL_ROOT")' in source


def test_where_reports_both_roots_and_what_each_component_resolved_to(
    monkeypatch, tmp_path
):
    # So a job that ran off the volume can be told apart from one that ran
    # off the image by reading the report, not by inferring it from how
    # long the worker took to start.
    piper = tmp_path / "piper"
    piper.mkdir()
    (piper / "en-us-ryan-high.onnx").write_text("x")
    mr = _reload(monkeypatch, tmp_path)
    report = mr.where()
    assert report["volume_root"] == str(tmp_path)
    assert report["baked_root"] == "/app/models"
    assert report["volume_present"] is True
    assert report["components"]["piper"] == str(piper)
    assert report["components"]["ltx"] == "/app/models/ltx"
