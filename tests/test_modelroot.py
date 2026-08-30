"""The resolver's two contracts, which are deliberately not the same.

Owner directive 2026-08-30: experimental models FAIL CLOSED and never fall
back. Production keeps the baked image as a fallback so that attaching a
volume for an experiment cannot take LTX down.

The single most important test in this file is the one asserting a Hunyuan
request never returns an LTX path. A silent fallback would produce a
benchmark that compares LTX against LTX and reports it as Hunyuan.
"""

import importlib
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modelroot  # noqa: E402

MODEL = "HUNYUAN_15_I2V_480_STEP"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def volume(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_VOLUME_ROOT", str(tmp_path))
    importlib.reload(modelroot)
    yield tmp_path
    monkeypatch.delenv("MODEL_VOLUME_ROOT", raising=False)
    importlib.reload(modelroot)


def _hydrate(path, files, revision=None):
    os.makedirs(path, exist_ok=True)
    sizes = {}
    for name, body in files.items():
        full = os.path.join(path, name)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as fh:
            fh.write(body)
        sizes[name] = len(body)
    marker = {
        "revision": revision or modelroot.EXPERIMENTAL[MODEL]["revision"],
        "files": sizes,
        "bytes": sum(sizes.values()),
    }
    with open(os.path.join(path, modelroot.READY_MARKER), "w") as fh:
        json.dump(marker, fh)


# --------------------------------------------------- finding the volume


def test_an_explicit_template_path_always_wins(monkeypatch, tmp_path):
    # An operator who names a path has answered the question; the detector
    # must not second-guess them.
    monkeypatch.setenv("MODEL_VOLUME_ROOT", str(tmp_path))
    reloaded = importlib.reload(modelroot)
    assert reloaded.VOLUME_ROOT == str(tmp_path)
    assert "template" in reloaded.VOLUME_ROOT_SOURCE
    assert reloaded.volume_mounted() is True


def test_a_plain_directory_in_the_image_is_not_a_mounted_volume(monkeypatch):
    """The failure this detector exists to prevent.

    /workspace can exist inside the image. If mere existence counted as a
    mounted volume, every model would resolve to a real-looking path with
    nothing in it, and the fail-closed contract would report
    MODEL_NOT_HYDRATED for a volume that was never attached at all —
    a true statement pointing at the wrong problem.
    """
    monkeypatch.delenv("MODEL_VOLUME_ROOT", raising=False)
    reloaded = importlib.reload(modelroot)
    # "/" is a directory and is trivially not a separate filesystem from
    # itself, which is exactly the discrimination being made.
    assert reloaded._is_real_mount("/") is False
    assert reloaded._is_real_mount("/definitely/not/here") is False


def test_with_nothing_mounted_the_refusal_names_the_documented_path(monkeypatch):
    monkeypatch.delenv("MODEL_VOLUME_ROOT", raising=False)
    reloaded = importlib.reload(modelroot)
    assert reloaded.VOLUME_ROOT == "/runpod-volume"
    assert "no mounted candidate" in reloaded.VOLUME_ROOT_SOURCE
    assert reloaded.volume_mounted() is False


def test_where_reports_how_the_root_was_chosen(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_VOLUME_ROOT", str(tmp_path))
    reloaded = importlib.reload(modelroot)
    report = reloaded.where()
    assert report["volume_root_source"]
    assert "/runpod-volume" in report["volume_candidates"]


# ------------------------------------------------- experimental: fail closed


def test_an_unmounted_volume_is_named_not_guessed(monkeypatch):
    monkeypatch.setenv("MODEL_VOLUME_ROOT", "/definitely/not/mounted")
    importlib.reload(modelroot)
    with pytest.raises(modelroot.ModelUnavailable) as exc:
        modelroot.resolve(MODEL)
    assert exc.value.code == "MODEL_VOLUME_UNAVAILABLE"


def test_a_mounted_but_unhydrated_volume_says_so(volume):
    with pytest.raises(modelroot.ModelUnavailable) as exc:
        modelroot.resolve(MODEL)
    assert exc.value.code == "MODEL_NOT_HYDRATED"


def test_files_without_a_marker_are_not_a_model(volume):
    # An interrupted download leaves files and no marker. Loading those on
    # a rented card is the expensive way to discover they are incomplete.
    path = modelroot.model_dir(MODEL)
    os.makedirs(os.path.join(path, "transformer"), exist_ok=True)
    with open(os.path.join(path, "transformer", "part.bin"), "w") as fh:
        fh.write("half a checkpoint")
    with pytest.raises(modelroot.ModelUnavailable) as exc:
        modelroot.resolve(MODEL)
    assert exc.value.code == "MODEL_NOT_HYDRATED"


def test_a_hydrated_model_resolves_to_the_deterministic_layout(volume):
    path = modelroot.model_dir(MODEL)
    _hydrate(path, {"model_index.json": "{}"})
    resolved = modelroot.resolve(MODEL)
    assert resolved == path
    assert resolved.endswith(
        os.path.join("models", "oniq", "hunyuan",
                     "HunyuanVideo-1.5-480P-I2V-step-distill")
    )


def test_a_different_revision_is_corrupt_not_acceptable(volume):
    _hydrate(modelroot.model_dir(MODEL), {"model_index.json": "{}"},
             revision="0" * 40)
    with pytest.raises(modelroot.ModelUnavailable) as exc:
        modelroot.resolve(MODEL)
    assert exc.value.code == "MODEL_CORRUPT"


def test_a_size_mismatch_is_corrupt(volume):
    path = modelroot.model_dir(MODEL)
    _hydrate(path, {"vae/weights.bin": "x" * 200})
    with open(os.path.join(path, "vae", "weights.bin"), "w") as fh:
        fh.write("x" * 5)
    with pytest.raises(modelroot.ModelUnavailable) as exc:
        modelroot.resolve(MODEL)
    assert exc.value.code == "MODEL_CORRUPT"


@pytest.mark.parametrize("state", ["unmounted", "unhydrated", "corrupt"])
def test_hunyuan_NEVER_resolves_to_an_ltx_path(volume, monkeypatch, state):
    """The one that matters.

    A benchmark whose Hunyuan run silently loaded LTX would compare LTX
    against LTX and report a decision on it. Every failure mode must raise
    rather than return, and nothing returned may point into the baked tree.
    """
    if state == "unmounted":
        monkeypatch.setenv("MODEL_VOLUME_ROOT", "/definitely/not/mounted")
        importlib.reload(modelroot)
    elif state == "corrupt":
        _hydrate(modelroot.model_dir(MODEL), {"a.json": "{}"}, revision="0" * 40)

    with pytest.raises(modelroot.ModelUnavailable) as exc:
        modelroot.resolve(MODEL)
    assert "ltx" not in str(exc.value).lower()
    assert exc.value.code in (
        "MODEL_VOLUME_UNAVAILABLE", "MODEL_NOT_HYDRATED", "MODEL_CORRUPT"
    )


def test_the_source_contains_no_experimental_to_production_fallback():
    with open(os.path.join(ROOT, "modelroot.py"), encoding="utf-8") as fh:
        source = fh.read()
    body = source[source.index("def resolve(model_id"):
                  source.index("def resolve_production(")]
    assert "BAKED_ROOT" not in body, "resolve() can reach the baked tree"
    assert body.count("raise ModelUnavailable") >= 4


# --------------------------------------------- production: fallback survives


def test_production_falls_back_to_the_image_when_the_volume_is_absent(monkeypatch):
    monkeypatch.setenv("MODEL_VOLUME_ROOT", "/definitely/not/mounted")
    importlib.reload(modelroot)
    assert modelroot.resolve_production("ltx") == "/app/models/ltx"
    assert modelroot.resolve_production_file("MODEL_ID") == "/app/models/MODEL_ID"


def test_an_empty_volume_does_not_break_production(volume):
    # The state EVERY worker is in before hydration finishes.
    os.makedirs(os.path.join(modelroot.oniq_root(), "ltx"), exist_ok=True)
    assert modelroot.resolve_production("ltx") == "/app/models/ltx"


def test_a_populated_volume_wins_for_production(volume):
    component = os.path.join(modelroot.oniq_root(), "ltx")
    os.makedirs(component, exist_ok=True)
    with open(os.path.join(component, "model_index.json"), "w") as fh:
        fh.write("{}")
    assert modelroot.resolve_production("ltx") == component


def test_production_components_resolve_independently(volume):
    ltx = os.path.join(modelroot.oniq_root(), "ltx")
    os.makedirs(ltx, exist_ok=True)
    with open(os.path.join(ltx, "model_index.json"), "w") as fh:
        fh.write("{}")
    assert modelroot.resolve_production("ltx") == ltx
    assert modelroot.resolve_production("story") == "/app/models/story"
    assert modelroot.resolve_production("piper") == "/app/models/piper"


def test_nothing_a_caller_sends_can_steer_the_root():
    """Owner constraint: a user may not specify GPU, provider, endpoint,
    model, checkpoint, runtime, budget, precision or offloading strategy.
    Where weights load from belongs on that list.

    Checked over the AST rather than the text. A substring scan matched
    the word "request" in this module's own docstring explaining the rule
    — the same false positive the GraphQL mutation guard hit, and a guard
    that fires on its own documentation is one that gets deleted rather
    than fixed.
    """
    import ast as _ast

    with open(os.path.join(ROOT, "modelroot.py"), encoding="utf-8") as fh:
        source = fh.read()
    tree = _ast.parse(source)

    imported = set()
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, _ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    # Nothing that carries a job, a request or a user.
    assert imported <= {"json", "os", "__future__"}, sorted(imported)

    # EVERY environment read names MODEL_VOLUME_ROOT and nothing else.
    #
    # Counting the reads was the earlier form of this check, and it broke
    # the moment mount detection legitimately needed a second one. The
    # count was never the property worth guarding: what matters is that no
    # OTHER variable can steer where weights load from. Asserting the
    # argument is both stricter and stable under refactoring.
    reads = []
    for node in _ast.walk(tree):
        if not isinstance(node, _ast.Call):
            continue
        func = node.func
        if not isinstance(func, _ast.Attribute) or func.attr != "get":
            continue
        target = func.value
        if isinstance(target, _ast.Attribute) and target.attr == "environ":
            assert node.args, "os.environ.get() with no argument"
            first = node.args[0]
            assert isinstance(first, _ast.Constant), _ast.dump(first)
            reads.append(first.value)
    assert reads, "no environment read found — the detector must read one"
    assert set(reads) == {"MODEL_VOLUME_ROOT"}, reads


def test_where_reports_the_refusal_code_rather_than_a_path(volume):
    report = modelroot.where()
    assert report["experimental"][MODEL] == {"error": "MODEL_NOT_HYDRATED"}
    assert report["production"]["ltx"] == "/app/models/ltx"
