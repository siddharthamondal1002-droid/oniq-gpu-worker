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
    # Both are set by the TEMPLATE and name a root an operator has chosen.
    # MODEL_CACHE_ROOT joined 2026-09-01 with the container-disk class. The
    # property guarded here is unchanged and is not the count: no variable
    # that could carry a job, a request or a user may steer where weights
    # load from, and an allowlist of two operator paths still satisfies it.
    assert set(reads) <= {"MODEL_VOLUME_ROOT", "MODEL_CACHE_ROOT"}, reads


def test_where_reports_the_refusal_code_rather_than_a_path(volume):
    report = modelroot.where()
    assert report["experimental"][MODEL] == {"error": "MODEL_NOT_HYDRATED"}
    assert report["production"]["ltx"] == "/app/models/ltx"


class TestTheTextEncoderLivesOnContainerDisk:
    """OWNER DIRECTIVE 2026-08-31, amended 2026-09-01.

    Moving the checkpoint to LTX-Video-0.9.7-distilled — needed so its vae
    matches the pinned spatial upsampler — put the media image at 57.97 GiB,
    and a hosted runner cannot build that (run 33434875038; the last
    SUCCESSFUL publish left 5.73 GiB free of 71.61). The text encoder is
    17.74 GiB of it, so it left the image.

    It went to a NETWORK VOLUME first, and came off one on 2026-09-01. The
    console's Releases tab showed what no API surface reports: attaching a
    volume narrows the endpoint's `locations` from ALL to that volume's
    single datacenter, and DETACHING DOES NOT WIDEN IT BACK. The endpoint is
    then pinned for life and cannot get a GPU the day that datacenter's
    approved tier runs dry — the daily-endpoint-recreation treadmill,
    explained. Owner chose option B: no volume, every datacenter, weights on
    the worker's own container disk.

    It is still a THIRD class and still fail-closed: ltx/story/piper fall
    back to the baked copy when their tree is empty, and this one has no
    baked copy to fall back to. What changed is WHERE it is expected and
    that an empty cache is now a normal cold-worker state to be fixed by
    fetching, rather than a configuration error."""

    ID = "LTX_TEXT_ENCODER"

    def test_it_is_registered_and_is_not_experimental(self):
        assert self.ID in modelroot.CACHE_RESIDENT
        assert self.ID not in modelroot.EXPERIMENTAL
        assert self.ID in modelroot.known_ids()
        assert modelroot.storage_class(self.ID) == "cache"

    def test_one_lookup_serves_both_registries(self):
        # Five call sites used to index EXPERIMENTAL directly. A second
        # registry only some of them knew about would resolve for hydration
        # and not for loading, or the reverse.
        assert modelroot.spec_for(self.ID) is modelroot.CACHE_RESIDENT[self.ID]
        assert modelroot.spec_for("HUNYUAN_15_I2V_480_STEP") is (
            modelroot.EXPERIMENTAL["HUNYUAN_15_I2V_480_STEP"])
        assert modelroot.spec_for("nope") is None

    def test_it_names_the_checkpoint_it_belongs_to(self):
        # The encoder and the transformer share an embedding space. A generic
        # "text-encoder" directory is how one checkpoint's encoder ends up
        # beside another's transformer — a silent quality failure, not a
        # crash.
        spec = modelroot.spec_for(self.ID)
        assert "0.9.7-distilled" in spec["directory"]
        assert spec["repo"] == "Lightricks/LTX-Video-0.9.7-distilled"
        assert spec["allow"] == ["text_encoder/*"]

    def test_the_revision_is_the_dockerfiles(self):
        docker = open("Dockerfile", encoding="utf-8").read()
        spec = modelroot.spec_for(self.ID)
        assert f'PINNED_REVISION = "{spec["revision"]}"' in docker
        assert f'("{spec["repo"]}", "")' in docker

    def test_the_declared_size_is_the_measured_one(self):
        # 19,049,290,370 bytes over text_encoder/*, run 33436186203. This
        # number feeds modelhydrate's disk check; an estimate breaks the gate.
        assert modelroot.spec_for(self.ID)["download_gib"] == 17.74
        assert round(19_049_290_370 / 1024**3, 2) == 17.74

    def test_an_absent_volume_no_longer_refuses_it(self, monkeypatch):
        """THE 2026-09-01 CHANGE. Under option B the production endpoint has
        NO volume on purpose, so demanding a mount here would refuse every
        clip on a correctly configured endpoint. The refusal must come from
        the cache being empty, not from storage that is meant to be absent."""
        monkeypatch.setattr(modelroot, "volume_mounted", lambda: False)
        with pytest.raises(modelroot.ModelUnavailable) as exc:
            modelroot.resolve(self.ID)
        assert exc.value.code == "MODEL_NOT_HYDRATED"

    def test_it_still_fails_closed_with_a_named_reason(self, monkeypatch):
        # THE POINT OF THE WHOLE CLASS, unchanged by the move. No baked
        # fallback exists, so a clip that cannot trust its text encoder must
        # REFUSE and say why — never run on an untrained embedding.
        with pytest.raises(modelroot.ModelUnavailable) as exc:
            modelroot.resolve(self.ID)
        assert exc.value.code in ("MODEL_NOT_HYDRATED", "MODEL_CORRUPT")
        assert exc.value.detail

    def test_it_lands_on_container_disk_not_the_volume(self):
        path = modelroot.model_dir(self.ID)
        assert path.startswith(modelroot.CACHE_ROOT), path
        assert not path.startswith(modelroot.VOLUME_ROOT), path
        # Same deterministic layout either side, so a model can move between
        # the two roots by changing registry and nothing else.
        assert path.endswith(
            os.path.join("ltx", "text-encoder-0.9.7-distilled"))

    def test_the_experimental_class_still_demands_a_real_volume(self,
                                                               monkeypatch):
        """The two classes fail DIFFERENTLY and must not be conflated. A
        Hunyuan probe with no volume is an operator error to fix before
        spending; an empty cache on a cold worker is not."""
        monkeypatch.setattr(modelroot, "volume_mounted", lambda: False)
        with pytest.raises(modelroot.ModelUnavailable) as exc:
            modelroot.resolve("HUNYUAN_15_I2V_480_STEP")
        assert exc.value.code == "MODEL_VOLUME_UNAVAILABLE"

    def test_ensure_fetches_only_when_the_cache_is_empty(self, monkeypatch):
        """A cold worker is the NORMAL first state under option B, so an
        empty cache is fetched rather than refused."""
        calls = []
        seen = {"n": 0}

        def fake_resolve(model_id):
            seen["n"] += 1
            if seen["n"] == 1:
                raise modelroot.ModelUnavailable("MODEL_NOT_HYDRATED", "cold")
            return "/app/cache/models/oniq/ltx/text-encoder-0.9.7-distilled"

        monkeypatch.setattr(modelroot, "resolve", fake_resolve)
        path = modelroot.ensure(self.ID, lambda mid: calls.append(mid))
        assert calls == [self.ID]
        assert path.startswith("/app/cache")
        # Resolved AGAIN after the fetch: the hydrator's own return value
        # would only say where it put files, not that the marker is present,
        # the revision is pinned and every length matches.
        assert seen["n"] == 2

    def test_ensure_never_refetches_over_a_corrupt_checkpoint(self,
                                                             monkeypatch):
        """Quietly re-downloading over a manifest that does not verify turns
        a diagnosable corruption into an intermittent one."""
        calls = []

        def fake_resolve(model_id):
            raise modelroot.ModelUnavailable("MODEL_CORRUPT", "bad manifest")

        monkeypatch.setattr(modelroot, "resolve", fake_resolve)
        with pytest.raises(modelroot.ModelUnavailable) as exc:
            modelroot.ensure(self.ID, lambda mid: calls.append(mid))
        assert exc.value.code == "MODEL_CORRUPT"
        assert calls == [], "a corrupt checkpoint must not trigger a re-fetch"

    def test_the_image_creates_that_cache_dir_and_gives_it_to_the_runtime_uid(
            self):
        """A directory made on demand under a root-owned /app fails EACCES —
        inside a job already being paid for, forty minutes into a cold
        start. That exact failure killed the first model probe when the hub
        tried to create its cache under a root-owned parent.

        The path is compared against modelroot's constant rather than
        retyped: a Dockerfile that prepares /app/cache while the code reads
        /app/cache2 looks correct in both files and fails only on a rented
        card."""
        docker = open("Dockerfile", encoding="utf-8").read()
        assert f"mkdir -p {modelroot.CACHE_ROOT}" in docker
        assert f"chown -R 10001:10001 {modelroot.CACHE_ROOT}" in docker
        # And it must happen while the build is still root.
        prepare = docker.index(f"mkdir -p {modelroot.CACHE_ROOT}")
        assert prepare < docker.rindex("USER oniq:oniq"), (
            "the cache is prepared after the image drops to uid 10001, so "
            "the chown cannot succeed"
        )

    def test_ensure_refuses_to_fetch_a_volume_model_mid_job(self):
        """Their contract is a deliberate hydrate before dispatch. Fetching
        one inside a job spends a booted worker on a download the preflight
        exists to make unnecessary."""
        with pytest.raises(modelroot.ModelUnavailable) as exc:
            modelroot.ensure("HUNYUAN_15_I2V_480_STEP", lambda mid: None)
        assert exc.value.code == "MODEL_NOT_CACHEABLE"

    def test_an_unknown_id_lists_every_known_one(self, monkeypatch):
        with pytest.raises(modelroot.ModelUnavailable) as exc:
            modelroot.model_dir("NOT_A_MODEL")
        assert exc.value.code == "MODEL_UNKNOWN"
        for known in modelroot.known_ids():
            assert known in exc.value.detail
