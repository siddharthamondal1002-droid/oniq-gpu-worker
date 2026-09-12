import json
import os

import pytest

import contract
import handler
import preprocess
import storage
import videogen


GOOD_EVENT = {
    "input": {
        "op": "image_preprocess",
        "input_key": "in/a.png",
        "output_key": "out/a.jpeg",
    }
}

VIDEO_EVENT = {
    "input": {
        "op": "video_generate",
        "input_key": "in/a.jpg",
        "output_key": "out/clip.mp4",
        "params": {"prompt": "the subject blinks"},
    }
}


@pytest.fixture
def wired(monkeypatch):
    """Wire storage + preprocess with fakes; record the workdir used."""
    seen = {}

    def fake_download(key, dest, max_bytes=None):
        seen["workdir"] = os.path.dirname(dest)
        with open(dest, "wb") as fh:
            fh.write(b"png-bytes")
        return 9

    def fake_run(job, input_path, output_path):
        with open(output_path, "wb") as fh:
            fh.write(b"jpeg-bytes")
        return {
            "width": 320,
            "height": 240,
            "format": "jpeg",
            "output_bytes": 10,
            "duration_ms": 5,
            "device": "cuda",
            "gpu_name": "Fake GPU",
            "vram_total_mb": 24576,
            "vram_peak_mb": 300,
        }

    monkeypatch.setattr(storage, "require_configured", lambda: None)
    monkeypatch.setattr(storage, "download", fake_download)
    monkeypatch.setattr(storage, "upload", lambda src, key: 10)
    monkeypatch.setattr(preprocess, "run", fake_run)
    return seen


def test_happy_path_returns_whitelisted_result(wired):
    result = handler.handle(GOOD_EVENT)
    assert result["ok"] is True
    assert result["op"] == "image_preprocess"
    assert result["output_key"] == "out/a.jpeg"
    assert result["device"] == "cuda"
    assert set(result) <= contract.OUTPUT_WHITELIST


def test_happy_path_cleans_up_workdir(wired):
    handler.handle(GOOD_EVENT)
    assert wired["workdir"].startswith("/")
    assert not os.path.exists(wired["workdir"])


def test_video_event_routes_to_videogen_with_an_mp4_path(wired, monkeypatch):
    seen = {}

    def fake_videogen_run(job, input_path, output_path):
        seen["output_path"] = output_path
        seen["prompt"] = job["params"]["prompt"]
        with open(output_path, "wb") as fh:
            fh.write(b"mp4-bytes")
        return {
            "model": "Lightricks/LTX-Video-0.9.7-distilled#distilled",
            "model_load_ms": 41000,
            "inference_ms": 95000,
            "encode_ms": 3500,
            "frames": 97,
            "fps": 24,
            "video_seconds": 4.04,
            "width": 704,
            "height": 480,
            "format": "mp4",
            "output_bytes": 9,
            "duration_ms": 140000,
            "device": "cuda",
            "gpu_name": "NVIDIA GeForce RTX 3090",
            "vram_total_mb": 24576,
            "vram_peak_mb": 9000,
        }

    monkeypatch.setattr(videogen, "run", fake_videogen_run)
    result = handler.handle(VIDEO_EVENT)
    assert result["ok"] is True
    assert result["op"] == "video_generate"
    assert seen["output_path"].endswith("/output.mp4")
    assert seen["prompt"] == "the subject blinks"
    assert result["model"].endswith("#distilled")
    assert result["frames"] == 97
    assert set(result) <= contract.OUTPUT_WHITELIST


def test_video_cuda_refusal_surfaces_as_its_code(wired, monkeypatch):
    def refusing_run(job, input_path, output_path):
        raise preprocess.GpuUnavailable(
            "video_generate requires CUDA; there is no CPU fallback"
        )

    monkeypatch.setattr(videogen, "run", refusing_run)
    result = handler.handle(VIDEO_EVENT)
    assert result["ok"] is False
    assert result["code"] == "cuda-unavailable"


def test_invalid_input_refused_before_any_io():
    result = handler.handle({"input": {"op": "rm -rf"}})
    assert result["ok"] is False
    assert result["code"] == "op-not-allowed"


def test_event_not_a_dict():
    result = handler.handle(None)
    assert result["ok"] is False
    assert result["code"] == "invalid-input"


def test_storage_not_configured_fails_closed_naming_vars():
    result = handler.handle(GOOD_EVENT)
    assert result["ok"] is False
    assert result["code"] == "storage-not-configured"
    assert "R2_S3_ENDPOINT" in result["error"]


def test_r2_read_failure(wired, monkeypatch):
    def failing_download(key, dest, max_bytes=None):
        raise storage.StorageError("r2-read-failed", "could not read")

    monkeypatch.setattr(storage, "download", failing_download)
    result = handler.handle(GOOD_EVENT)
    assert result["code"] == "r2-read-failed"


def test_r2_write_failure_still_cleans_up(wired, monkeypatch):
    def failing_upload(src, key):
        raise storage.StorageError("r2-write-failed", "could not write")

    monkeypatch.setattr(storage, "upload", failing_upload)
    result = handler.handle(GOOD_EVENT)
    assert result["code"] == "r2-write-failed"
    assert not os.path.exists(wired["workdir"])


def test_cuda_unavailable_surfaces_as_its_code(wired, monkeypatch):
    def refusing_run(job, input_path, output_path):
        raise preprocess.GpuUnavailable()

    monkeypatch.setattr(preprocess, "run", refusing_run)
    result = handler.handle(GOOD_EVENT)
    assert result["code"] == "cuda-unavailable"


def test_unexpected_exception_never_leaks_its_text(wired, monkeypatch):
    def exploding_run(job, input_path, output_path):
        raise RuntimeError("SECRET-TOKEN-VALUE-12345")

    monkeypatch.setattr(preprocess, "run", exploding_run)
    result = handler.handle(GOOD_EVENT)
    assert result["ok"] is False
    assert result["code"] == "unexpected-exception"
    assert "SECRET-TOKEN-VALUE-12345" not in json.dumps(result)
    assert result["error"] == "RuntimeError"


def test_unexpected_exception_still_cleans_up(wired, monkeypatch):
    monkeypatch.setattr(
        preprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("x")),
    )
    handler.handle(GOOD_EVENT)
    assert not os.path.exists(wired["workdir"])


def test_runtime_ceiling_enforced(wired, monkeypatch):
    ticks = iter([0.0, 0.0, 5000.0, 6000.0, 7000.0, 8000.0])
    monkeypatch.setattr(handler.time, "monotonic", lambda: next(ticks))
    result = handler.handle(GOOD_EVENT)
    assert result["ok"] is False
    assert result["code"] == "runtime-exceeded"


def test_cleanup_is_deterministic_object():
    c = handler.Cleanup(None)
    assert c.run() == {"ok": True}


def test_cleanup_reports_failure_without_raising(tmp_path, monkeypatch):
    target = tmp_path / "d"
    target.mkdir()
    monkeypatch.setattr(
        handler.shutil,
        "rmtree",
        lambda p: (_ for _ in ()).throw(OSError("busy")),
    )
    out = handler.Cleanup(str(target)).run()
    assert out["ok"] is False
    assert out["error"] == "OSError"


def test_result_never_contains_unwhitelisted_keys(wired, monkeypatch):
    def leaky_run(job, input_path, output_path):
        with open(output_path, "wb") as fh:
            fh.write(b"x")
        return {"device": "cuda", "internal_path": "/tmp/x", "width": 1,
                "height": 1, "format": "jpeg", "output_bytes": 1,
                "duration_ms": 1, "gpu_name": "g", "vram_total_mb": 1,
                "vram_peak_mb": 1}

    monkeypatch.setattr(preprocess, "run", leaky_run)
    result = handler.handle(GOOD_EVENT)
    assert "internal_path" not in result


WORKER_FILES = (
    "contract.py",
    "preprocess.py",
    "storage.py",
    "videogen.py",
    "audio.py",
    "handler.py",
)


def _source(name):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, name), encoding="utf-8") as fh:
        return fh.read()


@pytest.mark.parametrize("forbidden", [
    "eval(", "exec(", "subprocess", "os.system", "pickle", "shell=True",
])
def test_no_shell_surface_in_worker_files(forbidden):
    for name in WORKER_FILES:
        assert forbidden not in _source(name), f"{forbidden} in {name}"


def test_serverless_start_is_main_guarded():
    src = _source("handler.py")
    assert 'if __name__ == "__main__":' in src
    guard_pos = src.index('if __name__ == "__main__":')
    assert src.index("import runpod") > guard_pos
    assert "runpod.serverless.start" in src


def test_the_probe_ceiling_applies_to_model_probe_only():
    """Owner directive 2026-08-29: do not alter production. A benchmark that
    downloads its checkpoint at job time needs a wider window than a job whose
    weights are already in the image — and giving it one must not give one to
    anything else."""
    assert handler._ceiling("model_probe") == contract.PROBE_RUNTIME_CEILING_SECONDS
    for op in ("image_preprocess", "video_generate", "image_generate",
               "audio_mux", "video_concat", "story_generate", ""):
        assert handler._ceiling(op) == contract.RUNTIME_CEILING_SECONDS, op


def test_a_probe_inside_the_probe_ceiling_is_not_a_runtime_error(monkeypatch):
    """900s would have killed every candidate after paying for it in full:
    the deadline is checked AFTER the work returns, so the window is billed,
    the clip is discarded, and the measurements go with it."""
    over_production = contract.RUNTIME_CEILING_SECONDS + 10
    monkeypatch.setattr(handler.time, "monotonic", lambda: over_production)
    handler._check_deadline(0.0, "model_probe")
    with pytest.raises(contract.ContractError) as exc:
        handler._check_deadline(0.0, "video_generate")
    assert exc.value.code == "runtime-exceeded"


def test_a_probe_downloads_its_reference_before_it_fetches_any_weights(monkeypatch):
    """This ordering is what makes a missing reference cheap. Reversed, a
    probe would fetch 44-118 GiB of weights and only then discover there is
    nothing to condition on — the whole rented window spent to learn what one
    R2 GET knew. require_reference's fallback rests on this, so it is asserted
    rather than read."""
    import modelprobe

    called = []
    monkeypatch.setattr(handler.storage, "require_configured", lambda: None)

    def refuse_download(key, path, limit):
        called.append(("download", key))
        raise storage.StorageError("r2-read-failed", "no such key")

    monkeypatch.setattr(handler.storage, "download", refuse_download)
    monkeypatch.setattr(
        modelprobe, "run",
        lambda *a, **k: called.append(("probe", None)) or {},
    )
    result = handler.handle({"input": {
        "op": "model_probe", "model": "cogvideox-i2v",
        "input_key": "validation/out/probe-reference.png",
        "output_key": "validation/out/probe-cogvideox-i2v.mp4",
        "params": {"prompt": "the woman turns toward the camera"},
    }})
    assert result["ok"] is False
    assert [name for name, _ in called] == ["download"]


# ---------------------------------------------------------------------------
# ONIQ's own refusals must carry their message (2026-09-12)
#
# On 2026-09-12 the first still of Story job a7b9c3b9 came back three times
# as `engine job FAILED: CheckpointInconsistent` — a bare class name with
# five possible causes behind it, none of them recoverable from outside the
# container or from the worker's log. These tests pin the two halves of the
# fix: ONIQ's own refusals say what happened, and a dependency's exception
# still says only its class.
# ---------------------------------------------------------------------------

import ltxcaps
import modelhydrate
import modelroot
import storygen
import weights_r2


def test_checkpoint_inconsistent_reports_which_of_its_five_causes(wired, monkeypatch):
    """The failure that cost 2026-09-12. `ltxcaps` raises this with five
    distinct messages; before the fix all five read `CheckpointInconsistent`."""
    def boom(key, dest, max_bytes=None):
        raise ltxcaps.CheckpointInconsistent(
            "distillation evidence disagrees: "
            "{'name_says_distilled': True, 'scheduler_says_distilled': False}"
            " — refusing to choose a sampler profile"
        )

    monkeypatch.setattr(storage, "download", boom)
    result = handler.handle(GOOD_EVENT)

    assert result["ok"] is False
    assert result["code"] == "checkpoint-inconsistent"
    assert "distillation evidence disagrees" in result["error"]
    assert result["error"] != "CheckpointInconsistent"


@pytest.mark.parametrize("exc,code,needle", [
    (ltxcaps.CheckpointInconsistent("model_index.json missing or unreadable"),
     "checkpoint-inconsistent", "model_index.json"),
    (weights_r2.WeightsUnavailable("not-staged", "could not read the manifest"),
     "not-staged", "manifest"),
    (modelroot.ModelUnavailable("MODEL_UNKNOWN", "'x' is not a known model"),
     "MODEL_UNKNOWN", "not a known model"),
    (modelhydrate.HydrationRefused("DISK_INSUFFICIENT", "37.82 GiB free, need 39.48"),
     "DISK_INSUFFICIENT", "37.82 GiB free"),
    (videogen.ReferenceUnsupported("the reference could not be honoured"),
     "reference-unsupported", "could not be honoured"),
    (videogen.OutOfMemory("the canvas did not fit"),
     "out-of-memory", "did not fit"),
])
def test_every_shipped_refusal_class_reports_its_own_diagnosis(exc, code, needle):
    """All six were shipped, none was named in the except-chain, and every
    one of them therefore arrived as its class name alone. Two of these
    classes say in their own docstring that "`code` is the whole diagnosis"
    — and the code was exactly what was being dropped."""
    own = handler._own_refusal(exc)
    assert own is not None, f"{type(exc).__name__} is not recognised as ONIQ's own"
    assert own[0] == code
    assert needle in own[1]


def test_a_dependency_exception_still_says_only_its_class():
    """THE RULE THAT IS NOT BEING RELAXED. Text from a module nobody here
    wrote never reaches the caller, because it could carry anything."""
    leaky = ValueError("Bearer sk-live-0000000000 leaked from a dependency")
    assert handler._own_refusal(leaky) is None
    assert handler._foreign_detail(leaky) == "ValueError"
    assert "sk-live" not in handler._foreign_detail(leaky)


def test_a_dependency_exception_reaches_the_caller_unchanged(wired, monkeypatch):
    def boom(key, dest, max_bytes=None):
        raise ValueError("Bearer sk-live-0000000000 leaked from a dependency")

    monkeypatch.setattr(storage, "download", boom)
    result = handler.handle(GOOD_EVENT)

    assert result["code"] == "unexpected-exception"
    assert result["error"] == "ValueError"
    assert "sk-live" not in result["error"]


def test_permission_error_names_the_path_it_could_not_write():
    """2026-09-12 attempt 1 of that same still was `PermissionError` and
    nothing else. errno and filename are STRUCTURED fields the OS sets,
    not a dependency's prose, and the path is the whole diagnosis."""
    exc = PermissionError(13, "Permission denied")
    exc.filename = "/app/cache/models/oniq/ltx/LTX-Video-0.9.7-distilled"
    detail = handler._foreign_detail(exc)

    assert "PermissionError" in detail
    assert "errno=13" in detail
    assert "/app/cache/models/oniq/ltx" in detail


def test_an_oserror_without_errno_or_path_still_names_its_class():
    assert handler._foreign_detail(OSError()) == "OSError"


def test_the_detail_is_bounded():
    exc = ltxcaps.CheckpointInconsistent("x" * 5000)
    code, detail = handler._own_refusal(exc)
    assert len(detail) == handler.MAX_REFUSAL_DETAIL


def test_a_refusal_defined_in_a_test_is_not_mistaken_for_oniqs_own():
    """`_is_own_exception` keys on the DIRECTORY the class was defined in,
    so this class — defined in tests/ — must not be recognised. Without
    that, the guarantee would be 'anything importable', which is every
    dependency in the image."""
    class NotOurs(RuntimeError):
        pass

    assert handler._own_refusal(NotOurs("secret")) is None
    assert handler._foreign_detail(NotOurs("secret")) == "NotOurs"


def test_the_code_is_derived_from_the_class_name():
    assert handler._code_for(ltxcaps.CheckpointInconsistent("x")) == "checkpoint-inconsistent"
    assert handler._code_for(videogen.OutOfMemory("x")) == "out-of-memory"


def test_every_exception_class_the_worker_ships_is_reportable():
    """THE GUARANTEE THAT CANNOT DRIFT, and the reason this is not a list.

    Six refusal classes were added after the except-chain was written and
    not one was added to it. This walks the modules the Dockerfile actually
    COPYs, finds every exception class defined in them, and requires each to
    be recognised as ONIQ's own with a non-empty code and detail — so the
    seventh is covered on the day it is written rather than on the day it
    costs a job.
    """
    import ast
    import importlib

    shipped = []
    for line in _source("Dockerfile").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "COPY" and parts[1].endswith(".py"):
            shipped.append(parts[1])
    assert "ltxcaps.py" in shipped, "the COPY parser found no ltxcaps — it is wrong"

    checked = []
    for name in shipped:
        tree = ast.parse(_source(name))
        module = importlib.import_module(name[:-3])
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            cls = getattr(module, node.name, None)
            if not isinstance(cls, type) or not issubclass(cls, BaseException):
                continue
            try:
                instance = cls("a-code", "a detail")
            except TypeError:
                instance = cls("a detail")
            own = handler._own_refusal(instance)
            assert own is not None, f"{name}:{node.name} is not recognised as ONIQ's own"
            assert own[0], f"{name}:{node.name} reports an empty code"
            assert own[1], f"{name}:{node.name} reports an empty detail"
            checked.append(f"{name}:{node.name}")

    # Measured 2026-09-12: 13 classes across the 14 shipped modules.
    assert len(checked) >= 13, f"the walk found only {checked}"


@pytest.mark.parametrize("exc,code", [
    (contract.ContractError("invalid-input", "prompt exceeds 1000 characters"),
     "invalid-input"),
    (storage.StorageNotConfigured(["R2_BUCKET"]), "storage-not-configured"),
    (storage.StorageError("r2-read-failed", "the read failed"), "r2-read-failed"),
    (preprocess.GpuUnavailable("CUDA is not available"), "cuda-unavailable"),
    (videogen.ConcatRefused("concat-refused", "clips disagree"), "concat-refused"),
])
def test_the_codes_the_app_already_reads_are_unchanged(wired, monkeypatch, exc, code):
    """The six named clauses run BEFORE the new fallback and must keep the
    exact codes they had. `oniqImage.ts` classifies retryability off this
    text, so a changed code is a changed retry decision in production —
    and `StoryModelUnavailable` in particular maps to `local-model-unavailable`,
    which is NOT what deriving it from the class name would produce."""
    def boom(key, dest, max_bytes=None):
        raise exc

    monkeypatch.setattr(storage, "download", boom)
    assert handler.handle(GOOD_EVENT)["code"] == code


def test_story_model_unavailable_keeps_its_hand_written_code():
    """Deriving from the class name would give `story-model-unavailable`.
    The clause above the fallback gives `local-model-unavailable`, and that
    is the one the app has always seen."""
    exc = storygen.StoryModelUnavailable("no checkpoint")
    assert handler._code_for(exc) == "story-model-unavailable"
    assert 'return _error("local-model-unavailable", str(exc))' in _source("handler.py")
