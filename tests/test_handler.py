import json
import os

import pytest

import contract
import handler
import preprocess
import storage


GOOD_EVENT = {
    "input": {
        "op": "image_preprocess",
        "input_key": "in/a.png",
        "output_key": "out/a.jpeg",
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
