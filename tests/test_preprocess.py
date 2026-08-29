import sys
import types

import pytest
from PIL import Image

import contract
import preprocess


def _fake_torch(available: bool):
    torch = types.ModuleType("torch")
    cuda = types.SimpleNamespace(
        is_available=lambda: available,
        reset_peak_memory_stats=lambda: None,
        get_device_name=lambda idx: "Fake GPU",
        get_device_properties=lambda idx: types.SimpleNamespace(
            total_memory=24 * 1024**3
        ),
        max_memory_allocated=lambda: 512 * 1024**2,
    )
    torch.cuda = cuda
    return torch


def test_default_requires_cuda_when_torch_missing(monkeypatch):
    # Simulate the §16g rig: torch not installed at all. A None entry in
    # sys.modules makes `import torch` raise ImportError deterministically,
    # even on a machine that does have torch.
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(preprocess.GpuUnavailable) as exc:
        preprocess.run_gpu_op(lambda t: 1, lambda: 2)
    assert exc.value.code == "cuda-unavailable"


def test_torch_missing_with_optin_uses_cpu_path(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setenv("ONIQ_ALLOW_CPU_FALLBACK", "1")
    result, metrics = preprocess.run_gpu_op(lambda t: "gpu", lambda: "cpu")
    assert result == "cpu"
    assert metrics["device"] == "cpu"


def test_default_requires_cuda_when_cuda_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(False))
    with pytest.raises(preprocess.GpuUnavailable):
        preprocess.run_gpu_op(lambda t: 1, lambda: 2)


def test_explicit_env_optin_allows_cpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(False))
    monkeypatch.setenv("ONIQ_ALLOW_CPU_FALLBACK", "1")
    result, metrics = preprocess.run_gpu_op(lambda t: "gpu", lambda: "cpu")
    assert result == "cpu"
    assert metrics["device"] == "cpu"
    assert metrics["gpu_name"] is None


def test_env_value_other_than_1_does_not_opt_in(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(False))
    monkeypatch.setenv("ONIQ_ALLOW_CPU_FALLBACK", "true")
    with pytest.raises(preprocess.GpuUnavailable):
        preprocess.run_gpu_op(lambda t: 1, lambda: 2)


def test_cuda_path_reports_device_and_vram(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True))
    result, metrics = preprocess.run_gpu_op(lambda t: "gpu", lambda: "cpu")
    assert result == "gpu"
    assert metrics["device"] == "cuda"
    assert metrics["gpu_name"] == "Fake GPU"
    assert metrics["vram_total_mb"] == 24 * 1024
    assert metrics["vram_peak_mb"] == 512


def test_require_cuda_false_still_prefers_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True))
    _, metrics = preprocess.run_gpu_op(
        lambda t: 1, lambda: 2, require_cuda=False
    )
    assert metrics["device"] == "cuda"


def _write_png(path, size=(64, 48)):
    Image.new("RGB", size, (10, 200, 30)).save(path, format="PNG")


def test_run_end_to_end_on_cpu_fallback(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(False))
    monkeypatch.setenv("ONIQ_ALLOW_CPU_FALLBACK", "1")
    src = tmp_path / "in.png"
    dst = tmp_path / "out.jpeg"
    _write_png(src, (640, 480))
    job = contract.validate_job(
        {
            "op": "image_preprocess",
            "input_key": "in/a.png",
            "output_key": "out/a.jpeg",
            "params": {"target_max_dim": 320},
        }
    )
    metrics = preprocess.run(job, str(src), str(dst))
    assert metrics["width"] == 320 and metrics["height"] == 240
    assert metrics["format"] == "jpeg"
    assert metrics["device"] == "cpu"
    assert metrics["output_bytes"] > 0
    assert dst.exists()


def test_run_never_upscales(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(False))
    monkeypatch.setenv("ONIQ_ALLOW_CPU_FALLBACK", "1")
    src = tmp_path / "in.png"
    dst = tmp_path / "out.jpeg"
    _write_png(src, (100, 60))
    job = contract.validate_job(
        {
            "op": "image_preprocess",
            "input_key": "k",
            "output_key": "o",
            "params": {"target_max_dim": 4096},
        }
    )
    metrics = preprocess.run(job, str(src), str(dst))
    assert (metrics["width"], metrics["height"]) == (100, 60)


def test_undecodable_input_is_invalid_image(tmp_path):
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"not an image at all")
    with pytest.raises(contract.ContractError) as exc:
        preprocess._decode(str(bad))
    assert exc.value.code == "invalid-image"


def test_decode_pixel_bound_enforced(tmp_path, monkeypatch):
    src = tmp_path / "big.png"
    _write_png(src, (200, 200))
    monkeypatch.setattr(contract, "MAX_IMAGE_PIXELS", 100)
    # _decode copies the bound into PIL's PROCESS-WIDE Image.MAX_IMAGE_PIXELS,
    # which monkeypatch cannot know about and so cannot restore. Left at 100
    # it silently turns every later test's ordinary image into a
    # "decompression bomb" — measured 2026-08-29, when it began failing
    # tests in an unrelated module that merely open a 64x48 JPEG.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", Image.MAX_IMAGE_PIXELS)
    with pytest.raises(contract.ContractError) as exc:
        preprocess._decode(str(src))
    assert exc.value.code == "input-too-large"


def test_gpu_op_default_is_require_cuda():
    import inspect

    sig = inspect.signature(preprocess.run_gpu_op)
    assert sig.parameters["require_cuda"].default is True
