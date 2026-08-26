"""videogen on the CPU rig: everything except the CUDA pass itself.

The fake pipeline stands in for LTXImageToVideoPipeline, so these tests
prove the orchestration — canvas fit, server-constant sampler kwargs,
metric measurement, and a REAL h264 mp4 written through imageio — while
the CUDA-only refusal proves no CPU ever runs the real model.
"""

import os
from types import SimpleNamespace

import pytest
from PIL import Image

import contract
import videogen
from preprocess import GpuUnavailable


class FakePipe:
    """Records the sampler kwargs and returns real PIL frames."""

    def __init__(self, n_frames=9):
        self.n_frames = n_frames
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        frames = [
            Image.new(
                "RGB",
                (contract.VIDEO_WIDTH, contract.VIDEO_HEIGHT),
                (i * 20 % 255, 80, 160),
            )
            for i in range(self.n_frames)
        ]
        return SimpleNamespace(frames=[frames])


def _video_job(prompt="the subject blinks"):
    return contract.validate_job(
        {
            "op": "video_generate",
            "input_key": "in/a.jpg",
            "output_key": "out/a.mp4",
            "params": {"prompt": prompt},
        }
    )


@pytest.fixture
def paths(tmp_path):
    input_path = str(tmp_path / "input.jpg")
    Image.new("RGB", (900, 600), (200, 120, 40)).save(input_path, "JPEG")
    return input_path, str(tmp_path / "output.mp4")


def test_run_writes_a_real_mp4_and_measures(paths):
    input_path, output_path = paths
    pipe = FakePipe()
    metrics = videogen.run(
        _video_job(), input_path, output_path, load_pipeline=lambda: pipe
    )
    with open(output_path, "rb") as fh:
        head = fh.read(12)
    assert head[4:8] == b"ftyp"  # a real ISO-BMFF mp4, not renamed bytes
    assert metrics["output_bytes"] == os.path.getsize(output_path) > 0
    assert metrics["frames"] == pipe.n_frames
    assert metrics["fps"] == contract.VIDEO_FPS
    assert metrics["video_seconds"] == round(
        pipe.n_frames / contract.VIDEO_FPS, 2
    )
    assert metrics["width"] == contract.VIDEO_WIDTH
    assert metrics["height"] == contract.VIDEO_HEIGHT
    assert metrics["format"] == "mp4"
    for measured in ("model_load_ms", "inference_ms", "encode_ms", "duration_ms"):
        assert isinstance(metrics[measured], int)


def test_pipe_receives_only_server_decided_settings(paths, tmp_path, monkeypatch):
    monkeypatch.setattr(videogen, "MODEL_ID_FILE", str(tmp_path / "MODEL_ID"))
    input_path, output_path = paths
    pipe = FakePipe()
    videogen.run(
        _video_job(prompt="turns toward camera"),
        input_path,
        output_path,
        load_pipeline=lambda: pipe,
    )
    kwargs = pipe.calls[0]
    assert kwargs["prompt"] == "turns toward camera"
    assert kwargs["negative_prompt"] == videogen.NEGATIVE_PROMPT
    assert kwargs["width"] == contract.VIDEO_WIDTH
    assert kwargs["height"] == contract.VIDEO_HEIGHT
    assert kwargs["num_frames"] == contract.VIDEO_NUM_FRAMES
    # MODEL_ID absent on this rig -> not distilled -> the full step count.
    assert kwargs["num_inference_steps"] == videogen.STEPS_FULL
    assert kwargs["image"].size == (contract.VIDEO_WIDTH, contract.VIDEO_HEIGHT)


def test_distilled_model_id_selects_the_distilled_steps(paths, tmp_path, monkeypatch):
    marker = tmp_path / "MODEL_ID"
    marker.write_text("Lightricks/LTX-Video-0.9.7-distilled#distilled\n")
    monkeypatch.setattr(videogen, "MODEL_ID_FILE", str(marker))
    input_path, output_path = paths
    pipe = FakePipe()
    metrics = videogen.run(
        _video_job(), input_path, output_path, load_pipeline=lambda: pipe
    )
    assert pipe.calls[0]["num_inference_steps"] == videogen.STEPS_DISTILLED
    assert metrics["model"] == "Lightricks/LTX-Video-0.9.7-distilled#distilled"


def test_model_id_reads_missing_without_the_baked_file(tmp_path, monkeypatch):
    monkeypatch.setattr(videogen, "MODEL_ID_FILE", str(tmp_path / "absent"))
    assert videogen.model_id() == "missing"


def test_cuda_refused_even_with_the_cpu_fallback_env(paths, monkeypatch):
    # image_preprocess honors ONIQ_ALLOW_CPU_FALLBACK; video NEVER does —
    # a 2B diffusion pass on CPU blows the ceiling and proves nothing.
    monkeypatch.setenv("ONIQ_ALLOW_CPU_FALLBACK", "1")
    input_path, output_path = paths
    with pytest.raises(GpuUnavailable):
        videogen.run(_video_job(), input_path, output_path)
    assert not os.path.exists(output_path)


@pytest.mark.parametrize(
    "size", [(900, 600), (100, 400), (2000, 500), (704, 480), (33, 21)]
)
def test_fit_to_canvas_always_yields_the_video_canvas(size):
    fitted = videogen._fit_to_canvas(Image.new("RGB", size, (1, 2, 3)))
    assert fitted.size == (contract.VIDEO_WIDTH, contract.VIDEO_HEIGHT)


def test_frame_count_constant_satisfies_the_8k_plus_1_rule():
    # LTX generates 8k+1 frames; any other count gets silently adjusted
    # by the pipeline, which would falsify the measured video_seconds.
    assert contract.VIDEO_NUM_FRAMES % 8 == 1
