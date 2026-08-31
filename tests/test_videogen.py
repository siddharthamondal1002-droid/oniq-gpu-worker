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


def test_run_writes_a_real_mp4_and_measures(paths, fake_checkpoint):
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


def test_pipe_receives_only_server_decided_settings(paths, tmp_path, monkeypatch, fake_checkpoint):
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
    # The fake checkpoint ships a stock scheduler and a non-distilled name,
    # so the derived profile is the full one.
    import ltxcaps

    assert kwargs["num_inference_steps"] == ltxcaps.STEPS_FULL
    # Guidance was never sent before this change; the pipeline's default
    # applied silently and nothing recorded it.
    assert kwargs["guidance_scale"] == 3.0
    assert kwargs["guidance_rescale"] == 0.0
    assert kwargs["image"].size == (contract.VIDEO_WIDTH, contract.VIDEO_HEIGHT)


def test_a_distilled_checkpoint_selects_the_distilled_profile(
    paths, tmp_path, monkeypatch
):
    """Distillation is read from the CHECKPOINT, not from the repo name.

    The fake snapshot below ships a scheduler carrying its own short schedule
    — what a timestep-distilled release actually contains — and a name that
    agrees. Both signals point the same way, so the profile resolves.
    """
    import ltxcaps
    from conftest import write_fake_checkpoint

    root = write_fake_checkpoint(tmp_path / "ltx", distilled=True)
    marker = tmp_path / "MODEL_ID"
    marker.write_text("Lightricks/LTX-Video-0.9.7-distilled\n")
    monkeypatch.setattr(videogen, "MODEL_DIR", root)
    monkeypatch.setattr(videogen, "MODEL_ID_FILE", str(marker))

    input_path, output_path = paths
    pipe = FakePipe()
    metrics = videogen.run(
        _video_job(), input_path, output_path, load_pipeline=lambda: pipe
    )
    assert pipe.calls[0]["num_inference_steps"] == ltxcaps.STEPS_DISTILLED
    # A distilled checkpoint does not want classifier-free guidance.
    assert pipe.calls[0]["guidance_scale"] == 1.0
    assert metrics["distilled"] is True


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


def test_watermark_defaults_on_and_changes_the_frame(paths, fake_checkpoint):
    # The default job carries watermark=True; the corner pixels of the
    # encoded output must differ from an unmarked run of the SAME frames.
    input_path, output_path = paths
    pipe = FakePipe()
    marked = videogen.run(
        _video_job(), input_path, output_path, load_pipeline=lambda: pipe
    )
    assert marked["watermarked"] is True

    clean_path = output_path.replace(".mp4", ".clean.mp4")
    clean_job = contract.validate_job(
        {
            "op": "video_generate",
            "input_key": "in/a.jpg",
            "output_key": "out/a.mp4",
            "params": {"prompt": "the subject blinks", "watermark": False},
        }
    )
    clean = videogen.run(
        clean_job, input_path, clean_path, load_pipeline=lambda: FakePipe()
    )
    assert clean["watermarked"] is False

    import imageio.v3 as iio
    import numpy as np

    frame_marked = iio.imread(output_path, index=0, plugin="pyav")
    frame_clean = iio.imread(clean_path, index=0, plugin="pyav")
    h, w = frame_marked.shape[:2]
    corner_marked = frame_marked[h - 60 :, w - 160 :]
    corner_clean = frame_clean[h - 60 :, w - 160 :]
    # The mark lives in the bottom-right corner and nowhere else.
    assert np.abs(
        corner_marked.astype(int) - corner_clean.astype(int)
    ).max() > 20
    top_marked = frame_marked[: h // 2]
    top_clean = frame_clean[: h // 2]
    assert np.abs(top_marked.astype(int) - top_clean.astype(int)).max() <= 6


def test_watermark_frame_is_pure_and_preserves_size():
    import numpy as np

    frame = Image.new("RGB", (contract.VIDEO_WIDTH, contract.VIDEO_HEIGHT), (10, 10, 10))
    marked = videogen._watermark_frame(frame)
    assert marked.size == frame.size
    assert marked.mode == "RGB"
    original = np.asarray(frame)
    result = np.asarray(marked)
    # The original frame object is untouched, and the mark landed in the
    # bottom-right corner region — nowhere else.
    assert original.max() == 10
    corner = result[-70:, -180:]
    assert corner.max() > 60
    assert np.array_equal(result[: result.shape[0] // 2], original[: original.shape[0] // 2])


# ---------------------------------------------------------------- concat


def _tiny_clip(path, n_frames, shade):
    frames = [
        Image.new(
            "RGB",
            (contract.VIDEO_WIDTH, contract.VIDEO_HEIGHT),
            (shade, 90, 200 - shade),
        )
        for _ in range(n_frames)
    ]
    videogen._encode_mp4(frames, path)


def _concat_job_for(keys):
    return contract.validate_job(
        {
            "op": "video_concat",
            "input_key": keys[0],
            "output_key": "films/final.mp4",
            "params": {"segment_keys": list(keys)},
        }
    )


def test_concat_joins_ordered_segments_and_measures(tmp_path):
    a = str(tmp_path / "a.mp4")
    b = str(tmp_path / "b.mp4")
    c = str(tmp_path / "c.mp4")
    _tiny_clip(a, 9, 30)
    _tiny_clip(b, 17, 120)
    _tiny_clip(c, 9, 220)
    out = str(tmp_path / "final.mp4")
    metrics = videogen.run_concat(
        _concat_job_for(["k/a.mp4", "k/b.mp4", "k/c.mp4"]), [a, b, c], out
    )
    # Decoded, not trusted: every frame of the final must decode.
    assert metrics["segments"] == 3
    assert metrics["frames"] == 9 + 17 + 9
    assert metrics["video_seconds"] == round(35 / contract.VIDEO_FPS, 2)
    assert metrics["width"] == contract.VIDEO_WIDTH
    assert metrics["format"] == "mp4"
    assert metrics["output_bytes"] > 0
    with open(out, "rb") as fh:
        assert fh.read(12)[4:8] == b"ftyp"

    # And the frames arrive in declared order: first shade, then second.
    import imageio.v3 as iio

    first = iio.imread(out, index=0, plugin="pyav")
    later = iio.imread(out, index=12, plugin="pyav")
    assert abs(int(first[10, 10, 0]) - 30) < 20
    assert abs(int(later[10, 10, 0]) - 120) < 20


def test_concat_refuses_a_mismatched_canvas(tmp_path):
    a = str(tmp_path / "a.mp4")
    _tiny_clip(a, 9, 30)
    small = str(tmp_path / "small.mp4")
    frames = [Image.new("RGB", (352, 240), (50, 50, 50)) for _ in range(9)]
    import imageio.v2 as imageio

    writer = imageio.get_writer(small, fps=contract.VIDEO_FPS, codec="libx264")
    try:
        import numpy as np

        for f in frames:
            writer.append_data(np.asarray(f))
    finally:
        writer.close()

    with pytest.raises(videogen.ConcatRefused) as exc:
        videogen.run_concat(
            _concat_job_for(["k/a.mp4", "k/small.mp4"]),
            [a, small],
            str(tmp_path / "out.mp4"),
        )
    assert exc.value.code == "concat-dims-mismatch"


def test_frame_count_constant_satisfies_the_8k_plus_1_rule():
    # LTX generates 8k+1 frames; any other count gets silently adjusted
    # by the pipeline, which would falsify the measured video_seconds.
    assert contract.VIDEO_NUM_FRAMES % 8 == 1


# ------------------------------------------- the in-house image engine
# Same baked snapshot, text-to-video pipeline, frame 0 kept as the still.
# The CPU rig injects a fake pipeline, exactly as the video tests do, so
# everything except the CUDA pass itself is exercised here.


class _FakeTextPipe:
    """Records what the engine asked for and returns real PIL frames."""

    def __init__(self, n_frames=contract.IMAGE_GEN_NUM_FRAMES):
        self.calls = []
        self._n = n_frames
        self.vae = SimpleNamespace(enable_tiling=lambda: None)

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        frames = [
            Image.new("RGB", (kwargs["width"], kwargs["height"]),
                      (10 * i, 20, 30))
            for i in range(self._n)
        ]
        return SimpleNamespace(frames=[frames])


def _image_job():
    return contract.validate_job(
        {"op": "image_generate", "output_key": "out/still.png",
         "params": {"prompt": "a lantern in the rain"}}
    )


def test_image_engine_draws_from_the_prompt_at_the_video_canvas(tmp_path, fake_checkpoint):
    pipe = _FakeTextPipe()
    out = str(tmp_path / "still.png")
    metrics = videogen.run_image(_image_job(), out, load_pipeline=lambda: pipe)

    call = pipe.calls[0]
    assert call["prompt"] == "a lantern in the rain"
    assert "image" not in call  # text-to-video: there is no source frame
    assert call["width"] == contract.VIDEO_WIDTH
    assert call["height"] == contract.VIDEO_HEIGHT
    assert call["num_frames"] == contract.IMAGE_GEN_NUM_FRAMES
    assert metrics["width"] == contract.VIDEO_WIDTH
    assert metrics["height"] == contract.VIDEO_HEIGHT
    assert metrics["format"] == contract.IMAGE_GEN_FORMAT


def test_image_engine_keeps_frame_zero(tmp_path, fake_checkpoint):
    pipe = _FakeTextPipe()
    out = str(tmp_path / "still.png")
    videogen.run_image(_image_job(), out, load_pipeline=lambda: pipe)
    # Frame 0 of the fake is (0, 20, 30); frame 1 would be (10, 20, 30).
    assert Image.open(out).convert("RGB").getpixel((0, 0)) == (0, 20, 30)


def test_image_engine_writes_a_real_measured_artifact(tmp_path, fake_checkpoint):
    out = str(tmp_path / "still.png")
    metrics = videogen.run_image(
        _image_job(), out, load_pipeline=_FakeTextPipe
    )
    assert os.path.getsize(out) == metrics["output_bytes"] > 0
    assert metrics["ok"] is True and metrics["op"] == "image_generate"
    for measured in ("model_load_ms", "inference_ms", "encode_ms",
                     "duration_ms"):
        assert isinstance(metrics[measured], int)
    assert set(metrics) <= set(contract.OUTPUT_WHITELIST)


def test_image_engine_never_marks_the_conditioning_frame(tmp_path, fake_checkpoint):
    # A watermark here would be burned twice: once on the still and again
    # on the film that animates it.
    flat = _FakeTextPipe()
    out = str(tmp_path / "still.png")
    videogen.run_image(_image_job(), out, load_pipeline=lambda: flat)
    img = Image.open(out).convert("RGB")
    corner = img.crop((img.width - 90, img.height - 40, img.width, img.height))
    assert corner.getextrema() == ((0, 0), (20, 20), (30, 30))


def test_image_engine_refuses_cpu_fallback(tmp_path):
    with pytest.raises(GpuUnavailable):
        videogen.run_image(_image_job(), str(tmp_path / "x.png"))


def test_image_engine_refuses_an_empty_generation(tmp_path, fake_checkpoint):
    with pytest.raises(contract.ContractError) as exc:
        videogen.run_image(
            _image_job(), str(tmp_path / "x.png"),
            load_pipeline=lambda: _FakeTextPipe(n_frames=0),
        )
    assert exc.value.code == "no-frames"


class TestTheTextEncoderComesFromTheVolume:
    """The image no longer carries text_encoder/ (owner directive
    2026-08-31), so every pipeline load has to supply it."""

    def test_all_three_pipelines_are_handed_the_encoder(self):
        # Three from_pretrained sites: LTXConditionPipeline, the
        # LTXImageToVideoPipeline fallback, and LTXPipeline for stills. One of
        # them left un-supplied is an OSError on a rented card, in the one
        # code path that only runs when the primary has already failed.
        source = open("videogen.py", encoding="utf-8").read()
        loads = [l for l in source.splitlines() if ".from_pretrained(" in l
                 and "Pipeline" in l]
        assert len(loads) == 3, loads
        assert source.count("text_encoder=_text_encoder()") == 3

    def test_it_resolves_through_modelroot_and_never_from_the_image(self):
        source = open("videogen.py", encoding="utf-8").read()
        body = source.split("def _text_encoder():", 1)[1].split("\ndef ", 1)[0]
        assert "modelroot.resolve(LTX_TEXT_ENCODER)" in body
        # The baked pipeline directory must not be where it looks.
        assert "_model_dir()" not in body

    def test_the_refusal_is_not_swallowed(self, monkeypatch):
        # modelroot raises a NAMED refusal. If _text_encoder caught it and
        # returned None, diffusers would build a pipeline with no encoder and
        # the job would fail somewhere far less legible.
        import modelroot
        import videogen

        def refuse(_id):
            raise modelroot.ModelUnavailable("MODEL_NOT_HYDRATED", "not there")

        monkeypatch.setattr(modelroot, "resolve", refuse)
        with pytest.raises(modelroot.ModelUnavailable) as exc:
            videogen._text_encoder()
        assert exc.value.code == "MODEL_NOT_HYDRATED"

    def test_the_diffusers_contract_this_relies_on_is_recorded(self):
        # Passing a component is only safe because from_pretrained skips
        # loading it. That is quoted from diffusers 0.38.0's own
        # pipeline_utils.py rather than assumed, so a reader can check it.
        source = open("videogen.py", encoding="utf-8").read()
        assert "if name in passed_class_obj:" in source
        assert "loaded_sub_model = passed_class_obj[name]" in source
