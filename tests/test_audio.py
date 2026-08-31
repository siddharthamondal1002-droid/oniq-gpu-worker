"""audio_mux — contract, gain math, the duration gate, and a REAL mux.

The mux and probe run for real on CPU (PyAV needs no GPU and no
subprocess); only piper itself is stubbed, through the same injection
pattern videogen uses for its pipeline. The CI workflow downloads the
sha256-pinned voice and sets ONIQ_TEST_VOICE_DIR, which arms the
real-synthesis tests at the bottom; locally they skip.
"""

import math
import os

import numpy as np
import pytest

import audio
import contract
import handler
import storage


# ----------------------------------------------------------------- fixtures
def make_video(path, frames=9, w=64, h=48, fps=24, with_audio=False):
    import av

    out = av.open(str(path), "w")
    vs = out.add_stream("libx264", rate=fps)
    vs.width, vs.height, vs.pix_fmt = w, h, "yuv420p"
    astream = None
    if with_audio:
        # Streams must all exist before the first packet is muxed.
        astream = out.add_stream("aac", rate=8000)
        astream.layout = "mono"
    for i in range(frames):
        img = np.full((h, w, 3), i * 20 % 255, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        for pkt in vs.encode(frame):
            out.mux(pkt)
    for pkt in vs.encode():
        out.mux(pkt)
    if astream is not None:
        tone = (np.sin(np.arange(4000) * 0.2) * 8000).astype(np.int16)
        af = av.AudioFrame.from_ndarray(tone[np.newaxis, :], format="s16", layout="mono")
        af.sample_rate = 8000
        af.pts = 0
        for pkt in astream.encode(af):
            out.mux(pkt)
        for pkt in astream.encode():
            out.mux(pkt)
    out.close()
    return str(path)


def fake_synth_seconds(seconds, rate=8000, amplitude=0.5):
    """A synth stub returning a sine 'voice' of a chosen length."""

    def synth(_text):
        n = int(seconds * rate)
        pcm = (np.sin(np.arange(n) * 0.05) * amplitude * 32767).astype(np.int16)
        return pcm, rate

    return synth


def audio_job(narration="A short line."):
    return {
        "op": "audio_mux",
        "input_key": "media/video/j1/ltx-001.mp4",
        "output_key": "media/video/j1/final-001.mp4",
        "params": {"narration": narration},
    }


# ----------------------------------------------------------------- contract
def test_contract_accepts_bounded_audio_mux():
    job = contract.validate_job(
        {
            "op": "audio_mux",
            "input_key": "media/video/x/ltx-001.mp4",
            "output_key": "media/video/x/final-001.mp4",
            "params": {"narration": "  He turns to face us.  "},
        }
    )
    assert job["op"] == "audio_mux"
    assert job["params"] == {"narration": "He turns to face us."}


def test_contract_refuses_unknown_audio_params():
    with pytest.raises(contract.ContractError) as err:
        contract.validate_job(
            {
                "op": "audio_mux",
                "input_key": "a/b.mp4",
                "output_key": "a/c.mp4",
                "params": {"narration": "hi", "voice": "Charon"},
            }
        )
    assert err.value.code == "invalid-input"
    assert "voice" in err.value.message


@pytest.mark.parametrize("narration", [None, "", "   ", 7])
def test_contract_refuses_missing_narration(narration):
    with pytest.raises(contract.ContractError):
        contract.validate_job(
            {
                "op": "audio_mux",
                "input_key": "a/b.mp4",
                "output_key": "a/c.mp4",
                "params": {} if narration is None else {"narration": narration},
            }
        )


def test_contract_bounds_narration_length():
    with pytest.raises(contract.ContractError) as err:
        contract.validate_job(
            {
                "op": "audio_mux",
                "input_key": "a/b.mp4",
                "output_key": "a/c.mp4",
                "params": {"narration": "x" * (contract.MAX_NARRATION_CHARS + 1)},
            }
        )
    assert str(contract.MAX_NARRATION_CHARS) in err.value.message


def test_video_generate_contract_is_untouched_by_audio():
    """Audio must not widen the video op.

    The surface is the prompt, the server-derived watermark entitlement
    (monetization resolution loop, 2026-08-27), and the two per-shot sampler
    inputs added by the quality work of 2026-08-31 — seed and negative_prompt,
    both of which the app DERIVES rather than letting a user type.

    The guard's point was never the number of fields; it is that AUDIO cannot
    reach a video job. Widening the assertion to an exact set keeps that: a
    narration field appearing here still fails, and so does any other name
    nobody has argued for.
    """
    assert contract._VIDEO_PARAM_FIELDS == frozenset(
        {"prompt", "watermark", "seed", "negative_prompt"}
    )
    assert "narration" not in contract._VIDEO_PARAM_FIELDS
    with pytest.raises(contract.ContractError):
        contract.validate_job(
            {
                "op": "video_generate",
                "input_key": "a/b.jpg",
                "output_key": "a/c.mp4",
                "params": {"prompt": "moves", "narration": "hi"},
            }
        )


# ---------------------------------------------------------------- gain math
def test_gain_limits_a_hot_signal_to_the_peak_ceiling():
    pcm = (np.sin(np.arange(8000) * 0.1) * 32767).astype(np.int16)
    gain, peak_db, _rms_db = audio.measure_gain(pcm)
    assert peak_db == pytest.approx(0.0, abs=0.1)
    out_peak_db = peak_db + 20 * math.log10(gain)
    assert out_peak_db <= audio.PEAK_CEILING_DB + 0.1


def test_gain_raises_a_quiet_signal_toward_rms_target_without_clipping():
    pcm = (np.sin(np.arange(8000) * 0.1) * 300).astype(np.int16)
    gain, peak_db, rms_db = audio.measure_gain(pcm)
    assert gain > 1.0
    assert peak_db + 20 * math.log10(gain) <= audio.PEAK_CEILING_DB + 0.1
    assert rms_db + 20 * math.log10(gain) <= audio.RMS_TARGET_DB + 0.1


def test_gain_on_silence_is_unity():
    gain, _p, _r = audio.measure_gain(np.zeros(1000, dtype=np.int16))
    assert gain == 1.0


def test_verdict_constants_mirror_the_shared_media_contract():
    """videoAudio.ts owns these numbers app-side; both ends must agree."""
    assert audio.AV_DRIFT_TOLERANCE_S == 0.25
    assert audio.SILENCE_FLOOR_DB == -60.0


# ------------------------------------------------------------------ run()
def test_run_refuses_garbage_input(tmp_path):
    src = tmp_path / "input.bin"
    src.write_bytes(b"not a video at all")
    with pytest.raises(contract.ContractError) as err:
        audio.run(audio_job(), str(src), str(tmp_path / "out.mp4"),
                  synth=fake_synth_seconds(0.1))
    assert err.value.code == "input-not-video"


def test_run_refuses_input_that_already_has_audio(tmp_path):
    src = make_video(tmp_path / "in.mp4", with_audio=True)
    with pytest.raises(contract.ContractError) as err:
        audio.run(audio_job(), src, str(tmp_path / "out.mp4"),
                  synth=fake_synth_seconds(0.1))
    assert err.value.code == "input-already-has-audio"


def test_run_refuses_narration_longer_than_the_video(tmp_path):
    src = make_video(tmp_path / "in.mp4", frames=9)  # 0.375 s
    out = tmp_path / "out.mp4"
    with pytest.raises(contract.ContractError) as err:
        audio.run(audio_job(), src, str(out), synth=fake_synth_seconds(2.0))
    assert err.value.code == "narration-too-long"
    assert "truncated" in err.value.message
    assert not out.exists(), "the mux must never run for an over-long narration"


def test_run_muxes_and_verifies_for_real(tmp_path):
    src = make_video(tmp_path / "in.mp4", frames=24)  # 1.0 s
    out = tmp_path / "out.mp4"
    metrics = audio.run(audio_job(), src, str(out), synth=fake_synth_seconds(0.5))

    assert metrics["has_audio"] is True
    assert metrics["narration_seconds"] == pytest.approx(0.5, abs=0.01)
    assert metrics["audio_sample_rate"] == 8000
    assert abs(metrics["audio_seconds"] - metrics["video_seconds"]) <= audio.AV_DRIFT_TOLERANCE_S
    assert metrics["audio_peak_dbfs"] > audio.SILENCE_FLOOR_DB
    assert metrics["output_bytes"] == os.path.getsize(out)
    assert metrics["tts_ms"] >= 0 and metrics["mux_ms"] >= 0

    probe = audio.probe_media(str(out))
    assert probe["video_streams"] == 1 and probe["audio_streams"] == 1

    evidence_keys = set(metrics) - {"video_seconds"}
    assert evidence_keys <= contract.OUTPUT_WHITELIST


def test_run_pads_short_narration_to_the_video_length(tmp_path):
    src = make_video(tmp_path / "in.mp4", frames=24)  # 1.0 s
    out = tmp_path / "out.mp4"
    metrics = audio.run(audio_job(), src, str(out), synth=fake_synth_seconds(0.2))
    assert metrics["audio_seconds"] == pytest.approx(metrics["video_seconds"], abs=0.25)


# ---------------------------------------------------------------- handler
AUDIO_EVENT = {
    "input": {
        "op": "audio_mux",
        "input_key": "media/video/j1/ltx-001.mp4",
        "output_key": "media/video/j1/final-001.mp4",
        "params": {"narration": "He turns."},
    }
}


def test_audio_event_routes_to_audio_with_an_mp4_path(monkeypatch):
    seen = {}

    def fake_download(key, dest, max_bytes=None):
        with open(dest, "wb") as fh:
            fh.write(b"mp4-bytes")
        return 9

    def fake_audio_run(job, input_path, output_path):
        seen["output_path"] = output_path
        with open(output_path, "wb") as fh:
            fh.write(b"final-bytes")
        return {
            "has_audio": True,
            "narration_seconds": 1.2,
            "audio_seconds": 4.04,
            "audio_sample_rate": 22050,
            "audio_peak_dbfs": -3.8,
            "audio_gain_db": -3.7,
            "tts_ms": 1700,
            "mux_ms": 45,
            "format": "mp4",
            "output_bytes": 11,
            "duration_ms": 1800,
        }

    monkeypatch.setattr(storage, "require_configured", lambda: None)
    monkeypatch.setattr(storage, "download", fake_download)
    monkeypatch.setattr(storage, "upload", lambda src, key: 11)
    monkeypatch.setattr(audio, "run", fake_audio_run)

    result = handler.handle(AUDIO_EVENT)
    assert result["ok"] is True
    assert result["op"] == "audio_mux"
    assert result["has_audio"] is True
    assert seen["output_path"].endswith("/output.mp4")
    assert set(result) <= contract.OUTPUT_WHITELIST


def test_audio_contract_error_surfaces_as_its_code(monkeypatch):
    monkeypatch.setattr(storage, "require_configured", lambda: None)
    monkeypatch.setattr(
        storage, "download",
        lambda key, dest, max_bytes=None: open(dest, "wb").write(b"x"),
    )

    def refuse(job, input_path, output_path):
        raise contract.ContractError("narration-too-long", "measured 9.0s vs 4.04s")

    monkeypatch.setattr(audio, "run", refuse)
    result = handler.handle(AUDIO_EVENT)
    assert result["ok"] is False
    assert result["code"] == "narration-too-long"


# ------------------------------------------------------------- real piper
VOICE_DIR = os.environ.get("ONIQ_TEST_VOICE_DIR")
needs_voice = pytest.mark.skipif(
    not VOICE_DIR, reason="ONIQ_TEST_VOICE_DIR not set (CI downloads the voice)"
)


@needs_voice
def test_real_piper_speaks_at_its_declared_rate():
    pcm, rate = audio.synthesize("The character turns to face us.", VOICE_DIR)
    assert rate == 22050
    seconds = len(pcm) / rate
    assert 0.5 < seconds < 5.0
    _gain, peak_db, _rms = audio.measure_gain(pcm)
    assert peak_db > audio.SILENCE_FLOOR_DB


@needs_voice
def test_real_piper_end_to_end_mux(tmp_path):
    src = make_video(tmp_path / "in.mp4", frames=97, w=64, h=48)  # 4.042 s
    out = tmp_path / "final.mp4"

    def real_synth(text):
        return audio.synthesize(text, VOICE_DIR)

    metrics = audio.run(
        audio_job("The character slowly turns towards the camera."),
        src,
        str(out),
        synth=real_synth,
    )
    assert metrics["has_audio"] is True
    assert metrics["audio_sample_rate"] == 22050
    assert 0.5 < metrics["narration_seconds"] < 4.042
    assert abs(metrics["audio_seconds"] - metrics["video_seconds"]) <= 0.25
    assert metrics["audio_peak_dbfs"] > -20
