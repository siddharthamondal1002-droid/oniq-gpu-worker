"""The audio_mux workload — narration onto an already-generated video.

In-process only, like everything in this worker: piper (onnxruntime on
the CPU) speaks the narration, PyAV muxes it under the video. No process
is ever spawned and there is no shell here, by the same construction and
the same CI scan as the rest of the worker. The video stream is COPIED
packet by packet — the pixels this worker already generated and verified
are never re-encoded — and an AAC track is added next to them.

Voice policy is a SERVER decision baked at build time, exactly like the
video model: /app/models/piper holds the en-us-ryan-high voice (the same
sha256-pinned release asset the ONIQ story worker's in-house engine
runs), and the caller's only degree of freedom is the narration text.
No voice field, no rate field, no format field.

THE DURATION RULE, inherited from the Story pipeline's hardest lesson:
the narration is MEASURED after synthesis and the job is REFUSED, not
truncated, when it exceeds the video. Silently cutting a voice mid-word
and silently stretching a video are both banned; the caller is told the
measured number instead. Shorter narration is padded with real silence
to exactly the video's length, so the two streams always agree.

LOUDNESS, documented rather than hand-waved: piper output arrives near
0 dBFS peak. The mix normalizes to an RMS target of -20 dBFS with a hard
peak ceiling of -1.5 dBFS (gain is the smaller of the two corrections,
never an amplification past the ceiling), then the OUTPUT is decoded and
probed: an "audio track" quieter than -60 dBFS is silence with extra
steps and fails the job, and audio/video durations further apart than
0.25 s fail it too. Those two floors mirror ONIQ's shared media verdict
(supabase/functions/_shared/videoAudio.ts) so both ends of the wire
refuse the same artifacts.
"""

from __future__ import annotations

import math
import os
import time

import numpy as np

import contract
import modelroot

# Lazy, like videogen/storygen: resolved per call so a hydration that
# lands mid-life is seen. None means "ask modelroot".
VOICE_DIR = None


def _voice_dir() -> str:
    return VOICE_DIR or modelroot.resolve("piper")
VOICE_MODEL = "en-us-ryan-high.onnx"

# The documented loudness strategy (RMS target with a true-peak-ish
# ceiling; single pass, deterministic given the samples).
RMS_TARGET_DB = -20.0
PEAK_CEILING_DB = -1.5

# Mirrors of the app-side verdict constants in videoAudio.ts — the mux
# refuses here what the application would refuse on arrival.
AV_DRIFT_TOLERANCE_S = 0.25
SILENCE_FLOOR_DB = -60.0

AAC_BITRATE = 96_000
_ENCODE_CHUNK = 1024  # aac frame size; one frame per encode call


def voice_paths(voice_dir: str | None = None):
    voice_dir = voice_dir or _voice_dir()
    model = os.path.join(voice_dir, VOICE_MODEL)
    return model, model + ".json"


def load_voice(voice_dir: str | None = None):
    voice_dir = voice_dir or _voice_dir()
    """Load the baked piper voice. Never touches the network."""
    model, config = voice_paths(voice_dir)
    if not (os.path.exists(model) and os.path.exists(config)):
        raise contract.ContractError(
            "voice-not-baked",
            "the piper voice is not present in this image; audio_mux "
            "requires the media build",
        )
    from piper import PiperVoice

    return PiperVoice.load(model, config_path=config)


def synthesize(text: str, voice_dir: str | None = None):
    voice_dir = voice_dir or _voice_dir()
    """Speak one narration in-process. Returns (int16 mono pcm, rate)."""
    voice = load_voice(voice_dir)
    chunks = b"".join(voice.synthesize_stream_raw(text))
    pcm = np.frombuffer(chunks, dtype=np.int16)
    if pcm.size == 0:
        raise contract.ContractError(
            "narration-empty", "the voice produced no audio for that text"
        )
    return pcm, int(voice.config.sample_rate)


def measure_gain(pcm: np.ndarray) -> tuple[float, float, float]:
    """The normalization gain for one pcm buffer, plus what was measured.

    Returns (linear_gain, in_peak_db, in_rms_db). Pure math, no I/O —
    the unit tests own every branch. The gain is the SMALLER of the two
    corrections so the peak ceiling always binds last, and silence gets
    unity gain rather than an infinite one.
    """
    x = pcm.astype(np.float64) / 32768.0
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
    if peak <= 0.0 or rms <= 0.0:
        return 1.0, -math.inf, -math.inf
    peak_gain = (10.0 ** (PEAK_CEILING_DB / 20.0)) / peak
    rms_gain = (10.0 ** (RMS_TARGET_DB / 20.0)) / rms
    gain = min(peak_gain, rms_gain)
    return gain, 20.0 * math.log10(peak), 20.0 * math.log10(rms)


def _dbfs(value: float) -> float:
    return 20.0 * math.log10(value if value > 0 else 1e-9)


def probe_media(path: str) -> dict:
    """Streams and durations of one container, plus the audio peak.

    The probe DECODES the audio rather than trusting headers — a track
    present at -91 dB is silence with extra steps, and only the samples
    can say so.
    """
    import av

    try:
        container = av.open(path)
    except Exception as exc:
        raise contract.ContractError(
            "input-not-video", f"could not open media: {type(exc).__name__}"
        ) from exc
    try:
        video = [s for s in container.streams if s.type == "video"]
        audio = [s for s in container.streams if s.type == "audio"]
        out = {
            "video_streams": len(video),
            "audio_streams": len(audio),
            "video_seconds": 0.0,
            "audio_seconds": 0.0,
            "audio_peak_dbfs": None,
            "audio_sample_rate": None,
        }
        if video:
            v = video[0]
            if v.duration is not None:
                out["video_seconds"] = float(v.duration * v.time_base)
            elif container.duration is not None:
                out["video_seconds"] = container.duration / 1_000_000.0
        if audio:
            a = audio[0]
            out["audio_sample_rate"] = int(a.codec_context.sample_rate or 0)
            if a.duration is not None:
                out["audio_seconds"] = float(a.duration * a.time_base)
            peak = 0.0
            for frame in container.decode(a):
                arr = frame.to_ndarray()
                if arr.size:
                    m = float(np.max(np.abs(arr.astype(np.float64))))
                    if arr.dtype.kind == "i":
                        m /= 32768.0
                    peak = max(peak, m)
            out["audio_peak_dbfs"] = round(_dbfs(peak), 1)
        return out
    finally:
        container.close()


def mux(video_path: str, pcm16: np.ndarray, rate: int, output_path: str) -> None:
    """Copy the video stream, add one AAC track. Deterministic settings."""
    import av

    src = av.open(video_path)
    try:
        in_video = src.streams.video[0]
        dst = av.open(output_path, "w")
        try:
            out_video = dst.add_stream(template=in_video)
            out_audio = dst.add_stream("aac", rate=rate)
            out_audio.layout = "mono"
            out_audio.bit_rate = AAC_BITRATE

            for start in range(0, len(pcm16), _ENCODE_CHUNK):
                segment = pcm16[start : start + _ENCODE_CHUNK]
                frame = av.AudioFrame.from_ndarray(
                    segment[np.newaxis, :], format="s16", layout="mono"
                )
                frame.sample_rate = rate
                frame.pts = start
                for packet in out_audio.encode(frame):
                    dst.mux(packet)
            for packet in out_audio.encode():
                dst.mux(packet)

            for packet in src.demux(in_video):
                if packet.dts is None:
                    continue
                packet.stream = out_video
                dst.mux(packet)
        finally:
            dst.close()
    finally:
        src.close()


def run(job: dict, input_path: str, output_path: str, synth=None) -> dict:
    """Probe, speak, gate, normalize, mux, verify. Measured metrics only.

    `synth` exists for the CPU test rig: injecting a fake (pcm, rate)
    source exercises everything here except piper itself, exactly the
    `load_pipeline` pattern videogen uses.
    """
    started = time.monotonic()

    source = probe_media(input_path)
    if source["video_streams"] < 1 or not source["video_seconds"] > 0:
        raise contract.ContractError(
            "input-not-video", "audio_mux input has no playable video stream"
        )
    if source["audio_streams"] > 0:
        raise contract.ContractError(
            "input-already-has-audio",
            "audio_mux input already carries an audio stream; refusing to "
            "stack or replace a voice that is already there",
        )
    video_seconds = source["video_seconds"]

    tts_started = time.monotonic()
    if synth is None:
        pcm, rate = synthesize(job["params"]["narration"])
    else:
        pcm, rate = synth(job["params"]["narration"])
    tts_ms = int((time.monotonic() - tts_started) * 1000)
    narration_seconds = len(pcm) / rate

    # THE GATE: refused, never truncated. TTS seconds are the only spend
    # so far; the mux never happens for an over-long narration.
    if narration_seconds > video_seconds:
        raise contract.ContractError(
            "narration-too-long",
            f"narration measured {narration_seconds:.2f}s against the "
            f"video's {video_seconds:.2f}s — nothing was truncated; "
            "shorten the line",
        )

    gain, in_peak_db, in_rms_db = measure_gain(pcm)
    shaped = np.clip(pcm.astype(np.float64) / 32768.0 * gain, -1.0, 1.0)
    target_samples = int(round(video_seconds * rate))
    padded = np.zeros(target_samples, dtype=np.int16)
    keep = min(len(shaped), target_samples)
    padded[:keep] = (shaped[:keep] * 32767.0).astype(np.int16)

    mux_started = time.monotonic()
    mux(input_path, padded, rate, output_path)
    mux_ms = int((time.monotonic() - mux_started) * 1000)

    # The money check, on the OUTPUT: the artifact must carry both
    # streams, agree with itself on duration, and actually be audible.
    produced = probe_media(output_path)
    if produced["video_streams"] < 1 or produced["audio_streams"] < 1:
        raise contract.ContractError(
            "mux-verify-failed", "output lacks a video or audio stream"
        )
    drift = abs(produced["audio_seconds"] - produced["video_seconds"])
    if drift > AV_DRIFT_TOLERANCE_S:
        raise contract.ContractError(
            "mux-verify-failed",
            f"audio {produced['audio_seconds']:.2f}s vs video "
            f"{produced['video_seconds']:.2f}s exceeds the "
            f"{AV_DRIFT_TOLERANCE_S}s drift tolerance",
        )
    peak_db = produced["audio_peak_dbfs"]
    if peak_db is None or peak_db <= SILENCE_FLOOR_DB:
        raise contract.ContractError(
            "mux-verify-failed",
            f"output audio peaks at {peak_db} dBFS — that is silence "
            "with extra steps, not a voice track",
        )

    return {
        "has_audio": True,
        "narration_seconds": round(narration_seconds, 3),
        "audio_seconds": round(produced["audio_seconds"], 3),
        "video_seconds": round(produced["video_seconds"], 3),
        "audio_sample_rate": rate,
        "audio_peak_dbfs": peak_db,
        "audio_gain_db": round(_dbfs(gain) if gain > 0 else 0.0, 1),
        "tts_ms": tts_ms,
        "mux_ms": mux_ms,
        "format": "mp4",
        "output_bytes": os.path.getsize(output_path),
        "duration_ms": int((time.monotonic() - started) * 1000),
    }
