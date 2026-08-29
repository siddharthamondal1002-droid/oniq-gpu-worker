"""The first real quality verdicts on LTX clips — measured, not assumed.

WHY THIS EXISTS. On 2026-08-29 three real LTX clips were pulled and
frame-inspected (validation/fixtures/fixtures.json holds the record).
All three were COMPLETED jobs, all three were valid mp4s, all three had
the right frame count — and one was a PASS (real character motion,
stable identity), one a PARTIAL (motion, then the officer's face became
a different man's and the in-frame BANK BOSS text corrupted), and one a
FAIL (the cartoon plate held about 1.5 seconds and then the entire scene
was replaced by an unrelated photorealistic woman's face). Technical
success is not visual success, and the variance between those three is
the whole reason this module exists: nothing in a status field separates
them, and a metadata-only pipeline would have shipped all three.

TWO VALIDATORS, both computed from a tiny 32x18 luma plane per frame —
the story-worker's own sheet-adapter technique: enough signal to see
change, cheap enough to run on every clip everywhere.

  A. TEMPORAL_ALIVENESS — mean consecutive-frame difference across the
     clip. A frozen clip (the model smearing its plate for four
     seconds) scores ~0. This is the "did anything MOVE" question that
     frame_pull's contact sheet answers for a human eye, made numeric.

  B. PLATE_ANCHOR_DISTANCE — difference of sampled frames against FRAME
     0, the conditioning plate. The PASS fixture stays near its plate
     end to end; the FAIL fixture jumps off a cliff late, once the
     scene is replaced wholesale. The LATE end of this curve is where
     both observed failures live.

Every decision is a pure function with the I/O injected — the only
subprocess (ffmpeg decoding to raw gray) arrives as a `run` argument —
so the whole path is proved offline by tests/test_clip_quality.py
against synthetic byte frames: no network, no ffmpeg, no GPU.

NOTHING HERE CAN COST MONEY. It reads a clip a run already produced and
computes arithmetic on its pixels. A metrics failure must never fail a
generation that succeeded — callers (frame_pull.attach_quality) treat
any error here as QUALITY_REVIEW_REQUIRED, and so does this module:
uncertainty is never converted into a pass.
"""

from __future__ import annotations

# ------------------------------------------------------------ verdicts
#
# String values equal their names so a verdict survives a trip through
# JSON, a log line and a step summary without a decoder ring.

# Motion happened and the clip stayed anchored to its plate. The only
# verdict that is a pass, and only a CALIBRATED run can reach it.
GOOD_MOTION = "GOOD_MOTION"

# The clip barely changed frame to frame — the model held its
# conditioning plate instead of animating it. A frozen clip is a
# failure even though it is a perfectly valid mp4.
STATIC_MOTION_FAILED = "STATIC_MOTION_FAILED"

# IDENTITY DRIFT is the observed failure this flags — the PARTIAL
# fixture's officer reads as a different man by the back half, and the
# in-frame text deforms. A 32x18 luma plane cannot see faces; what it
# CAN see is how far the picture has moved from its conditioning frame,
# and on 2026-08-29 that distance is the best cheap proxy we have for
# "the plate is no longer being respected".
PLATE_DRIFT = "PLATE_DRIFT"

# The scene was REPLACED, not drifted — the FAIL fixture held its
# cartoon plate ~1.5s and then cut to unrelated photoreal content. Far
# beyond drift on the same axis, hence a higher threshold on the same
# late-anchor number.
CONTENT_COLLAPSE = "CONTENT_COLLAPSE"

# The refusal verdict: metrics missing, decode failed, or thresholds
# uncalibrated. Deliberately NOT a pass and NOT a fail — it routes the
# clip to a human eye, which is where every judgement lived before this
# module existed.
QUALITY_REVIEW_REQUIRED = "QUALITY_REVIEW_REQUIRED"


# The luma plane. 32x18 keeps a 704x480 frame's gross composition and
# throws away everything a face or a glyph lives in — which is the
# point: cheap enough to run on every clip, honest about what it sees.
GRAY_WIDTH = 32
GRAY_HEIGHT = 18


def ffmpeg_gray_args(src: str, width: int = GRAY_WIDTH, height: int = GRAY_HEIGHT) -> list:
    """Every frame of one clip as raw 8-bit gray planes on stdout.

    `-nostdin` because a CI runner has no terminal to hand over, `-f
    rawvideo -pix_fmt gray` because a width*height byte string per frame
    is the cheapest thing arithmetic can be done on, and `-` so no
    temporary file has to be named or cleaned up.
    """
    return [
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-i", src,
        "-vf", f"scale={width}x{height}",
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]


def decode_gray(path: str, *, run, width: int = GRAY_WIDTH, height: int = GRAY_HEIGHT) -> list:
    """The clip as a list of width*height-byte gray frames, or [].

    [] rather than a raise, for every failure shape at once — a missing
    ffmpeg (OSError), a decoder refusal (returncode), a text-mode runner
    (str stdout), an empty pipe. The caller's job is a verdict, and the
    verdict for "could not see the frames" is QUALITY_REVIEW_REQUIRED,
    never an exception that could lose the frames already cut.
    """
    try:
        done = run(ffmpeg_gray_args(path, width, height), capture_output=True)
    except OSError:
        return []
    if getattr(done, "returncode", 1) != 0:
        return []
    stdout = getattr(done, "stdout", None)
    if not isinstance(stdout, bytes) or not stdout:
        return []
    frame_bytes = width * height
    # A trailing partial frame is dropped, not padded: padding would
    # invent pixels and the arithmetic below would score the invention.
    count = len(stdout) // frame_bytes
    return [stdout[i * frame_bytes:(i + 1) * frame_bytes] for i in range(count)]


def mean_abs_diff(a: bytes, b: bytes) -> float:
    """Mean absolute per-pixel difference of two equal-length byte
    strings, normalized to 0..1 (a byte's range is 255).

    Pure arithmetic, and the single primitive both validators are built
    from: identical frames answer 0.0, a black-to-white cut answers 1.0.
    """
    if not a:
        return 0.0
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a) / 255.0


def aliveness(frames) -> float:
    """Validator A, TEMPORAL_ALIVENESS: mean consecutive-frame
    difference across the whole clip. A frozen clip scores ~0.

    None below 2 frames — one frame has no motion to measure, and
    answering 0.0 there would let an undecodable clip impersonate a
    frozen one, which are different problems with different fixes.
    """
    if not frames or len(frames) < 2:
        return None
    diffs = [mean_abs_diff(a, b) for a, b in zip(frames, frames[1:])]
    return sum(diffs) / len(diffs)


def anchor_curve(frames, points=(0.0, 0.25, 0.5, 0.75, 1.0)) -> list:
    """Validator B, PLATE_ANCHOR_DISTANCE: how far the frame at each
    relative position sits from FRAME 0, the conditioning plate.

    Frame 0 is the baseline deliberately — it is the plate the model was
    handed, so distance from it IS distance from the conditioning. The
    2026-08-29 inspection shaped the sampling: the PASS fixture stays
    near its plate at every point, while the FAIL fixture holds for
    ~1.5s of its 4.04s and then jumps off a cliff — a failure the EARLY
    points cannot see and the LATE points cannot miss.

    None below 2 frames, same reasoning as aliveness.
    """
    if not frames or len(frames) < 2:
        return None
    last = len(frames) - 1
    return [mean_abs_diff(frames[min(last, round(p * last))], frames[0]) for p in points]


def metrics(frames) -> dict:
    """Both validators over one decoded clip, ready for a log line.

    Floats are rounded to 4 places so logs stay readable and the
    calibration numbers copied out of them are stable — the thresholds
    in CALIBRATION will be transcribed from exactly these digits, and a
    transcription that loses precision the comparison still has would
    move a boundary nobody meant to move.
    """
    curve = anchor_curve(frames)
    rounded = [round(v, 4) for v in curve] if curve is not None else None
    alive = aliveness(frames)
    return {
        "frames": len(frames or []),
        "aliveness": round(alive, 4) if alive is not None else None,
        "anchor": rounded,
        # The mean of the LAST TWO anchor points, because that is where
        # both observed failures live: the PARTIAL fixture drifts in its
        # back half, the FAIL fixture has already been replaced by then.
        "anchor_late": (
            round(sum(rounded[-2:]) / 2, 4) if rounded and len(rounded) >= 2 else None
        ),
    }


# THRESHOLDS ARE MEASURED, NEVER GUESSED. These are derived ONLY from
# the measured fixture numbers (validation/fixtures/fixtures.json run
# through these exact validators) and inserted here after the
# calibration run, with the numbers that justified them recorded beside
# them. Arbitrary universal thresholds are forbidden by owner directive
# 2026-08-29 — the whole lesson of the three fixtures is that intuition
# about "how much change is normal" was wrong three different ways.
# None means UNCALIBRATED, and classify() refuses to pass anything
# while any needed threshold is None.
CALIBRATION = {
    "static_below": None,   # aliveness at or under this: frozen clip
    "collapse_above": None, # anchor_late at or over this: scene replaced
    "drift_above": None,    # anchor_late at or over this: plate abandoned
}


def classify(m: dict, calibration: dict = CALIBRATION) -> str:
    """One verdict from one clip's metrics — CONSERVATIVE by construction.

    Every gap in the evidence answers QUALITY_REVIEW_REQUIRED: missing
    metrics, an undecodable clip, an uncalibrated threshold. Owner
    directive 2026-08-29 — uncalibrated NEVER converts to a pass;
    uncertainty is never PASS. The failure checks run worst-first
    (frozen, then replaced, then drifted) so a clip that fails two ways
    is named by the worse one, and every comparison is inclusive: a
    value EXACTLY on a boundary lands on the failure side, because a
    boundary case is by definition the case we are least sure about.
    """
    if not m or m.get("aliveness") is None or m.get("anchor_late") is None:
        return QUALITY_REVIEW_REQUIRED
    static_below = calibration.get("static_below")
    collapse_above = calibration.get("collapse_above")
    drift_above = calibration.get("drift_above")
    if static_below is None or collapse_above is None or drift_above is None:
        return QUALITY_REVIEW_REQUIRED
    if m["aliveness"] <= static_below:
        return STATIC_MOTION_FAILED
    if m["anchor_late"] >= collapse_above:
        return CONTENT_COLLAPSE
    if m["anchor_late"] >= drift_above:
        return PLATE_DRIFT
    return GOOD_MOTION


def assess(path: str, *, run) -> dict:
    """One clip, one verdict, with the numbers that produced it.

    The metrics ride along in the report because a verdict without its
    evidence cannot be argued with — and arguing with these numbers
    against the fixtures is exactly how CALIBRATION gets filled.
    """
    frames = decode_gray(path, run=run)
    if not frames:
        return {"verdict": QUALITY_REVIEW_REQUIRED, "frames": 0}
    m = metrics(frames)
    report = {"verdict": classify(m)}
    report.update(m)
    return report
