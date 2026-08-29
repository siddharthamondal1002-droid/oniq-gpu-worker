"""The clip-quality validators, proved offline against synthetic frames.

Owner directive 2026-08-29: quality judgements are made from actual
frames, thresholds come only from measured fixture numbers, and
uncertainty never converts into a pass. So the properties proved here,
in order of what they cost if wrong:

  1. an UNCALIBRATED validator can never say GOOD_MOTION — not even for
     perfect metrics;
  2. missing evidence (no frames, no metrics, a failed decode) always
     answers QUALITY_REVIEW_REQUIRED, never a raise and never a pass;
  3. a value exactly on a boundary lands on the FAILURE side — the
     comparisons are inclusive, because the boundary case is the case
     we are least sure about;
  4. the arithmetic itself: a frozen clip scores ~0 aliveness, a
     late-replaced clip scores a large anchor_late while its early
     anchor points stay small — the exact signatures of the 2026-08-29
     STATIC and CONTENT_COLLAPSE fixtures.

No network, no ffmpeg execution, no GPU: the decoder's runner is a fake
and every frame is a handmade byte string.
"""

from __future__ import annotations

import pytest

from validation import clip_quality


FRAME_BYTES = clip_quality.GRAY_WIDTH * clip_quality.GRAY_HEIGHT  # 32*18 = 576


def _frame(value: int) -> bytes:
    """One synthetic 32x18 gray frame of a single luma value."""
    return bytes([value]) * FRAME_BYTES


class _Result:
    def __init__(self, returncode: int = 0, stdout=b""):
        self.returncode = returncode
        self.stdout = stdout


def _runner(stdout=b"", returncode: int = 0):
    """A subprocess.run-shaped fake that records nothing and runs nothing."""

    def run(argv, capture_output=False, **kwargs):
        assert capture_output, "decode_gray must capture stdout"
        assert argv[0] == "ffmpeg"
        return _Result(returncode, stdout)

    return run


# A filled calibration for the classify tests. These numbers are TEST
# scaffolding chosen to make every verdict reachable from synthetic
# frames — they are not the real thresholds, which may only come from
# the measured fixture run (owner directive 2026-08-29).
CAL = {"static_below": 0.002, "collapse_above": 0.5, "drift_above": 0.2}


# ---------------------------------------------------------- the arithmetic


def test_identical_frames_score_zero_everywhere():
    frames = [_frame(100)] * 8
    assert clip_quality.aliveness(frames) == 0.0
    assert clip_quality.anchor_curve(frames) == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_gently_changing_frames_are_alive_but_stay_anchored():
    # Luma creeps by 1 per frame — visible motion, tiny plate distance.
    frames = [_frame(100 + i) for i in range(8)]
    alive = clip_quality.aliveness(frames)
    assert alive == pytest.approx(1 / 255)
    assert alive > 0
    curve = clip_quality.anchor_curve(frames)
    assert max(curve) <= 7 / 255  # never further than the last frame's creep
    assert max(curve) < 0.05


def test_a_late_scene_replacement_spikes_anchor_late_only():
    # The CONTENT_COLLAPSE fixture's shape: the plate holds for the
    # front half, then the scene is replaced by very different content.
    frames = [_frame(10)] * 4 + [_frame(245)] * 4
    m = clip_quality.metrics(frames)
    curve = m["anchor"]
    assert curve[0] == 0.0  # the plate against itself
    assert curve[1] == 0.0  # early: still the plate
    assert m["anchor_late"] == pytest.approx(235 / 255, abs=1e-3)
    assert m["anchor_late"] > 0.5


def test_mean_abs_diff_spans_zero_to_one():
    assert clip_quality.mean_abs_diff(_frame(7), _frame(7)) == 0.0
    assert clip_quality.mean_abs_diff(_frame(0), _frame(255)) == 1.0


def test_below_two_frames_there_is_no_measurement():
    assert clip_quality.aliveness([]) is None
    assert clip_quality.aliveness([_frame(1)]) is None
    assert clip_quality.anchor_curve([]) is None
    assert clip_quality.anchor_curve([_frame(1)]) is None
    m = clip_quality.metrics([_frame(1)])
    assert m["frames"] == 1
    assert m["aliveness"] is None
    assert m["anchor"] is None
    assert m["anchor_late"] is None


def test_metrics_rounds_to_four_places_for_stable_logs():
    frames = [_frame(100 + i) for i in range(8)]
    m = clip_quality.metrics(frames)
    assert m["aliveness"] == round(1 / 255, 4)
    assert all(v == round(v, 4) for v in m["anchor"])
    assert m["anchor_late"] == round(m["anchor_late"], 4)


# ------------------------------------------------------------ the verdict


def test_uncalibrated_never_converts_to_a_pass():
    # Perfect metrics, shipped CALIBRATION (all None): still review.
    perfect = {"frames": 97, "aliveness": 0.02, "anchor": [0.0] * 5, "anchor_late": 0.01}
    assert clip_quality.CALIBRATION == {
        "static_below": None, "collapse_above": None, "drift_above": None,
    }
    assert clip_quality.classify(perfect) == clip_quality.QUALITY_REVIEW_REQUIRED


@pytest.mark.parametrize("missing", ["static_below", "collapse_above", "drift_above"])
def test_one_missing_threshold_is_enough_to_refuse(missing):
    cal = dict(CAL)
    cal[missing] = None
    good = {"aliveness": 0.02, "anchor_late": 0.01}
    assert clip_quality.classify(good, cal) == clip_quality.QUALITY_REVIEW_REQUIRED


def test_every_verdict_is_reachable_once_calibrated():
    review = clip_quality.QUALITY_REVIEW_REQUIRED
    cases = [
        ({"aliveness": 0.001, "anchor_late": 0.01}, clip_quality.STATIC_MOTION_FAILED),
        ({"aliveness": 0.02, "anchor_late": 0.6}, clip_quality.CONTENT_COLLAPSE),
        ({"aliveness": 0.02, "anchor_late": 0.3}, clip_quality.PLATE_DRIFT),
        ({"aliveness": 0.02, "anchor_late": 0.01}, clip_quality.GOOD_MOTION),
    ]
    for m, verdict in cases:
        assert clip_quality.classify(m, CAL) == verdict
        assert verdict != review


@pytest.mark.parametrize("m", [None, {}, {"aliveness": None, "anchor_late": 0.1},
                               {"aliveness": 0.02, "anchor_late": None}])
def test_missing_metrics_answer_review_even_when_calibrated(m):
    assert clip_quality.classify(m, CAL) == clip_quality.QUALITY_REVIEW_REQUIRED


def test_a_value_exactly_on_a_boundary_lands_on_the_failure_side():
    # The comparisons are inclusive (<= / >=): a clip AT the threshold
    # is treated as the failure, never as the pass — the boundary case
    # is the case the calibration is least sure about.
    on_static = {"aliveness": CAL["static_below"], "anchor_late": 0.01}
    assert clip_quality.classify(on_static, CAL) == clip_quality.STATIC_MOTION_FAILED
    on_collapse = {"aliveness": 0.02, "anchor_late": CAL["collapse_above"]}
    assert clip_quality.classify(on_collapse, CAL) == clip_quality.CONTENT_COLLAPSE
    on_drift = {"aliveness": 0.02, "anchor_late": CAL["drift_above"]}
    assert clip_quality.classify(on_drift, CAL) == clip_quality.PLATE_DRIFT


def test_a_worse_failure_outranks_a_lesser_one():
    # Frozen AND far from the plate is named by the frozen check first;
    # collapse outranks drift on the same number.
    both = {"aliveness": 0.0, "anchor_late": 0.9}
    assert clip_quality.classify(both, CAL) == clip_quality.STATIC_MOTION_FAILED


# ------------------------------------------------------------- the decode


def test_decode_splits_frames_and_drops_the_trailing_partial():
    stdout = _frame(1) + _frame(2) + _frame(3) + b"partial"
    frames = clip_quality.decode_gray("clip.mp4", run=_runner(stdout))
    assert len(frames) == 3
    assert all(len(f) == FRAME_BYTES for f in frames)
    assert frames[1] == _frame(2)


def test_a_failing_runner_yields_no_frames():
    assert clip_quality.decode_gray("clip.mp4", run=_runner(_frame(1), returncode=1)) == []


def test_text_mode_stdout_yields_no_frames():
    # A text-mode runner hands back str; arithmetic on it would be a
    # TypeError three calls later — refuse it at the door instead.
    assert clip_quality.decode_gray("clip.mp4", run=_runner(stdout="not bytes")) == []


def test_empty_stdout_yields_no_frames():
    assert clip_quality.decode_gray("clip.mp4", run=_runner(b"")) == []


def test_a_raising_runner_yields_no_frames():
    def run(argv, capture_output=False, **kwargs):
        raise OSError("no ffmpeg on this runner")

    assert clip_quality.decode_gray("clip.mp4", run=run) == []


def test_gray_args_carry_the_decode_contract():
    args = clip_quality.ffmpeg_gray_args("clip.mp4")
    assert args[0] == "ffmpeg"
    assert "-nostdin" in args
    assert "rawvideo" in args
    assert "gray" in args
    assert "scale=32x18" in args
    assert "clip.mp4" in args
    assert args[-1] == "-"


def test_gray_args_respect_a_custom_plane_size():
    assert "scale=8x4" in clip_quality.ffmpeg_gray_args("c.mp4", width=8, height=4)


# ------------------------------------------------------------- the report


def test_assess_returns_verdict_beside_every_metric():
    stdout = b"".join(_frame(100 + i) for i in range(6))
    report = clip_quality.assess("clip.mp4", run=_runner(stdout))
    # Shipped CALIBRATION is all None, so even a moving clip is review.
    assert report["verdict"] == clip_quality.QUALITY_REVIEW_REQUIRED
    assert report["frames"] == 6
    assert report["aliveness"] == round(1 / 255, 4)
    assert len(report["anchor"]) == 5
    assert report["anchor_late"] is not None


def test_assess_with_no_decodable_frames_is_review_not_a_raise():
    report = clip_quality.assess("clip.mp4", run=_runner(returncode=1))
    assert report == {"verdict": clip_quality.QUALITY_REVIEW_REQUIRED, "frames": 0}
