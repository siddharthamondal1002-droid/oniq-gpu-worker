"""LTX quality is judged from frames, so getting the frames must be proved.

Owner directive 2026-08-29: generation -> R2 -> frame extraction -> CI
artifact -> inspection, and "do not launch another GPU job merely to
test this plumbing". So every decision in frame_pull is exercised here
against fakes: no network, no ffmpeg, no bucket, no endpoint.

The properties that matter, in order of what they cost if wrong:

  1. the public base can never be a credential (a presigned URL is one);
  2. a downloaded clip is proved before it is trusted;
  3. one unfetchable clip does not lose the other four;
  4. a missing base is loud but never fails a generation that succeeded.
"""

from __future__ import annotations

import json
import os

import pytest

from validation import frame_pull


BASE = "https://pub-example.r2.dev"


def _mp4(size: int = 64) -> bytes:
    return b"\x00\x00\x00\x18ftypisom" + b"\x00" * (size - 12)


class _Result:
    def __init__(self, returncode: int = 0):
        self.returncode = returncode


# ------------------------------------------------------------ the base


def test_a_presigned_url_is_refused_because_it_is_a_credential():
    signed = BASE + "/story/clip.mp4?X-Amz-Signature=deadbeef&X-Amz-Expires=900"
    with pytest.raises(frame_pull.FramePullError) as exc:
        frame_pull.normalise_base(signed)
    assert exc.value.code == "base-has-query"


@pytest.mark.parametrize(
    "raw, code",
    [
        ("", "base-unset"),
        ("   ", "base-unset"),
        ("http://pub-example.r2.dev", "base-not-https"),
        ("s3://bucket", "base-not-https"),
        ("https://", "base-no-host"),
        ("https://key:secret@pub-example.r2.dev", "base-has-userinfo"),
        ("https://pub-example.r2.dev#token", "base-has-fragment"),
    ],
)
def test_every_shape_that_could_carry_authority_is_refused(raw, code):
    with pytest.raises(frame_pull.FramePullError) as exc:
        frame_pull.normalise_base(raw)
    assert exc.value.code == code


def test_a_plain_public_base_is_accepted_and_its_trailing_slash_dropped():
    assert frame_pull.normalise_base(BASE + "/") == BASE
    assert frame_pull.normalise_base("  " + BASE + "  ") == BASE


def test_only_the_host_is_logged_never_the_path():
    assert frame_pull.safe_host(BASE + "/oniq-gpu") == "pub-example.r2.dev"


# ------------------------------------------------------------- the key


def test_a_key_names_one_object_in_one_bucket():
    assert (
        frame_pull.public_url(BASE, "validation/out/battery-1-intro.mp4")
        == BASE + "/validation/out/battery-1-intro.mp4"
    )


@pytest.mark.parametrize(
    "key", ["", "/etc/passwd", "../../secret", "https://evil.example/x", "a\\b"]
)
def test_a_key_that_could_leave_the_bucket_is_refused(key):
    with pytest.raises(frame_pull.FramePullError):
        frame_pull.public_url(BASE, key)


# -------------------------------------------------------- the artifact


def test_the_bytes_that_arrived_must_be_the_bytes_the_worker_wrote():
    data = _mp4(64)
    frame_pull.verify_artifact(data, 64)  # exact — fine
    with pytest.raises(frame_pull.FramePullError) as exc:
        frame_pull.verify_artifact(data, 65)
    assert exc.value.code == "artifact-size-mismatch"


def test_an_empty_or_non_mp4_download_is_refused():
    with pytest.raises(frame_pull.FramePullError) as exc:
        frame_pull.verify_artifact(b"", 0)
    assert exc.value.code == "artifact-empty"
    with pytest.raises(frame_pull.FramePullError) as exc:
        frame_pull.verify_artifact(b"<!DOCTYPE html><html>404</html>", None)
    assert exc.value.code == "artifact-not-mp4"


def test_an_unreported_size_still_checks_the_magic():
    frame_pull.verify_artifact(_mp4(), None)
    frame_pull.verify_artifact(_mp4(), 0)


# ---------------------------------------------------------- the sampling


def test_frames_are_spread_across_the_clip_and_skip_the_plate():
    times = frame_pull.frame_times(4.04, 6)
    assert len(times) == 6
    assert times[0] > 0, "frame 0 is the conditioning plate, not a result"
    assert times[-1] < 4.04, "seeking to the duration lands past the last frame"
    assert times == sorted(times)


@pytest.mark.parametrize("bad", [0, -1, None, "", "n/a"])
def test_a_clip_with_no_measured_duration_yields_no_frames(bad):
    assert frame_pull.frame_times(bad) == []


def test_the_ffmpeg_calls_are_non_interactive_and_overwrite():
    args = frame_pull.ffmpeg_frame_args("in.mp4", 1.5, "out.png")
    assert args[0] == "ffmpeg"
    for flag in ("-nostdin", "-y"):
        assert flag in args, f"{flag} missing — a runner has no tty"
    assert args[args.index("-ss") + 1] == "1.5"
    assert args[-1] == "out.png"
    sheet = frame_pull.ffmpeg_sheet_args("in.mp4", "sheet.png")
    assert "-nostdin" in sheet and "tile=6x1" in " ".join(sheet)


# ---------------------------------------------------------- the manifest


def test_only_video_rows_reach_the_manifest():
    rows = [
        {"op": "video_generate", "output_key": "a.mp4", "scene": "intro"},
        {"op": "image_generate", "output_key": "plate.png"},
        {"op": "video_generate"},  # no key — nothing to fetch
        "not-a-row",
    ]
    assert [c["output_key"] for c in frame_pull.video_rows(rows)] == ["a.mp4"]


def test_the_manifest_carries_evidence_and_never_a_price(tmp_path):
    path = str(tmp_path / "m.json")
    rows = [
        {
            "op": "video_generate",
            "scene": "intro",
            "output_key": "validation/out/battery-1-intro.mp4",
            "output_bytes": 512,
            "video_seconds": "4.04",
            "frames": 97,
            "fps": 24,
            "resolution": "704x480",
            "model": "LTX_VIDEO_2B",
            "job_id": "gpu-1",
            "cost_usd": "0.01",
            "reservation_usd": "0.07",
        }
    ]
    frame_pull.write_manifest(rows, path)
    written = json.loads(open(path, encoding="utf-8").read())
    clip = written["clips"][0]
    assert clip["output_key"] == "validation/out/battery-1-intro.mp4"
    assert clip["output_bytes"] == 512
    blob = json.dumps(written)
    for forbidden in ("cost_usd", "reservation_usd", "0.07"):
        assert forbidden not in blob, "the manifest is evidence, not accounting"


def test_a_missing_manifest_reads_as_no_clips(tmp_path):
    assert frame_pull.read_manifest(str(tmp_path / "absent.json")) == []


# -------------------------------------------------------------- the pull


def test_a_good_clip_yields_its_frames_and_a_sheet(tmp_path):
    calls = []
    data = _mp4(128)
    report = frame_pull.pull_clip(
        {
            "scene": "intro",
            "output_key": "validation/out/battery-1-intro.mp4",
            "output_bytes": 128,
            "video_seconds": "4.04",
        },
        BASE,
        out_dir=str(tmp_path),
        fetch=lambda url: data,
        run=lambda args: (calls.append(args), _Result(0))[1],
    )
    assert "error" not in report
    assert report["bytes"] == 128
    # the mp4, six frames, and the sheet
    assert len(report["artifacts"]) == 8
    assert any(a.endswith("-sheet.png") for a in report["artifacts"])
    assert len(calls) == 7
    assert os.path.exists(tmp_path / "intro.mp4")


def test_one_unfetchable_clip_does_not_lose_the_others(tmp_path):
    report = frame_pull.pull_clip(
        {"scene": "walk", "output_key": "validation/out/b.mp4", "output_bytes": 99},
        BASE,
        out_dir=str(tmp_path),
        fetch=lambda url: (_ for _ in ()).throw(OSError("connection reset")),
        run=lambda args: _Result(0),
    )
    assert report["error"] == "fetch-failed:OSError"
    assert report["artifacts"] == []


def test_a_wrong_sized_download_is_reported_not_cut_into_frames(tmp_path):
    ran = []
    report = frame_pull.pull_clip(
        {"scene": "hero", "output_key": "validation/out/c.mp4", "output_bytes": 999},
        BASE,
        out_dir=str(tmp_path),
        fetch=lambda url: _mp4(64),
        run=lambda args: (ran.append(args), _Result(0))[1],
    )
    assert report["error"] == "artifact-size-mismatch"
    assert ran == [], "nothing is extracted from bytes that failed verification"


def test_the_fetch_url_is_built_from_the_public_base_and_the_key(tmp_path):
    seen = []
    frame_pull.pull_clip(
        {"scene": "s", "output_key": "validation/out/d.mp4", "output_bytes": 64,
         "video_seconds": 0},
        BASE + "/",
        out_dir=str(tmp_path),
        fetch=lambda url: (seen.append(url), _mp4(64))[1],
        run=lambda args: _Result(0),
    )
    assert seen == [BASE + "/validation/out/d.mp4"]


def test_a_failed_ffmpeg_call_drops_that_frame_and_keeps_the_rest(tmp_path):
    report = frame_pull.pull_clip(
        {"scene": "x", "output_key": "validation/out/e.mp4", "output_bytes": 64,
         "video_seconds": 4.04},
        BASE,
        out_dir=str(tmp_path),
        fetch=lambda url: _mp4(64),
        run=lambda args: _Result(1),
    )
    assert report["artifacts"] == ["x.mp4"], "only the clip itself survives"
    assert "error" not in report


# ----------------------------------------------------------------- main


def test_no_manifest_is_a_quiet_success(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert frame_pull.main() == 0
    assert "nothing to fetch" in capsys.readouterr().out


def test_an_unset_base_is_loud_but_never_fails_a_successful_generation(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    frame_pull.write_manifest(
        [{"op": "video_generate", "output_key": "a.mp4", "scene": "intro"}]
    )
    monkeypatch.delenv("R2_PUBLIC_BASE_URL", raising=False)
    # The GPU work already succeeded and its artifacts are in the bucket.
    # What is lost is the seeing, and that must not turn a paid, verified
    # generation into a red run.
    assert frame_pull.main() == 0
    out = capsys.readouterr().out
    assert "frame-pull REFUSED: base-unset" in out
    assert "repository VARIABLE" in out


def test_a_presigned_base_in_the_variable_refuses_rather_than_authenticating(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    frame_pull.write_manifest(
        [{"op": "video_generate", "output_key": "a.mp4", "scene": "intro"}]
    )
    monkeypatch.setenv("R2_PUBLIC_BASE_URL", BASE + "/?X-Amz-Signature=abc")
    assert frame_pull.main() == 0
    out = capsys.readouterr().out
    assert "base-has-query" in out
    assert "abc" not in out, "a refused credential is never echoed back"


# ---------------------------------------------------------- the report


def test_the_run_report_names_every_artifact_and_its_measured_evidence():
    clips = [
        {
            "scene": "intro",
            "output_key": "validation/out/battery-1-intro.mp4",
            "frames": 97,
            "fps": 24,
            "video_seconds": "4.04",
            "resolution": "704x480",
        }
    ]
    reports = [
        {
            "scene": "intro",
            "output_key": "validation/out/battery-1-intro.mp4",
            "bytes": 512,
            "artifacts": ["intro.mp4", "intro-f1-0.577s.png", "intro-sheet.png"],
        }
    ]
    md = frame_pull.summary_markdown(clips, reports)
    assert "intro-sheet.png" in md
    assert "97f @ 24fps" in md and "704x480" in md
    assert "1/1 clip(s) retrieved" in md


def test_a_clip_with_no_frames_says_so_in_the_report():
    clips = [{"scene": "walk", "output_key": "b.mp4"}]
    reports = [{"scene": "walk", "output_key": "b.mp4", "error": "artifact-not-mp4"}]
    md = frame_pull.summary_markdown(clips, reports)
    assert "none — artifact-not-mp4" in md
    assert "0/1 clip(s) retrieved" in md


def test_the_summary_is_written_where_the_run_page_reads_it(tmp_path, monkeypatch):
    dest = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(dest))
    frame_pull.write_summary([], [])
    assert "LTX frames" in dest.read_text(encoding="utf-8")


def test_no_summary_path_is_simply_no_summary(monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    frame_pull.write_summary([], [])  # must not raise
