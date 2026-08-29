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
    def __init__(self, returncode: int = 0, stdout: str = ""):
        self.returncode = returncode
        self.stdout = stdout


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
        run=lambda args, **kw: (calls.append(args), _Result(0))[1],
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
        run=lambda args, **kw: _Result(0),
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
        run=lambda args, **kw: (ran.append(args), _Result(0))[1],
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
        run=lambda args, **kw: _Result(0),
    )
    assert seen == [BASE + "/validation/out/d.mp4"]


def test_a_failed_ffmpeg_call_drops_that_frame_and_keeps_the_rest(tmp_path):
    report = frame_pull.pull_clip(
        {"scene": "x", "output_key": "validation/out/e.mp4", "output_bytes": 64,
         "video_seconds": 4.04},
        BASE,
        out_dir=str(tmp_path),
        fetch=lambda url: _mp4(64),
        run=lambda args, **kw: _Result(1),
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


# ------------------------------------------- keys named directly ($0 path)


def test_keys_become_clips_named_after_the_object():
    clips = frame_pull.clips_from_keys(
        " validation/out/ltx-001.mp4 , validation/video-test/ltx-001.mp4 ,, "
    )
    assert [c["output_key"] for c in clips] == [
        "validation/out/ltx-001.mp4",
        "validation/video-test/ltx-001.mp4",
    ]
    # The WHOLE key, because two keys sharing a file name must not share a
    # scene — the second's frames overwrote the first's when they did.
    assert clips[0]["scene"] == "validation-out-ltx-001"
    assert clips[1]["scene"] == "validation-video-test-ltx-001"
    assert clips[0]["scene"] != clips[1]["scene"]
    # No worker report stands behind a hand-named key, so no size claim
    # is invented for it — the mp4 magic is still checked.
    assert "output_bytes" not in clips[0]
    assert "video_seconds" not in clips[0]


def test_no_keys_is_a_quiet_success(monkeypatch, capsys):
    monkeypatch.setenv("FRAME_KEYS", "")
    assert frame_pull.main(["keys"]) == 0
    assert "names no object" in capsys.readouterr().out


def test_a_duration_with_no_worker_report_is_measured_not_assumed():
    probed = frame_pull.probe_seconds(
        "clip.mp4", run=lambda args, **kw: _Result(0, "4.041667\n")
    )
    assert probed == pytest.approx(4.041667)
    assert "ffprobe" in frame_pull.ffprobe_args("clip.mp4")[0]


@pytest.mark.parametrize(
    "result", [_Result(1, "4.0"), _Result(0, ""), _Result(0, "N/A"), _Result(0, "0")]
)
def test_an_unmeasurable_duration_is_none_never_a_guess(result):
    assert frame_pull.probe_seconds("clip.mp4", run=lambda args, **kw: result) is None


def test_a_hand_named_clip_is_probed_then_cut(tmp_path):
    seen = []

    def fake_run(args, **kw):
        seen.append(args[0])
        if args[0] == "ffprobe":
            return _Result(0, "4.04\n")
        return _Result(0)

    report = frame_pull.pull_clip(
        frame_pull.clips_from_keys("validation/out/ltx-001.mp4")[0],
        BASE,
        out_dir=str(tmp_path),
        fetch=lambda url: _mp4(256),
        run=fake_run,
    )
    assert report["measured_seconds"] == pytest.approx(4.04)
    assert seen[0] == "ffprobe", "the duration is measured before frames are cut"
    assert len(report["artifacts"]) == 8


def test_a_flag_shaped_base_says_so_instead_of_talking_about_schemes():
    with pytest.raises(frame_pull.FramePullError) as exc:
        frame_pull.normalise_base("on")
    assert exc.value.code == "base-not-a-url"
    assert "not an on/off flag" in exc.value.message


def test_a_failed_fetch_reports_the_status_not_just_that_it_failed(tmp_path):
    # MEASURED 2026-08-29: the first real pull said `fetch-failed:HTTPError`
    # and that sentence contains no diagnosis. 403 and 404 are different
    # problems with different fixes. Same mistake story-still made when its
    # throw discarded the engine's reason.
    import urllib.error

    def forbidden(url):
        raise urllib.error.HTTPError(url, 403, "Forbidden", {}, None)

    report = frame_pull.pull_clip(
        {"scene": "s", "output_key": "validation/out/ltx-001.mp4"},
        BASE,
        out_dir=str(tmp_path),
        fetch=forbidden,
        run=lambda args, **kw: _Result(0),
    )
    assert report["error"] == "fetch-failed:HTTP 403"
    assert "public read" in report["hint"]
    # The URL is still never echoed — a status is not a secret, a URL can be.
    assert BASE not in json.dumps(report)


def test_a_404_points_at_the_key_not_at_permissions(tmp_path):
    import urllib.error

    report = frame_pull.pull_clip(
        {"scene": "s", "output_key": "validation/out/missing.mp4"},
        BASE,
        out_dir=str(tmp_path),
        fetch=lambda url: (_ for _ in ()).throw(
            urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        ),
        run=lambda args, **kw: _Result(0),
    )
    assert report["error"] == "fetch-failed:HTTP 404"
    assert "key" in report["hint"]


def test_a_network_failure_is_still_named_by_class_only(tmp_path):
    report = frame_pull.pull_clip(
        {"scene": "s", "output_key": "validation/out/ltx-001.mp4"},
        BASE,
        out_dir=str(tmp_path),
        fetch=lambda url: (_ for _ in ()).throw(OSError("reset")),
        run=lambda args, **kw: _Result(0),
    )
    assert report["error"] == "fetch-failed:OSError"


# ============================================================================
# THE PRIVATE-BUCKET PATH (owner directive 2026-08-29: R2 stays private)
#
# A presigned URL is a credential: the signature is in the query and anyone
# holding the string can read that object until it expires. These tests exist
# because the whole point of not opening the bucket is lost if the credential
# then leaks into a public CI log.
# ============================================================================

SIGNED = (
    "https://oniq-gpu.abc123.r2.cloudflarestorage.com/validation/out/ltx-001.mp4"
    "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAEXAMPLE%2F20260829"
    "&X-Amz-Signature=1234deadbeefcafe&X-Amz-Expires=900"
)
SECRETS = ("X-Amz-Signature", "1234deadbeefcafe", "AKIAEXAMPLE", "abc123")


def test_a_signed_url_is_accepted_where_a_base_would_be_refused():
    # The deliberate opposite of normalise_base: here the query IS the
    # point. The two paths stay separate for exactly that reason.
    with pytest.raises(frame_pull.FramePullError):
        frame_pull.normalise_base(SIGNED)
    clips = frame_pull.clips_from_signed(SIGNED)
    assert len(clips) == 1
    assert clips[0]["_signed"] == SIGNED


def test_a_signed_url_is_never_sent_in_the_clear():
    with pytest.raises(frame_pull.FramePullError) as exc:
        frame_pull.clips_from_signed("http://oniq-gpu.example/x.mp4?X-Amz-Signature=a")
    assert exc.value.code == "signed-not-https"


def test_only_the_object_name_survives_into_anything_visible():
    assert frame_pull.safe_label(SIGNED) == "validation-out-ltx-001"
    for secret in SECRETS:
        assert secret not in frame_pull.safe_label(SIGNED)


def test_the_credential_never_reaches_the_report(tmp_path):
    report = frame_pull.pull_clip(
        frame_pull.clips_from_signed(SIGNED)[0],
        "",  # no public base at all on this path
        out_dir=str(tmp_path),
        fetch=lambda url: _mp4(256),
        run=lambda args, **kw: _Result(0, "4.04\n"),
    )
    assert report["scene"] == "validation-out-ltx-001"
    assert report["source"] == "signed"
    blob = json.dumps(report)
    for secret in SECRETS:
        assert secret not in blob, f"{secret} leaked into the report"
    assert "_signed" not in blob


def test_the_credential_never_reaches_the_run_summary():
    clips = frame_pull.clips_from_signed(SIGNED)
    reports = [{"scene": "ltx-001", "source": "signed", "bytes": 9, "artifacts": ["a.png"]}]
    md = frame_pull.summary_markdown(clips, reports)
    for secret in SECRETS:
        assert secret not in md
    assert "signed URL (private bucket)" in md


def test_the_credential_never_reaches_stdout(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SIGNED_URLS", SIGNED)
    monkeypatch.setattr(frame_pull, "_fetch", lambda url: _mp4(256))
    monkeypatch.setattr(
        frame_pull.subprocess, "run", lambda args, **kw: _Result(0, "4.04\n")
    )
    assert frame_pull.main(["signed"]) == 0
    out = capsys.readouterr().out
    for secret in SECRETS:
        assert secret not in out, f"{secret} printed to a public CI log"
    assert "ltx-001" in out


def test_the_signed_path_still_proves_the_bytes(tmp_path):
    # Authentication is not a reason to trust the payload: a 200 carrying
    # an HTML error page is still not an mp4.
    report = frame_pull.pull_clip(
        frame_pull.clips_from_signed(SIGNED)[0],
        "",
        out_dir=str(tmp_path),
        fetch=lambda url: b"<!DOCTYPE html><html>nope</html>",
        run=lambda args, **kw: _Result(0),
    )
    assert report["error"] == "artifact-not-mp4"


def test_a_403_on_the_signed_path_reports_the_status_without_the_url(tmp_path):
    import urllib.error

    report = frame_pull.pull_clip(
        frame_pull.clips_from_signed(SIGNED)[0],
        "",
        out_dir=str(tmp_path),
        fetch=lambda url: (_ for _ in ()).throw(
            urllib.error.HTTPError(url, 403, "Forbidden", {}, None)
        ),
        run=lambda args, **kw: _Result(0),
    )
    assert report["error"] == "fetch-failed:HTTP 403"
    for secret in SECRETS:
        assert secret not in json.dumps(report)


def test_several_signed_urls_may_be_given_at_once():
    clips = frame_pull.clips_from_signed(SIGNED + "\n" + SIGNED.replace("ltx-001", "ltx-002"))
    assert [c["scene"] for c in clips] == [
        "validation-out-ltx-001",
        "validation-out-ltx-002",
    ]


def test_no_signed_urls_is_a_quiet_success(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SIGNED_URLS", "")
    assert frame_pull.main(["signed"]) == 0
    assert "names no object" in capsys.readouterr().out


# --------------------------------------------- the UA and the 403 retry
#
# MEASURED 2026-08-29: both CI's probe and an independent probe from the
# app's sandbox got HTTP 403 with the 17-byte body `error code: 1010` —
# Cloudflare's UA-signature ban on r2.dev — while the app's Deno fetch
# reads the same objects successfully. A 403 can mean "your client's UA
# is on a scraper list", not "the bucket is private", and the retry
# exists to tell those apart in one run.

import io as _io
import urllib.error as _ue


class _Resp:
    def __init__(self, data: bytes):
        self._data = data
        self.headers = {"Content-Length": str(len(data))}

    def read(self, n=-1):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    class headers:  # overwritten in __init__
        pass


def _http_error(code: int, body: bytes = b"") -> _ue.HTTPError:
    return _ue.HTTPError("https://x", code, "err", {}, _io.BytesIO(body))


def test_the_primary_ua_names_the_tool_honestly():
    assert "frame-pull" in frame_pull.UA_PRIMARY
    assert "Mozilla" not in frame_pull.UA_PRIMARY


def test_a_403_earns_one_browser_ua_retry_and_says_so(capsys):
    seen = []

    def opener(url, ua, timeout=120):
        seen.append(ua)
        if ua == frame_pull.UA_PRIMARY:
            raise _http_error(403, b"error code: 1010")
        return _Resp(_mp4(64))

    data = frame_pull._fetch("https://x/clip.mp4", opener=opener)
    assert data == _mp4(64)
    assert seen == [frame_pull.UA_PRIMARY, frame_pull.UA_BROWSER]
    assert "UA-FILTERED, not private" in capsys.readouterr().out


def test_a_404_is_answered_once_because_a_different_ua_changes_nothing():
    seen = []

    def opener(url, ua, timeout=120):
        seen.append(ua)
        raise _http_error(404)

    with pytest.raises(_ue.HTTPError):
        frame_pull._fetch("https://x/clip.mp4", opener=opener)
    assert seen == [frame_pull.UA_PRIMARY]


def test_a_403_under_both_identities_raises_the_403():
    with pytest.raises(_ue.HTTPError) as exc:
        frame_pull._fetch(
            "https://x/clip.mp4",
            opener=lambda url, ua, timeout=120: (_ for _ in ()).throw(
                _http_error(403, b"error code: 1010")
            ),
        )
    assert exc.value.code == 403


def test_the_error_body_becomes_a_bounded_printable_detail():
    assert frame_pull.http_detail(_http_error(403, b"error code: 1010\n")) == (
        "error code: 1010"
    )
    # Binary junk and control characters never reach a log line.
    # read(64) takes the first 64 raw bytes (5 junk + 59 A's); the filter
    # then drops the 3 non-printables.
    noisy = frame_pull.http_detail(_http_error(403, b"\x00\x01ok\x7f" + b"A" * 500))
    assert noisy == "ok" + "A" * 59
    assert frame_pull.http_detail(_ue.HTTPError("https://x", 403, "e", {}, None)) == ""


def test_the_detail_rides_in_the_report_beside_the_status(tmp_path):
    report = frame_pull.pull_clip(
        {"scene": "s", "output_key": "validation/out/ltx-001.mp4"},
        BASE,
        out_dir=str(tmp_path),
        fetch=lambda url: (_ for _ in ()).throw(_http_error(403, b"error code: 1010")),
        run=lambda args, **kw: _Result(0),
    )
    assert report["error"] == "fetch-failed:HTTP 403"
    assert report["detail"] == "error code: 1010"
