"""Frames from the clips a run actually produced — LTX quality, seen.

WHY THIS EXISTS. Every LTX decision so far has been made from metadata:
a COMPLETED status, a frame count, an inference time, a byte count. None
of those can tell you whether the character turned their head or whether
the model smeared the plate for four seconds. Owner directive
2026-08-29: quality judgements are made from ACTUAL FRAMES, and the
route from generation to a human eye has to be automatic, or it will not
happen when it matters.

    LTX generation -> R2 -> frame extraction -> CI artifact -> inspection

WHAT THIS MODULE MAY DO, and the boundary is the point. It READS
already-generated validation artifacts over the bucket's PUBLIC read
base and nothing else. It holds no credential, signs nothing, and
writes nothing to R2 — the bucket's write credentials live in the RunPod
endpoint's environment and gate 6 of the workflow keeps them out of CI.

R2_PUBLIC_BASE_URL IS PUBLIC CONFIGURATION, not a secret: it is the same
read base the application already fetches artifacts from with no
Authorization header. It is carried as a repository VARIABLE rather than
a secret so that stays true and visible.

    But a URL that carries authority is not a public base, and the
    difference is invisible to a human pasting one in. A presigned S3 or
    R2 URL is a credential wearing a URL's clothes — the signature rides
    in the query string. So a base with ANY query, fragment or userinfo
    is REFUSED here rather than used, and the refusal names the reason.
    That makes "never use it for authentication" a property of the code
    instead of a rule somebody has to remember.

NOTHING HERE CAN COST MONEY. No RunPod call, no job submission, no
endpoint mutation. It runs after the spend job's work is finished and
reads what that work left behind, so a failure to fetch a frame can
never fail an authorized generation that already succeeded — the
artifacts are still in the bucket and the run's economics are still
recorded.

Every decision is a pure function with the I/O injected, so the whole
path is proved offline by tests/test_frame_pull.py against fakes. The
owner's rule stands: do not launch a GPU job to test plumbing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

# The manifest the spend run leaves behind, naming what it generated.
MANIFEST = "artifact-manifest.json"

# Where extracted frames land for upload-artifact to collect.
FRAME_DIR = "frames"

# How many stills to pull out of one clip. Six across ~4.04s samples
# roughly every 0.67s, which is enough to see whether anything MOVED
# between them — the question a single thumbnail cannot answer.
FRAMES_PER_CLIP = 6

# A contact sheet as well as the individual frames: motion reads far
# better as a strip than as six files somebody has to open in order.
SHEET_NAME = "sheet"

# Bounded read. A validation clip is a few hundred KB; anything at this
# size is not a clip and should not be pulled into a runner's memory.
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024


# What a status usually means for THIS bucket, so a refusal points at a
# fix instead of at a number. Guidance only — never a claim of fact.
HTTP_HINTS = {
    403: "the bucket has no public read enabled (r2.dev dev URL off, or "
         "no public access), or this object is not public",
    404: "no object at that key under this base — check the key, and "
         "whether the base already includes the bucket name",
    401: "the base is not a public read base; it wants credentials",
}


class FramePullError(Exception):
    """A refusal. `code` is stable so tests and logs can name it."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


# ------------------------------------------------------------ the base


def normalise_base(raw: str) -> str:
    """The public read base, or a refusal saying exactly what is wrong.

    Refuses anything that could carry authority. A presigned URL's
    signature lives in the query string, so a base with a query is not a
    base — it is one object's credential, and using it would silently
    turn this module into an authenticated reader.
    """
    base = (raw or "").strip()
    if not base:
        raise FramePullError(
            "base-unset",
            "R2_PUBLIC_BASE_URL is not set; frames cannot be retrieved",
        )
    parts = urllib.parse.urlsplit(base)
    if not parts.scheme:
        # MEASURED 2026-08-29: this variable was first set to the literal
        # string "on", the shape the three motion routing flags take. A
        # flag-shaped value is the likely mistake here, so it gets its own
        # message rather than the confusing "scheme '' is not https".
        raise FramePullError(
            "base-not-a-url",
            f"{base!r} is not a URL — this is a read base like "
            "https://pub-<id>.r2.dev, not an on/off flag",
        )
    if parts.scheme != "https":
        raise FramePullError("base-not-https", f"scheme {parts.scheme!r} is not https")
    if not parts.netloc:
        raise FramePullError("base-no-host", "no host in the base URL")
    if "@" in parts.netloc:
        raise FramePullError("base-has-userinfo", "the base carries userinfo")
    if parts.query:
        raise FramePullError(
            "base-has-query",
            "the base carries a query string — that is a presigned URL, "
            "which is a credential, not a public read base",
        )
    if parts.fragment:
        raise FramePullError("base-has-fragment", "the base carries a fragment")
    return base.rstrip("/")


def safe_host(base: str) -> str:
    """The host alone, for logging. The path may name a bucket layout
    nobody needs in a public log; the host is enough to prove which
    origin was read."""
    return urllib.parse.urlsplit(base).netloc


def public_url(base: str, key: str) -> str:
    """A read URL for one already-generated object.

    The key comes from the run's OWN manifest — a value this repository
    computed — never from a caller. It is still checked, because a key
    that could open a `..` or a scheme would make this a fetcher of
    arbitrary URLs instead of a fetcher of one bucket's objects.
    """
    k = (key or "").strip()
    if not k:
        raise FramePullError("key-empty", "no output key")
    if k.startswith("/") or ".." in k or "://" in k or "\\" in k:
        raise FramePullError("key-unsafe", f"refusing key {k!r}")
    return f"{normalise_base(base)}/{urllib.parse.quote(k)}"


# ------------------------------------------------------- the artifact


def verify_artifact(data: bytes, expected_bytes) -> None:
    """The bytes that ARRIVED must be the bytes the worker said it wrote.

    The same cross-check the application makes before it trusts a clip
    (gpuVideoCore.verifyStoredArtifact): a 200 carrying a truncated
    object, or somebody else's file at that key, is exactly what a
    status field cannot tell you. `ftyp` at offset 4 is an mp4's own
    claim about itself.
    """
    if not data:
        raise FramePullError("artifact-empty", "the download was empty")
    if isinstance(expected_bytes, int) and expected_bytes > 0:
        if len(data) != expected_bytes:
            raise FramePullError(
                "artifact-size-mismatch",
                f"downloaded {len(data)} bytes, worker reported {expected_bytes}",
            )
    if data[4:8] != b"ftyp":
        raise FramePullError("artifact-not-mp4", "no ftyp box at offset 4")


def frame_times(video_seconds, count: int = FRAMES_PER_CLIP) -> list:
    """When to sample, in seconds.

    Evenly spaced across the clip and deliberately NOT starting at 0 or
    ending at the last frame: frame 0 is the conditioning plate the
    model was handed, so including it would flatter the result — the
    interesting question is what the frames AFTER it look like. The end
    is stepped back because seeking exactly to the duration lands past
    the final frame on some decoders.
    """
    try:
        total = float(video_seconds)
    except (TypeError, ValueError):
        total = 0.0
    if total <= 0 or count < 1:
        return []
    step = total / (count + 1)
    return [round(step * (i + 1), 3) for i in range(count)]


def ffmpeg_frame_args(src: str, when: float, dest: str) -> list:
    """One PNG at one timestamp. `-ss` before `-i` seeks fast, `-frames:v
    1` takes exactly one, `-y` overwrites a rerun's leftovers."""
    return [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
        "-ss", f"{when}", "-i", src, "-frames:v", "1", dest,
    ]


def ffmpeg_sheet_args(src: str, dest: str, count: int = FRAMES_PER_CLIP) -> list:
    """One contact strip: `count` frames spread across the clip, tiled in
    a row. Motion reads as a strip in a way it does not as six files."""
    return [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", src,
        "-vf",
        f"select='not(mod(n\\,{max(1, 97 // count)}))',scale=352:-1,"
        f"tile={count}x1",
        "-frames:v", "1", "-fps_mode", "vfr", dest,
    ]


# --------------------------------------------------------- the manifest


def video_rows(rows) -> list:
    """The rows that named a clip. Anything else in a run's economics —
    a plate, a preprocess, an audio mux — has no frames to pull."""
    out = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        if row.get("op") != "video_generate":
            continue
        if not row.get("output_key"):
            continue
        out.append(row)
    return out


def write_manifest(rows, path: str = MANIFEST) -> dict:
    """What this run generated, in the shape frame_pull reads back.

    Deliberately carries no price and no credential — only what is
    needed to find an object and prove it arrived intact.
    """
    manifest = {
        "clips": [
            {
                "scene": row.get("scene"),
                "output_key": row.get("output_key"),
                "output_bytes": row.get("output_bytes"),
                "video_seconds": row.get("video_seconds"),
                "frames": row.get("frames"),
                "fps": row.get("fps"),
                "resolution": row.get("resolution"),
                "model": row.get("model"),
                "job_id": row.get("job_id"),
            }
            for row in video_rows(rows)
        ]
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


def read_manifest(path: str = MANIFEST) -> list:
    try:
        with open(path, encoding="utf-8") as handle:
            return list(json.load(handle).get("clips") or [])
    except FileNotFoundError:
        return []


# ------------------------------------------------------------- the pull


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=120) as response:
        length = response.headers.get("Content-Length")
        if length and int(length) > MAX_ARTIFACT_BYTES:
            raise FramePullError(
                "artifact-too-large", f"{length} bytes exceeds the read bound"
            )
        return response.read(MAX_ARTIFACT_BYTES + 1)


def pull_clip(
    clip: dict,
    base: str,
    out_dir: str = FRAME_DIR,
    *,
    fetch=_fetch,
    run=subprocess.run,
    makedirs=os.makedirs,
    write=None,
) -> dict:
    """One clip: read it, prove it, cut frames out of it.

    Returns a report row rather than raising, EXCEPT for a refusal about
    the base itself — one clip that cannot be fetched must not lose the
    frames of the four that can.
    """
    key = clip.get("output_key")
    url = public_url(base, key)
    stem = (clip.get("scene") or key or "clip").replace("/", "-")
    report = {"scene": clip.get("scene"), "output_key": key, "artifacts": []}
    try:
        data = fetch(url)
        verify_artifact(data, clip.get("output_bytes"))
    except FramePullError as exc:
        report["error"] = exc.code
        return report
    except urllib.error.HTTPError as exc:
        # THE STATUS, NOT JUST "IT FAILED". Measured 2026-08-29: the first
        # real pull reported `fetch-failed:HTTPError` and that sentence
        # contains no diagnosis — 403 (the bucket has no public r2.dev
        # access enabled) and 404 (the key is not where we think it is)
        # are completely different problems with completely different
        # fixes, and this said neither. It is the same mistake
        # story-still made when its throw discarded the engine's reason.
        # A status code is not a secret; the URL still never appears.
        report["error"] = f"fetch-failed:HTTP {exc.code}"
        report["hint"] = HTTP_HINTS.get(exc.code, "")
        return report
    except (urllib.error.URLError, OSError) as exc:
        # The class of failure, never the URL: an error string can carry
        # a redirect target nobody meant to publish.
        report["error"] = f"fetch-failed:{type(exc).__name__}"
        return report

    makedirs(out_dir, exist_ok=True)
    local = os.path.join(out_dir, f"{stem}.mp4")
    writer = write or _write_file
    writer(local, data)
    report["bytes"] = len(data)
    report["artifacts"].append(os.path.basename(local))

    # The worker's reported duration when there is one; otherwise measure
    # it. A key named by hand carries no report, and guessing a duration
    # would sample the wrong moments of somebody else's clip.
    seconds = clip.get("video_seconds")
    if not seconds:
        seconds = probe_seconds(local, run=run)
        report["measured_seconds"] = seconds
    for index, when in enumerate(frame_times(seconds), start=1):
        dest = os.path.join(out_dir, f"{stem}-f{index}-{when}s.png")
        if run(ffmpeg_frame_args(local, when, dest)).returncode == 0:
            report["artifacts"].append(os.path.basename(dest))
    sheet = os.path.join(out_dir, f"{stem}-{SHEET_NAME}.png")
    if run(ffmpeg_sheet_args(local, sheet)).returncode == 0:
        report["artifacts"].append(os.path.basename(sheet))
    return report


def _write_file(path: str, data: bytes) -> None:
    with open(path, "wb") as handle:
        handle.write(data)


def ffprobe_args(src: str) -> list:
    return [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", src,
    ]


def probe_seconds(path: str, *, run=subprocess.run):
    """A clip's real duration, measured. Used when the manifest does not
    carry one — a key named by hand has no worker report behind it."""
    try:
        done = run(ffprobe_args(path), capture_output=True, text=True)
    except OSError:
        return None
    if getattr(done, "returncode", 1) != 0:
        return None
    try:
        seconds = float((done.stdout or "").strip())
    except ValueError:
        return None
    return seconds if seconds > 0 else None


def clips_from_keys(raw: str) -> list:
    """A manifest built from object keys named directly.

    THE FREE HALF OF THE PROOF. Objects a previous run already generated
    are still in the bucket, and pulling frames out of them costs nothing
    — no endpoint, no job, no GPU second. It exercises exactly the chain
    a paid canary needs (base -> URL -> download -> verify -> ffmpeg ->
    artifact) against real LTX output, which is the difference between
    believing the plumbing works and having seen it work.

    No output_bytes and no video_seconds: a key named by hand has no
    worker report behind it, so the size cross-check is skipped (the mp4
    magic still is not) and the duration is MEASURED with ffprobe rather
    than assumed.
    """
    clips = []
    for key in [k.strip() for k in (raw or "").split(",") if k.strip()]:
        clips.append({"scene": key.rsplit("/", 1)[-1].rsplit(".", 1)[0], "output_key": key})
    return clips


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "keys":
        clips = clips_from_keys(os.environ.get("FRAME_KEYS", ""))
        if not clips:
            print("frame-pull: FRAME_KEYS names no object — nothing to fetch")
            return 0
    else:
        clips = read_manifest()
    if not clips:
        print("frame-pull: no clips in this run's manifest — nothing to fetch")
        return 0
    try:
        base = normalise_base(os.environ.get("R2_PUBLIC_BASE_URL", ""))
    except FramePullError as exc:
        # LOUD, and still not a failure: the GPU work succeeded and its
        # artifacts are in the bucket. What is lost is the seeing.
        print(f"frame-pull REFUSED: {exc.code} — {exc.message}")
        print("Set the R2_PUBLIC_BASE_URL repository VARIABLE (public config,")
        print("never a secret, never a presigned URL) to retrieve frames.")
        return 0
    print(f"frame-pull: reading {len(clips)} clip(s) from {safe_host(base)}")
    # Made here, not only inside pull_clip: a run where every fetch fails
    # still has a report to write, and it belongs beside the frames.
    os.makedirs(FRAME_DIR, exist_ok=True)
    reports = [pull_clip(clip, base) for clip in clips]
    for report in reports:
        if report.get("error"):
            hint = report.get("hint")
            print(
                f"  {report['scene']}: NO FRAMES ({report['error']})"
                + (f" — {hint}" if hint else "")
            )
        else:
            print(
                f"  {report['scene']}: {report['bytes']} bytes, "
                f"{len(report['artifacts'])} artifact(s)"
            )
    with open(os.path.join(FRAME_DIR, "frames.json"), "w", encoding="utf-8") as handle:
        json.dump(reports, handle, indent=2, sort_keys=True)
    got = sum(1 for r in reports if not r.get("error"))
    print(f"frame-pull: {got}/{len(reports)} clip(s) retrieved and cut")
    write_summary(clips, reports)
    return 0


def summary_markdown(clips, reports) -> str:
    """The run's own report of what came back — artifact names and the
    measured evidence beside them.

    In the run page rather than only inside the downloaded zip, because
    the first question after a canary is "did the frames come back", and
    that should be answerable without downloading anything.
    """
    by_key = {c.get("output_key"): c for c in clips}
    lines = ["### LTX frames", "", "| scene | clip | measured | artifacts |", "|---|---|---|---|"]
    for report in reports:
        clip = by_key.get(report.get("output_key"), {})
        measured = (
            f"{clip.get('frames')}f @ {clip.get('fps')}fps, "
            f"{clip.get('video_seconds')}s, {clip.get('resolution')}"
        )
        if report.get("error"):
            names = f"**none — {report['error']}**"
        else:
            names = "<br>".join(report.get("artifacts") or []) or "none"
        lines.append(
            f"| {report.get('scene') or '—'} | `{report.get('output_key')}` "
            f"| {measured} | {names} |"
        )
    got = sum(1 for r in reports if not r.get("error"))
    lines += ["", f"{got}/{len(reports)} clip(s) retrieved. Download the "
                  "`ltx-frames` artifact to inspect them."]
    return "\n".join(lines) + "\n"


def write_summary(clips, reports) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(summary_markdown(clips, reports))
    except OSError:
        # A report that cannot be written is not a reason to fail a run
        # whose generation already succeeded.
        pass


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
