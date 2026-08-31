# LTX calibration fixtures — immutable

Recorded 2026-08-29. The three clips named in `fixtures.json` are the
ONLY real LTX output the validators have ever been calibrated against.
Every judgement the gates encode — what real motion looks like, what
identity drift looks like, what a scene being replaced outright looks
like — was made by a human looking at THESE frames. Regenerating or
overwriting them would recalibrate the gates against nothing: a fresh
generation carries a different outcome and no human verdict, and the
conditioning plate (`validation/input.jpg`, a cartoon bank office) has
since been deleted from the bucket, so the runs cannot be reproduced
even in principle. Fields the worker did not report are `"not
recorded"` — the worker does not report a seed at all, and the prompts
were not captured. An honest gap beats a plausible reconstruction.

## What each fixture shows

All three: 4.04s, 704x480 @ 24fps, LTX_VIDEO_2B on an NVIDIA RTX A5000,
conditioned on the same plate.

- **GOOD_MOTION** — `validation/out/ltx-001.mp4`, PASS. Real character
  motion across the full span with identity, cartoon style, composition
  and the in-frame BANK BOSS text all stable end to end. This is what
  the gates must let through.
- **IDENTITY_DRIFT** — `validation/audio-canary/final-001.mp4`,
  PARTIAL. Strongest motion of the three early on, then the officer's
  face reads as a different man by the back half and the in-frame text
  deforms. Carries a narration audio track (audio_mux output). This is
  the failure that motion metrics alone cannot see.
- **CONTENT_COLLAPSE** — `validation/video-test/ltx-001.mp4`, FAIL.
  Holds the plate about 1.5 seconds, then the entire scene is replaced
  by an unrelated photorealistic face — full content collapse, not weak
  motion. This is what the gates must never let through.

## R2 access posture (owner directive, stated plainly)

- **PUBLIC_R2_ARTIFACT_READ = CURRENT.** r2.dev serves these objects to
  normal clients with no credential. The 403s previously observed were
  Cloudflare's UA-signature filtering (error code 1010) refusing known
  scraper agents — a filter on WHO asked, not privacy. Measured
  2026-08-29 and recorded in `validation/frame_pull.py`: the probes
  that got the 17-byte 1010 body were python-urllib-shaped, while the
  app's own fetch read the same objects unauthenticated and succeeded.
- **SIGNED_PRIVATE_READ = AVAILABLE.** Built and tested in
  `validation/frame_pull.py` (the `signed` path: presigned, read-only,
  single-object, short-lived, never logged). Live use is backlogged
  pending an owner decision that must not break existing Story/GPU
  artifact playback.

## Why a battery can never overwrite a fixture

The battery writes only under `validation/out/shot-*`. None of the
three fixture keys sits in that namespace, and
`tests/test_fixtures.py` holds that line — a fixture key that ever
moved under the battery prefix fails the suite before it can fail the
calibration.

## Why a canary can no longer overwrite a fixture

Found in adversarial review, 2026-08-29: the phase-16 video canary and the
audio canary used to write `ltx-001.mp4` and `final-001.mp4` — the exact
basenames these fixtures live at — so one ordinary re-run with the matching
output prefix would have replaced the only real calibration data. Both
canaries were renamed (`ltx-canary-001.mp4`, `audio-final-001.mp4`), the
battery writes only `shot-*` names, and `tests/test_fixtures.py` asserts
against `spend_run`'s actual producible names, not against a constant.
