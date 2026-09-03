# oniq-gpu-worker

The ONIQ GPU worker: a RunPod serverless handler for exactly one workload,
`image_preprocess`, rebuilt from the design recorded in the ONIQ financial
ledger (`oniq-sparkle-pay` → `docs/video/ONIQ_AI_FINANCIAL_CONTROL.md`,
§§15–16h) after the original was lost with its build container.

This repository is deliberately separate from the ONIQ application.
The worker must never move into `oniq-sparkle-pay`.

## Shape

| file                      | role                                                        |
| ------------------------- | ----------------------------------------------------------- |
| `contract.py`             | bounded job contract, error codes, explicit output whitelist |
| `preprocess.py`           | the workload; CUDA-only by default via `run_gpu_op`         |
| `videogen.py`             | `video_generate` (LTX image-to-video), `image_generate` (LTX text-to-video, frame 0), `video_concat` — all CUDA |
| `audio.py`                | `audio_mux` — in-house narration muxed under a video, CPU   |
| `storygen.py`             | `story_generate` — ONIQ's own causal LLM, loaded and UNLOADED before LTX |
| `storage.py`              | R2 by reference (bucket `oniq-gpu`), fails closed           |
| `handler.py`              | serverless handler, runtime ceiling, deterministic cleanup  |
| `modelprobe.py`           | `model_probe` — the open-source video model benchmark, off the production path |
| `preview.py`              | bounded JPEG thumbnails returned in the reply, for a private bucket |
| `runpod_client.py`        | CI harness client — never shipped in the image              |
| `validation/admission.py` | financial admission — pure functions, no network            |

`story_generate` (2026-08-27, owner directive: Qwen3-8B conditionally
approved) runs ONIQ's own causal LLM. The weights are baked behind a
LICENCE GATE — the build reads the checkpoint's licence from the HF
metadata and refuses to download anything that is not Apache-2.0, so
"we believe it is Apache" is something the image cannot be built
without. At job time the model loads `local_files_only`, and the card is
handed back in a `finally`: load, generate, DELETE, `empty_cache`. LTX
peaked at 15.9GB of 24GB, so a story job that left the model resident
would not fail loudly — it would make the NEXT video job fail
mysteriously. The worker deliberately does NOT parse or validate the
story: the Story IR validator lives in the application, where an invalid
story must stop before any GPU job is planned.

`image_generate` (2026-08-27, fully in-house directive) is ONIQ's OWN
image engine, and it is deliberately not a second model: the same baked
LTX snapshot opened as `LTXPipeline` (text-to-video) instead of
`LTXImageToVideoPipeline`, sampled for the shortest legal clip, frame 0
kept as a PNG at the VIDEO canvas. No new weights, nothing downloaded,
`local_files_only` unchanged — the component that used to be an
outsourced image API is a different pipeline class over bytes this image
already carries. It is TEXT-ONLY (an `input_key` is refused, not
ignored) and carries NO watermark field: a conditioning frame is an
intermediate, and the mark belongs to the film the video stage burns it
into from the entitlement of record.

`audio_mux` (2026-08-26) speaks a narration with piper (the sha256-pinned
`en-us-ryan-high` voice, baked into the media image like the LTX weights)
via onnxruntime in-process, MEASURES it, refuses — never truncates — a
line longer than the video, normalizes to RMS −20 dBFS under a −1.5 dBFS
peak ceiling, and muxes an AAC track under the COPIED video stream with
PyAV. The output is verified by decoding: both streams, ≤0.25 s drift,
peak above −60 dBFS (mirroring the app's shared `videoAudio` verdict).
No process is ever spawned; the scan covers `audio.py` too.

Only the first four files (plus `requirements.txt`) go into the Docker
image; `COPY` names them individually and `.dockerignore` denies
everything else as a second lock.

## Invariants, all tested

- **CUDA-only by default.** `run_gpu_op(require_cuda=True)` raises
  `cuda-unavailable` rather than falling back; `ONIQ_ALLOW_CPU_FALLBACK=1`
  is the only escape hatch, and it is for test rigs.
- **R2 by reference.** Jobs name object keys; bytes never travel through
  the queue. Input size is bounded via `head_object` before the body.
- **Bounded input.** Strict field whitelist, bounded keys, bounded
  dimensions, decode pixel bound. No model field, no GPU field.
- **Non-root runtime.** `USER oniq:oniq` (uid/gid 10001) sits between the
  last `COPY` and `CMD`: `/app` is root-owned and read-only to the
  process. `PYTHONDONTWRITEBYTECODE=1` and `HOME=/home/oniq` keep a
  non-root process from writing where it cannot.
- **No shell surface.** No eval/exec/subprocess/os.system/pickle —
  enforced by a scan in CI and in the tests.
- **Runtime ceiling.** 900 seconds, refused (never clamped) above it.
- **Financial admission.** Reservations round UP to the cent, charge the
  FULL ceiling window, treat a null price as no capacity (never free),
  refuse in a fixed order with VRAM before price, and cap each job at
  $0.50. There is no default price argument and no historical price
  constant anywhere in this repository — a test greps for the stale
  figures by digit string.
- **Deterministic cleanup.** `Cleanup.run()` in a `finally`.

## Running the tests

```
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

CI (`worker-ci.yml`) additionally runs the suite as a real uid-10001 user
against a root-owned read-only `/app`, builds the image, and proves
`torch.version.cuda == "12.1"`, the imports, and the runtime uid — all
without CUDA hardware. `torch.cuda.is_available()` is printed, never
gated, on CPU runners: FALSE there carries no information about the image.

## The two workflows

- **`worker-ci.yml`** runs on every push. It holds no secrets and cannot
  spend money.
- **`gpu-validation.yml`** is `workflow_dispatch` ONLY and is the only
  workflow that may hold `RUNPOD_API_KEY`. Its default mode is `discover`:
  read-only, $0.00 — RAW provider responses first, parsed view second,
  because the parser is not trusted until the two agree. Spending requires
  BOTH typing `SPEND` into the run input AND approval of the `gpu-spend`
  environment. `cancel-in-progress` is false, the orphan sweep runs
  `if: always()` and fails the run when it cannot confirm zero, CI can
  only verify an endpoint (`min_workers=0` / `max_workers=1`), never
  create one, and R2 credentials reach exactly one job, from the
  `gpu-spend` environment, so touching the bucket costs a reviewer's
  click. (Amended 2026-09-01. The rule was "R2 credentials are not GitHub
  secrets — they live in the RunPod endpoint's environment", and it held
  until the gated text encoder had to move off the network volume, which
  permanently pins an endpoint to one datacenter. CI holds the
  HuggingFace token and no R2; a worker holds R2 and no HuggingFace
  token; nothing had both, so nothing could stage those weights. What is
  kept is the part that mattered: nothing which runs without a human
  reaches the bucket, and `worker-ci` — the workflow that fires on push —
  still holds no secret at all.)

## Media inference — measured baseline (2026-08-26)

The second workload, `video_generate`, ran once for real on the gated
pipeline (gpu-validation run #33, job `242948dc…-u2`): one LTX-Video 2B
image-to-video clip on a rented RTX 3090. Measured, not estimated:

| Figure                    | Value                                     |
| ------------------------- | ----------------------------------------- |
| Model (from MODEL_ID)     | `Lightricks/LTX-Video` (2B, 30 steps)     |
| Model load (baked, local) | 11.5 s                                    |
| Inference (97f, 704x480)  | 29.0 s CUDA                               |
| Encode (h264)             | 1.0 s                                     |
| Clip                      | 4.04 s @ 24 fps, 330,061 bytes            |
| Peak VRAM                 | 15,916 MB of 24,126                       |
| Execution / cost          | 45.5 s → $0.01 (of a $0.13 reservation)   |
| Per generated second      | $0.0025 ($0.15/min), ceiled               |
| Cold pull of media image  | 399 s delayTime, once per release         |

The full evidence chain, the TERMINATION_UNKNOWN honesty note
(`workersStandby: 1`), and the stop-after-one-job record live in the
financial ledger: `oniq-sparkle-pay/docs/video/ONIQ_AI_FINANCIAL_CONTROL.md`
§16o. Visual quality is judged by the owner from
`oniq-gpu/validation/video-test/ltx-001.mp4`, not by this repo.

## Open-source video model probe — measured (2026-08-29)

Owner directive 2026-08-29: benchmark the open-source image-to-video
candidates on an A5000 and name ONE, or `NO_MODEL_READY`. The probe is a
FIFTH operation, `model_probe`, added as a strict superset — no production
model, provider, price or user-visible behaviour was touched, and
`RUNTIME_CEILING_SECONDS` is still 900 for everything else.
`PROBE_RUNTIME_CEILING_SECONDS` (1800) applies to `model_probe` alone.

**Verdict: `NO_MODEL_READY`.** Nothing measured here beats the LTX-2B the
worker already runs, so nothing changes. Full report, with the frames:
`claude.ai/code/artifact/636d69f5-9b97-4402-8226-5e0f1d7e25f1`.

Infrastructure, measured on the worker rather than from the console:
container disk 214,748,364,800 B (200.0 GiB) total, 199.98 GiB free before
any download; VRAM 25,283,526,656 B (23.55 GiB); cold image pull 17.6 min
on the first job of a release and 8–16 s cached; production LTX-2B re-ran
at 43.2 s against 42.7 s historical — no regression from the new image.

| Candidate            | Outcome            | Inference    | Peak VRAM  | Cost  |
| -------------------- | ------------------ | ------------ | ---------- | ----- |
| `cogvideox-i2v` (5B) | clip returned      | 548,024 ms   | 13.58 GiB  | $0.05 |
| `ltx-13b`            | clip returned      | 306,653 ms   | 2.27 GiB   | $0.03 |
| `wan21-i2v-480p`     | executionTimeout   | —            | —          | $0.14 |
| `wan22-i2v-a14b`     | provider timeout   | —            | —          | $0.02 |
| `hunyuan-i2v`        | NOT_EVALUATED      | —            | —          | $0.00 |

Human inspection against the shared reference, per axis:

- `cogvideox-i2v` — ACTION/SCENE/CONTENT/TEMPORAL **PASS**, IDENTITY
  **PARTIAL**. The best of the four and still not good enough to switch to
  at 9.1 minutes for 49 frames.
- `ltx-13b` — SCENE **PASS**, ACTION/TEMPORAL/CONTENT **PARTIAL**,
  IDENTITY **FAIL**.
- The two Wan candidates produced no clip at all, so they are UNMEASURED,
  not bad: `wan21` was cut by the 1800 s ceiling and `wan22` by a
  provider-initiated retry at 157 s. Neither result says anything about
  the model's quality, and neither is recorded as a failure of it.
- `hunyuan-i2v` was never probed — ARCHITECTURE_NOT_RESOLVED.

Two things worth keeping. **A prediction of mine that the measurement
refuted:** I expected sequential CPU offload to be prohibitively slow, and
LTX-13B (13B params, 97 frames, sequential) finished in 6.0 min using
2.27 GiB while CogVideoX (5B, 49 frames, model offload) took 9.8 min. Peak
VRAM under sequential offload is a submodule, not the transformer, so
parameter count stops predicting either footprint or wall time. **The
bucket is private and `R2_PUBLIC_BASE_URL` is misconfigured**, so the clips
could not be fetched for inspection; `preview.py` returns bounded
thumbnails in the job reply instead, which is why there are frames to look
at at all. That is a durable fix, not a workaround for one run.

Total spend for the whole benchmark: **$0.30**, zero orphans
(`in_progress 0, in_queue 0, running 0` at the sweep).

## Owner actions this repo cannot perform

1. Add the `RUNPOD_API_KEY` repository secret (Actions).
2. Create the `gpu-spend` GitHub environment with required reviewers —
   that reviewer approval is the second half of the spend gate.
3. Create the RunPod serverless endpoint (min 0 / max 1, RTX 3090,
   Secure Cloud) and set the three `R2_*` variables in its environment.
4. Mint the scoped R2 credentials for bucket `oniq-gpu` (separate from
   `oniq-chat-media`, on purpose).

## Absolute rules

Discover first; discovery costs $0. Never reuse a historical hourly
price — every reservation starts from a price quoted in the same run.
Never provision without live price + financial admission + the literal
`SPEND` input + `gpu-spend` approval. A worker that cannot confirm its
own termination reports UNKNOWN, and UNKNOWN is never converted to
success.
