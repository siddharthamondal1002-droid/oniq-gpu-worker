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
| `storage.py`              | R2 by reference (bucket `oniq-gpu`), fails closed           |
| `handler.py`              | serverless handler, runtime ceiling, deterministic cleanup  |
| `runpod_client.py`        | CI harness client — never shipped in the image              |
| `validation/admission.py` | financial admission — pure functions, no network            |

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
  create one, and R2 credentials are not GitHub secrets — they live in
  the RunPod endpoint's environment.

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
