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
| `storage.py`              | R2 by reference (bucket `oniq-gpu`), fails closed           |
| `handler.py`              | serverless handler, runtime ceiling, deterministic cleanup  |
| `runpod_client.py`        | CI harness client — never shipped in the image              |
| `validation/admission.py` | financial admission — pure functions, no network            |

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
