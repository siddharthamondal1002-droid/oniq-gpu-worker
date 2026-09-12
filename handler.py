"""RunPod serverless handler for the ONIQ GPU worker.

One workload (image_preprocess), R2 by reference, CUDA-only by default,
a hard runtime ceiling, an explicit output whitelist, and deterministic
cleanup in a finally block. There is no shell surface here: nothing that
spawns a process, evaluates a string, or deserializes an object — by
construction, and by the scan in CI and the tests that greps the worker
files for the forbidden call names.

The only writable location is a mkdtemp directory under TMPDIR; /app is
root-owned and merely readable by the uid-10001 runtime user.
"""

from __future__ import annotations

# FIRST, and above every project module on purpose. cudaenv sets
# PYTORCH_CUDA_ALLOC_CONF, which PyTorch reads exactly once when its CUDA
# allocator initialises; a value set after that point is present in the
# environment and ignored by the allocator, which is the worst kind of
# configuration because every report says it is on. Importing it here,
# before anything that could pull in torch, is what makes the setting
# real — and cudaenv.evidence() records the ordering rather than asserting
# it. Owner directive 2026-08-30: "Do not claim it is configured merely
# because it appears in source."
import cudaenv  # noqa: F401  (imported for its import-time effect)

import os
import shutil
import sys
import tempfile
import time

import audio
import contract
import preprocess
import storage
import storygen
import videogen


class Cleanup:
    """Deterministic removal of the job's temp directory."""

    def __init__(self, path):
        self.path = path

    def run(self) -> dict:
        if not self.path:
            return {"ok": True}
        try:
            shutil.rmtree(self.path)
            return {"ok": True}
        except Exception as exc:
            return {"ok": False, "error": type(exc).__name__}


def _ceiling(op: str) -> int:
    """model_probe downloads its checkpoint at job time; production never
    does. One ceiling for the benchmark, production's untouched."""
    if op == "model_probe":
        return contract.PROBE_RUNTIME_CEILING_SECONDS
    return contract.RUNTIME_CEILING_SECONDS


def _check_deadline(started: float, op: str = "") -> None:
    ceiling = _ceiling(op)
    if time.monotonic() - started > ceiling:
        raise contract.ContractError(
            "runtime-exceeded",
            f"job exceeded the {ceiling}s ceiling",
        )


def _error(code: str, message: str) -> dict:
    return contract.filter_output(
        {"ok": False, "code": code, "error": message}
    )


# Every module this image ships sits beside handler.py under /app, and
# nothing installed from PyPI does. That is what lets `_own_refusal`
# recognise ONIQ's own exceptions WITHOUT a list to forget to extend —
# and forgetting to extend one is exactly what happened: six shipped
# refusal classes were added after the except-chain below was written and
# not one of them was added to it.
_OWN_DIR = os.path.dirname(os.path.abspath(__file__))

# A refusal's detail travels into an error body and a log on the caller's
# side. ONIQ's own messages are short by construction; the bound is here
# so that stays true of one written later.
MAX_REFUSAL_DETAIL = 400

# A path is the whole diagnosis of an OSError and is not free-form text,
# but it is still unbounded, so it is bounded here.
MAX_REFUSAL_PATH = 200


def _is_own_exception(exc: BaseException) -> bool:
    """Was this class defined in THIS worker, rather than a dependency?"""
    module = sys.modules.get(type(exc).__module__)
    path = getattr(module, "__file__", None)
    if not path:
        return False
    return os.path.dirname(os.path.abspath(path)) == _OWN_DIR


def _code_for(exc: BaseException) -> str:
    """CheckpointInconsistent -> checkpoint-inconsistent.

    Only for a refusal that carries no code of its own. Deriving it from
    the class name means a new one is reportable the day it is written,
    rather than on the day somebody remembers to name it here.
    """
    name = type(exc).__name__
    out = []
    for index, char in enumerate(name):
        if char.isupper() and index:
            out.append("-")
        out.append(char.lower())
    return "".join(out)


def _own_refusal(exc: BaseException):
    """(code, detail) if ONIQ raised this, else None.

    THE MESSAGE IS OURS, SO IT MAY TRAVEL. The rule the except-chain below
    protects is that text from a DEPENDENCY never reaches the caller,
    because nobody here wrote it and it could carry anything. A refusal
    this repository raises is the opposite: the message is the diagnosis,
    written here, and `storage.StorageError` has returned its own since
    the beginning — so this extends an existing precedent rather than
    relaxing a rule.

    Every message of the six classes that reach this path was read before
    it was widened (2026-09-12): they carry object keys, paths, byte
    counts, digests and revisions. The two that embed a caught exception
    wrap `storage.StorageError`, whose own docstring promises it "never
    carries key material".

    Three shapes exist and all three are read, because a refusal that
    reports `None: None` is no better than the class name it replaced:
    `.code`/`.message` (ContractError, StorageError, GpuUnavailable,
    ConcatRefused), `.code`/`.detail` (WeightsUnavailable,
    ModelUnavailable), `.state`/`.detail` (HydrationRefused). A bare one
    such as CheckpointInconsistent falls back to its class name and
    `str(exc)`.
    """
    if not _is_own_exception(exc):
        return None
    code = getattr(exc, "code", None) or getattr(exc, "state", None) or _code_for(exc)
    detail = getattr(exc, "message", None) or getattr(exc, "detail", None) or str(exc)
    return str(code)[:80], str(detail)[:MAX_REFUSAL_DETAIL]


def _foreign_detail(exc: BaseException) -> str:
    """What may be said about an exception ONIQ did not write.

    The class name, as before — its message is a dependency's text and
    could carry anything.

    OSError IS THE ONE EXCEPTION, and it is a measured one. `errno` and
    `filename` are STRUCTURED fields the operating system sets, not
    free-form text, and the path is the entire diagnosis: on 2026-09-12
    the first still of job a7b9c3b9 came back `PermissionError` with
    nothing else, and which directory could not be written was
    unknowable from outside the container.
    """
    name = type(exc).__name__
    if not isinstance(exc, OSError):
        return name
    parts = [name]
    if exc.errno is not None:
        parts.append(f"errno={exc.errno}")
    if exc.filename:
        parts.append(f"path={str(exc.filename)[:MAX_REFUSAL_PATH]}")
    return " ".join(parts)


def handle(event) -> dict:
    started = time.monotonic()
    workdir = None
    cleanup_result = None
    try:
        job = contract.validate_job(
            event.get("input") if isinstance(event, dict) else None
        )
        storage.require_configured()

        workdir = tempfile.mkdtemp(prefix="oniq-gpu-")
        input_path = f"{workdir}/input.bin"
        if job["op"] == "model_hydrate":
            # No GPU, no artifact, no upload. It puts a checkpoint on the
            # persistent volume so that every later change to that model is
            # a configuration edit rather than a 25 GiB image rebuild.
            import modelhydrate

            record = modelhydrate.hydrate(job["model"])
            _check_deadline(started, job["op"])
            return contract.filter_output({**record, "ok": True,
                                           "cleanup_ok": True})

        if job["op"] == "story_generate":
            # Text in the response, no artifact: nothing to upload.
            metrics = storygen.run(job)
            _check_deadline(started, job["op"])
            return contract.filter_output({**metrics, "cleanup_ok": True})

        if job["op"] in ("video_generate", "audio_mux", "video_concat", "model_probe"):
            output_path = f"{workdir}/output.mp4"
        elif job["op"] == "image_generate":
            output_path = f"{workdir}/output.{contract.IMAGE_GEN_FORMAT}"
        else:
            output_path = f"{workdir}/output.{job['params']['format']}"

        if job["op"] == "video_concat":
            # Every segment is a bounded download in declared order; the
            # deadline is re-checked between fetches so a slow bucket can
            # never carry the job past the ceiling.
            segment_paths = []
            for index, key in enumerate(job["params"]["segment_keys"]):
                seg_path = f"{workdir}/seg{index:03d}.mp4"
                storage.download(key, seg_path, contract.MAX_INPUT_BYTES)
                segment_paths.append(seg_path)
                _check_deadline(started, job["op"])
            metrics = videogen.run_concat(job, segment_paths, output_path)
        elif job["op"] == "image_generate":
            # Text-only, UNLESS the job named a canonical character
            # reference. The engine draws from the prompt on this worker's
            # own GPU either way; a reference makes the draw start partway
            # from that person instead of from noise.
            #
            # THE KEY IS ALREADY PROVEN by the time it reaches here: the
            # contract pins it to the server-owned story/ref/ prefix and a
            # bounded id, so this download can only ever name a published
            # canonical reference — not another user's still, not a clip, not
            # anything else in the bucket. The bytes are then bounded by the
            # same MAX_INPUT_BYTES every other input is, and decoded by the
            # same magic-byte-and-pixel-bounded decoder.
            reference_path = None
            reference_key = job["params"].get("reference_key")
            if reference_key:
                reference_path = f"{workdir}/reference.img"
                storage.download(reference_key, reference_path, contract.MAX_INPUT_BYTES)
                _check_deadline(started, job["op"])
            metrics = videogen.run_image(job, output_path, reference_path=reference_path)
        else:
            storage.download(job["input_key"], input_path, contract.MAX_INPUT_BYTES)
            _check_deadline(started, job["op"])

            if job["op"] == "video_generate":
                metrics = videogen.run(job, input_path, output_path)
            elif job["op"] == "model_probe":
                # The benchmark path. It downloads a candidate checkpoint at
                # job time, which production never does — and it times that
                # download as its own phase, because on a 14B candidate the
                # fetch is expected to cost more than the inference and
                # folding it into "model load" would misreport every row.
                import modelprobe

                try:
                    metrics = modelprobe.run(job, input_path, output_path)
                except modelprobe.ProbeStop as stop:
                    # A FAILED PROBE IS STILL A MEASUREMENT. The generic
                    # handler below would answer "unexpected-exception:
                    # ProbeStop" — the stage, the detail and every timing and
                    # byte count taken before it broke, all discarded, on a
                    # GPU that was rented and billed regardless. For a
                    # benchmark that is the one thing that must not happen,
                    # so the partial report comes back with the failure.
                    return contract.filter_output({
                        "ok": False,
                        "code": stop.failure,
                        "error": stop.detail,
                        **stop.report,
                    })
            elif job["op"] == "audio_mux":
                metrics = audio.run(job, input_path, output_path)
            else:
                metrics = preprocess.run(job, input_path, output_path)
        _check_deadline(started, job["op"])

        storage.upload(output_path, job["output_key"])
        _check_deadline(started, job["op"])

        return contract.filter_output(
            {
                "ok": True,
                "op": job["op"],
                "output_key": job["output_key"],
                "duration_ms": int((time.monotonic() - started) * 1000),
                **metrics,
            }
        )
    except contract.ContractError as exc:
        return _error(exc.code, exc.message)
    except storage.StorageNotConfigured as exc:
        return _error(exc.code, exc.message)
    except storage.StorageError as exc:
        return _error(exc.code, exc.message)
    except preprocess.GpuUnavailable as exc:
        return _error(exc.code, exc.message)
    except storygen.StoryModelUnavailable as exc:
        # LOCAL_MODEL_UNAVAILABLE. Never a reason to call a provider.
        return _error("local-model-unavailable", str(exc))
    except videogen.ConcatRefused as exc:
        return _error(exc.code, exc.message)
    except Exception as exc:
        # ONIQ'S OWN REFUSALS CARRY THEIR MESSAGE; NOTHING ELSE DOES.
        #
        # The clauses above name six refusal classes. Six MORE are shipped
        # and named nowhere — CheckpointInconsistent, WeightsUnavailable,
        # ModelUnavailable, HydrationRefused, ReferenceUnsupported and
        # OutOfMemory — so every one of them arrived here and was reduced
        # to its class name. Two of those classes say in their own
        # docstring that "`code` is the whole diagnosis", and the code was
        # the thing being dropped.
        #
        # It cost a day. `ltxcaps.CheckpointInconsistent` is raised with
        # five DISTINCT messages — a missing directory, an unreadable
        # model_index.json, a non-LTX pipeline, contradictory distillation
        # evidence, and missing components — and on 2026-09-12 job
        # a7b9c3b9 reported the bare word `CheckpointInconsistent` three
        # times. Which of the five had happened was not knowable from
        # outside the container, and is not in the worker's log either:
        # the only print here is the cleanup line.
        #
        # A DIAGNOSTIC MAY NOT FALL BACK TO THE THING IT WAS BUILT TO
        # EXPLAIN. The same shape has now cost this project three
        # investigations: Firebase's `auth/internal-error` hiding
        # `customData.serverResponse`, `vertexPost` reporting `http 404`
        # instead of Google's sentence, and this.
        own = _own_refusal(exc)
        if own is not None:
            return _error(own[0], own[1])
        return _error("unexpected-exception", _foreign_detail(exc))
    finally:
        cleanup_result = Cleanup(workdir).run()
        if not cleanup_result["ok"]:
            print(
                "cleanup-failed:",
                cleanup_result.get("error", "unknown"),
                flush=True,
            )


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handle})
