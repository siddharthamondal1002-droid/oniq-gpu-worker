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

import shutil
import tempfile
import time

import audio
import contract
import preprocess
import storage
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


def _check_deadline(started: float) -> None:
    if time.monotonic() - started > contract.RUNTIME_CEILING_SECONDS:
        raise contract.ContractError(
            "runtime-exceeded",
            f"job exceeded the {contract.RUNTIME_CEILING_SECONDS}s ceiling",
        )


def _error(code: str, message: str) -> dict:
    return contract.filter_output(
        {"ok": False, "code": code, "error": message}
    )


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
        if job["op"] in ("video_generate", "audio_mux"):
            output_path = f"{workdir}/output.mp4"
        else:
            output_path = f"{workdir}/output.{job['params']['format']}"

        storage.download(job["input_key"], input_path, contract.MAX_INPUT_BYTES)
        _check_deadline(started)

        if job["op"] == "video_generate":
            metrics = videogen.run(job, input_path, output_path)
        elif job["op"] == "audio_mux":
            metrics = audio.run(job, input_path, output_path)
        else:
            metrics = preprocess.run(job, input_path, output_path)
        _check_deadline(started)

        storage.upload(output_path, job["output_key"])
        _check_deadline(started)

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
    except Exception as exc:
        # Never echo arbitrary exception text to the caller: the class
        # name is diagnostic enough and cannot carry a credential.
        return _error("unexpected-exception", type(exc).__name__)
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
