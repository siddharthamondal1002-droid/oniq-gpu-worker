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
            # Text-only: there is no source object to fetch. The engine
            # draws from the prompt on this worker's own GPU.
            metrics = videogen.run_image(job, output_path)
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

                metrics = modelprobe.run(job, input_path, output_path)
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
