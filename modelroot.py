"""Where a model's weights live, and the refusal when they do not.

OWNER DIRECTIVE, 2026-08-30 — "MODEL WEIGHTS ARE DATA. SOFTWARE IS THE
IMAGE." Changing a checkpoint, a revision, a frame count or an offload
mode must never require a Docker rebuild again. Ten hours of 2026-08-29
and 2026-08-30 went on cold-pulling a 25 GiB image so that a value in a
table could change.

So there are two classes of model here, and they are deliberately NOT
symmetrical.

PRODUCTION components — ltx, story, piper — are baked into the image and
stay baked. They resolve volume-first with the baked path as fallback, so
attaching a volume cannot break a production job and neither can an empty
one. Nothing about the production path changes.

EXPERIMENTAL models — Hunyuan today, whatever is next tomorrow — live ONLY
on the persistent volume and FAIL CLOSED. There is no fallback, by
design: the one outcome worse than a Hunyuan job that cannot run is a
Hunyuan job that quietly runs LTX and gets compared against LTX. Every
failure below names itself, so a refusal is a diagnosis rather than a
mystery.

    MODEL_VOLUME_UNAVAILABLE   the volume is not mounted
    MODEL_NOT_HYDRATED         mounted, but this model was never fetched
    MODEL_CORRUPT              hydrated, but the manifest does not verify

`MODEL_VOLUME_ROOT` is set by the TEMPLATE, never by a job. The owner's
standing constraint is that a user may not choose GPU, provider, endpoint,
model, checkpoint, runtime, budget, precision or offloading strategy;
where weights are read from belongs on that list, so nothing in this
module takes caller input and no request-shaped value reaches it.
"""

from __future__ import annotations

import json
import os

# RunPod mounts a serverless network volume here. Overridable by the
# template only — a provider that changes its mount path should not need a
# code change — but never by a job.
VOLUME_ROOT = os.environ.get("MODEL_VOLUME_ROOT") or "/runpod-volume"

# The deterministic layout the owner specified: <volume>/models/oniq/<family>/<dir>
ONIQ_TREE = os.path.join("models", "oniq")

# Baked into the image by the Dockerfile. Production only.
BAKED_ROOT = "/app/models"
PRODUCTION_COMPONENTS = ("ltx", "story", "piper")

# Written by the hydrator as the LAST act of a successful fetch, so its
# presence means "complete", never "started". Anything else on disk
# without it is an interrupted download.
READY_MARKER = ".oniq-ready.json"

# Experimental models. The id is the contract; the path is derived, never
# typed twice.
EXPERIMENTAL = {
    "HUNYUAN_15_I2V_480_STEP": {
        "family": "hunyuan",
        "directory": "HunyuanVideo-1.5-480P-I2V-step-distill",
        "repo": (
            "hunyuanvideo-community/"
            "HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled"
        ),
        "revision": "854c04a4c8a53d990b418c7478f0802c0fc8c726",
        "licence": (
            "Tencent HunyuanVideo-1.5 Community License "
            "(no EU/UK/KR, incl. outputs)"
        ),
        # Every component the pipeline declares. Measured against the live
        # listing by the $0 model-bench read, where COVERAGE confirmed no
        # declared component falls outside these patterns — a missing
        # guider/ or feature_extractor/ surfaces as a load failure on a
        # rented card, which is the expensive place to find it.
        "allow": [
            "model_index.json", "transformer/*", "text_encoder/*",
            "tokenizer/*", "text_encoder_2/*", "tokenizer_2/*",
            "image_encoder/*", "feature_extractor/*", "scheduler/*",
            "vae/*", "guider/*",
        ],
        # MEASURED from the listing's own byte counts over exactly those
        # patterns, and confirmed on the wire: the 2026-08-30 probe
        # downloaded 34,636,996,520 bytes in 40.0 s.
        "download_gib": 32.26,
    },
}


class ModelUnavailable(Exception):
    """A named refusal. `code` is the whole diagnosis."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def oniq_root() -> str:
    return os.path.join(VOLUME_ROOT, ONIQ_TREE)


def volume_mounted() -> bool:
    """Is the volume actually there?

    A mount point that exists as a plain directory inside the image would
    pass a bare isdir(), so the check is on VOLUME_ROOT itself, which the
    image does not create.
    """
    return os.path.isdir(VOLUME_ROOT)


def model_dir(model_id: str) -> str:
    """The path a model WOULD occupy. Pure; touches no disk."""
    spec = EXPERIMENTAL.get(model_id)
    if spec is None:
        raise ModelUnavailable(
            "MODEL_UNKNOWN",
            f"{model_id!r} is not an experimental model; known ids are "
            + ", ".join(sorted(EXPERIMENTAL)),
        )
    return os.path.join(oniq_root(), spec["family"], spec["directory"])


def read_marker(path: str):
    """The completion record, or None. A marker that will not parse is
    treated as absent — a half-written JSON file is not a completion."""
    try:
        with open(os.path.join(path, READY_MARKER), encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    return doc if isinstance(doc, dict) else None


def verify(path: str, marker: dict) -> list:
    """Files the marker promises that disk does not deliver.

    Size, not content hash, on the per-request path: re-hashing 32 GiB on
    every job would cost more than the job. The hydrator does the full
    digest once, at write time, and records it; this is the cheap check
    that the files are still all there at the right lengths.
    """
    missing = []
    for name, size in sorted((marker.get("files") or {}).items()):
        full = os.path.join(path, name)
        try:
            actual = os.path.getsize(full)
        except OSError:
            missing.append(f"{name}: absent")
            continue
        if actual != size:
            missing.append(f"{name}: {actual} bytes, marker says {size}")
    return missing


def resolve(model_id: str) -> str:
    """Absolute path to an EXPERIMENTAL model's directory, or raise.

    Never falls back. A Hunyuan request that cannot find Hunyuan must not
    become an LTX run wearing Hunyuan's name in the report.
    """
    spec = EXPERIMENTAL.get(model_id)
    if spec is None:
        raise ModelUnavailable(
            "MODEL_UNKNOWN",
            f"{model_id!r} is not an experimental model; known ids are "
            + ", ".join(sorted(EXPERIMENTAL)),
        )
    if not volume_mounted():
        raise ModelUnavailable(
            "MODEL_VOLUME_UNAVAILABLE",
            f"{VOLUME_ROOT} is not mounted. Attach the network volume to "
            "the endpoint (networkVolumeId) before dispatching an "
            "experimental probe; there is deliberately no baked fallback.",
        )
    path = model_dir(model_id)
    marker = read_marker(path)
    if marker is None:
        raise ModelUnavailable(
            "MODEL_NOT_HYDRATED",
            f"no {READY_MARKER} at {path}. Run model_hydrate({model_id!r}) "
            "first; an unmarked directory is an interrupted download, not a "
            "usable model.",
        )
    if marker.get("revision") != spec["revision"]:
        raise ModelUnavailable(
            "MODEL_CORRUPT",
            f"hydrated revision {marker.get('revision')!r} is not the pinned "
            f"{spec['revision']!r}. Measuring a different checkpoint than the "
            "one recorded would make every number in the report wrong.",
        )
    broken = verify(path, marker)
    if broken:
        raise ModelUnavailable(
            "MODEL_CORRUPT",
            f"{len(broken)} file(s) do not match the manifest: "
            + "; ".join(broken[:5]),
        )
    return path


def resolve_production(component: str) -> str:
    """Production components only: volume first, baked image as fallback.

    Asymmetric with `resolve` on purpose. Production must survive an
    absent volume, an empty one, and a partially hydrated one, because
    attaching storage for an experiment must not be able to take LTX down.
    """
    if component not in PRODUCTION_COMPONENTS:
        raise ModelUnavailable(
            "MODEL_UNKNOWN",
            f"{component!r} is not a production component; known are "
            + ", ".join(PRODUCTION_COMPONENTS),
        )
    candidate = os.path.join(oniq_root(), component)
    try:
        with os.scandir(candidate) as it:
            if any(True for _ in it):
                return candidate
    except OSError:
        pass
    return os.path.join(BAKED_ROOT, component)


def resolve_production_file(name: str) -> str:
    """A production marker file (MODEL_ID, STORY_MODEL_ID, LTX_REPO)."""
    candidate = os.path.join(oniq_root(), name)
    if os.path.isfile(candidate):
        return candidate
    return os.path.join(BAKED_ROOT, name)


def where() -> dict:
    """What resolved to what, for the worker's own report — so a job that
    ran off the volume can be told from one that ran off the image without
    inferring it from how long the worker took to start."""
    report = {
        "volume_root": VOLUME_ROOT,
        "volume_mounted": volume_mounted(),
        "oniq_root": oniq_root(),
        "baked_root": BAKED_ROOT,
        "production": {
            c: resolve_production(c) for c in PRODUCTION_COMPONENTS
        },
        "experimental": {},
    }
    for model_id in sorted(EXPERIMENTAL):
        try:
            report["experimental"][model_id] = {"path": resolve(model_id)}
        except ModelUnavailable as exc:
            report["experimental"][model_id] = {"error": exc.code}
    return report
