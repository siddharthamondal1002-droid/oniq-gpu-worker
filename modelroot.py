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

# Where the persistent volume actually is.
#
# RunPod mounts a serverless network volume at /runpod-volume. That is the
# documented path and almost certainly the right one — but "almost
# certainly" is a two-hour rebuild if it is wrong, because the value would
# be baked into an image that takes that long to cold-pull. So the path is
# DETECTED rather than asserted, from a short ordered list, and the choice
# is reported.
#
# The test is not "does this directory exist": the image itself could
# contain an empty /workspace, and mounting nothing there would then look
# like a mounted volume. A real mount is on a DIFFERENT filesystem from /,
# so the device id is what decides.
#
# An explicit MODEL_VOLUME_ROOT from the template always wins and is never
# second-guessed — an operator naming a path has answered the question.
_CANDIDATES = ("/runpod-volume", "/workspace", "/mnt/volume")


def _is_real_mount(path: str) -> bool:
    try:
        here = os.stat(path)
        root = os.stat("/")
    except OSError:
        return False
    if not os.path.isdir(path):
        return False
    return here.st_dev != root.st_dev


def _detect_volume_root() -> tuple:
    """(path, how). `how` records the reasoning for the worker's report."""
    explicit = os.environ.get("MODEL_VOLUME_ROOT")
    if explicit:
        return explicit, "MODEL_VOLUME_ROOT (template)"
    for candidate in _CANDIDATES:
        if _is_real_mount(candidate):
            return candidate, f"detected: {candidate} is a separate filesystem"
    # Nothing mounted. Return the documented default so the refusal names a
    # path an operator recognises rather than an empty string.
    return _CANDIDATES[0], "default (no mounted candidate found)"


VOLUME_ROOT, VOLUME_ROOT_SOURCE = _detect_volume_root()

# The deterministic layout the owner specified: <volume>/models/oniq/<family>/<dir>
ONIQ_TREE = os.path.join("models", "oniq")

# Baked into the image by the Dockerfile. Production only.
BAKED_ROOT = "/app/models"
PRODUCTION_COMPONENTS = ("ltx", "story", "piper")

# THE WORKER'S OWN CONTAINER DISK, and the third place weights can live.
#
# OWNER DECISION 2026-09-01, option B: no network volume, every datacenter.
# The reason is not storage cost, it is placement. Attaching a network
# volume narrows an endpoint's `locations` from ALL to that volume's single
# datacenter — and DETACHING DOES NOT WIDEN IT BACK (read off the console's
# Releases tab, releases #3 and #7; no API surface reports the field). The
# endpoint is then pinned for life, and the day that one datacenter's
# approved VRAM tier runs dry it has zero placement candidates and simply
# stops getting workers. That is what the daily endpoint recreation was
# working around: a new endpoint is born at locations: ALL.
#
# So the 17.74 GiB text encoder cannot live on a volume. It also cannot go
# back into the image: that was measured at 57.97 GiB (run 33434875038) and
# a hosted runner could not build it, with the last SUCCESSFUL publish
# leaving 5.73 GiB free of 71.61. Container disk is the remaining place,
# and the template already provisions 201 GB of it.
#
# THE COST IS A COLD START, and it is real: a worker that has never held
# these weights fetches them before its first clip, inside the endpoint's
# execution ceiling. A warm worker pays nothing. That trade was taken
# deliberately — an endpoint that is slow to start beats an endpoint that
# cannot get a GPU at all.
CACHE_ROOT = os.environ.get("MODEL_CACHE_ROOT") or "/app/cache"

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

# CACHE-RESIDENT PRODUCTION. A third class, and it is deliberately NOT
# "experimental" — this is the production text encoder every clip goes
# through.
#
# IT WAS VOLUME-RESIDENT UNTIL 2026-09-01. The owner moved it to container
# disk (option B) once the console's Releases tab showed that attaching any
# network volume permanently pins the endpoint to one datacenter. See
# CACHE_ROOT above for the whole finding; the short version is that storage
# and placement breadth are mutually exclusive on RunPod, and placement
# breadth is what keeps the product running.
#
# WHY IT LEFT THE IMAGE, owner directive 2026-08-31. Moving the checkpoint to
# LTX-Video-0.9.7-distilled — needed so its vae matches the pinned spatial
# upsampler — put the media image at 57.97 GiB. Measured: a hosted runner
# cannot build that (run 33434875038, and the last SUCCESSFUL publish left
# 5.73 GiB free of 71.61). The text encoder is the single largest component,
# so it moved here and the image came back to a size that builds.
#
# IT FAILS CLOSED, exactly like an experimental model, and that is a REAL
# CHANGE from how production components behave. ltx/story/piper resolve
# volume-first with the baked path as a fallback, so an unmounted or empty
# volume cannot break them. This one has no baked copy to fall back to. The
# owner accepted that dependency knowingly; the alternative was paying for a
# larger build runner or giving up the 13B checkpoint.
#
# A clip that cannot find its text encoder must REFUSE and say so. It must
# never quietly run with an untrained embedding, which is the failure a
# silent fallback would produce.
CACHE_RESIDENT = {
    "LTX_TEXT_ENCODER": {
        "family": "ltx",
        # Named for the CHECKPOINT, not just "text-encoder". The encoder and
        # the transformer share an embedding space; hydrating one checkpoint's
        # encoder beside another's transformer is a silent quality failure,
        # and a shared directory name is how that would happen.
        "directory": "text-encoder-0.9.7-distilled",
        "repo": "Lightricks/LTX-Video-0.9.7-distilled",
        "revision": "057509edea1493cae5e62e9d8f780ebda3fb4333",
        "licence": "LTX Open Weights 0.X (accepted by the owner 2026-08-31)",
        "allow": ["text_encoder/*"],
        # MEASURED, run 33436186203, from the registry's own byte counts over
        # exactly that pattern: 19,049,290,370 bytes = 17.74 GiB. Not derived
        # by subtracting known components from the pipeline total — this
        # number feeds modelhydrate's disk check, and an estimate there breaks
        # the gate it exists to feed.
        #
        # Corroborated independently: the last build that BAKED this component
        # (run 33327318610) logged "TEXT ENCODER COMPLETE: 6 file(s),
        # 19049290411 bytes" for the previous checkpoint. 41 bytes apart, so
        # the T5 encoder is the same model either side of the repoint.
        "download_gib": 17.74
    },
}


def spec_for(model_id: str):
    """The spec for a fetched model of EITHER class, or None.

    One lookup, because there are two registries and five call sites that
    used to index EXPERIMENTAL directly. A second registry that only some of
    them knew about would resolve for hydration and not for loading, or the
    reverse.
    """
    return EXPERIMENTAL.get(model_id) or CACHE_RESIDENT.get(model_id)


def known_ids() -> list:
    return sorted({**EXPERIMENTAL, **CACHE_RESIDENT})


def storage_class(model_id: str) -> str:
    """'volume' or 'cache' — WHERE this model's weights are expected.

    The two classes fail differently and must not be conflated. A volume
    model with no volume is a configuration error the operator has to fix.
    A cache model with nothing on disk is normal on a cold worker and is
    fixed by fetching, which is why `ensure` exists.
    """
    if model_id in CACHE_RESIDENT:
        return "cache"
    if model_id in EXPERIMENTAL:
        return "volume"
    raise ModelUnavailable(
        "MODEL_UNKNOWN",
        f"{model_id!r} is not a known model; known ids are "
        + ", ".join(known_ids()),
    )


class ModelUnavailable(Exception):
    """A named refusal. `code` is the whole diagnosis."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def oniq_root() -> str:
    return os.path.join(VOLUME_ROOT, ONIQ_TREE)


def cache_root() -> str:
    """The same deterministic tree, on the worker's own container disk.

    Deliberately the SAME layout as the volume — <root>/models/oniq/<family>/
    <dir> — so a model can move between the two by changing which registry
    it is in and nothing else. The directory names are the contract; the
    root is where that contract is satisfied today.
    """
    return os.path.join(CACHE_ROOT, ONIQ_TREE)


def root_for(model_id: str) -> str:
    """The tree this model's weights belong under."""
    return cache_root() if storage_class(model_id) == "cache" else oniq_root()


def volume_mounted() -> bool:
    """Is the volume actually there?

    An explicitly configured root is trusted as a directory check, because
    an operator who named a path has made the decision. An auto-detected
    one must be a real mount: a plain directory baked into the image would
    otherwise read as a volume and every model would resolve to a path
    with nothing in it.
    """
    if os.environ.get("MODEL_VOLUME_ROOT"):
        return os.path.isdir(VOLUME_ROOT)
    return _is_real_mount(VOLUME_ROOT)


def model_dir(model_id: str) -> str:
    """The path a model WOULD occupy. Pure; touches no disk."""
    spec = spec_for(model_id)
    if spec is None:
        raise ModelUnavailable(
            "MODEL_UNKNOWN",
            f"{model_id!r} is not a fetched model; known ids are "
            + ", ".join(known_ids()),
        )
    return os.path.join(root_for(model_id), spec["family"], spec["directory"])


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
    spec = spec_for(model_id)
    if spec is None:
        raise ModelUnavailable(
            "MODEL_UNKNOWN",
            f"{model_id!r} is not a fetched model; known ids are "
            + ", ".join(known_ids()),
        )
    # ONLY VOLUME MODELS NEED A VOLUME. A cache-resident model lives on the
    # worker's own container disk, which is always there; demanding a mount
    # for it would refuse every clip on a correctly configured endpoint that
    # deliberately has no volume attached.
    if storage_class(model_id) == "volume" and not volume_mounted():
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


def ensure(model_id: str, hydrator) -> str:
    """Resolve a CACHE-resident model, fetching it first if it is absent.

    THE COLD WORKER IS THE NORMAL CASE, not an error. A worker that has
    never held these weights has an empty cache, and refusing there would
    refuse every first clip on every fresh worker — which under option B is
    every worker RunPod has just placed in a new datacenter.

    So: resolve, and on MODEL_NOT_HYDRATED alone, fetch and resolve again.
    Every OTHER refusal propagates untouched. MODEL_CORRUPT in particular
    must NOT trigger a re-fetch here: a checkpoint whose manifest does not
    verify is a fault to report, and quietly re-downloading over it would
    turn a diagnosable corruption into an intermittent one.

    VOLUME models are refused outright. Their whole contract is that an
    operator attaches storage and hydrates deliberately, before anything is
    dispatched; fetching one mid-job would spend a booted worker on a
    download the preflight exists to make unnecessary.

    `hydrator` IS REQUIRED, not defaulted to an import. This module imports
    nothing but json and os, and a test asserts that over the AST — because
    where weights load from is on the owner's list of things a caller may
    not steer, and the cheapest way to keep that true is to keep every
    module that handles a job out of this one's import graph. The caller
    supplies the fetcher; modelroot stays a registry and a path calculator.
    """
    if storage_class(model_id) != "cache":
        raise ModelUnavailable(
            "MODEL_NOT_CACHEABLE",
            f"{model_id!r} is volume-resident; hydrate it deliberately with "
            "model_hydrate before dispatching, rather than fetching it "
            "inside a job that is already being paid for.",
        )
    try:
        return resolve(model_id)
    except ModelUnavailable as first:
        if first.code != "MODEL_NOT_HYDRATED":
            raise
    hydrator(model_id)
    # RESOLVED AGAIN, NOT TRUSTED. The fetch's own return value would say
    # where it put the files; only resolve() says the marker is present, the
    # revision is the pinned one, and every file is the length the manifest
    # promises.
    return resolve(model_id)


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
        "volume_root_source": VOLUME_ROOT_SOURCE,
        "volume_candidates": list(_CANDIDATES),
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
