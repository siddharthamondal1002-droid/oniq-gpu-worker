# syntax=docker/dockerfile:1.7
# ONIQ GPU worker image.
#
# Structure is the security posture:
# - COPY names every shipped file individually — there is no COPY . .,
#   so the image cannot receive a stray .env even if .dockerignore were
#   wrong (.dockerignore is a second lock, not the only one). The set is
#   not a number to remember: test_workflow_gates derives handler.py's
#   own first-party import closure and fails if any module in it is
#   missing from either lock. storygen.py reached CI missing from both
#   (2026-08-27) and the image died on `import storygen` at start-up;
# - no ARG and no ENV credential anywhere, so no build argument can bake a
#   secret into a layer. The ONE credential this build needs — a read-only
#   Hugging Face token, for the gated LTX checkpoint — arrives through a
#   BUILDKIT SECRET MOUNT instead (owner directive 2026-08-28, option 1).
#   That is a different mechanism, not a loophole: --mount=type=secret
#   exposes the value on a tmpfs for the duration of ONE RUN, and it is
#   written to no layer, recorded in no image history entry, and present
#   in no image config. ARG and ENV are both permanently readable off the
#   published image with `docker history` / `docker inspect`; this is not.
#   The token is used to DOWNLOAD weights and never survives beside them;
# - each stage builds as root and executes as oniq (uid/gid 10001):
#   /app, site-packages and the baked model weights end up root-owned
#   and merely readable, so a compromised job cannot rewrite the code —
#   or the model — it runs;
# - exec-form CMD only — no shell surface.
#
# Two stages, one security boundary:
# - `base` is the complete worker (code + deps). worker-ci builds THIS
#   stage on every push, because it is small enough to build per commit.
# - `media` bakes the LTX-Video weights on top at BUILD time, so a job
#   never fetches a model over the network (videogen loads with
#   local_files_only). Which model ships is a server decision made here,
#   recorded in /app/models/MODEL_ID, with a size guard so a 13B-class
#   checkpoint can never slip in under a 2B name.
#
# OWNER DIRECTIVE 2026-08-28: the media image is built in CI and pushed to
# a container registry, and the RunPod template names the pushed image.
# This replaces the earlier rule that GitHub runners must never download
# the model. That rule was written when RunPod's own builder held the only
# copy of the image — and deleting the worker destroyed it. RunPod's API
# cannot build from a repository (POST /templates requires an imageName
# and accepts no repo field, measured 2026-08-28), so an image that
# already exists somewhere is now a precondition for the endpoint working
# at all. The media build runs ON DEMAND, never on every push: it is far
# too large to build per commit.
#
# torch installs from the cu121 index. That host is unreachable from some
# dev containers; RunPod's builder and GitHub's runners are normal hosts.
# Do NOT repoint it at PyPI to suit a development environment.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/oniq

WORKDIR /app

RUN groupadd --gid 10001 oniq \
    && useradd --uid 10001 --gid 10001 --create-home --home-dir /home/oniq oniq

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt \
    && pip install --no-cache-dir torch==2.5.1+cu121 \
        --index-url https://download.pytorch.org/whl/cu121

COPY contract.py /app/contract.py
COPY preprocess.py /app/preprocess.py
COPY storage.py /app/storage.py
COPY videogen.py /app/videogen.py
COPY storygen.py /app/storygen.py
COPY audio.py /app/audio.py
COPY handler.py /app/handler.py

# base is itself a complete, secure worker image: uid-10001 runtime,
# read-only /app. CI proves THIS stage. media re-escalates to root
# below only for the bake, and drops back before its own CMD.
USER oniq:oniq

CMD ["python3", "-u", "handler.py"]

FROM base AS media

USER root

# Bake the model — ONE model, the distilled 2B the owner chose, and no
# alternative. It is REJECTED FROM METADATA before a byte is downloaded:
# the HF API lists every file with its size, so a transformer over 16GiB
# (a 13B-class checkpoint wearing a 2B name), a missing model_index.json,
# a missing component, or an answer for a different repository stops the
# build at $0 network cost. Only a surveyed model is downloaded, pinned to
# the exact commit the survey saw, and only its pipeline components (the
# repo also carries a multi-GB single-file checkpoint this image must not
# haul in). After download: the class is an LTX pipeline, every declared
# component is one the worker's LTXImageToVideoPipeline can actually
# accept (a mismatch here would otherwise become a PAID TypeError at job
# time), and the size guard is re-checked against what landed on disk.
# The resolved id, its revision and its licence land in /app/models/ so
# the worker reports exactly what it ran and under what terms.
RUN --mount=type=secret,id=hf_token python3 - <<'EOF'
import inspect, json, os, shutil

from huggingface_hub import HfApi, snapshot_download


def auth_refused(exc):
    """Was this a refusal to AUTHENTICATE, rather than a judgement?

    Measured 2026-08-28 (validation run 51): the first candidate answers
    HTTP 401 — gated, not gone. The loop below caught that with every
    other failure and moved on, so an INFRASTRUCTURE problem silently
    became a MODEL SUBSTITUTION: a build today would ship a different
    checkpoint than the owner chose, under the same image name, and
    nothing in the log would read as an error.

    Falling through because a candidate fails the size or licence gate is
    a judgement about the model and is exactly what the list is for.
    Falling through because we could not log in is not a judgement at
    all. UNKNOWN never becomes success anywhere else in this repo, and it
    does not become a model choice here.
    """
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status in (401, 403)

# OWNER DIRECTIVE 2026-08-28 (option 1): ONE candidate, and it is the
# distilled 2B. The list previously carried two fallbacks, and run 51
# measured what that bought: with the head gated behind HTTP 401 the build
# walked past it and would have shipped Lightricks/LTX-Video — a different
# model, under the same image name, silently. Guarding a fall-through is
# weaker than not having one, so the list has one member. If this
# checkpoint cannot be fetched the build FAILS; nothing else is acceptable
# and there is nothing else to reach for.
CANDIDATES = [
    ("Lightricks/LTX-Video-0.9.8-2B-distilled", "#distilled"),
]
# The exact revision, once measured. Empty means "resolve it from the
# registry and RECORD it" — the previous image was destroyed with its
# builder, so there is no known-good sha to pin to yet. Whatever is
# resolved is written into the image and printed, so the next build can
# pin it here and be byte-reproducible.
PINNED_REVISION = ""
DEST = "/app/models/ltx"
SIZE_GUARD_BYTES = 16 * 1024**3
COMPONENTS = ("transformer", "vae", "text_encoder", "tokenizer", "scheduler")

# The credential arrives on a tmpfs for the duration of this RUN only, via
# BuildKit's secret mount. Read here, passed explicitly to the two calls
# that need it, and never written anywhere. It is deliberately NOT put in
# os.environ: huggingface_hub picks HF_TOKEN up implicitly, and an
# implicit credential is one that can travel somewhere unnoticed.
def build_token():
    try:
        with open("/run/secrets/hf_token", encoding="utf-8") as fh:
            return fh.read().strip() or None
    except OSError:
        return None


TOKEN = build_token()
if not TOKEN:
    raise SystemExit(
        "NO CREDENTIAL: /run/secrets/hf_token is absent or empty. The chosen "
        "checkpoint is gated, so there is nothing to do but stop — the owner "
        "directive forbids substituting an ungated model."
    )


def survey(api, repo):
    info = api.model_info(repo, files_metadata=True, token=TOKEN)

    # IDENTITY, before a byte moves. A rename or a redirect must not
    # quietly become a different model wearing the requested name.
    answered = getattr(info, "id", None) or getattr(info, "modelId", None)
    if answered != repo:
        raise RuntimeError(f"asked for {repo}, registry answered for {answered!r}")

    paths = {s.rfilename: (s.size or 0) for s in info.siblings}
    if "model_index.json" not in paths:
        raise RuntimeError("no model_index.json (not a diffusers snapshot)")
    for component in COMPONENTS:
        if not any(p.startswith(component + "/") for p in paths):
            raise RuntimeError(f"lacks component {component}")
    transformer_bytes = sum(
        size for p, size in paths.items()
        if p.startswith("transformer/") and p.endswith(".safetensors")
    )
    if transformer_bytes > SIZE_GUARD_BYTES:
        raise RuntimeError(
            f"transformer {transformer_bytes} metadata bytes exceed the 2B-class guard"
        )

    # The commit this build will pin to. A repository name names a moving
    # branch; a sha names bytes.
    revision = getattr(info, "sha", None)
    if not revision:
        raise RuntimeError(f"{repo} reports no commit sha to pin to")
    if PINNED_REVISION and revision != PINNED_REVISION:
        raise RuntimeError(
            f"{repo} is at {revision}, but this build is pinned to "
            f"{PINNED_REVISION} — refusing to bake a different revision"
        )

    licence = None
    card = getattr(info, "card_data", None) or {}
    try:
        licence = card.get("license")
    except AttributeError:
        licence = None
    if licence is None:
        licence = next(
            (t.split(":", 1)[1] for t in (getattr(info, "tags", None) or [])
             if t.startswith("license:")),
            None,
        )
    return transformer_bytes, revision, licence


api = HfApi()
os.makedirs("/app/models", exist_ok=True)
resolved = None
for repo, tag in CANDIDATES:
    try:
        surveyed, revision, licence = survey(api, repo)
        print(f"SURVEY {repo}: {surveyed} transformer bytes by metadata")
        print(f"REVISION {revision}")
        print(f"LICENCE {licence!r}")
        snapshot_download(
            repo,
            revision=revision,
            token=TOKEN,
            local_dir=DEST,
            allow_patterns=["model_index.json"] + [c + "/*" for c in COMPONENTS],
        )
        with open(os.path.join(DEST, "model_index.json")) as fh:
            index = json.load(fh)
        if "LTX" not in str(index.get("_class_name") or ""):
            raise RuntimeError("model_index.json is not an LTX pipeline")
        from diffusers import LTXImageToVideoPipeline

        accepted = set(
            inspect.signature(LTXImageToVideoPipeline.__init__).parameters
        ) - {"self"}
        declared = {k for k, v in index.items() if isinstance(v, list)}
        if not declared <= accepted:
            raise RuntimeError(
                f"components {sorted(declared - accepted)} are not loadable "
                "by LTXImageToVideoPipeline"
            )
        on_disk = 0
        for root, _, files in os.walk(os.path.join(DEST, "transformer")):
            for name in files:
                if name.endswith(".safetensors"):
                    on_disk += os.path.getsize(os.path.join(root, name))
        if not 0 < on_disk <= SIZE_GUARD_BYTES:
            raise RuntimeError(f"downloaded transformer is {on_disk} bytes")
        resolved = repo + tag
        resolved_revision = revision
        resolved_licence = licence
        print(f"BAKED {resolved} at {revision} ({on_disk} transformer bytes on disk)")
        break
    except Exception as exc:
        # There is nothing to fall through to — the list has one member by
        # owner directive — so every failure here is terminal. It is spelled
        # out rather than left to the loop ending, because a build that
        # quietly produced an image with no model would be far worse than
        # one that stops.
        if auth_refused(exc):
            raise SystemExit(
                f"AUTH REFUSED for {repo}: the registry will not serve it with the "
                "credential presented. The token may be absent, expired, or issued "
                "by an account that has not accepted this model's terms. Refusing "
                "to substitute another checkpoint."
            )
        raise SystemExit(f"CANNOT BAKE {repo}: {type(exc).__name__}: {exc}")

if resolved is None:
    raise SystemExit("the intended model could not be baked")
with open("/app/models/MODEL_ID", "w") as fh:
    fh.write(resolved + "\n")
# The revision and licence travel INSIDE the image, so the running worker
# can report exactly which bytes it is executing and under what terms —
# rather than that being knowable only from a build log that scrolls away.
with open("/app/models/LTX_REVISION", "w") as fh:
    fh.write(resolved_revision + "\n")
with open("/app/models/LTX_LICENCE", "w") as fh:
    fh.write(str(resolved_licence) + "\n")
shutil.rmtree(os.path.join(DEST, ".cache"), ignore_errors=True)
# Both possible cache homes, not just the one HOME points at: this stage
# runs as root while HOME is /home/oniq, and a token cache written to
# either would otherwise ride into the published layer.
shutil.rmtree(os.path.expanduser("~/.cache/huggingface"), ignore_errors=True)
shutil.rmtree("/root/.cache/huggingface", ignore_errors=True)
shutil.rmtree("/home/oniq/.cache/huggingface", ignore_errors=True)
EOF

# Bake the STORY model — ONIQ's local causal LLM (owner directive
# 2026-08-27, Qwen3-8B conditionally approved). Same discipline as LTX:
# candidates tried in order, each REJECTED FROM METADATA before a byte is
# downloaded, with a size guard so a 14B/32B slip cannot enter the image.
#
# ONE ADDITION THE LTX BAKE DOES NOT HAVE: a LICENCE GATE. The owner's
# approval is conditional on the checkpoint actually being Apache-2.0, and
# a model card read by a human months ago is not a check. The HF metadata
# carries the licence tag, so the build asserts it and REFUSES to download
# anything else. That turns "we believe it is Apache" into something the
# image cannot be built without.
#
# Verified 2026-08-27 from the authors' own repository
# (github.com/QwenLM/Qwen3): "All our open-weight models are licensed
# under Apache 2.0", with 8B among the released dense models. Apache-2.0
# section 2 grants a perpetual, royalty-free, irrevocable right to
# reproduce and distribute the Work, which is what ONIQ needs to ship the
# weights inside its own private image; section 4 conditions (licence
# copy, NOTICE, change notices) are satisfied by keeping the files the
# snapshot ships.
RUN python3 - <<'EOF'
import json, os, shutil

from huggingface_hub import HfApi, snapshot_download


def auth_refused(exc):
    """A refusal to authenticate is not a judgement about the model.

    Same rule as the LTX bake above, and here it also protects the
    LICENCE GATE: a 401 caught as an ordinary skip would let the build
    move to the next candidate without ever having read the licence it
    was supposed to check. A gate that cannot run must stop the build,
    not wave it through.
    """
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status in (401, 403)


# Quantised first (a smaller image pulls faster on a cold worker), then
# the base weights, which always exist. A candidate that does not exist
# simply SKIPs, exactly as in the LTX bake.
CANDIDATES = [
    "Qwen/Qwen3-8B-AWQ",
    "Qwen/Qwen3-8B",
]
DEST = "/app/models/story"
ALLOWED_LICENCES = {"apache-2.0"}
# An 8B checkpoint in bf16 is ~16.4GB; 20GB refuses a larger class.
SIZE_GUARD_BYTES = 20 * 1024**3
NEEDED = ("config.json", "tokenizer_config.json")


def survey(api, repo):
    info = api.model_info(repo, files_metadata=True)

    # THE LICENCE GATE. No licence, or the wrong one, and nothing is
    # downloaded — the build fails rather than baking weights ONIQ may
    # not redistribute inside its image.
    licence = (info.card_data or {}).get("license") if info.card_data else None
    if licence is None:
        licence = next(
            (t.split(":", 1)[1] for t in (info.tags or []) if t.startswith("license:")),
            None,
        )
    if str(licence).lower() not in ALLOWED_LICENCES:
        raise RuntimeError(
            f"licence {licence!r} is not in {sorted(ALLOWED_LICENCES)} — refusing to bake"
        )

    paths = {s.rfilename: (s.size or 0) for s in info.siblings}
    for needed in NEEDED:
        if needed not in paths:
            raise RuntimeError(f"lacks {needed} (not a transformers checkpoint)")
    weight_bytes = sum(
        size for p, size in paths.items() if p.endswith(".safetensors")
    )
    if not 0 < weight_bytes <= SIZE_GUARD_BYTES:
        raise RuntimeError(f"{weight_bytes} metadata weight bytes fail the 8B-class guard")
    return licence, weight_bytes


api = HfApi()
os.makedirs("/app/models", exist_ok=True)
resolved = None
for repo in CANDIDATES:
    try:
        licence, surveyed = survey(api, repo)
        print(f"SURVEY {repo}: licence={licence}, {surveyed} weight bytes by metadata")
        snapshot_download(
            repo,
            local_dir=DEST,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "LICENSE*", "NOTICE*"],
        )
        with open(os.path.join(DEST, "config.json")) as fh:
            config = json.load(fh)
        if "qwen3" not in str(config.get("model_type") or "").lower():
            raise RuntimeError(f"config model_type is {config.get('model_type')!r}, not qwen3")
        on_disk = 0
        for root, _, files in os.walk(DEST):
            for name in files:
                if name.endswith(".safetensors"):
                    on_disk += os.path.getsize(os.path.join(root, name))
        if not 0 < on_disk <= SIZE_GUARD_BYTES:
            raise RuntimeError(f"downloaded weights are {on_disk} bytes")
        resolved = repo
        print(f"BAKED {resolved} ({on_disk} weight bytes on disk, licence {licence})")
        break
    except Exception as exc:
        if auth_refused(exc):
            raise SystemExit(
                f"AUTH REFUSED for {repo}: the registry will not serve it without "
                "credentials, so the licence gate never ran. Refusing to fall "
                "through — a gate that cannot run must stop the build."
            )
        print(f"SKIP {repo}: {type(exc).__name__}: {exc}")
        shutil.rmtree(DEST, ignore_errors=True)
        shutil.rmtree(os.path.expanduser("~/.cache/huggingface"), ignore_errors=True)

if resolved is None:
    raise SystemExit("no story model could be baked")
with open("/app/models/STORY_MODEL_ID", "w") as fh:
    fh.write(resolved + "\n")
shutil.rmtree(os.path.join(DEST, ".cache"), ignore_errors=True)
shutil.rmtree(os.path.expanduser("~/.cache/huggingface"), ignore_errors=True)
EOF

# Bake the piper voice for audio_mux — the SAME sha256-pinned release
# asset the ONIQ story worker's in-house engine runs (rhasspy v0.0.2,
# en-us-ryan-high). stdlib download + digest check, tarfile with the
# data filter; a job never fetches a voice over the network.
RUN python3 - <<'EOF'
import hashlib, os, tarfile, urllib.request

URL = ("https://github.com/rhasspy/piper/releases/download/v0.0.2/"
       "voice-en-us-ryan-high.tar.gz")
SHA256 = "de346b054703a190782f49acb9b93c50678a884fede49cfd85429d204802d678"
DEST = "/app/models/piper"

os.makedirs(DEST, exist_ok=True)
tarball = "/tmp/voice.tar.gz"
urllib.request.urlretrieve(URL, tarball)
digest = hashlib.sha256(open(tarball, "rb").read()).hexdigest()
if digest != SHA256:
    raise SystemExit(f"voice sha256 mismatch: {digest}")
with tarfile.open(tarball) as tar:
    tar.extractall(DEST, filter="data")
os.remove(tarball)
for name in ("en-us-ryan-high.onnx", "en-us-ryan-high.onnx.json"):
    if not os.path.exists(os.path.join(DEST, name)):
        raise SystemExit(f"voice tarball lacked {name}")
print("BAKED piper voice en-us-ryan-high")
EOF

USER oniq:oniq

CMD ["python3", "-u", "handler.py"]
# (restated so the shipped media image never depends on inheritance)
