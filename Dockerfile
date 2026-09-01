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
COPY cudaenv.py /app/cudaenv.py
COPY modelroot.py /app/modelroot.py
COPY modelhydrate.py /app/modelhydrate.py
COPY preview.py /app/preview.py
COPY videogen.py /app/videogen.py
# ltxcaps derives the inference profile from the BAKED checkpoint. videogen
# imports it, so it must ship or the worker dies on import at start-up.
COPY ltxcaps.py /app/ltxcaps.py
COPY modelprobe.py /app/modelprobe.py
COPY storygen.py /app/storygen.py
COPY audio.py /app/audio.py
COPY handler.py /app/handler.py
# The spatial-upscaler revision pin, read by the media stage's bake. Empty by
# default, which is why the image is unchanged until an owner fills it in.
COPY ltx-upscaler.pin /app/ltx-upscaler.pin

# base is itself a complete, secure worker image: uid-10001 runtime,
# read-only /app. CI proves THIS stage. media re-escalates to root
# below only for the bake, and drops back before its own CMD.
USER oniq:oniq

CMD ["python3", "-u", "handler.py"]

FROM base AS media

USER root

# Bake the model — ONE model, the one the owner chose on 2026-08-28, at
# one pinned revision, and no alternative. It is REJECTED FROM METADATA
# before a byte is downloaded:
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

    Kept for the case it names, and worth being precise about after this
    one misled four runs. Validation run 51 saw HTTP 401 here and it was
    recorded as "gated". It was not: unauthenticated, Hugging Face answers
    401 for gated AND absent repositories alike, deliberately, so that
    existence cannot be probed without credentials. Run 7 presented a
    working token and got 404 — the repository did not exist. A 401 is
    therefore never evidence of gating on its own; it is evidence that the
    question was asked without a credential.

    The distinction this function draws still holds. Failing a size or
    licence gate is a judgement about the model. Failing to authenticate
    is not a judgement at all, and UNKNOWN never becomes success anywhere
    else in this repo — so it does not become a model choice here.
    """
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status in (401, 403)

# OWNER MODEL DECISION 2026-08-28: Lightricks/LTX-Video, pinned to
# revision 8984fa25007f376c1a299016d0957a37a2f797bb.
#
# The name this file carried before — Lightricks/LTX-Video-0.9.8-2B-distilled
# — DOES NOT EXIST, measured on validation run 7 with a working credential.
# Unauthenticated the registry answers 401 for gated and for absent
# repositories alike, so four runs read "gated" where the truth was "gone".
# There is no 2B at 0.9.8: the 0.9.8 distilled release is 13B (24.29 GiB
# transformer, refused by the guard below), and the genuine distilled 2B
# (LTX-Video-2B-0.9.6-Distilled-04-25) ships as a single-file checkpoint
# with no model_index.json, which this loader cannot consume.
#
# Run 53 catalogued every Lightricks LTX repository against the gates this
# bake applies; three passed, and the owner chose this one: the current
# main release, a diffusers snapshot, all five components present, a
# 7.17 GiB transformer inside the 2B-class guard, consumable by the
# existing loader without a single-file path.
#
# ONE candidate, deliberately. The list previously carried fallbacks, and
# they were the only reason any image ever built — every historical build
# walked past the absent head and landed somewhere unrecorded. Guarding a
# fall-through is weaker than not having one, so there is nothing to fall
# through to and every failure below is terminal.
# OWNER DIRECTIVE 2026-08-31 — the 13B distilled 0.9.7 checkpoint.
#
# This moved because of a MEASUREMENT, not a preference. Run 33426496040
# found that the vae at Lightricks/LTX-Video@8984fa25 is a different network
# from the one the pinned spatial upsampler was trained beside — four stages
# against five, no timestep-conditioned decoder — so multi-scale could not be
# enabled without silently refining latents the upsampler had never seen.
# Run 33432424021 then measured every Lightricks candidate against the
# upsampler's own vae config; this one PAIRS, and the owner chose it over the
# 2B-class 0.9.5 that also pairs.
CANDIDATES = [
    ("Lightricks/LTX-Video-0.9.7-distilled", ""),
]
# THE PIN. A repository name names a moving branch; a sha names bytes —
# and it also fixes the LICENCE TERMS, because the terms at a commit
# cannot change after the fact. The build refuses any other revision.
PINNED_REVISION = "057509edea1493cae5e62e9d8f780ebda3fb4333"

# THE LTX LICENCE GATE — owner directive 2026-08-28.
#
# LTX is NOT Apache-2.0. Every Lightricks repository reports the Hugging
# Face licence tag "other", which means "see the repository's own licence
# file": the LTX Open Weights Licence. The owner was shown that, in those
# words, and selected this checkpoint anyway on 2026-08-28. That decision
# is the acceptance; this gate is its record, and it is written the same
# way the Qwen Apache gate below is — a set the metadata must match, so
# the image cannot be built if the terms ever change.
#
# "other" alone would be a weak gate: it admits ANY non-standard licence.
# It is not the whole gate. The revision pin above is, because the licence
# at 8984fa25 is fixed forever, and the build additionally refuses to ship
# weights whose LICENCE TEXT did not land beside them — redistributing
# someone's model without their terms attached is the failure this
# prevents, and it is why LICENSE*/NOTICE* joined allow_patterns.
ALLOWED_LICENCES = {"other"}
DEST = "/app/models/ltx"
# RAISED FROM 16 GiB TO 32 GiB — owner directive 2026-08-31, taken knowingly
# alongside the checkpoint choice above.
#
# The old value was a 2B-CLASS guard: it existed so a 13B checkpoint could not
# enter the image by accident. The owner has now chosen one on purpose, and
# 0.9.7-distilled's transformer measures 24.29 GiB, so the guard had to move
# or the build would refuse the very model it was told to bake.
#
# It is RAISED, NOT REMOVED, and the new value is still a real refusal: the
# same registry read measured Lightricks/LTX-2-Pre-Trained at 70.75 GiB of
# transformer, which this still rejects. A guard that admits everything is
# not a guard.
SIZE_GUARD_BYTES = 32 * 1024**3
COMPONENTS = ("transformer", "vae", "text_encoder", "tokenizer", "scheduler")
# WHICH COMPONENTS THIS RUN DOWNLOADS. The survey above still checks that
# EVERY component in COMPONENTS exists before a byte moves; this narrower
# tuple only decides what lands in THIS layer.
#
# A container layer is one download stream with NO RESUME. Measured
# 2026-08-30 from the published manifest: baking the whole pipeline in one
# RUN produced a single 16.05 GiB layer, and a worker that lost the stream
# at any point restarted all 16 GiB — which is exactly what a RunPod
# worker was observed doing for over three hours, re-downloading layers it
# had already completed. Splitting the text encoder into its own passes
# caps the worst-case restart at roughly a third of that.
# THE TRANSFORMER IS NO LONGER HERE. At 8984fa25 it was 7.17 GiB and rode
# along with the small components; 0.9.7-distilled's is 24.29 GiB, which would
# make this single no-resume layer ~27 GiB — worse than the 16.05 GiB layer
# that was measured re-downloading itself for over three hours. It now goes
# through its own three-way split below, the same midpoint mechanism the text
# encoder already uses, capping the worst-case restart at roughly 8 GiB.
FIRST_PASS = ("vae", "tokenizer", "scheduler")

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
            f"transformer {transformer_bytes} metadata bytes exceed the "
            f"{SIZE_GUARD_BYTES} guard"
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
    if str(licence).lower() not in ALLOWED_LICENCES:
        raise RuntimeError(
            f"licence {licence!r} is not in {sorted(ALLOWED_LICENCES)} — refusing "
            "to bake. The owner accepted the LTX Open Weights terms as they "
            "stood at the pinned revision; a different licence is a different "
            "decision and is the owner's to make again."
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
            allow_patterns=(
                # MEASURED 2026-08-28 (run 55): this repository's terms are
                # NOT in a file called LICENSE. They ship as
                # LTX-Video-Open-Weights-License-0.X.txt, plus a
                # per-checkpoint ltx-video-2b-v0.9.N.license.txt. Run 8's
                # build downloaded everything and then refused, because
                # "LICENSE*" anchors at the start of the name and these
                # start with "LTX-" and "ltx-". The gate was right; the
                # glob was wrong. These are kilobytes, so nothing about
                # the image size changes.
                ["model_index.json", "LICENSE*", "NOTICE*",
                 "*icense*.txt", "*icence*.txt"]
                + [c + "/*" for c in FIRST_PASS]
            ),
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
        # A BLOCK VERIFIES WHAT IT DOWNLOADED, AND ONLY THAT.
        #
        # This used to weigh DEST/transformer, which was right while the
        # transformer rode in FIRST_PASS. It no longer does — it goes through
        # the three-way split below — so the check found 0 bytes and refused
        # the build (run 33464498069). The metadata guard in survey() is the
        # one that refuses an oversized model BEFORE anything downloads, and
        # it still runs; the on-disk guard moved to the final transformer
        # pass, where the bytes actually are.
        first_pass_bytes = 0
        for component in FIRST_PASS:
            for root, _, files in os.walk(os.path.join(DEST, component)):
                for name in files:
                    first_pass_bytes += os.path.getsize(os.path.join(root, name))
        if first_pass_bytes <= 0:
            raise RuntimeError(
                f"first pass downloaded nothing for {list(FIRST_PASS)}"
            )

        # The terms must travel WITH the weights. An image that
        # redistributes someone's model without their licence text beside
        # it is the compliance failure this catches, and it is caught here
        # rather than by a human noticing later.
        licence_files = sorted(
            name for name in os.listdir(DEST)
            if "LICENSE" in name.upper() or "LICENCE" in name.upper()
            or name.upper().startswith("NOTICE")
        )
        if not licence_files:
            raise RuntimeError(
                f"{repo} shipped no LICENSE or NOTICE file at {revision} — "
                "refusing to redistribute the weights without their terms"
            )
        print(f"LICENCE FILES {licence_files}")

        resolved = repo + tag
        resolved_revision = revision
        resolved_licence = licence
        print(f"BAKED {resolved} at {revision} "
              f"({first_pass_bytes} bytes for {list(FIRST_PASS)}; the "
              f"transformer follows in its own passes)")
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
# The BARE repo id, for the transformer passes below. MODEL_ID carries
# repo+tag and is what the worker reports; this is what the hub is asked
# for, and keeping them separate stops a display string from becoming a
# download argument.
with open("/app/models/LTX_REPO", "w") as fh:
    fh.write(repo + "\n")
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

# THE TRANSFORMER, IN THREE LAYERS. Identical mechanism to the text encoder
# below — list the component's shards from metadata, split them by cumulative
# size on each file's MIDPOINT, download this pass's share — and here for the
# same reason: a container layer is one download stream with NO RESUME, so a
# worker that loses the stream restarts the whole layer. 0.9.7-distilled's
# transformer is 24.29 GiB; in one layer that is the failure mode this image
# has already paid for once, and in three it is roughly 8 GiB.
#
# A repository whose transformer is one unsharded file puts it all in pass 0
# and leaves the others empty — still correct, just unsplit.
RUN --mount=type=secret,id=hf_token LTX_TX_PASS=0 python3 - <<'EOF'
import os

from huggingface_hub import HfApi, snapshot_download

TOKEN = None
if os.path.exists("/run/secrets/hf_token"):
    with open("/run/secrets/hf_token") as fh:
        TOKEN = fh.read().strip() or None
DEST = "/app/models/ltx"
PREFIX = "transformer/"
PASSES = 3
# The same ceiling the first block's survey() applies from metadata, repeated
# here because this is a separate process and this is where the bytes land.
SIZE_GUARD_BYTES = 32 * 1024**3
PASS = int(os.environ["LTX_TX_PASS"])

with open("/app/models/LTX_REPO") as fh:
    repo = fh.read().strip()
with open("/app/models/LTX_REVISION") as fh:
    revision = fh.read().strip()

info = HfApi().model_info(repo, revision=revision, files_metadata=True, token=TOKEN)
if (getattr(info, "id", None) or getattr(info, "modelId", None)) != repo:
    raise SystemExit(f"registry answered for a different repository than {repo}")
files = sorted(
    (s.rfilename, s.size or 0) for s in info.siblings
    if s.rfilename.startswith(PREFIX)
)
if not files:
    raise SystemExit("the pinned revision carries no transformer files")
total = sum(size for _, size in files)
mine, running = [], 0
for name, size in files:
    # The file's MIDPOINT decides its group, not its start: a large shard
    # beginning just before a boundary would otherwise land wholly in the
    # earlier pass and rebuild the imbalance this split exists to remove.
    group = min(int((running + size / 2) * PASSES / total) if total else 0,
                PASSES - 1)
    if group == PASS:
        mine.append(name)
    running += size
print(f"TRANSFORMER PASS {PASS}: {len(mine)} of {len(files)} file(s), "
      f"{sum(s for n, s in files if n in set(mine))} bytes")
if not mine:
    print("nothing for this pass — the transformer is not sharded this finely")
else:
    snapshot_download(repo, revision=revision, token=TOKEN,
                      local_dir=DEST, allow_patterns=mine)

# THE COMPONENT MUST BE WHOLE AFTER THE LAST PASS. A split download that
# silently landed two thirds of a transformer would fail at job time on a
# rented card, which is the class of failure this whole bake exists to
# prevent. Only the FINAL pass can know the component is complete.
if PASS == PASSES - 1:
    missing = [name for name, _ in files
               if not os.path.exists(os.path.join(DEST, name))]
    if missing:
        raise SystemExit(
            f"transformer incomplete after all passes: {missing[:5]}")
    landed = sum(os.path.getsize(os.path.join(DEST, name))
                 for name, _ in files)
    # THE ON-DISK SIZE GUARD, moved here from the first bake block when the
    # transformer moved into these passes. survey() already refused an
    # oversized model from METADATA before any byte moved; this is the same
    # ceiling applied to what actually landed, so a registry that under-reports
    # a size cannot smuggle a bigger model past both.
    weights = sum(os.path.getsize(os.path.join(DEST, name))
                  for name, _ in files if name.endswith(".safetensors"))
    if not 0 < weights <= SIZE_GUARD_BYTES:
        raise SystemExit(
            f"transformer weighs {weights} bytes on disk, outside the "
            f"{SIZE_GUARD_BYTES} guard")
    print(f"TRANSFORMER COMPLETE: {len(files)} file(s), {landed} bytes "
          f"({weights} in weights)")
import shutil
shutil.rmtree(os.path.join(DEST, ".cache"), ignore_errors=True)
for home in ("~/.cache/huggingface", "/root/.cache/huggingface",
             "/home/oniq/.cache/huggingface"):
    shutil.rmtree(os.path.expanduser(home), ignore_errors=True)
EOF

RUN --mount=type=secret,id=hf_token LTX_TX_PASS=1 python3 - <<'EOF'
import os

from huggingface_hub import HfApi, snapshot_download

TOKEN = None
if os.path.exists("/run/secrets/hf_token"):
    with open("/run/secrets/hf_token") as fh:
        TOKEN = fh.read().strip() or None
DEST = "/app/models/ltx"
PREFIX = "transformer/"
PASSES = 3
# The same ceiling the first block's survey() applies from metadata, repeated
# here because this is a separate process and this is where the bytes land.
SIZE_GUARD_BYTES = 32 * 1024**3
PASS = int(os.environ["LTX_TX_PASS"])

with open("/app/models/LTX_REPO") as fh:
    repo = fh.read().strip()
with open("/app/models/LTX_REVISION") as fh:
    revision = fh.read().strip()

info = HfApi().model_info(repo, revision=revision, files_metadata=True, token=TOKEN)
if (getattr(info, "id", None) or getattr(info, "modelId", None)) != repo:
    raise SystemExit(f"registry answered for a different repository than {repo}")
files = sorted(
    (s.rfilename, s.size or 0) for s in info.siblings
    if s.rfilename.startswith(PREFIX)
)
if not files:
    raise SystemExit("the pinned revision carries no transformer files")
total = sum(size for _, size in files)
mine, running = [], 0
for name, size in files:
    # The file's MIDPOINT decides its group, not its start: a large shard
    # beginning just before a boundary would otherwise land wholly in the
    # earlier pass and rebuild the imbalance this split exists to remove.
    group = min(int((running + size / 2) * PASSES / total) if total else 0,
                PASSES - 1)
    if group == PASS:
        mine.append(name)
    running += size
print(f"TRANSFORMER PASS {PASS}: {len(mine)} of {len(files)} file(s), "
      f"{sum(s for n, s in files if n in set(mine))} bytes")
if not mine:
    print("nothing for this pass — the transformer is not sharded this finely")
else:
    snapshot_download(repo, revision=revision, token=TOKEN,
                      local_dir=DEST, allow_patterns=mine)

# THE COMPONENT MUST BE WHOLE AFTER THE LAST PASS. A split download that
# silently landed two thirds of a transformer would fail at job time on a
# rented card, which is the class of failure this whole bake exists to
# prevent. Only the FINAL pass can know the component is complete.
if PASS == PASSES - 1:
    missing = [name for name, _ in files
               if not os.path.exists(os.path.join(DEST, name))]
    if missing:
        raise SystemExit(
            f"transformer incomplete after all passes: {missing[:5]}")
    landed = sum(os.path.getsize(os.path.join(DEST, name))
                 for name, _ in files)
    # THE ON-DISK SIZE GUARD, moved here from the first bake block when the
    # transformer moved into these passes. survey() already refused an
    # oversized model from METADATA before any byte moved; this is the same
    # ceiling applied to what actually landed, so a registry that under-reports
    # a size cannot smuggle a bigger model past both.
    weights = sum(os.path.getsize(os.path.join(DEST, name))
                  for name, _ in files if name.endswith(".safetensors"))
    if not 0 < weights <= SIZE_GUARD_BYTES:
        raise SystemExit(
            f"transformer weighs {weights} bytes on disk, outside the "
            f"{SIZE_GUARD_BYTES} guard")
    print(f"TRANSFORMER COMPLETE: {len(files)} file(s), {landed} bytes "
          f"({weights} in weights)")
import shutil
shutil.rmtree(os.path.join(DEST, ".cache"), ignore_errors=True)
for home in ("~/.cache/huggingface", "/root/.cache/huggingface",
             "/home/oniq/.cache/huggingface"):
    shutil.rmtree(os.path.expanduser(home), ignore_errors=True)
EOF

RUN --mount=type=secret,id=hf_token LTX_TX_PASS=2 python3 - <<'EOF'
import os

from huggingface_hub import HfApi, snapshot_download

TOKEN = None
if os.path.exists("/run/secrets/hf_token"):
    with open("/run/secrets/hf_token") as fh:
        TOKEN = fh.read().strip() or None
DEST = "/app/models/ltx"
PREFIX = "transformer/"
PASSES = 3
# The same ceiling the first block's survey() applies from metadata, repeated
# here because this is a separate process and this is where the bytes land.
SIZE_GUARD_BYTES = 32 * 1024**3
PASS = int(os.environ["LTX_TX_PASS"])

with open("/app/models/LTX_REPO") as fh:
    repo = fh.read().strip()
with open("/app/models/LTX_REVISION") as fh:
    revision = fh.read().strip()

info = HfApi().model_info(repo, revision=revision, files_metadata=True, token=TOKEN)
if (getattr(info, "id", None) or getattr(info, "modelId", None)) != repo:
    raise SystemExit(f"registry answered for a different repository than {repo}")
files = sorted(
    (s.rfilename, s.size or 0) for s in info.siblings
    if s.rfilename.startswith(PREFIX)
)
if not files:
    raise SystemExit("the pinned revision carries no transformer files")
total = sum(size for _, size in files)
mine, running = [], 0
for name, size in files:
    # The file's MIDPOINT decides its group, not its start: a large shard
    # beginning just before a boundary would otherwise land wholly in the
    # earlier pass and rebuild the imbalance this split exists to remove.
    group = min(int((running + size / 2) * PASSES / total) if total else 0,
                PASSES - 1)
    if group == PASS:
        mine.append(name)
    running += size
print(f"TRANSFORMER PASS {PASS}: {len(mine)} of {len(files)} file(s), "
      f"{sum(s for n, s in files if n in set(mine))} bytes")
if not mine:
    print("nothing for this pass — the transformer is not sharded this finely")
else:
    snapshot_download(repo, revision=revision, token=TOKEN,
                      local_dir=DEST, allow_patterns=mine)

# THE COMPONENT MUST BE WHOLE AFTER THE LAST PASS. A split download that
# silently landed two thirds of a transformer would fail at job time on a
# rented card, which is the class of failure this whole bake exists to
# prevent. Only the FINAL pass can know the component is complete.
if PASS == PASSES - 1:
    missing = [name for name, _ in files
               if not os.path.exists(os.path.join(DEST, name))]
    if missing:
        raise SystemExit(
            f"transformer incomplete after all passes: {missing[:5]}")
    landed = sum(os.path.getsize(os.path.join(DEST, name))
                 for name, _ in files)
    # THE ON-DISK SIZE GUARD, moved here from the first bake block when the
    # transformer moved into these passes. survey() already refused an
    # oversized model from METADATA before any byte moved; this is the same
    # ceiling applied to what actually landed, so a registry that under-reports
    # a size cannot smuggle a bigger model past both.
    weights = sum(os.path.getsize(os.path.join(DEST, name))
                  for name, _ in files if name.endswith(".safetensors"))
    if not 0 < weights <= SIZE_GUARD_BYTES:
        raise SystemExit(
            f"transformer weighs {weights} bytes on disk, outside the "
            f"{SIZE_GUARD_BYTES} guard")
    print(f"TRANSFORMER COMPLETE: {len(files)} file(s), {landed} bytes "
          f"({weights} in weights)")
import shutil
shutil.rmtree(os.path.join(DEST, ".cache"), ignore_errors=True)
for home in ("~/.cache/huggingface", "/root/.cache/huggingface",
             "/home/oniq/.cache/huggingface"):
    shutil.rmtree(os.path.expanduser(home), ignore_errors=True)
EOF

# THE TEXT ENCODER IS NOT IN THIS IMAGE — owner directive 2026-08-31.
#
# It used to be baked here in two layers. Moving the checkpoint to
# LTX-Video-0.9.7-distilled, so its vae matches the pinned spatial upsampler,
# put the media image at 57.97 GiB, and a hosted runner cannot build that:
# run 33434875038 measured 33.76 GiB after reclaim, and the last SUCCESSFUL
# publish (33327318610) finished a 40.10 GiB image with 5.73 GiB to spare.
#
# The text encoder is the single largest component the image can do without —
# 17.74 GiB, measured in run 33436186203 as 19,049,290,370 bytes over
# text_encoder/*. Taking it out brings the image to ~40.2 GiB, which is the
# size that already builds. The owner chose this over paying for a larger
# build runner or giving up the 13B checkpoint.
#
# IT NOW LIVES ON THE NETWORK VOLUME as modelroot.VOLUME_RESIDENT
# ["LTX_TEXT_ENCODER"], hydrated once by the model_hydrate op and loaded by
# videogen through modelroot.resolve. That is a REAL CHANGE IN FAILURE MODE
# and it is deliberate: ltx/story/piper fall back to the baked copy when the
# volume is absent, and this one has no baked copy to fall back to. A clip
# that cannot find its text encoder REFUSES and names the reason. It must
# never quietly run with an untrained embedding, which is what a silent
# fallback would produce.
#
# The survey above still requires text_encoder to EXIST in the repository
# before a byte moves — the pipeline is incomplete without it. What changed is
# only where the bytes are fetched to.


# ---------------------------------------------------------------------------
# THE SPATIAL LATENT UPSCALER — OFF BY DEFAULT, AND OFF MEANS ABSENT.
#
# WHY IT MATTERS. Measured arithmetic, not opinion: the generation canvas is
# 704 wide and the delivered film is 1080 wide, so assembly resamples every
# frame UP by 1.534x. A pixel resampler cannot invent detail that was never
# sampled, and that 53% is exactly the softness a viewer reads as "hazy" — it
# is baked in before Remotion ever opens the clip. Upstream's answer is
# multi-scale: generate latents at the base canvas, upsample them IN LATENT
# SPACE, run a short second denoise so the transformer actually synthesises
# the new detail, then decode. videogen.py implements all three passes and
# ltxcaps.py turns them on the moment this component is present.
#
# WHY IT IS A PIN FILE AND NOT A DEFAULT. Three reasons, and each is a rule
# this repository already holds:
#
#   1. A REVISION IS THE LICENCE. The LTX bake above pins a sha because the
#      owner accepted the LTX Open Weights terms AS THEY STOOD at those
#      bytes. The upscaler ships under the same family of terms, so it gets
#      the same treatment: no sha, no bake. Inventing a plausible-looking
#      revision would be worse than shipping without the capability.
#   2. UNSET MUST BE A NO-OP. With the pin file empty this stage does nothing
#      at all, so the image is byte-identical to the one before this change
#      and no build that works today can start failing because of it.
#   3. THE BUILD IS WHERE VERIFICATION CAN HAPPEN. The gates below — repo
#      reachable, revision matches, licence text present, component config
#      loadable by the installed diffusers, size inside the guard — need the
#      registry. They run there and fail closed.
#
# WHY A FILE RATHER THAN A BUILD ARG. `test_the_image_takes_no_build_argument`
# bans ARG outright, because ARG and ENV both survive into the published image
# where `docker history` can read them, and a blanket ban is stronger than
# case-by-case judgement about which ARG is sensitive. That guard stands. A
# committed pin is better here anyway: enabling a licence-bearing model becomes
# a reviewable one-line diff rather than an invisible build flag, which is
# exactly how the LTX revision above is already pinned.
#
# SIZE, computed from the architecture in diffusers 0.38.0
# (pipelines/ltx/modeling_latent_upsampler.py) rather than guessed:
# in_channels 128, mid_channels 512, 4 res blocks per stage, Conv3d ->
# 126.25 M parameters, ~482 MiB at fp32 and ~241 MiB at bf16. Against the
# ~40 GiB media image that is roughly 1.2%, so the disk finding that blocked
# earlier work does NOT block this. The guard below is set well above the
# computed figure and well below anything that could be a different model.
RUN --mount=type=secret,id=hf_token python3 - <<'EOF'
import json, os, shutil

# The pin file: blank lines and # comments ignored, then "<repo> <revision>".
REPO = REVISION = ""
try:
    with open("/app/ltx-upscaler.pin", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise SystemExit(
                    f"ltx-upscaler.pin: expected '<repo> <revision>', got {line!r}"
                )
            REPO, REVISION = parts
            break
except FileNotFoundError:
    pass
DEST = "/app/models/ltx/latent_upsampler"
# 2x the computed fp32 size. Comfortably above the real component and far
# below anything that could be a transformer arriving under the wrong name.
SIZE_GUARD_BYTES = 1024**3

if not (REPO and REVISION):
    print("UPSCALER SKIPPED: ltx-upscaler.pin names no revision. The image "
          "ships without multi-scale; videogen records "
          "upscaler_absent_reason=latent-upsampler-not-baked on every clip.")
    raise SystemExit(0)

from huggingface_hub import HfApi, snapshot_download


def build_token():
    try:
        with open("/run/secrets/hf_token") as fh:
            return fh.read().strip() or None
    except OSError:
        return None


TOKEN = build_token()
api = HfApi()
info = api.model_info(REPO, revision=REVISION, files_metadata=True, token=TOKEN)

# A NAME NAMES A BRANCH; A SHA NAMES BYTES. Same rule as the LTX bake.
if getattr(info, "sha", None) != REVISION:
    raise SystemExit(
        f"{REPO} resolved to {getattr(info, 'sha', None)} but this build was "
        f"pinned to {REVISION} — refusing to bake different bytes"
    )

# WHICH FILES ARE THE COMPONENT? Two repository shapes ship this model and
# the difference is load-bearing.
#
#   BARE COMPONENT  config.json at the root, weights beside it.
#   PIPELINE        model_index.json at the root, the component in a
#                   subfolder, and a vae alongside it.
#
# MEASURED 2026-08-31 against the registry. a-r-r-o-w/LTX-0.9.8-Latent-Upsampler
# is the first shape and ships NO licence of any kind, so this bake refuses it.
# Lightricks/ltxv-spatial-upscaler-0.9.7 is the second, ships
# LTX-Video-Open-Weights-License-0.X.txt, and its latent_upsampler component is
# 505,009,832 bytes — byte-identical to the third party's, so they are the same
# weights and only one of them carries terms ONIQ can accept. Reading both
# shapes is exactly what lets the licence-clean copy be the one baked.
names = {f.rfilename: (f.size or 0) for f in info.siblings}
subdirs = sorted({p.split("/", 1)[0] for p in names if "/" in p})
config_rel = next(
    (c for c in ["config.json"] + [f"{d}/config.json" for d in subdirs
                                   if "upsampl" in d.lower() or "upscal" in d.lower()]
     if c in names),
    None,
)
if not config_rel:
    raise SystemExit(
        f"{REPO} at {REVISION} carries no upsampler config.json (root files "
        f"{sorted(p for p in names if '/' not in p)}, subfolders {subdirs})"
    )
PREFIX = config_rel.rsplit("/", 1)[0] + "/" if "/" in config_rel else ""

# THE GUARD IS ON THE COMPONENT, NOT THE REPOSITORY. A pipeline repo carries a
# vae as well; counting its bytes against a component guard would refuse a
# repository that is entirely correct.
weights = sum(
    size for path, size in names.items()
    if path.startswith(PREFIX) and path.endswith((".safetensors", ".bin"))
)
if not 0 < weights <= SIZE_GUARD_BYTES:
    raise SystemExit(
        f"{REPO} component {PREFIX or '(root)'} carries {weights} weight "
        f"bytes; the upscaler guard is {SIZE_GUARD_BYTES}. That is not a "
        "spatial latent upsampler."
    )

STAGE = DEST + "_src"
snapshot_download(
    REPO, revision=REVISION, token=TOKEN, local_dir=STAGE,
    allow_patterns=[f"{PREFIX}*.json", f"{PREFIX}*.safetensors",
                    # The vae CONFIG only — kilobytes, and the thing the
                    # latent-space gate below compares against. Its weights
                    # are deliberately not fetched: this image decodes with
                    # its OWN vae, and a second one would be gigabytes.
                    "vae/config.json",
                    "LICENSE*", "NOTICE*", "*icense*.txt", "*icence*.txt"],
)

# Read before the staging tree is removed; the flatten below only moves files
# that sit at the root or in the component folder.
upstream_vae_path = os.path.join(STAGE, "vae", "config.json")
upstream_vae = None
if os.path.isfile(upstream_vae_path):
    with open(upstream_vae_path, encoding="utf-8") as fh:
        upstream_vae = json.load(fh)

# FLATTEN, so DEST *is* the component. ltxcaps finds the upsampler by scanning
# UPSCALER_DIRS under the model root and videogen calls
# LTXLatentUpsamplerModel.from_pretrained(DEST); both expect the config and the
# weights directly in DEST rather than one level down.
os.makedirs(DEST, exist_ok=True)
component_dir = os.path.join(STAGE, PREFIX.rstrip("/")) if PREFIX else STAGE
for name in os.listdir(component_dir):
    src = os.path.join(component_dir, name)
    if os.path.isfile(src):
        shutil.move(src, os.path.join(DEST, name))
# The terms travel WITH the weights. In a pipeline repo they sit at the root,
# so they are moved down beside the component they license.
for name in os.listdir(STAGE):
    src = os.path.join(STAGE, name)
    if os.path.isfile(src):
        shutil.move(src, os.path.join(DEST, name))
shutil.rmtree(STAGE, ignore_errors=True)

# THE TERMS MUST TRAVEL WITH THE WEIGHTS — the same compliance check the LTX
# bake makes, for the same reason: an image that redistributes someone's model
# without their licence text beside it is a failure caught here rather than by
# a human noticing later.
licence_files = sorted(
    n for n in os.listdir(DEST)
    if "LICENSE" in n.upper() or "LICENCE" in n.upper() or n.upper().startswith("NOTICE")
)
if not licence_files:
    raise SystemExit(
        f"{REPO} shipped no LICENSE or NOTICE at {REVISION} — refusing to "
        "redistribute the weights without their terms"
    )

# IT MUST BE THE MODEL THE CODE WILL CONSTRUCT.
#
# CHECKED ON THE CONSTRUCTED MODEL, NOT ON THE RAW JSON. Measured 2026-08-31:
# Lightricks' latent_upsampler/config.json declares _class_name and leaves the
# architecture to LTXLatentUpsamplerModel's own defaults, so asserting raw keys
# refused a component that is in fact correct. Constructing it and reading the
# resolved config back is STRICTER rather than looser — it verifies what the
# model IS once defaults resolve, instead of what someone happened to write in
# a file.
#
# A config that merely LOADS under the right class is still not enough: a
# temporal upsampler, a 2-D variant or a different channel width would all
# construct happily and then produce latents the refine pass cannot use — at
# job time, on a rented card, which is the class of failure this whole bake
# exists to prevent. Hence the resolved values are asserted too.
#
# The expected shape is VERIFIED FROM UPSTREAM (diffusers 0.38.0,
# pipelines/ltx/modeling_latent_upsampler.py) and cross-checks against the
# published artifact size: 128/512/4 with dims=3 is 126.25 M parameters, which
# at fp32 is 505 MB — the size BOTH published copies ship at, to the byte. Two
# independent facts agreeing is why these are assertions rather than hopes.
with open(os.path.join(DEST, "config.json")) as fh:
    cfg = json.load(fh)
if cfg.get("_class_name") != "LTXLatentUpsamplerModel":
    raise SystemExit(
        f"{DEST}/config.json declares {cfg.get('_class_name')!r}, not "
        "LTXLatentUpsamplerModel"
    )
EXPECTED = {
    "dims": 3,
    "in_channels": 128,
    "mid_channels": 512,
    "num_blocks_per_stage": 4,
    "spatial_upsample": True,
    # A TEMPORAL upsampler would change the frame count, and the clip contract
    # is 97 frames. This one must move pixels, not time.
    "temporal_upsample": False,
}
from diffusers.pipelines.ltx.modeling_latent_upsampler import LTXLatentUpsamplerModel
model = LTXLatentUpsamplerModel.from_config(cfg)  # raises if not loadable
resolved = {k: getattr(model.config, k, None) for k in EXPECTED}
wrong = {k: resolved[k] for k, v in EXPECTED.items() if resolved[k] != v}
if wrong:
    raise SystemExit(
        f"{DEST} constructs to {wrong}, which is not the spatial upsampler "
        f"this pipeline needs (expected {EXPECTED})"
    )
print(f"UPSCALER CONFIG resolved {resolved} from {config_rel}")

# THE UPSAMPLER'S LATENT SPACE MUST BE THE ONE THIS IMAGE BAKES.
#
# MEASURED 2026-08-31, run 33426496040, and this gate exists because the
# measurement came back wrong. LTXLatentUpsamplePipeline takes (vae,
# latent_upsampler) and normalises with self.vae.latents_mean/latents_std —
# ONIQ'S OWN baked vae, never the one the upsampler shipped beside. So an
# upsampler trained in a DIFFERENT vae's latent space would load, construct,
# pass every check above, and then refine latents it was never trained on:
# a silent quality failure on a rented card, visible only by watching the
# output.
#
# The two configs measured, side by side:
#
#   field                    baked LTX-Video@8984fa25   upscaler 0.9.7
#   block_out_channels       [128, 256, 512, 512]       [128, 256, 512, 1024, …5]
#   layers_per_block         [4, 3, 3, 3, …5]           [4, 6, 6, 2, …5]
#   spatio_temporal_scaling  [T, T, T, F]               [T, T, T, T]
#   down_block_types         absent                     LTXVideo095DownBlock3D ×4
#   timestep_conditioning    absent                     True
#   decoder_* (4 fields)     absent                     present
#
# Different encoders. latents_mean, latents_std and latent_channels DO agree,
# so the normalisation constants match — but the network producing those 128
# channels is not the same network, and matching per-channel statistics is not
# a shared latent space.
#
# WHAT IS COMPARED, and what deliberately is not. Encoder shape and latent
# normalisation decide what the upsampler receives. decoder_* fields do not:
# this image decodes with its own vae whatever happens, so a decoder
# difference changes the picture, not the compatibility. spatial/temporal
# compression ratios are DERIVED from the fields below, so including them
# would only add noise — newer configs write them out, older ones do not.
LATENT_SPACE = (
    "latent_channels", "latents_mean", "latents_std", "in_channels",
    "block_out_channels", "layers_per_block", "spatio_temporal_scaling",
    "down_block_types", "patch_size", "patch_size_t",
)
baked_vae_path = "/app/models/ltx/vae/config.json"
if upstream_vae is None:
    raise SystemExit(
        f"{REPO} at {REVISION} ships no vae/config.json, so the latent space "
        "the upsampler was trained in CANNOT be compared against the one this "
        "image bakes. That is unverified, not verified — refusing rather than "
        "enabling multi-scale on faith."
    )
with open(baked_vae_path, encoding="utf-8") as fh:
    baked_vae = json.load(fh)
# The identity the LTX bake recorded, read rather than re-declared: this RUN
# is a separate Python process and PINNED_REPO/PINNED_REVISION are not in it.
with open("/app/models/LTX_REPO", encoding="utf-8") as fh:
    baked_repo = fh.read().strip()
with open("/app/models/LTX_REVISION", encoding="utf-8") as fh:
    baked_revision = fh.read().strip()
latent_delta = {
    k: (baked_vae.get(k), upstream_vae.get(k))
    for k in LATENT_SPACE if baked_vae.get(k) != upstream_vae.get(k)
}
if latent_delta:
    lines = "\n".join(
        f"    {k}\n      baked    {b!r:.120}\n      upscaler {u!r:.120}"
        for k, (b, u) in sorted(latent_delta.items())
    )
    raise SystemExit(
        f"LATENT SPACE MISMATCH — refusing to bake {REPO} at {REVISION}.\n"
        f"The upsampler was trained beside a vae that differs from the one "
        f"this image bakes ({baked_repo} at {baked_revision}) on "
        f"{len(latent_delta)} field(s):\n{lines}\n"
        "  Fix the PAIRING, never this check: bake the checkpoint whose vae "
        "matches the upsampler, or pin an upsampler trained for this vae. "
        "Clearing ltx-upscaler.pin disables multi-scale and leaves the "
        "single-scale path running unchanged."
    )
print(f"LATENT SPACE VERIFIED — the baked vae and {REPO}'s agree on "
      f"all of {list(LATENT_SPACE)}")

on_disk = sum(
    os.path.getsize(os.path.join(r, n))
    for r, _, fs in os.walk(DEST) for n in fs if n.endswith(".safetensors")
)
with open("/app/models/LTX_UPSCALER_ID", "w") as fh:
    fh.write(f"{REPO}\n")
with open("/app/models/LTX_UPSCALER_REVISION", "w") as fh:
    fh.write(f"{REVISION}\n")
print(f"UPSCALER BAKED {REPO} at {REVISION} ({on_disk} bytes on disk); "
      f"licence files {licence_files}")

shutil.rmtree(os.path.join(DEST, ".cache"), ignore_errors=True)
for home in ("~/.cache/huggingface", "/root/.cache/huggingface",
             "/home/oniq/.cache/huggingface"):
    shutil.rmtree(os.path.expanduser(home), ignore_errors=True)
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
