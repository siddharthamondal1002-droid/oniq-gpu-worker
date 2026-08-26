# ONIQ GPU worker image.
#
# Structure is the security posture:
# - COPY names the six shipped files individually — there is no COPY . .,
#   so the image cannot receive a stray .env even if .dockerignore were
#   wrong (.dockerignore is a second lock, not the only one);
# - no ARG anywhere, so no build argument can bake a secret into a layer;
# - each stage builds as root and executes as oniq (uid/gid 10001):
#   /app, site-packages and the baked model weights end up root-owned
#   and merely readable, so a compromised job cannot rewrite the code —
#   or the model — it runs;
# - exec-form CMD only — no shell surface.
#
# Two stages, one security boundary:
# - `base` is the complete worker (code + deps). CI builds THIS stage
#   only (--target base): GitHub runners must never download the model.
# - `media` bakes the LTX-Video weights on top at BUILD time, so a job
#   never fetches a model over the network (videogen loads with
#   local_files_only). Which model ships is a server decision made here,
#   recorded in /app/models/MODEL_ID, with a size guard so a 13B-class
#   checkpoint can never slip in under a 2B name.
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
COPY handler.py /app/handler.py

# base is itself a complete, secure worker image: uid-10001 runtime,
# read-only /app. CI proves THIS stage. media re-escalates to root
# below only for the bake, and drops back before its own CMD.
USER oniq:oniq

CMD ["python3", "-u", "handler.py"]

FROM base AS media

USER root

# Bake the model. Candidates are tried in order — distilled 2B first per
# the owner's model decision — and each is REJECTED FROM METADATA before
# a byte is downloaded: the HF API lists every file with its size, so a
# transformer over 16GiB (a 13B-class checkpoint wearing a 2B name), a
# missing model_index.json, or a missing component skips the candidate
# at $0 network cost. Only a surveyed candidate is downloaded, and only
# its pipeline components (the repos also carry multi-GB single-file
# checkpoints this image must not haul in). After download: the class
# is an LTX pipeline, every declared component is one the worker's
# LTXImageToVideoPipeline can actually accept (a mismatch here would
# otherwise become a PAID TypeError at job time), and the size guard is
# re-checked against what landed on disk. The resolved id lands in
# /app/models/MODEL_ID (suffixed #distilled when applicable) so the
# worker reports exactly what it ran.
RUN python3 - <<'EOF'
import inspect, json, os, shutil

from huggingface_hub import HfApi, snapshot_download

CANDIDATES = [
    ("Lightricks/LTX-Video-0.9.8-2B-distilled", "#distilled"),
    ("Lightricks/LTX-Video-0.9.7-distilled", "#distilled"),
    ("Lightricks/LTX-Video", ""),
]
DEST = "/app/models/ltx"
SIZE_GUARD_BYTES = 16 * 1024**3
COMPONENTS = ("transformer", "vae", "text_encoder", "tokenizer", "scheduler")


def survey(api, repo):
    info = api.model_info(repo, files_metadata=True)
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
    return transformer_bytes


api = HfApi()
os.makedirs("/app/models", exist_ok=True)
resolved = None
for repo, tag in CANDIDATES:
    try:
        print(f"SURVEY {repo}: {survey(api, repo)} transformer bytes by metadata")
        snapshot_download(
            repo,
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
        print(f"BAKED {resolved} ({on_disk} transformer bytes on disk)")
        break
    except Exception as exc:
        print(f"SKIP {repo}: {type(exc).__name__}: {exc}")
        shutil.rmtree(DEST, ignore_errors=True)
        shutil.rmtree(os.path.expanduser("~/.cache/huggingface"), ignore_errors=True)

if resolved is None:
    raise SystemExit("no candidate model could be baked")
with open("/app/models/MODEL_ID", "w") as fh:
    fh.write(resolved + "\n")
shutil.rmtree(os.path.join(DEST, ".cache"), ignore_errors=True)
shutil.rmtree(os.path.expanduser("~/.cache/huggingface"), ignore_errors=True)
EOF

USER oniq:oniq

CMD ["python3", "-u", "handler.py"]
# (restated so the shipped media image never depends on inheritance)
