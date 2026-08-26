# ONIQ GPU worker image.
#
# Structure is the security posture:
# - COPY names the six shipped files individually — there is no COPY . .,
#   so the image cannot receive a stray .env even if .dockerignore were
#   wrong (.dockerignore is a second lock, not the only one);
# - no ARG anywhere, so no build argument can bake a secret into a layer;
# - everything above USER builds as root, everything below executes as
#   oniq (uid/gid 10001): /app, site-packages and the baked model weights
#   end up root-owned and merely readable, so a compromised job cannot
#   rewrite the code — or the model — it runs;
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

FROM base AS media

# Bake the model. Candidates are tried in order — distilled 2B first per
# the owner's model decision — and each is validated the same way the
# worker will load it (from_pretrained on the local snapshot) plus a
# size guard: a transformer over 16GB on disk is a 13B-class checkpoint
# wearing the wrong name, and is refused. The resolved id lands in
# /app/models/MODEL_ID (suffixed #distilled when applicable) so the
# worker reports exactly what it ran.
RUN python3 - <<'EOF'
import os, shutil

from huggingface_hub import snapshot_download

CANDIDATES = [
    ("Lightricks/LTX-Video-0.9.7-distilled", "#distilled"),
    ("Lightricks/LTX-Video", ""),
]
DEST = "/app/models/ltx"
SIZE_GUARD_BYTES = 16 * 1024**3

os.makedirs("/app/models", exist_ok=True)
resolved = None
for repo, tag in CANDIDATES:
    try:
        path = snapshot_download(repo)
        transformer = os.path.join(path, "transformer")
        total = 0
        for root, _, files in os.walk(
            transformer if os.path.isdir(transformer) else path
        ):
            for name in files:
                if name.endswith(".safetensors"):
                    total += os.path.getsize(os.path.join(root, name))
        if total > SIZE_GUARD_BYTES:
            print(f"REJECT {repo}: transformer weights {total} bytes exceed the 2B-class guard")
            continue
        index = os.path.join(path, "model_index.json")
        with open(index) as fh:
            head = fh.read()
        if "LTX" not in head:
            print(f"REJECT {repo}: model_index.json is not an LTX pipeline")
            continue
        for component in ("transformer", "vae", "text_encoder", "tokenizer", "scheduler"):
            if not os.path.isdir(os.path.join(path, component)):
                raise RuntimeError(f"{repo} lacks component {component}")
        shutil.copytree(path, DEST, dirs_exist_ok=True)
        resolved = repo + tag
        print(f"BAKED {resolved} ({total} transformer bytes)")
        break
    except Exception as exc:
        print(f"SKIP {repo}: {type(exc).__name__}: {exc}")

if resolved is None:
    raise SystemExit("no candidate model could be baked")
with open("/app/models/MODEL_ID", "w") as fh:
    fh.write(resolved + "\n")
shutil.rmtree(os.path.expanduser("~/.cache/huggingface"), ignore_errors=True)
EOF

USER oniq:oniq

CMD ["python3", "-u", "handler.py"]
