# ONIQ GPU worker image.
#
# Structure is the security posture:
# - COPY names the five shipped files individually — there is no COPY . .,
#   so the image cannot receive a stray .env even if .dockerignore were
#   wrong (.dockerignore is a second lock, not the only one);
# - no ARG anywhere, so no build argument can bake a secret into a layer;
# - everything above USER builds as root, everything below executes as
#   oniq (uid/gid 10001): /app and site-packages end up root-owned and
#   merely readable, so a compromised job cannot rewrite the code it runs;
# - exec-form CMD only — no shell surface.
#
# torch installs from the cu121 index. That host is unreachable from some
# dev containers; RunPod's builder and GitHub's runners are normal hosts.
# Do NOT repoint it at PyPI to suit a development environment.

FROM python:3.11-slim

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
COPY handler.py /app/handler.py

USER oniq:oniq

CMD ["python3", "-u", "handler.py"]
