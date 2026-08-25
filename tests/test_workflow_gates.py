"""The six spend gates, verified by parsing the files — not by reading
them. A gate that only exists in prose is not a gate."""

import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    with open(os.path.join(ROOT, ".github", "workflows", name), encoding="utf-8") as fh:
        return yaml.safe_load(fh), open(
            os.path.join(ROOT, ".github", "workflows", name), encoding="utf-8"
        ).read()


def _triggers(doc):
    # YAML 1.1 parses a bare `on` key as boolean True.
    return doc.get("on", doc.get(True))


# ------------------------------------------------------- gpu-validation.yml


def test_gate1_workflow_dispatch_only():
    doc, _ = _load("gpu-validation.yml")
    assert set(_triggers(doc)) == {"workflow_dispatch"}


def test_gate2_default_mode_is_readonly_discover():
    doc, _ = _load("gpu-validation.yml")
    inputs = _triggers(doc)["workflow_dispatch"]["inputs"]
    assert inputs["mode"]["default"] == "discover"
    assert inputs["spend"]["default"] == ""


def test_gate2_spend_job_requires_the_literal_word():
    doc, _ = _load("gpu-validation.yml")
    cond = doc["jobs"]["spend"]["if"]
    assert "inputs.mode == 'spend'" in cond
    assert "inputs.spend == 'SPEND'" in cond


def test_gate3_cancel_in_progress_false():
    doc, _ = _load("gpu-validation.yml")
    assert doc["concurrency"]["cancel-in-progress"] is False


def test_gate4_orphan_sweep_always_runs():
    doc, _ = _load("gpu-validation.yml")
    sweep = doc["jobs"]["sweep"]
    assert sweep["if"].strip() == "always()"
    assert set(sweep["needs"]) == {"discover", "spend"}


def test_gate5_no_endpoint_creation_anywhere():
    _, raw = _load("gpu-validation.yml")
    assert "check_endpoint_config" in raw
    import runpod_client

    assert not hasattr(runpod_client, "create_endpoint")
    assert not hasattr(runpod_client, "create_pod")


def test_gate6_r2_credentials_are_not_github_secrets():
    for name in ("gpu-validation.yml", "worker-ci.yml"):
        _, raw = _load(name)
        assert "R2_ACCESS_KEY_ID" not in raw
        assert "R2_SECRET_ACCESS_KEY" not in raw
        assert "R2_S3_ENDPOINT" not in raw


def test_spend_job_requires_the_gpu_spend_environment():
    doc, _ = _load("gpu-validation.yml")
    assert doc["jobs"]["spend"]["environment"] == "gpu-spend"


# ----------------------------------------------------------- worker-ci.yml


def test_worker_ci_holds_no_credential():
    _, raw = _load("worker-ci.yml")
    assert "RUNPOD_API_KEY" not in raw
    assert "secrets." not in raw


def test_worker_ci_runs_on_push_because_it_cannot_spend():
    doc, _ = _load("worker-ci.yml")
    assert "push" in _triggers(doc)


# ------------------------------------------------------------- Dockerfile


def _dockerfile_instructions():
    """Instruction lines only — the comments talk ABOUT the forbidden
    shapes, so a raw-text scan would trip on its own documentation."""
    with open(os.path.join(ROOT, "Dockerfile"), encoding="utf-8") as fh:
        return [
            line.strip()
            for line in fh
            if line.strip() and not line.strip().startswith("#")
        ]


def test_dockerfile_has_no_arg():
    assert not any(l.startswith("ARG") for l in _dockerfile_instructions())


def test_dockerfile_cmd_is_exec_form():
    lines = _dockerfile_instructions()
    assert 'CMD ["python3", "-u", "handler.py"]' in lines
    assert not any(l.startswith("ENTRYPOINT") for l in lines)


def test_dockerfile_user_sits_between_last_copy_and_cmd():
    lines = _dockerfile_instructions()
    last_copy = max(i for i, l in enumerate(lines) if l.startswith("COPY"))
    user = lines.index("USER oniq:oniq")
    cmd = next(i for i, l in enumerate(lines) if l.startswith("CMD"))
    assert last_copy < user < cmd


def test_dockerfile_copies_exactly_the_five_files():
    copies = [l for l in _dockerfile_instructions() if l.startswith("COPY")]
    copied = [l.split()[1] for l in copies]
    assert copied == [
        "requirements.txt",
        "/app/contract.py",
        "/app/preprocess.py",
        "/app/storage.py",
        "/app/handler.py",
    ] or copied == [
        "requirements.txt",
        "contract.py",
        "preprocess.py",
        "storage.py",
        "handler.py",
    ]


def test_dockerfile_never_copies_the_context_wholesale():
    for line in _dockerfile_instructions():
        if line.startswith("COPY"):
            sources = line.split()[1:-1]
            assert "." not in sources and "./" not in sources


def test_dockerignore_denies_by_default():
    with open(os.path.join(ROOT, ".dockerignore"), encoding="utf-8") as fh:
        lines = [l.strip() for l in fh if l.strip() and not l.startswith("#")]
    assert lines[0] == "*"
    assert set(lines[1:]) == {
        "!requirements.txt",
        "!contract.py",
        "!preprocess.py",
        "!storage.py",
        "!handler.py",
    }


def test_requirements_are_the_three_recorded_pins():
    with open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8") as fh:
        pins = [l.strip() for l in fh if l.strip()]
    assert pins == ["runpod==1.7.7", "pillow==11.0.0", "boto3==1.35.76"]


def test_dockerfile_env_protects_the_nonroot_runtime():
    lines = _dockerfile_instructions()
    joined = " ".join(lines)
    assert "PYTHONDONTWRITEBYTECODE=1" in joined
    assert "HOME=/home/oniq" in joined
