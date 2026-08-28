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
    # The load-bearing part: the sweep must run after the job that can
    # PROVISION a worker, or an orphan outlives the run that made it.
    assert "spend" in sweep["needs"]
    # The full set, exact, so a new job cannot quietly skip the sweep.
    # standby_probe joined 2026-08-27 (read-only standby diagnostic).
    assert set(sweep["needs"]) == {
        "discover",
        "spend",
        "standby",
        "standby_probe",
    }


def test_gate5_no_endpoint_creation_anywhere():
    _, raw = _load("gpu-validation.yml")
    assert "validation.spend_run" in raw  # gates live in tested code
    import runpod_client

    assert not hasattr(runpod_client, "create_endpoint")
    assert not hasattr(runpod_client, "create_pod")


def test_no_separate_preflight_job_blocks_the_spend_path():
    # Owner directive 2026-08-25: every check runs INSIDE the spend job,
    # immediately before provisioning, behind the environment pause — no
    # separate preflight job may gate (or block) the authorized run.
    doc, _ = _load("gpu-validation.yml")
    assert "preflight" not in doc["jobs"]
    assert "needs" not in doc["jobs"]["spend"]


def test_spend_job_runs_the_tested_driver():
    doc, _ = _load("gpu-validation.yml")
    runs = [s.get("run", "") for s in doc["jobs"]["spend"]["steps"]]
    assert any("validation.spend_run run" in r for r in runs)


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


def test_dockerfile_copies_exactly_the_shipped_files():
    copies = [l for l in _dockerfile_instructions() if l.startswith("COPY")]
    copied = [l.split()[1] for l in copies]
    assert copied == [
        "requirements.txt",
        "contract.py",
        "preprocess.py",
        "storage.py",
        "videogen.py",
        "storygen.py",
        "audio.py",
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
        "!videogen.py",
        "!storygen.py",
        "!audio.py",
        "!handler.py",
    }


# The two lists above are enumerations, and on 2026-08-27 both were
# wrong in the same way: storygen.py was added to the repo and imported
# by handler.py, and neither list learned about it. Every offline test
# stayed green — nothing tied the enumerations to what the worker
# actually imports — and the defect surfaced only when the built image
# ran `import storygen` and died. So the enumerations are no longer the
# gate. The gate is the closure below: it reads handler.py, follows its
# first-party imports transitively, and demands that every module it
# reaches be present in BOTH locks. Adding an engine and forgetting to
# ship it is now a test failure, not a broken image.


def _shipped_module_closure():
    """Every first-party module the worker's entrypoint actually needs.

    A first-party module is one that exists as a top-level .py file in
    the repo; stdlib and site-packages imports are not our problem.
    Deferred imports (inside a function, as videogen and storygen do for
    torch) count exactly as much as top-level ones — the module still has
    to be in the image — so this walks the whole AST, not just its head.
    """
    import ast

    local = {
        name[:-3]
        for name in os.listdir(ROOT)
        if name.endswith(".py") and os.path.isfile(os.path.join(ROOT, name))
    }

    seen, pending = set(), ["handler"]
    while pending:
        module = pending.pop()
        if module in seen:
            continue
        seen.add(module)
        with open(os.path.join(ROOT, module + ".py"), encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                # level > 0 is a relative import; the worker has none.
                found = [(node.module or "").split(".")[0]]
            else:
                continue
            pending.extend(name for name in found if name in local)
    return seen


def test_every_module_the_handler_imports_is_in_both_locks():
    needed = {module + ".py" for module in _shipped_module_closure()}

    copied = {
        l.split()[1]
        for l in _dockerfile_instructions()
        if l.startswith("COPY")
    }
    missing_copy = needed - copied
    assert not missing_copy, (
        f"{sorted(missing_copy)} reachable from handler.py but never COPYed "
        "into the image — the worker will die on import at start-up"
    )

    with open(os.path.join(ROOT, ".dockerignore"), encoding="utf-8") as fh:
        admitted = {
            l.strip()[1:]
            for l in fh
            if l.strip().startswith("!")
        }
    missing_admit = needed - admitted
    assert not missing_admit, (
        f"{sorted(missing_admit)} is COPYed but denied by .dockerignore — "
        "the build itself will fail"
    )


def test_the_closure_actually_reaches_the_engines():
    # Guards the guard: a closure walk that silently found nothing would
    # make the test above vacuously true.
    closure = _shipped_module_closure()
    assert closure == {
        "handler",
        "contract",
        "preprocess",
        "storage",
        "videogen",
        "storygen",
        "audio",
    }
    # runpod_client is the CI harness's, not the worker's. It must NOT be
    # in the image: it is the only module that talks to the RunPod API.
    assert "runpod_client" not in closure


def test_requirements_are_the_recorded_pins():
    # numpy joined 2026-08-26: torch 2.x does not depend on it, and the
    # first job to reach the GPU path died on tensor.numpy() without it.
    # The media pins joined the same day for LTX-Video image-to-video.
    with open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8") as fh:
        pins = [l.strip() for l in fh if l.strip() and not l.startswith("#")]
    assert pins == [
        "runpod==1.7.7",
        "pillow==11.0.0",
        "boto3==1.35.76",
        "numpy==2.1.3",
        "diffusers==0.33.1",
        # Bumped 2026-08-27 for Qwen3, which raises KeyError: 'qwen3'
        # below 4.51. Proven against the diffusers/torch pins LTX runs on
        # by the image job's coexistence step before anything was baked.
        "transformers==4.51.3",
        "accelerate==1.2.1",
        "sentencepiece==0.2.0",
        "protobuf==5.29.3",
        "imageio==2.36.1",
        "imageio-ffmpeg==0.5.1",
        # audio_mux (2026-08-26): in-process TTS + mux — no shell surface.
        "piper-tts==1.2.0",
        "piper-phonemize==1.1.0",
        "onnxruntime==1.29.0",
        "av==13.1.0",
        # 4-bit loading for the story model, which runs ALONE by design.
        "bitsandbytes==0.45.0",
    ]


def test_dockerfile_env_protects_the_nonroot_runtime():
    lines = _dockerfile_instructions()
    joined = " ".join(lines)
    assert "PYTHONDONTWRITEBYTECODE=1" in joined
    assert "HOME=/home/oniq" in joined


# --------------------------------------------------------- media workload


def test_op_input_defaults_to_the_image_workload():
    # video_generate and audio_mux must be explicit dispatch choices,
    # never a default a habitual re-run could trip into.
    doc, raw = _load("gpu-validation.yml")
    op = _triggers(doc)["workflow_dispatch"]["inputs"]["op"]
    assert op["default"] == "image_preprocess"
    # image_generate is ONIQ's own image engine (fully in-house directive,
    # 2026-08-27): a real generative pass, so it is an explicit choice on
    # the same footing as the other paid workloads — never the default.
    assert op["options"] == [
        "image_preprocess",
        "image_generate",
        "video_generate",
        "audio_mux",
    ]
    assert "OP: ${{ inputs.op }}" in raw


def test_standby_zero_mode_is_gated_and_carries_no_worker_count():
    # The one endpoint mutation: its own dispatch mode, its own job, and
    # NO input anywhere that could carry a worker count — the zero lives
    # as a literal in runpod_client.set_workers_standby_zero.
    doc, raw = _load("gpu-validation.yml")
    mode = _triggers(doc)["workflow_dispatch"]["inputs"]["mode"]
    # "advisory" (2026-08-27) is the standalone free preflight — a $0 read
    # of the live endpoint that gates nothing and can spend nothing.
    # standby-probe joined 2026-08-27: the READ-ONLY twin of standby-zero,
    # so the diagnostic can be re-run without re-attempting a mutation the
    # API has already refused.
    # queue-probe and stale-cancel joined 2026-08-28. The first is a
    # pure read of queue counts. The second is the ONLY job-level
    # mutation on this workflow, and it is deliberately the narrow one:
    # cancel ONE named run, never purge_queue.
    assert mode["options"] == [
        "discover",
        "spend",
        "standby-zero",
        "standby-probe",
        "advisory",
        "queue-probe",
        "stale-cancel",
        "template-probe",
    ]
    assert mode["default"] == "discover"
    standby = doc["jobs"]["standby"]
    assert standby["if"].strip() == "inputs.mode == 'standby-zero'"
    runs = [s.get("run", "") for s in standby["steps"]]
    assert any("validation.standby_zero" in r for r in runs)
    for name, spec in _triggers(doc)["workflow_dispatch"]["inputs"].items():
        assert "standby" not in name
        assert "worker" not in name


def test_worker_ci_builds_only_the_weightless_base_stage():
    # GitHub runners must never download the model: CI builds --target
    # base; the media stage bakes weights only on RunPod's builder.
    _, raw = _load("worker-ci.yml")
    assert "--target base" in raw


def test_dockerfile_media_stage_loads_locally_only():
    with open(os.path.join(ROOT, "Dockerfile"), encoding="utf-8") as fh:
        raw = fh.read()
    assert "FROM base AS media" in raw
    assert "snapshot_download" in raw  # bake at BUILD time...
    with open(os.path.join(ROOT, "videogen.py"), encoding="utf-8") as fh:
        assert "local_files_only=True" in fh.read()  # ...never at job time


def test_the_standby_probe_job_cannot_spend_or_write():
    """The read-only twin. It exists so the diagnostic is repeatable
    without re-attempting a mutation the API has already refused, so the
    thing to prove is that it stayed read-only."""
    doc, _ = _load("gpu-validation.yml")
    job = doc["jobs"]["standby_probe"]
    assert job["if"].strip() == "inputs.mode == 'standby-probe'"
    commands = " ".join(
        str(step.get("run", "")) for step in job["steps"]
    )
    # The probe argv, and nothing that patches or submits.
    assert "validation.standby_zero probe" in commands
    for forbidden in ("spend_run", "SPEND", "submit", "standby_zero\n"):
        assert forbidden not in commands


def test_the_queue_probe_job_is_read_only():
    doc, _ = _load("gpu-validation.yml")
    job = doc["jobs"]["queue_probe"]
    assert job["if"].strip() == "inputs.mode == 'queue-probe'"
    commands = " ".join(str(step.get("run", "")) for step in job["steps"])
    assert "validation.queue_probe" in commands
    for forbidden in ("stale_run", "spend_run", "standby_zero", "SPEND"):
        assert forbidden not in commands


def test_the_stale_cancel_job_needs_the_literal_token_and_never_purges():
    """The narrow mutation. Owner authorization 2026-08-28 covered ONE
    stranded run, so the gate proves the job cannot become a queue drain:
    it is token-gated, it names one run and one endpoint from the
    dispatch, and purge_queue appears nowhere in it or in the module it
    calls."""
    doc, _ = _load("gpu-validation.yml")
    job = doc["jobs"]["stale_cancel"]
    condition = job["if"].strip()
    assert "inputs.mode == 'stale-cancel'" in condition
    assert "inputs.cancel_token == 'CANCEL-STALE-RUN'" in condition

    commands = " ".join(str(step.get("run", "")) for step in job["steps"])
    assert "validation.stale_run" in commands
    assert "inputs.stale_job_id" in commands
    assert "inputs.endpoint_id" in commands
    for forbidden in ("purge", "purge_queue", "submit", "SPEND", "spend_run"):
        assert forbidden not in commands

    with open(os.path.join(ROOT, "validation", "stale_run.py"), encoding="utf-8") as fh:
        module = fh.read()
    assert "purge_queue" not in module
    assert "submit_job" not in module
