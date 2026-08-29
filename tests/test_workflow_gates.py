"""The six spend gates, verified by parsing the files — not by reading
them. A gate that only exists in prose is not a gate."""

import json
import os
import pathlib

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
        # preview joined 2026-08-29: the bucket is private, so the only way
        # to LOOK at what the worker made is for the worker to hand a
        # thumbnail back with the reply.
        "preview.py",
        "videogen.py",
        # modelprobe joined 2026-08-29 with the benchmark op.
        "modelprobe.py",
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
        "!preview.py",
        "!videogen.py",
        "!modelprobe.py",
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
        "preview",
        "videogen",
        "storygen",
        "audio",
        # modelprobe joined 2026-08-29 with the benchmark op. It is imported
        # inside a handler branch rather than at module top, and the closure
        # walk finds it anyway — which is the point: storygen once reached CI
        # missing from both locks and the image died on import at start-up.
        "modelprobe",
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
        # 0.35.2 since 2026-08-29: the first minor whose Wan i2v pipeline
        # carries transformer_2/boundary_ratio, without which Wan2.2's
        # two-expert mixture cannot be driven at all.
        "diffusers==0.35.2",
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
        # Model-benchmark runtime deps (2026-08-29). Named rather than left
        # transitive: a pin that arrives only through somebody else's
        # dependency tree is one a resolver can take away, and Wan's i2v
        # pipeline imports regex at module top — a hard import, not an
        # optional one.
        "huggingface_hub==0.34.6",
        "regex==2026.7.19",
        "ftfy==6.3.1",
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
        # model_probe joined 2026-08-29 with the open-source benchmark. It
        # spends like the others and is an explicit choice like the others;
        # what makes it different is that it downloads a checkpoint the image
        # does not carry, which is why it can never be a default.
        "model_probe",
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
        # image-size joined 2026-08-28 with the owner's route-2 directive:
        # it builds only the base stage and reads file sizes from the model
        # registry, so it spends nothing and can mutate nothing.
        "image-size",
        # template-attach is the other half of that directive and the
        # second job-level mutation on this workflow: ONE template created
        # naming an already-published image, ONE endpoint repointed. It
        # sends templateId alone, so no spend bound can move.
        "template-attach",
        # template-env joined 2026-08-28 too, and is the third job-level
        # mutation: it writes the env field of ONE template with RunPod
        # secret REFERENCES, never a credential, and sends env alone so
        # the image cannot move underneath the endpoint. This list stays
        # pinned so a new mutation is a deliberate act rather than
        # something that arrives inside a diff.
        "template-env",
        # ltx-discover joined 2026-08-28 after run 7 measured that the
        # Dockerfile names a checkpoint which does not exist. Read-only,
        # and it picks nothing.
        "ltx-discover",
        # frames-pull joined 2026-08-29 (owner directive: judge LTX from
        # frames, not metadata). It READS objects that already exist over
        # the bucket's public base and cuts stills out of them. It holds
        # no RunPod credential at all, so it is a $0 mode that cannot
        # become a paid one by mistake.
        "frames-pull",
        # model-bench joined 2026-08-29 with the owner's expanded benchmark
        # brief. It reads the HuggingFace registry and does arithmetic on
        # file sizes; it submits no job and holds no RunPod credential, so
        # like frames-pull it is a $0 mode that cannot turn into a paid one.
        "model-bench",
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


# ------------------------------------------------------- image-publish.yml


def _publish_steps():
    doc, _ = _load("image-publish.yml")
    return doc["jobs"]["publish"]["steps"]


def test_publish_is_dispatch_only_behind_a_literal_token():
    doc, _ = _load("image-publish.yml")
    assert set(_triggers(doc)) == {"workflow_dispatch"}
    assert doc["jobs"]["publish"]["if"] == "inputs.publish == 'PUBLISH-MEDIA-IMAGE'"


def test_publish_carries_no_runpod_credential():
    """Pushing an image must not be able to start a worker. Keeping the
    two credentials in different workflows means a mistake here cannot
    become a GPU bill. Asserted on the CREDENTIAL and the CALLERS, not on
    the word: this workflow's own header explains at length why RunPod
    cannot build the image, and prose is not a capability."""
    doc, _ = _load("image-publish.yml")
    job = doc["jobs"]["publish"]

    # Comments stripped, the same way _dockerfile_instructions does it:
    # this workflow's header explains at length why RunPod cannot build
    # the image, and a raw-text scan flags its own documentation. That
    # trap has now been walked into twice in this file.
    def _instructions(step):
        return "\n".join(
            line for line in (step.get("run") or "").splitlines()
            if not line.strip().startswith("#")
        )

    referenced = repr(job.get("env") or {}) + repr(doc.get("env") or {})
    for step in job["steps"]:
        referenced += repr(step.get("env") or {}) + _instructions(step)

    assert "RUNPOD" not in referenced.upper()
    assert "runpod_client" not in referenced
    assert "template_attach" not in referenced
    assert "runpod.io" not in referenced


def test_publish_never_passes_the_credential_as_a_build_argument():
    """ARG and ENV both survive into the published image where docker
    history and docker inspect can read them; a secret mount does not."""
    _, raw = _load("image-publish.yml")
    assert "--build-arg" not in raw.replace("grep -qiE 'build-arg", "")
    assert "--secret id=hf_token,env=HF_TOKEN" in raw


def test_every_python_call_happens_before_the_toolcache_is_deleted():
    """The reclaim removes /opt/hostedtoolcache, which is where
    actions/setup-python puts the interpreter. A python call after it
    would fail on a runner that had just been made able to build."""
    steps = _publish_steps()
    reclaim_at = next(
        i for i, s in enumerate(steps)
        if "/opt/hostedtoolcache || true" in (s.get("run") or "")
    )
    for step in steps[reclaim_at:]:
        run = step.get("run") or ""
        # Join backslash continuations FIRST: `docker run ... \` followed
        # by `python3 -c ...` is one command, and reading it line by line
        # would flag the image's own interpreter as if it were the
        # runner's — which is exactly what this test did on its first run.
        for command in run.replace("\\\n", " ").splitlines():
            stripped = command.strip()
            if stripped.startswith("#"):
                continue
            # python3 inside `docker run` is the IMAGE's interpreter, which
            # the runner's toolcache has nothing to do with.
            if "docker run" in stripped or "/proofs/" in stripped:
                continue
            assert not stripped.startswith("python"), (step.get("name"), stripped)
            assert " python " not in f" {stripped} ", (step.get("name"), stripped)


def test_the_proofs_all_run_before_the_push():
    """A pushed image that cannot start, or carries the wrong model, turns
    a visible dangling reference into a worker that fails at cost."""
    steps = _publish_steps()
    names = [s.get("name") or s.get("uses") for s in steps]
    push_at = next(i for i, n in enumerate(names) if n and "Push" in n)
    for needle in ("The image starts", "uid 10001", "LTX 2B", "credential did NOT"):
        at = next(i for i, n in enumerate(names) if n and needle in n)
        assert at < push_at, needle


# ------------------------------------------- gate 7: frames, never metadata
#
# Owner directive 2026-08-29: LTX quality is judged from actual frames.
# The route generation -> R2 -> extraction -> artifact must be automatic,
# and it must be incapable of costing money or holding a credential.


def _spend_steps():
    doc, _ = _load("gpu-validation.yml")
    return doc["jobs"]["spend"]["steps"]


def test_gate7_frames_are_pulled_only_after_the_spend_has_finished():
    steps = _spend_steps()
    names = [s.get("name") or s.get("uses") or "" for s in steps]
    spend = next(i for i, n in enumerate(names) if "Phases 12-19" in n)
    pull = next(i for i, n in enumerate(names) if "Frames from the clips" in n)
    upload = next(i for i, n in enumerate(names) if "The frames, for inspection" in n)
    assert spend < pull < upload, (
        "a frame step before the spend could delay or block an authorized run"
    )


def test_gate7_no_frame_step_can_fail_a_paid_generation():
    # The GPU work is finished and billed by the time these run. A
    # missing ffmpeg, an unset variable or a 404 must never turn a
    # verified, paid generation into a red run.
    for step in _spend_steps():
        name = step.get("name") or step.get("uses") or ""
        if any(k in name for k in ("ffmpeg", "Frames from", "Frames inline", "The frames")):
            assert step.get("if") == "always()", f"{name} is not if: always()"


def test_gate7_the_read_base_is_a_variable_and_never_a_secret():
    _, raw = _load("gpu-validation.yml")
    assert "vars.R2_PUBLIC_BASE_URL" in raw
    assert "secrets.R2_PUBLIC_BASE_URL" not in raw, (
        "a public read base carried as a secret invites a presigned URL"
    )
    # And no R2 WRITE credential ever reaches CI (gate 6, restated here
    # because this is the change that made CI touch the bucket at all).
    for forbidden in (
        "secrets.R2_ACCESS_KEY_ID",
        "secrets.R2_SECRET_ACCESS_KEY",
        "secrets.R2_S3_ENDPOINT",
    ):
        assert forbidden not in raw


def test_gate7_the_frame_puller_never_submits_a_job():
    # It reads objects that already exist. If it could reach RunPod it
    # could spend, and "do not launch a GPU job to test plumbing" would
    # depend on nobody making a mistake.
    path = os.path.join(ROOT, "validation", "frame_pull.py")
    src = open(path, encoding="utf-8").read()
    for forbidden in ("runpod", "RUNPOD_API_KEY", "/run", "submit"):
        assert forbidden not in src, f"frame_pull references {forbidden!r}"


def test_gate7_the_free_frame_pull_job_holds_no_runpod_credential():
    # It reads objects that already exist. Giving it the key would make a
    # $0 job one mistake away from a paid one.
    doc, _ = _load("gpu-validation.yml")
    job = doc["jobs"]["frames_pull"]
    assert "RUNPOD_API_KEY" not in json.dumps(job)
    assert "environment" not in job, "gpu-spend is for jobs that can spend"


def test_gate7_dispatch_inputs_never_reach_a_shell_directly():
    # A workflow_dispatch input interpolated into a `run:` script is a
    # command injection. public_base and frame_keys travel as env.
    doc, _ = _load("gpu-validation.yml")
    for job in doc["jobs"].values():
        for step in job.get("steps", []):
            script = step.get("run") or ""
            for name in ("inputs.public_base", "inputs.frame_keys"):
                assert name not in script, f"{name} interpolated into a run: script"


def test_gate7_the_read_base_input_overrides_the_variable_everywhere():
    _, raw = _load("gpu-validation.yml")
    # Three sites now: the spend step itself (the battery PROVES both
    # conditioning plates exist over the base before its first paid
    # submission — the run-72 lesson, times five), the spend job's frame
    # step, and the free frame pull.
    assert raw.count("inputs.public_base || vars.R2_PUBLIC_BASE_URL") == 3, (
        "spend preflight, spend frames, and the free frame pull must all "
        "honour the override"
    )


# ------------------------------- gate 8: the private-bucket read path
#
# Owner directive 2026-08-29: production R2 stays PRIVATE. Validation
# clips are read through a presigned, read-only, single-object,
# short-lived URL. That URL is a credential, and the whole point of not
# opening the bucket is lost if it then leaks into a public CI log.


def test_gate8_signed_urls_are_masked_before_any_step_can_echo_one():
    steps = _spend_frame_steps()
    names = [s.get("name") or "" for s in steps]
    mask = next(i for i, n in enumerate(names) if "Mask the signed URLs" in n)
    use = next(i for i, n in enumerate(names) if "signed URLs (private bucket)" in n)
    assert mask < use, "a credential must be masked before it is used"
    assert "::add-mask::" in (steps[mask].get("run") or "")


def test_gate8_the_signed_url_never_reaches_a_shell_argument():
    # An argv is visible in `ps`; the URL travels as env on every step.
    for step in _spend_frame_steps():
        assert "inputs.signed_urls" not in (step.get("run") or "")


def test_gate8_the_two_read_paths_are_mutually_exclusive():
    steps = {s.get("name"): s for s in _spend_frame_steps()}
    public = steps["Frames from objects that already exist (public read)"]
    signed = steps["Frames through signed URLs (private bucket)"]
    assert public["if"] == "inputs.signed_urls == ''"
    assert signed["if"] == "inputs.signed_urls != ''"
    # The signed path is handed no public base — the URL carries its own
    # authority and a base has nothing to contribute.
    assert "R2_PUBLIC_BASE_URL" not in json.dumps(signed.get("env", {}))


def test_gate8_the_frame_job_still_cannot_reach_r2_with_credentials():
    doc, _ = _load("gpu-validation.yml")
    blob = json.dumps(doc["jobs"]["frames_pull"])
    for forbidden in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_S3_ENDPOINT"):
        assert forbidden not in blob, "the read path never gains write access"


def _spend_frame_steps():
    doc, _ = _load("gpu-validation.yml")
    return doc["jobs"]["frames_pull"]["steps"]


def test_gate9_the_plate_input_is_a_closed_choice():
    # Multi-reference conditioning (owner 2026-08-29): the dispatch picks
    # WHICH plate to draw, never what it contains — a closed a/b choice,
    # with the prompts as module constants in the tested driver.
    doc, raw = _load("gpu-validation.yml")
    plate = _triggers(doc)["workflow_dispatch"]["inputs"]["plate"]
    assert plate["type"] == "choice"
    # "ref" joined 2026-08-29: the model benchmark's controlled reference —
    # ONE adult, plain background, drawn once and shared by all five
    # candidates. Still a closed choice; the dispatch picks WHICH reference,
    # never what is in it.
    assert plate["options"] == ["a", "b", "ref"]
    assert plate["default"] == "a"
    assert "PLATE: ${{ inputs.plate }}" in raw


def test_gate9_the_probe_model_input_is_a_closed_choice():
    """The dispatch names a benchmark ROW, never a repository. Every
    checkpoint, revision, precision and offload strategy is a server-side
    constant in modelprobe, so a dispatch cannot redirect what is downloaded
    or how it is run."""
    import modelprobe

    doc, raw = _load("gpu-validation.yml")
    probe = _triggers(doc)["workflow_dispatch"]["inputs"]["probe_model"]
    assert probe["type"] == "choice"
    assert probe["default"] == ""
    assert set(probe["options"]) == {""} | set(modelprobe.PROBE_MODELS)
    assert "PROBE_MODEL: ${{ inputs.probe_model }}" in raw
    # Hunyuan is not offerable: its architecture did not resolve without
    # guessing, and an option nobody can select is how that stays true.
    for absent in modelprobe.NOT_EVALUATED:
        assert absent not in probe["options"]


def test_the_model_bench_job_cannot_spend():
    """The expanded benchmark's free half. Measuring which models MIGHT be
    worth paying for must never itself become a way to pay for one, so the
    job holds no RunPod credential and submits nothing."""
    doc, _ = _load("gpu-validation.yml")
    job = doc["jobs"]["model_bench"]
    assert job["if"].strip() == "inputs.mode == 'model-bench'"
    assert "RUNPOD_API_KEY" not in json.dumps(job)
    runs = " ".join(s.get("run", "") for s in job["steps"])
    assert "validation.model_bench" in runs
    for forbidden in ("runpod", "/run", "submit"):
        assert forbidden not in runs.lower()


def test_the_model_bench_module_never_reaches_runpod():
    """Belt and braces: the job is fenced above, and the module it runs has
    no way to reach the provider even if the workflow changed."""
    source = pathlib.Path("validation/model_bench.py").read_text(encoding="utf-8")
    source += pathlib.Path("validation/model_registry.py").read_text(encoding="utf-8")
    for forbidden in ("runpod", "RUNPOD_API_KEY", "api.runpod.ai"):
        assert forbidden not in source.lower()
