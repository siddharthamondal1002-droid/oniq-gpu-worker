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
        # modelroot joined 2026-08-30: every engine reaches its weights
        # through it, so the migration to a network volume is one seam
        # rather than three hardcoded paths.
        # cudaenv joined 2026-08-30 and is the handler's FIRST import:
        # PyTorch reads PYTORCH_CUDA_ALLOC_CONF once, when its allocator
        # initialises, so a value set later is present and ignored.
        "cudaenv.py",
        "modelroot.py",
        # modelhydrate puts an experimental checkpoint on the volume. It is
        # what makes a model change a data change.
        "modelhydrate.py",
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



def test_no_workflow_exceeds_githubs_dispatch_input_cap():
    """GitHub allows at most 25 workflow_dispatch inputs and refuses the
    whole file past that — the workflow becomes undispatchable, not just
    the new mode.

    Hit on 2026-08-30 at 27, after four modes each added their own token
    and value. The fix was to fold decisions already taken (a 45-minute
    ceiling, the datacenter the account's only volume already lives in)
    into module constants, where they are one line to edit rather than a
    value to re-type per run. This gate is what turns the next occurrence
    into a local failure instead of a rejected dispatch.
    """
    import yaml

    workflows = os.path.join(ROOT, ".github", "workflows")
    for name in sorted(os.listdir(workflows)):
        if not name.endswith((".yml", ".yaml")):
            continue
        with open(os.path.join(workflows, name), encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
        triggers = doc.get(True) or doc.get("on") or {}
        dispatch = (triggers or {}).get("workflow_dispatch") or {}
        inputs = dispatch.get("inputs") or {}
        assert len(inputs) <= 25, f"{name} has {len(inputs)} inputs"

def test_no_workflow_has_a_duplicate_key():
    """PyYAML accepts duplicate mapping keys and keeps the last one.
    GitHub's parser refuses the file outright.

    On 2026-08-30 an inserted input split an existing one, leaving two
    `default:` keys in the same block. yaml.safe_load said the file was
    fine; the dispatch came back "'default' is already defined" and the
    run never started. A gate that is more permissive than the thing it
    guards is not a gate.
    """
    import yaml

    class StrictLoader(yaml.SafeLoader):
        pass

    def _no_duplicates(loader, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = loader.construct_object(key_node, deep=deep)
            assert key not in seen, (
                f"duplicate key {key!r} at line {key_node.start_mark.line + 1}"
            )
            seen.add(key)
        return yaml.SafeLoader.construct_mapping(loader, node, deep)

    StrictLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicates
    )

    workflows = os.path.join(ROOT, ".github", "workflows")
    for name in sorted(os.listdir(workflows)):
        if not name.endswith((".yml", ".yaml")):
            continue
        with open(os.path.join(workflows, name), encoding="utf-8") as fh:
            yaml.load(fh, Loader=StrictLoader)

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
        "!cudaenv.py",
        "!modelroot.py",
        "!modelhydrate.py",
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
        # modelroot joined 2026-08-30 with the owner's directive to move
        # the weights onto a network volume. Every engine reaches its
        # weights through it, so it is the one module whose absence from
        # the image would break all three at once.
        "modelroot",
        # cudaenv sets PYTORCH_CUDA_ALLOC_CONF and MUST be the handler's
        # first import: PyTorch reads that variable once, when its CUDA
        # allocator initialises, and a value set afterwards is present and
        # ignored.
        "cudaenv",
        # modelhydrate fetches an experimental checkpoint onto the volume.
        # It is reached from the handler's hydrate op, which is how a model
        # change stops being a Docker rebuild.
        "modelhydrate",
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
        # 0.38.0 since 2026-08-29 (was 0.35.2 earlier the same day):
        # HunyuanVideo15ImageToVideoPipeline first ships in 0.36.0, and
        # 0.38.0 is the last release whose floors fit the baked torch
        # 2.5.1+cu121 and huggingface_hub 0.34.6 — 0.39.0 raises the torch
        # floor to >=2.6.
        "diffusers==0.38.0",
        # 4.57.1 since 2026-08-29 (was 4.51.3): the hunyuan pipeline
        # imports Qwen2_5_VLTextModel, absent below 4.52.0; 4.57.1 is the
        # version Tencent's own repository pins. Qwen3 still needs >=4.51,
        # which this satisfies.
        "transformers==4.57.1",
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
        # model_hydrate joined 2026-08-30 with the owner's "MODEL WEIGHTS
        # ARE DATA" directive. It is the one op here that runs NO
        # inference: it puts a checkpoint on the persistent volume so a
        # later model change is a configuration edit rather than a rebuild.
        # It still costs a booted worker, so it is an explicit choice like
        # every other paid op and never the default.
        "model_hydrate",
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
        # template-retarget joined 2026-08-29. It exists because RunPod
        # refuses a second template with the same name (500, measured), and
        # because updating in place keeps the env holding the R2 secret
        # references — a fresh template starts with none, so the endpoint
        # would run a worker that could not upload its output.
        "template-retarget",
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
        # volume-probe joined 2026-08-30, with the owner's directive to
        # move the baked weights onto a network volume. It is the READ
        # that has to come first: a network volume pins the endpoint to
        # one datacenter, so the migration is only possible if the A5000
        # is sold there. Every call in validation/volume_probe.py is a
        # GET and the module has no write path at all, so like
        # frames-pull and model-bench it cannot become a paid mode.
        "volume-probe",
        # endpoint-timeout joined 2026-08-30 with the owner's authorization
        # to raise the execution ceiling. It is the fourth job-level
        # mutation and the narrowest: ONE endpoint, ONE field, and the
        # module refuses the run if the PATCH moved a worker bound, a
        # template or a volume alongside it — the failure the template
        # retarget hit the same day, where a write that was not read back
        # dropped a registry credential nobody could see was gone.
        "endpoint-timeout",
        # hunyuan-preflight joined 2026-08-30 as section 12's free gate. It
        # reads checkpoint configs and endpoint state and submits nothing;
        # like frames-pull and model-bench it is a $0 mode that cannot
        # become a paid one.
        "hunyuan-preflight",
        # volume-setup joined 2026-08-30 as the fifth job-level mutation and
        # the first that creates recurring SPEND. Token-gated, one volume,
        # one endpoint, one field on the PATCH, and reversible by
        # detaching. The rate it reports is measured from this account's
        # own billing, never recalled.
        "volume-setup",
        # endpoint-template joined 2026-08-30 as the sixth job-level
        # mutation. The owner deleted the previous endpoints and created
        # 9gh6qbou1in8yb in the console; it came up naming a templateId the
        # account does not have — the hhhdwtjw0y dangling reference again.
        # This points the endpoint at a template that ALREADY EXISTS, so
        # unlike template-attach it creates nothing and cannot leave a
        # correct image sitting on a template with no R2 environment. It
        # refuses a target it cannot prove is on the account, which is the
        # difference between repairing a dangling reference and writing a
        # second one over it.
        "endpoint-template",
        # workers-min-zero joined 2026-08-30 with the owner's decision to
        # drop the floor rather than relax the admission gate. It is the
        # SECOND mode with no token, and it earns that the same way
        # standby-zero does: the writer it calls takes no value, so there
        # is no argument by which a spend reduction becomes an increase.
        "workers-min-zero",
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


def test_the_volume_probe_can_only_read():
    """Owner directive 2026-08-30 approved MOVING the weights to a network
    volume. It did not approve creating one blind, so the probe that
    informs it is read-only by construction: no mutating verb reaches the
    module, and the module names no write call. The datacenter cross-check
    is asserted too, because that — not the storage rate — is what decides
    whether the migration is possible at all: a volume pins the endpoint
    to one datacenter, and an endpoint pinned where the A5000 is not sold
    has traded a slow pull for no GPU."""
    doc, _ = _load("gpu-validation.yml")
    job = doc["jobs"]["volume_probe"]
    assert job["if"].strip() == "inputs.mode == 'volume-probe'"

    commands = " ".join(str(step.get("run", "")) for step in job["steps"])
    assert "validation.volume_probe" in commands
    for forbidden in ("SPEND", "spend_run", "submit", "create", "retarget"):
        assert forbidden not in commands

    with open(os.path.join(ROOT, "validation", "volume_probe.py"), encoding="utf-8") as fh:
        module = fh.read()
    # No mutating client call, and no mutating HTTP method.
    for forbidden in ("submit_job", "create_template", "retarget_template",
                      "set_template_env", "attach_template",
                      "set_workers_standby_zero", "purge_queue",
                      "cancel_job", 'method="PATCH"', 'method="DELETE"',
                      'method="PUT"'):
        assert forbidden not in module, forbidden
    # POST is admitted for exactly one reason: GraphQL sends READS over
    # POST, and the datacenter document lives there — the REST API does
    # not declare a datacenters path at all. So the guard is on the verb
    # the payload carries, not on the HTTP method: a GraphQL mutation is
    # what must never appear, and every POST must go to GRAPHQL_URL.
    # A GraphQL mutation would have to live in a STRING literal to be
    # sent, so that is where the check belongs. Scanning the whole file
    # matched the word in this module's own comment explaining the rule —
    # a guard that fires on its own documentation is a guard that gets
    # deleted rather than fixed.
    import ast as _ast
    for node in _ast.walk(_ast.parse(module)):
        if isinstance(node, _ast.Constant) and isinstance(node.value, str):
            assert "mutation" not in node.value.lower(), node.value[:120]
    for line in module.splitlines():
        if 'method="POST"' in line:
            assert "GRAPHQL_URL" in line, line
    assert module.count('method="POST"') == 1
    # The feasibility answer the probe exists to produce.
    assert "datacenter_overlap" in module
    assert "a5000_datacenters" in module
    # A rate the API does not state must not be filled in from memory.
    assert "NOT INVENTED HERE" in module

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
    # Four since 2026-08-30: the hunyuan preflight joined, because it now
    # fetches the reference to prove it is really an image, and a gate that
    # could not honour the override would read from a base nobody chose.
    assert raw.count("inputs.public_base || vars.R2_PUBLIC_BASE_URL") == 4, (
        "spend preflight, spend frames, the free frame pull and the hunyuan "
        "preflight must all honour the override"
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
    # The set is closed over two server-side tables and nothing else: the
    # benchmark ROWS a model_probe may measure, and the experimental MODEL
    # IDS a model_hydrate may fetch. Both are constants in this repository,
    # so a dispatch still cannot name a repository, a revision, a precision
    # or an offload strategy — it can only pick from what was reviewed.
    import modelroot

    assert set(probe["options"]) == (
        {""} | set(modelprobe.PROBE_MODELS) | set(modelroot.EXPERIMENTAL)
    )
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


def test_every_op_the_workflow_offers_is_dispatchable_and_admitted():
    """Registration is one property, not four.

    model_hydrate had a contract entry, a workflow choice, a driver branch
    and a handler — and was still rejected at dispatch, because the
    driver's own gate was a separate inline tuple nobody had updated. The
    first three being right said nothing about the fourth, and the failure
    only appeared after a workflow had spun up and run.

    These three lists must agree in both directions: anything the workflow
    lets someone choose must be dispatchable by the driver and admitted by
    the contract. A one-way check would let an op be offered and refused.
    """
    import contract
    from validation.spend_run import DISPATCHABLE_OPS

    doc, _ = _load("gpu-validation.yml")
    offered = {
        o for o in _triggers(doc)["workflow_dispatch"]["inputs"]["op"]["options"] if o
    }

    unroutable = offered - set(DISPATCHABLE_OPS)
    assert not unroutable, (
        f"the workflow offers {sorted(unroutable)} but spend_run refuses them — "
        "a dispatch would fail after the run has already started"
    )
    unadmitted = set(DISPATCHABLE_OPS) - set(contract.ALLOWED_OPS)
    assert not unadmitted, (
        f"the driver would dispatch {sorted(unadmitted)} but the worker's "
        "contract refuses them — the job would be rejected on a booted worker"
    )


def test_no_step_run_block_contains_a_yaml_key_from_a_neighbour():
    """A split step is a step that runs YAML as shell.

    On 2026-08-30 an inserted step landed mid-block and left
    `retention-days: 30` inside a `run:` script. Bash tried to execute it,
    the step died with 127 (command not found), and the diagnosis it
    existed to print was lost — the exact failure it had been added to
    prevent.

    yaml.safe_load cannot see this: the file parses, the script is just a
    string. So the scripts themselves are scanned for lines that are
    obviously an action input rather than a command.
    """
    import re

    import yaml

    workflows = os.path.join(ROOT, ".github", "workflows")
    suspicious = re.compile(
        r"^\s*(retention-days|if-no-files-found|compression-level|"
        r"overwrite|include-hidden-files|python-version|fetch-depth|"
        r"path|name):\s*\S", re.M
    )
    for filename in sorted(os.listdir(workflows)):
        if not filename.endswith((".yml", ".yaml")):
            continue
        with open(os.path.join(workflows, filename), encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
        for job_name, job in (doc.get("jobs") or {}).items():
            for step in job.get("steps") or []:
                script = step.get("run")
                if not isinstance(script, str):
                    continue
                # A heredoc legitimately contains arbitrary text; only the
                # shell-level lines are checked.
                lines, in_heredoc = [], False
                for line in script.splitlines():
                    if re.search(r"<<\s*'?[A-Z_]+'?", line):
                        in_heredoc = True
                        continue
                    if in_heredoc and re.match(r"^\s*[A-Z_]+\s*$", line):
                        in_heredoc = False
                        continue
                    if not in_heredoc:
                        lines.append(line)
                hit = suspicious.search("\n".join(lines))
                assert not hit, (
                    f"{filename} job {job_name} step "
                    f"{step.get('name', '?')!r} has an action input inside "
                    f"its run script: {hit.group(0).strip()!r} — the step "
                    "was split and bash will try to execute it"
                )


def test_the_readonly_workflow_cannot_spend_and_runs_beside_the_paid_one():
    """gpu-readonly.yml exists so a $0 read can happen WHILE a paid run
    holds gpu-validation's lock.

    Two properties make it safe, and both are asserted rather than
    intended: a DIFFERENT concurrency group (otherwise it queues behind
    the very run it is meant to observe, which is the blindness it was
    written to remove), and no reachable mutating verb.
    """
    doc, raw = _load("gpu-readonly.yml")
    paid, _ = _load("gpu-validation.yml")

    assert doc["concurrency"]["group"] != paid["concurrency"]["group"], (
        "sharing the paid workflow's group would queue every read behind "
        "the run it exists to look at"
    )
    assert doc["concurrency"]["cancel-in-progress"] is False

    # Only read-only modules, and only the three read-only modes.
    mode = _triggers(doc)["workflow_dispatch"]["inputs"]["mode"]
    assert mode["options"] == ["queue-probe", "template-probe", "volume-probe"]
    for allowed in ("validation.queue_probe", "validation.template_probe",
                    "validation.volume_probe"):
        assert allowed in raw

    # The spend driver and every endpoint mutation are unreachable from here.
    for forbidden in ("spend_run", "standby_zero", "workers_min_zero",
                      "template_attach", "template_retarget", "template_env",
                      "endpoint_timeout", "endpoint_template", "volume_setup",
                      "stale_run", "SPEND"):
        assert forbidden not in raw, forbidden

    # And the modules it DOES name hold no mutating verb themselves.
    import inspect

    from validation import queue_probe, template_probe, volume_probe

    for module in (queue_probe, template_probe, volume_probe):
        source = inspect.getsource(module)
        for verb in ("attach_template", "attach_network_volume",
                     "create_network_volume", "set_execution_timeout",
                     "retarget_template", "create_template",
                     "set_template_env", "set_workers_min_zero",
                     "set_workers_standby_zero", "purge_queue", "run_sync"):
            assert verb not in source, f"{module.__name__}: {verb}"
