import os

import pytest

import contract
import modelprobe


class FakeCuda:  # noqa: D101
    """A CUDA that reports what we tell it to, so the whole probe runs on CPU."""

    def __init__(self, available=True, total=24 * 1024**3, peak=10 * 1024**3):
        self._available = available
        self._total = total
        self._peak = peak
        self.reset_calls = 0

    def is_available(self):
        return self._available

    def mem_get_info(self):
        return (self._total // 2, self._total)

    def memory_allocated(self):
        return 1024**3

    def memory_reserved(self):
        return 2 * 1024**3

    def max_memory_allocated(self):
        return self._peak

    def max_memory_reserved(self):
        return self._peak + 1024**3

    def reset_peak_memory_stats(self):
        self.reset_calls += 1

    def get_device_name(self, index):
        return "NVIDIA RTX A5000"


class FakeTorch:
    def __init__(self, **kw):
        self.cuda = FakeCuda(**kw)


def fake_pipe(frames=None):
    def pipe(**kwargs):
        pipe.called_with = kwargs
        return frames if frames is not None else [b"frame"] * 3
    return pipe


def probe_job(model="cogvideox-i2v", prompt="a person turns toward the camera"):
    return {"op": "model_probe", "model": model, "input_key": "ref.png",
            "output_key": "out.mp4", "params": {"prompt": prompt}}


@pytest.fixture
def rig(tmp_path, monkeypatch):
    """Everything except the CUDA pass, wired to fakes."""
    ref = tmp_path / "ref.png"
    ref.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    out = tmp_path / "out.mp4"

    monkeypatch.setattr(modelprobe, "_load_reference", lambda p: "IMAGE")
    monkeypatch.setattr(
        modelprobe, "_encode",
        lambda frames, path, fps: open(path, "wb").write(b"\x00\x00\x00\x18ftypmp42"),
    )
    monkeypatch.setattr(modelprobe, "free_disk_bytes", lambda p="/tmp": 500 * 1024**3)
    # The fetch is a seam of its own now: no test reaches the network, and
    # the download phase is exercised on the same path the real one takes.
    monkeypatch.setattr(modelprobe, "dir_bytes", lambda path: 7 * 1024**3)
    return {"ref": str(ref), "out": str(out), "fetch": lambda: str(tmp_path)}


# ------------------------------------------------------------ the model table


def test_only_the_four_authorised_candidates_are_probeable():
    assert set(modelprobe.PROBE_MODELS) == {
        "ltx-13b", "wan21-i2v-480p", "wan22-i2v-a14b", "cogvideox-i2v",
    }


def test_hunyuan_is_absent_and_named_not_evaluated():
    """Owner rule: probe it only if the architecture resolves without guessing.
    On 2026-08-29 it did not, so it is recorded as unevaluated rather than
    given a row built on inference."""
    assert "hunyuanvideo-1.5-i2v" not in modelprobe.PROBE_MODELS
    assert modelprobe.NOT_EVALUATED["hunyuanvideo-1.5-i2v"] == "ARCHITECTURE_NOT_RESOLVED"


def test_every_candidate_pins_a_revision_and_records_its_licence():
    """A repository name names a moving branch; the licence recorded is the
    licence at that commit, which is the only form of the claim that stays
    true."""
    for key, row in modelprobe.PROBE_MODELS.items():
        assert len(row["revision"]) == 40, key
        assert row["licence"], key
        assert row["repo"].count("/") == 1, key


def test_wan21_and_wan22_are_separate_rows_with_separate_downloads():
    """Owner rule: independent candidates, never combined, and 14B does not
    imply identical memory behaviour."""
    a = modelprobe.PROBE_MODELS["wan21-i2v-480p"]
    b = modelprobe.PROBE_MODELS["wan22-i2v-a14b"]
    assert a["repo"] != b["repo"]
    assert a["revision"] != b["revision"]
    assert a["download_gib"] != b["download_gib"]


def test_every_candidate_bounds_what_it_downloads():
    """`snapshot_download` with no allow_patterns takes whatever the repository
    happens to contain. That is not a size estimate being slightly off — it is
    an unbounded fetch onto a disk the probe shares with production."""
    for key, row in modelprobe.PROBE_MODELS.items():
        assert row["allow"], key
        assert "model_index.json" in row["allow"], key
        assert "*" not in row["allow"], key
        assert "**" not in row["allow"], key


def test_the_ltx_repo_names_its_vae_files_rather_than_globbing_them():
    """huggingface_hub's fnmatch lets `*` CROSS a slash, and this repository
    nests a second complete copy of itself under vae/ — 42 GiB of it. `vae/*`
    would quietly pull the lot, which is exactly the bug the allow list is
    here to prevent."""
    allow = modelprobe.PROBE_MODELS["ltx-13b"]["allow"]
    assert "vae/*" not in allow
    assert "vae/config.json" in allow
    assert "vae/diffusion_pytorch_model.safetensors" in allow


def test_each_candidate_runs_at_a_shape_it_actually_supports():
    """Forcing ONIQ's 704x480x97 onto every model would measure the mismatch,
    not the model. CogVideoX in particular is trained at a fixed 720x480x49."""
    cog = modelprobe.PROBE_MODELS["cogvideox-i2v"]
    assert (cog["width"], cog["height"], cog["frames"]) == (720, 480, 49)
    wan = modelprobe.PROBE_MODELS["wan21-i2v-480p"]
    assert (wan["width"], wan["height"]) == (832, 480)
    for row in modelprobe.PROBE_MODELS.values():
        seconds = row["frames"] / row["fps"]
        assert 3.0 <= seconds <= 6.5, row["label"]


def test_spec_refuses_an_unknown_or_unevaluated_key():
    with pytest.raises(modelprobe.ProbeStop) as unknown:
        modelprobe.spec("veo")
    assert unknown.value.failure == "MODEL_ERROR"
    with pytest.raises(modelprobe.ProbeStop) as hunyuan:
        modelprobe.spec("hunyuanvideo-1.5-i2v")
    assert "NOT_EVALUATED" in hunyuan.value.detail


# ------------------------------------------------------------------ disk gate


def test_a_candidate_larger_than_the_disk_is_refused_before_downloading(rig, monkeypatch):
    """Running out of disk 60 GiB into a 118 GiB fetch burns the whole
    watchdog window and produces no evidence about the model at all."""
    monkeypatch.setattr(modelprobe, "free_disk_bytes", lambda p="/tmp": 20 * 1024**3)
    with pytest.raises(modelprobe.ProbeStop) as stop:
        modelprobe.run(probe_job("wan22-i2v-a14b"), rig["ref"], rig["out"],
                       fetch=rig["fetch"], load_pipeline=lambda local: fake_pipe(), torch=FakeTorch())
    assert stop.value.failure == "LOAD_FAILED"
    assert "no download attempted" in stop.value.detail
    assert not os.path.exists(rig["out"])


def test_the_disk_gate_keeps_headroom_for_the_incoming_file():
    """A snapshot download briefly holds an arriving file beside what it has
    already written, so exactly-enough disk is not enough."""
    exact = int(modelprobe.PROBE_MODELS["cogvideox-i2v"]["download_gib"] * 1024**3)
    assert not modelprobe.fits_disk("cogvideox-i2v", exact)
    assert modelprobe.fits_disk("cogvideox-i2v", int(exact * 1.2))


# ------------------------------------------------------------------- failures


def test_an_oom_is_reported_as_vram_oom_not_as_a_model_error():
    """The finding the owner asked for: evidence about this card."""
    class OutOfMemoryError(RuntimeError):
        pass

    assert modelprobe.classify(OutOfMemoryError("CUDA out of memory")) == "VRAM_OOM"
    assert modelprobe.classify(RuntimeError("CUDA error: no kernel image")) == "CUDA_FAILURE"
    assert modelprobe.classify(ValueError("bad config")) == "MODEL_ERROR"


def test_a_driver_fault_is_never_collapsed_into_an_oom():
    """A kernel fault says nothing about whether the model fits, and calling
    it an OOM would turn a broken worker into a verdict on a model."""
    assert modelprobe.classify(RuntimeError("CUDA error: device-side assert")) != "VRAM_OOM"


def test_quality_fail_is_not_a_failure_this_module_can_emit():
    """This code cannot see frames. Quality is decided by looking."""
    assert "QUALITY_FAIL" not in modelprobe.FAILURES


def test_a_failure_at_each_stage_is_named_for_that_stage(rig, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("nope")

    monkeypatch.setattr(modelprobe, "_load_reference", boom)
    with pytest.raises(modelprobe.ProbeStop) as stop:
        modelprobe.run(probe_job(), rig["ref"], rig["out"],
                       fetch=rig["fetch"], load_pipeline=lambda local: fake_pipe(), torch=FakeTorch())
    assert stop.value.failure == "CONDITIONING_FAILURE"


def test_an_encoder_that_writes_nothing_is_an_artifact_failure(rig, monkeypatch):
    monkeypatch.setattr(modelprobe, "_encode", lambda f, p, fps: open(p, "wb").close())
    with pytest.raises(modelprobe.ProbeStop) as stop:
        modelprobe.run(probe_job(), rig["ref"], rig["out"],
                       fetch=rig["fetch"], load_pipeline=lambda local: fake_pipe(), torch=FakeTorch())
    assert stop.value.failure == "ARTIFACT_FAILURE"


# ------------------------------------------------------------- what it records


def test_a_successful_probe_records_every_column_the_owner_asked_for(rig):
    report = modelprobe.run(probe_job(), rig["ref"], rig["out"],
                            fetch=rig["fetch"], load_pipeline=lambda local: fake_pipe(), torch=FakeTorch())
    assert report["failure"] == "SUCCESS"
    for phase in ("model_load_ms", "conditioning_load_ms", "inference_ms",
                  "encode_ms", "total_wall_ms"):
        assert phase in report, phase
    for vram in ("vram_total_bytes", "vram_before_allocated_bytes",
                 "vram_before_reserved_bytes", "peak_allocated_bytes",
                 "peak_reserved_bytes"):
        assert vram in report, vram
    assert report["output_bytes"] > 0
    assert report["revision"] and report["licence"]


def test_the_download_is_timed_apart_from_the_model_load():
    """On a 14B candidate the fetch is expected to dominate. Folding it into
    model load would report the cost of the network as the cost of the model."""
    phases = modelprobe.Phases(clock=iter([0, 0, 10, 10, 12, 99]).__next__)
    phases.time("download", lambda: None)
    phases.time("model_load", lambda: None)
    assert phases.timings["download_ms"] == 10_000
    assert phases.timings["model_load_ms"] == 2_000


def test_a_phase_is_timed_even_when_it_raises():
    phases = modelprobe.Phases(clock=iter([0, 0, 5, 9]).__next__)
    with pytest.raises(RuntimeError):
        phases.time("inference", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    assert phases.timings["inference_ms"] == 5_000


def test_cost_is_never_computed_by_the_worker():
    """The worker does not know the live rate, and a rate remembered from an
    earlier run is not a price."""
    row = modelprobe.probe_report(
        "cogvideox-i2v", modelprobe.PROBE_MODELS["cogvideox-i2v"],
        modelprobe.Phases(), before={}, peaks={}, failure="SUCCESS",
    )
    assert not any("usd" in k or "cost" in k for k in row)


def test_vram_is_read_from_the_device_not_from_checkpoint_size(rig):
    torch = FakeTorch(peak=17 * 1024**3)
    report = modelprobe.run(probe_job(), rig["ref"], rig["out"],
                            fetch=rig["fetch"], load_pipeline=lambda local: fake_pipe(), torch=torch)
    assert report["peak_allocated_bytes"] == 17 * 1024**3
    assert torch.cuda.reset_calls == 1, "peaks must be reset before the probe"


def test_a_cpu_worker_reports_no_vram_rather_than_zero(rig):
    """Zero allocated bytes and no GPU at all are different findings."""
    report = modelprobe.run(probe_job(), rig["ref"], rig["out"],
                            fetch=rig["fetch"], load_pipeline=lambda local: fake_pipe(),
                            torch=FakeTorch(available=False))
    assert "peak_allocated_bytes" not in report
    assert "vram_total_bytes" not in report


def test_the_pipeline_is_called_at_the_candidates_own_shape(rig):
    pipe = fake_pipe()
    modelprobe.run(probe_job("cogvideox-i2v"), rig["ref"], rig["out"],
                   fetch=rig["fetch"], load_pipeline=lambda local: pipe, torch=FakeTorch())
    assert pipe.called_with["width"] == 720
    assert pipe.called_with["num_frames"] == 49


# -------------------------------------------------------------- the contract


def test_the_contract_admits_only_the_authorised_model_keys():
    for key in modelprobe.PROBE_MODELS:
        job = contract.validate_job(probe_job(key))
        assert job["model"] == key


def test_the_contract_refuses_a_repository_in_place_of_a_key():
    """A caller may name WHICH row of an authorised benchmark to run; it may
    never name what to download."""
    with pytest.raises(contract.ContractError):
        contract.validate_job(probe_job("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"))


def test_the_contract_refuses_hunyuan_by_name_with_its_reason():
    with pytest.raises(contract.ContractError, match="ARCHITECTURE_NOT_RESOLVED"):
        contract.validate_job(probe_job("hunyuanvideo-1.5-i2v"))


def test_model_probe_carries_no_extra_top_level_fields():
    job = dict(probe_job())
    job["gpu"] = "H100"
    with pytest.raises(contract.ContractError):
        contract.validate_job(job)


def test_production_ops_are_unchanged_by_the_new_op():
    """The probe is additive. Nothing a user can reach behaves differently."""
    assert "model_probe" in contract.ALLOWED_OPS
    video = contract.validate_job({
        "op": "video_generate", "input_key": "a.png", "output_key": "b.mp4",
        "params": {"prompt": "x"},
    })
    assert "model" not in video


# ------------------------------------------- the report actually gets returned


def test_every_field_the_probe_measures_survives_filter_output(rig):
    """filter_output drops anything unlisted, so a measurement missing from
    the whitelist is a measurement the worker takes, pays for, and throws
    away. This walks the real report against the real filter."""
    report = modelprobe.run(probe_job(), rig["ref"], rig["out"],
                            fetch=rig["fetch"], load_pipeline=lambda local: fake_pipe(), torch=FakeTorch())
    dropped = set(report) - set(contract.OUTPUT_WHITELIST)
    assert not dropped, f"these measurements would be discarded: {sorted(dropped)}"


def test_a_failed_probe_also_survives_the_filter():
    """A refusal carries its own evidence — which stage, and why."""
    report = modelprobe.probe_report(
        "wan22-i2v-a14b", modelprobe.PROBE_MODELS["wan22-i2v-a14b"],
        modelprobe.Phases(), before={}, peaks={}, failure="VRAM_OOM",
        detail="CUDA out of memory",
    )
    dropped = set(report) - set(contract.OUTPUT_WHITELIST)
    assert not dropped, sorted(dropped)
    assert report["failure"] == "VRAM_OOM"
    assert report["detail"]


def test_the_worker_reports_the_disk_it_actually_has(rig, monkeypatch):
    """The template asks for 200 GB; the owner's instruction is not to trust
    that nominal figure. This is read from the running container."""
    monkeypatch.setattr(modelprobe, "free_disk_bytes", lambda p="/tmp": 150 * 1024**3)
    monkeypatch.setattr(modelprobe, "total_disk_bytes", lambda p="/tmp": 200 * 1024**3)
    report = modelprobe.run(probe_job(), rig["ref"], rig["out"],
                            fetch=rig["fetch"], load_pipeline=lambda local: fake_pipe(), torch=FakeTorch())
    assert report["disk_free_bytes"] == 150 * 1024**3
    assert report["disk_total_bytes"] == 200 * 1024**3


def test_the_probe_proves_which_card_it_ran_on(rig):
    """The same proof production demands: a CPU fallback is not success and
    the wrong card is not the benchmark. A result measured on some other GPU
    would be worse than none, because it would look like an answer."""
    report = modelprobe.run(probe_job(), rig["ref"], rig["out"],
                            fetch=rig["fetch"], load_pipeline=lambda local: fake_pipe(), torch=FakeTorch())
    assert report["device"] == "cuda"
    assert report["gpu_name"] == "NVIDIA RTX A5000"
    assert report["vram_peak_mb"] > 0
    assert report["vram_total_mb"] > 0


def test_a_cpu_run_reports_cpu_so_the_harness_can_refuse_it(rig):
    report = modelprobe.run(probe_job(), rig["ref"], rig["out"],
                            fetch=rig["fetch"], load_pipeline=lambda local: fake_pipe(),
                            torch=FakeTorch(available=False))
    assert report["device"] == "cpu"
    assert "gpu_name" not in report


# ------------------------------------------- the download, timed and bounded


def test_the_download_has_a_budget_smaller_than_the_job_ceiling():
    """A fetch allowed to run to the job ceiling leaves nothing for the load
    and the generation — it would spend the whole rental and still have no
    clip, which is the most expensive way to learn nothing."""
    assert modelprobe.DOWNLOAD_BUDGET_SECONDS < contract.PROBE_RUNTIME_CEILING_SECONDS
    assert modelprobe.DOWNLOAD_BUDGET_SECONDS > 0


def test_the_probe_ceiling_is_separate_from_productions():
    """Owner directive: do not alter production. A benchmark that downloads
    118 GiB at job time needs a different window from a job whose checkpoint
    is already baked into the image, and it must not move production's."""
    assert contract.RUNTIME_CEILING_SECONDS == 900
    assert contract.PROBE_RUNTIME_CEILING_SECONDS > contract.RUNTIME_CEILING_SECONDS


def test_a_fetch_that_finishes_in_time_returns_its_directory():
    calls = []
    result = modelprobe.fetch_within_budget(
        lambda: "/local/weights",
        lambda: 0,
        60,
        clock=lambda: 0.0,
        sleeper=calls.append,
        spawn=lambda target: target(),
    )
    assert result == "/local/weights"
    assert calls == []


def test_a_fetch_error_reaches_the_caller_rather_than_being_swallowed():
    with pytest.raises(RuntimeError):
        modelprobe.fetch_within_budget(
            _raiser(RuntimeError("connection reset")),
            lambda: 0,
            60,
            clock=lambda: 0.0,
            sleeper=lambda s: None,
            spawn=lambda target: target(),
        )


def _raiser(exc):
    def fn():
        raise exc
    return fn


def test_an_overrunning_fetch_stops_with_the_rate_it_actually_achieved():
    """The finding is the measurement. A candidate that cannot be fetched
    here is a real result about this worker, and only a result if the rate
    is measured rather than guessed."""
    ticks = iter([0.0, 100.0, 900.0, 900.0, 900.0])
    with pytest.raises(modelprobe.ProbeStop) as stop:
        modelprobe.fetch_within_budget(
            lambda: "never",
            lambda: 45 * 1024**3,
            900,
            clock=lambda: next(ticks),
            sleeper=lambda s: None,
            spawn=lambda target: None,   # never runs: the fetch is still going
        )
    assert stop.value.failure == "DOWNLOAD_TIMEOUT"
    assert "45.00GiB" in stop.value.detail
    assert "MiB/s" in stop.value.detail
    assert "no GPU time was spent" in stop.value.detail


def test_download_timeout_is_a_named_failure_not_a_quality_verdict():
    assert "DOWNLOAD_TIMEOUT" in modelprobe.FAILURES
    assert "QUALITY_FAIL" not in modelprobe.FAILURES


def test_the_download_is_timed_apart_from_the_model_load_on_a_real_run(rig):
    """Owner directive: do not hide cold-start cost. Fetching 117 GiB and
    building a pipeline out of it are two costs with two different fixes, and
    one combined number would report the network as the model."""
    report = modelprobe.run(probe_job(), rig["ref"], rig["out"],
                            fetch=rig["fetch"],
                            load_pipeline=lambda local: fake_pipe(),
                            torch=FakeTorch())
    assert "download_ms" in report
    assert "model_load_ms" in report
    assert report["download_bytes"] == 7 * 1024**3


def test_the_loader_is_handed_the_directory_the_fetch_returned(rig):
    """The two phases are separate but not independent: the pipeline must be
    built from the snapshot that was just measured, never re-resolved."""
    seen = {}

    def loader(local):
        seen["local"] = local
        return fake_pipe()

    modelprobe.run(probe_job(), rig["ref"], rig["out"],
                   fetch=lambda: "/snapshot/here",
                   load_pipeline=loader, torch=FakeTorch())
    assert seen["local"] == "/snapshot/here"


def test_a_download_timeout_survives_the_output_filter():
    """The measurement is the deliverable. A timeout that reached the harness
    stripped of its numbers would be indistinguishable from a crash."""
    kept = contract.filter_output({
        "ok": False, "code": "DOWNLOAD_TIMEOUT",
        "download_bytes": 45 * 1024**3, "download_ms": 900_000,
        "disk_free_bytes": 1, "disk_total_bytes": 2,
    })
    assert kept["download_bytes"] == 45 * 1024**3
    assert kept["download_ms"] == 900_000


# --------------------------------------- sampling settings, cited not chosen


def test_every_sampling_setting_names_where_it_came_from():
    """A step count with no citation is a preference wearing a number's
    clothes. Read on 2026-08-29 by validation/probe_settings from each
    publisher's card at the PINNED revision."""
    for key, row in modelprobe.PROBE_MODELS.items():
        if row.get("steps") is not None or row.get("guidance") is not None:
            assert row.get("sampling_source"), key
            assert "card" in row["sampling_source"], key


def test_a_candidate_whose_card_states_no_step_count_gets_no_step_count():
    """Wan2.1's card sets guidance_scale=5.0 and states no step count. Filling
    that in from the neighbouring row would make the comparison a comparison
    of my invention."""
    wan21 = modelprobe.PROBE_MODELS["wan21-i2v-480p"]
    assert "steps" not in wan21
    assert wan21["guidance"] == 5.0


def test_the_sampling_knobs_passed_are_only_the_cited_ones():
    assert modelprobe._sampling(modelprobe.PROBE_MODELS["wan22-i2v-a14b"]) == {
        "num_inference_steps": 40, "guidance_scale": 3.5,
    }
    assert modelprobe._sampling(modelprobe.PROBE_MODELS["wan21-i2v-480p"]) == {
        "guidance_scale": 5.0,
    }
    assert modelprobe._sampling(modelprobe.PROBE_MODELS["ltx-13b"]) == {
        "num_inference_steps": 30,
    }
    assert modelprobe._sampling({}) == {}


def test_the_report_says_which_sampling_was_used(rig):
    """Two candidates compared at different step counts is a legitimate
    benchmark only if the report says so on every row."""
    report = modelprobe.run(probe_job("wan21-i2v-480p"), rig["ref"], rig["out"],
                            fetch=rig["fetch"],
                            load_pipeline=lambda local: fake_pipe(),
                            torch=FakeTorch())
    assert report["steps"] == "PIPELINE_DEFAULT"
    assert report["guidance"] == 5.0
    kept = contract.filter_output({"ok": True, **report})
    assert kept["steps"] == "PIPELINE_DEFAULT"
    assert kept["sampling_source"] == report["sampling_source"]


def test_the_cited_knobs_reach_the_pipeline(rig):
    pipe = fake_pipe()
    modelprobe.run(probe_job("cogvideox-i2v"), rig["ref"], rig["out"],
                   fetch=rig["fetch"], load_pipeline=lambda local: pipe,
                   torch=FakeTorch())
    assert pipe.called_with["num_inference_steps"] == 50
    assert pipe.called_with["guidance_scale"] == 6.0


# ------------------------------------------------- the batched pipeline output


def test_a_batched_pipeline_output_is_unwrapped_to_one_clip():
    """Every diffusers video pipeline answers with frames BATCHED — a list of
    clips. Encoding the outer list writes a file nothing can play, and it does
    it without raising."""
    class Output:
        frames = [["f0", "f1", "f2"]]

    assert modelprobe._frames_of(Output()) == ["f0", "f1", "f2"]


def test_a_plain_list_of_frames_passes_through():
    assert modelprobe._frames_of(["f0", "f1"]) == ["f0", "f1"]


def test_a_pipeline_that_returned_nothing_is_a_generation_failure(rig):
    with pytest.raises(modelprobe.ProbeStop) as stop:
        modelprobe.run(probe_job(), rig["ref"], rig["out"],
                       fetch=rig["fetch"],
                       load_pipeline=lambda local: fake_pipe(frames=[]),
                       torch=FakeTorch())
    assert stop.value.failure == "GENERATION_FAILURE"


# ----------------------------- offload, decided by arithmetic not by taste


def test_a_candidate_whose_transformer_exceeds_the_card_gets_sequential_offload():
    """enable_model_cpu_offload() keeps ONE pipeline component resident, so
    the peak is the transformer: params x 2 bytes in bf16. An OOM caused by
    choosing model-level offload for a 26 GiB transformer would be a fact
    about the configuration wearing the costume of a fact about the model."""
    A5000_BYTES = 24 * 1000 ** 3  # the card's marketed 24GB
    approx_params = {
        "cogvideox-i2v": 5e9,
        "ltx-13b": 13e9,
        "wan21-i2v-480p": 14e9,
        "wan22-i2v-a14b": 14e9,
    }
    for key, params in approx_params.items():
        resident = params * 2  # bf16
        row = modelprobe.PROBE_MODELS[key]
        if resident >= A5000_BYTES * 0.85:
            assert row["offload"] == "sequential", key
        else:
            assert row["offload"] == "model", key


def test_no_candidate_runs_without_an_offload_strategy():
    for key, row in modelprobe.PROBE_MODELS.items():
        assert row["offload"] in {"model", "sequential", "none"}, key


def test_the_offload_choice_is_the_servers_and_no_caller_can_change_it():
    """A caller names a benchmark ROW; every knob behind it is fixed here."""
    import contract

    job = {"op": "model_probe", "model": "wan21-i2v-480p",
           "input_key": "out/ref.png", "output_key": "out/x.mp4",
           "offload": "model", "params": {"prompt": "turn"}}
    with pytest.raises(contract.ContractError):
        contract.validate_job(job)
