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
    return {"ref": str(ref), "out": str(out)}


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
                       load_pipeline=lambda: fake_pipe(), torch=FakeTorch())
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
                       load_pipeline=lambda: fake_pipe(), torch=FakeTorch())
    assert stop.value.failure == "CONDITIONING_FAILURE"


def test_an_encoder_that_writes_nothing_is_an_artifact_failure(rig, monkeypatch):
    monkeypatch.setattr(modelprobe, "_encode", lambda f, p, fps: open(p, "wb").close())
    with pytest.raises(modelprobe.ProbeStop) as stop:
        modelprobe.run(probe_job(), rig["ref"], rig["out"],
                       load_pipeline=lambda: fake_pipe(), torch=FakeTorch())
    assert stop.value.failure == "ARTIFACT_FAILURE"


# ------------------------------------------------------------- what it records


def test_a_successful_probe_records_every_column_the_owner_asked_for(rig):
    report = modelprobe.run(probe_job(), rig["ref"], rig["out"],
                            load_pipeline=lambda: fake_pipe(), torch=FakeTorch())
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
                            load_pipeline=lambda: fake_pipe(), torch=torch)
    assert report["peak_allocated_bytes"] == 17 * 1024**3
    assert torch.cuda.reset_calls == 1, "peaks must be reset before the probe"


def test_a_cpu_worker_reports_no_vram_rather_than_zero(rig):
    """Zero allocated bytes and no GPU at all are different findings."""
    report = modelprobe.run(probe_job(), rig["ref"], rig["out"],
                            load_pipeline=lambda: fake_pipe(),
                            torch=FakeTorch(available=False))
    assert "peak_allocated_bytes" not in report
    assert "vram_total_bytes" not in report


def test_the_pipeline_is_called_at_the_candidates_own_shape(rig):
    pipe = fake_pipe()
    modelprobe.run(probe_job("cogvideox-i2v"), rig["ref"], rig["out"],
                   load_pipeline=lambda: pipe, torch=FakeTorch())
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
                            load_pipeline=lambda: fake_pipe(), torch=FakeTorch())
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
                            load_pipeline=lambda: fake_pipe(), torch=FakeTorch())
    assert report["disk_free_bytes"] == 150 * 1024**3
    assert report["disk_total_bytes"] == 200 * 1024**3


def test_the_probe_proves_which_card_it_ran_on(rig):
    """The same proof production demands: a CPU fallback is not success and
    the wrong card is not the benchmark. A result measured on some other GPU
    would be worse than none, because it would look like an answer."""
    report = modelprobe.run(probe_job(), rig["ref"], rig["out"],
                            load_pipeline=lambda: fake_pipe(), torch=FakeTorch())
    assert report["device"] == "cuda"
    assert report["gpu_name"] == "NVIDIA RTX A5000"
    assert report["vram_peak_mb"] > 0
    assert report["vram_total_mb"] > 0


def test_a_cpu_run_reports_cpu_so_the_harness_can_refuse_it(rig):
    report = modelprobe.run(probe_job(), rig["ref"], rig["out"],
                            load_pipeline=lambda: fake_pipe(),
                            torch=FakeTorch(available=False))
    assert report["device"] == "cpu"
    assert "gpu_name" not in report
