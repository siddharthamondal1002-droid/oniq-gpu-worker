import pytest

from validation import model_bench as mb
from validation import model_registry as mr
from validation import vram


# ------------------------------------------------------------ config reading


def test_arch_reads_hidden_directly_when_published():
    arch, notes = mb.arch_from_configs(
        {"hidden_size": 4096, "num_layers": 30},
        {
            "spatial_compression_ratio": 8,
            "temporal_compression_ratio": 4,
            "base_channels": 128,
        },
    )
    assert arch.hidden == 4096 and arch.layers == 30
    assert any("hidden=4096" in n for n in notes)


def test_arch_derives_hidden_from_heads_when_not_published_directly():
    """Wan and CogVideoX publish heads x head_dim rather than a hidden size."""
    arch, notes = mb.arch_from_configs(
        {"num_attention_heads": 40, "attention_head_dim": 128, "num_layers": 40},
        {
            "spatial_compression_ratio": 8,
            "temporal_compression_ratio": 4,
            "dim": 96,
        },
    )
    assert arch.hidden == 5120
    assert any("x" in n and "hidden=5120" in n for n in notes)


def test_arch_is_none_when_a_required_number_cannot_be_read():
    """The whole point: no defaults. A guessed architecture yields a confident
    wrong VRAM figure, and nobody probes a number that already looks fine."""
    arch, notes = mb.arch_from_configs({"num_layers": 30}, {})
    assert arch is None
    assert any("UNREADABLE" in n for n in notes)
    assert any("hidden" in n for n in notes if "UNREADABLE" in n)


def test_patch_size_list_is_read_as_temporal_then_spatial():
    """Wan writes patch_size as [t, h, w]; reading it as spatial would shrink
    the token count fourfold and understate every projection."""
    arch, _ = mb.arch_from_configs(
        {
            "num_attention_heads": 40,
            "attention_head_dim": 128,
            "num_layers": 40,
            "patch_size": [1, 2, 2],
        },
        {
            "spatial_compression_ratio": 8,
            "temporal_compression_ratio": 4,
            "dim": 96,
        },
    )
    assert arch.patch_spatial == 2
    assert arch.patch_temporal == 1


def test_missing_patch_size_defaults_to_identity():
    """Absent patching has one unambiguous meaning, unlike an absent hidden
    size — so this default is safe where the others are not."""
    arch, _ = mb.arch_from_configs(
        {"hidden_size": 2048, "num_layers": 28},
        {
            "spatial_compression_ratio": 32,
            "temporal_compression_ratio": 8,
            "base_channels": 128,
        },
    )
    assert arch.patch_spatial == 1 and arch.patch_temporal == 1


def test_block_out_channels_gives_the_full_resolution_width_not_the_widest():
    """Decoders run NARROW at full size and WIDE at low size, and because
    channels roughly double each time the area quarters, the full-size map is
    the one that dominates memory. Taking the maximum is what made the first
    matrix project 750 GiB of decode for a model that runs on one card."""
    arch, notes = mb.arch_from_configs(
        {"hidden_size": 2048, "num_layers": 28},
        {
            "spatial_compression_ratio": 8,
            "temporal_compression_ratio": 4,
            "block_out_channels": [128, 256, 512],
        },
    )
    assert arch.vae_channels == 128
    assert any("fullres" in n for n in notes)


def test_vae_ratios_are_derived_when_no_ratio_is_published():
    """Wan publishes neither compression ratio, but does publish its stage
    list and which stages are temporal — including the `temperal_downsample`
    spelling, which is matched as published rather than as intended."""
    arch, notes = mb.arch_from_configs(
        {"hidden_size": 2048, "num_layers": 28},
        {
            "block_out_channels": [96, 192, 384, 384],
            "temperal_downsample": [False, True, True],
        },
    )
    assert arch.vae_spatial == 8, "three halvings after the first stage"
    assert arch.vae_temporal == 4, "two temporal halvings"
    assert any("derived" in n for n in notes)


def test_booleans_are_never_mistaken_for_dimensions():
    """`True` is an int in Python. Reading a flag as a hidden size would
    produce a VRAM figure of essentially zero."""
    assert mb._as_int(True) is None
    arch, _ = mb.arch_from_configs({"hidden_size": True, "num_layers": 4}, {})
    assert arch is None


def test_pick_config_paths_finds_the_two_that_matter():
    paths = [
        "scheduler/scheduler_config.json",
        "transformer/config.json",
        "vae/config.json",
        "text_encoder/config.json",
    ]
    assert mb.pick_config_paths(paths) == ("transformer/config.json", "vae/config.json")


def test_pick_config_paths_tolerates_a_repo_with_neither():
    assert mb.pick_config_paths(["README.md"]) == (None, None)


# ----------------------------------------------------------------- projection


def _row(roles, arch, *, dtypes=None):
    cand = next(c for c in mr.CANDIDATES if c.key == "ltx-2b")
    row = mr.Row(candidate=cand, repo="Lightricks/LTX-Video")
    row.measurement = {"roles": roles, "revision": "x"}
    row.configs = {
        "arch": arch,
        "notes": [],
        "effective_roles": roles,
        "dtypes": dtypes or {},
    }
    return row


ARCH = vram.Arch(hidden=2048, layers=28, vae_spatial=32, vae_temporal=8,
                 vae_channels=128)


def test_project_returns_one_plan_per_configuration():
    plans = mb.project(_row({"transformer": 4 * vram.GIB}, ARCH))
    assert len(plans) == len(mb.CONFIGS)
    assert {p["label"] for p in plans} == {label for label, _ in mb.CONFIGS}


def test_project_is_empty_without_an_architecture():
    assert mb.project(_row({"transformer": 1}, None)) == []


def test_project_is_empty_without_measured_weights():
    assert mb.project(_row({}, ARCH)) == []


def test_offloading_never_raises_the_peak():
    """Each configuration in the ladder should be at least as deployable as
    the one before it, or the ladder is describing something else."""
    plans = {p["label"]: p for p in mb.project(_row(
        {"transformer": 26 * vram.GIB, "text_encoder": 9 * vram.GIB}, ARCH
    ))}
    assert (plans["bf16 + model offload"]["peak_bytes"]
            <= plans["as published"]["peak_bytes"])
    assert (plans["fp8 + offload + tiled VAE"]["peak_bytes"]
            <= plans["bf16 + offload + tiled VAE"]["peak_bytes"])


def test_every_plan_carries_an_a5000_verdict_and_a_minimum_card():
    for plan in mb.project(_row({"transformer": 2 * vram.GIB}, ARCH)):
        assert isinstance(plan["fits_a5000"], bool)
        assert "minimum_gpu" in plan


def test_a_14b_class_model_is_not_excluded_before_the_ladder_is_tried():
    """The brief forbids ruling 14B out just because the card is an A5000.
    A model too big as published can still fit once offloaded, and the
    projection has to be able to say so."""
    big = _row({"transformer": 27 * vram.GIB, "text_encoder": 9 * vram.GIB}, ARCH)
    plans = mb.project(big)
    assert not plans[0]["fits_a5000"], "as-published should not fit 24GB"
    assert any(p["fits_a5000"] for p in plans), (
        "the offload ladder must be able to rescue it"
    )


# --------------------------------------------------------------------- report


def test_report_refuses_without_a_credential():
    code, rows = mb.report(None)
    assert code == 2 and rows == []


def test_report_survives_a_publisher_whose_listing_fails(capsys):
    """One unreachable org must not take the whole benchmark down — the other
    eight rows are still evidence."""

    def get(url, token, timeout=60):
        if "author=Wan-AI" in url:
            raise OSError("network")
        if "api/models?" in url:
            return []
        return {}

    code, rows = mb.report("tok", get=get)
    assert code == 0
    assert len(rows) == len(mr.CANDIDATES)
    assert "listing failed" in capsys.readouterr().out


def test_report_names_unmatched_candidates_rather_than_inventing_one(capsys):
    def get(url, token, timeout=60):
        return [] if "api/models?" in url else {}

    mb.report("tok", get=get)
    assert "NOT-PUBLISHED" in capsys.readouterr().out


# ------------------------------------------------- resolving what is loaded


def _measured(key, **kw):
    cand = next(c for c in mr.CANDIDATES if c.key == key)
    row = mr.Row(candidate=cand, repo="r")
    row.measurement = kw
    return row


def test_a_named_variant_is_chosen_out_of_several_complete_checkpoints():
    """HunyuanVideo-1.5 ships eleven checkpoints under transformer/, one per
    resolution and task. Summing them describes a machine nobody will build."""
    row = _measured(
        "hunyuanvideo-1.5-i2v",
        roles={"vae": 5},
        role_variants={"transformer": {"480p_i2v": 31, "720p_sr_distilled": 31,
                                       "480p_t2v": 31}},
    )
    roles, notes = mb.effective_roles(row)
    assert roles["transformer"] == 31, "one checkpoint, not the sum of three"
    assert any("chosen from 3" in n for n in notes)


def test_unnamed_variants_are_reported_ambiguous_rather_than_summed():
    row = _measured(
        "ltx-2b",
        roles={"vae": 5},
        role_variants={"transformer": {"a": 31, "b": 31}},
    )
    roles, notes = mb.effective_roles(row)
    assert roles == {}
    assert any("AMBIGUOUS" in n for n in notes)


def test_a_root_level_checkpoint_file_replaces_the_component_directory():
    """Three of the owner's four LTX rows are files inside one repository."""
    row = _measured(
        "ltx-13b-fp8",
        roles={"transformer": 7, "text_encoder": 17, "vae": 1},
        single_files=[{"name": "ltxv-13b-0.9.8-dev-fp8.safetensors", "bytes": 99}],
    )
    roles, notes = mb.effective_roles(row)
    assert roles["transformer"] == 99, "the named file, not the 2B directory"
    assert roles["text_encoder"] == 17, "shared components still come from the repo"
    assert any("root-level file" in n for n in notes)


def test_a_missing_named_file_is_a_finding_not_a_silent_fallback():
    row = _measured("ltx-13b", roles={"transformer": 7}, single_files=[])
    roles, notes = mb.effective_roles(row)
    assert roles == {}
    assert any("MISSING" in n for n in notes)


def test_nested_pipeline_copies_are_reported_as_excluded():
    """LTX-Video-0.9.8-13B-distilled ships a second copy of itself under vae/.
    A prefix match read that as a 44 GiB VAE."""
    row = _measured("ltx-2b", roles={"vae": 1}, nested_bytes=44 * vram.GIB)
    _, notes = mb.effective_roles(row)
    assert any("nested pipeline" in n for n in notes)


def test_an_fp32_checkpoint_halves_when_loaded_bf16():
    """Wan ships fp32 transformers. Assuming bf16 on disk overstates a bf16
    load by exactly two, which is the difference between one card and two."""
    plans = mb.project(_row(
        {"transformer": 60 * vram.GIB}, ARCH, dtypes={"transformer": "fp32"}
    ))
    as_published = next(p for p in plans if p["label"] == "as published")
    assert as_published["resident_weight_bytes"] == 30 * vram.GIB


def test_ltx_spatial_ratio_includes_the_vaes_own_patchify():
    """LTX compresses 32x spatially: three halvings AND a 4x patch inside the
    VAE. Deriving from the stage list alone gives 8 and understates the token
    count — and therefore the working set — fourfold."""
    arch, notes = mb.arch_from_configs(
        {"hidden_size": 2048, "num_layers": 28},
        {
            "block_out_channels": [128, 256, 512, 512],
            "spatio_temporal_scaling": [True, True, True, False],
            "patch_size": 4,
        },
    )
    assert arch.vae_spatial == 32
    assert arch.vae_temporal == 8
    assert any("patch 4" in n for n in notes)


def test_config_path_follows_the_chosen_variant():
    """Reading the first transformer config alphabetically would describe a
    1080p super-resolution model while the measured bytes are the 480p I2V
    one."""
    paths = [
        "transformer/1080p_sr_distilled/config.json",
        "transformer/480p_i2v/config.json",
        "vae/config.json",
    ]
    tpath, _ = mb.pick_config_paths(paths, "480p_i2v")
    assert tpath == "transformer/480p_i2v/config.json"


def test_config_path_prefers_the_component_root_when_no_variant_is_named():
    paths = ["transformer/a/config.json", "transformer/config.json"]
    assert mb.pick_config_paths(paths)[0] == "transformer/config.json"
