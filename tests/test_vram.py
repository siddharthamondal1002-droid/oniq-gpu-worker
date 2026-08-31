import pytest

from validation import vram
from validation.vram import Arch, Config, Shape

ONIQ = Shape(width=704, height=480, frames=97)

# Stand-in numbers, not any real model: these tests are about the arithmetic.
LTXish = Arch(hidden=2048, layers=28, vae_spatial=32, vae_temporal=8, vae_channels=128)
WANish = Arch(hidden=5120, layers=40, patch_spatial=2, vae_spatial=8, vae_temporal=4,
              vae_channels=96)


# ------------------------------------------------------------------- geometry


def test_latent_grid_applies_vae_compression_then_dit_patching():
    lf, lh, lw = vram.latent_grid(Shape(704, 480, 97), WANish)
    assert lf == 1 + 96 // 4  # 25 latent frames
    assert lh == (480 // 8) // 2  # 30
    assert lw == (704 // 8) // 2  # 44


def test_single_frame_never_collapses_to_zero_latent_frames():
    lf, _, _ = vram.latent_grid(Shape(704, 480, 1), WANish)
    assert lf == 1


def test_tokens_grow_with_frames_and_area_not_with_parameters():
    """Cost 2 is a function of the SHAPE. This is why a small model at 720p
    can want more working memory than a big one at 480p."""
    small = vram.tokens(Shape(704, 480, 97), WANish)
    wider = vram.tokens(Shape(1280, 720, 97), WANish)
    longer = vram.tokens(Shape(704, 480, 193), WANish)
    assert wider > small and longer > small


# ------------------------------------------------------------------- weights


def test_fp8_halves_the_transformer_and_leaves_the_rest_alone():
    roles = {"transformer": 1000, "text_encoder": 400, "vae": 200}
    out = vram.weight_bytes(roles, precision="fp8", shipped="bf16")
    assert out == {"transformer": 500, "text_encoder": 400, "vae": 200}


def test_fp8_also_halves_a_second_expert():
    roles = {"transformer": 1000, "transformer_2": 1000}
    out = vram.weight_bytes(roles, precision="fp8", shipped="bf16")
    assert out == {"transformer": 500, "transformer_2": 500}


def test_bf16_to_bf16_is_the_measured_bytes_unchanged():
    roles = {"transformer": 27_000, "vae": 900}
    assert vram.weight_bytes(roles, precision="bf16") == roles


# ------------------------------------------------------------------ residency


def test_no_offload_holds_everything_at_once():
    roles = {"transformer": 100, "transformer_2": 100, "vae": 10}
    assert vram.resident_bytes(roles, "none") == (210, "every component resident at once")


def test_model_offload_on_a_mixture_of_experts_holds_one_expert():
    """The Wan2.2 case. The experts run at different denoising stages, so
    one-at-a-time is the honest floor — and it is the difference between
    fitting on 24 GB and not."""
    roles = {"transformer": 100, "transformer_2": 100, "text_encoder": 40}
    resident, why = vram.resident_bytes(roles, "model")
    assert resident == 100
    assert "largest single component" in why


def test_sequential_offload_streams_the_transformer():
    roles = {"transformer": 800, "text_encoder": 400}
    resident, why = vram.resident_bytes(roles, "sequential")
    assert resident == 100
    assert "streamed" in why


def test_unknown_offload_strategy_is_refused_not_guessed():
    with pytest.raises(ValueError):
        vram.resident_bytes({"transformer": 1}, "magic")


def test_no_weights_measured_is_zero_and_says_so():
    assert vram.resident_bytes({}, "none") == (0, "no weights measured")


# ---------------------------------------------------------------------- plan


def test_peak_is_the_worse_stage_never_the_sum():
    """Denoise frees its working set before the VAE runs. Summing them would
    invent a ceiling no real run ever hits, and would wrongly rule models
    out."""
    plan = vram.plan(
        roles={"transformer": 2 * vram.GIB},
        arch=LTXish,
        shape=ONIQ,
        config=Config(),
    )
    assert plan["peak_bytes"] == max(plan["denoise_peak_bytes"], plan["decode_peak_bytes"])
    assert plan["peak_bytes"] < plan["denoise_peak_bytes"] + plan["decode_peak_bytes"]


def test_plan_names_which_stage_binds():
    plan = vram.plan(
        roles={"transformer": vram.GIB}, arch=LTXish, shape=ONIQ, config=Config()
    )
    assert plan["binding_stage"] in ("denoise", "decode")


def test_vae_tiling_lowers_the_decode_peak():
    """A model that only OOMs with whole-clip decode is deployable; reporting
    just the naive figure would wrongly condemn it."""
    whole = vram.plan(roles={}, arch=LTXish, shape=ONIQ, config=Config())
    tiled = vram.plan(
        roles={}, arch=LTXish, shape=ONIQ, config=Config(vae_tile_frames=16)
    )
    assert tiled["projected_decode_bytes"] < whole["projected_decode_bytes"]


def test_measured_and_projected_are_reported_separately():
    plan = vram.plan(
        roles={"transformer": 123}, arch=LTXish, shape=ONIQ, config=Config()
    )
    assert plan["measured_weight_bytes"] == 123
    assert "projected_activation_bytes" in plan
    assert "projected_decode_bytes" in plan


def test_fp8_plan_still_computes_activations_in_bf16():
    """fp8 checkpoints compute in bf16; pretending activations shrink too
    would under-report the peak, which is the dangerous direction."""
    bf16 = vram.plan(roles={}, arch=WANish, shape=ONIQ, config=Config(precision="bf16"))
    fp8 = vram.plan(roles={}, arch=WANish, shape=ONIQ, config=Config(precision="fp8"))
    assert fp8["projected_activation_bytes"] == bf16["projected_activation_bytes"]


# ------------------------------------------------------------------ hardware


def test_fits_leaves_headroom_for_the_cuda_context():
    """23.9 GiB does not fit on a 24 GB card. A plan that says it does OOMs."""
    assert not vram.fits(int(23.9 * vram.GIB), 24)
    assert vram.fits(int(20 * vram.GIB), 24)


def test_minimum_gpu_picks_the_smallest_card_that_works():
    # Each figure is chosen to clear the usable ceiling of one card and bust
    # the one below it: 24GB yields 22.32 GiB, 32GB yields 29.76, 40GB 37.2,
    # 48GB 44.64. Nominal capacity is never the number that decides.
    assert vram.minimum_gpu(int(20 * vram.GIB)) == "RTX A5000 24GB"
    assert vram.minimum_gpu(int(25 * vram.GIB)) == "RTX 5090 32GB"
    assert vram.minimum_gpu(int(35 * vram.GIB)) == "A100 40GB"
    assert vram.minimum_gpu(int(42 * vram.GIB)) == "A6000 48GB"


def test_minimum_gpu_never_reaches_for_an_a100_that_is_not_needed():
    """The brief's explicit warning: do not pay for an A100/H100 merely
    because a model is labelled 14B."""
    chosen = vram.minimum_gpu(int(28 * vram.GIB))
    assert chosen == "RTX 5090 32GB"
    assert "A100" not in chosen and "H100" not in chosen


def test_minimum_gpu_returns_none_when_nothing_listed_is_big_enough():
    assert vram.minimum_gpu(int(200 * vram.GIB)) is None


def test_a40_class_ordering_prefers_the_earlier_listed_card_on_a_tie():
    """A5000 and 4090 are both 24 GB; the owner listed the A5000 first, and
    it is the card ONIQ already runs."""
    assert vram.minimum_gpu(int(15 * vram.GIB)) == "RTX A5000 24GB"


# ------------------------------------------------------------------- anchor


LTX_MEASURED = vram.Arch(hidden=2048, layers=28, vae_spatial=32, vae_temporal=8,
                         vae_channels=128)


def test_the_projection_reproduces_the_one_real_measurement():
    """LTX-Video 2B, bf16, all resident, tiling on, 9 frames at 704x480, on an
    A5000: 13,837 MB observed. Arithmetic that cannot reproduce the single case
    ONIQ has actually run has no business ranking eight it has not."""
    check = vram.anchor_check(LTX_MEASURED)
    assert 0.95 <= check["ratio"] <= 1.05, check


def test_tiled_decode_is_bounded_by_area_as_well_as_by_frames():
    """Capping only the frame count overestimated ONIQ's own production
    configuration more than tenfold: enable_tiling() splits SPATIALLY, so the
    biggest feature map is a tile however large the canvas."""
    small = vram.vae_decode_bytes(Shape(704, 480, 97), LTX_MEASURED, tile_frames=16)
    huge = vram.vae_decode_bytes(Shape(1920, 1080, 97), LTX_MEASURED, tile_frames=16)
    assert small == huge, "a bigger canvas must not enlarge a tiled decode"


def test_untiled_decode_still_grows_with_the_canvas():
    """The OOM case has to remain visible, or a model gets called deployable
    on a configuration nobody enabled."""
    small = vram.vae_decode_bytes(Shape(704, 480, 97), LTX_MEASURED)
    huge = vram.vae_decode_bytes(Shape(1920, 1080, 97), LTX_MEASURED)
    assert huge > small * 4


def test_tiling_is_never_worse_than_not_tiling():
    whole = vram.vae_decode_bytes(Shape(704, 480, 97), LTX_MEASURED)
    tiled = vram.vae_decode_bytes(Shape(704, 480, 97), LTX_MEASURED, tile_frames=16)
    assert tiled < whole
