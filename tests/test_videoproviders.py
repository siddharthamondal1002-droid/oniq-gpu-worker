import pytest

from validation import videoproviders as vp
from validation.videoproviders import ActionContract, Reference

TURN = ActionContract(
    slug="maya-turns",
    subject="Maya",
    start_state="standing on the platform beside the girl",
    action="slowly turns her head and upper body toward the distant railway",
    end_state="facing toward the railway",
    camera_action="gentle push-in",
    environment_action="the girl remains beside her",
    required_motion="head_and_body_rotation",
    reference="a",
)
PLATE_A = Reference(ref_id="a", key="plate-a.png",
                    describes="An abandoned railway platform in the rain.")
PLATE_B = Reference(ref_id="b", key="plate-b.png", describes="A black train.")


def ltx():
    return vp.LTXProvider(vae_temporal=8, vae_spatial=32)


def wan(cls=vp.Wan21I2VProvider):
    return cls(vae_temporal=4, vae_spatial=8, patch_spatial=2)


def cog():
    return vp.CogVideoXI2VProvider(vae_temporal=4, vae_spatial=8)


# ------------------------------------------------------------------ the seam


def test_all_five_adapters_the_owner_named_are_present():
    keys = {p.key for p in vp.PROVIDERS}
    assert len(vp.PROVIDERS) == 5
    assert vp.for_key("ltx") is vp.LTXProvider
    assert vp.for_key("wan21-i2v") is vp.Wan21I2VProvider
    assert vp.for_key("wan22-i2v-a14b") is vp.Wan22I2VA14BProvider
    assert vp.for_key("hunyuan-1.5-i2v") is vp.HunyuanVideo15Provider
    assert vp.for_key("cogvideox-i2v") is vp.CogVideoXI2VProvider
    assert len(keys) == 5, "provider keys must be distinct"


def test_unknown_provider_is_refused_rather_than_defaulted():
    with pytest.raises(KeyError):
        vp.for_key("veo")


def test_the_contract_carries_no_model_specific_syntax():
    """If the shared contract contained one model's prompt dialect, every
    other adapter would inherit it and the comparison would be rigged."""
    text = " ".join(
        [TURN.subject, TURN.action, TURN.start_state, TURN.end_state,
         TURN.camera_action, TURN.environment_action]
    ).lower()
    for dialect in ("cinematic realism", "photographic realism", "4k", "masterpiece"):
        assert dialect not in text


def test_each_family_renders_its_own_dialect_from_the_same_contract():
    """The point of the layer. If every adapter emitted the same text, the
    benchmark would be measuring tolerance of one model's dialect and
    reporting it as quality.

    Four dialects across five adapters, not five: Wan2.2 deliberately shares
    Wan2.1's caption style because it shares its text distribution. What
    separates those two is the machine — two experts and a handover boundary
    — not the words, and inventing a spurious difference in the prompt would
    confound exactly the comparison the owner asked for."""
    prompts = {
        p.key: p(vae_temporal=4, vae_spatial=8).generate_video(TURN, PLATE_A).prompt
        for p in vp.PROVIDERS
    }
    assert len(set(prompts.values())) == 4
    assert prompts["wan22-i2v-a14b"] == prompts["wan21-i2v"]
    others = [v for k, v in prompts.items() if not k.startswith("wan2")]
    assert len(set(others)) == 3


def test_every_adapter_states_the_action_from_the_contract():
    """Different dialects, same instruction — otherwise they are not being
    asked for the same shot."""
    for provider in vp.PROVIDERS:
        request = provider(vae_temporal=4, vae_spatial=8).generate_video(TURN, PLATE_A)
        assert "turns her head" in request.prompt
        assert TURN.subject.lower() in request.prompt.lower()


# ---------------------------------------------------------- conditioning fence


def test_a_shot_cannot_be_conditioned_on_the_wrong_reference():
    """The failure the multi-reference architecture exists to prevent: a
    Maya shot silently conditioned on the train plate."""
    with pytest.raises(ValueError, match="reference"):
        ltx().generate_video(TURN, PLATE_B)


def test_the_request_carries_the_reference_key_it_was_given():
    assert ltx().generate_video(TURN, PLATE_A).reference_key == "plate-a.png"


def test_wan_declares_the_clip_vision_path_ltx_does_not_have():
    """Wan conditions through an image encoder as well as the VAE, which is
    why the registry measures an image_encoder role for it and not for LTX.
    A benchmark that ignored this would misattribute Wan's identity
    retention to its prompt."""
    assert "CLIP" in wan().generate_video(TURN, PLATE_A).conditioning
    assert "CLIP" not in ltx().generate_video(TURN, PLATE_A).conditioning


# -------------------------------------------------------------- legal shapes


def test_frames_are_snapped_to_each_models_own_vae_period():
    """1 + k*temporal, derived from the checkpoint — never a magic number."""
    assert (ltx().snap_frames(4.0) - 1) % 8 == 0
    assert (wan().snap_frames(4.0) - 1) % 4 == 0


def test_duration_is_held_constant_not_frame_count():
    """LTX is 24fps, Wan 16, CogVideoX 8. Asking all three for 97 frames
    would ask for 4.0, 6.1 and 12.1 seconds and then compare them."""
    requests = [ltx().generate_video(TURN, PLATE_A),
                wan().generate_video(TURN, PLATE_A),
                cog().generate_video(TURN, PLATE_A)]
    assert len({r.num_frames for r in requests}) > 1, "frame counts differ"
    for r in requests:
        assert abs(r.seconds - vp.TARGET_SECONDS) < 0.6, (
            f"{r.provider} got {r.seconds}s, not ~{vp.TARGET_SECONDS}s"
        )


def test_ltx_at_four_seconds_is_oniqs_proven_97_frames():
    """Sanity anchor: the incumbent's benchmark clip should come out as the
    shape already proven on this endpoint."""
    assert ltx().generate_video(TURN, PLATE_A).num_frames == 97


def test_canvas_snaps_down_never_up():
    """Rounding up would raise the token count and the VRAM peak, turning
    the authorised benchmark into a bigger, costlier one."""
    provider = vp.LTXProvider(vae_temporal=8, vae_spatial=32, width=704, height=480)
    assert provider.width <= 704 and provider.height <= 480
    assert provider.width % 32 == 0 and provider.height % 32 == 0


def test_patching_tightens_the_size_multiple():
    """Wan patches 2x on top of an 8x VAE, so its canvas must be divisible
    by 16, not 8. Using the VAE ratio alone would emit an illegal size."""
    assert wan().size_multiple == 16
    assert ltx().size_multiple == 32


def test_an_odd_canvas_is_snapped_into_legality():
    provider = vp.Wan21I2VProvider(
        vae_temporal=4, vae_spatial=8, patch_spatial=2, width=700, height=477
    )
    assert (provider.width, provider.height) == (688, 464)


def test_a_single_frame_request_stays_legal():
    assert ltx().snap_frames(0.01) == 9


# ------------------------------------------------------------------ directives


def test_no_adapter_emits_a_negative_prompt():
    """Owner directive 2026-08-29: positive descriptions only."""
    for provider in vp.PROVIDERS:
        request = provider(vae_temporal=4, vae_spatial=8).generate_video(TURN, PLATE_A)
        assert not any("negative" in k.lower() for k in request.extra)


def test_the_seam_refuses_a_negative_prompt_smuggled_through_extra():
    """The realistic way one would come back: copied from a model card."""

    class Sneaky(vp.LTXProvider):
        def native_extra(self):
            return {"negative_prompt": "blurry, deformed"}

    with pytest.raises(ValueError, match="negative prompting"):
        Sneaky(vae_temporal=8, vae_spatial=32).generate_video(TURN, PLATE_A)


def test_an_empty_prompt_is_refused_rather_than_sent():
    class Silent(vp.LTXProvider):
        def render_prompt(self, contract, reference):
            return "   "

    with pytest.raises(ValueError, match="empty prompt"):
        Silent(vae_temporal=8, vae_spatial=32).generate_video(TURN, PLATE_A)


def test_the_base_class_has_no_dialect_of_its_own():
    with pytest.raises(NotImplementedError):
        vp.VideoProvider(vae_temporal=4, vae_spatial=8).generate_video(TURN, PLATE_A)


def test_wan22_is_a_separate_adapter_carrying_its_expert_boundary():
    """The brief: treat Wan2.2 as a separate candidate, and do not assume
    A14B is Wan2.1 14B. The handover point between its two experts is a real
    generation parameter, so it travels rather than defaulting."""
    request = wan(vp.Wan22I2VA14BProvider).generate_video(TURN, PLATE_A)
    assert request.extra["experts"] == 2
    assert 0 < request.extra["boundary_ratio"] <= 1
    assert request.model_repo != vp.Wan21I2VProvider.model_repo


def test_declared_facts_are_flagged_for_verification_on_the_probe():
    """fps and conditioning are not stated by any config field, so they are
    declared here — and must be confirmed against the real pipeline before a
    comparison rests on them."""
    for provider in vp.PROVIDERS:
        assert "native_fps" in provider.verify_on_probe
        assert "conditioning" in provider.verify_on_probe
    assert "boundary_ratio" in vp.Wan22I2VA14BProvider.verify_on_probe
