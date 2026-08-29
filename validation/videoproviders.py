"""VideoProvider — one action contract, five native input formats.

Owner brief 2026-08-29: "Each adapter converts ONIQ's common action contract
into the model's native input format. The application must not know
model-specific prompt syntax."

So the contract is the currency and the adapter is the exchange. ONIQ says
what must HAPPEN — subject, action, start and end state, what the camera is
allowed to contribute — and each provider decides how to ask its own model
for it. Nothing above this layer ever writes a prompt, and no prompt written
here ever leaks upward.

WHY THE PROMPT CANNOT SIMPLY BE REUSED ACROSS MODELS. It reads like the
cheap option and it would quietly invalidate the benchmark. These models
were trained on different caption distributions: LTX on long descriptive
paragraphs, Wan on shorter action-led text, CogVideoX on caption-style
sentences. Feeding all five the paragraph LTX likes measures which model
best tolerates LTX's prompt style, and then reports that as quality. The
brief's "implement the closest native equivalent rather than weakening the
test" is the instruction to avoid exactly that.

WHAT IS DERIVED AND WHAT IS DECLARED — the same split vram.py keeps:

  DERIVED   frame-count and resolution legality, computed from the model's
            OWN measured config. Causal video VAEs accept 1 + k*temporal
            frames and sizes divisible by (vae_spatial * patch); both
            numbers were read out of the checkpoint, so the rule travels
            with the model instead of being remembered per family.
  DECLARED  native fps, and the conditioning mechanism. These are model
            facts that no config field states outright. Each is marked
            `verify_on_probe` and must be confirmed against the real
            pipeline before any of them is trusted in a comparison.

DURATION, NOT FRAME COUNT, IS WHAT IS HELD CONSTANT. LTX is a 24fps model,
Wan 16fps, CogVideoX 8fps; asking all three for 97 frames asks for 4.0, 6.1
and 12.1 seconds of footage and then compares them. Every adapter is handed
a target in SECONDS and snaps to the nearest count its own VAE permits.

NO NEGATIVE PROMPTS. Owner directive 2026-08-29: positive descriptions only.
The base class refuses to emit one, so an adapter cannot reintroduce it by
copying a model card.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The benchmark's fixed target. ONIQ's proven production clip is 97 frames at
# 24fps; every model is asked for this many SECONDS, not this many frames.
TARGET_SECONDS = 4.0


@dataclass(frozen=True)
class ActionContract:
    """What must happen in a shot. Model-agnostic by construction.

    Mirrors the `contract` block the five-shot battery already carries, and
    is deliberately free of any word that belongs to one model's prompt
    dialect. This is what the footage is JUDGED against; what the model is
    ASKED is whatever its adapter renders below.
    """

    slug: str
    subject: str
    start_state: str
    action: str
    end_state: str
    camera_action: str
    environment_action: str
    required_motion: str
    reference: str  # which controlled reference conditions this shot

    def as_clause(self) -> str:
        """The action as one plain clause. The shared root of every prompt."""
        return f"{self.subject} {self.action}"


@dataclass(frozen=True)
class Reference:
    """The image a shot is conditioned on.

    A reference id, never a path: the same fence the production route keeps,
    where a caller can name WHICH reference but never WHERE it lives.
    """

    ref_id: str
    key: str  # server-derived storage key
    describes: str  # what is actually in it, for the adapter's prompt


@dataclass(frozen=True)
class NativeRequest:
    """One provider's native inputs. What actually reaches the pipeline."""

    provider: str
    model_repo: str
    prompt: str
    width: int
    height: int
    num_frames: int
    fps: int
    reference_key: str
    conditioning: str
    extra: dict = field(default_factory=dict)

    @property
    def seconds(self) -> float:
        return round(self.num_frames / self.fps, 2)


class VideoProvider:
    """The seam. `generate_video` is the only entry point above this line.

    Subclasses supply a native prompt dialect and the model's own identity;
    everything about legality — frame counts, dimensions — is derived from
    the measured architecture rather than restated per family.
    """

    key = "abstract"
    label = "abstract"
    model_repo = ""
    native_fps = 24  # DECLARED — verify_on_probe
    conditioning = "single-image"  # DECLARED — verify_on_probe
    verify_on_probe = ("native_fps", "conditioning")

    def __init__(self, *, vae_temporal: int, vae_spatial: int,
                 patch_spatial: int = 1, width: int = 704, height: int = 480):
        self.vae_temporal = vae_temporal
        self.vae_spatial = vae_spatial
        self.patch_spatial = patch_spatial
        self.width, self.height = self.snap_size(width, height)

    # ------------------------------------------------------------ legality

    @property
    def size_multiple(self) -> int:
        """Both compressions apply: the VAE's, then the DiT's patching."""
        return self.vae_spatial * self.patch_spatial

    def snap_size(self, width: int, height: int) -> tuple[int, int]:
        """Nearest legal canvas at or below the request.

        Rounds DOWN. Rounding up would silently enlarge the canvas, which
        raises both the token count and the VRAM peak — turning a benchmark
        into a different, more expensive benchmark than the one authorised.
        """
        m = self.size_multiple
        return (max(m, (width // m) * m), max(m, (height // m) * m))

    def snap_frames(self, seconds: float = TARGET_SECONDS) -> int:
        """Nearest legal frame count to `seconds` of footage at native fps.

        Causal video VAEs encode a leading keyframe plus groups of
        `vae_temporal`, so legal counts are 1 + k*temporal. Derived from the
        checkpoint's own config, never from a number remembered per family.
        """
        wanted = max(1, round(seconds * self.native_fps))
        k = max(1, round((wanted - 1) / self.vae_temporal))
        return int(k * self.vae_temporal + 1)

    # ------------------------------------------------------------- dialect

    def render_prompt(self, contract: ActionContract,
                      reference: Reference) -> str:
        raise NotImplementedError

    # --------------------------------------------------------------- entry

    def generate_video(self, contract: ActionContract, reference: Reference,
                       *, seconds: float = TARGET_SECONDS) -> NativeRequest:
        """ONIQ's one call. Returns the model's native request.

        The name is the owner's (`VideoProvider.generateVideo()`); the body
        is deliberately a REQUEST BUILDER rather than an invocation, because
        the invocation needs the checkpoint resident on a GPU and this must
        stay testable — and comparable across five models — for nothing.
        """
        if reference.ref_id != contract.reference:
            raise ValueError(
                f"{contract.slug} asks for reference {contract.reference!r} "
                f"but was handed {reference.ref_id!r}"
            )
        prompt = self.render_prompt(contract, reference)
        if not prompt.strip():
            raise ValueError(f"{self.key} rendered an empty prompt")
        request = NativeRequest(
            provider=self.key,
            model_repo=self.model_repo,
            prompt=prompt,
            width=self.width,
            height=self.height,
            num_frames=self.snap_frames(seconds),
            fps=self.native_fps,
            reference_key=reference.key,
            conditioning=self.conditioning,
            extra=self.native_extra(),
        )
        # Owner directive 2026-08-29: positive descriptions only. The real
        # risk is an adapter copying a model card's recommended negative
        # prompt into `extra`, so that is what is checked — scanning the
        # positive text for words like "no" would only ever misfire on
        # "notices" and "north".
        banned = [k for k in request.extra if "negative" in k.lower()]
        if banned:
            raise ValueError(f"negative prompting is not permitted: {banned}")
        return request

    def native_extra(self) -> dict:
        return {}


class LTXProvider(VideoProvider):
    """LTX-Video. Long, descriptive, present-tense paragraphs.

    LTX's captions are dense scene descriptions, and its own documentation
    asks for detail; a terse action line under-specifies it. This is the
    incumbent, so its dialect is the one already proven on this endpoint.
    """

    key = "ltx"
    label = "LTX-Video"
    model_repo = "Lightricks/LTX-Video"
    native_fps = 24
    conditioning = "single-image (VAE-encoded first frame)"

    def render_prompt(self, contract, reference) -> str:
        return (
            f"{reference.describes} {contract.subject.capitalize()} begins "
            f"{contract.start_state} and {contract.action}, ending "
            f"{contract.end_state}. {contract.environment_action.capitalize()}. "
            f"Camera: {contract.camera_action}. Faces, clothing, body "
            f"proportions and the surrounding scene stay consistent "
            f"throughout. Cinematic, photographic realism."
        )


class Wan21I2VProvider(VideoProvider):
    """Wan2.1 I2V-14B. Action-led, shorter, subject first.

    Wan conditions on the reference through a CLIP vision encoder as well as
    the VAE latent — the `image_encoder` component the registry measures — so
    the image carries more of the identity than it does for LTX, and the text
    is left to carry the MOVEMENT.
    """

    key = "wan21-i2v"
    label = "Wan2.1 I2V-14B"
    model_repo = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    native_fps = 16
    conditioning = "single-image (VAE latent + CLIP vision embedding)"

    def render_prompt(self, contract, reference) -> str:
        return (
            f"{contract.as_clause()}, ending {contract.end_state}. "
            f"{contract.environment_action.capitalize()}. "
            f"Camera {contract.camera_action}. Consistent identity and "
            f"stable scene."
        )


class Wan22I2VA14BProvider(Wan21I2VProvider):
    """Wan2.2 I2V-A14B. Same dialect as 2.1, different machine underneath.

    A separate candidate by the owner's instruction, and separate in fact:
    it is a two-expert mixture that switches between a high-noise and a
    low-noise transformer partway through denoising. That boundary is a real
    generation parameter, so it travels in `extra` rather than being left to
    a default — and it is why this class exists at all instead of an alias.
    """

    key = "wan22-i2v-a14b"
    label = "Wan2.2 I2V-A14B"
    model_repo = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    # DECLARED: the denoising fraction at which the high-noise expert hands
    # over to the low-noise one. Read from the checkpoint's own config on the
    # probe — a wrong boundary silently degrades every clip this model makes.
    boundary_ratio = 0.9
    verify_on_probe = ("native_fps", "conditioning", "boundary_ratio")

    def native_extra(self) -> dict:
        return {"boundary_ratio": self.boundary_ratio, "experts": 2}


class HunyuanVideo15Provider(VideoProvider):
    """HunyuanVideo-1.5 I2V. Structured, comma-separated scene grammar."""

    key = "hunyuan-1.5-i2v"
    label = "HunyuanVideo-1.5 I2V"
    model_repo = "tencent/HunyuanVideo-1.5"
    native_fps = 24
    conditioning = "single-image"

    def render_prompt(self, contract, reference) -> str:
        return (
            f"{contract.subject}, {contract.action}, from "
            f"{contract.start_state} to {contract.end_state}, "
            f"{contract.environment_action}, camera {contract.camera_action}, "
            f"consistent identity, stable scene, cinematic realism"
        )


class CogVideoXI2VProvider(VideoProvider):
    """CogVideoX-5B-I2V. One flowing caption sentence.

    Trained on long single-sentence captions, and notably intolerant of
    prompt shapes far from that — the lower-resource comparison row.
    """

    key = "cogvideox-i2v"
    label = "CogVideoX-5B-I2V"
    model_repo = "THUDM/CogVideoX-5b-I2V"
    native_fps = 8
    conditioning = "single-image"

    def render_prompt(self, contract, reference) -> str:
        return (
            f"{contract.subject.capitalize()} {contract.action}, beginning "
            f"{contract.start_state} and ending {contract.end_state}, while "
            f"{contract.environment_action}, with the camera "
            f"{contract.camera_action}, in a consistent and stable scene."
        )


PROVIDERS: tuple[type[VideoProvider], ...] = (
    LTXProvider,
    Wan21I2VProvider,
    Wan22I2VA14BProvider,
    HunyuanVideo15Provider,
    CogVideoXI2VProvider,
)


def for_key(key: str) -> type[VideoProvider]:
    for provider in PROVIDERS:
        if provider.key == key:
            return provider
    raise KeyError(f"no provider named {key!r}")
