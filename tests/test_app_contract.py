"""THE PAYLOADS THE APP ACTUALLY SENDS, THROUGH THE REAL VALIDATOR.

Every other contract test here starts from what `contract.py` says it accepts.
This one starts from the other end: the exact shapes ONIQ's edge functions put
on the wire, run through `validate_job` unmodified. A contract can be
self-consistent and still be the wrong contract, and only the caller's real
payload can tell the two apart.

WHY THE SHAPES ARE PINNED AS LITERALS RATHER THAN READ FROM THE APP REPO.
`characterRefAuthorization.test.ts` tried reading `../oniq-gpu-worker/contract.py`
across the repo boundary on 2026-08-31. It passed locally and failed CI with
ENOENT, because the two repos are only ever checked out together on a
developer's disk. So each side pins the other's shape and a comment names the
counterpart file; drift shows up as a failure here rather than as a 400 from a
paid GPU job.

The counterparts, as of app main dd72955:

    supabase/functions/_shared/inHouseMotion.ts
        stillKeyFor  -> story/still/<jobId>-<sceneId>-<shotId>.png
        unitKey      -> <projectId>/<sceneId>/<shotId>/v<version>/<index>
        outputKeyFor -> story/clip/<unitKey>/ltx-001.mp4
        buildClipPayload -> op/input_key/output_key + params
                            {prompt, watermark, seed, negative_prompt}
    supabase/functions/_shared/oniqImage.ts
        image_generate params
                            {prompt, seed, negative_prompt,
                             reference_key, reference_strength}
    supabase/functions/_shared/characterRef.ts
        characterRefKey -> story/ref/canon/<characterId>/v<n>.png
        MIN/MAX_REFERENCE_STRENGTH = 0.05 / 0.95

Note what the app's ids may contain: `assertSafeId` is [A-Za-z0-9_-], and the
job id is a Supabase uuid, so UNDERSCORES and HYPHENS both reach these keys.
That is why the cases below use them rather than tidy alphanumerics.
"""

import pytest

import contract

# One realistic identity, in the app's own shapes.
JOB = "550e8400-e29b-41d4-a716-446655440000"
SCENE = "scene_1"
SHOT = "shot_01"
PROJECT = "proj_abc-123"
STILL_KEY = f"story/still/{JOB}-{SCENE}-{SHOT}.png"
CLIP_KEY = f"story/clip/{PROJECT}/{SCENE}/{SHOT}/v1/0/ltx-001.mp4"
REF_KEY = "story/ref/canon/83193e07-2b3c-465b-8479-7a3f03d73464/v1.png"


def _motion_job(**params):
    """story-motion's payload, via _shared/oniqMotion.ts."""
    base = {"prompt": "a slow push in on her face", "watermark": True}
    base.update(params)
    return {
        "op": "video_generate",
        "input_key": STILL_KEY,
        "output_key": CLIP_KEY,
        "params": base,
    }


def _still_job(**params):
    """story-still's payload, via _shared/oniqImage.ts."""
    base = {"prompt": "a woman at a market stall"}
    base.update(params)
    return {"op": "image_generate", "output_key": STILL_KEY, "params": base}


class TestTheKeysTheAppBuilds:
    def test_the_still_key_is_accepted(self):
        assert contract.validate_job(_motion_job())["input_key"] == STILL_KEY

    def test_the_clip_key_is_accepted(self):
        assert contract.validate_job(_motion_job())["output_key"] == CLIP_KEY

    def test_underscores_and_hyphens_survive(self):
        # assertSafeId admits both; a key charset that refused either would
        # reject real jobs only once a paid one was dispatched.
        assert "_" in CLIP_KEY and "-" in CLIP_KEY
        contract.validate_job(_motion_job())

    def test_every_clip_index_of_a_shot_is_a_legal_key(self):
        for index in range(0, 8):
            job = _motion_job()
            job["output_key"] = (
                f"story/clip/{PROJECT}/{SCENE}/{SHOT}/v1/{index}/ltx-001.mp4"
            )
            contract.validate_job(job)

    def test_a_version_bump_is_a_legal_key(self):
        for version in (1, 2, 17, 999):
            job = _motion_job()
            job["output_key"] = (
                f"story/clip/{PROJECT}/{SCENE}/{SHOT}/v{version}/0/ltx-001.mp4"
            )
            contract.validate_job(job)


class TestTheMotionParams:
    def test_prompt_and_watermark_alone_are_enough(self):
        contract.validate_job(_motion_job())

    def test_seed_and_negative_prompt_are_accepted(self):
        # The app holds these camelCase and renames them on the wire; this is
        # the snake_case end of that rename.
        out = contract.validate_job(
            _motion_job(seed=6890573110812446, negative_prompt="blurry, low quality")
        )
        assert out["params"]["seed"] == 6890573110812446

    def test_the_apps_widest_seed_is_accepted(self):
        # deriveSeed returns `% Number.MAX_SAFE_INTEGER`, so 2**53-1 bounds
        # every seed the app can emit. The worker's own bound is uint64, which
        # is wider — the app being the stricter end is what keeps a seed from
        # losing precision in JSON transit.
        contract.validate_job(_motion_job(seed=2**53 - 1))

    def test_a_reference_is_refused_on_video(self):
        # Identity conditioning is an image_generate capability. A video job
        # that carried one would be silently ignored by the sampler, which is
        # worse than a refusal.
        with pytest.raises(contract.ContractError):
            contract.validate_job(_motion_job(reference_key=REF_KEY))

    def test_an_absent_watermark_still_watermarks(self):
        # videogen reads params.get("watermark", True): the fail-safe default
        # is marked, so a dropped field cannot silently ship a clean clip.
        job = _motion_job()
        del job["params"]["watermark"]
        contract.validate_job(job)


class TestTheReferenceContract:
    def test_the_apps_canonical_key_is_accepted(self):
        contract.validate_job(_still_job(reference_key=REF_KEY, reference_strength=0.5))

    @pytest.mark.parametrize("strength", [0.05, 0.5, 0.95])
    def test_the_usable_band_is_accepted(self, strength):
        contract.validate_job(
            _still_job(reference_key=REF_KEY, reference_strength=strength)
        )

    @pytest.mark.parametrize("strength", [0.0, 0.04, 0.951, 1.0, -0.5])
    def test_outside_the_band_is_refused_not_clamped(self, strength):
        # Both ends refuse. A clamp would silently give a caller something
        # other than what they asked for.
        with pytest.raises(contract.ContractError):
            contract.validate_job(
                _still_job(reference_key=REF_KEY, reference_strength=strength)
            )

    @pytest.mark.parametrize(
        "key",
        [
            "story/ref/canon/abc/v0.png",  # v0 is not a version
            "story/ref/canon/abc/v1.jpg",  # references are png
            "story/ref/other/abc/v1.png",  # only the canon scope exists
            "story/ref/canon/../abc/v1.png",  # traversal
            "story/ref/canon/abc/v1.png/../../x",  # traversal, suffixed
            "https://oniqhub.com/x.png",  # a URL is not an identity
            "//evil.example/x.png",
            "story/still/anything.png",  # a still is not a reference
            "",
        ],
    )
    def test_only_a_canonical_identity_is_accepted(self, key):
        # THE SECURITY PROPERTY: the browser names a characterId, never a path.
        # Everything that is not a published canonical reference resolves to
        # nothing, so no client-chosen path can reach the sampler.
        with pytest.raises(contract.ContractError):
            contract.validate_job(_still_job(reference_key=key))


class TestWhatComesBack:
    def test_the_result_carries_no_secret_and_no_url(self):
        # filter_output is the checkable form of "no credentials leave the
        # worker". Whitelist, not blacklist, so a new field is dropped by
        # default rather than shipped by accident.
        leaked = {
            "ok": True,
            "op": "video_generate",
            "output_key": CLIP_KEY,
            "r2_secret_access_key": "SHOULD-NEVER-APPEAR",
            "presigned_url": "https://example.invalid/x?sig=abc",
            "authorization": "Bearer y",
        }
        out = contract.filter_output(leaked)
        assert set(out) == {"ok", "op", "output_key"}

    def test_the_prompt_text_never_echoes_back(self):
        # Only its LENGTH is whitelisted. A worker that echoed prompts would
        # put user text into every log that records a job result.
        assert "negative_prompt" not in contract.OUTPUT_WHITELIST
        assert "negative_prompt_chars" in contract.OUTPUT_WHITELIST

    def test_the_fields_the_app_reads_back_all_survive(self):
        # _shared/oniqMotion.ts verifies the artifact against output_bytes and
        # settles the ledger on the worker's measured time. If any of these
        # were dropped the app would fail every clip it correctly generated.
        for field in ("ok", "output_key", "output_bytes", "video_seconds", "frames", "fps"):
            assert field in contract.OUTPUT_WHITELIST, field
