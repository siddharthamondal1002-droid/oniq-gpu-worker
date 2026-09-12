import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_SCRUB = (
    "R2_S3_ENDPOINT",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "ONIQ_ALLOW_CPU_FALLBACK",
    "ONIQ_STORY_PROVIDER",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_API_URL",
    "RUNPOD_API_KEY",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ambient credentials or fallback flags must never flip a test."""
    for name in _SCRUB:
        monkeypatch.delenv(name, raising=False)


def write_fake_checkpoint(root, *, distilled=False, upscaler=False, components=None):
    """Materialise the minimum LTX snapshot ltxcaps.inspect_checkpoint reads.

    A real directory rather than a monkeypatched return value: the point of
    ltxcaps is that inference settings come from files the bake produced, so a
    test that stubbed the reading would prove nothing about the reading.
    """
    import json
    import os

    root = str(root)
    comps = ("transformer", "vae", "text_encoder", "tokenizer", "scheduler") \
        if components is None else tuple(components)
    os.makedirs(root, exist_ok=True)
    for c in comps:
        os.makedirs(os.path.join(root, c), exist_ok=True)
    if upscaler:
        os.makedirs(os.path.join(root, "latent_upsampler"), exist_ok=True)
    with open(os.path.join(root, "model_index.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {"_class_name": "LTXImageToVideoPipeline",
             **{c: ["diffusers", "Thing"] for c in comps}},
            fh,
        )
    if "scheduler" in comps:
        cfg = {"_class_name": "FlowMatchEulerDiscreteScheduler", "shift": 1.0}
        if distilled:
            # What a timestep-distilled checkpoint ships: its own schedule.
            cfg["timesteps"] = [1000, 875, 750, 625, 500, 375, 250, 125]
        with open(
            os.path.join(root, "scheduler", "scheduler_config.json"),
            "w",
            encoding="utf-8",
        ) as fh:
            json.dump(cfg, fh)
    return root


@pytest.fixture
def fake_checkpoint(tmp_path, monkeypatch):
    """A non-distilled baked checkpoint, wired into videogen."""
    import videogen

    root = write_fake_checkpoint(tmp_path / "ltx")
    mid = tmp_path / "MODEL_ID"
    mid.write_text("Lightricks/LTX-Video\n", encoding="utf-8")
    monkeypatch.setattr(videogen, "MODEL_DIR", root)
    monkeypatch.setattr(videogen, "MODEL_ID_FILE", str(mid))
    return root
