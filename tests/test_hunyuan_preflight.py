"""The free gate, and the frame count it derives.

Owner directive 2026-08-30 section 10: the frame count must come from "the
exact legal frame-count requirements of the selected Hunyuan I2V
implementation/checkpoint", not from a README. Section 12: zero GPU jobs.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation import hunyuan_preflight as hp  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_legal_counts_follow_the_vae_rule_not_a_readme():
    # A causal video VAE with temporal compression R encodes frame 0 alone
    # and the rest in blocks of R: legal counts are R*k+1.
    for row in hp.legal_frame_counts(4, 24):
        assert (row["frames"] - 1) % 4 == 0
    for row in hp.legal_frame_counts(8, 24):
        assert (row["frames"] - 1) % 8 == 0


def test_the_chosen_count_is_the_shortest_useful_one():
    chosen = hp.choose_frames(hp.legal_frame_counts(4, 24))
    assert chosen["frames"] == 49
    assert chosen["seconds"] >= hp.MIN_USEFUL_SECONDS


def test_counts_below_the_useful_floor_are_excluded():
    # 5 frames is legal for ratio 4 and cannot show a person slowly
    # turning. Legality is necessary, not sufficient.
    frames = [r["frames"] for r in hp.legal_frame_counts(4, 24)]
    assert 5 not in frames and 9 not in frames


def test_counts_above_the_baseline_window_are_excluded():
    # Going markedly longer than the ~4.04s LTX baseline would compare a
    # harder job against an easier one, in Hunyuan's disfavour.
    for row in hp.legal_frame_counts(4, 24):
        assert row["seconds"] <= hp.MAX_USEFUL_SECONDS


def test_an_unreadable_ratio_refuses_rather_than_assuming_four():
    for bad in (None, "4", 0, -1):
        with pytest.raises(hp.PreflightFailure) as exc:
            hp.legal_frame_counts(bad, 24)
        assert exc.value.gate == "frame-rule-unreadable"


def test_gates_fail_when_the_endpoint_has_no_volume():
    facts = {
        "vae/config.json": {"temporal_compression_ratio": 4,
                            "spatial_compression_ratio": 16},
        "transformer/config.json": {"task_type": "i2v"},
        "model_index.json": {"_class_name": "HunyuanVideo15ImageToVideoPipeline",
                             "vae": [], "text_encoder": [], "text_encoder_2": [],
                             "scheduler": []},
        "revision": "a" * 40,
    }
    rows, chosen, _ = hp.gates(facts, {"gpuTypeIds": ["NVIDIA RTX A5000"],
                                       "networkVolumeId": ""})
    failed = [name for name, ok, _ in rows if not ok]
    assert "network volume attached" in failed
    assert chosen["frames"] == 49


def test_gates_pass_on_a_fully_prepared_endpoint():
    facts = {
        "vae/config.json": {"temporal_compression_ratio": 4,
                            "spatial_compression_ratio": 16},
        "transformer/config.json": {"task_type": "i2v"},
        "model_index.json": {"_class_name": "HunyuanVideo15ImageToVideoPipeline",
                             "vae": [], "text_encoder": [], "text_encoder_2": [],
                             "scheduler": []},
        "revision": "a" * 40,
    }
    rows, _, _ = hp.gates(facts, {"gpuTypeIds": ["NVIDIA RTX A5000"],
                                  "networkVolumeId": "vol-123"})
    assert [name for name, ok, _ in rows if not ok] == []


def test_a_non_a5000_endpoint_fails_the_gate():
    facts = {
        "vae/config.json": {"temporal_compression_ratio": 4,
                            "spatial_compression_ratio": 16},
        "transformer/config.json": {"task_type": "i2v"},
        "model_index.json": {"_class_name": "HunyuanVideo15ImageToVideoPipeline",
                             "vae": [], "text_encoder": [], "text_encoder_2": [],
                             "scheduler": []},
        "revision": "a" * 40,
    }
    rows, _, _ = hp.gates(facts, {"gpuTypeIds": ["NVIDIA RTX A5000", "NVIDIA A100"],
                                  "networkVolumeId": "vol-123"})
    assert "A5000 is the only GPU on the endpoint" in [
        name for name, ok, _ in rows if not ok
    ]


def test_the_module_submits_no_job():
    with open(os.path.join(ROOT, "validation", "hunyuan_preflight.py"),
              encoding="utf-8") as fh:
        source = fh.read()
    for forbidden in ("submit_job", "purge_queue", "cancel_job",
                      "retarget_template", "set_template_env",
                      "create_template", "set_execution_timeout"):
        assert forbidden not in source, forbidden
