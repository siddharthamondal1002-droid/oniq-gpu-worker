"""The Qwen licence gate, checked against metadata without a build."""

import pytest

from validation import qwen_assets


DOCKERFILE = open("Dockerfile", encoding="utf-8").read()

GATE = {
    "candidates": ["Qwen/Qwen3-8B-AWQ", "Qwen/Qwen3-8B"],
    "allowed_licences": {"apache-2.0"},
    "size_guard_bytes": 20 * 1024**3,
    "needed": ("config.json", "tokenizer_config.json"),
}


def _info(licence="apache-2.0", *, gib=16.4, files=("config.json", "tokenizer_config.json"), tag=False):
    siblings = [{"rfilename": f, "size": 1} for f in files]
    siblings.append({"rfilename": "model.safetensors", "size": int(gib * 1024**3)})
    doc = {"siblings": siblings}
    if tag:
        doc["tags"] = [f"license:{licence}"] if licence else []
    else:
        doc["cardData"] = {"license": licence} if licence else {}
    return doc


# ----------------------------------------------------------- parsing


def test_the_gate_is_read_from_the_real_dockerfile():
    gate = qwen_assets.parse_gate(DOCKERFILE)
    assert gate["candidates"] == ["Qwen/Qwen3-8B-AWQ", "Qwen/Qwen3-8B"]
    assert gate["allowed_licences"] == {"apache-2.0"}
    assert gate["needed"] == ("config.json", "tokenizer_config.json")


def test_it_reads_the_story_block_not_the_ltx_one():
    # Both bakes declare CANDIDATES and SIZE_GUARD_BYTES. Matching the
    # first one would check Lightricks repos against the Qwen licence
    # gate and report a confident, meaningless answer.
    gate = qwen_assets.parse_gate(DOCKERFILE)
    assert not any("Lightricks" in c for c in gate["candidates"])


def test_a_dockerfile_that_stops_declaring_the_gate_raises():
    with pytest.raises(qwen_assets.GateParseError):
        qwen_assets.parse_gate("FROM python:3.11-slim\nRUN echo hi\n")


# ------------------------------------------------------------ verdicts


def test_a_non_apache_licence_is_refused_not_skipped():
    row = qwen_assets.survey("x", GATE, lambda r: _info("llama3"))
    assert row["verdict"] == "REFUSE"


def test_a_missing_licence_is_refused():
    row = qwen_assets.survey("x", GATE, lambda r: _info(None))
    assert row["verdict"] == "REFUSE"


def test_the_licence_tag_is_read_when_card_data_has_none():
    row = qwen_assets.survey("x", GATE, lambda r: _info("apache-2.0", tag=True))
    assert row["verdict"] == "BAKE"


def test_a_checkpoint_missing_its_config_is_skipped():
    row = qwen_assets.survey("x", GATE, lambda r: _info(files=("config.json",)))
    assert row["verdict"] == "SKIP"


def test_a_larger_model_class_fails_the_size_guard():
    row = qwen_assets.survey("x", GATE, lambda r: _info(gib=64))
    assert row["verdict"] == "SKIP"


def test_a_good_candidate_bakes():
    row = qwen_assets.survey("x", GATE, lambda r: _info())
    assert row["verdict"] == "BAKE"


def test_an_unreachable_registry_is_never_a_pass():
    def boom(repo):
        raise OSError("no route")

    row = qwen_assets.survey("x", GATE, boom)
    assert row["verdict"] == "UNREACHABLE"


# ------------------------------------------------------------- report


def test_a_refused_licence_fails_the_report_even_if_a_later_one_would_bake():
    def fetch(repo):
        return _info("llama3") if repo.endswith("AWQ") else _info()

    code, rows = qwen_assets.report(DOCKERFILE, fetch)
    assert code == 1
    assert rows[0]["verdict"] == "REFUSE"
    assert rows[1]["verdict"] == "BAKE"


def test_no_candidate_baking_fails_the_report():
    code, _ = qwen_assets.report(DOCKERFILE, lambda r: _info(gib=64))
    assert code == 1


def test_an_unreachable_registry_fails_the_report():
    def boom(repo):
        raise OSError("no route")

    code, _ = qwen_assets.report(DOCKERFILE, boom)
    assert code == 1


def test_the_report_passes_when_a_candidate_would_bake():
    code, _ = qwen_assets.report(DOCKERFILE, lambda r: _info())
    assert code == 0
