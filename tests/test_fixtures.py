"""The first real LTX evidence is fixtures now, and fixtures must not rot.

The three clips recorded in validation/fixtures/fixtures.json are the
ONLY real calibration data the validators have — every gate's idea of
good motion, identity drift and content collapse was set by a human
looking at those frames (2026-08-29). These tests hold the record's
shape and the one boundary that keeps it safe: the battery writes only
under validation/out/shot-*, so no fixture key may ever sit there.

Offline by construction: the record is a file in this repository. No
network, no ffmpeg, no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "validation" / "fixtures"
FIXTURES_JSON = FIXTURE_DIR / "fixtures.json"
FIXTURES_MD = FIXTURE_DIR / "FIXTURES.md"

# The namespace the battery is allowed to write. A fixture key under it
# would be one shot id collision away from being overwritten.
BATTERY_PREFIX = "validation/out/shot-"

# One fixture per failure mode the gates were calibrated to tell apart —
# exactly these, no more and no fewer. A fourth role would be evidence
# nobody looked at; a missing one would be a gate with no calibration.
ROLES = {"GOOD_MOTION", "IDENTITY_DRIFT", "CONTENT_COLLAPSE"}

# Every field a fixture must carry. `prompt` and `seed` are required
# even though their VALUES are "not recorded" — the worker does not
# report a seed at all — because an honest gap must be stated, not
# omitted where it could later be filled in with a reconstruction.
REQUIRED_FIELDS = {
    "role",
    "key",
    "bytes",
    "duration_seconds",
    "resolution",
    "fps",
    "model",
    "gpu",
    "prompt",
    "seed",
    "reference",
    "verdict",
    "observed",
    "immutable",
}

VERDICTS = {"PASS", "PARTIAL", "FAIL"}


def _record() -> dict:
    return json.loads(FIXTURES_JSON.read_text(encoding="utf-8"))


def _fixtures() -> list:
    return _record()["fixtures"]


def test_the_record_parses_and_holds_exactly_three_fixtures():
    record = _record()
    assert record["recorded"] == "2026-08-29"
    assert len(record["fixtures"]) == 3


def test_the_roles_are_exactly_the_three_calibrated_failure_modes():
    assert {f["role"] for f in _fixtures()} == ROLES


def test_every_fixture_carries_every_required_field():
    for fixture in _fixtures():
        missing = REQUIRED_FIELDS - set(fixture)
        assert not missing, f"{fixture.get('key')!r} is missing {sorted(missing)}"
        assert fixture["verdict"] in VERDICTS


def test_immutable_is_true_on_every_fixture():
    for fixture in _fixtures():
        assert fixture["immutable"] is True, (
            f"{fixture['key']!r} is not marked immutable — these clips are "
            "the only real calibration data the validators have"
        )


def test_fixture_keys_are_unique():
    keys = [f["key"] for f in _fixtures()]
    assert len(keys) == len(set(keys))


def test_no_spend_run_output_can_ever_be_a_fixture_key():
    # THE REAL GUARD, not a constant about a fictional boundary. The first
    # version of this test checked the fixture keys against a prefix no
    # code wrote, while the ACTUAL canary outputs (ltx-001.mp4,
    # final-001.mp4) were byte-identical to the fixture keys — one re-run
    # with the matching output prefix would have silently overwritten the
    # only real calibration data ONIQ has (found in adversarial review,
    # 2026-08-29). So this asserts against the code that spends: every
    # basename spend_run can write — battery shots, both canaries, the
    # plate — must differ from every fixture basename, whatever prefix a
    # dispatch chooses.
    from validation import spend_run

    fixture_basenames = {f["key"].rsplit("/", 1)[-1] for f in _fixtures()}
    producible = {shot["output"] for shot in spend_run.ACTION_BATTERY}
    producible |= {"ltx-canary-001.mp4", "audio-final-001.mp4"}
    overlap = fixture_basenames & producible
    assert not overlap, f"spend_run can overwrite fixture(s): {sorted(overlap)}"

    source = open("validation/spend_run.py", encoding="utf-8").read()
    for forbidden in ("/ltx-001.mp4", "/final-001.mp4"):
        assert forbidden not in source, (
            f"{forbidden!r} is back in spend_run — that key IS a fixture"
        )

    # And the battery really does write under shot-*, which is what lets
    # FIXTURES.md promise the namespace boundary.
    for shot in spend_run.ACTION_BATTERY:
        assert shot["output"].startswith("shot-"), shot["output"]
        assert not any(f["key"].startswith(BATTERY_PREFIX) for f in _fixtures())


def test_fixtures_md_states_both_read_postures():
    text = FIXTURES_MD.read_text(encoding="utf-8")
    assert "PUBLIC_R2_ARTIFACT_READ" in text
    assert "SIGNED_PRIVATE_READ" in text
