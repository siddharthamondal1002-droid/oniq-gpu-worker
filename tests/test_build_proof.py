"""The zero-spend build proof runs, and it actually fails when it should.

A proof that has never been seen to fail is a proof nobody should trust. Each
case below breaks one thing the module claims to catch and checks that it does.
"""

import json
import os
import shutil

import pytest

from validation import build_proof


def test_the_proof_passes_on_the_repository_as_it_stands():
    p = build_proof.run()
    assert p.failures == [], p.failures
    assert p.checks > 25


def _sandbox(tmp_path, monkeypatch):
    """A copy of the repo the proof can be run against destructively."""
    root = tmp_path / "repo"
    root.mkdir()
    for name in os.listdir(build_proof.ROOT):
        if name in ("tests", "__pycache__", ".git", "validation"):
            continue
        src = os.path.join(build_proof.ROOT, name)
        if os.path.isfile(src):
            shutil.copy2(src, root / name)
    shutil.copytree(os.path.join(build_proof.ROOT, "validation"), root / "validation")
    monkeypatch.setattr(build_proof, "ROOT", str(root))
    return root


def test_it_catches_a_module_that_is_imported_but_never_copied(tmp_path, monkeypatch):
    """The failure mode that kills a worker on start-up, after the endpoint has
    already scaled up and the job is already someone's."""
    root = _sandbox(tmp_path, monkeypatch)
    (root / "orphan.py").write_text("x = 1\n", encoding="utf-8")
    src = (root / "videogen.py").read_text(encoding="utf-8")
    (root / "videogen.py").write_text("import orphan\n" + src, encoding="utf-8")
    assert "every first-party import is COPY'd and un-ignored" in build_proof.run().failures


def test_it_catches_a_syntax_error_in_a_bake(tmp_path, monkeypatch):
    """Surfaces only after tens of GiB have downloaded, if it surfaces at all."""
    root = _sandbox(tmp_path, monkeypatch)
    docker = (root / "Dockerfile").read_text(encoding="utf-8")
    docker = docker.replace("import json, os, shutil", "import json, os, shutil,", 1)
    (root / "Dockerfile").write_text(docker, encoding="utf-8")
    assert any(f.startswith("bake block") for f in build_proof.run().failures)


def test_it_catches_a_second_unverified_revision_declared_in_the_pin(tmp_path, monkeypatch):
    """A fabricated sha is a licence problem, not a bug — the owner accepted
    terms at specific bytes, and nobody else can do that for them.

    The way this goes wrong is somebody appending repositories the owner never
    accepted. The build would bake whichever it read first, which is exactly
    the ambiguity a pin exists to remove — so more than one declaration must
    fail however many there are, and the pin shipping ZERO today (the pairing
    is refused, see the file) must not make the check unreachable."""
    root = _sandbox(tmp_path, monkeypatch)
    with open(root / "ltx-upscaler.pin", "a", encoding="utf-8") as fh:
        fh.write("a-r-r-o-w/LTX-0.9.8-Latent-Upsampler " + "d" * 40 + "\n")
        fh.write("someone/else " + "e" * 40 + "\n")
    failures = build_proof.run().failures
    assert any("at most one line" in f for f in failures), failures


def test_a_sha_quoted_in_the_pins_PROSE_is_not_a_declaration():
    """The pin explains itself by citing revisions and content hashes. A
    raw-text scan would fail on the documentation rather than on the data, and
    a check that accuses the explanation is one people delete.

    THE PROPERTY, NOT THE PARTICULAR HASHES. This test named three specific
    shas and broke twice in one day as the prose was rewritten — first when
    the latent-space mismatch was recorded, then when the checkpoint moved to
    meet it. What matters is that the file quotes SEVERAL hex identifiers and
    the parser declares exactly ONE line, so it now asserts that instead."""
    import re

    pin = build_proof._read("ltx-upscaler.pin")
    quoted = set(re.findall(r"\b[0-9a-f]{40}\b|\b[0-9a-f]{64}\b", pin))
    assert len(quoted) >= 3, sorted(quoted)

    active = build_proof._active_lines(pin)
    assert len(active) == 1, active
    repo, revision = active[0].split()
    assert repo == "Lightricks/ltxv-spatial-upscaler-0.9.7"
    # Every OTHER hex identifier in the file is prose, and none of them leaked
    # into the declaration.
    for other in quoted - {revision}:
        assert other not in active[0]
    assert build_proof.run().failures == []

def test_it_catches_a_signature_manifest_for_the_wrong_diffusers(tmp_path, monkeypatch):
    """A manifest describing a library the image does not install would pass
    while checking nothing."""
    root = _sandbox(tmp_path, monkeypatch)
    sigs = json.loads((root / "ltx_signatures.json").read_text(encoding="utf-8"))
    sigs["_diffusers_version"] = "0.99.0"
    (root / "ltx_signatures.json").write_text(json.dumps(sigs), encoding="utf-8")
    assert "manifest version equals the requirements pin" in build_proof.run().failures


def test_it_names_what_it_cannot_prove(capsys):
    build_proof.main()
    out = capsys.readouterr().out
    # Honesty about scope is part of the proof: a daemon-free run cannot
    # produce layer sizes or a digest, and must not imply that it did.
    assert "NOT PROVEN HERE" in out
    for absent in ("digest", "image size", "layer sizes"):
        assert absent in out


class TestTheUpscalerPinIsPolicedNotJustAllowed:
    """The pin went from "must be empty" to "must be well-formed" on
    2026-08-31, once the registry had actually been read. That is a tightening
    only if a half-declaration still fails, so each way of getting it wrong is
    broken here and asserted to be caught."""

    def _pin(self, tmp_path, monkeypatch, line):
        root = _sandbox(tmp_path, monkeypatch)
        path = os.path.join(root, "ltx-upscaler.pin")
        original = open(path, encoding="utf-8").read()
        kept = "\n".join(l for l in original.splitlines()
                         if l.strip().startswith("#") or not l.strip())
        open(path, "w", encoding="utf-8").write(kept + "\n" + line + "\n")
        return build_proof.run()

    def test_the_real_pin_passes(self):
        p = build_proof.run()
        assert p.failures == [], p.failures

    def test_a_branch_name_instead_of_a_sha_is_caught(self, tmp_path, monkeypatch):
        p = self._pin(tmp_path, monkeypatch,
                      "Lightricks/ltxv-spatial-upscaler-0.9.7 main")
        assert any("40-char sha" in f for f in p.failures), p.failures

    def test_a_short_sha_is_caught(self, tmp_path, monkeypatch):
        p = self._pin(tmp_path, monkeypatch,
                      "Lightricks/ltxv-spatial-upscaler-0.9.7 c96c168")
        assert any("40-char sha" in f for f in p.failures), p.failures

    def test_a_revision_with_no_repository_is_caught(self, tmp_path, monkeypatch):
        p = self._pin(tmp_path, monkeypatch,
                      "c96c168c2bd8bbc82c9fe8259e5f89f8b2ea293f")
        assert any("<repo> <revision>" in f for f in p.failures), p.failures

    def test_an_unnamespaced_repository_is_caught(self, tmp_path, monkeypatch):
        p = self._pin(tmp_path, monkeypatch,
                      "ltxv-spatial-upscaler c96c168c2bd8bbc82c9fe8259e5f89f8b2ea293f")
        assert any("namespaced" in f for f in p.failures), p.failures

    def test_a_pin_without_the_owner_licence_record_is_caught(self, tmp_path, monkeypatch):
        root = _sandbox(tmp_path, monkeypatch)
        path = os.path.join(root, "ltx-upscaler.pin")
        # A declaration with the acceptance record stripped out: the revision
        # IS the licence, so a pin that does not say who accepted it, and when,
        # is a pin nobody can audit.
        open(path, "w", encoding="utf-8").write(
            "# no acceptance recorded here\n"
            "Lightricks/ltxv-spatial-upscaler-0.9.7 "
            "c96c168c2bd8bbc82c9fe8259e5f89f8b2ea293f\n")
        p = build_proof.run()
        assert any("licence acceptance" in f for f in p.failures), p.failures

    def test_two_declarations_are_caught(self, tmp_path, monkeypatch):
        p = self._pin(
            tmp_path, monkeypatch,
            "Lightricks/ltxv-spatial-upscaler-0.9.7 "
            "c96c168c2bd8bbc82c9fe8259e5f89f8b2ea293f\n"
            "a-r-r-o-w/LTX-0.9.8-Latent-Upsampler "
            "e0c981533db26531c47dec16a124586cea53f11f")
        assert any("at most one line" in f for f in p.failures), p.failures
