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


def test_it_catches_an_unverified_revision_declared_in_the_pin(tmp_path, monkeypatch):
    """A fabricated sha is a licence problem, not a bug — the owner accepted
    terms at specific bytes, and nobody else can do that for them."""
    root = _sandbox(tmp_path, monkeypatch)
    with open(root / "ltx-upscaler.pin", "a", encoding="utf-8") as fh:
        fh.write("a-r-r-o-w/LTX-0.9.8-Latent-Upsampler " + "d" * 40 + "\n")
    failures = build_proof.run().failures
    assert "no unverified revision is declared" in failures
    assert "pin declares no revision, so the stage is a no-op" in failures


def test_a_sha_quoted_in_the_pins_PROSE_is_not_a_declaration():
    """The pin explains itself by citing the LTX transformer's own verified
    revision. A raw-text scan would fail on the documentation rather than on
    the data, and a check that accuses the explanation is one people delete."""
    pin = build_proof._read("ltx-upscaler.pin")
    assert "8984fa25007f376c1a299016d0957a37a2f797bb" in pin
    assert build_proof._active_lines(pin) == []
    assert "no unverified revision is declared" not in build_proof.run().failures


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
