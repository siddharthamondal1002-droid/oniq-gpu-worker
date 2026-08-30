"""The allocator setting is real, and its ordering is proved not claimed.

Owner directive 2026-08-30: "Do not claim it is configured merely because
it appears in source. The startup log/test must prove: CUDA allocator
configuration loaded before torch.cuda initialization."

PyTorch reads PYTORCH_CUDA_ALLOC_CONF once, when its CUDA allocator comes
up. Set it afterwards and the variable is present while the setting is
ignored — a configuration that every report calls on and the allocator
calls off. These tests pin the ordering, not the presence.
"""

import ast
import importlib
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cudaenv  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_the_setting_is_applied():
    assert cudaenv.APPLIED == "expandable_segments:True"
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_an_operator_value_is_respected_not_overridden(monkeypatch):
    # The template must be able to set a different strategy without this
    # module quietly stamping over it.
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    reloaded = importlib.reload(cudaenv)
    assert reloaded.APPLIED == "max_split_size_mb:128"
    assert reloaded.evidence()["source"] == "inherited"
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    importlib.reload(cudaenv)


def test_handler_imports_cudaenv_before_every_other_module():
    """The property that makes the setting take effect.

    Checked over the AST, in source order: any project module imported
    first could pull torch in and initialise the allocator before cudaenv
    ever runs.
    """
    with open(os.path.join(ROOT, "handler.py"), encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    imports = [
        n for n in tree.body
        if isinstance(n, (ast.Import, ast.ImportFrom))
    ]
    # __future__ is a compiler directive and must stay first; cudaenv is
    # the first real import after it.
    real = [
        n for n in imports
        if not (isinstance(n, ast.ImportFrom) and n.module == "__future__")
    ]
    assert isinstance(real[0], ast.Import)
    assert real[0].names[0].name == "cudaenv", ast.dump(real[0])


def test_evidence_reports_a_bad_ordering_rather_than_hiding_it():
    """Import torch first in a subprocess, then cudaenv, and confirm the
    module says the ordering was wrong instead of reporting success.

    A guard that cannot fail is not a guard — the same lesson as the CI
    import proof that ran an empty program and went green.
    """
    script = (
        "import sys, types;"
        # A stand-in for torch: this container has no CUDA and no torch,
        # and the property under test is 'was torch in sys.modules', which
        # a stub exercises exactly.
        "m = types.ModuleType('torch');"
        "m.cuda = types.SimpleNamespace(is_initialized=lambda: True);"
        "sys.modules['torch'] = m;"
        "sys.path.insert(0, %r);"
        "import cudaenv;"
        "e = cudaenv.evidence();"
        "print(e['ordering_ok'], e['torch_imported_before_cudaenv'],"
        " e['torch_cuda_initialized_before_cudaenv'])"
    ) % ROOT
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert out == "False True True", out


def test_the_module_never_imports_torch_itself():
    # Importing torch to check on torch would create the condition the
    # check exists to detect.
    with open(os.path.join(ROOT, "cudaenv.py"), encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(a.name != "torch" for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module != "torch"
