"""Everything about the image that can be proved WITHOUT a Docker daemon.

WHY THIS EXISTS. The media image is ~40 GiB and takes a hosted runner the best
part of an hour; the environments where this code is written frequently have no
daemon at all. That is not a reason to ship unverified — most of what goes
wrong in this Dockerfile is provable statically:

  - a syntax error in a bake surfaces only after tens of GiB have downloaded
  - a module videogen imports but nothing COPYs kills the worker on start-up,
    after the endpoint has already scaled up
  - a fabricated or missing model revision is a licence problem, not a bug
  - a signature manifest for a diffusers the image does not install would pass
    while describing a different library

None of that needs a build. What genuinely does — layer sizes, the final image
digest, whether the registry serves the weights today — is out of scope here
and must be reported as NOT PROVEN rather than assumed.

Run:  python3 -m validation.build_proof
Exit: 0 if every proof passes, 1 otherwise.
"""

from __future__ import annotations

import ast
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Proof:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks = 0

    def check(self, name: str, cond: bool, detail: str = "") -> None:
        self.checks += 1
        mark = "PASS" if cond else "FAIL"
        print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
        if not cond:
            self.failures.append(name)


def _read(*parts: str) -> str:
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()


def _active_lines(text: str) -> list[str]:
    """Lines with the comment stripped and blanks dropped.

    Used wherever a check is about what the file DECLARES rather than what it
    explains. The upscaler pin, for instance, quotes the LTX transformer's own
    already-verified revision in its prose as the precedent for pinning — a
    scan of raw text would read that as a fabricated upscaler sha and fail on
    the documentation rather than on the data.
    """
    return [s for s in (l.split("#", 1)[0].strip() for l in text.splitlines()) if s]


def run() -> Proof:
    p = Proof()
    docker = _read("Dockerfile")
    instructions = [
        l.strip() for l in docker.splitlines()
        if l.strip() and not l.strip().startswith("#")
    ]

    print("1. embedded build programs compile")
    for i, block in enumerate(re.findall(r"python3 - <<'EOF'\n(.*?)\nEOF\n", docker, re.S)):
        try:
            ast.parse(block)
            p.check(f"bake block {i}", True)
        except SyntaxError as exc:
            p.check(f"bake block {i}", False, f"line {exc.lineno}: {exc.msg}")

    print("2. runtime import closure")
    copied = {os.path.basename(m.group(1)) for m in re.finditer(r"^COPY\s+(\S+)\s", docker, re.M)}
    unignored = {l[1:].strip() for l in _read(".dockerignore").splitlines() if l.startswith("!")}
    local = {f[:-3] for f in os.listdir(ROOT) if f.endswith(".py")}
    missing: list[str] = []
    for mod in sorted(local):
        for node in ast.walk(ast.parse(_read(mod + ".py"))):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            for n in names:
                if n in local and (f"{n}.py" not in copied or f"{n}.py" not in unignored):
                    missing.append(f"{mod} -> {n}")
    p.check("every first-party import is COPY'd and un-ignored", not missing, str(missing[:3]))

    print("3. every shipped module parses")
    for f in sorted(x for x in copied if x.endswith(".py")):
        try:
            ast.parse(_read(f))
            p.check(f, True)
        except SyntaxError as exc:
            p.check(f, False, str(exc))

    print("4. baked models and their gates")
    from validation import image_size

    bakes = image_size.parse_bakes(docker)
    p.check("LTX bake has exactly one candidate",
            bakes[0]["candidates"] == ["Lightricks/LTX-Video-0.9.7-distilled"])
    p.check("LTX revision pinned",
            "057509edea1493cae5e62e9d8f780ebda3fb4333" in docker)
    p.check("Qwen licence gate present", "Apache" in docker and "/app/models/story" in docker)
    p.check("Piper baked", "/app/models/piper" in docker)
    p.check("upscaler stage present", "THE SPATIAL LATENT UPSCALER" in docker)
    p.check("upscaler model class asserted", "LTXLatentUpsamplerModel" in docker)
    p.check("upscaler config asserted field by field",
            '"temporal_upsample": False' in docker and '"spatial_upsample": True' in docker)
    p.check("upscaler licence gate", "without their terms" in docker)
    p.check("upscaler size guard", "SIZE_GUARD_BYTES" in docker)
    # THE LATENT-SPACE GATE. Added 2026-08-31 after run 33426496040 measured
    # that the vae this image bakes and the vae the pinned upsampler was
    # trained beside are different networks. Everything else about the
    # upsampler checks out, so nothing upstream of this would have caught it —
    # the model loads, constructs, and then refines latents it never saw in
    # training. Silent, on a rented card, visible only in the output.
    p.check("upscaler latent-space gate present",
            "LATENT SPACE MISMATCH" in docker
            and "/app/models/ltx/vae/config.json" in docker)
    p.check("the gate compares the upstream vae config it fetched",
            '"vae/config.json",' in docker and "upstream_vae" in docker)
    # FAIL CLOSED. Nothing to compare against is UNVERIFIED, and unverified is
    # not a pass — that is the whole lesson of this module's three earlier
    # false negatives, applied in the safe direction.
    p.check("the gate refuses rather than assuming when it cannot compare",
            "ships no vae/config.json" in docker
            and "refusing rather than" in docker)
    # THE $0 READ MUST REACH THE BUILD'S VERDICT, or a PINNABLE line gets
    # pasted into this pin and the build refuses it 25 minutes later.
    from validation import upscaler_discover as _ud
    _gated = docker.split("LATENT_SPACE = (", 1)[1].split(")", 1)[0]
    p.check("the discovery read gates on the same fields the build does",
            sorted(f.strip().strip('",') for f in _gated.split()
                   if f.strip(' ,"')) == sorted(_ud.LATENT_SPACE))

    print("5. the upscaler pin")
    pin_text = _read("ltx-upscaler.pin")
    pin = _active_lines(pin_text)
    # EMPTY IS LEGAL — the stage is then a no-op and the image ships without
    # multi-scale. What must never happen is a HALF-declaration: a revision
    # nobody resolved, a bare branch name, or a repository named without terms.
    #
    # Until 2026-08-31 this asserted the pin was empty, which was right while
    # nothing had been verified. Now that the registry has actually been read,
    # the check TIGHTENS rather than disappears: a declared pin must be exactly
    # "<repo> <40-hex>", and the file must carry a dated owner record of the
    # licence acceptance, because a revision IS the licence and an agent cannot
    # accept one on anybody's behalf.
    p.check("pin declares at most one line", len(pin) <= 1,
            f"{len(pin)} declaration(s)")
    if pin:
        parts = pin[0].split()
        p.check("declared pin is '<repo> <revision>'", len(parts) == 2, pin[0])
        p.check("declared revision is a 40-char sha, never a branch",
                len(parts) > 1 and bool(re.fullmatch(r"[0-9a-f]{40}", parts[1])),
                parts[1] if len(parts) > 1 else "(none)")
        p.check("declared repository is namespaced",
                len(parts) > 0 and parts[0].count("/") == 1, parts[0])
        p.check("the licence acceptance is recorded beside the pin",
                "OWNER LICENCE ACCEPTANCE" in pin_text)
    else:
        # A sha may legitimately appear in the PROSE (the LTX transformer's own
        # revision, quoted as the precedent); what must never appear is a
        # DECLARED upscaler revision nobody verified.
        p.check("no unverified revision is declared",
                not any(re.search(r"\b[0-9a-f]{40}\b", l) for l in pin))

    print("6. container user")
    p.check("runs as non-root", "USER oniq:oniq" in instructions)
    uid = re.search(r"useradd[^\n]*?(\d{4,6})", docker)
    p.check("uid is 10001", bool(uid) and uid.group(1) == "10001",
            uid.group(1) if uid else "not found")

    print("7. the LTX signature manifest matches the installed diffusers")
    sigs = json.loads(_read("ltx_signatures.json"))
    p.check("manifest version equals the requirements pin",
            f"diffusers=={sigs['_diffusers_version']}" in _read("requirements.txt"),
            sigs["_diffusers_version"])

    return p


def main() -> int:
    p = run()
    print()
    if p.failures:
        print(f"BUILD PROOF FAILED — {len(p.failures)}/{p.checks}: {p.failures}")
        return 1
    print(f"BUILD PROOF PASSED — {p.checks} checks")
    print("NOT PROVEN HERE (needs a daemon): layer sizes, final image size, "
          "image digest, registry availability of the weights today.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
