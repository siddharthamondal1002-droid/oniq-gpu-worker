"""WHICH SPATIAL LATENT UPSAMPLER, AT WHICH REVISION, UNDER WHICH LICENCE?

`ltx-upscaler.pin` needs one line — `<repo> <40-char sha>` — and it has stayed
empty because the three facts it encodes (the revision, the licence, and the
component config) are registry metadata that exists nowhere else. The build
container's egress proxy refuses huggingface.co (403 to CONNECT, confirmed for
huggingface.co, hf.co and cdn-lfs.huggingface.co), so they cannot be read from
there, and a plausible-looking sha typed by hand is a fabrication wearing the
costume of a fact.

A GitHub runner CAN reach the registry — `ltx-discover` and `model-bench`
already do, with the same HF_TOKEN. This module is the $0 read that resolves
the upsampler specifically, so the owner names a real revision instead of
copying one out of a browser.

WHY IT MEASURES TWO CANDIDATES RATHER THAN ONE. The diffusers LTX documentation
(fetched 2026-08-31 from raw.githubusercontent.com, which this environment CAN
reach) names both:

    Lightricks/ltxv-spatial-upscaler-0.9.7   paired with LTX-Video-0.9.7-dev
    a-r-r-o-w/LTX-0.9.8-Latent-Upsampler     paired with LTX-Video-0.9.8-13B-distilled

The first is Lightricks' own, so it falls inside the LTX Open Weights terms the
owner has accepted. The second is a personal namespace the docs point at with
upstream's own note "TODO: Update the checkpoint here once updated in LTX org",
so its right to redistribute derived weights is a separate question that its
LICENSE/NOTICE files have to answer. Measuring both, side by side, is what lets
that be decided on evidence.

It CHOOSES NOTHING. Like hf_discover, it lists and measures; naming the model
ONIQ ships is the owner's decision, and the licence is theirs to accept.

The token is used and never printed — the same discipline as hf_auth.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from validation.hf_discover import INFO_API, _get, _why

# The two the diffusers documentation names. Hardcoded rather than taken as a
# workflow input because GitHub caps workflow_dispatch at 10 inputs and
# test_no_workflow_exceeds_githubs_dispatch_input_cap enforces it — a new mode
# costs nothing, a new input costs one of ten.
CANDIDATES = (
    "Lightricks/ltxv-spatial-upscaler-0.9.7",
    "a-r-r-o-w/LTX-0.9.8-Latent-Upsampler",
)

RAW = "https://huggingface.co/{repo}/resolve/{revision}/{path}"

# The tree endpoint, taken from huggingface_hub's HfApi.list_repo_tree:
#   f"{endpoint}/api/{repo_type}s/{repo_id}/tree/{revision}{path_in_repo}"
# It is the only read here that returns a Xet hash — model-info's siblings
# list does not carry one. See _vae_tree for why that matters.
TREE_API = "https://huggingface.co/api/models/{repo}/tree/{revision}/{path}"

# The component whose latent space the upsampler was trained against.
VAE_PREFIX = "vae"

# Exactly what the Dockerfile's upscaler stage asserts field by field. Kept
# here so the read reports the SAME verdict the build would reach, rather than
# a looser one that lets a doomed pin through to a 25-minute build.
MODEL_CLASS = "LTXLatentUpsamplerModel"
EXPECTED = {
    "dims": 3,
    "in_channels": 128,
    "mid_channels": 512,
    "num_blocks_per_stage": 4,
    "spatial_upsample": True,
    "temporal_upsample": False,
}

# The Dockerfile's own guard for this component: a spatial upsampler that
# weighs more than this is not a spatial upsampler.
SIZE_GUARD_BYTES = 1024**3

# A licence has to SHIP with the weights: the build refuses a bake whose
# snapshot carries no terms beside the bytes.
#
# MATCHED BY SUBSTRING, not by an exact filename list. The first run of this
# module (2026-08-31) reported "licence files NONE" for
# Lightricks/ltxv-spatial-upscaler-0.9.7, which actually ships
# LTX-Video-Open-Weights-License-0.X.txt — a false negative that would have
# condemned the one licence-clean candidate. Publishers name their terms after
# the licence, not after the convention.
LICENCE_MARKERS = ("LICEN", "NOTICE", "COPYING", "TERMS")


def _licence_files(root_files) -> list:
    return [p for p in root_files
            if any(m in p.upper() for m in LICENCE_MARKERS)]


def _get_text(url: str, token, timeout: int = 60) -> str:
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def _lfs_sha256(entry) -> str | None:
    """The sha256 git-LFS records for one file, or None if none came back.

    ONE reader for both callers. The cross-check used to inline its own copy
    that read only `oid`, while this one also fell back to `sha256`; two
    readings of one field is how a laxer path and a stricter path drift apart,
    and the stricter one then reports an absence the other would not have.
    """
    lfs = entry.get("lfs") or {}
    return lfs.get("oid") or lfs.get("sha256")


def measure(repo: str, token, get=_get, get_text=_get_text, revision=None) -> dict:
    """One candidate, against the gates the build would apply to it."""
    row: dict = {"repo": repo}
    url = INFO_API.format(repo=repo)
    if revision:
        url = url.replace("/api/models/", "/api/models/").replace(
            f"{repo}?", f"{repo}/revision/{revision}?")
    try:
        info = get(url, token)
    except Exception as exc:
        row.update(verdict="UNREADABLE", detail=_why(exc))
        return row

    row["revision"] = info.get("sha")
    card = info.get("cardData") or {}
    row["licence"] = card.get("license") or next(
        (t.split(":", 1)[1] for t in (info.get("tags") or []) if t.startswith("license:")),
        None,
    )
    row["licence_name"] = card.get("license_name")
    row["licence_link"] = card.get("license_link")

    paths = {
        (s.get("rfilename") or ""): (s.get("size") or 0)
        for s in info.get("siblings") or []
    }
    # THE CONTENT HASH, where the registry gives one. Two repositories can
    # publish files of identical SIZE and different bytes; only the hash
    # settles whether they are the same artifact. Git-LFS stores a sha256 per
    # blob and HuggingFace returns it with blobs=true, so this is free.
    row["blob_hashes"] = {
        (s.get("rfilename") or ""): _lfs_sha256(s)
        for s in info.get("siblings") or []
        if (s.get("rfilename") or "").endswith((".safetensors", ".bin"))
        and (s.get("lfs") or {})
    }
    row["root_files"] = sorted(p for p in paths if "/" not in p)
    row["licence_files"] = _licence_files(row["root_files"])

    # TWO REPOSITORY SHAPES, and the difference is not cosmetic.
    #
    #   BARE COMPONENT   config.json at the root, weights beside it. Loaded
    #                    with LTXLatentUpsamplerModel.from_pretrained, which
    #                    is what the Dockerfile's bake does today.
    #   PIPELINE         model_index.json at the root and the component in a
    #                    subfolder. Loaded with
    #                    LTXLatentUpsamplePipeline.from_pretrained, and it
    #                    carries a VAE too — which is why its byte total is
    #                    several GiB and the component guard cannot be applied
    #                    to the repository as a whole.
    row["is_pipeline"] = "model_index.json" in paths
    subfolders = sorted({p.split("/", 1)[0] for p in paths if "/" in p})
    row["subfolders"] = subfolders
    candidates = ["config.json"] + [f"{d}/config.json" for d in subfolders
                                    if "upsampl" in d.lower() or "upscal" in d.lower()]
    row["config_path"] = next((c for c in candidates if c in paths), None)

    prefix = row["config_path"].rsplit("/", 1)[0] + "/" if (
        row["config_path"] and "/" in row["config_path"]) else ""
    row["component_prefix"] = prefix or "(repository root)"
    row["weight_bytes"] = sum(
        size for p, size in paths.items()
        if p.startswith(prefix) and p.endswith((".safetensors", ".bin", ".pt", ".pth"))
    )
    row["repo_bytes"] = sum(paths.values())
    row["within_size_guard"] = 0 < row["weight_bytes"] <= SIZE_GUARD_BYTES

    # The config decides whether this component is even the right SHAPE. Read
    # at the resolved revision, never at a branch name, so what is reported is
    # what a pin to that sha would actually bake.
    if row["revision"] and row["config_path"]:
        try:
            cfg = json.loads(get_text(
                RAW.format(repo=repo, revision=row["revision"],
                           path=row["config_path"]), token))
            row["config"] = {k: cfg.get(k) for k in EXPECTED}
            row["declared_class"] = cfg.get("_class_name")
            # ABSENT IS NOT WRONG. The build constructs the model with
            # LTXLatentUpsamplerModel.from_config and asserts the RESOLVED
            # config, so a field the publisher left to the class default is
            # verified there rather than here. Lightricks' config declares only
            # _class_name; treating that as six mismatches condemned the one
            # licence-clean candidate (measured 2026-08-31). Only a field that
            # is PRESENT AND WRONG is a mismatch.
            row["defaulted"] = sorted(k for k in EXPECTED if k not in cfg)
            row["config_mismatch"] = {
                k: cfg.get(k) for k, v in EXPECTED.items()
                if k in cfg and cfg.get(k) != v
            }
            if cfg.get("_class_name") != MODEL_CLASS:
                row["config_mismatch"]["_class_name"] = cfg.get("_class_name")
        except Exception as exc:
            row["config"] = None
            row["config_mismatch"] = {"config.json": _why(exc)}
    else:
        row["config"] = None
        row["config_mismatch"] = {
            "config.json": f"not found (searched {candidates})"
            if row["revision"] else "revision not resolved"}

    row["verdict"] = (
        "PINNABLE"
        if row["revision"] and not row["config_mismatch"]
        and row["licence_files"] and row["within_size_guard"]
        else "NOT PINNABLE"
    )
    return row


# THE CHECKPOINT ONIQ ACTUALLY BAKES, and the revision its Dockerfile pins.
# The upsampler was TRAINED against a particular VAE latent distribution, and
# LTXLatentUpsamplePipeline normalises with self.vae.latents_mean/latents_std —
# ONIQ's own baked vae. If that vae is byte-identical to the one shipped beside
# the upsampler, the latent space is provably the same and the 0.9.7-vs-2B
# question is closed rather than argued.
BAKED_REPO = "Lightricks/LTX-Video"
BAKED_REVISION = "8984fa25007f376c1a299016d0957a37a2f797bb"


def _vae_tree(repo: str, revision: str, token, get) -> dict:
    """The vae component's content hashes, read from the TREE endpoint.

    Returns {"files": {path: {"sha256", "xet", "raw"}}, "error": str | None}.

    WHY NOT THE SIBLINGS LIST, which every other read here uses. Run
    33424489532 printed `baked vae None` for
    Lightricks/LTX-Video@8984fa25's vae/diffusion_pytorch_model.safetensors:
    model-info returned an lfs dict for that file with no oid inside it, and
    an absence compared against real hashes answered SAME LATENTS: False.

    huggingface_hub's own parsers say where a hash actually lives.
    RepoSibling — what model-info returns — carries only rfilename, size,
    blob_id and lfs. RepoFile — what THIS endpoint returns — carries lfs.oid
    AND xetHash. And HfApi.copy_files reads both off one file, `if not
    src_file.lfs: continue` then `if not src_file.xet_hash: raise`, which
    means a Xet-backed blob still keeps its LFS sha256. So Xet storage does
    not on its own explain a missing oid, and this is the read that can
    return a hash in either currency.

    One page is read, not the full Link-header pagination: this is pointed at
    a single component folder holding a handful of files, and a vae that
    needed a second page would be a different repository shape entirely.
    """
    url = TREE_API.format(repo=repo, revision=revision, path=VAE_PREFIX)
    try:
        entries = get(url, token)
    except Exception as exc:
        return {"files": {}, "error": _why(exc)}
    files = {}
    for entry in entries if isinstance(entries, list) else []:
        path = entry.get("path") or ""
        if entry.get("type") == "directory" or not path.endswith(
                (".safetensors", ".bin")):
            continue
        files[path] = {
            "sha256": _lfs_sha256(entry),
            "xet": entry.get("xetHash"),
            # Same weights at a different dtype differ in BYTES and in SIZE;
            # same weights re-serialised differ in bytes alone. The size is
            # what tells those two apart.
            "size": entry.get("size"),
            # KEPT SO A BARREN READ DIAGNOSES ITSELF. The last unexplained
            # absence cost three runs and a wrong note in the pin, because
            # nothing recorded what the registry had actually returned.
            "raw": {k: v for k, v in entry.items()
                    if k in ("oid", "lfs", "xetHash", "size", "type")},
        }
    return {"files": files, "error": None}


def _same_content(mine: dict, theirs: dict) -> str:
    """SAME / DIFFERENT / UNKNOWN for two sets of file hashes.

    LIKE IS COMPARED WITH LIKE. A sha256 and a Xet hash are different
    functions over the same bytes, so they are never compared to each other —
    that would answer DIFFERENT for two identical files, which is precisely
    the class of false negative this cross-check has already produced three
    times. Either currency settles it alone; both are deterministic over
    content.

    UNKNOWN IS NOT FALSE. A hash that did not come back is not a hash that
    disagreed, so the third value exists and the caller must handle it.
    """
    for currency in ("sha256", "xet"):
        ours = {h[currency] for h in mine.values() if h.get(currency)}
        yours = {h[currency] for h in theirs.values() if h.get(currency)}
        if ours and yours:
            return (f"SAME ({currency})" if ours & yours
                    else f"DIFFERENT ({currency})")
    return "UNKNOWN — no content hash returned in either currency"


def _vae_config(repo: str, revision: str, token, get_text) -> dict:
    """The vae's own config.json at that revision."""
    try:
        return json.loads(get_text(
            RAW.format(repo=repo, revision=revision,
                       path=f"{VAE_PREFIX}/config.json"), token))
    except Exception as exc:
        return {"_error": _why(exc)}


# Metadata about WHERE a config came from, never about what it configures.
_PROVENANCE = {"_diffusers_version", "_name_or_path", "_class_name"}


def _config_delta(mine: dict, theirs: dict) -> dict:
    """Every configured value the two vaes disagree on.

    DIFFERENT BYTES IS NOT DIFFERENT LATENT SPACE. Two safetensors files can
    hold the same weights at a different dtype, or the same weights
    re-serialised, and hash differently either way. What
    LTXLatentUpsamplePipeline actually reads off the vae is its CONFIG —
    latents_mean and latents_std are the normalisation it applies, and
    latent_channels is the shape the upsampler was built for. If those agree,
    a byte difference is a packaging difference; if they disagree, the two
    checkpoints do not share a latent space and no hash was needed to know it.
    """
    keys = (set(mine) | set(theirs)) - _PROVENANCE
    return {k: [mine.get(k), theirs.get(k)] for k in sorted(keys)
            if mine.get(k) != theirs.get(k)}


def _brief(value, keep: int = 4):
    """A long list printed as its head and its length, never in full.

    latents_mean and latents_std are 128 floats each. Printing two of them
    turns the one line that matters into four screens nobody reads.
    """
    if isinstance(value, list) and len(value) > keep:
        return f"[{', '.join(repr(v) for v in value[:keep])}, ... {len(value)} values]"
    return repr(value)


def vae_crosscheck(rows, token, get=_get, get_text=_get_text) -> dict:
    """Is the vae beside the upsampler the vae ONIQ bakes?"""
    out: dict = {"baked": f"{BAKED_REPO}@{BAKED_REVISION}"}
    baked = _vae_tree(BAKED_REPO, BAKED_REVISION, token, get)
    out["baked_vae"] = baked
    if baked["error"]:
        out["error"] = baked["error"]
        return out

    out["candidate_vae"] = {}
    for row in rows:
        if not row.get("revision"):
            continue
        if not any(k.startswith(VAE_PREFIX + "/")
                   for k in (row.get("blob_hashes") or {})):
            continue  # a bare component repo ships no vae to compare
        out["candidate_vae"][row["repo"]] = _vae_tree(
            row["repo"], row["revision"], token, get)

    out["matches"] = {}
    for repo, theirs in out["candidate_vae"].items():
        out["matches"][repo] = (
            f"UNREADABLE — {theirs['error']}" if theirs["error"]
            else _same_content(baked["files"], theirs["files"]))

    # WHAT A BYTE DIFFERENCE MEANS. Only asked when the bytes actually differ:
    # identical files need no interpreting, and an absent hash is not a
    # difference to interpret.
    baked_cfg = None
    out["config_delta"] = {}
    for repo, verdict in out["matches"].items():
        if not verdict.startswith("DIFFERENT"):
            continue
        if baked_cfg is None:
            baked_cfg = _vae_config(BAKED_REPO, BAKED_REVISION, token, get_text)
        theirs = _vae_config(
            repo, dict((r["repo"], r.get("revision")) for r in rows)[repo],
            token, get_text)
        if "_error" in baked_cfg or "_error" in theirs:
            out["config_delta"][repo] = {
                "_error": baked_cfg.get("_error") or theirs.get("_error")}
        else:
            out["config_delta"][repo] = _config_delta(baked_cfg, theirs)

    if any(v.startswith("UNKNOWN") for v in out["matches"].values()) or (
            not baked["files"]):
        out["note"] = (
            "no content hash came back in either currency, so this question "
            "is UNRESOLVED rather than answered — never DIFFERENT. The raw "
            "registry metadata for each file is printed above; read it rather "
            "than guessing a cause, which is what the last note in the pin did"
        )
    return out


def _print_vae(label: str, block: dict, indent: str = "    ") -> None:
    """One component folder's hashes, in BOTH currencies, plus the raw entry.

    The raw line only appears when neither hash came back — that is the case
    that has misled this module before, and it is the one where the next
    reader needs the registry's own words rather than a summary of them.
    """
    if block.get("error"):
        print(f"{indent}{label:<14} UNREADABLE — {block['error']}")
        return
    if not block.get("files"):
        print(f"{indent}{label:<14} no .safetensors returned for this folder")
        return
    for name, h in sorted(block["files"].items()):
        print(f"{indent}{label:<14} {name}")
        print(f"{indent}  sha256       {h.get('sha256')}")
        print(f"{indent}  xet          {h.get('xet')}")
        if not h.get("sha256") and not h.get("xet"):
            print(f"{indent}  RAW          {h.get('raw')}")


def report(token, get=_get, get_text=_get_text) -> tuple:
    if not token:
        print("BLOCKED: no credential, and an anonymous read cannot tell a "
              "gated repository from one that does not exist")
        return 2, []

    rows = [measure(repo, token, get, get_text) for repo in CANDIDATES]
    for row in rows:
        print("")
        print(f"  {row['repo']}")
        if row.get("verdict") == "UNREADABLE":
            print(f"    UNREADABLE — {row['detail']}")
            continue
        print(f"    revision      {row.get('revision')}")
        print(f"    licence tag   {row.get('licence')!r}")
        print(f"    license_name  {row.get('licence_name')!r}")
        print(f"    license_link  {row.get('licence_link')!r}")
        print(f"    licence files {row.get('licence_files') or 'NONE — the build refuses this'}")
        print(f"    root files    {row.get('root_files')}")
        print(f"    shape         {'PIPELINE (model_index.json)' if row.get('is_pipeline') else 'BARE COMPONENT'}")
        print(f"    subfolders    {row.get('subfolders')}")
        print(f"    config at     {row.get('config_path')}")
        print(f"    component     {row.get('component_prefix')}")
        print(f"    component wt  {row.get('weight_bytes')} "
              f"(guard {SIZE_GUARD_BYTES}, within={row.get('within_size_guard')})")
        print(f"    repo bytes    {row.get('repo_bytes')}")
        for name, digest in sorted((row.get("blob_hashes") or {}).items()):
            print(f"    sha256        {digest}  {name}")
        print(f"    class         {row.get('declared_class')!r}")
        print(f"    config        {row.get('config')}")
        if row.get("defaulted"):
            print(f"    defaulted     {row['defaulted']} "
                  f"(verified at build time by from_config)")
        if row.get("config_mismatch"):
            print(f"    MISMATCH      {row['config_mismatch']}")
        print(f"    VERDICT       {row['verdict']}")

    cross = vae_crosscheck(rows, token, get, get_text)
    print("")
    print("  VAE CROSS-CHECK — is the latent space the same one ONIQ bakes?")
    print(f"    baked          {cross.get('baked')}")
    if cross.get("error"):
        print(f"    UNREADABLE     {cross['error']}")
    else:
        _print_vae("baked vae", cross.get("baked_vae") or {})
        for repo, block in (cross.get("candidate_vae") or {}).items():
            print(f"    {repo}")
            _print_vae("vae", block, indent="      ")
        print(f"    SAME LATENTS   {cross.get('matches')}")
        for repo, delta in (cross.get("config_delta") or {}).items():
            if delta.get("_error"):
                print(f"    vae config     UNREADABLE — {delta['_error']}")
            elif not delta:
                print("    vae config     IDENTICAL — the weights differ in "
                      "bytes, but every configured value agrees, including "
                      "the latents_mean/latents_std this pipeline normalises "
                      "with. A packaging difference, not a latent-space one.")
            else:
                print(f"    vae config     DIFFERS on {len(delta)} field(s) — "
                      "the two checkpoints do NOT share a latent space:")
                for key, (ours, theirs_) in delta.items():
                    print(f"      {key}")
                    print(f"        baked {_brief(ours)}")
                    print(f"        {repo.split('/')[-1]} {_brief(theirs_)}")
        if cross.get("note"):
            print(f"    NOTE           {cross['note']}")

    print("")
    print("A VERDICT OF PINNABLE IS NOT PERMISSION. It says the revision "
          "resolves, the config is the right shape, terms ship beside the "
          "weights and the size is sane. Whether those terms permit ONIQ to "
          "redistribute the weights inside a private commercial image is the "
          "owner's judgement, and reading the licence text is part of making "
          "it. To enable, put one line in ltx-upscaler.pin:")
    print("")
    for row in rows:
        if row.get("verdict") == "PINNABLE":
            print(f"    {row['repo']} {row['revision']}")
    print("")
    return (0 if any(r.get("verdict") == "PINNABLE" for r in rows) else 1), rows


def main(argv) -> int:
    import os

    from validation.hf_auth import TOKEN_VAR

    code, _ = report(os.environ.get(TOKEN_VAR))
    return code


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
