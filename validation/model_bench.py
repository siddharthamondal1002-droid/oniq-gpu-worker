"""The benchmark's free half: measure all nine candidates, project the matrix.

Runs where HF_TOKEN lives (CI), costs nothing, and touches no GPU. What it
produces is the evidence the owner's questions 1-7 actually require:

    MODEL | PRECISION | VRAM PEAK | GPU | RUNTIME | COST | QUALITY
             ^measured   ^projected  ^derived  ^--- only a real run fills these

RUNTIME, COST and QUALITY are deliberately left empty here. They cannot be
computed, only observed, and a matrix that quietly modelled them would be the
exact failure the owner has been guarding against all week — a plausible
number standing in for a measurement. This module fills the columns that
arithmetic can honestly fill and marks the rest PENDING-PROBE.

ARCHITECTURE IS READ, NEVER DEFAULTED. If a model's config does not yield the
hidden size, depth or VAE ratios, the row is ARCH-INCOMPLETE and carries no
VRAM projection at all. A defaulted architecture produces a confident number
that is wrong, which is worse than no number: nobody schedules a probe to
check a figure that already looks fine.
"""

from __future__ import annotations

import json
import os

from validation import model_registry as mr
from validation import vram
from validation.vram import Arch, Config, Shape

# ONIQ's proven production shape. Every model is measured at ITS closest
# supported size but reported against this, because the decision is "what
# should ONIQ ship", not "which model wins at its own favourite resolution".
ONIQ_FRAMES = 97

# The configurations worth projecting. Each is a real deployment option, not a
# spread of hypotheticals: ship it as published, quantise the DiT, or lean on
# the offloading the model cards document.
CONFIGS: tuple[tuple[str, Config], ...] = (
    ("as published", Config(precision="bf16", offload="none")),
    ("bf16 + model offload", Config(precision="bf16", offload="model")),
    ("bf16 + offload + tiled VAE",
     Config(precision="bf16", offload="model", vae_tile_frames=16)),
    ("fp8 + offload + tiled VAE",
     Config(precision="fp8", offload="model", vae_tile_frames=16)),
    ("fp8 + sequential + tiled VAE",
     Config(precision="fp8", offload="sequential", vae_tile_frames=16)),
)

# Where each architectural number may live, in priority order. Families spell
# these differently; the resolver records WHICH key answered so the output
# shows its work rather than asserting a number from nowhere.
HIDDEN_KEYS = ("hidden_size", "inner_dim", "dim", "d_model")
HEAD_KEYS = ("num_attention_heads", "num_heads", "attention_head_num")
HEAD_DIM_KEYS = ("attention_head_dim", "head_dim", "attention_head_size")
LAYER_KEYS = ("num_layers", "num_hidden_layers", "depth", "num_blocks")
VAE_SPATIAL_KEYS = ("spatial_compression_ratio", "scaling_factor_spatial")
VAE_TEMPORAL_KEYS = ("temporal_compression_ratio", "scaling_factor_temporal")
VAE_CHANNEL_KEYS = ("base_channels", "block_out_channels", "z_dim", "base_dim")


def _first(cfg: dict, keys) -> tuple[object, str | None]:
    for key in keys:
        if cfg.get(key) is not None:
            return cfg[key], key
    return None, None


def _as_int(value) -> int | None:
    """Configs carry ints, lists (block_out_channels) and lists-of-patch-dims.

    A list is reduced to its widest entry, which is the one that sets the
    memory high-water mark — the narrow blocks never bind.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, (list, tuple)) and value:
        ints = [v for v in value if isinstance(v, int) and not isinstance(v, bool)]
        return max(ints) if ints else None
    return None


def _patch(cfg: dict) -> tuple[int, int]:
    """DiT patch size on top of VAE compression: (spatial, temporal).

    Wan writes it as [t, h, w]; LTX as separate scalars. Absent means 1, which
    is the identity and therefore safe to default — unlike hidden size, a
    missing patch size has an unambiguous meaning.
    """
    raw = cfg.get("patch_size")
    temporal = _as_int(cfg.get("patch_size_t")) or 1
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        return int(max(raw[1], raw[2])), int(raw[0])
    spatial = _as_int(raw) or 1
    return spatial, temporal


def arch_from_configs(transformer: dict, vae: dict) -> tuple[Arch | None, list[str]]:
    """Architecture as the model publishes it. None when it cannot be read."""
    notes: list[str] = []

    hidden, key = _first(transformer, HIDDEN_KEYS)
    hidden = _as_int(hidden)
    if hidden:
        notes.append(f"hidden={hidden} from {key!r}")
    else:
        heads, hk = _first(transformer, HEAD_KEYS)
        head_dim, dk = _first(transformer, HEAD_DIM_KEYS)
        heads, head_dim = _as_int(heads), _as_int(head_dim)
        if heads and head_dim:
            hidden = heads * head_dim
            notes.append(f"hidden={hidden} from {hk!r}x{dk!r}")

    layers, lk = _first(transformer, LAYER_KEYS)
    layers = _as_int(layers)
    if layers:
        notes.append(f"layers={layers} from {lk!r}")

    patch_spatial, patch_temporal = _patch(transformer)
    notes.append(f"patch={patch_spatial}s/{patch_temporal}t")

    vae_spatial = _as_int(_first(vae, VAE_SPATIAL_KEYS)[0])
    vae_temporal = _as_int(_first(vae, VAE_TEMPORAL_KEYS)[0])
    vae_channels = _as_int(_first(vae, VAE_CHANNEL_KEYS)[0])
    if vae_spatial:
        notes.append(f"vae_spatial={vae_spatial}")
    if vae_temporal:
        notes.append(f"vae_temporal={vae_temporal}")
    if vae_channels:
        notes.append(f"vae_channels={vae_channels}")

    missing = [
        name
        for name, value in (
            ("hidden", hidden),
            ("layers", layers),
            ("vae_spatial", vae_spatial),
            ("vae_temporal", vae_temporal),
            ("vae_channels", vae_channels),
        )
        if not value
    ]
    if missing:
        notes.append(f"UNREADABLE: {', '.join(missing)}")
        return None, notes

    return (
        Arch(
            hidden=hidden,
            layers=layers,
            patch_spatial=patch_spatial,
            patch_temporal=patch_temporal,
            vae_spatial=vae_spatial,
            vae_temporal=vae_temporal,
            vae_channels=vae_channels,
        ),
        notes,
    )


def pick_config_paths(paths: list[str]) -> tuple[str | None, str | None]:
    """The transformer's and VAE's config.json out of the repository listing."""
    transformer = next(
        (p for p in paths if p.startswith("transformer/")), None
    )
    vae = next((p for p in paths if p.startswith("vae/")), None)
    return transformer, vae


def project(row: mr.Row) -> list[dict]:
    """Every configuration for one measured model. Empty when unmeasurable."""
    arch = row.configs.get("arch")
    roles = row.measurement.get("roles") or {}
    if not arch or not roles:
        return []
    width, height = row.candidate.target_shape or mr.ONIQ_SHAPE
    shape = Shape(width=width, height=height, frames=ONIQ_FRAMES)
    out = []
    for label, config in CONFIGS:
        plan = vram.plan(roles=roles, arch=arch, shape=shape, config=config)
        plan["label"] = label
        plan["shape"] = f"{width}x{height}x{ONIQ_FRAMES}"
        plan["fits_a5000"] = vram.fits(plan["peak_bytes"], 24)
        plan["minimum_gpu"] = vram.minimum_gpu(plan["peak_bytes"])
        out.append(plan)
    return out


def gather(token, get=mr._get) -> list[mr.Row]:
    """Resolve, measure and read the architecture of every candidate."""
    listings: dict[str, list[str]] = {}
    rows: list[mr.Row] = []

    for candidate in mr.CANDIDATES:
        if candidate.author not in listings:
            try:
                listings[candidate.author] = mr.repo_ids(
                    mr.catalogue(candidate.author, token, get)
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  ! {candidate.author} listing failed: {mr._why(exc)}")
                listings[candidate.author] = []

        ids = listings[candidate.author]
        matches = mr.resolve(candidate, ids)
        row = mr.Row(candidate=candidate, alternates=matches)
        row.repo = mr.preferred(candidate, matches)
        if row.repo:
            row.measurement = mr.measure(row.repo, token, get)
            paths = row.measurement.get("config_paths") or []
            tpath, vpath = pick_config_paths(paths)
            tcfg, vcfg = {}, {}
            for path, sink in ((tpath, "t"), (vpath, "v")):
                if not path:
                    continue
                try:
                    cfg = mr.fetch_config(
                        row.repo, row.measurement.get("revision"), path, token, get
                    )
                    if sink == "t":
                        tcfg = cfg
                    else:
                        vcfg = cfg
                except Exception as exc:  # noqa: BLE001
                    print(f"  ! {row.repo}:{path} unreadable ({mr._why(exc)})")
            arch, notes = arch_from_configs(tcfg, vcfg)
            row.configs = {"arch": arch, "notes": notes}
        rows.append(row)
    return rows


def _print_row(row: mr.Row) -> None:
    c = row.candidate
    print("")
    print(f"### {c.label}  [{c.key} -> {c.decision}]")
    if c.note:
        print(f"    {c.note}")
    if not row.repo:
        print("    NOT-PUBLISHED under this predicate — nothing matched the "
              "publisher's real listing, and the predicate is not loosened "
              "to manufacture a match.")
        return

    m = row.measurement
    print(f"    repo      {row.repo}")
    if len(row.alternates) > 1:
        print(f"    also      {[a for a in row.alternates if a != row.repo]}")
    if m.get("verdict") == "UNREADABLE":
        print(f"    UNREADABLE {m.get('detail')}")
        return

    print(f"    revision  {m.get('revision')}")
    print(f"    licence   {m.get('licence')!r}"
          + (f"  name={m.get('licence_name')!r}" if m.get("licence_name") else "")
          + (f"  link={m.get('licence_link')!r}" if m.get("licence_link") else "")
          + ("  GATED" if m.get("gated") else ""))
    print(f"    diffusers {m.get('is_diffusers')}")
    roles = m.get("roles") or {}
    if roles:
        parts = ", ".join(f"{k}={vram.gib(v)}GiB" for k, v in sorted(roles.items()))
        print(f"    weights   {vram.gib(m.get('total_weight_bytes', 0))}GiB total  ({parts})")
    if "transformer_2" in roles:
        print("    NOTE      two experts present — this is a mixture, and only "
              "one expert is resident per denoising stage. The label 'A14B' "
              "does not describe these bytes.")
    for role, files in sorted((m.get("role_files") or {}).items()):
        for name, size in files[:6]:
            print(f"    {role:14s} {name}  {vram.gib(size)}GiB")
        if len(files) > 6:
            print(f"    {role:14s} ... and {len(files) - 6} more")
    for f in (m.get("single_files") or [])[:8]:
        print(f"    file      {f['name']}  {vram.gib(f['bytes'])}GiB  "
              f"[{mr.variant_of(f['name'])}]")

    notes = row.configs.get("notes") or []
    print(f"    arch      {'; '.join(notes) if notes else 'no config read'}")

    plans = project(row)
    if not plans:
        print("    ARCH-INCOMPLETE — no VRAM projection. A defaulted "
              "architecture would produce a confident wrong number.")
        return
    print(f"    projection at {plans[0]['shape']} (measured weights, "
          f"projected working set):")
    for p in plans:
        verdict = "A5000 OK" if p["fits_a5000"] else "A5000 NO"
        print(
            f"      {p['label']:30s} peak={vram.gib(p['peak_bytes']):6.2f}GiB  "
            f"(resident={vram.gib(p['resident_weight_bytes']):5.2f} + "
            f"{p['binding_stage']}={vram.gib(max(p['projected_activation_bytes'], p['projected_decode_bytes'])):5.2f})  "
            f"{verdict}  min={p['minimum_gpu'] or 'NONE LISTED'}"
        )


def report(token, get=mr._get) -> tuple[int, list[mr.Row]]:
    if not token:
        print("BLOCKED: no credential. Anonymous listing cannot distinguish a "
              "gated repository from one that does not exist, so a run without "
              "a token would report absences it cannot support.")
        return 2, []

    print("=== VIDEO MODEL BENCHMARK — measured half, $0, no GPU ===")
    print("Weights are MEASURED from the registry's file listing. Working-set "
          "and decode figures are PROJECTED by validation/vram.py with stated "
          "constants. Runtime, cost and quality are PENDING-PROBE: they cannot "
          "be computed, only observed.")

    rows = gather(token, get)
    for row in rows:
        _print_row(row)

    print("")
    print("=== A5000 SUMMARY (24GB, ONIQ's current card) ===")
    runnable, blocked, unknown = [], [], []
    for row in rows:
        plans = project(row)
        if not plans:
            unknown.append(row.candidate.label)
        elif any(p["fits_a5000"] for p in plans):
            best = next(p for p in plans if p["fits_a5000"])
            runnable.append(f"{row.candidate.label} — via {best['label']}")
        else:
            cheapest = min(plans, key=lambda p: p["peak_bytes"])
            blocked.append(
                f"{row.candidate.label} — best case {vram.gib(cheapest['peak_bytes'])}GiB "
                f"needs {cheapest['minimum_gpu'] or 'more than any listed card'}"
            )
    for name, group in (("PROJECTED TO FIT", runnable),
                        ("PROJECTED NOT TO FIT", blocked),
                        ("NOT PROJECTABLE", unknown)):
        print(f"  {name}:")
        for line in group or ["    (none)"]:
            print(f"    {line}")

    print("")
    print("NOT A DEPLOYABILITY VERDICT. Every 'fits' above is arithmetic, and "
          "arithmetic has never once loaded a checkpoint. The next step is one "
          "bounded probe per surviving row that reports torch.cuda peak memory "
          "and wall-clock from a real load — which is spend, and the owner's "
          "call.")
    return 0, rows


def main(argv) -> int:
    from validation.hf_auth import TOKEN_VAR

    code, rows = report(os.environ.get(TOKEN_VAR))
    out = os.environ.get("BENCH_JSON")
    if out and rows:
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(
                [
                    {
                        "key": r.candidate.key,
                        "decision": r.candidate.decision,
                        "repo": r.repo,
                        "revision": r.measurement.get("revision"),
                        "licence": r.measurement.get("licence"),
                        "roles": r.measurement.get("roles"),
                        "plans": [
                            {k: v for k, v in p.items() if k != "shape"}
                            for p in project(r)
                        ],
                    }
                    for r in rows
                ],
                fh,
                indent=1,
            )
    return code


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
