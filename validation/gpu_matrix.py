"""The hardware half of the benchmark: which card, at what LIVE price.

Owner brief 2026-08-29: "Create a measured hardware matrix — MODEL | PRECISION
| VRAM PEAK | GPU | RUNTIME | COST | QUALITY. Evaluate the minimum viable GPU.
Only include a GPU if the chosen model/configuration actually requires it. Do
not pay for an A100/H100 merely because a model is labelled 14B."

That last sentence is the whole design. This module never starts from a card
and asks what runs on it; it starts from a projected VRAM peak and asks for the
cheapest card that clears it. A GPU appears in the output only where some
configuration actually needs it.

PRICES ARE READ LIVE, ALWAYS. Standing rule since the first spend gate: a quote
remembered from an earlier run is not a price, and a card that has fallen out
of stock has no price at all. A missing price parses as None and is reported as
NO CAPACITY — never as free, and never as the last figure that happened to
work.

WHAT THIS CANNOT DO. Cost per clip needs a runtime, and runtime cannot be
computed — only observed. Every dollar figure here is per HOUR, live, with the
per-clip column left PENDING-PROBE. Multiplying an hourly rate by a guessed
duration would produce exactly the kind of invented billing number the owner
has ruled out repeatedly.
"""

from __future__ import annotations

from decimal import Decimal

from validation import vram

# The owner's list, mapped to the catalogue's canonical `id` strings. Matching
# is by substring against the live listing rather than by exact name, because
# the provider renames cards ("NVIDIA GeForce RTX 4090" vs "RTX 4090") and an
# exact-match miss would silently drop a card from the matrix — reading as
# "unavailable" when it is merely spelled differently.
CARDS: tuple[tuple[str, int, tuple[str, ...]], ...] = (
    ("RTX A5000 24GB", 24, ("a5000",)),
    ("RTX 4090 24GB", 24, ("4090",)),
    ("RTX 5090 32GB", 32, ("5090",)),
    ("A6000 48GB", 48, ("a6000",)),
    ("A100 40GB", 40, ("a100",)),
    ("A100 80GB", 80, ("a100",)),
    ("H100 80GB", 80, ("h100",)),
)

# How far a listing's reported memory may sit from the row's nominal capacity
# and still be that card. Vendors report 40960 MiB as 40 and 81920 as 80, and
# occasionally shade by a gigabyte; anything further apart is a different card.
MEMORY_TOLERANCE_GIB = 4


def match(catalogue, needles: tuple[str, ...], memory_gib: int | None = None):
    """Live entries of this family AND this capacity.

    THE NAME IS NOT THE CARD. The first live pull matched "A100 40GB" to an
    "NVIDIA A100 80GB PCIe" listing and "A100 80GB" to an "A100-SXM4-40GB" one,
    because both display names carry the family string — so each A100 row was
    priced from the other one's hardware. On a page whose entire purpose is
    choosing what to rent, that is the worst error available: it reports 80 GB
    at $1/h when the quote belongs to a 40 GB card.

    So the family string only narrows the search and the capacity the provider
    REPORTS decides. Same discipline as the model side: measure the bytes, never
    read them off the label.
    """
    out = []
    for entry in catalogue or []:
        haystack = f"{entry.get('id') or ''} {entry.get('display_name') or ''}".lower()
        if not any(n in haystack for n in needles):
            continue
        if memory_gib is not None:
            reported = entry.get("memory_gb")
            if not isinstance(reported, (int, float)):
                # No reported capacity means the card cannot be confirmed as
                # this row's, and an unconfirmed card is not a price.
                continue
            if abs(reported - memory_gib) > MEMORY_TOLERANCE_GIB:
                continue
        out.append(entry)
    return out


def cheapest_hourly(entries) -> tuple[Decimal | None, str]:
    """The lowest live on-demand price across matching entries, and its source.

    Spot/bid prices are deliberately NOT considered. A preemptible rental that
    can be reclaimed mid-generation is not a substrate for a paid render, and
    quoting its price would understate what ONIQ would actually pay.
    """
    best, source = None, "NO CAPACITY"
    for entry in entries:
        for field, label in (
            ("secure_price", "secure"),
            ("community_price", "community"),
            ("on_demand_price", "on-demand"),
        ):
            raw = entry.get(field)
            if raw is None:
                continue
            try:
                price = Decimal(str(raw))
            except Exception:  # noqa: BLE001
                continue
            if price <= 0:
                continue
            if best is None or price < best:
                best, source = price, f"{label} / {entry.get('id')}"
    return best, source


def live_cards(catalogue) -> list[dict]:
    """The owner's card list, priced from the live catalogue."""
    rows = []
    for name, gib, needles in CARDS:
        entries = match(catalogue, needles, gib)
        price, source = cheapest_hourly(entries)
        rows.append({
            "card": name,
            "vram_gib": gib,
            "usd_per_hour": price,
            "source": source,
            "listed": len(entries),
        })
    return rows


def cheapest_for(peak_bytes: int, rows: list[dict]) -> dict | None:
    """The cheapest LIVE card that clears this peak.

    Cheapest, not smallest: a larger card that happens to be cheaper per hour is
    the better buy, and the brief's warning is against paying MORE for capacity
    that is not needed — not against taking a bargain. A card with no live price
    is not a candidate at any size.
    """
    affordable = [
        r for r in rows
        if r["usd_per_hour"] is not None
        and vram.fits(peak_bytes, r["vram_gib"])
    ]
    if not affordable:
        return None
    return min(affordable, key=lambda r: (r["usd_per_hour"], r["vram_gib"]))


def summarise(plans_by_model: dict, rows: list[dict]) -> list[dict]:
    """One matrix line per model: its cheapest deployable configuration.

    A model is represented by the configuration that reaches the cheapest card,
    because that is the question actually being asked — not "how big can this
    get" but "what would ONIQ have to rent to run it".
    """
    out = []
    for label, plans in plans_by_model.items():
        best = None
        for plan in plans:
            card = cheapest_for(plan["peak_bytes"], rows)
            if not card:
                continue
            if best is None or card["usd_per_hour"] < best["card"]["usd_per_hour"]:
                best = {"plan": plan, "card": card}
        out.append({
            "model": label,
            "configuration": best["plan"]["label"] if best else None,
            "precision": best["plan"]["precision"] if best else None,
            "peak_gib": vram.gib(best["plan"]["peak_bytes"]) if best else None,
            "card": best["card"]["card"] if best else None,
            "usd_per_hour": best["card"]["usd_per_hour"] if best else None,
            "runtime": "PENDING-PROBE",
            "usd_per_clip": "PENDING-PROBE",
            "quality": "PENDING-PROBE",
            "verdict": "deployable" if best else "NO LIVE CARD FITS",
        })
    return out


def print_cards(rows: list[dict]) -> None:
    print("=== LIVE GPU PRICES (quoted now, never remembered) ===")
    for row in rows:
        price = (f"${row['usd_per_hour']}/h" if row["usd_per_hour"] is not None
                 else "NO CAPACITY")
        print(f"  {row['card']:18s} {row['vram_gib']:3d}GiB  {price:16s} "
              f"{row['listed']} live listing(s)  [{row['source']}]")
    missing = [r["card"] for r in rows if r["usd_per_hour"] is None]
    if missing:
        print(f"  NOT PRICEABLE RIGHT NOW: {', '.join(missing)} — reported as "
              f"no capacity, never as free.")


def print_matrix(matrix: list[dict]) -> None:
    print("")
    print("=== MODEL | PRECISION | VRAM PEAK | GPU | RUNTIME | COST | QUALITY ===")
    for row in matrix:
        if not row["card"]:
            print(f"  {row['model']:26s} NO LIVE CARD FITS ANY CONFIGURATION")
            continue
        print(
            f"  {row['model']:26s} {row['precision']:5s} "
            f"{row['peak_gib']:6.2f}GiB  {row['card']:16s} "
            f"${row['usd_per_hour']}/h  runtime={row['runtime']}  "
            f"cost/clip={row['usd_per_clip']}  quality={row['quality']}"
        )
    print("")
    print("RUNTIME, COST/CLIP and QUALITY stay PENDING-PROBE by construction. "
          "An hourly rate times a guessed duration is an invented billing "
          "number, and only a real generation can fill those three.")


def main(argv) -> int:
    """Live card prices only. Runs in the read-only discovery job, which
    already holds the provider credential and already reads this catalogue.

    Deliberately NOT folded into model-bench: that job holds no RunPod
    credential at all, which is the fence keeping a free measurement mode from
    turning into a paid one, and a convenient price table is not worth
    spending it.
    """
    import runpod_client

    _raw, parsed = runpod_client.gpu_catalogue()
    rows = live_cards(parsed)
    print_cards(rows)
    print("")
    print("THESE ARE POD-MARKET QUOTES, not ONIQ's serverless rate. The same "
          "card that lists here bills the production endpoint at a different, "
          "separately quoted figure — read that one from the preflight facts "
          "above, never from this table. A pod price is a signal about capacity "
          "and relative cost, not the number that lands on the invoice, and no "
          "rate is written into this file: the historical-price guard in "
          "test_admission forbids it, for exactly this reason.")
    print("")
    print("A card priced here is a card that could be rented NOW. Pair it with "
          "the VRAM peaks from `model-bench` to read the hardware matrix: the "
          "cheapest row that clears a model's peak is that model's minimum "
          "viable GPU, and nothing larger is required to run it.")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
