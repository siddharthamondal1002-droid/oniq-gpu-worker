import pytest
from decimal import Decimal

from validation import gpu_matrix as gm
from validation import vram


CATALOGUE = [
    {"id": "NVIDIA RTX A5000", "display_name": "RTX A5000",
     "secure_price": 0.27, "community_price": None, "on_demand_price": 0.29},
    {"id": "NVIDIA GeForce RTX 4090", "display_name": "RTX 4090",
     "secure_price": 0.69, "community_price": 0.34, "on_demand_price": 0.44},
    {"id": "NVIDIA A100 80GB PCIe", "display_name": "A100 80GB",
     "secure_price": 1.64, "community_price": None, "on_demand_price": None},
    {"id": "NVIDIA H100 PCIe", "display_name": "H100",
     "secure_price": None, "community_price": None, "on_demand_price": None},
]


# ------------------------------------------------------------------ matching


def test_cards_match_by_substring_because_providers_rename_them():
    """An exact-name miss would silently drop a card from the matrix and read
    as 'unavailable' when it is merely spelled differently."""
    assert gm.match(CATALOGUE, ("rtx a5000",))[0]["id"] == "NVIDIA RTX A5000"
    assert gm.match(CATALOGUE, ("rtx 4090",))[0]["display_name"] == "RTX 4090"
    assert gm.match(CATALOGUE, ("rtx 5090",)) == []


def test_every_card_the_owner_listed_has_a_matcher():
    listed = {name for name, _, _ in gm.CARDS}
    assert listed == {
        "RTX A5000 24GB", "RTX 4090 24GB", "RTX 5090 32GB", "A6000 48GB",
        "A100 40GB", "A100 80GB", "H100 80GB",
    }


# ------------------------------------------------------------------- pricing


def test_cheapest_hourly_takes_the_lowest_live_quote():
    price, source = gm.cheapest_hourly(gm.match(CATALOGUE, ("rtx 4090",)))
    assert price == Decimal("0.34")
    assert "community" in source


def test_a_card_with_no_live_price_is_no_capacity_never_free():
    """A missing price has bitten this repository before. Zero is a price;
    absent is not."""
    price, source = gm.cheapest_hourly(gm.match(CATALOGUE, ("h100",)))
    assert price is None
    assert source == "NO CAPACITY"


def test_a_zero_or_negative_quote_is_refused():
    price, _ = gm.cheapest_hourly([{"id": "x", "secure_price": 0}])
    assert price is None


def test_an_unparseable_quote_is_skipped_rather_than_crashing():
    price, _ = gm.cheapest_hourly(
        [{"id": "x", "secure_price": "n/a", "community_price": 0.5}]
    )
    assert price == Decimal("0.5")


def test_spot_prices_are_never_quoted():
    """A preemptible rental that can be reclaimed mid-generation is not a
    substrate for a paid render, and quoting it would understate the bill."""
    price, _ = gm.cheapest_hourly(
        [{"id": "x", "secure_price": 2.0, "spot_price": 0.01}]
    )
    assert price == Decimal("2.0")


def test_live_cards_reports_every_listed_card_even_when_unpriced():
    rows = gm.live_cards(CATALOGUE)
    assert len(rows) == len(gm.CARDS)
    unpriced = [r for r in rows if r["usd_per_hour"] is None]
    assert {r["card"] for r in unpriced} >= {"RTX 5090 32GB", "H100 80GB"}


# ------------------------------------------------------------ card selection


def test_the_cheapest_fitting_card_wins_not_the_smallest():
    """The brief's warning is against paying MORE for capacity that is not
    needed, not against taking a bargain on a larger card."""
    rows = gm.live_cards(CATALOGUE)
    chosen = gm.cheapest_for(int(10 * vram.GIB), rows)
    assert chosen["card"] == "RTX A5000 24GB"


def test_a_peak_over_24gb_skips_the_24gb_cards():
    rows = gm.live_cards(CATALOGUE)
    chosen = gm.cheapest_for(int(30 * vram.GIB), rows)
    assert chosen["card"] == "A100 80GB"


def test_an_unpriced_card_is_not_a_candidate_at_any_size():
    """H100 has capacity in the listing and no price. It must not be selected
    just because it is large."""
    rows = gm.live_cards(CATALOGUE)
    chosen = gm.cheapest_for(int(70 * vram.GIB), rows)
    assert chosen["card"] == "A100 80GB", "never the unpriced H100"


def test_nothing_fits_returns_none_rather_than_the_biggest_card():
    rows = gm.live_cards(CATALOGUE)
    assert gm.cheapest_for(int(500 * vram.GIB), rows) is None


# -------------------------------------------------------------------- matrix


def _plan(label, peak_gib, precision="bf16"):
    return {"label": label, "peak_bytes": int(peak_gib * vram.GIB),
            "precision": precision}


def test_the_matrix_picks_the_configuration_reaching_the_cheapest_card():
    rows = gm.live_cards(CATALOGUE)
    matrix = gm.summarise({
        "Model X": [
            _plan("as published", 60),
            _plan("bf16 + offload", 30),
            _plan("fp8 + sequential", 12, "fp8"),
        ]
    }, rows)
    assert matrix[0]["card"] == "RTX A5000 24GB"
    assert matrix[0]["configuration"] == "fp8 + sequential"
    assert matrix[0]["precision"] == "fp8"


def test_runtime_cost_and_quality_are_never_computed():
    """An hourly rate times a guessed duration is an invented billing number."""
    rows = gm.live_cards(CATALOGUE)
    matrix = gm.summarise({"Model X": [_plan("cheap", 10)]}, rows)
    assert matrix[0]["runtime"] == "PENDING-PROBE"
    assert matrix[0]["usd_per_clip"] == "PENDING-PROBE"
    assert matrix[0]["quality"] == "PENDING-PROBE"


def test_a_model_no_live_card_fits_is_reported_not_dropped():
    rows = gm.live_cards(CATALOGUE)
    matrix = gm.summarise({"Enormous": [_plan("as published", 400)]}, rows)
    assert matrix[0]["verdict"] == "NO LIVE CARD FITS"
    assert matrix[0]["card"] is None


def test_print_paths_do_not_crash_on_unpriced_or_unfitting_rows(capsys):
    rows = gm.live_cards(CATALOGUE)
    gm.print_cards(rows)
    gm.print_matrix(gm.summarise({"Enormous": [_plan("as published", 400)]}, rows))
    out = capsys.readouterr().out
    assert "NO CAPACITY" in out
    assert "PENDING-PROBE" in out
