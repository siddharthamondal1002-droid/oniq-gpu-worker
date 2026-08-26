import inspect
import os
from decimal import Decimal

import pytest

import contract
import runpod_client as rp
from validation import admission


# ---------------------------------------------------------------- reserve


def test_reservation_rounds_up_to_the_cent():
    # A cent too little silently defeats the ceiling: 900s at $0.22/h is
    # $0.055 exactly, and the reservation must be $0.06, never $0.05.
    assert admission.reserve_usd("0.22", 900) == Decimal("0.06")


def test_exact_cent_stays_exact():
    assert admission.reserve_usd("0.20", 900) == Decimal("0.05")


def test_fractional_cent_always_ceils():
    assert admission.reserve_usd("0.201", 900) == Decimal("0.06")


def test_null_price_is_no_capacity_not_free():
    with pytest.raises(admission.AdmissionRefused) as exc:
        admission.reserve_usd(None, 900)
    assert exc.value.code == "gpu-unpriced"


def test_zero_and_negative_price_refused():
    for bad in (0, -1, "0", "-0.5"):
        with pytest.raises(admission.AdmissionRefused) as exc:
            admission.reserve_usd(bad, 900)
        assert exc.value.code == "gpu-unpriced"


def test_nonpositive_runtime_refused():
    with pytest.raises(admission.AdmissionRefused):
        admission.reserve_usd("0.30", 0)


def test_no_default_price_argument_exists():
    # There is deliberately no default, so a stale figure cannot be
    # reached by forgetting one.
    sig = inspect.signature(admission.reserve_usd)
    assert sig.parameters["price_per_hour"].default is inspect.Parameter.empty
    sig2 = inspect.signature(admission.admit)
    assert sig2.parameters["price_per_hour"].default is inspect.Parameter.empty


def test_no_historical_price_in_any_source_file():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    offenders = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in {".git", "tests", "__pycache__", ".pytest_cache"}
        ]
        for name in filenames:
            if name.endswith((".py", ".yml", ".yaml", ".txt", "Dockerfile")):
                path = os.path.join(dirpath, name)
                text = open(path, encoding="utf-8", errors="ignore").read()
                for stale in ("0.22", "0.27"):
                    if stale in text:
                        offenders.append((os.path.relpath(path, root), stale))
    assert offenders == []


# ---------------------------------------------------------------- admit


def _admit(**overrides):
    args = {
        "gpu_name": admission.TARGET_GPU,
        "vram_gb": 24,
        "runtime_seconds": 900,
        "price_per_hour": "0.30",
    }
    args.update(overrides)
    return admission.admit(**args)


def test_admit_happy_path():
    res = _admit()
    assert res.reserved_usd == Decimal("0.08")  # 0.30*900/3600=0.075 -> up
    assert res.runtime_seconds == 900
    assert res.gpu == admission.TARGET_GPU


def test_reservation_charges_the_full_ceiling_not_the_expectation():
    # A job expected to take 20s reserves the same money as one that
    # wedges for the full window: at admission they are indistinguishable.
    short = _admit(runtime_seconds=20)
    full = _admit(runtime_seconds=900)
    assert short.reserved_usd == full.reserved_usd
    assert short.runtime_seconds == admission.RUNTIME_CEILING_SECONDS


def test_gpu_not_on_allow_list_refused_first():
    with pytest.raises(admission.AdmissionRefused) as exc:
        _admit(gpu_name="NVIDIA H100 80GB HBM3", vram_gb=80,
               price_per_hour=None, runtime_seconds=5000)
    assert exc.value.code == "gpu-type-not-allowed"


def test_vram_is_checked_before_price():
    with pytest.raises(admission.AdmissionRefused) as exc:
        _admit(vram_gb=10, price_per_hour=None)
    assert exc.value.code == "insufficient-vram"


def test_runtime_is_checked_before_price():
    with pytest.raises(admission.AdmissionRefused) as exc:
        _admit(runtime_seconds=1000, price_per_hour=None)
    assert exc.value.code == "runtime-exceeds-ceiling"


def test_unpriced_is_checked_before_cap():
    with pytest.raises(admission.AdmissionRefused) as exc:
        _admit(price_per_hour=None)
    assert exc.value.code == "gpu-unpriced"


def test_over_cap_refused():
    with pytest.raises(admission.AdmissionRefused) as exc:
        _admit(price_per_hour="2.04")  # 0.51 reserved
    assert exc.value.code == "over-job-cap"


def test_exactly_at_cap_admitted():
    assert _admit(price_per_hour="2.00").reserved_usd == Decimal("0.50")


def test_runtime_above_ceiling_refused_not_clamped():
    with pytest.raises(admission.AdmissionRefused) as exc:
        _admit(runtime_seconds=901)
    assert exc.value.code == "runtime-exceeds-ceiling"


def test_missing_vram_refused():
    with pytest.raises(admission.AdmissionRefused) as exc:
        _admit(vram_gb=None)
    assert exc.value.code == "insufficient-vram"


# ---------------------------------------------------------------- constants


def test_job_cap_is_fifty_cents_and_its_own_capability():
    assert admission.JOB_CAP_USD == Decimal("0.50")


def test_runtime_ceiling_agrees_with_the_contract():
    assert admission.RUNTIME_CEILING_SECONDS == 900
    assert admission.RUNTIME_CEILING_SECONDS == contract.RUNTIME_CEILING_SECONDS


def test_target_is_the_a5000_and_allow_list_is_closed():
    # Owner directive 2026-08-26 (third card of the day, settled).
    assert admission.TARGET_GPU == "NVIDIA RTX A5000"
    assert set(admission.ALLOWED_GPUS) == {admission.TARGET_GPU}


# ---------------------------------------------------------------- available


def _gpu(gpu_id, mem, secure=True, price="0.31", on_demand="0.31", display=None,
         community=True):
    return {
        "id": gpu_id,
        "display_name": display or gpu_id,
        "memory_gb": mem,
        "secure_cloud": secure,
        "secure_price": price,
        "on_demand_price": on_demand,
        # Tri-state, like the parser emits: True/False are explicit
        # provider booleans; None models a missing/malformed field.
        "community_cloud": community,
    }


def test_available_target_is_returned():
    cat = [_gpu("NVIDIA RTX A5000", 24, display="RTX A5000")]
    assert admission.require_available(cat)["memory_gb"] == 24


def test_missing_target_raises_with_alternatives():
    cat = [
        _gpu("NVIDIA GeForce RTX 4090", 24, price="0.40"),
        _gpu("Small Card", 8, price="0.10"),
    ]
    with pytest.raises(admission.UnavailableGpu) as exc:
        admission.require_available(cat)
    alts = [a["id"] for a in exc.value.alternatives]
    assert alts == ["NVIDIA GeForce RTX 4090"]  # allocatable, >=24GB only


def test_unpriced_target_is_unavailable_never_free():
    cat = [_gpu("NVIDIA RTX A5000", 24, price=None, on_demand=None)]
    with pytest.raises(admission.UnavailableGpu):
        admission.require_available(cat)


def test_list_price_without_capacity_is_unavailable():
    # The A5000's exact state: a securePrice in the catalogue while
    # lowestPrice is null — a list price is not capacity.
    cat = [_gpu("NVIDIA RTX A5000", 24, price="0.27", on_demand=None)]
    with pytest.raises(admission.UnavailableGpu):
        admission.require_available(cat)


def test_unavailable_never_substitutes():
    cat = [_gpu("NVIDIA GeForce RTX 4090", 24, price="0.40")]
    with pytest.raises(admission.UnavailableGpu):
        admission.require_available(cat)  # a 4090 in hand changes nothing


def test_community_only_target_is_not_secure_capacity():
    cat = [_gpu("NVIDIA RTX A5000", 24, secure=False)]
    with pytest.raises(admission.UnavailableGpu):
        admission.require_available(cat)


def test_secure_only_card_admits_on_its_secure_price_alone():
    # The L4 shape, verbatim semantics from run #14's raw bytes: no
    # community market exists, so a null lowestPrice carries no signal.
    cat = [_gpu("NVIDIA L4", 24, price=0.49, on_demand=None, community=False)]
    entry = admission.require_available(cat, target="NVIDIA L4")
    assert entry["secure_price"] == 0.49
    # Reservation math on the eligible price; admit()'s closed allow-list
    # is the TARGET card's own gate and is tested with the target.
    assert str(admission.reserve_usd(entry["secure_price"],
                                     admission.RUNTIME_CEILING_SECONDS)) == "0.13"


def test_secure_only_card_without_a_secure_price_is_unavailable():
    cat = [_gpu("NVIDIA L4", 24, price=None, on_demand=None, community=False)]
    with pytest.raises(admission.UnavailableGpu):
        admission.require_available(cat, target="NVIDIA L4")


def test_secure_only_card_over_cap_is_refused_at_admit():
    # Eligibility is not admission: a secure-only card can be ELIGIBLE at
    # a price the cap then refuses. The cap fires on the TARGET name (the
    # closed allow-list would refuse any other name even earlier).
    cat = [_gpu("NVIDIA L4", 24, price=2.04, on_demand=None, community=False)]
    entry = admission.require_available(cat, target="NVIDIA L4")  # eligible
    with pytest.raises(admission.AdmissionRefused) as exc:
        admission.admit(
            gpu_name=admission.TARGET_GPU, vram_gb=entry["memory_gb"],
            runtime_seconds=admission.RUNTIME_CEILING_SECONDS,
            price_per_hour=entry["secure_price"],
        )
    assert exc.value.code == "over-job-cap"


def test_missing_or_malformed_community_flag_rejects_conservatively():
    # None models a missing field; a string models a malformed one. The
    # parser only ever emits True/False/None, and None must never slip
    # into the lenient secure-only branch.
    for bad in (None, "false"):
        cat = [_gpu("NVIDIA L4", 24, price=0.49, on_demand=None, community=bad)]
        with pytest.raises(admission.UnavailableGpu):
            admission.require_available(cat, target="NVIDIA L4")


def test_community_card_still_requires_the_market_signal():
    # The A5000 lesson, unweakened: a community-market card with a secure
    # list price but a null lowestPrice is NOT capacity.
    cat = [_gpu("NVIDIA L4", 24, price=0.49, on_demand=None, community=True)]
    with pytest.raises(admission.UnavailableGpu):
        admission.require_available(cat, target="NVIDIA L4")


def test_matching_is_on_id_never_display_name():
    # displayName carries the short name in real payloads; a catalogue
    # whose ids are short names must NOT satisfy the target.
    cat = [_gpu("RTX A5000", 24, display="NVIDIA RTX A5000")]
    with pytest.raises(admission.UnavailableGpu):
        admission.require_available(cat)


# ------------------------------------------------- real payload, 2026-08-25


def _real_catalogue():
    import json
    import os

    import runpod_client as rp

    path = os.path.join(
        os.path.dirname(__file__), "fixtures", "gpu_types_actual.json"
    )
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    return [rp.parse_gpu_type(g) for g in doc["data"]["gpuTypes"]]


A5000_FRESH_RAW = {
    # Verbatim provider bytes, gpu-validation run #15 (2026-08-26T17:36Z):
    # the A5000's community-market signal is live again.
    "id": "NVIDIA RTX A5000", "displayName": "RTX A5000", "memoryInGb": 24,
    "secureCloud": True, "communityCloud": True,
    "securePrice": 0.27, "communityPrice": 0.16,
    "lowestPrice": {"uninterruptablePrice": 0.16, "minimumBidPrice": 0.16},
}


def test_real_payload_2026_08_25_a5000_still_refuses():
    # The recorded 2026-08-25 bytes are the LESSON bytes: securePrice with
    # a null lowestPrice on a community-market card is not capacity.
    with pytest.raises(admission.UnavailableGpu):
        admission.require_available(_real_catalogue())


def test_real_payload_fresh_a5000_is_available_on_secure_cloud():
    entry = admission.require_available([rp.parse_gpu_type(A5000_FRESH_RAW)])
    assert entry["id"] == "NVIDIA RTX A5000"
    assert entry["memory_gb"] == 24
    assert entry["secure_price"] == 0.27


def test_real_payload_admits_the_a5000_at_seven_cents():
    # Fresh secure price on run #15 bytes: CEIL(0.27 * 900 / 3600, $0.01)
    # = $0.07, under the $0.50 job cap. The reservation math on real
    # numbers, the cheapest card of the day.
    entry = admission.require_available([rp.parse_gpu_type(A5000_FRESH_RAW)])
    res = admission.admit(
        gpu_name=entry["id"],
        vram_gb=entry["memory_gb"],
        runtime_seconds=900,
        price_per_hour=entry["secure_price"],
    )
    assert res.reserved_usd == Decimal("0.07")
    assert res.reserved_usd <= admission.JOB_CAP_USD


def test_real_payload_a5000_is_still_not_provisionable():
    with pytest.raises(admission.UnavailableGpu):
        admission.require_available(
            _real_catalogue(), target="NVIDIA RTX A5000"
        )


# ---------------------------------------------------------------- endpoint


def test_endpoint_zero_one_required():
    admission.check_endpoint_config(0, 1)


@pytest.mark.parametrize("mn,mx", [(1, 1), (0, 2), (None, None), (0, 0)])
def test_endpoint_other_configs_refused(mn, mx):
    with pytest.raises(admission.AdmissionRefused) as exc:
        admission.check_endpoint_config(mn, mx)
    assert exc.value.code == "endpoint-config-refused"


# ---------------------------------------------------------------- spend gate


def test_spend_gate_requires_both():
    admission.check_spend_gate("SPEND", True)


@pytest.mark.parametrize(
    "spend,approval",
    [
        ("SPEND", False),
        ("SPEND", None),
        ("SPEND", "yes"),  # approval must be the literal True
        ("SPEND", 1),
        ("spend", True),  # case-sensitive
        ("", True),
        (None, True),
        ("", False),
    ],
)
def test_spend_gate_refuses_anything_partial(spend, approval):
    with pytest.raises(admission.AdmissionRefused) as exc:
        admission.check_spend_gate(spend, approval)
    assert exc.value.code == "spend-not-approved"
    assert "BLOCKED — DO NOT SPEND" in exc.value.message
