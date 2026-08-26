"""Financial admission for GPU spend. No network — every decision that can
spend money is a pure function, testable to the cent.

Encodes findings this workstream already paid for:

- reservations round UP to the cent — a cent too little silently defeats
  the ceiling;
- the reservation charges the FULL runtime ceiling, never an expected
  runtime — at admission a 20-second job and a wedged 900-second job are
  indistinguishable, and only one is affordable to be wrong about;
- a null price means NO CAPACITY, never free (the A5000's exact state);
- an unavailable target GPU raises with alternatives instead of silently
  substituting;
- there is deliberately no default price argument, so a stale figure
  cannot be reached by forgetting one — no historical $/h appears
  anywhere in this repository's code.

Refusal order (VRAM before price):
  gpu-type-not-allowed -> insufficient-vram -> runtime-exceeds-ceiling
  -> gpu-unpriced -> over-job-cap
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_UP, Decimal

# GPU is its own ledger capability. It happens to share the SEARCH
# ceiling's number and nothing else; collapsing them would mean a change
# to one silently moved the other.
JOB_CAP_USD = Decimal("0.50")

RUNTIME_CEILING_SECONDS = 900

# Owner directive 2026-08-26 (audio canary, third card of the day): the
# endpoint now offers the RTX A5000 24GB. The 3090's pool flapped through
# three $0 refusals in an hour; the L4 (secure-only) admitted under the
# community/secure-only branch but the owner moved on before a dispatch
# landed. The A5000 is the cheapest card tried today, its secure price is
# well under the cap, and its community-market signal is LIVE again in
# today's pulls — this is the very card whose 2026-08-25 null-lowestPrice
# bytes taught the strict rule, and that strict rule still governs it.
# The AUDIO workload is CPU-by-design and GPU-agnostic; VIDEO remains
# measured on the 3090 only, which is why verify_gpu_success still pins
# "3090" — a video job on this endpoint refuses rather than running on
# an unmeasured card.
TARGET_GPU = "NVIDIA RTX A5000"

# Server-side allow-list: which card ONIQ rents is an owner decision, and
# the allow-list is why "give me 8x H100" cannot be typed at all.
ALLOWED_GPUS = {
    TARGET_GPU: {"min_vram_gb": 24},
}

WORKLOAD_MIN_VRAM_GB = 24

_CENT = Decimal("0.01")


class AdmissionRefused(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class UnavailableGpu(Exception):
    """The target GPU cannot be provisioned right now. Carries priced
    alternatives for the owner to decide on; never substitutes."""

    def __init__(self, target: str, alternatives):
        self.target = target
        self.alternatives = list(alternatives)
        alt_text = (
            "; alternatives: "
            + ", ".join(
                f"{a['display_name']} {a['memory_gb']}GB ${a['secure_price']}/h"
                for a in self.alternatives
            )
            if self.alternatives
            else "; no priced secure-cloud alternative at sufficient VRAM"
        )
        super().__init__(
            f"{target} is not provisionable (unlisted or unpriced){alt_text}"
        )


@dataclass(frozen=True)
class Reservation:
    gpu: str
    runtime_seconds: int
    price_per_hour_usd: Decimal
    reserved_usd: Decimal


def reserve_usd(price_per_hour, runtime_seconds: int) -> Decimal:
    """CEIL(price x runtime / 3600, $0.01). No default price, on purpose."""
    if price_per_hour is None:
        raise AdmissionRefused(
            "gpu-unpriced",
            "price is null — the provider has no capacity to quote, "
            "and null is not free",
        )
    price = Decimal(str(price_per_hour))
    if price <= 0:
        raise AdmissionRefused("gpu-unpriced", "price must be positive")
    if runtime_seconds <= 0:
        raise AdmissionRefused("invalid-runtime", "runtime must be positive")
    exact = price * Decimal(runtime_seconds) / Decimal(3600)
    return exact.quantize(_CENT, rounding=ROUND_UP)


def admit(
    *,
    gpu_name: str,
    vram_gb,
    runtime_seconds: int,
    price_per_hour,
    min_vram_gb: int = WORKLOAD_MIN_VRAM_GB,
) -> Reservation:
    """The caller cannot choose what it costs. Refuses in a fixed order,
    VRAM before price; a runtime above the ceiling is refused, not
    clamped."""
    if gpu_name not in ALLOWED_GPUS:
        raise AdmissionRefused(
            "gpu-type-not-allowed",
            f"GPU type is not on the server-side allow-list",
        )
    if vram_gb is None or vram_gb < min_vram_gb:
        raise AdmissionRefused(
            "insufficient-vram",
            f"workload needs >= {min_vram_gb}GB",
        )
    if runtime_seconds > RUNTIME_CEILING_SECONDS:
        raise AdmissionRefused(
            "runtime-exceeds-ceiling",
            f"runtime {runtime_seconds}s exceeds the "
            f"{RUNTIME_CEILING_SECONDS}s ceiling (refused, not clamped)",
        )
    # The reservation is a TIME budget: charge the full ceiling window,
    # never the caller's expectation of how long the job should take.
    reserved = reserve_usd(price_per_hour, RUNTIME_CEILING_SECONDS)
    if reserved > JOB_CAP_USD:
        raise AdmissionRefused(
            "over-job-cap",
            f"reservation ${reserved} exceeds the ${JOB_CAP_USD} job cap",
        )
    return Reservation(
        gpu=gpu_name,
        runtime_seconds=RUNTIME_CEILING_SECONDS,
        price_per_hour_usd=Decimal(str(price_per_hour)),
        reserved_usd=reserved,
    )


def require_available(catalogue, target: str = TARGET_GPU, min_vram_gb: int = WORKLOAD_MIN_VRAM_GB):
    """Find the target GPU as secure-cloud, ALLOCATABLE capacity — or
    raise UnavailableGpu listing allocatable alternatives. Never
    substitutes.

    Two lessons from real payloads are load-bearing here. Matching is on
    the `id` field (the canonical full name), never `displayName` (the
    short name). And a catalogue LIST price is not capacity: the A5000
    carries a securePrice while its lowestPrice is null for both
    on-demand and spot — RunPod saying it has none to allocate — so
    availability additionally requires a non-null lowestPrice
    (`on_demand_price` in the parsed view)."""
    entry = None
    alternatives = []
    for gpu in catalogue:
        # Provider-semantics branch (owner directive 2026-08-26). The
        # community-market lowestPrice is a capacity proxy ONLY for cards
        # that HAVE a community market: the A5000 lesson (a secure list
        # price with a null lowestPrice is not capacity) stands for
        # communityCloud=true cards. A secure-ONLY card (the L4:
        # communityCloud=false, verbatim raw bytes, run #14) has no such
        # market, so its null lowestPrice carries no signal — its secure
        # price is the whole quote. A missing or malformed communityCloud
        # (parsed as None) rejects conservatively; it is never inferred.
        community = gpu.get("community_cloud")
        if community is True:
            allocatable_secure = (
                gpu.get("secure_cloud")
                and gpu.get("secure_price") is not None
                and gpu.get("on_demand_price") is not None
            )
        elif community is False:
            allocatable_secure = (
                gpu.get("secure_cloud") and gpu.get("secure_price") is not None
            )
        else:
            allocatable_secure = False
        if gpu.get("id") == target:
            if allocatable_secure:
                entry = gpu
        elif allocatable_secure and (gpu.get("memory_gb") or 0) >= min_vram_gb:
            alternatives.append(gpu)
    if entry is None:
        raise UnavailableGpu(target, alternatives)
    return entry


def check_endpoint_config(min_workers, max_workers) -> None:
    """CI verifies an endpoint's configuration; it never authors one."""
    if min_workers != 0 or max_workers != 1:
        raise AdmissionRefused(
            "endpoint-config-refused",
            f"endpoint must be min_workers=0/max_workers=1, "
            f"found {min_workers}/{max_workers}",
        )


def check_spend_gate(spend_input, approval_granted) -> None:
    """Spending requires BOTH the literal SPEND input and the gpu-spend
    approval. No implicit approval, no automatic approval."""
    if spend_input != "SPEND" or approval_granted is not True:
        raise AdmissionRefused(
            "spend-not-approved",
            "BLOCKED — DO NOT SPEND: requires the literal input 'SPEND' "
            "and gpu-spend approval, both",
        )
