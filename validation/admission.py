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
# VIDEO has since been measured on this card too (production launch,
# then the 2026-08-27 watermark canary: 97 frames in 42.7s billed), so
# verify_gpu_success pins TARGET_GPU itself. Its earlier hardcoded
# "3090" outlived that card's retirement and stopped a canary whose
# generation had actually succeeded on the A5000 — one target, one name.
#
# SUPERSEDED 2026-08-30, and the reason is worth keeping. The A5000 was
# retired when the owner moved to a 48 GB card; the endpoint then sat with
# gpuTypeIds ["NVIDIA RTX A6000"] and workers {throttled 1} — RunPod
# wanting to place a worker and unable to get that one card — while this
# constant still named the A5000, so every spend refused
# endpoint-not-target before reaching the GPU at all.
#
# ONE CARD NAMED IS A SINGLE POINT OF FAILURE. Owner directive 2026-08-30:
# RTX A6000 and A40, and nothing else. Both are 48 GB, so a model that fits
# one fits the other and the fallback changes where a job lands and roughly
# how long it takes — never whether it can run. A 24 or 32 GB card would
# turn "a bit slower" into "out of memory", which is why the set is these
# two and why it stays an allow-list rather than becoming "any card".
APPROVED_GPUS = ("NVIDIA RTX A6000", "NVIDIA A40")

# The preferred card, for the places that need ONE name — a message, a
# default, a quote. Membership, not equality, is what the gates test.
TARGET_GPU = APPROVED_GPUS[0]

# Server-side allow-list: which card ONIQ rents is an owner decision, and
# the allow-list is why "give me 8x H100" cannot be typed at all.
ALLOWED_GPUS = {name: {"min_vram_gb": 48} for name in APPROVED_GPUS}

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
    ceiling_seconds: int = RUNTIME_CEILING_SECONDS,
) -> Reservation:
    """The caller cannot choose what it costs. Refuses in a fixed order,
    VRAM before price; a runtime above the ceiling is refused, not
    clamped.

    `ceiling_seconds` DEFAULTS to production's and is passed explicitly by
    exactly one caller: the model_probe path, whose jobs download a
    checkpoint before they start and so run in a wider window (see
    contract.PROBE_RUNTIME_CEILING_SECONDS). Reserving 900s for an
    1800s-capable job is not conservative — it is wrong in the expensive
    direction, because the reservation check fires AFTER the job has run and
    would refuse a measurement already paid for. The cap is untouched: a
    wider window still has to clear JOB_CAP_USD, and it does.
    """
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
    if runtime_seconds > ceiling_seconds:
        raise AdmissionRefused(
            "runtime-exceeds-ceiling",
            f"runtime {runtime_seconds}s exceeds the "
            f"{ceiling_seconds}s ceiling (refused, not clamped)",
        )
    # The reservation is a TIME budget: charge the full ceiling window,
    # never the caller's expectation of how long the job should take.
    reserved = reserve_usd(price_per_hour, ceiling_seconds)
    if reserved > JOB_CAP_USD:
        raise AdmissionRefused(
            "over-job-cap",
            f"reservation ${reserved} exceeds the ${JOB_CAP_USD} job cap",
        )
    return Reservation(
        gpu=gpu_name,
        runtime_seconds=ceiling_seconds,
        price_per_hour_usd=Decimal(str(price_per_hour)),
        reserved_usd=reserved,
    )


def require_available(catalogue, target: str = TARGET_GPU, min_vram_gb: int = WORKLOAD_MIN_VRAM_GB):
    """Find the target GPU as secure-cloud, ALLOCATABLE capacity — or
    raise UnavailableGpu listing allocatable alternatives. Never
    substitutes.

    Two lessons from real payloads are load-bearing here. Matching is on
    the `id` field (the canonical full name), never `displayName` (the
    short name). And "null is never free": a null SECURE price refuses
    (reserve_usd raises gpu-unpriced), never admits.

    THE SERVERLESS RULE (owner directive 2026-08-26, evening record):
    lowestPrice is a community/pod-market rental signal, and this harness
    only ever admits SERVERLESS jobs, which bill the secure price against
    an endpoint that manages its own worker pool. Across eight catalogue
    pulls in one evening the lowestPrice field flapped null on the 3090,
    then the L4, then the A5000 — each while the other cards' signals
    were live and while every secure price stayed firm — costing five $0
    admission refusals across three owner-pinned cards. Serverless
    eligibility therefore reads the secure side only; the spend is
    actually protected by the cap, the full-ceiling reservation, the
    runtime ceiling, the single submit, the watchdog and the orphan
    sweep. The 2026-08-25 A5000 lesson was calibrated against POD
    provisioning and its bytes now pin the reinterpretation in tests."""
    entry = None
    alternatives = []
    for gpu in catalogue:
        # THE SERVERLESS RULE (see docstring): secure side only. The
        # pod-market lowestPrice and the communityCloud flag are not
        # consulted for serverless eligibility.
        allocatable_secure = (
            gpu.get("secure_cloud") and gpu.get("secure_price") is not None
        )
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
