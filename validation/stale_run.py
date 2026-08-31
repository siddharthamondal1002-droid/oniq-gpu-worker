"""Cancel ONE stale RunPod run, on an endpoint proven to own it.

Owner authorization 2026-08-28: remove the stranded LTX run
ac95fea6-…-e2 before template hhhdwtjw0y receives an image, so repairing
the template cannot turn yesterday's failed request into today's paid
execution. The authorization is for THAT RUN. Four things follow:

- The endpoint is VERIFIED, not assumed. ONIQ stores no endpoint on the
  job row and RUNPOD_ENDPOINT_ID is a secret this process cannot read, so
  which endpoint owns the run is established from RunPod itself. A
  mismatch stops rather than cancelling somewhere plausible.
- The queue-wide drain is never called. It would stop every queued job
  on the endpoint, which is broader than what was authorized. The
  single-job cancel is the whole mutation surface used here, and a gate
  asserts the broad verb appears nowhere in this module - including in
  this sentence, which is why it is not written out.
- A run already in a terminal state is NOT re-cancelled. There is
  nothing to stop, and a POST that changes nothing is still a mutation.
- Nothing runs without the explicit token. Import and dry-run are safe.
"""

from __future__ import annotations

import json

from validation import queue_probe

AUTHORIZED_TOKEN = "CANCEL-STALE-RUN"


class StaleRunStop(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def locate(client, job_id: str, rows=None) -> dict:
    """Which endpoints RunPod says know this job, and in what state."""
    if rows is None:
        ids = queue_probe.endpoint_ids(client)
        rows = [queue_probe.probe_endpoint(client, ep, job_id) for ep in ids]
    owners = [r for r in rows if r.get("job_status") is not None]
    return {"rows": rows, "owners": owners}


def resolve(client, job_id: str, expected_endpoint: str, token: str = "") -> dict:
    """Verify, then cancel exactly one run. Returns the measured result."""
    found = locate(client, job_id)
    rows, owners = found["rows"], found["owners"]
    for row in rows:
        print(json.dumps(row, sort_keys=True))

    if not owners:
        raise StaleRunStop(
            "run-not-located",
            f"no endpoint reports a status for {job_id}; the endpoint that owns "
            "it is unproven, so there is nothing this authorization covers",
        )
    if len(owners) > 1:
        raise StaleRunStop(
            "run-ambiguous",
            f"{len(owners)} endpoints report a status for {job_id}; refusing to "
            "guess which one the authorization meant",
        )

    owner = owners[0]
    if owner["endpoint"] != expected_endpoint:
        raise StaleRunStop(
            "endpoint-mismatch",
            f"{job_id} lives on {owner['endpoint']}, not the authorized "
            f"{expected_endpoint}; cancelling elsewhere is not what was approved",
        )

    previous = owner.get("job_status")
    if previous in queue_probe.TERMINAL:
        print(
            f"NO MUTATION NEEDED: {job_id} is already {previous} — a cancel would "
            "change nothing, and a POST that changes nothing is still a mutation"
        )
        return {"previous": previous, "final": previous, "cancelled": False}

    if token != AUTHORIZED_TOKEN:
        raise StaleRunStop(
            "not-authorized",
            f"{job_id} is {previous} and would need a cancel, but the explicit "
            f"token was not supplied; refusing to mutate",
        )

    status, raw = client.cancel_job(expected_endpoint, job_id)
    print(f"cancel {job_id} on {expected_endpoint} -> HTTP {status}")
    if status not in (200, 201, 202):
        raise StaleRunStop(
            "cancel-failed",
            f"cancel returned HTTP {status}; body {raw[:300]!r}. Not retrying",
        )

    after = queue_probe.probe_endpoint(client, expected_endpoint, job_id)
    print(json.dumps(after, sort_keys=True))
    final = after.get("job_status")
    if final not in queue_probe.TERMINAL:
        raise StaleRunStop(
            "cancel-unconfirmed",
            f"after cancelling, {job_id} reads {final!r}, which is not a terminal "
            "state; the run is not provably stopped",
        )
    return {"previous": previous, "final": final, "cancelled": True}


def main(argv) -> int:
    if len(argv) < 3:
        print("usage: python -m validation.stale_run <job_id> <endpoint_id> [token]")
        return 2
    import runpod_client

    job_id, endpoint_id = argv[1], argv[2]
    token = argv[3] if len(argv) > 3 else ""
    try:
        result = resolve(runpod_client, job_id, endpoint_id, token)
    except StaleRunStop as stop:
        print(f"STOP [{stop.code}]: {stop.message}")
        return 1

    print(f"RESULT: previous={result['previous']} final={result['final']}")

    # Re-verify the whole account read-only: the point of the cancel was
    # that nothing can execute when the template gets an image.
    code, _ = queue_probe.report(runpod_client, job_id)
    sweep = runpod_client.sweep_orphans()
    print(f"sweep: {sweep}")
    if sweep is None:
        print("STOP [sweep-unconfirmed]: could not confirm zero; None is never 0")
        return 1
    if sweep.get("pods") != 0 or sweep.get("endpoint_min_workers") != 0:
        print(f"STOP [not-idle]: {sweep}")
        return 1
    return code


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
