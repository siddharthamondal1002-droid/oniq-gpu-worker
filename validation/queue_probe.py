"""Can a stale RunPod run still execute if the template gets an image?

Repairing an empty template is the one action that could turn yesterday's
stranded request into today's paid execution, so the question has to be
answered from the provider's own queue counts BEFORE the image lands —
never inferred from "active pods = 0", which says nothing about what is
waiting in a queue.

Everything here is a GET. There is no cancel and no purge in this module
on purpose: resolving a queued job is a mutation and needs the owner's
explicit authorization, so the probe reports and stops.

Fail-closed throughout. An unreadable count is UNKNOWN, and UNKNOWN is
never rounded down to zero — the same rule sweep_orphans follows when it
returns None rather than 0.
"""

from __future__ import annotations

import json

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}
LIVE = {"IN_QUEUE", "IN_PROGRESS"}

CAN_EXECUTE = "CAN EXECUTE"
CANNOT_EXECUTE = "CANNOT EXECUTE"
UNKNOWN = "UNKNOWN"


def _queue_counts(health) -> tuple:
    """(in_queue, in_progress) or (None, None) when the shape is not what
    we think it is. A missing key is not a zero."""
    if not isinstance(health, dict):
        return (None, None)
    jobs = health.get("jobs")
    if not isinstance(jobs, dict):
        return (None, None)
    in_queue = jobs.get("inQueue")
    in_progress = jobs.get("inProgress")
    if not isinstance(in_queue, int) or not isinstance(in_progress, int):
        return (None, None)
    return (in_queue, in_progress)


JOB_KEYS = ("completed", "failed", "inProgress", "inQueue", "retried")
WORKER_KEYS = ("idle", "initializing", "ready", "running", "throttled", "unhealthy")


def _facts(health, section: str, keys) -> dict:
    """Every key the provider reports for one section, fail-closed.

    A key the payload does not carry is None, never 0 — the same rule
    _queue_counts follows. Read from the health document already fetched
    for the queue counts, so this costs no extra call and cannot spend.
    """
    if not isinstance(health, dict):
        return {k: None for k in keys}
    block = health.get(section)
    if not isinstance(block, dict):
        return {k: None for k in keys}
    return {k: (block.get(k) if isinstance(block.get(k), int) else None) for k in keys}


def probe_endpoint(client, endpoint_id: str, job_id: str) -> dict:
    row = {"endpoint": endpoint_id}

    health = None
    try:
        _, health = client.endpoint_health(endpoint_id)
        in_queue, in_progress = _queue_counts(health)
    except Exception as exc:
        in_queue = in_progress = None
        row["health_error"] = type(exc).__name__
    row["in_queue"] = in_queue
    row["in_progress"] = in_progress
    # WHY THE WHOLE PAYLOAD AND NOT JUST THE TWO COUNTS. Story job
    # 1481d262 (2026-08-28) died when story-still's image_generate returned
    # FAILED twice on shot 1, and "in_queue 0, in_progress 0" cannot tell a
    # healthy idle endpoint from one whose workers are unhealthy or
    # throttled. failed/retried and the worker states are the difference
    # between "cold start" and "the endpoint is broken", so they get read.
    row["jobs"] = _facts(health, "jobs", JOB_KEYS)
    row["workers"] = _facts(health, "workers", WORKER_KEYS)

    try:
        _, doc = client.job_status(endpoint_id, job_id)
        row["job_status"] = (doc or {}).get("status") if isinstance(doc, dict) else None
    except Exception as exc:
        row["job_status"] = None
        # The exception NAME is not the answer. RunPodApiError carries the
        # HTTP code in its message ("GET <url> -> 404"), and 404-the-record-
        # is-gone reads very differently from 401-we-cannot-look. The URL
        # holds the endpoint and job id only; the credential travels in a
        # header, so this is safe to print.
        row["status_error"] = f"{type(exc).__name__}: {exc}"

    return row


def verdict(rows: list) -> str:
    """CAN EXECUTE beats UNKNOWN beats CANNOT EXECUTE.

    Anything live anywhere is decisive on its own. Otherwise every
    endpoint must be READABLE and empty before the answer is 'cannot' —
    one unreadable endpoint keeps the whole answer UNKNOWN, because the
    stale run could be sitting in exactly the queue we failed to read.
    """
    if not rows:
        return UNKNOWN

    for row in rows:
        if row.get("job_status") in LIVE:
            return CAN_EXECUTE
        if (row.get("in_queue") or 0) > 0 or (row.get("in_progress") or 0) > 0:
            return CAN_EXECUTE

    for row in rows:
        if row.get("in_queue") is None or row.get("in_progress") is None:
            return UNKNOWN
        status = row.get("job_status")
        if status is not None and status not in TERMINAL:
            return UNKNOWN

    return CANNOT_EXECUTE


def endpoint_ids(client) -> list:
    _, endpoints = client.get_endpoints()
    ep_list = (
        endpoints if isinstance(endpoints, list) else endpoints.get("endpoints", [])
    )
    return [e.get("id") for e in ep_list if e.get("id")]


def report(client, job_id: str) -> tuple:
    """Exit code and rows. 0 ONLY when the run provably cannot execute."""
    ids = endpoint_ids(client)
    print(f"endpoints seen: {ids}")
    rows = [probe_endpoint(client, ep, job_id) for ep in ids]
    for row in rows:
        print(json.dumps(row, sort_keys=True))
    answer = verdict(rows)
    print(f"STALE RUN {job_id}: {answer}")
    if answer == CANNOT_EXECUTE:
        print("SAFE: no queued or running work anywhere; repairing the template "
              "cannot resurrect this run")
        return 0, rows
    if answer == CAN_EXECUTE:
        print("BLOCKED: queued or running work exists — repairing the template "
              "could cause unattended execution. Cancelling it is a MUTATION "
              "and requires explicit owner authorization.")
        return 1, rows
    print("BLOCKED: the queue could not be read; UNKNOWN is never treated as "
          "empty. Do not repair the template on this evidence.")
    return 1, rows


def main(argv) -> int:
    if len(argv) < 2:
        print("usage: python -m validation.queue_probe <runpod_job_id>")
        return 2
    import runpod_client

    code, _ = report(runpod_client, argv[1])
    return code


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
