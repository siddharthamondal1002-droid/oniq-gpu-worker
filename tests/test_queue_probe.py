"""The stale-run probe: read-only, and fail-closed on anything unread."""

from validation import queue_probe as qp


class Client:
    def __init__(self, endpoints, health=None, statuses=None):
        self._endpoints = endpoints
        self._health = health or {}
        self._statuses = statuses or {}

    def get_endpoints(self):
        return "{}", [{"id": e} for e in self._endpoints]

    def endpoint_health(self, endpoint_id):
        value = self._health.get(endpoint_id, {"jobs": {"inQueue": 0, "inProgress": 0}})
        if isinstance(value, Exception):
            raise value
        return "{}", value

    def job_status(self, endpoint_id, job_id):
        value = self._statuses.get(endpoint_id, {"status": "FAILED"})
        if isinstance(value, Exception):
            raise value
        return "{}", value


def _empty(*eps):
    return {e: {"jobs": {"inQueue": 0, "inProgress": 0}} for e in eps}


def test_a_queued_job_can_execute():
    rows = [{"endpoint": "a", "in_queue": 1, "in_progress": 0, "job_status": "IN_QUEUE"}]
    assert qp.verdict(rows) == qp.CAN_EXECUTE


def test_a_nonzero_queue_alone_can_execute_even_with_no_job_status():
    rows = [{"endpoint": "a", "in_queue": 3, "in_progress": 0, "job_status": None}]
    assert qp.verdict(rows) == qp.CAN_EXECUTE


def test_work_in_progress_can_execute():
    rows = [{"endpoint": "a", "in_queue": 0, "in_progress": 1, "job_status": None}]
    assert qp.verdict(rows) == qp.CAN_EXECUTE


def test_empty_queues_and_a_terminal_job_cannot_execute():
    rows = [{"endpoint": "a", "in_queue": 0, "in_progress": 0, "job_status": "FAILED"}]
    assert qp.verdict(rows) == qp.CANNOT_EXECUTE


def test_an_unreadable_queue_is_unknown_never_empty():
    rows = [{"endpoint": "a", "in_queue": None, "in_progress": None, "job_status": None}]
    assert qp.verdict(rows) == qp.UNKNOWN


def test_one_unreadable_endpoint_keeps_the_whole_answer_unknown():
    rows = [
        {"endpoint": "a", "in_queue": 0, "in_progress": 0, "job_status": "FAILED"},
        {"endpoint": "b", "in_queue": None, "in_progress": None, "job_status": None},
    ]
    assert qp.verdict(rows) == qp.UNKNOWN


def test_a_live_endpoint_outvotes_an_empty_one():
    rows = [
        {"endpoint": "a", "in_queue": 0, "in_progress": 0, "job_status": "FAILED"},
        {"endpoint": "b", "in_queue": 2, "in_progress": 0, "job_status": None},
    ]
    assert qp.verdict(rows) == qp.CAN_EXECUTE


def test_an_unrecognised_status_is_unknown_not_terminal():
    rows = [{"endpoint": "a", "in_queue": 0, "in_progress": 0, "job_status": "WEIRD"}]
    assert qp.verdict(rows) == qp.UNKNOWN


def test_no_endpoints_at_all_is_unknown():
    assert qp.verdict([]) == qp.UNKNOWN


def test_a_missing_jobs_key_is_none_not_zero():
    assert qp._queue_counts({"workers": {}}) == (None, None)


def test_a_non_integer_count_is_none_not_zero():
    assert qp._queue_counts({"jobs": {"inQueue": "0", "inProgress": 0}}) == (None, None)


def test_real_shaped_health_parses():
    assert qp._queue_counts({"jobs": {"inQueue": 2, "inProgress": 1}}) == (2, 1)


def test_report_exits_zero_only_when_provably_safe():
    code, rows = qp.report(Client(["ep1", "ep2"], health=_empty("ep1", "ep2")), "stale-1")
    assert code == 0
    assert len(rows) == 2


def test_report_blocks_on_a_live_queue():
    client = Client(
        ["ep1", "ep2"],
        health={**_empty("ep1"), "ep2": {"jobs": {"inQueue": 1, "inProgress": 0}}},
    )
    assert qp.report(client, "stale-1")[0] == 1


def test_report_blocks_when_health_cannot_be_read():
    assert qp.report(Client(["ep1"], health={"ep1": OSError("no route")}), "s")[0] == 1


def test_a_missing_job_record_alone_does_not_make_it_safe():
    client = Client(["ep1"], health={"ep1": OSError("boom")}, statuses={"ep1": OSError("404")})
    assert qp.report(client, "stale-1")[0] == 1


def test_the_probe_never_calls_a_mutation():
    class Strict(Client):
        def cancel_job(self, *a, **k):
            raise AssertionError("probe attempted a cancel")

        def purge_queue(self, *a, **k):
            raise AssertionError("probe attempted a purge")

        def submit_job(self, *a, **k):
            raise AssertionError("probe attempted a submit")

    qp.report(Strict(["ep1"], health=_empty("ep1")), "stale-1")


def test_the_module_carries_no_mutation_verbs():
    source = open("validation/queue_probe.py", encoding="utf-8").read()
    for verb in ("cancel_job", "purge_queue", "submit_job", "set_workers_standby_zero"):
        assert f"client.{verb}" not in source
        assert f"runpod_client.{verb}" not in source


# ---------------------------------------------------- per-worker detail


def test_worker_detail_is_reported_and_never_fails_the_probe():
    """The counts cannot tell a worker that is downloading from one that
    died and was recreated — both read {"initializing": 1}. The detail
    answers it, and must never be able to break the probe that carries it:
    a diagnostic that takes the report down with it is worse than no
    diagnostic."""
    import validation.queue_probe as qp

    class Client:
        def endpoint_health(self, ep):
            return "{}", {"jobs": {"inQueue": 0, "inProgress": 0},
                          "workers": {"initializing": 1}}

        def worker_detail_graphql(self, ep):
            raise RuntimeError("graphql is down")

        def job_status(self, ep, job):
            return "{}", {"status": "COMPLETED"}

    row = qp.probe_endpoint(Client(), "ep1", "job1")
    assert row["worker_detail"] is None
    assert "RuntimeError" in row["worker_detail_note"]
    # The rest of the row still answered.
    assert row["in_queue"] == 0
    assert row["workers"]["initializing"] == 1


def test_worker_detail_rows_reach_the_row():
    import validation.queue_probe as qp

    class Client:
        def endpoint_health(self, ep):
            return "{}", {"jobs": {"inQueue": 0, "inProgress": 0}, "workers": {}}

        def worker_detail_graphql(self, ep):
            return [{"id": "w-abc", "status": "INITIALIZING"}], "fields: ['id']"

        def job_status(self, ep, job):
            return "{}", {"status": "COMPLETED"}

    row = qp.probe_endpoint(Client(), "ep1", "job1")
    assert row["worker_detail"] == [{"id": "w-abc", "status": "INITIALIZING"}]
