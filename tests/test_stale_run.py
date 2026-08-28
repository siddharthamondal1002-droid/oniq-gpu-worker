"""One run, one endpoint, one cancel — and never the whole queue."""

import pytest

from validation import stale_run
from validation.stale_run import StaleRunStop

JOB = "ac95fea6-1014-4b8c-85e2-54df29cefbf1-e2"
OWNER_EP = "p3zmlv8ek10dzt"
OTHER_EP = "b6iw91zuvkrra9"
TOKEN = stale_run.AUTHORIZED_TOKEN


class Client:
    def __init__(self, endpoints, statuses, health=None, cancel=(200, "{}")):
        self._endpoints = endpoints
        self._statuses = statuses
        self._health = health or {
            e: {"jobs": {"inQueue": 0, "inProgress": 0}} for e in endpoints
        }
        self._cancel = cancel
        self.cancels = []
        self.purges = []

    def get_endpoints(self):
        return "{}", [{"id": e} for e in self._endpoints]

    def endpoint_health(self, endpoint_id):
        return "{}", self._health[endpoint_id]

    def job_status(self, endpoint_id, job_id):
        value = self._statuses.get(endpoint_id)
        if value is None:
            raise RuntimeError("404")
        if isinstance(value, list):
            value = value.pop(0)
        return "{}", {"status": value}

    def cancel_job(self, endpoint_id, job_id):
        self.cancels.append((endpoint_id, job_id))
        return self._cancel

    def purge_queue(self, endpoint_id):  # must never be reached
        self.purges.append(endpoint_id)
        raise AssertionError("purge_queue is broader than the authorization")


def _client(**kw):
    return Client([OWNER_EP, OTHER_EP], **kw)


# ------------------------------------------------------- locating first


def test_a_run_no_endpoint_claims_is_refused():
    client = _client(statuses={})
    with pytest.raises(StaleRunStop) as exc:
        stale_run.resolve(client, JOB, OWNER_EP, TOKEN)
    assert exc.value.code == "run-not-located"
    assert client.cancels == []


def test_a_run_on_the_wrong_endpoint_is_refused():
    client = _client(statuses={OTHER_EP: "IN_QUEUE"})
    with pytest.raises(StaleRunStop) as exc:
        stale_run.resolve(client, JOB, OWNER_EP, TOKEN)
    assert exc.value.code == "endpoint-mismatch"
    assert client.cancels == []


def test_two_claiming_endpoints_are_refused_rather_than_guessed():
    client = _client(statuses={OWNER_EP: "IN_QUEUE", OTHER_EP: "IN_QUEUE"})
    with pytest.raises(StaleRunStop) as exc:
        stale_run.resolve(client, JOB, OWNER_EP, TOKEN)
    assert exc.value.code == "run-ambiguous"
    assert client.cancels == []


# ------------------------------------------------------------ the cancel


def test_a_queued_run_on_the_right_endpoint_is_cancelled_once():
    client = _client(statuses={OWNER_EP: ["IN_QUEUE", "CANCELLED"]})
    result = stale_run.resolve(client, JOB, OWNER_EP, TOKEN)
    assert result == {"previous": "IN_QUEUE", "final": "CANCELLED", "cancelled": True}
    assert client.cancels == [(OWNER_EP, JOB)]


def test_an_already_terminal_run_is_not_re_cancelled():
    client = _client(statuses={OWNER_EP: "FAILED"})
    result = stale_run.resolve(client, JOB, OWNER_EP, TOKEN)
    assert result == {"previous": "FAILED", "final": "FAILED", "cancelled": False}
    assert client.cancels == []


def test_without_the_token_a_live_run_is_never_cancelled():
    client = _client(statuses={OWNER_EP: "IN_QUEUE"})
    with pytest.raises(StaleRunStop) as exc:
        stale_run.resolve(client, JOB, OWNER_EP, token="")
    assert exc.value.code == "not-authorized"
    assert client.cancels == []


def test_a_wrong_token_is_not_authorization():
    client = _client(statuses={OWNER_EP: "IN_QUEUE"})
    with pytest.raises(StaleRunStop):
        stale_run.resolve(client, JOB, OWNER_EP, token="please")
    assert client.cancels == []


def test_a_failed_cancel_stops_and_does_not_retry():
    client = _client(statuses={OWNER_EP: ["IN_QUEUE"]}, cancel=(500, "boom"))
    with pytest.raises(StaleRunStop) as exc:
        stale_run.resolve(client, JOB, OWNER_EP, TOKEN)
    assert exc.value.code == "cancel-failed"
    assert client.cancels == [(OWNER_EP, JOB)]


def test_a_cancel_that_does_not_reach_a_terminal_state_is_not_success():
    client = _client(statuses={OWNER_EP: ["IN_QUEUE", "IN_QUEUE"]})
    with pytest.raises(StaleRunStop) as exc:
        stale_run.resolve(client, JOB, OWNER_EP, TOKEN)
    assert exc.value.code == "cancel-unconfirmed"


# ------------------------------------------------------ scope of the ask


def test_the_whole_queue_is_never_purged():
    client = _client(statuses={OWNER_EP: ["IN_QUEUE", "CANCELLED"]})
    stale_run.resolve(client, JOB, OWNER_EP, TOKEN)
    assert client.purges == []


def test_the_module_never_names_the_broad_mutations():
    source = open("validation/stale_run.py", encoding="utf-8").read()
    assert "client.purge_queue" not in source
    assert "runpod_client.purge_queue" not in source
    assert "client.submit_job" not in source
    assert "runpod_client.submit_job" not in source


def test_exactly_one_cancel_call_exists_in_the_module():
    source = open("validation/stale_run.py", encoding="utf-8").read()
    assert source.count("client.cancel_job(") == 1
