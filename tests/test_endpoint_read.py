"""The two-route endpoint read: $0, redacting, and honest about absence."""

import inspect
import json

from validation import endpoint_read as er


class FakeClient:
    REST_BASE = "https://rest.example/v1"

    def __init__(self, docs):
        self.docs = docs
        self.seen = []

    def _get_json(self, url):
        self.seen.append(url)
        path = url[len(self.REST_BASE):]
        if path not in self.docs:
            raise RuntimeError(f"404 {path}")
        return "", self.docs[path]


def test_absence_is_a_disagreement_not_an_equal_none():
    """The finding that started this module was a key one route OMITS.

    Comparing with .get() alone reports None == None and hides exactly the
    case worth printing.
    """
    diff = er.disagreements({"dataCenterIds": ["EU-RO-1"]}, {})
    assert diff["dataCenterIds"] == {"detail": ["EU-RO-1"], "list": "<ABSENT>"}

    diff = er.disagreements({}, {"dataCenterIds": None})
    assert diff["dataCenterIds"] == {"detail": "<ABSENT>", "list": None}

    # A key both carry with the same value is not a disagreement.
    assert er.disagreements({"a": 1}, {"a": 1}) == {}


def test_a_credential_shaped_key_is_never_printed_by_value():
    out = er.safe({
        "id": "mbnif6m7jbyf93",
        "containerRegistryAuthId": "super-secret-value",
        "env": {"RUNPOD_API_KEY": "sk-live-abcdef"},
        "gpuTypeIds": ["NVIDIA A40"],
    })
    blob = json.dumps(out)
    assert "super-secret-value" not in blob
    assert "sk-live-abcdef" not in blob
    # Ids and names still come through, because that is what a config read
    # is for.
    assert out["id"] == "mbnif6m7jbyf93"
    assert out["gpuTypeIds"] == ["NVIDIA A40"]


def test_it_reads_both_routes_and_every_volume_either_one_names(capsys):
    client = FakeClient({
        "/endpoints/e1": {
            "id": "e1", "networkVolumeId": "vol-50", "dataCenterIds": ["EU-RO-1"],
        },
        "/endpoints": [
            {"id": "e1", "networkVolumeId": "vol-10"},
        ],
        "/networkvolumes/vol-50": {"id": "vol-50", "dataCenterId": "EU-RO-1", "size": 50},
        "/networkvolumes/vol-10": {"id": "vol-10", "dataCenterId": "EU-RO-1", "size": 10},
    })
    assert er.report(client, "e1") == 0
    text = capsys.readouterr().out
    # Both volumes are resolved, because either route's answer could be the
    # one that is true.
    assert "vol-50" in text and "vol-10" in text
    assert "ROUTES DISAGREE" in text
    # dataCenterIds absent from the list route is reported as absence.
    assert "<ABSENT>" in text
    assert f"{client.REST_BASE}/endpoints/e1" in client.seen
    assert f"{client.REST_BASE}/endpoints" in client.seen


def test_an_unreadable_route_is_reported_not_rendered_as_empty(capsys):
    client = FakeClient({"/endpoints": [{"id": "e1"}]})
    assert er.report(client, "e1") == 0
    text = capsys.readouterr().out
    assert "UNREADABLE" in text


def test_the_module_holds_no_write_path_at_all():
    source = inspect.getsource(er)
    for verb in ("method=", "POST", "PATCH", "DELETE", "PUT"):
        assert verb not in source, verb
