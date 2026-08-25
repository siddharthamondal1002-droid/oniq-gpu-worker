import pytest

import contract
import storage


def _configure(monkeypatch):
    monkeypatch.setenv("R2_S3_ENDPOINT", "https://example.r2.cloudflarestorage.com")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "test-access-key-id")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "test-secret-value")


class FakeS3:
    def __init__(self, size=100, head_error=None, get_error=None, put_error=None):
        self.size = size
        self.head_error = head_error
        self.get_error = get_error
        self.put_error = put_error
        self.calls = []

    def head_object(self, Bucket, Key):
        self.calls.append(("head", Bucket, Key))
        if self.head_error:
            raise self.head_error
        return {"ContentLength": self.size}

    def download_file(self, Bucket, Key, Filename):
        self.calls.append(("get", Bucket, Key))
        if self.get_error:
            raise self.get_error
        with open(Filename, "wb") as fh:
            fh.write(b"x" * self.size)

    def upload_file(self, Filename, Bucket, Key):
        self.calls.append(("put", Bucket, Key))
        if self.put_error:
            raise self.put_error


def test_unconfigured_names_all_missing_vars():
    with pytest.raises(storage.StorageNotConfigured) as exc:
        storage.require_configured()
    assert exc.value.code == "storage-not-configured"
    for name in storage.REQUIRED_VARS:
        assert name in exc.value.message


def test_partially_configured_names_only_the_missing(monkeypatch):
    monkeypatch.setenv("R2_S3_ENDPOINT", "https://example.com")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "present-key-id")
    with pytest.raises(storage.StorageNotConfigured) as exc:
        storage.require_configured()
    assert exc.value.missing == ("R2_SECRET_ACCESS_KEY",)
    assert "R2_ACCESS_KEY_ID" not in exc.value.missing


def test_error_names_variables_never_values(monkeypatch):
    monkeypatch.setenv("R2_S3_ENDPOINT", "https://example.com")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "SUPERSECRETID")
    with pytest.raises(storage.StorageNotConfigured) as exc:
        storage.require_configured()
    assert "SUPERSECRETID" not in exc.value.message


def test_bucket_is_oniq_gpu_not_chat_media():
    assert storage.BUCKET == "oniq-gpu"
    assert storage.BUCKET != "oniq-chat-media"


def test_client_constructs_when_configured(monkeypatch):
    _configure(monkeypatch)
    s3 = storage.client()
    assert s3.meta.endpoint_url.startswith("https://example.r2")


def test_schemeless_endpoint_is_a_typed_misconfiguration(monkeypatch):
    # Regression for the first live job (2026-08-25, ea308ecd…-u1): a
    # scheme-less R2_S3_ENDPOINT made boto3 raise a bare ValueError at
    # client construction, which surfaced as unexpected-exception. The
    # stop must name the variable — and never echo its value.
    _configure(monkeypatch)
    monkeypatch.setenv("R2_S3_ENDPOINT", "accid.r2.cloudflarestorage.com")
    with pytest.raises(storage.StorageError) as exc:
        storage.client()
    assert exc.value.code == "r2-misconfigured"
    assert "R2_S3_ENDPOINT" in exc.value.message
    assert "accid" not in exc.value.message


def test_http_endpoint_is_a_typed_misconfiguration(monkeypatch):
    _configure(monkeypatch)
    monkeypatch.setenv("R2_S3_ENDPOINT", "http://accid.r2.cloudflarestorage.com")
    with pytest.raises(storage.StorageError) as exc:
        storage.client()
    assert exc.value.code == "r2-misconfigured"


def test_download_bounds_size_before_body(monkeypatch, tmp_path):
    _configure(monkeypatch)
    fake = FakeS3(size=contract.MAX_INPUT_BYTES + 1)
    monkeypatch.setattr(storage, "client", lambda: fake)
    with pytest.raises(contract.ContractError) as exc:
        storage.download("k", str(tmp_path / "f"), contract.MAX_INPUT_BYTES)
    assert exc.value.code == "input-too-large"
    assert [c[0] for c in fake.calls] == ["head"]  # never fetched the body


def test_download_happy_path(monkeypatch, tmp_path):
    _configure(monkeypatch)
    fake = FakeS3(size=42)
    monkeypatch.setattr(storage, "client", lambda: fake)
    dest = tmp_path / "f"
    assert storage.download("k", str(dest), 100) == 42
    assert dest.read_bytes() == b"x" * 42


def test_download_head_failure_is_r2_read_failed(monkeypatch, tmp_path):
    _configure(monkeypatch)
    fake = FakeS3(head_error=RuntimeError("boom"))
    monkeypatch.setattr(storage, "client", lambda: fake)
    with pytest.raises(storage.StorageError) as exc:
        storage.download("k", str(tmp_path / "f"), 100)
    assert exc.value.code == "r2-read-failed"


def test_download_body_failure_is_r2_read_failed(monkeypatch, tmp_path):
    _configure(monkeypatch)
    fake = FakeS3(get_error=RuntimeError("boom"))
    monkeypatch.setattr(storage, "client", lambda: fake)
    with pytest.raises(storage.StorageError) as exc:
        storage.download("k", str(tmp_path / "f"), 100)
    assert exc.value.code == "r2-read-failed"


def test_upload_failure_is_r2_write_failed(monkeypatch, tmp_path):
    _configure(monkeypatch)
    fake = FakeS3(put_error=RuntimeError("boom"))
    monkeypatch.setattr(storage, "client", lambda: fake)
    src = tmp_path / "src"
    src.write_bytes(b"data")
    with pytest.raises(storage.StorageError) as exc:
        storage.upload(str(src), "k")
    assert exc.value.code == "r2-write-failed"


def test_upload_happy_path_returns_bytes(monkeypatch, tmp_path):
    _configure(monkeypatch)
    fake = FakeS3()
    monkeypatch.setattr(storage, "client", lambda: fake)
    src = tmp_path / "src"
    src.write_bytes(b"12345")
    assert storage.upload(str(src), "out/k") == 5
    assert fake.calls == [("put", storage.BUCKET, "out/k")]


def test_operations_fail_closed_when_unconfigured(tmp_path):
    with pytest.raises(storage.StorageNotConfigured):
        storage.download("k", str(tmp_path / "f"), 100)
    with pytest.raises(storage.StorageNotConfigured):
        storage.upload(__file__, "k")
