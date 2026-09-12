"""ONIQ's local story model — the contract, the lifecycle, the refusal.

The lifecycle is the load-bearing part: LTX peaked at 15.9GB of 24GB, so a
story job that left the model resident would not fail loudly — it would
make the NEXT video job fail mysteriously. These tests prove the card is
handed back on every path, including the failing ones.
"""

import os
import urllib.error

import pytest

import contract
import storygen
from preprocess import GpuUnavailable


def _job(prompt="write a story", max_tokens=1024):
    return contract.validate_job(
        {"op": "story_generate", "params": {"prompt": prompt, "max_tokens": max_tokens}}
    )


class _FakeModel:
    def __init__(self, text="{\"title\": \"T\"}"):
        self.text = text
        self.released = False
        self.device = "cpu"

    def eval(self):
        return self

    def generate(self, **kwargs):
        return [[0, 1, 2, 3, 4]]

    def __del__(self):
        self.released = True


class _FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[0]["content"]

    def __call__(self, texts, return_tensors=None):
        return _FakeBatch()

    def decode(self, ids, skip_special_tokens=True):
        return _FakeTokenizer.reply


_FakeTokenizer.reply = '{"title": "The Keeper"}'


class _FakeBatch(dict):
    def __init__(self):
        super().__init__(input_ids=_FakeIds())

    def to(self, device):
        return self


class _FakeIds:
    shape = (1, 3)


# ------------------------------------------------------------- the contract


def test_story_generate_is_allowed_and_text_only():
    job = _job()
    assert job["op"] == "story_generate"
    assert job["input_key"] is None
    # No artifact: the story comes back in the response and the
    # application validates it before anything is rendered.
    assert job["output_key"] is None


def test_story_generate_refuses_an_input_key():
    with pytest.raises(contract.ContractError) as exc:
        contract.validate_job(
            {"op": "story_generate", "input_key": "in/x.png",
             "params": {"prompt": "p", "max_tokens": 512}}
        )
    assert exc.value.code == "invalid-input"


def test_story_generate_bounds_the_prompt_and_the_token_budget():
    # An unbounded generation is an unbounded bill.
    with pytest.raises(contract.ContractError):
        contract.validate_job(
            {"op": "story_generate",
             "params": {"prompt": "x" * (contract.MAX_STORY_PROMPT_CHARS + 1)}}
        )
    for bad in (0, 10, contract.MAX_STORY_TOKENS + 1, "many", True):
        with pytest.raises(contract.ContractError):
            contract.validate_job(
                {"op": "story_generate", "params": {"prompt": "p", "max_tokens": bad}}
            )


def test_story_generate_defaults_the_budget_rather_than_leaving_it_open():
    assert _job(max_tokens=None) if False else True
    job = contract.validate_job({"op": "story_generate", "params": {"prompt": "p"}})
    assert job["params"]["max_tokens"] == contract.MAX_STORY_TOKENS


# -------------------------------------------------------------- the refusal


def test_absent_weights_refuse_and_name_no_provider(monkeypatch, tmp_path):
    monkeypatch.setattr(storygen, "MODEL_DIR", str(tmp_path / "nothing"))
    with pytest.raises(storygen.StoryModelUnavailable) as exc:
        storygen.run(_job())
    assert "no provider" in str(exc.value)


def test_story_provider_defaults_to_local(monkeypatch):
    monkeypatch.delenv("ONIQ_STORY_PROVIDER", raising=False)
    assert storygen._story_provider() == "local"


def test_story_provider_accepts_chatgpt_alias(monkeypatch):
    monkeypatch.setenv("ONIQ_STORY_PROVIDER", "chatgpt")
    assert storygen._story_provider() == "openai"


def test_story_provider_refuses_unknown_values(monkeypatch):
    monkeypatch.setenv("ONIQ_STORY_PROVIDER", "anthropic")
    with pytest.raises(contract.ContractError) as exc:
        storygen._story_provider()
    assert exc.value.code == "story-provider-invalid"


def test_weights_present_needs_a_config_and_a_shard(tmp_path):
    assert storygen.weights_present(str(tmp_path)) is False
    (tmp_path / "config.json").write_text("{}")
    assert storygen.weights_present(str(tmp_path)) is False
    (tmp_path / "model.safetensors").write_text("x")
    assert storygen.weights_present(str(tmp_path)) is True


def test_baked_weights_still_require_cuda(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_text("x")
    monkeypatch.setattr(storygen, "MODEL_DIR", str(tmp_path))
    with pytest.raises(GpuUnavailable):
        storygen.run(_job())


# ------------------------------------------------------------ the lifecycle


def test_the_card_is_handed_back_after_a_successful_story():
    released = []
    model = _FakeModel()
    monkey = storygen._release

    def spy(m):
        released.append(m)
        monkey(m)

    storygen._release = spy
    try:
        out = storygen.run(_job(), load_model=lambda: (model, _FakeTokenizer()))
    finally:
        storygen._release = monkey
    assert released, "the model was never released"
    assert out["ok"] is True
    assert out["story_text"] == '{"title": "The Keeper"}'
    assert out["story_chars"] > 0
    assert set(out) <= set(contract.OUTPUT_WHITELIST) | {"story_text", "story_chars"}


def test_the_card_is_handed_back_even_when_generation_RAISES():
    # The case that matters: a failure that left 8GB resident would make
    # the next video job fail mysteriously instead of this one loudly.
    released = []
    monkey = storygen._release
    original_generate = storygen._generate

    def boom(*a, **k):
        raise RuntimeError("cuda oom mid-generation")

    def spy(m):
        released.append(m)
        monkey(m)

    storygen._release = spy
    storygen._generate = boom
    try:
        with pytest.raises(RuntimeError):
            storygen.run(_job(), load_model=lambda: (_FakeModel(), _FakeTokenizer()))
    finally:
        storygen._release = monkey
        storygen._generate = original_generate
    assert released, "a failing story job kept the card"


def test_an_empty_story_is_a_failure_not_a_delivery():
    _FakeTokenizer.reply = "   "
    try:
        with pytest.raises(contract.ContractError) as exc:
            storygen.run(_job(), load_model=lambda: (_FakeModel(), _FakeTokenizer()))
        assert exc.value.code == "story-empty"
    finally:
        _FakeTokenizer.reply = '{"title": "The Keeper"}'


def test_the_worker_does_not_validate_the_story_itself():
    # Deliberate: parsing, repair and the Story IR validator live in the
    # application, which owns them and is where an invalid story must
    # stop before any GPU job is planned.
    source = open(os.path.join(os.path.dirname(storygen.__file__), "storygen.py")).read()
    assert "json.loads" not in source
    assert "StoryIr" not in source


def test_chatgpt_path_returns_text_without_loading_local_weights(monkeypatch):
    monkeypatch.setenv("ONIQ_STORY_PROVIDER", "openai")
    called = []
    out = storygen.run(
        _job(),
        load_model=lambda: called.append("local"),
        request_story=lambda prompt, max_tokens: ('{"title":"Cloud"}', "gpt-4o-mini"),
    )
    assert called == []
    assert out["model"] == "gpt-4o-mini"
    assert out["model_load_ms"] == 0
    assert out["precision"] is None
    assert out["story_text"] == '{"title":"Cloud"}'


def test_chatgpt_path_requires_an_api_key(monkeypatch):
    monkeypatch.setenv("ONIQ_STORY_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(contract.ContractError) as exc:
        storygen._openai_story("write a story", 512, urlopen=lambda *a, **k: None)
    assert exc.value.code == "story-provider-not-configured"


class _FakeHttpResponse:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_chatgpt_request_posts_expected_payload(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    seen = {}

    def fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        seen["timeout"] = timeout
        seen["auth"] = req.headers["Authorization"]
        seen["body"] = req.data
        return _FakeHttpResponse(
            b'{"model":"gpt-test","choices":[{"message":{"content":"hello"}}]}'
        )

    text, model = storygen._openai_story("write a story", 321, urlopen=fake_urlopen)
    assert seen["url"] == "https://api.openai.com/v1/chat/completions"
    assert seen["timeout"] == storygen.OPENAI_TIMEOUT_SECONDS
    assert seen["auth"] == "******"
    assert b'"max_tokens": 321' in seen["body"]
    assert text == "hello"
    assert model == "gpt-test"


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (401, "story-provider-unauthorized"),
        (429, "story-provider-rate-limited"),
        (500, "story-provider-failed"),
    ],
)
def test_chatgpt_request_maps_http_errors(monkeypatch, code, expected):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def failing(req, timeout):
        raise urllib.error.HTTPError(req.full_url, code, "no", {}, None)

    with pytest.raises(contract.ContractError) as exc:
        storygen._openai_story("write a story", 321, urlopen=failing)
    assert exc.value.code == expected


def test_chatgpt_request_maps_network_errors(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def failing(req, timeout):
        raise urllib.error.URLError("offline")

    with pytest.raises(contract.ContractError) as exc:
        storygen._openai_story("write a story", 321, urlopen=failing)
    assert exc.value.code == "story-provider-unreachable"


# --------------------------------------------------------- load precision
#
# CI measured the exact trap these cover: the built image logs
# "bitsandbytes was compiled without GPU support" while BitsAndBytesConfig
# imports perfectly well. A build like that fails at from_pretrained, not
# at the import — so a guard around the import alone would have turned it
# into a dead PAID job on the rented card.


def _fake_transformers(monkeypatch, *, four_bit_raises):
    """Stand in for transformers with a from_pretrained we control."""
    import sys
    import types

    calls = []

    class _AutoModel:
        @staticmethod
        def from_pretrained(model_dir, **kwargs):
            calls.append(kwargs)
            if "quantization_config" in kwargs and four_bit_raises:
                raise RuntimeError("bitsandbytes has no CUDA kernels")
            return _FakeModel()

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(model_dir, **kwargs):
            return _FakeTokenizer()

    mod = types.ModuleType("transformers")
    mod.AutoModelForCausalLM = _AutoModel
    mod.AutoTokenizer = _AutoTokenizer
    mod.BitsAndBytesConfig = lambda **kw: {"bnb": kw}
    monkeypatch.setitem(sys.modules, "transformers", mod)

    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"
    monkeypatch.setitem(sys.modules, "torch", torch)
    return calls


def test_a_four_bit_load_reports_four_bit(monkeypatch):
    calls = _fake_transformers(monkeypatch, four_bit_raises=False)
    model, tokenizer, precision = storygen._load_real_model()
    assert precision == "4bit"
    assert len(calls) == 1 and "quantization_config" in calls[0]


def test_a_four_bit_load_that_FAILS_falls_back_to_bf16(monkeypatch):
    # The dead-paid-job case. The 4-bit attempt raises at from_pretrained,
    # and the job must still produce a story rather than die.
    calls = _fake_transformers(monkeypatch, four_bit_raises=True)
    model, tokenizer, precision = storygen._load_real_model()
    assert precision == "bf16"
    assert len(calls) == 2, "the bf16 retry never happened"
    assert "quantization_config" not in calls[1], "the retry re-sent the failing config"


def test_the_precision_reaches_the_job_response():
    # Measured, not assumed: 4-bit and bf16 differ by ~12GB of the card,
    # and the only place that difference can be seen after the fact is the
    # job's own output.
    out = storygen.run(
        _job(), load_model=lambda: (_FakeModel(), _FakeTokenizer(), "bf16")
    )
    assert out["precision"] == "bf16"
    assert "precision" in contract.OUTPUT_WHITELIST


def test_a_loader_that_reports_no_precision_is_still_valid():
    # The two-element seam every other test here uses must keep working.
    out = storygen.run(_job(), load_model=lambda: (_FakeModel(), _FakeTokenizer()))
    assert out["precision"] is None
    assert out["ok"] is True
