"""The story_generate workload — local by default, ChatGPT-capable by config.

Owner directive 2026-08-27 (Qwen3-8B conditionally approved). The default
path keeps the local model baked into MODEL_DIR behind a licence gate and
loaded with local_files_only. When ONIQ_STORY_PROVIDER=openai is set, the
same op may be served by ChatGPT instead.

THE LIFECYCLE IS THE POINT. LTX peaked at 15.9GB of the A5000's 24GB on
the measured 2026-08-27 job. The story model therefore runs ALONE:

    load -> generate -> parse -> DELETE -> empty_cache -> (LTX may start)

`run` releases the model in a finally block, so a refusal, a timeout or a
crash inside generation still hands the card back. A story job that left
8GB resident would not fail loudly; it would make the NEXT video job fail
mysteriously, which is the failure this ordering exists to prevent.

Heavy imports happen inside the functions, exactly as videogen does, so
the CPU test rig can exercise everything except the CUDA pass itself.
"""

from __future__ import annotations

import os
import time
import io
import json
import urllib.error
import urllib.parse
import urllib.request

import contract
import modelroot
from preprocess import GpuUnavailable

# Lazy for the same reason as videogen's: a warm worker can be hydrated
# mid-life, and the next job must see it. Tests override the attribute.
MODEL_DIR = None
MODEL_ID_FILE = None


def _model_dir() -> str:
    return MODEL_DIR or modelroot.resolve_production("story")


def _model_id_file() -> str:
    return MODEL_ID_FILE or modelroot.resolve_production_file("STORY_MODEL_ID")

# Server decisions, like the video sampler's: the caller chooses none of
# them. A story is long-form structured JSON, so the budget is generous
# and the sampling is deliberately low-variance.
MAX_NEW_TOKENS = 8192
TEMPERATURE = 0.7
TOP_P = 0.9
SEED = 42
OPENAI_TIMEOUT_SECONDS = 120
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
OPENAI_PROVIDERS = frozenset({"openai", "chatgpt"})


class StoryModelUnavailable(Exception):
    """No local checkpoint. NEVER a reason to call a provider."""


def _story_provider() -> str:
    raw = (os.environ.get("ONIQ_STORY_PROVIDER") or "").strip().lower()
    if not raw:
        return "local"
    if raw == "local":
        return "local"
    if raw in OPENAI_PROVIDERS:
        return "openai"
    raise contract.ContractError(
        "story-provider-invalid",
        "ONIQ_STORY_PROVIDER must be local, openai, or chatgpt",
    )


def _openai_model() -> str:
    chosen = (os.environ.get("OPENAI_MODEL") or "").strip()
    return chosen or DEFAULT_OPENAI_MODEL


def _openai_url() -> str:
    # This worker sends the Chat Completions payload shape below. A custom
    # endpoint is therefore supported only when it is that exact HTTPS route;
    # pointing the bearer token at some other path is misconfiguration.
    chosen = (os.environ.get("OPENAI_API_URL") or "").strip()
    url = chosen or "https://api.openai.com/v1/chat/completions"
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise contract.ContractError(
            "story-provider-invalid",
            "OPENAI_API_URL must be an https:// chat completions endpoint",
        )
    if not parsed.netloc or not parsed.path.endswith("/chat/completions"):
        raise contract.ContractError(
            "story-provider-invalid",
            "OPENAI_API_URL must end in /chat/completions",
        )
    return url


def _json_from_bytes(raw: bytes) -> object:
    return json.load(io.StringIO(raw.decode("utf-8")))


def _openai_story(prompt: str, max_new_tokens: int, urlopen=None) -> tuple[str, str]:
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise contract.ContractError(
            "story-provider-not-configured",
            "OPENAI_API_KEY is required when ONIQ_STORY_PROVIDER selects ChatGPT",
        )

    body = json.dumps(
        {
            "model": _openai_model(),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": max_new_tokens,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        _openai_url(),
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    opener = urlopen or urllib.request.urlopen
    try:
        with opener(req, timeout=OPENAI_TIMEOUT_SECONDS) as response:
            try:
                payload = _json_from_bytes(response.read())
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise contract.ContractError(
                    "story-provider-failed",
                    "ChatGPT returned an unreadable response",
                ) from exc
            if not isinstance(payload, dict):
                raise contract.ContractError(
                    "story-provider-failed",
                    "ChatGPT returned a non-object response",
                )
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise contract.ContractError(
                "story-provider-unauthorized",
                "ChatGPT refused the API key",
            ) from exc
        if exc.code == 429:
            raise contract.ContractError(
                "story-provider-rate-limited",
                "ChatGPT rate-limited the request",
            ) from exc
        raise contract.ContractError(
            "story-provider-failed",
            f"ChatGPT request failed with HTTP {exc.code}",
        ) from exc
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise contract.ContractError(
            "story-provider-unreachable",
            "ChatGPT could not be reached",
        ) from exc

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise contract.ContractError(
            "story-provider-failed",
            "ChatGPT returned no choices",
        )
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    text = message.get("content") if isinstance(message, dict) else None
    if isinstance(text, list):
        parts = []
        for part in text:
            if not isinstance(part, dict):
                continue
            value = part.get("text")
            if isinstance(value, str):
                parts.append(value)
        text = "".join(parts)
    if not isinstance(text, str):
        raise contract.ContractError(
            "story-provider-failed",
            "ChatGPT returned no text",
        )
    return text, str(payload.get("model") or _openai_model())


def model_id() -> str:
    try:
        with open(_model_id_file(), encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return "missing"


def weights_present(model_dir: str | None = None) -> bool:
    model_dir = model_dir or _model_dir()
    """A baked checkpoint is a config plus at least one weight shard."""
    if not os.path.isdir(model_dir):
        return False
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        return False
    return any(name.endswith(".safetensors") for name in os.listdir(model_dir))


def _load_real_model():
    """Load the baked model onto CUDA. Never touches the network.

    Returns (model, tokenizer, precision) — the precision is REPORTED, not
    assumed, because the two modes have very different footprints (~4.6GB
    at 4-bit against ~16.4GB at bf16) and the job's own response is the
    only place that difference can be seen after the fact.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(_model_dir(), local_files_only=True)
    kwargs = {"local_files_only": True, "torch_dtype": torch.bfloat16, "device_map": "cuda"}

    quantised = None
    try:
        from transformers import BitsAndBytesConfig

        quantised = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    except Exception:
        quantised = None

    if quantised is not None:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                _model_dir(), quantization_config=quantised, **kwargs
            )
            model.eval()
            return model, tokenizer, "4bit"
        except Exception as exc:
            # GUARDING THE IMPORT IS NOT ENOUGH, and CI proved it: the
            # built image logs "bitsandbytes was compiled without GPU
            # support" while BitsAndBytesConfig imports perfectly well.
            # A build like that fails HERE, at load — so a try around the
            # import alone would have turned it into a dead PAID job on
            # the card. bf16 is a real fallback rather than a nicety: the
            # model runs alone by design, and ~16.4GB fits the 24GB A5000
            # with LTX not resident.
            print(
                "story model: 4-bit load failed "
                f"({type(exc).__name__}: {exc}); falling back to bf16"
            )

    model = AutoModelForCausalLM.from_pretrained(_model_dir(), **kwargs)
    model.eval()
    return model, tokenizer, "bf16"


def _release(model) -> None:
    """Hand the card back. Called in a finally, always."""
    try:
        del model
    except Exception:
        pass
    try:
        import gc

        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        # A cleanup that cannot run must not mask the job's own result.
        pass


def _generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    # torch is imported lazily and tolerated absent, exactly as videogen
    # does: the CPU rig injects a fake model and exercises everything
    # here except the CUDA pass itself.
    try:
        import torch
    except ImportError:
        torch = None

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    if torch is None:
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated = out[0][inputs["input_ids"].shape[-1] :]
        return tokenizer.decode(generated, skip_special_tokens=True)
    torch.manual_seed(SEED)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def run(job: dict, load_model=None, request_story=None) -> dict:
    """Generate one story. Returns the model's raw text plus measurements.

    THE WORKER DOES NOT VALIDATE THE STORY. Parsing, repair and the Story
    IR validator live in the application, which already owns them and is
    where an invalid story must stop before any GPU job is planned. This
    op's contract is narrower and therefore checkable: produce text from
    ONIQ's own model, measure it, and release the card.
    """
    started = time.monotonic()
    provider = _story_provider()

    if provider == "openai":
        request_story = request_story or _openai_story
        infer_started = time.monotonic()
        text, remote_model = request_story(
            job["params"]["prompt"], job["params"]["max_tokens"]
        )
        inference_ms = int((time.monotonic() - infer_started) * 1000)
        if not text.strip():
            raise contract.ContractError("story-empty", "the model produced no text")
        return {
            "ok": True,
            "op": "story_generate",
            "model": remote_model,
            "model_load_ms": 0,
            "precision": None,
            "inference_ms": inference_ms,
            "story_text": text,
            "story_chars": len(text),
            "vram_peak_mb": None,
            "duration_ms": int((time.monotonic() - started) * 1000),
        }

    if load_model is None:
        if not weights_present():
            raise StoryModelUnavailable(
                "no story model is baked at " + _model_dir() + "; story generation is "
                "unavailable and no provider substitutes for it"
            )
        try:
            import torch
        except ImportError:
            torch = None
        if torch is None or not torch.cuda.is_available():
            raise GpuUnavailable(
                "story_generate requires CUDA; there is no CPU fallback"
            )
        load_model = _load_real_model

    load_started = time.monotonic()
    # A loader may report the precision it achieved as a third element.
    # Two elements stays valid so an injected test double — and any older
    # caller — needs no knowledge of quantisation to exercise this path.
    loaded = load_model()
    model, tokenizer = loaded[0], loaded[1]
    precision = loaded[2] if len(loaded) > 2 else None
    model_load_ms = int((time.monotonic() - load_started) * 1000)

    peak_mb = None
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        torch = None

    try:
        infer_started = time.monotonic()
        text = _generate(
            model, tokenizer, job["params"]["prompt"], job["params"]["max_tokens"]
        )
        inference_ms = int((time.monotonic() - infer_started) * 1000)
        if torch is not None and torch.cuda.is_available():
            peak_mb = int(torch.cuda.max_memory_allocated() / (1024 * 1024))
    finally:
        # ALWAYS. A refusal must not leave the card occupied for LTX.
        _release(model)

    if not text.strip():
        raise contract.ContractError("story-empty", "the model produced no text")

    return {
        "ok": True,
        "op": "story_generate",
        "model": model_id(),
        "model_load_ms": model_load_ms,
        "precision": precision,
        "inference_ms": inference_ms,
        "story_text": text,
        "story_chars": len(text),
        "vram_peak_mb": peak_mb,
        "duration_ms": int((time.monotonic() - started) * 1000),
    }
