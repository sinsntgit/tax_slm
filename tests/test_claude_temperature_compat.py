"""2026-11-23: regression tests for the reported crash
"TypeError: Messages.create() got an unexpected keyword argument 'temperature'"
- anthropic SDK >= 1.0.0 removed `temperature` from Messages.create, and the
Settings page had temperature=0.0 saved, so every Claude call failed.
Fake clients only - no network, no real .env."""
import dataclasses
from types import SimpleNamespace

import pytest

from framework.core.infra import llm_client


def _response(text="ok"):
    return SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=3, output_tokens=2),
    )


class NewSdkMessages:
    """Mirrors anthropic>=1.0: explicit keyword-only params, no temperature, no **kwargs."""
    def __init__(self):
        self.calls = []

    def create(self, *, model, max_tokens, messages, system=None, stop_sequences=None):
        self.calls.append(dict(model=model, max_tokens=max_tokens, system=system))
        return _response()


class OldSdkMessages:
    def __init__(self):
        self.calls = []

    def create(self, *, model, max_tokens, messages, system=None, temperature=None):
        self.calls.append(dict(model=model, temperature=temperature))
        return _response()


class OpaqueMessages:
    """Signature says **kwargs (wrapped method) but the real call rejects temperature."""
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        if "temperature" in kwargs:
            raise TypeError("Messages.create() got an unexpected keyword argument 'temperature'")
        self.calls.append(kwargs)
        return _response()


class OtherTypeErrorMessages:
    def create(self, **kwargs):
        raise TypeError("something unrelated")


def _config(temperature=0.0):
    fields = {f.name for f in dataclasses.fields(llm_client.ModelConfig)}
    base = dict(use_model="CLAUDE", model_name="claude-test", timeout=30.0, temperature=temperature)
    for f in dataclasses.fields(llm_client.ModelConfig):
        if f.name not in base and f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            base[f.name] = None
    return llm_client.ModelConfig(**{k: v for k, v in base.items() if k in fields})


@pytest.fixture
def use_client(monkeypatch):
    monkeypatch.setattr(llm_client, "_claude_temperature_supported", None)

    def install(messages):
        client = SimpleNamespace(messages=messages)
        monkeypatch.setattr(llm_client, "_get_client", lambda config: client)
        return messages
    return install


def test_new_sdk_temperature_not_sent(use_client):
    msgs = use_client(NewSdkMessages())
    text, usage = llm_client._complete_claude(_config(0.0), "sys", "q", 100)
    assert text == "ok" and usage == {"input": 3, "output": 2, "total": 5}
    assert len(msgs.calls) == 1
    assert llm_client._claude_temperature_supported is False


def test_old_sdk_temperature_still_sent(use_client):
    msgs = use_client(OldSdkMessages())
    llm_client._complete_claude(_config(0.3), "sys", "q", 100)
    assert msgs.calls[0]["temperature"] == 0.3


def test_no_temperature_configured_never_inspects(use_client):
    msgs = use_client(NewSdkMessages())
    llm_client._complete_claude(_config(None), "sys", "q", 100)
    assert len(msgs.calls) == 1
    assert llm_client._claude_temperature_supported is None


def test_runtime_rejection_retries_without_and_remembers(use_client):
    msgs = use_client(OpaqueMessages())
    text, _ = llm_client._complete_claude(_config(0.0), "sys", "q", 100)
    assert text == "ok"
    assert "temperature" not in msgs.calls[0]
    assert llm_client._claude_temperature_supported is False
    llm_client._complete_claude(_config(0.0), "sys", "q2", 100)
    assert len(msgs.calls) == 2  # second call went straight through, no retry needed


def test_unrelated_type_error_still_raises(use_client):
    use_client(OtherTypeErrorMessages())
    with pytest.raises(TypeError, match="unrelated"):
        llm_client._complete_claude(_config(0.0), "sys", "q", 100)


def test_real_installed_sdk_signature_is_handled(use_client):
    """Whatever anthropic version is installed here, the check must not crash
    and must agree with the real signature."""
    anthropic = pytest.importorskip("anthropic")
    import inspect
    client = anthropic.Anthropic(api_key="sk-test-not-real")
    expected = "temperature" in inspect.signature(client.messages.create).parameters
    assert llm_client._claude_accepts_temperature(client) is expected
