import os

from src.ai.client import OpenAIClient, create_ai_client
from src.ai.openclaw_provider import resolve_openclaw_ai_config
from src.models import AIConfig, AIProvider


def test_resolves_explicit_openclaw_model_spec_to_openai_config():
    config = AIConfig(provider=AIProvider.OPENCLAW, model="openai/gpt-5.5", languages=["ru"])

    resolved = resolve_openclaw_ai_config(config)

    assert resolved.provider == AIProvider.OPENAI
    assert resolved.model == "gpt-5.5"
    assert resolved.api_key_env == "OPENAI_API_KEY"
    assert resolved.languages == ["ru"]


def test_resolves_model_from_openclaw_defaults_model_primary(tmp_path, monkeypatch):
    openclaw_config = tmp_path / "openclaw.json"
    openclaw_config.write_text(
        """
        {
          agents: {
            defaults: {
              model: {
                primary: "anthropic/claude-sonnet-4.5"
              }
            }
          }
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("HORIZON_OPENCLAW_CONFIG_PATH", str(openclaw_config))
    config = AIConfig(provider=AIProvider.OPENCLAW, model="openclaw")

    resolved = resolve_openclaw_ai_config(config)

    assert resolved.provider == AIProvider.ANTHROPIC
    assert resolved.model == "claude-sonnet-4.5"
    assert resolved.api_key_env == "ANTHROPIC_API_KEY"


def test_resolves_openclaw_agent_override_before_defaults(tmp_path, monkeypatch):
    openclaw_config = tmp_path / "openclaw.json"
    openclaw_config.write_text(
        """
        {
          agents: {
            defaults: { model: { primary: "openai/gpt-default" } },
            list: [
              { id: "main" },
              {
                id: "spermwhale",
                model: { primary: "anthropic/claude-agent" }
              }
            ]
          }
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("HORIZON_OPENCLAW_CONFIG_PATH", str(openclaw_config))
    config = AIConfig(
        provider=AIProvider.OPENCLAW,
        model="openclaw",
        openclaw_agent_id="spermwhale",
    )

    resolved = resolve_openclaw_ai_config(config)

    assert resolved.provider == AIProvider.ANTHROPIC
    assert resolved.model == "claude-agent"
    assert resolved.api_key_env == "ANTHROPIC_API_KEY"


def test_resolves_openclaw_custom_openai_provider_config(tmp_path, monkeypatch):
    openclaw_config = tmp_path / "openclaw.json"
    openclaw_config.write_text(
        """
        {
          agents: {
            defaults: { model: { primary: "moonshot/kimi-k2" } }
          },
          models: {
            providers: {
              moonshot: {
                api: "openai-completions",
                apiKey: "${MOONSHOT_API_KEY}",
                baseUrl: "https://api.moonshot.ai/v1"
              }
            }
          }
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("HORIZON_OPENCLAW_CONFIG_PATH", str(openclaw_config))
    config = AIConfig(provider=AIProvider.OPENCLAW, model="openclaw")

    resolved = resolve_openclaw_ai_config(config)

    assert resolved.provider == AIProvider.OPENAI
    assert resolved.model == "kimi-k2"
    assert resolved.api_key_env == "MOONSHOT_API_KEY"
    assert resolved.base_url == "https://api.moonshot.ai/v1"


def test_loads_openclaw_env_file_without_overriding_existing_values(tmp_path, monkeypatch):
    env_file = tmp_path / "openclaw.env"
    env_file.write_text(
        "OPENAI_API_KEY=from-file\nHORIZON_OPENCLAW_MODEL=openai/gpt-from-env\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "already-set")
    config = AIConfig(
        provider=AIProvider.OPENCLAW,
        model="openclaw",
        openclaw_env_path=str(env_file),
    )

    resolved = resolve_openclaw_ai_config(config)

    assert resolved.provider == AIProvider.OPENAI
    assert resolved.model == "gpt-from-env"
    assert resolved.api_key_env == "OPENAI_API_KEY"
    assert os.environ["OPENAI_API_KEY"] == "already-set"


def test_ollama_model_uses_openai_compatible_loopback_base_url(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    config = AIConfig(provider=AIProvider.OPENCLAW, model="ollama/gemma4:e2b")

    resolved = resolve_openclaw_ai_config(config)

    assert resolved.provider == AIProvider.OPENAI
    assert resolved.model == "gemma4:e2b"
    assert resolved.api_key_env is None
    assert resolved.base_url == "http://127.0.0.1:11434/v1"


def test_create_ai_client_delegates_openclaw_provider_to_existing_client(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    client = create_ai_client(AIConfig(provider=AIProvider.OPENCLAW, model="openai/gpt-test"))

    assert isinstance(client, OpenAIClient)
    assert client.model == "gpt-test"