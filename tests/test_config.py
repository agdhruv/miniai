import os
import pytest
from unittest.mock import patch
from miniai.config import Config
from miniai.providers import PROVIDERS

@pytest.fixture
def config():
    return Config()

def test_initialization(config):
    assert config.provider == "openai"
    assert config.get_active_provider() == "openai"
    assert config.api_keys["mock"] == "mock-key"
    for provider in PROVIDERS.keys():
        assert provider in config.models
        assert config.models[provider] == PROVIDERS[provider]["default_model"]

def test_provider_management(config):
    assert config.provider == "openai"
    # Test valid provider
    config.use("anthropic")
    assert config.provider == "anthropic"
    
    # Test invalid provider
    with pytest.raises(ValueError):
        config.use("invalid_provider")

def test_api_key_management(config):
    # Test setting and getting API key
    config.set_api_key("openai", "test-key")
    assert config.get_api_key() == "test-key"
    assert config.api_keys["openai"] == "test-key"
    
    # Test getting API key for specific provider
    config.set_api_key("anthropic", "anthropic-key")
    assert config.get_api_key("anthropic") == "anthropic-key"
    
    # Test invalid provider
    with pytest.raises(ValueError):
        config.set_api_key("invalid_provider", "test-key")
    
    # Test missing API key
    config.set_api_key("openai", "")
    with pytest.raises(ValueError):
        config.get_api_key()

def test_model_management(config):
    # Test setting and getting model
    assert config.get_model() == "gpt-4o"
    config.set_model("gpt-4")
    assert config.get_model() == "gpt-4"
    
    # Test setting and getting model for specific provider
    config.set_model("claude-2", "anthropic")
    assert config.get_model("anthropic") == "claude-2"
    
    # Test invalid provider
    with pytest.raises(ValueError):
        config.set_model("test-model", "invalid_provider")
    
    # Test non-existent provider in models
    assert config.get_active_provider() == "openai"
    del config.models["openai"]
    with pytest.raises(ValueError):
        config.get_model()

def test_environment_variables():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
        config = Config()
        assert config.get_active_provider() == "openai"
        assert config.provider == "openai"
        assert config.get_api_key("openai") == "env-key"
