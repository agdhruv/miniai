import pytest
from unittest.mock import patch, MagicMock
from miniai.core import AI
from miniai.config import Config
from miniai.providers import BaseProvider, PROVIDERS

@pytest.fixture
def ai():
    return AI()

# Test Initialization
def test_initialization(ai):
    assert isinstance(ai.config, Config)
    assert ai.get_active_provider() == Config._default_provider
    assert ai.get_active_provider() == ai.config.get_active_provider()
    assert ai.get_available_providers() == list(PROVIDERS.keys())
    assert ai.config.get_model() == PROVIDERS[Config._default_provider]["default_model"]
    
    for provider in PROVIDERS.keys():
        assert ai.get_model(provider) == PROVIDERS[provider]["default_model"]

# Test Provider Management
def test_provider_management(ai):
    ai.use("anthropic")
    assert ai.get_active_provider() == "anthropic"
    assert ai.get_active_provider() == ai.config.get_active_provider()
    assert ai.get_model() == PROVIDERS["anthropic"]["default_model"]
    assert ai.config.get_model() == ai.get_model()
    
    with pytest.raises(ValueError):
        ai.use("invalid_provider")

# Test API Key Management
def test_api_key_management(ai):
    ai.set_api_key("openai", "test-key")
    assert ai.get_api_key() == "test-key"
    assert ai.config.get_api_key() == "test-key"
    with pytest.raises(ValueError):
        ai.set_api_key("invalid_provider", "test-key")

# Test Model Management
def test_model_management(ai):
    ai.set_model("gpt-4")
    assert ai.get_model() == "gpt-4"
    assert ai.config.get_model() == "gpt-4"
    
    ai.use("anthropic")
    ai.set_model("model-name")
    assert ai.get_model() == "model-name"
    assert ai.config.get_model() == "model-name"
    
    ai.set_model("gpt-5", "openai")
    assert ai.get_model("openai") == "gpt-5"
    assert ai.config.get_model("openai") == "gpt-5"
    
    # Check that the default model is still the same
    assert ai.get_model() == "model-name"
    
    with pytest.raises(ValueError):
        ai.set_model("test-model", "invalid_provider")

# Test Functionality
def test_functionality(ai):
    ai.use("mock")
    assert ai.get_active_provider() == "mock"
    assert ai.get_model() == "mock"
    assert ai.get_api_key() == "mock-key"
    
    # Check functionality with provider and model parameters
    result = ai.classify("text", ["category1", "category2"])
    assert result in ["category1", "category2"]
    
    result = ai.extract("test text", ["person", "location"])
    assert isinstance(result, dict)
    assert "person" in result
    assert "location" in result
    assert isinstance(result["person"], list)
    assert isinstance(result["location"], list)
    
    result = ai.summarize("Long text")
    assert result == "This is a summary of the text."
    
    result = ai.translate("Hello world", "french")
    assert result == "Bonjour le monde"
    
    result = ai.ask("What is the capital of France?")
    assert isinstance(result, str)
    assert len(result) > 0
    
    result = ai.embedding("Hello world")
    assert len(result) == 1536
    
    # Check that the provider is cached
    provider1 = ai._get_provider("mock")
    provider2 = ai._get_provider()
    assert provider1 is provider2
    
    # Check provider clearing on API key change
    ai.set_api_key("mock", "new-key")
    provider3 = ai._get_provider()
    assert provider3 is not provider1

# Test Function Decorator
def test_function_decorator(ai):
    ai.use("mock")
    
    @ai.function
    def generate_poem(topic, style):
        """Generate a poem about {topic} in the style of {style}."""
    
    result = generate_poem("nature", "haiku")
    assert isinstance(result, str)
    assert len(result) > 0
    
    @ai.function(system_prompt="You are a poet")
    def generate_song(theme):
        """Write a song about {theme}."""
    
    result = generate_song("love")
    assert isinstance(result, str)
    assert len(result) > 0
