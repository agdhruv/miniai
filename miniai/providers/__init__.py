"""Provider registry for MiniAI."""

from miniai.providers.base import BaseProvider
from miniai.providers.openai import OpenAIProvider
from miniai.providers.anthropic import AnthropicProvider
from miniai.providers.mock import MockProvider
from miniai.providers.constants import PROVIDERS

PROVIDERS["openai"]["class"] = OpenAIProvider
PROVIDERS["anthropic"]["class"] = AnthropicProvider
PROVIDERS["mock"]["class"] = MockProvider

def get_provider(config, provider_name):
    """Get a provider instance by name."""
    if provider_name not in PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider_name}")
    
    provider_class = PROVIDERS[provider_name]["class"]
    return provider_class(config)

__all__ = [
    'BaseProvider',
    'OpenAIProvider',
    'AnthropicProvider',
    'MockProvider',
    'PROVIDERS',
    'get_provider'
] 