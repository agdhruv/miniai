import os
from typing import Optional
from miniai.providers import PROVIDERS

class Config:
    """Configuration for MiniAI."""
    
    # Default provider
    _default_provider = "openai"
    
    def __init__(self):
        # API keys (read from environment variables by default)
        self.api_keys = {
            provider: os.environ.get(f"{provider.upper()}_API_KEY", "")
            for provider in PROVIDERS.keys()
        }
        self.api_keys["mock"] = "mock-key"  # Mock provider always has a key
        
        # Current provider selection
        self.provider = self._default_provider
        
        # Model configurations
        self.models = {provider: PROVIDERS[provider]["default_model"] for provider in PROVIDERS.keys()}
    
    def use(self, provider: str) -> None:
        """Set the AI provider to use."""
        if provider not in PROVIDERS.keys():
            raise ValueError(f"Unsupported provider: {provider}")
        self.provider = provider
    
    def get_active_provider(self) -> str:
        """Returns which provider would be used for the next call."""
        return self.provider
    
    def set_api_key(self, provider: str, key: str) -> None:
        """Set the API key for a provider."""
        if provider not in PROVIDERS.keys():
            raise ValueError(f"Unsupported provider: {provider}")
        self.api_keys[provider] = key
    
    def get_api_key(self, provider: Optional[str] = None) -> str:
        """Get the API key for the current or specified provider."""
        provider = provider or self.provider
        key = self.api_keys.get(provider, "")
        if not key:
            raise ValueError(f"No API key found for provider {provider}. "
                             f"Set it with ai.set_api_key('{provider.upper()}', 'your-key') or set the environment variable {provider.upper()}_API_KEY")
        return key
    
    def set_model(self, model: str, provider: Optional[str] = None) -> None:
        """Set the model to use for a provider."""
        provider = provider or self.provider
        if provider not in PROVIDERS.keys():
            raise ValueError(f"Unsupported provider: {provider}")
        self.models[provider] = model
    
    def get_model(self, provider: Optional[str] = None) -> str:
        """Get the model for the current or specified provider."""
        provider = provider or self.provider
        if provider not in self.models:
            raise ValueError(f"No model configured for provider {provider}")
        return self.models[provider]