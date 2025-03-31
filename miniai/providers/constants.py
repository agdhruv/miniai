"""Provider-specific constants for MiniAI."""

# Registry of available providers with their details/defaults
PROVIDERS = {
    "openai": {
        "default_model": "gpt-4o",
        "default_embedding_model": "text-embedding-3-small",
        "default_tts_model": "gpt-4o-mini-tts",
        "default_tts_voice": "alloy",
        "default_stt_model": "gpt-4o-transcribe",
        "class": None  # Will be set in __init__.py
    },
    "anthropic": {
        "default_model": "claude-3-5-haiku-latest",
        "class": None  # Will be set in __init__.py
    },
    "mock": {
        "default_model": "mock",
        "class": None  # Will be set in __init__.py
    }
}