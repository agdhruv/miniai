# MiniAI

A minimalist Python library for byte-sized AI tasks. No complex chains, no confusing abstractions, just AI that works.

## The Problem

Using AI models for simple tasks comes with too much overhead:
- Writing boilerplate API calls
- Managing prompts and message lists
- Configuring providers and models
- Learning complex frameworks like LangChain

These barriers make it difficult to quickly prototype and integrate AI capabilities into your workflow.

## The Solution

MiniAI provides a dead-simple interface for common AI tasks:

```python
from miniai import ai

# Simple tasks with just one line
text = ai.ask("Write a haiku about Python")
category = ai.classify("I love this product!", ["positive", "negative", "neutral"])
entities = ai.extract("Apple was founded by Steve Jobs in 1976", ["people", "organizations", "dates"])
summary = ai.summarize("Very long text goes here...")
translated = ai.translate("Hello world", to="spanish")
answer = ai.ask("Who was Ada Lovelace?")
image_analysis = ai.ask("What's in this image?", images=["Logo.png"])
image_url_analysis = ai.ask("What's in this image?", images=["https://example.com/image.jpg"])

# Configure once, use anywhere
ai.set_api_key("openai", "your-api-key-here")  # Or use environment variables
ai.use("anthropic")  # Switch providers dynamically
ai.set_model("gpt-4-turbo")  # Set model for current provider

# Mock mode for testing without API keys
ai.config.mock(True)
result = ai.ask("This will return a mock response")  # No API key needed
```

## Turn Any Function into an AI Function

The most powerful feature is the function decorator:

```python
@ai.function
def generate_poem(topic, style):
    """Generate a poem about {topic} in the style of {style}."""

poem = generate_poem("autumn leaves", "haiku")
print(poem)

# With system prompt
@ai.function(system_prompt="You are a professional software engineer.")
def write_code(task, language):
    """Write {language} code to {task}. Include comments."""

code = write_code("sort a list", "python")
```

## Installation

```bash
pip install miniai
```

## Why Choose MiniAI?

- 🚀 **Simple API**: Just one import, intuitive methods
- 🔧 **Zero configuration**: Works out of the box (with environment variables)
- 🧠 **Smart defaults**: Uses appropriate models for each task
- 🔄 **Model agnostic**: Works with OpenAI, Anthropic, and more
- 📦 **Lightweight**: No heavy dependencies
- 🧩 **Extensible**: Easy to add new providers and tasks
- 🧪 **Test-friendly**: Mock mode for development without API keys

## API Reference

### Core Functions

- `ai.ask(question, format_instructions=None, images=None, raw_response=False, **kwargs)` - Answer a question, optionally with images and format instructions
- `ai.classify(text, categories, raw_response=False, **kwargs)` - Classify text into categories
- `ai.extract(text, entities, raw_response=False, **kwargs)` - Extract entities from text
- `ai.summarize(text, raw_response=False, **kwargs)` - Summarize text
- `ai.translate(text, to, raw_response=False, **kwargs)` - Translate text to another language
- `ai.embedding(text, raw_response=False, **kwargs)` - Get embedding vector for text
- `ai.text_to_speech(text, raw_response=False, **kwargs)` - Convert text to speech (OpenAI only)
- `ai.speech_to_text(audio_data, raw_response=False, **kwargs)` - Convert speech to text (OpenAI only)

### Configuration

- `ai.set_api_key(provider, key)` - Set API key for a provider
- `ai.use(provider)` - Switch to a different provider
    - `ai.use('mock')` - Use the mock provider for testing
- `ai.set_model(model, provider=None)` - Set model for current or specified provider
- `ai.get_active_provider()` - Get current provider
- `ai.get_available_providers()` - List all available providers

### Response Format

By default, all functions return the processed response that makes sense for the task (e.g. a string for text generation, a list of entities for entity extraction, etc.).

All functions also support a `raw_response` parameter that returns a `Response` object with:
- `content`: The processed response
- `raw_response`: The complete response from the provider

Example:
```python
response = ai.ask("What is the capital of France?", raw_response=True)
print(response.content)  # The text answer
print(response.raw_response)  # Full provider response
```

More examples in the [examples](examples) directory.

## Decorator

- `@ai.function(system_prompt=None)` - Turn any function into an AI function

## License

MIT
