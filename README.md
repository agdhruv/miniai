# MiniAI 

A minimalist Python library for byte-sized AI tasks. No complex chains, no confusing abstractions, just AI that works.

```bash
pip install miniai
```

## The Problem

Using AI models for simple tasks comes with too much overhead: Writing boilerplate API calls, managing prompts and message lists, configuring providers and models, learning complex frameworks like LangChain.

These barriers make it annoying to quickly prototype and integrate AI capabilities into your workflow.

## The Solution

MiniAI provides a dead-simple interface for common AI tasks.

```python
from miniai import ai

text = ai.ask("Write a haiku about Python")
```

`ai.ask()` is a powerful function that can do anything. But here are some helper functions for common tasks:
```python
image_analysis = ai.ask("What's in this image?", images=["image.png", "https://example.com/image.png"])

category = ai.classify("I love this product!", ["positive", "negative", "neutral"])

entities = ai.extract("Apple was founded by Steve Jobs in 1976", ["people", "organizations", "dates"])

summary = ai.summarize("Very long text")

translated = ai.translate("Hello world", to="spanish")

audio = ai.text_to_speech("Hello world")

text = ai.speech_to_text(audio) # Transcription
```

See more examples in the [examples](examples) directory.

## Intuitive and Flexible

The only terminology you need to know is "provider" and "model". A provider is an AI service/company like OpenAI or Anthropic. A model is a specific AI model like GPT-4o or Claude 3.5.

The defaults will usually get your work done, but if you need more control, it's super intuitive. Need a different provider? A different model? A system prompt? Additional settings? just pass a parameter to `ai.ask()`.

```python
text = ai.ask("Write a haiku about Python", system_prompt="Respond in Spanish") # uses gpt-4o from openai by default

text = ai.ask("Write a haiku about Python", provider="anthropic")

text = ai.ask("Write a haiku about Python", provider="openai", model="gpt-3.5-turbo", temperature=0.5, max_tokens=100)
```

If you need the raw response from the provider, just pass `raw_response=True`.
```python
response = ai.ask("Write a haiku about Python", raw_response=True)
print(response.content) # The text answer
print(response.raw_response) # Full provider response
```

## Turn Any Function into an AI Function

The function decorator is a powerful way to turn any function into an AI function.

```python
@ai.function
def generate_poem(topic, style):
    """Generate a poem about {topic} in the style of {style}."""

poem = generate_poem("autumn leaves", "haiku")
print(poem)

# With system prompt and model
@ai.function(system_prompt="You are a professional software engineer.", model="gpt-4o-mini")
def write_code(task, language):
    """Write {language} code to {task}. Include comments."""

code = write_code("sort a list", "python")
```

## Why Choose MiniAI?

- 🚀 **Simple API**: Just one import, intuitive methods
- 🔧 **Zero configuration**: Works out of the box (with environment variables)
- 🧠 **Smart defaults**: Uses appropriate models for each task
- 🔄 **Model agnostic**: Works with OpenAI, Anthropic, and more coming soon
- 📦 **Lightweight**: No heavy dependencies
- 🧩 **Extensible**: Easy to add new providers and tasks
- 🛠️ **Error handling**: Clear and helpful error messages

## API Reference

**API Keys**: MiniAI requires API keys for the desired providers. You can set these using environment variables (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) or programmatically using `ai.set_api_key(provider, key)`.

### Core Function

| Function | Description |
|----------|-------------|
| `ai.ask(question, system_prompt=None, messages=None, format_instructions=None, images=None, raw_response=False, **kwargs)` | General-purpose AI request supporting text, images, format instructions, system prompts, and conversation history (`messages`). Uses default provider/model (OpenAI GPT-4o initially), override via arguments (`provider=`, `model=`, `**kwargs`). |

### Helper Functions

| Function | Description |
|----------|-------------|
| `ai.classify(text, categories, raw_response=False, **kwargs)` | Classify text into categories |
| `ai.extract(text, entities, raw_response=False, **kwargs)` | Extract entities from text |
| `ai.summarize(text, raw_response=False, **kwargs)` | Summarize text |
| `ai.translate(text, to, raw_response=False, **kwargs)` | Translate text to another language |
| `ai.embedding(text, raw_response=False, **kwargs)` | Get embedding vector for text |
| `ai.text_to_speech(text, raw_response=False, **kwargs)` | Convert text to speech (OpenAI only) |
| `ai.speech_to_text(audio_data, raw_response=False, **kwargs)` | Convert speech to text (OpenAI only) |

**Accessing Raw Provider Output**: By default, MiniAI functions return a directly usable result (e.g., a string for `ai.ask`, a list for `ai.extract`). To get the complete, unmodified response from the underlying AI provider's API, pass `raw_response=True`. This returns a `Response` object containing:
- `content`: The processed output (same as when `raw_response=False`).
- `raw_response`: The full, untouched response object from the provider.

### Configuration

| Function | Description |
|----------|-------------|
| `ai.set_api_key(provider, key)` | Set API key for a provider |
| `ai.use(provider)` | Switch to a different provider |
| `ai.set_model(model, provider=None)` | Set model for current or specified provider |
| `ai.get_active_provider()` | Get current provider |
| `ai.get_available_providers()` | List all available providers |

> **Note:** Use `ai.use('mock')` to enable the mock provider for testing without API keys.

### Decorator

| Decorator | Description |
|-----------|-------------|
| `@ai.function(system_prompt=None, messages=None, format_instructions=None, images=None, raw_response=False, **kwargs)` | Turn any function into an AI function. The function's docstring is used as the prompt. See [examples](examples/decorator_examples.ipynb). |

## License

MIT

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for step-by-step guidelines.
