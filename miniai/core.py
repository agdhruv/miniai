from typing import List, Dict, Optional, Union, Callable
import functools

from miniai.config import Config
from miniai.providers import get_provider, BaseProvider, PROVIDERS
from miniai.providers.base import Response


class AI:
    """Main MiniAI interface."""
    
    def __init__(self):
        self.config = Config()
        self._providers = {}
    
    def __repr__(self):
        return f"AI(provider={self.get_active_provider()}, default_model={self.get_model()})"
    
    def _get_provider(self, provider_name: Optional[str] = None) -> BaseProvider:
        """Get or create a provider instance."""
        provider_name = provider_name or self.config.get_active_provider()
        
        if provider_name not in self._providers:
            self._providers[provider_name] = get_provider(self.config, provider_name)
        return self._providers[provider_name]
    
    def use(self, provider: str) -> None:
        """Set the AI provider to use."""
        self.config.use(provider)
    
    def get_active_provider(self) -> str:
        """Returns which provider would be used for the next call."""
        return self.config.get_active_provider()

    def get_available_providers(self) -> List[str]:
        """Returns a list of all available providers."""
        return list(PROVIDERS.keys())
    
    def set_api_key(self, provider: str, key: str) -> None:
        """Set the API key for a provider."""
        self.config.set_api_key(provider, key)
        # Clear cached provider since it used the old key
        self._providers.pop(provider, None)
    
    def get_api_key(self, provider: Optional[str] = None) -> str:
        """Get the API key for the current or specified provider."""
        return self.config.get_api_key(provider)
    
    def set_model(self, model: str, provider: Optional[str] = None) -> None:
        """Set the model to use for a provider."""
        self.config.set_model(model, provider)
    
    def get_model(self, provider: Optional[str] = None) -> str:
        """Get the model for the current or specified provider."""
        return self.config.get_model(provider)
    
    def classify(self, text: str, categories: List[str], *, raw_response: bool = False, **kwargs) -> Union[str, Response]:
        """Classify text into one of the provided categories."""
        provider = kwargs.pop("provider", None)
        return self._get_provider(provider).classify(text, categories, raw_response=raw_response, **kwargs)
    
    def extract(self, text: str, entities: List[str], *, raw_response: bool = False, **kwargs) -> Union[Dict[str, List[str]], Response]:
        """Extract entities from text."""
        provider = kwargs.pop("provider", None)
        return self._get_provider(provider).extract(text, entities, raw_response=raw_response, **kwargs)
    
    def summarize(self, text: str, *, raw_response: bool = False, **kwargs) -> Union[str, Response]:
        """Summarize text."""
        provider = kwargs.pop("provider", None)
        return self._get_provider(provider).summarize(text, raw_response=raw_response, **kwargs)
    
    def translate(self, text: str, to: str, *, raw_response: bool = False, **kwargs) -> Union[str, Response]:
        """Translate text to another language."""
        provider = kwargs.pop("provider", None)
        return self._get_provider(provider).translate(text, to, raw_response=raw_response, **kwargs)
    
    def ask(self, question: str, *, format_instructions: Optional[str] = None, images: Optional[List[Union[str, bytes]]] = None, raw_response: bool = False, **kwargs) -> Union[str, Response]:
        """Answer a question, optionally with structured output and images.
        
        Args:
            question: The question or instruction to answer
            format_instructions: Optional instructions for how to format the response
                                (e.g. "Return JSON with {person: true/false, animal: true/false}")
            images: Optional list of image data (file paths or bytes)
            raw_response: Whether to return the full response object
            **kwargs: Additional arguments to pass to the provider
            
        Returns:
            The response (formatted as specified by format_instructions) or Response object
            
        Example:
            # Regular question
            result = ai.ask("What are the planets in our solar system?")
            
            # With structured output
            result = ai.ask(
                "For the sentence 'John saw a cat in New York yesterday', tell me what it contains",
                format_instructions="Output in JSON format with these keys: person (true/false), animal (true/false), place (true/false)"
            )
            # Returns: JSON string '{"person": true, "animal": true, "place": true}'
            
            # With images
            result = ai.ask(
                "What's in this image?",
                images=["path/to/image.jpg"]
            )
            
            # With raw response
            response = ai.ask("Hello", raw_response=True)
            print(response.content)  # The text content
            print(response.raw_response)  # The full provider response
        """
        provider = kwargs.pop("provider", None)
        return self._get_provider(provider).ask(question, format_instructions=format_instructions, images=images, raw_response=raw_response, **kwargs)
    
    def embedding(self, text: str, *, raw_response: bool = False, **kwargs) -> Union[List[float], Response]:
        """Get embedding vector for the given text.
        
        Args:
            text: The text to get embeddings for
            raw_response: Whether to return the full response object
            **kwargs: Additional arguments to pass to the provider
            
        Returns:
            List of floats representing the text embedding or Response object
            
        Example:
            embedding = ai.embedding("Hello world")
            # Returns: List of floats representing the text embedding
        """
        provider = kwargs.pop("provider", None)
        return self._get_provider(provider).embedding(text, raw_response=raw_response, **kwargs)
    
    def text_to_speech(self, text: str, *, raw_response: bool = False, **kwargs) -> Union[bytes, Response]:
        """Convert text to speech.
        
        Args:
            text: The text to convert to speech
            raw_response: Whether to return the full response object
            **kwargs: Additional arguments to pass to the provider
                For provider openai:
                    voice: The voice to use (default: "alloy")
                    model: TTS model to use (default: "tts-1")
                    instructions: Additional instructions for the TTS model (e.g. "Speak in a British accent")
                    response_format: Audio format (mp3, opus, aac, flac) (default: "mp3")
                    speed: Speech speed, between 0.25 and 4.0 (default: 1.0)

        Returns:
            Audio data as bytes or Response object
            
        Example:
            audio = ai.text_to_speech("Hello world")
        """
        provider = kwargs.pop("provider", None)
        return self._get_provider(provider).text_to_speech(text, raw_response=raw_response, **kwargs)
    
    def speech_to_text(self, audio_data: Union[bytes, str], *, raw_response: bool = False, **kwargs) -> Union[Dict, Response]:
        """Convert speech to text.
        
        Args:
            audio_data: The audio data to convert to text (bytes) or a path to an audio file (str)
            raw_response: Whether to return the full response object
            **kwargs: Additional arguments to pass to the provider
            
        Returns:
            Dict with transcription and other metadata or Response object
            
        Example:
            transcription = ai.speech_to_text("audio.mp3")
        """
        provider = kwargs.pop("provider", None)
        return self._get_provider(provider).speech_to_text(audio_data, raw_response=raw_response, **kwargs)
    
    def function(self, func: Optional[Callable] = None, *, system_prompt: Optional[str] = None):
        """Decorator to convert a function to use AI for its implementation.
        
        Example:
            @ai.function
            def generate_poem(topic, style):
                '''Generate a poem about {topic} in the style of {style}.'''
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Get function signature and docstring
                sig = func.__doc__ or func.__name__
                
                # Replace placeholders in the docstring with actual arguments
                if args:
                    # Get parameter names from function definition
                    import inspect
                    param_names = list(inspect.signature(func).parameters.keys())
                    
                    # Create a dict mapping parameter names to argument values
                    args_dict = dict(zip(param_names, args))
                    
                    # Update with any keyword arguments
                    args_dict.update(kwargs)
                    
                    # Format the docstring with the arguments
                    try:
                        sig = sig.format(**args_dict)
                    except KeyError:
                        # If formatting fails, just use the original docstring
                        pass
                
                # Build the prompt
                prompt = sig
                if system_prompt:
                    kwargs["system"] = system_prompt
                
                # Call the AI
                return self.ask(prompt, **kwargs)
            
            return wrapper
        
        # Handle both @ai.function and @ai.function(system_prompt="...")
        if func is None:
            return decorator
        return decorator(func)


# Global AI instance
ai = AI()
