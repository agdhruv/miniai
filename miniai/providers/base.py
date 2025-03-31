from typing import List, Dict, Optional, Union, Type, Any
import json

class Response:
    """Wrapper for provider responses."""
    
    def __init__(self, content: Any, raw_response: Any):
        self.content = content
        self.raw_response = raw_response
    
    def __str__(self) -> str:
        return str(self.content)
    
    def __repr__(self) -> str:
        return f"Response(content={repr(self.content)}, raw_response={repr(self.raw_response)})"

class BaseProvider:
    """Base class for AI providers."""
    
    def __init__(self, config: Type):
        self._client = None
        self.config = config
    
    def classify(self, text: str, categories: List[str], *, raw_response: bool = False, **kwargs) -> Union[str, Response]:
        """Classify text into one of the provided categories."""
        prompt = f"Classify the following text into exactly one of these categories: {', '.join(categories)}.\n\nText: {text}\n\nCategory:"
        response = self.ask(prompt, raw_response=True, **kwargs)
        result = response.content.strip()
        return Response(result, response.raw_response) if raw_response else result
    
    def extract(self, text: str, entities: List[str], *, raw_response: bool = False, **kwargs) -> Union[Dict[str, List[str]], Response]:
        """Extract entities from text."""
        prompt = f"Extract the following entities from the text: {', '.join(entities)}.\n\nText: {text}\n\nProvide results in JSON format with entity types as keys and lists of extracted items as values."
        response = self.ask(prompt, raw_response=True, **kwargs)
        
        # Try to parse JSON response
        try:
            result = json.loads(response.content)
        except json.JSONDecodeError:
            # If parsing fails, throw an error
            raise ValueError(f"Got invalid JSON response from model: {response.content}.\nTry specifying the format in the system prompt.")
            
        return Response(result, response.raw_response) if raw_response else result
    
    def summarize(self, text: str, *, raw_response: bool = False, **kwargs) -> Union[str, Response]:
        """Summarize text."""
        prompt = f"Summarize the following text:\n\n{text}"
        response = self.ask(prompt, raw_response=True, **kwargs)
        result = response.content.strip()
        return Response(result, response.raw_response) if raw_response else result
    
    def translate(self, text: str, to: str, *, raw_response: bool = False, **kwargs) -> Union[str, Response]:
        """Translate text to another language."""
        prompt = f"Translate the following text to {to}:\n\n{text}"
        response = self.ask(prompt, raw_response=True, **kwargs)
        result = response.content.strip()
        return Response(result, response.raw_response) if raw_response else result
    
    def ask(self, question: str, *, format_instructions: Optional[str] = None, images: Optional[List[Union[str, bytes]]] = None, raw_response: bool = False, **kwargs) -> Union[str, Response]:
        """Answer a question using the provider."""
        raise NotImplementedError
    
    def embedding(self, text: str, *, raw_response: bool = False, **kwargs) -> Union[List[float], Response]:
        """Get embedding vector for the given text.
        
        Args:
            text: The text to get embeddings for
            raw_response: Whether to return the full response object
            **kwargs: Additional arguments to pass to the provider
            
        Returns:
            List of floats representing the text embedding or Response object
        """
        raise NotImplementedError

    def text_to_speech(self, text: str, *, raw_response: bool = False, **kwargs) -> Union[bytes, Response]:
        """Convert text to speech.
        
        Args:
            text: The text to convert to speech
            raw_response: Whether to return the full response object
            **kwargs: Additional arguments to pass to the provider
            
        Returns:
            Audio data as bytes or Response object
        """
        raise NotImplementedError

    def speech_to_text(self, audio_data: bytes, *, raw_response: bool = False, **kwargs) -> Union[Dict, Response]:
        """Convert speech to text.
        
        Args:
            audio_data: The audio data to convert to text
            raw_response: Whether to return the full response object
            **kwargs: Additional arguments to pass to the provider
            
        Returns:
            Dict with transcription and other metadata or Response object
        """
        raise NotImplementedError 