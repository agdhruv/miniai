from typing import List, Dict, Optional, Union, Type
import json
import random
from miniai.providers.base import BaseProvider, Response

class MockProvider(BaseProvider):
    """Mock provider for testing without API keys."""
    
    _mock_responses = {
        "classify": {
            "positive": "positive",
            "negative": "negative",
            "neutral": "neutral"
        },
        "extract": {
            "people": ["John", "Jane", "Bob"],
            "places": ["New York", "London", "Tokyo"],
            "organizations": ["Google", "Microsoft", "Apple"]
        },
        "summarize": "This is a summary of the text.",
        "translate": {
            "english": "Hello world",
            "spanish": "Hola mundo",
            "french": "Bonjour le monde",
            "german": "Hallo Welt",
            "japanese": "こんにちは世界"
        },
        "ask": [
            "Here is a response to your question.",
            "The answer depends on various factors.",
            "That's an interesting question. Let me think about it.",
            "Based on my knowledge, I would say that...",
            "I don't have enough information to answer that question."
        ]
    }
    
    def classify(self, text: str, categories: List[str], *, raw_response: bool = False, **kwargs) -> Union[str, Response]:
        """Mock implementation of classify."""
        result = random.choice(categories)
        mock_response = {"choices": [{"message": {"content": result}}]}
        return Response(result, mock_response) if raw_response else result
    
    def extract(self, text: str, entities: List[str], *, raw_response: bool = False, **kwargs) -> Union[Dict[str, List[str]], Response]:
        """Mock implementation of extract."""
        result = {}
        for entity in entities:
            entity_lower = entity.lower()
            if entity_lower in self._mock_responses["extract"]:
                result[entity] = self._mock_responses["extract"][entity_lower]
            else:
                result[entity] = []
        mock_response = {"choices": [{"message": {"content": json.dumps(result)}}]}
        return Response(result, mock_response) if raw_response else result
    
    def summarize(self, text: str, *, raw_response: bool = False, **kwargs) -> Union[str, Response]:
        """Mock implementation of summarize."""
        result = self._mock_responses["summarize"]
        mock_response = {"choices": [{"message": {"content": result}}]}
        return Response(result, mock_response) if raw_response else result
    
    def translate(self, text: str, to: str, *, raw_response: bool = False, **kwargs) -> Union[str, Response]:
        """Mock implementation of translate."""
        to_lower = to.lower()
        if to_lower in self._mock_responses["translate"]:
            result = self._mock_responses["translate"][to_lower]
        else:
            result = f"Translated to {to}: {text}"
        mock_response = {"choices": [{"message": {"content": result}}]}
        return Response(result, mock_response) if raw_response else result
    
    def ask(self, question: str, *, format_instructions: Optional[str] = None, images: Optional[List[Union[str, bytes]]] = None, raw_response: bool = False, **kwargs) -> Union[str, Response]:
        """Return a mock answer, optionally with structured output."""
        if images:
            result = "This is a mock response for an image-based query."
        elif format_instructions is None:
            result = random.choice(self._mock_responses["ask"])
        else:
            # Otherwise return a string with a mock structured response
            if "json" in format_instructions.lower():
                result = '{"sample": "This is a mock JSON response"}'
            else:
                result = random.choice(self._mock_responses["ask"]) + "\n\n(Note: This is a mock response)"
        
        mock_response = {"choices": [{"message": {"content": result}}]}
        return Response(result, mock_response) if raw_response else result
    
    def embedding(self, text: str, *, raw_response: bool = False, **kwargs) -> Union[List[float], Response]:
        """Return a mock embedding vector."""
        # Return a random vector of same size as OpenAI's embeddings
        result = [random.uniform(-1, 1) for _ in range(1536)]
        mock_response = {"data": [{"embedding": result}]}
        return Response(result, mock_response) if raw_response else result