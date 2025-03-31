from typing import List, Dict, Optional, Union, Type
import anthropic
from miniai.providers.base import BaseProvider, Response

class AnthropicProvider(BaseProvider):
    """Provider for Anthropic API."""
    
    def __init__(self, config: Type):
        super().__init__(config)
    
    def _get_client(self):
        """Get or create the Anthropic client."""
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.config.get_api_key("anthropic"))
        return self._client
    
    def embedding(self, text: str, *, raw_response: bool = False, **kwargs) -> Union[List[float], Response]:
        """Get embedding vector using Anthropic's embedding model."""
        raise NotImplementedError(
            "Anthropic does not currently provide an embedding API. "
            "Please use a different provider (e.g., openai) for embeddings."
        )
    
    def ask(self, question: str, *, format_instructions: Optional[str] = None, images: Optional[List[Union[str, bytes]]] = None, raw_response: bool = False, **kwargs) -> Union[str, Response]:
        """Answer a question using Anthropic, optionally with images.
        
        Args:
            question: The question or instruction to answer
            format_instructions: Optional instructions for how to format the response
            images: Optional list of image data (file paths, URLs, or bytes)
            raw_response: Whether to return the full response object
            **kwargs: Additional arguments to pass to the provider
        """
        client = self._get_client()
        model = kwargs.pop("model", self.config.get_model("anthropic"))
        
        messages = []
        # Handle images if provided
        if images:
            content = []
            for img in images:
                if isinstance(img, str):
                    if img.startswith(('http://', 'https://')):
                        # Handle URL
                        content.append({
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": img
                            }
                        })
                    else:
                        # Handle file path
                        import base64
                        import mimetypes
                        with open(img, "rb") as f:
                            img_data = base64.b64encode(f.read()).decode("utf-8")
                        media_type = mimetypes.guess_type(img)[0] or "image/jpeg"
                        content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": img_data
                            }
                        })
                else:
                    # Handle bytes
                    import base64
                    img_data = base64.b64encode(img).decode("utf-8")
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_data
                        }
                    })
            # Add the text/question to the content
            content.append({"type": "text", "text": question})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": question})
        
        # Add format instructions if provided
        if format_instructions:
            messages.append({"role": "user", "content": format_instructions})
        
        response = client.messages.create(
            model=model,
            max_tokens=kwargs.pop("max_tokens", 1000),
            messages=messages,
            **kwargs
        )
        
        result = response.content[0].text
        return Response(result, response) if raw_response else result