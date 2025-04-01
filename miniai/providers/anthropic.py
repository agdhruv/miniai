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
    
    def _ask(self, question: Optional[str] = None, *, system_prompt: Optional[str] = None, messages: Optional[List[Dict]] = None, format_instructions: str = None, images: List[Union[str, bytes]] = None, raw_response: bool = False, **kwargs) -> Union[str, Response]:
        """Answer a question using Anthropic, optionally with conversation history or images.
        
        Args:
            question: A string with the question to answer (mutually exclusive with messages)
            system_prompt: Optional system prompt to guide the AI
            messages: A list of message dicts for conversation history (mutually exclusive with question)
            format_instructions: Optional instructions for how to format the response
            images: Optional list of image data (file paths, URLs, or bytes)
            raw_response: Whether to return the full response object
            **kwargs: Additional arguments to pass to the provider
        """
        client = self._get_client()
        model = kwargs.pop("model", self.config.get_model("anthropic"))
        
        # Handle system prompt if provided
        # Anthropic uses the "system" key instead of "system_prompt"
        if system_prompt is not None:
            kwargs["system"] = system_prompt
        
        if messages is not None:
            # Use the provided messages list directly
            messages_list = messages
        else:
            # Create a new messages list from the question string
            messages_list = []
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
                messages_list.append({"role": "user", "content": content})
            else:
                messages_list.append({"role": "user", "content": question})
        
        # Add format instructions if provided
        if format_instructions:
            messages_list.append({"role": "user", "content": format_instructions})
        
        response = client.messages.create(
            model=model,
            max_tokens=kwargs.pop("max_tokens", 1000),
            messages=messages_list,
            **kwargs
        )
        
        result = response.content[0].text
        return Response(result, response) if raw_response else result