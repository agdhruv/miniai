from typing import List, Dict, Optional, Union, Type
import json
from pathlib import Path
import openai
from miniai.providers.constants import PROVIDERS
from miniai.providers.base import BaseProvider, Response

class OpenAIProvider(BaseProvider):
    """Provider for OpenAI API."""
    
    def __init__(self, config: Type):
        super().__init__(config)
    
    def _get_client(self):
        """Get or create the OpenAI client."""
        if self._client is None:
            self._client = openai.OpenAI(api_key=self.config.get_api_key("openai"))
        return self._client
    
    def embedding(self, text: str, *, raw_response: bool = False, **kwargs) -> Union[List[float], Response]:
        """Get embedding vector using OpenAI's embedding model."""
        client = self._get_client()
        model = kwargs.pop("model", PROVIDERS["openai"]["default_embedding_model"])
        
        response = client.embeddings.create(
            model=model,
            input=text,
            **kwargs
        )
        
        result = response.data[0].embedding
        return Response(result, response) if raw_response else result
    
    def ask(self, question: str, *, format_instructions: Optional[str] = None, images: Optional[List[Union[str, bytes]]] = None, raw_response: bool = False, **kwargs) -> Union[str, Response]:
        """Answer a question using OpenAI, optionally with images.
        
        Args:
            question: The question or instruction to answer
            format_instructions: Optional instructions for how to format the response
            images: Optional list of image data (file paths, URLs, or bytes)
            raw_response: Whether to return the full response object
            **kwargs: Additional arguments to pass to the provider
        """
        client = self._get_client()
        model = kwargs.pop("model", self.config.get_model("openai"))
        
        # Handle system prompt if provided
        messages = []
        if "system" in kwargs:
            system_prompt = kwargs.pop("system")
            messages.append({"role": "system", "content": system_prompt})
            
        # Handle images if provided
        if images:
            content = [{"type": "text", "text": question}]
            for img in images:
                if isinstance(img, str):
                    if img.startswith(('http://', 'https://')):
                        # Handle URL
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": img}
                        })
                    else:
                        # Handle file path
                        import base64
                        with open(img, "rb") as f:
                            img_data = base64.b64encode(f.read()).decode("utf-8")
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_data}"
                            }
                        })
                else:
                    # Handle bytes
                    import base64
                    img_data = base64.b64encode(img).decode("utf-8")
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_data}"
                        }
                    })
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": question})
        
        # Add format instructions if provided
        if format_instructions:
            messages.append({"role": "user", "content": format_instructions})
            if "json" in format_instructions.lower():
                kwargs["response_format"] = {"type": "json_object"}
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        
        result = response.choices[0].message.content
        return Response(result, response) if raw_response else result
    
    def extract(self, text: str, entities: List[str], *, raw_response: bool = False, **kwargs) -> Union[Dict[str, List[str]], Response]:
        """Extract entities from text."""
        prompt = f"Extract the following entities from the text: {', '.join(entities)}.\n\nText: {text}\n\nProvide results in JSON format with entity types as keys and lists of extracted items as values."
        
        # Force JSON response format
        kwargs["response_format"] = {"type": "json_object"}
        
        response = self.ask(prompt, raw_response=True, **kwargs)
        
        # Parse JSON response
        try:
            result = json.loads(response.content)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON response from model: {response.content}")
            
        return Response(result, response.raw_response) if raw_response else result

    def text_to_speech(self, text: str, *, raw_response: bool = False, **kwargs) -> Union[bytes, Response]:
        """Convert text to speech using OpenAI's TTS API.
        
        Args:
            text: The text to convert to speech
            raw_response: Whether to return the full response object
            **kwargs: Additional arguments to pass to the provider, including:
                - voice: The voice to use (default: "alloy")
                - model: TTS model to use (default: "tts-1")
                - instructions: Additional instructions for the TTS model (e.g. "Speak in a British accent")
                - response_format: Audio format (mp3, opus, aac, flac) (default: "mp3")
                - speed: Speech speed, between 0.25 and 4.0 (default: 1.0)
            
        Returns:
            Audio data as bytes or Response object
        """
        client = self._get_client()
        model = kwargs.pop("model", PROVIDERS["openai"]["default_tts_model"])
        voice = kwargs.pop("voice", PROVIDERS["openai"]["default_tts_voice"])
        
        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            **kwargs
        )
        
        result = response.content
        return Response(result, response) if raw_response else result

    def speech_to_text(self, audio_data: Union[bytes, str], *, raw_response: bool = False, **kwargs) -> Union[Dict, Response]:
        """Convert speech to text using OpenAI's Whisper API.
        
        Args:
            audio_data: The audio data to convert to text (bytes) or a path to an audio file (str)
            raw_response: Whether to return the full response object
            **kwargs: Additional arguments to pass to the provider, including:
                - model: Transcription model to use (default: "gpt-4o-transcribe")
                - prompt: Optional text to guide the model's style (e.g. "This is a conversation between a doctor and a patient")
                - Other arguments: https://platform.openai.com/docs/api-reference/audio/createTranscription
            
        Returns:
            Dict with transcription and other metadata or Response object
        """
        client = self._get_client()
        model = kwargs.pop("model", PROVIDERS["openai"]["default_stt_model"])
        
        if 'stream' in kwargs:
            raise ValueError("Streaming not supported yet")

        if isinstance(audio_data, str):
            filename = Path(audio_data).name
            with open(audio_data, 'rb') as f:
                audio_data = f.read()
            # OpenAI prefers a tuple of (filename, file_data) to help with the transcription
            # We actually have that available if the user passes a filepath and not just bytes
            file_argument = (filename, audio_data)
        else:
            # If the user passes bytes, we can just pass them as is since we don't have a filename
            file_argument = audio_data            
        
        response = client.audio.transcriptions.create(
            model=model,
            file=file_argument,
            **kwargs
        )
        
        result = response.dict()
        return Response(result, response) if raw_response else result