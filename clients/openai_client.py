from openai import AsyncOpenAI
from typing import List
import logging
import os
from dotenv import load_dotenv

# from config.settings import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL
from .token_tracker import TokenTracker

load_dotenv()

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")

class OpenAIClient:
    """Client for OpenAI API interactions."""
    
    def __init__(self, token_tracker: TokenTracker):
        if not OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY not found. OpenAIClient will not be functional.")
            self.client = None
        else:
            self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.token_tracker = token_tracker
    
    async def generate_embedding(self, text: str, model: str = OPENAI_EMBEDDING_MODEL) -> List[float]:
        """Generate embedding for text."""
        try:
            # Track token usage
            self.token_tracker.track_openai_embedding(text, model)
            
            response = await self.client.embeddings.create(
                model=model,
                dimensions=1024,
                input=text,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    async def chat_completion(self, prompt: str, model: str, temperature: float = 0.5, max_tokens: int = None) -> str:
        """Generate chat completion."""
        try:
            kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature
            }
            
            if max_tokens:
                kwargs["max_completion_tokens"] = max_tokens
            
            response = await self.client.chat.completions.create(**kwargs)
            
            # Check if response is valid
            if response and response.choices and len(response.choices) > 0:
                output_text = response.choices[0].message.content
                if output_text is not None:
                    output_text = output_text.strip()
                    self.token_tracker.track_openai_chat(prompt, output_text, model)
                    return output_text
            
            logger.error("Received an invalid response from OpenAI API.")
            return ""
            
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            raise

    async def reasoning_completion(self, prompt: str, model: str) -> str:
        """Generate reasoning completion."""
        try:
            kwargs = {
                "model": model,
                "reasoning": {"effort": "medium"},
                "input": [{"role": "user", "content": prompt}]
            }

            response = await self.client.responses.create(**kwargs)
            return response.output_text
        
        except Exception as e:
            logger.error(f"Error in reasoning completion: {e}")
            raise
            