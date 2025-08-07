# from google import genai
# from google.genai import types
# import logging
# import json
# from dotenv import load_dotenv
# import os

# from .token_tracker import TokenTracker

# logger = logging.getLogger(__name__)

# load_dotenv()

# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# # GEMINI_MODEL = os.getenv("GEMINI_MODEL")

# class GeminiClient:
#     """Client for Gemini API interactions."""
    
#     def __init__(self, token_tracker: TokenTracker):
#         if not GEMINI_API_KEY:
#             logger.warning("GEMINI_API_KEY not found. GeminiClient will not be functional.")
#             self.client = None
#         else:
#             self.client = genai.Client(api_key=GEMINI_API_KEY)
#         self.token_tracker = token_tracker
    
#     async def generate_content(self, prompt: str, model: str, 
#                         temperature: float = 0.7, max_output_tokens: int = None) -> str:
#         """Generate content using Gemini."""
#         if not self.client:
#             logger.error("Gemini client is not initialized due to missing API key")
#             return ""
            
#         try:
#             config_kwargs = {"temperature": temperature}
#             if max_output_tokens:
#                 config_kwargs["max_output_tokens"] = max_output_tokens
            
#             response = await self.client.aio.models.generate_content(
#                 model=model,
#                 contents=[prompt],
#                 config=types.GenerateContentConfig(**config_kwargs)
#             )
#              # Check if response and response.text are valid
#             if response is None:
#                 logger.error("Gemini API returned None response")
#                 return ""
            
#             if not hasattr(response, 'text') or response.text is None:
#                 logger.error("Gemini response has no text or text is None")
#                 return ""
            
#             output_text = response.text.strip()
            
#             # Only track if we have valid output
#             if output_text:
#                 self.token_tracker.track_gemini_generation(prompt, output_text, model)
            
#             return output_text
            
#         except Exception as e:
#             logger.error(f"Error in Gemini content generation: {e}")
#             raise


from google import genai
from google.genai import types
import logging
import json
from dotenv import load_dotenv
import os
from typing import List, Dict, Any, Optional, Generator, Tuple
from dataclasses import dataclass, field

from .token_tracker import TokenTracker

logger = logging.getLogger(__name__)

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

@dataclass
class ChatMessage:
    """Represents a chat message with role and content."""
    role: str  # 'user' or 'model'
    content: str
    thought_summary: Optional[str] = None
    thought_tokens: int = 0
    output_tokens: int = 0

@dataclass
class ChatSession:
    """Represents a chat session with context management."""
    messages: List[ChatMessage] = field(default_factory=list)
    thought_signatures: List[Any] = field(default_factory=list)
    total_thought_tokens: int = 0
    total_output_tokens: int = 0

@dataclass
class ThinkingResponse:
    """Response from Gemini with thinking capabilities."""
    text: str
    thought_summary: Optional[str] = None
    thought_tokens: int = 0
    output_tokens: int = 0
    thought_signatures: List[Any] = field(default_factory=list)

class GeminiClient:
    """Client for Gemini API interactions with thinking capabilities."""
    
    def __init__(self, token_tracker: TokenTracker):
        if not GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY not found. GeminiClient will not be functional.")
            self.client = None
        else:
            self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.token_tracker = token_tracker
    
    async def generate_content(self, prompt: str, model: str, 
                        temperature: float = 0.7, max_output_tokens: int = None) -> str:
        """Generate content using Gemini (legacy method for backward compatibility)."""
        if not self.client:
            logger.error("Gemini client is not initialized due to missing API key")
            return ""
            
        try:
            config_kwargs = {"temperature": temperature}
            if max_output_tokens:
                config_kwargs["max_output_tokens"] = max_output_tokens
            
            response = await self.client.aio.models.generate_content(
                model=model,
                contents=[prompt],
                config=types.GenerateContentConfig(**config_kwargs)
            )
             # Check if response and response.text are valid
            if response is None:
                logger.error("Gemini API returned None response")
                return ""
            
            if not hasattr(response, 'text') or response.text is None:
                logger.error("Gemini response has no text or text is None")
                return ""
            
            output_text = response.text.strip()
            
            # Only track if we have valid output
            if output_text:
                self.token_tracker.track_gemini_generation(prompt, output_text, model)
            
            return output_text
            
        except Exception as e:
            logger.error(f"Error in Gemini content generation: {e}")
            raise
    
    def create_chat_session(self) -> ChatSession:
        """Create a new chat session."""
        return ChatSession()
    
    async def generate_with_thinking(
        self, 
        prompt: str, 
        model: str = "gemini-2.5-flash",
        thinking_budget: int = -1,  # -1 for dynamic thinking
        include_thoughts: bool = True,
        temperature: float = 0.7,
        max_output_tokens: int = None
    ) -> ThinkingResponse:
        """Generate content with thinking capabilities."""
        if not self.client:
            logger.error("Gemini client is not initialized due to missing API key")
            return ThinkingResponse(text="", thought_summary="Error: Client not initialized")
        
        try:
            config_kwargs = {
                "temperature": temperature,
                "thinking_config": types.ThinkingConfig(
                    thinking_budget=thinking_budget,
                    include_thoughts=include_thoughts
                )
            }
            
            if max_output_tokens:
                config_kwargs["max_output_tokens"] = max_output_tokens
            
            response = await self.client.aio.models.generate_content(
                model=model,
                contents=[prompt],
                config=types.GenerateContentConfig(**config_kwargs)
            )
            
            if response is None:
                logger.error("Gemini API returned None response")
                return ThinkingResponse(text="", thought_summary="Error: No response")
            
            # Extract text, thought summary, and token counts
            text_content = ""
            thought_summary = ""
            thought_signatures = []
            
            # Validate response structure
            try:
                if not hasattr(response, 'candidates') or not response.candidates:
                    logger.error("Gemini response has no candidates")
                    return ThinkingResponse(text="", thought_summary="Error: No candidates in response")
                
                candidate = response.candidates[0]
                if not hasattr(candidate, 'content') or not candidate.content:
                    logger.error("Gemini response candidate has no content")
                    return ThinkingResponse(text="", thought_summary="Error: No content in response")
                
                if not hasattr(candidate.content, 'parts') or not candidate.content.parts:
                    logger.error("Gemini response content has no parts")
                    return ThinkingResponse(text="", thought_summary="Error: No parts in response")
                
                for part in candidate.content.parts:
                    if not hasattr(part, 'text') or not part.text:
                        continue
                    if hasattr(part, 'thought') and part.thought:
                        thought_summary += part.text
                    else:
                        text_content += part.text
                    
                    # Collect thought signatures if present
                    if hasattr(part, 'thought_signature') and part.thought_signature:
                        thought_signatures.append(part)
                        
            except (AttributeError, IndexError) as e:
                logger.error(f"Error accessing response structure: {e}")
                return ThinkingResponse(text="", thought_summary="Error: Invalid response structure")
            
            # Get token counts
            thought_tokens = getattr(response.usage_metadata, 'thoughts_token_count', 0)
            output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
            
            # Track tokens
            if text_content:
                self.token_tracker.track_gemini_generation(prompt, text_content, model)
            
            logger.info(f"Generated response - Thought tokens: {thought_tokens}, Output tokens: {output_tokens}")
            
            return ThinkingResponse(
                text=text_content.strip(),
                thought_summary=thought_summary.strip() if thought_summary else None,
                thought_tokens=thought_tokens,
                output_tokens=output_tokens,
                thought_signatures=thought_signatures
            )
            
        except Exception as e:
            logger.error(f"Error in Gemini thinking generation: {e}")
            raise
    
    async def chat_with_thinking(
        self,
        session: ChatSession,
        user_message: str,
        model: str = "gemini-2.5-flash",
        thinking_budget: int = -1,
        include_thoughts: bool = True,
        temperature: float = 0.7,
        max_output_tokens: int = None
    ) -> ThinkingResponse:
        """Continue a chat conversation with thinking capabilities."""
        if not self.client:
            logger.error("Gemini client is not initialized due to missing API key")
            return ThinkingResponse(text="", thought_summary="Error: Client not initialized")
        
        try:
            # Build conversation history
            contents = []
            
            # Add previous messages
            for msg in session.messages:
                contents.append(types.Content(
                    role=msg.role,
                    parts=[types.Part(text=msg.content)]
                ))
            
            # Add thought signatures from previous turns if available
            if session.thought_signatures:
                # Include the entire response with signatures from previous turns
                for signature_part in session.thought_signatures:
                    contents.append(signature_part)
            
            # Add current user message
            contents.append(types.Content(
                role="user",
                parts=[types.Part(text=user_message)]
            ))
            
            config_kwargs = {
                "temperature": temperature,
                "thinking_config": types.ThinkingConfig(
                    thinking_budget=thinking_budget,
                    include_thoughts=include_thoughts
                )
            }
            
            if max_output_tokens:
                config_kwargs["max_output_tokens"] = max_output_tokens
            
            response = await self.client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs)
            )
            
            if response is None:
                logger.error("Gemini API returned None response")
                return ThinkingResponse(text="", thought_summary="Error: No response")
            
            # Extract text, thought summary, and signatures
            text_content = ""
            thought_summary = ""
            new_signatures = []
            
            for part in response.candidates[0].content.parts:
                if not part.text:
                    continue
                if part.thought:
                    thought_summary += part.text
                else:
                    text_content += part.text
                
                # Collect new thought signatures
                if hasattr(part, 'thought_signature') and part.thought_signature:
                    new_signatures.append(part)
            
            # Get token counts
            thought_tokens = getattr(response.usage_metadata, 'thoughts_token_count', 0)
            output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
            
            # Update session
            session.messages.append(ChatMessage(
                role="user",
                content=user_message,
                thought_tokens=0,
                output_tokens=0
            ))
            
            session.messages.append(ChatMessage(
                role="model",
                content=text_content.strip(),
                thought_summary=thought_summary.strip() if thought_summary else None,
                thought_tokens=thought_tokens,
                output_tokens=output_tokens
            ))
            
            # Update thought signatures for next turn
            if new_signatures:
                session.thought_signatures = new_signatures
            
            # Update totals
            session.total_thought_tokens += thought_tokens
            session.total_output_tokens += output_tokens
            
            # Track tokens
            if text_content:
                self.token_tracker.track_gemini_generation(user_message, text_content, model)
            
            logger.info(f"Chat response - Thought tokens: {thought_tokens}, Output tokens: {output_tokens}")
            
            return ThinkingResponse(
                text=text_content.strip(),
                thought_summary=thought_summary.strip() if thought_summary else None,
                thought_tokens=thought_tokens,
                output_tokens=output_tokens,
                thought_signatures=new_signatures
            )
            
        except Exception as e:
            logger.error(f"Error in Gemini chat with thinking: {e}")
            raise
    
    async def stream_with_thinking(
        self,
        prompt: str,
        model: str = "gemini-2.5-flash",
        thinking_budget: int = -1,
        include_thoughts: bool = True,
        temperature: float = 0.7,
        max_output_tokens: int = None
    ) -> Generator[Tuple[str, bool], None, None]:
        """Stream content generation with thinking capabilities.
        
        Yields tuples of (text_chunk, is_thought) where is_thought indicates
        if the chunk is from thought summary or actual response.
        """
        if not self.client:
            logger.error("Gemini client is not initialized due to missing API key")
            yield ("Error: Client not initialized", False)
            return
        
        try:
            config_kwargs = {
                "temperature": temperature,
                "thinking_config": types.ThinkingConfig(
                    thinking_budget=thinking_budget,
                    include_thoughts=include_thoughts
                )
            }
            
            if max_output_tokens:
                config_kwargs["max_output_tokens"] = max_output_tokens
            
            stream = await self.client.aio.models.generate_content_stream(
                model=model,
                contents=[prompt],
                config=types.GenerateContentConfig(**config_kwargs)
            )
            
            thoughts_started = False
            answer_started = False
            
            async for chunk in stream:
                try:
                    # Validate chunk structure
                    if not hasattr(chunk, 'candidates') or not chunk.candidates:
                        continue
                    
                    candidate = chunk.candidates[0]
                    if not hasattr(candidate, 'content') or not candidate.content:
                        continue
                    
                    if not hasattr(candidate.content, 'parts') or not candidate.content.parts:
                        continue
                    
                    for part in candidate.content.parts:
                        if not hasattr(part, 'text') or not part.text:
                            continue
                        
                        if hasattr(part, 'thought') and part.thought:
                            if not thoughts_started:
                                yield ("=== THINKING ===\n", True)
                                thoughts_started = True
                            yield (part.text, True)
                            full_thought += part.text
                        else:
                            if not answer_started:
                                if thoughts_started:
                                    yield ("\n=== RESPONSE ===\n", False)
                                answer_started = True
                            yield (part.text, False)
                            full_answer += part.text
                        
                        # Collect signatures
                        if hasattr(part, 'thought_signature') and part.thought_signature:
                            new_signatures.append(part)
                            
                except (AttributeError, IndexError) as e:
                    logger.error(f"Error processing chunk: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Error in Gemini streaming with thinking: {e}")
            yield (f"Error: {str(e)}", False)
    
    async def stream_chat_with_thinking(
        self,
        session: ChatSession,
        user_message: str,
        model: str = "gemini-2.5-flash",
        thinking_budget: int = -1,
        include_thoughts: bool = True,
        temperature: float = 0.7,
        max_output_tokens: int = None
    ) -> Generator[Tuple[str, bool], None, None]:
        """Stream chat conversation with thinking capabilities."""
        if not self.client:
            logger.error("Gemini client is not initialized due to missing API key")
            yield ("Error: Client not initialized", False)
            return
        
        try:
            # Build conversation history
            contents = []
            
            # Add previous messages
            for msg in session.messages:
                contents.append(types.Content(
                    role=msg.role,
                    parts=[types.Part(text=msg.content)]
                ))
            
            # Add thought signatures from previous turns
            if session.thought_signatures:
                for signature_part in session.thought_signatures:
                    contents.append(signature_part)
            
            # Add current user message
            contents.append(types.Content(
                role="user",
                parts=[types.Part(text=user_message)]
            ))
            
            config_kwargs = {
                "temperature": temperature,
                "thinking_config": types.ThinkingConfig(
                    thinking_budget=thinking_budget,
                    include_thoughts=include_thoughts
                )
            }
            
            if max_output_tokens:
                config_kwargs["max_output_tokens"] = max_output_tokens
            
            stream = await self.client.aio.models.generate_content_stream(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs)
            )
            
            thoughts_started = False
            answer_started = False
            full_thought = ""
            full_answer = ""
            new_signatures = []
            
            async for chunk in stream:
                try:
                    # Validate chunk structure
                    if not hasattr(chunk, 'candidates') or not chunk.candidates:
                        continue
                    
                    candidate = chunk.candidates[0]
                    if not hasattr(candidate, 'content') or not candidate.content:
                        continue
                    
                    if not hasattr(candidate.content, 'parts') or not candidate.content.parts:
                        continue
                    
                    for part in candidate.content.parts:
                        if not part.text:
                            continue
                        
                        if part.thought:
                            if not thoughts_started:
                                yield ("=== THINKING ===\n", True)
                                thoughts_started = True
                            yield (part.text, True)
                            full_thought += part.text
                        else:
                            if not answer_started:
                                if thoughts_started:
                                    yield ("\n=== RESPONSE ===\n", False)
                                answer_started = True
                            yield (part.text, False)
                            full_answer += part.text
                        
                        # Collect signatures
                        if hasattr(part, 'thought_signature') and part.thought_signature:
                            new_signatures.append(part)
                            
                except (AttributeError, IndexError) as e:
                    logger.error(f"Error processing chunk: {e}")
                    continue
            
            # Update session after streaming completes
            session.messages.append(ChatMessage(
                role="user",
                content=user_message,
                thought_tokens=0,
                output_tokens=0
            ))
            
            session.messages.append(ChatMessage(
                role="model",
                content=full_answer.strip(),
                thought_summary=full_thought.strip() if full_thought else None,
                thought_tokens=0,  # Token counts not available in streaming
                output_tokens=0
            ))
            
            # Update signatures
            if new_signatures:
                session.thought_signatures = new_signatures
            
        except Exception as e:
            logger.error(f"Error in Gemini streaming chat with thinking: {e}")
            yield (f"Error: {str(e)}", False)
    
    def get_session_stats(self, session: ChatSession) -> Dict[str, Any]:
        """Get statistics for a chat session."""
        return {
            "total_messages": len(session.messages),
            "user_messages": len([msg for msg in session.messages if msg.role == "user"]),
            "model_messages": len([msg for msg in session.messages if msg.role == "model"]),
            "total_thought_tokens": session.total_thought_tokens,
            "total_output_tokens": session.total_output_tokens,
            "has_thought_signatures": len(session.thought_signatures) > 0
        }