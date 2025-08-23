import os
import sys
import logging
import asyncio
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse
import tiktoken

# Add clients directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'clients'))

from clients.mongo_client import MongoDBClient
from clients.gemini_client import GeminiClient, ChatSession
from clients.token_tracker import TokenTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentChunker:
    """Handles document chunking with token counting."""
    
    def __init__(self, max_tokens: int = 12000, model: str = "gpt-4o"):
        self.max_tokens = max_tokens
        self.model = model
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except Exception:
            # Fallback to cl100k_base encoding
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        try:
            return len(self.encoding.encode(text))
        except Exception:
            # Fallback estimation: ~4 chars per token
            return len(text) // 4
    
    def chunk_text(self, text: str, overlap_tokens: int = 200) -> List[str]:
        """
        Chunk text into segments with specified max tokens.
        
        Args:
            text: Text to chunk
            overlap_tokens: Number of tokens to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []
        
        # Split text into sentences for better chunking
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If single sentence exceeds max tokens, split it further
            if sentence_tokens > self.max_tokens:
                # Add current chunk if it has content
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_tokens = 0
                
                # Split long sentence into smaller parts
                sub_chunks = self._split_long_sentence(sentence)
                chunks.extend(sub_chunks)
                continue
            
            # Check if adding this sentence would exceed token limit
            if current_tokens + sentence_tokens > self.max_tokens and current_chunk.strip():
                # Add current chunk and start new one
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap from previous chunk
                overlap_text = self._get_overlap_text(current_chunk, overlap_tokens)
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                current_tokens = self.count_tokens(current_chunk)
            else:
                # Add sentence to current chunk
                current_chunk += " " + sentence if current_chunk else sentence
                current_tokens += sentence_tokens
        
        # Add final chunk if it has content
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        logger.info(f"Split document into {len(chunks)} chunks")
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        # Simple sentence splitting - can be improved with more sophisticated methods
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_long_sentence(self, sentence: str) -> List[str]:
        """Split a sentence that's too long into smaller parts."""
        words = sentence.split()
        chunks = []
        current_chunk = ""
        
        for word in words:
            test_chunk = current_chunk + " " + word if current_chunk else word
            if self.count_tokens(test_chunk) > self.max_tokens:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = word
                else:
                    # Single word is too long - just add it
                    chunks.append(word)
                    current_chunk = ""
            else:
                current_chunk = test_chunk
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """Get the last part of text for overlap."""
        if overlap_tokens <= 0:
            return ""
        
        words = text.split()
        overlap_text = ""
        
        # Work backwards to get approximately the right number of tokens
        for i in range(len(words) - 1, -1, -1):
            test_text = " ".join(words[i:])
            if self.count_tokens(test_text) > overlap_tokens:
                break
            overlap_text = test_text
        
        return overlap_text

class PolicyDocumentChat:
    """Main class for policy document chat functionality."""
    
    def __init__(self, collection_name: str = "policies"):
        self.collection_name = collection_name
        self.mongo_client = None
        self.gemini_client = None
        self.token_tracker = TokenTracker()
        self.chunker = DocumentChunker()
        self.chat_session = None
        self.document_chunks = []
        self.document_metadata = {}
        
    async def initialize(self):
        """Initialize database and AI clients."""
        try:
            # Initialize MongoDB client
            self.mongo_client = MongoDBClient()
            logger.info("MongoDB client initialized successfully")
            
            # Initialize Gemini client
            self.gemini_client = GeminiClient(self.token_tracker)
            self.chat_session = self.gemini_client.create_chat_session()
            logger.info("Gemini client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize clients: {e}")
            raise
    
    def search_document_by_url(self, pdf_url: str) -> Optional[Dict[str, Any]]:
        """
        Search for a document in MongoDB by PDF URL.
        
        Args:
            pdf_url: The PDF URL to search for
            
        Returns:
            Document dict if found, None otherwise
        """
        try:
            if not self.mongo_client:
                raise RuntimeError("MongoDB client not initialized")
            
            # Search for document with matching URL in multiple possible fields
            query = {
                "$or": [
                    {"url": pdf_url},
                    {"pdf_url": pdf_url},
                    {"pdf_link": pdf_url},
                    {"link": pdf_url}
                ]
            }
            document = self.mongo_client.find_one_document(self.collection_name, query)
            
            if document:
                logger.info(f"Found document for URL: {pdf_url}")
                return document
            else:
                logger.warning(f"No document found for URL: {pdf_url}")
                return None
                
        except Exception as e:
            logger.error(f"Error searching for document: {e}")
            raise
    
    def prepare_document_chunks(self, document: Dict[str, Any]) -> bool:
        """
        Prepare document chunks for chat.
        
        Args:
            document: Document from MongoDB
            
        Returns:
            True if successful, False otherwise
        """
        try:
            parsed_content = document.get("parsed_content", "")
            if not parsed_content:
                logger.error("No parsed_content found in document")
                return False
            
            # Store document metadata
            self.document_metadata = {
                "url": document.get("url", ""),
                "title": document.get("title", "Unknown Policy"),
                "document_id": str(document.get("_id", "")),
                "total_length": len(parsed_content)
            }
            
            # Chunk the document
            self.document_chunks = self.chunker.chunk_text(parsed_content)
            
            if not self.document_chunks:
                logger.error("Failed to create chunks from document content")
                return False
            
            # Log chunking statistics
            total_tokens = sum(self.chunker.count_tokens(chunk) for chunk in self.document_chunks)
            logger.info(f"Document chunked successfully:")
            logger.info(f"  - Total chunks: {len(self.document_chunks)}")
            logger.info(f"  - Total tokens: {total_tokens:,}")
            logger.info(f"  - Average tokens per chunk: {total_tokens // len(self.document_chunks):,}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error preparing document chunks: {e}")
            return False
    
    def _create_context_prompt(self, user_query: str, relevant_chunks: List[str]) -> str:
        """Create a context-aware prompt for the AI."""
        context = "\n\n".join(f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(relevant_chunks))
        
        prompt = f"""You are an expert insurance policy assistant. You have access to the following sections from an insurance policy document:

POLICY DOCUMENT SECTIONS:
{context}

Your role is to help users understand and evaluate this insurance policy. Please:

1. Answer questions accurately based ONLY on the provided policy content
2. Help users understand coverage, terms, benefits, and limitations
3. Explain complex insurance terminology in simple terms
4. Point out important details they should consider
5. If information is not available in the provided sections, clearly state that
6. Be helpful in guiding policy selection decisions
7. Provide Proper Citations from the given policy wherever required

USER QUESTION: {user_query}

Please provide a comprehensive, helpful response based on the policy information provided."""

        return prompt
    
    def _find_relevant_chunks(self, user_query: str, max_chunks: int = 3) -> List[str]:
        """
        Find the most relevant chunks for the user query.
        This is a simple implementation - could be enhanced with embeddings/semantic search.
        """
        query_lower = user_query.lower()
        query_words = set(query_lower.split())
        
        chunk_scores = []
        for i, chunk in enumerate(self.document_chunks):
            chunk_lower = chunk.lower()
            chunk_words = set(chunk_lower.split())
            
            # Simple relevance scoring based on word overlap
            common_words = query_words.intersection(chunk_words)
            score = len(common_words) / len(query_words) if query_words else 0
            
            # Boost score for exact phrase matches
            if query_lower in chunk_lower:
                score += 0.5
            
            chunk_scores.append((score, i, chunk))
        
        # Sort by relevance score and return top chunks
        chunk_scores.sort(key=lambda x: x[0], reverse=True)
        relevant_chunks = [chunk for _, _, chunk in chunk_scores[:max_chunks]]
        
        return relevant_chunks if relevant_chunks else self.document_chunks[:max_chunks]
    
    async def chat(self, user_query: str) -> str:
        """
        Process a user query and return AI response.
        
        Args:
            user_query: User's question about the policy
            
        Returns:
            AI response string
        """
        try:
            if not self.document_chunks:
                return "No policy document loaded. Please load a document first."
            
            if not self.gemini_client or not self.chat_session:
                return "AI client not initialized. Please check your setup."
            
            # Find relevant chunks for the query
            relevant_chunks = self._find_relevant_chunks(user_query)
            
            # Create context-aware prompt
            prompt = self._create_context_prompt(user_query, relevant_chunks)
            
            # Get response from Gemini
            response = await self.gemini_client.chat_with_thinking(
                session=self.chat_session,
                user_message=prompt
            )
            
            if response.text:
                logger.info(f"Generated response (tokens: {response.output_tokens})")
                return response.text
            else:
                logger.error("Empty response from Gemini")
                return "I apologize, but I couldn't generate a response. Please try rephrasing your question."
                
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return f"An error occurred while processing your question: {str(e)}"
    
    def get_document_info(self) -> Dict[str, Any]:
        """Get information about the loaded document."""
        if not self.document_metadata:
            return {"status": "No document loaded"}
        
        total_tokens = sum(self.chunker.count_tokens(chunk) for chunk in self.document_chunks)
        
        return {
            "title": self.document_metadata.get("title", "Unknown"),
            "url": self.document_metadata.get("url") or self.document_metadata.get("pdf_url") or self.document_metadata.get("pdf_link", ""),
            "total_chunks": len(self.document_chunks),
            "total_tokens": total_tokens,
            "average_tokens_per_chunk": total_tokens // len(self.document_chunks) if self.document_chunks else 0,
            "chat_session_messages": len(self.chat_session.messages) if self.chat_session else 0
        }
    
    def get_token_usage_summary(self) -> Dict[str, Any]:
        """Get token usage and cost summary."""
        return self.token_tracker.get_detailed_breakdown()
    
    def close(self):
        """Clean up resources."""
        if self.mongo_client:
            self.mongo_client.close()
        logger.info("PolicyDocumentChat closed successfully")

async def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Insurance Policy Chat Assistant")
    parser.add_argument("pdf_url", help="PDF URL to search for in database")
    parser.add_argument("--collection", default="policies", 
                       help="MongoDB collection name (default: policy_documents)")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Start interactive chat mode")
    
    args = parser.parse_args()
    
    # Initialize the chat system
    chat_system = PolicyDocumentChat(collection_name=args.collection)
    
    try:
        # Initialize clients
        await chat_system.initialize()
        
        # Search for document
        print(f"Searching for document with URL: {args.pdf_url}")
        document = chat_system.search_document_by_url(args.pdf_url)
        
        if not document:
            print("Document not found in database.")
            return
        
        # Prepare document chunks
        print("Processing document content...")
        if not chat_system.prepare_document_chunks(document):
            print("Failed to process document content.")
            return
        
        # Display document info
        doc_info = chat_system.get_document_info()
        print(f"\nDocument loaded successfully:")
        print(f"  Title: {doc_info['title']}")
        print(f"  Chunks: {doc_info['total_chunks']}")
        print(f"  Total tokens: {doc_info['total_tokens']:,}")
        
        if args.interactive:
            # Interactive chat mode
            print("\n" + "="*60)
            print("INSURANCE POLICY CHAT ASSISTANT")
            print("="*60)
            print("Ask questions about the policy document. Type 'quit' to exit.")
            print("Type 'info' for document information or 'usage' for token usage.")
            print("="*60)
            
            while True:
                try:
                    user_input = input("\nYour question: ").strip()
                    
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        break
                    elif user_input.lower() == 'info':
                        info = chat_system.get_document_info()
                        print(f"\nDocument Info:")
                        for key, value in info.items():
                            print(f"  {key}: {value}")
                        continue
                    elif user_input.lower() == 'usage':
                        usage = chat_system.get_token_usage_summary()
                        print(f"\nToken Usage Summary:")
                        print(f"  Total cost: ${usage['total_cost']:.4f}")
                        print(f"  Total tokens: {usage['totals']['total_tokens']:,}")
                        continue
                    elif not user_input:
                        continue
                    
                    print("\nProcessing your question...")
                    response = await chat_system.chat(user_input)
                    print(f"\nAssistant: {response}")
                    
                except KeyboardInterrupt:
                    print("\nExiting...")
                    break
                except Exception as e:
                    print(f"Error: {e}")
        else:
            # Single query mode - example
            example_query = "What are the main benefits of this insurance policy?"
            print(f"\nExample query: {example_query}")
            response = await chat_system.chat(example_query)
            print(f"\nResponse: {response}")
    
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"Error: {e}")
    
    finally:
        # Clean up
        chat_system.close()
        
        # Print final usage summary
        usage = chat_system.get_token_usage_summary()
        print(f"\nFinal token usage: {usage['totals']['total_tokens']:,} tokens, ${usage['total_cost']:.4f}")

if __name__ == "__main__":
    asyncio.run(main())


# python select_policy_chat.py "https://cms.zurichkotak.com/uploads/Health_Maximiser_Prospectus_fed9c8c2e1.pdf" --interactive