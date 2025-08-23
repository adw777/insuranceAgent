import os
import sys
import logging
import asyncio
from typing import List, Optional, Dict, Any, Set
from urllib.parse import urlparse
import tiktoken

# Add clients directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'clients'))

from clients.mongo_client import MongoDBClient
from clients.gemini_client import GeminiClient, ChatSession
from clients.token_tracker import TokenTracker
from clients.qdrant_client import get_qdrant_client, query_points_with_filter
from clients.openai_client import OpenAIClient

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

class PolicySelectorChat:
    """Main class for policy selection chat functionality."""
    
    def __init__(self, 
                 collection_name: str = "policy_chunks2",
                 mongo_collection_name: str = "policies"):
        self.collection_name = collection_name
        self.mongo_collection_name = mongo_collection_name
        self.qdrant_client = None
        self.mongo_client = None
        self.gemini_client = None
        self.openai_client = None
        self.token_tracker = TokenTracker()
        self.chunker = DocumentChunker()
        self.chat_session = None
        self.policy_documents = []
        self.policy_urls = []
        
    async def initialize(self):
        """Initialize all clients."""
        try:
            # Initialize Qdrant client
            self.qdrant_client = get_qdrant_client()
            logger.info("Qdrant client initialized successfully")
            
            # Initialize MongoDB client
            self.mongo_client = MongoDBClient()
            logger.info("MongoDB client initialized successfully")
            
            # Initialize OpenAI client
            self.openai_client = OpenAIClient(self.token_tracker)
            logger.info("OpenAI client initialized successfully")
            
            # Initialize Gemini client
            self.gemini_client = GeminiClient(self.token_tracker)
            self.chat_session = self.gemini_client.create_chat_session()
            logger.info("Gemini client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize clients: {e}")
            raise
    
    async def search_relevant_policies(self, user_query: str, filter_urls: List[str], top_k: int = 10) -> Set[str]:
        """
        Search for relevant policies using vector similarity and URL filters.
        
        Args:
            user_query: User's query to convert to embedding
            filter_urls: List of PDF URLs to filter by
            top_k: Number of top results to return
            
        Returns:
            Set of unique PDF URLs found
        """
        try:
            # Generate embedding for the user query
            logger.info("Generating embedding for user query...")
            query_vector = await self.openai_client.generate_embedding(user_query)
            
            # Search in Qdrant with filters
            logger.info(f"Searching in Qdrant with {len(filter_urls)} URL filters...")
            results = query_points_with_filter(
                client=self.qdrant_client,
                collection_name=self.collection_name,
                query_vector=query_vector,
                filters={"pdf_link": filter_urls},
                limit=top_k,
                with_payload=True
            )
            
            # Extract unique PDF URLs from results
            found_urls = set()
            for result in results:
                if hasattr(result, 'payload') and result.payload:
                    pdf_link = result.payload.get('pdf_link')
                    if pdf_link:
                        found_urls.add(pdf_link)
            
            logger.info(f"Found {len(found_urls)} unique policy URLs from {len(results)} chunks")
            return found_urls
            
        except Exception as e:
            logger.error(f"Error searching relevant policies: {e}")
            return set()
    
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
            document = self.mongo_client.find_one_document(self.mongo_collection_name, query)
            
            if document:
                logger.info(f"Found document for URL: {pdf_url}")
                return document
            else:
                logger.warning(f"No document found for URL: {pdf_url}")
                return None
                
        except Exception as e:
            logger.error(f"Error searching for document: {e}")
            return None
    
    def load_policy_documents(self, pdf_urls: Set[str]) -> bool:
        """
        Load policy documents from MongoDB by URLs.
        
        Args:
            pdf_urls: Set of PDF URLs to load
            
        Returns:
            True if at least one document was loaded successfully
        """
        try:
            self.policy_documents = []
            self.policy_urls = []
            
            for pdf_url in pdf_urls:
                document = self.search_document_by_url(pdf_url)
                if document:
                    parsed_content = document.get("parsed_content", "")
                    if parsed_content:
                        self.policy_documents.append({
                            "url": pdf_url,
                            "title": document.get("title", "Unknown Policy"),
                            "content": parsed_content,
                            "document_id": str(document.get("_id", ""))
                        })
                        self.policy_urls.append(pdf_url)
                        logger.info(f"Loaded policy: {document.get('title', 'Unknown')}")
                    else:
                        logger.warning(f"No parsed content found for URL: {pdf_url}")
                else:
                    logger.warning(f"Document not found for URL: {pdf_url}")
            
            if self.policy_documents:
                logger.info(f"Successfully loaded {len(self.policy_documents)} policy documents")
                return True
            else:
                logger.error("No policy documents were loaded")
                return False
                
        except Exception as e:
            logger.error(f"Error loading policy documents: {e}")
            return False
    
    def _create_initial_prompt(self, user_query: str) -> str:
        """Create the initial context prompt for policy analysis."""
        # Combine all policy contents
        policy_sections = []
        for i, policy in enumerate(self.policy_documents, 1):
            policy_sections.append(f"""
POLICY {i}: {policy['title']}
URL: {policy['url']}
CONTENT:
{policy['content']}
""")
        
        combined_policies = "\n" + "="*80 + "\n".join(policy_sections)
        
        prompt = f"""You are an expert insurance policy analyst and advisor. 
Your role is to help the user analyze, compare, and select the BEST policy out of the available ones, 
while keeping their needs, preferences, and biases at the center of your recommendation. 

You have been provided with {len(self.policy_documents)} insurance policy documents. 
Your task is to:

1. Analyze each selected policy thoroughly (coverage, benefits, exclusions, costs, fine print).
2. Compare the policies directly against each other.
3. Focus on what matters most for the USER's specific query, preferences, and priorities.
4. Explain complex terms in simple, user-friendly language.
5. Highlight key differences that actually impact the user's decision.
6. Call out hidden limitations or conditions that might influence the choice.
7. Recommend the single BEST policy (or top 2 if truly close), with clear reasoning tailored to the user's needs.

AVAILABLE POLICIES:
{combined_policies}

USER QUERY: {user_query}

Please provide a structured and comprehensive analysis that includes:
- A short summary of each policy's key features
- A direct comparison of relevant aspects (coverage, benefits, exclusions, costs)
- Your expert recommendation (the best option for the user) with clear reasoning
- Any critical considerations, caveats, or fine-print the user should be aware of

Be decisive, practical, and helpful. The goal is to guide the user to confidently pick the best policy for THEM, not just list information."""


        return prompt
    
    async def start_policy_analysis(self, user_query: str, filter_urls: List[str], top_k: int = 10) -> str:
        """
        Start the policy analysis process.
        
        Args:
            user_query: User's initial query
            filter_urls: List of PDF URLs to consider
            top_k: Number of top chunks to retrieve
            
        Returns:
            Initial analysis response
        """
        try:
            # Search for relevant policies
            found_urls = await self.search_relevant_policies(user_query, filter_urls, top_k)
            
            if not found_urls:
                return "I couldn't find any relevant policies based on your query and the provided URLs. Please check the URLs or try a different query."
            
            # Load policy documents from MongoDB
            if not self.load_policy_documents(found_urls):
                return "I found relevant policies but couldn't load their content from the database. Please check if the documents are properly stored."
            
            # Create initial prompt and get response from Gemini
            initial_prompt = self._create_initial_prompt(user_query)
            
            # response = await self.gemini_client.chat_with_thinking(
            #     session=self.chat_session,
            #     user_message=initial_prompt
            # )
            
            response = await self.gemini_client.generate_content(
                prompt=initial_prompt,
                model="gemini-2.5-flash",
                temperature=0.7,
                # max_output_tokens=10000
            )

            # if response.text:
            #     logger.info(f"Generated initial analysis (tokens: {response.output_tokens})")
            #     return response.text
            if response:
                logger.info(f"Generated initial analysis")
                return response
            else:
                logger.error("Empty response from Gemini")
                return "I apologize, but I couldn't generate an analysis. Please try rephrasing your question."
                
        except Exception as e:
            logger.error(f"Error in policy analysis: {e}")
            return f"An error occurred while analyzing the policies: {str(e)}"
    
    async def continue_chat(self, user_message: str) -> str:
        """
        Continue the chat conversation.
        
        Args:
            user_message: User's follow-up message
            
        Returns:
            AI response string
        """
        try:
            if not self.policy_documents:
                return "No policies are currently loaded. Please start a new analysis session."
            
            if not self.gemini_client or not self.chat_session:
                return "AI client not initialized. Please check your setup."
            
            # Get response from Gemini
            # response = await self.gemini_client.chat_with_thinking(
            #     session=self.chat_session,
            #     user_message=user_message
            # )
            
            response = await self.gemini_client.generate_content(
                prompt=user_message,
                model="gemini-2.5-flash",
                temperature=0.7,
                # max_output_tokens=10000
            )

            # if response.text:
            #     logger.info(f"Generated initial analysis (tokens: {response.output_tokens})")
            #     return response.text

            if response:
                # logger.info(f"Generated initial analysis (tokens: {response.output_tokens})")
                return response
            else:
                logger.error("Empty response from Gemini")
                return "I apologize, but I couldn't generate a response. Please try rephrasing your question."
                
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return f"An error occurred while processing your message: {str(e)}"
    
    def get_loaded_policies_info(self) -> Dict[str, Any]:
        """Get information about loaded policies."""
        if not self.policy_documents:
            return {"status": "No policies loaded"}
        
        policies_info = []
        total_tokens = 0
        
        for policy in self.policy_documents:
            content_tokens = self.chunker.count_tokens(policy['content'])
            total_tokens += content_tokens
            policies_info.append({
                "title": policy['title'],
                "url": policy['url'],
                "content_tokens": content_tokens
            })
        
        return {
            "total_policies": len(self.policy_documents),
            "total_tokens": total_tokens,
            "policies": policies_info,
            "chat_session_messages": len(self.chat_session.messages) if self.chat_session else 0
        }
    
    def get_token_usage_summary(self) -> Dict[str, Any]:
        """Get token usage and cost summary."""
        return self.token_tracker.get_detailed_breakdown()
    
    def close(self):
        """Clean up resources."""
        if self.mongo_client:
            self.mongo_client.close()
        logger.info("PolicySelectorChat closed successfully")

async def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Insurance Policy Selection Chat Assistant")
    parser.add_argument("query", help="Your question about insurance policies")
    parser.add_argument("urls", nargs="+", help="List of PDF URLs to consider")
    parser.add_argument("--collection", default="policy_chunks2", 
                       help="Qdrant collection name (default: policy_chunks2)")
    parser.add_argument("--mongo-collection", default="policies",
                       help="MongoDB collection name (default: policies)")
    parser.add_argument("--top-k", type=int, default=10,
                       help="Number of top chunks to retrieve (default: 10)")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Start interactive chat mode after initial analysis")
    
    args = parser.parse_args()
    
    # Initialize the chat system
    chat_system = PolicySelectorChat(
        collection_name=args.collection,
        mongo_collection_name=args.mongo_collection
    )
    
    try:
        # Initialize clients
        print("Initializing AI clients...")
        await chat_system.initialize()
        
        # Start policy analysis
        print(f"Analyzing policies for query: {args.query}")
        print(f"Considering {len(args.urls)} URLs...")
        
        initial_response = await chat_system.start_policy_analysis(
            user_query=args.query,
            filter_urls=args.urls,
            top_k=args.top_k
        )
        
        # Display initial analysis
        print("\n" + "="*80)
        print("INSURANCE POLICY ANALYSIS")
        print("="*80)
        print(initial_response)
        print("="*80)
        
        # Display loaded policies info
        policies_info = chat_system.get_loaded_policies_info()
        if policies_info.get("total_policies", 0) > 0:
            print(f"\nLoaded {policies_info['total_policies']} policies:")
            for policy in policies_info['policies']:
                print(f"  - {policy['title']} ({policy['content_tokens']:,} tokens)")
        
        if args.interactive:
            # Interactive chat mode
            print("\n" + "="*60)
            print("INTERACTIVE POLICY CHAT")
            print("="*60)
            print("Ask follow-up questions about the policies. Type 'quit' to exit.")
            print("Type 'info' for policy information or 'usage' for token usage.")
            print("="*60)
            
            while True:
                try:
                    user_input = input("\nYour question: ").strip()
                    
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        break
                    elif user_input.lower() == 'info':
                        info = chat_system.get_loaded_policies_info()
                        print(f"\nPolicy Info:")
                        for key, value in info.items():
                            if key != 'policies':
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
                    response = await chat_system.continue_chat(user_input)
                    print(f"\nAssistant: {response}")
                    
                except KeyboardInterrupt:
                    print("\nExiting...")
                    break
                except Exception as e:
                    print(f"Error: {e}")
    
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

# Example usage:
# python chat_with_multiple_policies.py "What is the best health insurance for a family of 4?" "https://cms.zurichkotak.com/uploads/Health_Maximiser_Prospectus_fed9c8c2e1.pdf" "https://cms.zurichkotak.com/uploads/Benefit_Illustration_Health_Premier_Advantage_Plan_003_8b878d9ed5.pdf" --interactive