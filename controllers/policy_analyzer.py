import os
import sys
import logging
import asyncio
import argparse
from typing import List, Optional, Dict, Any
import tiktoken

# Add clients directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'clients'))

from clients.mongo_client import MongoDBClient
from clients.gemini_client import GeminiClient, ThinkingResponse
from clients.token_tracker import TokenTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentChunker:
    """Handles document chunking with tiktoken for accurate token counting."""
    
    def __init__(self, max_tokens: int = 12000, overlap_tokens: int = 256, model: str = "gpt-4o"):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.model = model
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except Exception:
            # Fallback to cl100k_base encoding
            self.encoding = tiktoken.get_encoding("cl100k_base")
            logger.warning(f"Using fallback encoding for model {model}")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            # Fallback estimation: ~4 chars per token
            return len(text) // 4
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text into segments with specified max tokens and overlap.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []
        
        # Split text into sentences for better chunking boundaries
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
                # Add current chunk
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap from previous chunk
                overlap_text = self._get_overlap_text(current_chunk, self.overlap_tokens)
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
        # Split on sentence endings, but preserve the punctuation
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

class PolicyScenarioAnalyzer:
    """Main class for policy scenario analysis."""
    
    def __init__(self, collection_name: str = "policies"):
        self.collection_name = collection_name
        self.mongo_client = None
        self.gemini_client = None
        self.token_tracker = TokenTracker()
        self.chunker = DocumentChunker(max_tokens=12000, overlap_tokens=256)
        self.scenario_prompt = self._load_scenario_prompt()
        
    def _load_scenario_prompt(self) -> str:
        """Load the scenario analysis prompt from file."""
        try:
            with open('controllers\scenario_check.txt', 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.error("scenario_check.txt not found")
            raise
        except Exception as e:
            logger.error(f"Error reading scenario_check.txt: {e}")
            raise
    
    async def initialize(self):
        """Initialize database and AI clients."""
        try:
            # Initialize MongoDB client
            self.mongo_client = MongoDBClient()
            logger.info("MongoDB client initialized successfully")
            
            # Initialize Gemini client
            self.gemini_client = GeminiClient(self.token_tracker)
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
                    {"pdf_link": pdf_url}
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
    
    def prepare_document_chunks(self, document: Dict[str, Any]) -> List[str]:
        """
        Extract and chunk the document content.
        
        Args:
            document: Document from MongoDB
            
        Returns:
            List of document chunks
        """
        try:
            parsed_content = document.get("parsed_content", "")
            if not parsed_content:
                logger.error("No parsed_content found in document")
                return []
            
            # Chunk the document
            chunks = self.chunker.chunk_text(parsed_content)
            
            if not chunks:
                logger.error("Failed to create chunks from document content")
                return []
            
            # Log chunking statistics
            total_tokens = sum(self.chunker.count_tokens(chunk) for chunk in chunks)
            avg_tokens = total_tokens // len(chunks) if chunks else 0
            
            logger.info(f"Document chunked successfully:")
            logger.info(f"  - Total chunks: {len(chunks)}")
            logger.info(f"  - Total tokens: {total_tokens:,}")
            logger.info(f"  - Average tokens per chunk: {avg_tokens:,}")
            logger.info(f"  - Max tokens per chunk: {self.chunker.max_tokens:,}")
            logger.info(f"  - Overlap tokens: {self.chunker.overlap_tokens}")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error preparing document chunks: {e}")
            return []
    
    def create_analysis_prompt(self, document_chunks: List[str]) -> str:
        """
        Create the complete analysis prompt with document content and scenario instructions.
        
        Args:
            document_chunks: List of document chunks
            
        Returns:
            Complete prompt for analysis
        """
        # Combine all chunks into a single document
        full_document = "\n\n--- CHUNK SEPARATOR ---\n\n".join(document_chunks)
        
        prompt = f"""{self.scenario_prompt}

INSURANCE POLICY DOCUMENT TO ANALYZE:
=====================================

{full_document}

=====================================

Please analyze this insurance policy document thoroughly against all the scenarios mentioned in the instructions above. Provide a comprehensive, structured analysis following the exact format specified. Quote specific policy text wherever possible to support your findings.

Focus on providing actionable insights that would help a potential policyholder understand the coverage, limitations, and important considerations for each scenario type."""

        return prompt
    
    async def analyze_policy(self, pdf_url: str) -> Optional[str]:
        """
        Analyze a policy document against predefined scenarios.
        
        Args:
            pdf_url: URL of the PDF to analyze
            
        Returns:
            Analysis result or None if failed
        """
        try:
            # Search for document
            logger.info(f"Searching for document with URL: {pdf_url}")
            document = self.search_document_by_url(pdf_url)
            
            if not document:
                logger.error("Document not found in database")
                return None
            
            # Extract document metadata
            doc_title = document.get("title", "Unknown Policy")
            doc_id = str(document.get("_id", ""))
            logger.info(f"Found document: {doc_title} (ID: {doc_id})")
            
            # Prepare document chunks
            logger.info("Processing document content...")
            document_chunks = self.prepare_document_chunks(document)
            
            if not document_chunks:
                logger.error("Failed to process document content")
                return None
            
            # Create analysis prompt
            analysis_prompt = self.create_analysis_prompt(document_chunks)
            prompt_tokens = self.chunker.count_tokens(analysis_prompt)
            logger.info(f"Analysis prompt created with {prompt_tokens:,} tokens")
            
            # Generate analysis using Gemini with thinking
            logger.info("Generating policy analysis with Gemini thinking...")
            response: ThinkingResponse = await self.gemini_client.generate_with_thinking(
                prompt=analysis_prompt,
                model="gemini-2.5-flash",
                thinking_budget=-1,  # Dynamic thinking
                include_thoughts=True,
                temperature=0.3
            )
            
            if response.text:
                logger.info(f"Analysis generated successfully:")
                logger.info(f"  - Output tokens: {response.output_tokens}")
                logger.info(f"  - Thought tokens: {response.thought_tokens}")
                
                # Log thinking summary if available
                if response.thought_summary:
                    logger.info(f"  - Thinking summary length: {len(response.thought_summary)} chars")
                
                return response.text
            else:
                logger.error("Empty response from Gemini")
                return None
                
        except Exception as e:
            logger.error(f"Error in policy analysis: {e}")
            return None
    
    def get_token_usage_summary(self) -> Dict[str, Any]:
        """Get token usage and cost summary."""
        return self.token_tracker.get_detailed_breakdown()
    
    def close(self):
        """Clean up resources."""
        if self.mongo_client:
            self.mongo_client.close()
        logger.info("PolicyScenarioAnalyzer closed successfully")

async def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Insurance Policy Scenario Analyzer")
    parser.add_argument("pdf_url", help="PDF URL to search for in database")
    parser.add_argument("--collection", default="policies", 
                       help="MongoDB collection name (default: policies)")
    parser.add_argument("--output", "-o", 
                       help="Output file to save analysis results")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize the analyzer
    analyzer = PolicyScenarioAnalyzer(collection_name=args.collection)
    
    try:
        # Initialize clients
        await analyzer.initialize()
        
        # Analyze the policy
        print(f"Analyzing policy document: {args.pdf_url}")
        print("This may take a few minutes...")
        
        analysis_result = await analyzer.analyze_policy(args.pdf_url)
        
        if analysis_result:
            print("\n" + "="*80)
            print("POLICY SCENARIO ANALYSIS COMPLETE")
            print("="*80)
            
            if args.output:
                # Save to file
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(f"Policy Analysis for: {args.pdf_url}\n")
                    f.write("="*80 + "\n\n")
                    f.write(analysis_result)
                print(f"Analysis saved to: {args.output}")
            else:
                # Print to console
                print(analysis_result)
            
            print("\n" + "="*80)
        else:
            print("Failed to generate analysis. Check logs for details.")
            return 1
    
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"Error: {e}")
        return 1
    
    finally:
        # Clean up and show usage summary
        usage = analyzer.get_token_usage_summary()
        print(f"\nToken usage: {usage['totals']['total_tokens']:,} tokens, ${usage['total_cost']:.4f}")
        analyzer.close()
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)