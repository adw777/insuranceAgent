# # import os
# # import logging
# # import asyncio
# # import numpy as np
# # import json
# # import requests
# # import tempfile
# # import aiohttp
# # import aiofiles
# # from pathlib import Path
# # from typing import List, Dict, Any, Optional, Tuple, Set
# # from dataclasses import dataclass
# # import hashlib
# # import uuid
# # from concurrent.futures import ThreadPoolExecutor, as_completed
# # import threading
# # from queue import Queue
# # import time
# # from urllib.parse import urlparse
# # import re
# # from collections import defaultdict

# # from qdrant_client import QdrantClient
# # from qdrant_client.http import models
# # from qdrant_client.models import Distance, VectorParams, PointStruct
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from openai import AsyncOpenAI
# # from dotenv import load_dotenv
# # import pymongo
# # from pymongo import UpdateOne

# # from clients.parsing_client import PDFParsingClient
# # from clients.mongo_client import MongoDBClient
# # from clients.qdrant_client import get_qdrant_client, create_collection, insert_points_to_collection
# # from clients.token_tracker import TokenTracker

# # # Load environment variables
# # load_dotenv()

# # # Configure logging
# # logging.basicConfig(
# #     level=logging.INFO,
# #     format='%(asctime)s - %(levelname)s - %(message)s',
# #     handlers=[
# #         logging.FileHandler('rag_ingestion_mongo.log'),
# #         logging.StreamHandler()
# #     ]
# # )
# # logger = logging.getLogger(__name__)

# # @dataclass
# # class DocumentChunk:
# #     """Represents a document chunk with metadata and keywords."""
# #     text: str
# #     source_file: str
# #     chunk_index: int
# #     start_char: int
# #     end_char: int
# #     chunk_id: str
# #     keywords: List[str]
# #     embedding: Optional[List[float]] = None
# #     company_name: str = ""
# #     type: str = ""
# #     sub_type: str = ""
# #     title: str = ""
# #     pdf_link: str = ""

# # @dataclass
# # class ProcessingResult:
# #     """Result of processing a single document."""
# #     document_id: str
# #     success: bool
# #     chunks_count: int
# #     keywords_count: int
# #     error_message: str = ""
# #     processing_time: float = 0.0

# # class KeywordExtractor:
# #     """Advanced keyword extraction using multiple strategies."""
    
# #     def __init__(self, openai_client: AsyncOpenAI):
# #         self.openai_client = openai_client
        
# #         # Insurance domain specific terms
# #         self.insurance_terms = {
# #             'coverage', 'premium', 'deductible', 'policy', 'claim', 'benefit', 'exclusion',
# #             'copay', 'coinsurance', 'underwriting', 'actuarial', 'rider', 'annuity',
# #             'endowment', 'term insurance', 'whole life', 'disability', 'critical illness',
# #             'hospitalization', 'outpatient', 'inpatient', 'maternity', 'dental', 'vision',
# #             'pharmaceutical', 'pre-existing', 'waiting period', 'sum insured', 'family floater',
# #             'individual', 'group', 'corporate', 'health insurance', 'life insurance',
# #             'motor insurance', 'travel insurance', 'home insurance', 'fire insurance'
# #         }
        
# #         # Common stop words to exclude
# #         self.stop_words = {
# #             'the', 'is', 'at', 'which', 'on', 'and', 'a', 'to', 'are', 'as', 'was',
# #             'for', 'an', 'be', 'by', 'i', 'you', 'it', 'of', 'or', 'will', 'my',
# #             'one', 'have', 'from', 'or', 'had', 'but', 'not', 'what', 'all', 'were',
# #             'they', 'we', 'when', 'your', 'can', 'said', 'there', 'each', 'do',
# #             'their', 'time', 'if', 'up', 'out', 'many', 'then', 'them', 'these',
# #             'so', 'some', 'her', 'would', 'make', 'like', 'into', 'him', 'has',
# #             'two', 'more', 'very', 'after', 'words', 'first', 'where', 'much',
# #             'through', 'back', 'years', 'work', 'came', 'right', 'still', 'such',
# #             'because', 'turn', 'here', 'why', 'asked', 'went', 'men', 'read',
# #             'need', 'land', 'different', 'home', 'us', 'move', 'try', 'kind',
# #             'hand', 'picture', 'again', 'change', 'off', 'play', 'spell', 'air',
# #             'away', 'animal', 'house', 'point', 'page', 'letter', 'mother', 'answer',
# #             'found', 'study', 'still', 'learn', 'should', 'america', 'world'
# #         }
    
# #     def extract_statistical_keywords(self, text: str, max_keywords: int = 15) -> List[str]:
# #         """Extract keywords using statistical methods."""
# #         # Clean and tokenize text
# #         words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
# #         # Filter out stop words and get word frequencies
# #         word_freq = defaultdict(int)
# #         for word in words:
# #             if word not in self.stop_words and len(word) > 2:
# #                 word_freq[word] += 1
        
# #         # Prioritize insurance domain terms
# #         scored_words = []
# #         for word, freq in word_freq.items():
# #             score = freq
# #             if word in self.insurance_terms or any(term in word for term in self.insurance_terms):
# #                 score *= 3  # Boost insurance-related terms
# #             scored_words.append((word, score))
        
# #         # Sort by score and return top keywords
# #         scored_words.sort(key=lambda x: x[1], reverse=True)
# #         return [word for word, _ in scored_words[:max_keywords]]
    
# #     async def extract_llm_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
# #         """Extract keywords using LLM for semantic understanding."""
# #         try:
# #             prompt = f"""Extract the most important {max_keywords} keywords from this insurance policy text. 
# # Focus on:
# # - Insurance terms and concepts
# # - Coverage types and benefits
# # - Policy conditions and exclusions
# # - Financial terms
# # - Medical conditions or treatments mentioned

# # Text: {text[:2000]}...

# # Return only the keywords as a comma-separated list, no explanations."""

# #             response = await self.openai_client.chat.completions.create(
# #                 model="gpt-4o-mini",
# #                 messages=[{"role": "user", "content": prompt}],
# #                 max_tokens=150,
# #                 temperature=0.3
# #             )
            
# #             keywords_text = response.choices[0].message.content.strip()
# #             keywords = [kw.strip().lower() for kw in keywords_text.split(',') if kw.strip()]
# #             return keywords[:max_keywords]
            
# #         except Exception as e:
# #             logger.warning(f"LLM keyword extraction failed: {e}")
# #             return []
    
# #     async def extract_keywords(self, text: str, max_keywords: int = 20) -> List[str]:
# #         """Extract keywords using hybrid approach."""
# #         # Get statistical keywords
# #         stat_keywords = self.extract_statistical_keywords(text, max_keywords // 2)
        
# #         # Get LLM keywords for semantic understanding
# #         llm_keywords = await self.extract_llm_keywords(text, max_keywords // 2)
        
# #         # Combine and deduplicate
# #         all_keywords = list(set(stat_keywords + llm_keywords))
        
# #         # Ensure we don't exceed max_keywords
# #         return all_keywords[:max_keywords]

# # class PDFDownloader:
# #     """Efficient PDF downloader with retry logic and caching."""
    
# #     def __init__(self, temp_dir: str = None):
# #         self.temp_dir = temp_dir or tempfile.mkdtemp()
# #         Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
# #         self.session = None
    
# #     async def __aenter__(self):
# #         self.session = aiohttp.ClientSession(
# #             timeout=aiohttp.ClientTimeout(total=300),  # 5 minutes timeout
# #             connector=aiohttp.TCPConnector(limit=10)
# #         )
# #         return self
    
# #     async def __aexit__(self, exc_type, exc_val, exc_tb):
# #         if self.session:
# #             await self.session.close()
    
# #     async def download_pdf(self, url: str, document_id: str, max_retries: int = 3) -> Optional[str]:
# #         """Download PDF with retry logic."""
# #         for attempt in range(max_retries):
# #             try:
# #                 # Generate filename from URL and document ID
# #                 parsed_url = urlparse(url)
# #                 filename = f"{document_id}_{hashlib.md5(url.encode()).hexdigest()[:8]}.pdf"
# #                 file_path = os.path.join(self.temp_dir, filename)
                
# #                 # Check if file already exists
# #                 if os.path.exists(file_path):
# #                     logger.info(f"PDF already exists: {filename}")
# #                     return file_path
                
# #                 logger.info(f"Downloading PDF from {url} (attempt {attempt + 1})")
                
# #                 async with self.session.get(url) as response:
# #                     if response.status == 200:
# #                         content = await response.read()
                        
# #                         # Validate it's a PDF
# #                         if content.startswith(b'%PDF'):
# #                             async with aiofiles.open(file_path, 'wb') as f:
# #                                 await f.write(content)
                            
# #                             logger.info(f"Successfully downloaded: {filename}")
# #                             return file_path
# #                         else:
# #                             logger.warning(f"Downloaded content is not a valid PDF: {url}")
# #                             return None
# #                     else:
# #                         logger.warning(f"HTTP {response.status} for {url}")
                        
# #             except Exception as e:
# #                 logger.warning(f"Download attempt {attempt + 1} failed for {url}: {e}")
# #                 if attempt < max_retries - 1:
# #                     await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
# #         logger.error(f"Failed to download PDF after {max_retries} attempts: {url}")
# #         return None
    
# #     def cleanup(self):
# #         """Clean up temporary files."""
# #         import shutil
# #         try:
# #             shutil.rmtree(self.temp_dir)
# #             logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
# #         except Exception as e:
# #             logger.warning(f"Failed to cleanup temp directory: {e}")

# # class MongoRAGIngestionPipeline:
# #     """High-performance RAG ingestion pipeline for MongoDB insurance policies."""
    
# #     def __init__(self, max_workers: int = 5):
# #         # Configuration
# #         self.collection_name = "policies"
# #         self.chunk_collection = "policy_chunks2"
# #         self.keyword_collection = "policy_keywords2"
# #         self.max_workers = max_workers
        
# #         # Text splitter configuration
# #         self.chunk_size = 1024
# #         self.chunk_overlap = 200
        
# #         # OpenAI configuration
# #         self.embedding_model = "text-embedding-3-large"
# #         self.embedding_dimensions = 1024
        
# #         # Initialize clients
# #         self.token_tracker = TokenTracker()
# #         self.parsing_client = PDFParsingClient()
# #         self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# #         self.mongo_client = MongoDBClient()
# #         self.qdrant_client = get_qdrant_client()
# #         self.keyword_extractor = KeywordExtractor(self.openai_client)
        
# #         # Initialize text splitter
# #         self.text_splitter = RecursiveCharacterTextSplitter(
# #             chunk_size=self.chunk_size,
# #             chunk_overlap=self.chunk_overlap,
# #             length_function=len,
# #             separators=["\n\n", "\n", ".", "!", "?", " ", ""]
# #         )
        
# #         # Processing statistics
# #         self.stats = {
# #             'total_documents': 0,
# #             'processed_documents': 0,
# #             'failed_documents': 0,
# #             'total_chunks': 0,
# #             'total_keywords': 0,
# #             'processing_time': 0.0
# #         }
        
# #         # Thread-safe collections for batch updates
# #         self.mongo_updates_queue = Queue()
# #         self.qdrant_chunks_queue = Queue()
# #         self.qdrant_keywords_queue = Queue()
        
# #     def setup_qdrant_collections(self):
# #         """Setup Qdrant collections for chunks and keywords."""
# #         logger.info("Setting up Qdrant collections...")
        
# #         try:
# #             # Create chunk collection
# #             if not create_collection(self.qdrant_client, self.chunk_collection, self.embedding_dimensions):
# #                 logger.warning(f"Failed to create chunk collection: {self.chunk_collection}")
            
# #             # Create keyword collection
# #             if not create_collection(self.qdrant_client, self.keyword_collection, self.embedding_dimensions):
# #                 logger.warning(f"Failed to create keyword collection: {self.keyword_collection}")
            
# #             logger.info("Qdrant collections setup completed")
            
# #         except Exception as e:
# #             logger.error(f"Error setting up Qdrant collections: {e}")
# #             raise
    
# #     def get_documents_to_process(self, limit: int = None) -> List[Dict[str, Any]]:
# #         """Get documents from MongoDB that need processing."""
# #         query = {
# #             "$or": [
# #                 {"parsed_content": {"$exists": False}},
# #                 {"chunks": {"$exists": False}},
# #                 {"keywords": {"$exists": False}}
# #             ]
# #         }
        
# #         try:
# #             documents = self.mongo_client.find_documents(
# #                 collection_name=self.collection_name,
# #                 query=query,
# #                 limit=limit or 0
# #             )
            
# #             logger.info(f"Found {len(documents)} documents to process")
# #             return documents
            
# #         except Exception as e:
# #             logger.error(f"Error fetching documents from MongoDB: {e}")
# #             return []
    
# #     async def process_single_document(self, document: Dict[str, Any], downloader: PDFDownloader) -> ProcessingResult:
# #         """Process a single document: download, parse, chunk, extract keywords."""
# #         start_time = time.time()
# #         doc_id = str(document['_id'])
        
# #         try:
# #             # Get PDF URL (check multiple possible field names)
# #             pdf_url = document.get('pdf_link') or document.get('url') or document.get('pdf_url')
            
# #             if not pdf_url:
# #                 return ProcessingResult(
# #                     document_id=doc_id,
# #                     success=False,
# #                     chunks_count=0,
# #                     keywords_count=0,
# #                     error_message="No PDF URL found in document"
# #                 )
            
# #             # Download PDF
# #             pdf_path = await downloader.download_pdf(pdf_url, doc_id)
# #             if not pdf_path:
# #                 return ProcessingResult(
# #                     document_id=doc_id,
# #                     success=False,
# #                     chunks_count=0,
# #                     keywords_count=0,
# #                     error_message="Failed to download PDF"
# #                 )
            
# #             # Parse PDF to markdown
# #             try:
# #                 parsed_content = self.parsing_client.parse_single_pdf(pdf_path)
# #                 if not parsed_content or len(parsed_content.strip()) < 100:
# #                     return ProcessingResult(
# #                         document_id=doc_id,
# #                         success=False,
# #                         chunks_count=0,
# #                         keywords_count=0,
# #                         error_message="PDF parsing failed or content too short"
# #                     )
# #             except Exception as e:
# #                 return ProcessingResult(
# #                     document_id=doc_id,
# #                     success=False,
# #                     chunks_count=0,
# #                     keywords_count=0,
# #                     error_message=f"PDF parsing failed: {str(e)}"
# #                 )
            
# #             # Create chunks
# #             text_chunks = self.text_splitter.split_text(parsed_content)
            
# #             if not text_chunks:
# #                 return ProcessingResult(
# #                     document_id=doc_id,
# #                     success=False,
# #                     chunks_count=0,
# #                     keywords_count=0,
# #                     error_message="No chunks created from parsed content"
# #                 )
            
# #             # Extract metadata
# #             company_name = document.get('company_name', '')
# #             type_field = document.get('type', '')
# #             sub_type = document.get('sub_type', '')
# #             title = document.get('title', '')
            
# #             # Process each chunk
# #             document_chunks = []
# #             all_keywords_set = set()
            
# #             char_position = 0
# #             for i, chunk_text in enumerate(text_chunks):
# #                 # Find chunk position in original text
# #                 chunk_start = parsed_content.find(chunk_text, char_position)
# #                 if chunk_start == -1:
# #                     chunk_start = char_position
# #                 chunk_end = chunk_start + len(chunk_text)
                
# #                 # Extract keywords for this chunk
# #                 chunk_keywords = await self.keyword_extractor.extract_keywords(chunk_text, max_keywords=15)
# #                 all_keywords_set.update(chunk_keywords)
                
# #                 # Create DocumentChunk
# #                 chunk_id = self._generate_chunk_id(doc_id, i, chunk_text)
# #                 chunk = DocumentChunk(
# #                     text=chunk_text.strip(),
# #                     source_file=title,
# #                     chunk_index=i,
# #                     start_char=chunk_start,
# #                     end_char=chunk_end,
# #                     chunk_id=chunk_id,
# #                     keywords=chunk_keywords,
# #                     company_name=company_name,
# #                     type=type_field,
# #                     sub_type=sub_type,
# #                     title=title,
# #                     pdf_link=pdf_url
# #                 )
                
# #                 document_chunks.append(chunk)
# #                 char_position = chunk_end
            
# #             # Generate embeddings for chunks
# #             chunks_with_embeddings = await self.generate_embeddings(document_chunks)
            
# #             # Prepare data for MongoDB update
# #             chunks_data = []
# #             for chunk in chunks_with_embeddings:
# #                 chunks_data.append({
# #                     'chunk_id': chunk.chunk_id,
# #                     'text': chunk.text,
# #                     'chunk_index': chunk.chunk_index,
# #                     'keywords': chunk.keywords,
# #                     'start_char': chunk.start_char,
# #                     'end_char': chunk.end_char
# #                 })
            
# #             all_keywords = list(all_keywords_set)
            
# #             # Queue MongoDB update
# #             mongo_update = {
# #                 'document_id': doc_id,
# #                 'update_data': {
# #                     'parsed_content': parsed_content,
# #                     'chunks': chunks_data,
# #                     'keywords': all_keywords,
# #                     'processing_timestamp': time.time()
# #                 }
# #             }
# #             self.mongo_updates_queue.put(mongo_update)
            
# #             # Queue Qdrant uploads
# #             for chunk in chunks_with_embeddings:
# #                 if chunk.embedding:
# #                     self.qdrant_chunks_queue.put(chunk)
            
# #             # Create keyword embeddings and queue them
# #             if all_keywords:
# #                 keyword_embeddings = await self.generate_keyword_embeddings(all_keywords, document)
# #                 for keyword_data in keyword_embeddings:
# #                     self.qdrant_keywords_queue.put(keyword_data)
            
# #             processing_time = time.time() - start_time
            
# #             logger.info(f"Successfully processed document {doc_id}: {len(chunks_with_embeddings)} chunks, {len(all_keywords)} keywords")
            
# #             return ProcessingResult(
# #                 document_id=doc_id,
# #                 success=True,
# #                 chunks_count=len(chunks_with_embeddings),
# #                 keywords_count=len(all_keywords),
# #                 processing_time=processing_time
# #             )
            
# #         except Exception as e:
# #             processing_time = time.time() - start_time
# #             logger.error(f"Error processing document {doc_id}: {e}")
# #             return ProcessingResult(
# #                 document_id=doc_id,
# #                 success=False,
# #                 chunks_count=0,
# #                 keywords_count=0,
# #                 error_message=str(e),
# #                 processing_time=processing_time
# #             )
    
# #     def _generate_chunk_id(self, doc_id: str, chunk_index: int, chunk_text: str) -> str:
# #         """Generate a unique ID for a chunk."""
# #         content_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:8]
# #         return f"{doc_id}_{chunk_index:04d}_{content_hash}"
    
# #     async def generate_embeddings(self, chunks: List[DocumentChunk], batch_size: int = 50) -> List[DocumentChunk]:
# #         """Generate embeddings for chunks in batches."""
# #         logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        
# #         processed_chunks = []
        
# #         for i in range(0, len(chunks), batch_size):
# #             batch = chunks[i:i + batch_size]
            
# #             try:
# #                 # Prepare texts for embedding
# #                 texts = [chunk.text for chunk in batch]
                
# #                 # Generate embeddings
# #                 response = await self.openai_client.embeddings.create(
# #                     model=self.embedding_model,
# #                     input=texts,
# #                     dimensions=self.embedding_dimensions,
# #                     encoding_format="float"
# #                 )
                
# #                 # Assign embeddings to chunks
# #                 for chunk, embedding_data in zip(batch, response.data):
# #                     chunk.embedding = embedding_data.embedding
# #                     processed_chunks.append(chunk)
                
# #                 # Track token usage
# #                 for text in texts:
# #                     self.token_tracker.track_openai_embedding(text, self.embedding_model)
                
# #                 # Small delay to respect rate limits
# #                 await asyncio.sleep(0.1)
                
# #             except Exception as e:
# #                 logger.error(f"Error generating embeddings for batch: {e}")
# #                 # Add chunks without embeddings to avoid losing data
# #                 for chunk in batch:
# #                     chunk.embedding = None
# #                     processed_chunks.append(chunk)
        
# #         valid_chunks = [chunk for chunk in processed_chunks if chunk.embedding is not None]
# #         logger.info(f"Generated embeddings for {len(valid_chunks)}/{len(chunks)} chunks")
# #         return valid_chunks
    
# #     async def generate_keyword_embeddings(self, keywords: List[str], document: Dict[str, Any]) -> List[Dict[str, Any]]:
# #         """Generate embeddings for keywords."""
# #         try:
# #             # Create keyword phrases for better semantic representation
# #             keyword_phrases = [f"insurance {keyword}" for keyword in keywords]
            
# #             response = await self.openai_client.embeddings.create(
# #                 model=self.embedding_model,
# #                 input=keyword_phrases,
# #                 dimensions=self.embedding_dimensions,
# #                 encoding_format="float"
# #             )
            
# #             keyword_data = []
# #             for keyword, embedding_data in zip(keywords, response.data):
# #                 keyword_data.append({
# #                     'keyword': keyword,
# #                     'embedding': embedding_data.embedding,
# #                     'company_name': document.get('company_name', ''),
# #                     'type': document.get('type', ''),
# #                     'sub_type': document.get('sub_type', ''),
# #                     'title': document.get('title', ''),
# #                     'pdf_link': document.get('pdf_link') or document.get('url') or document.get('pdf_url', '')
# #                 })
            
# #             return keyword_data
            
# #         except Exception as e:
# #             logger.error(f"Error generating keyword embeddings: {e}")
# #             return []
    
# #     def batch_update_mongodb(self):
# #         """Process MongoDB updates in batches."""
# #         updates = []
        
# #         # Collect updates from queue
# #         while not self.mongo_updates_queue.empty():
# #             update_data = self.mongo_updates_queue.get()
            
# #             updates.append(
# #                 UpdateOne(
# #                     {"_id": update_data['document_id']},
# #                     {"$set": update_data['update_data']}
# #                 )
# #             )
            
# #             # Process in batches of 100
# #             if len(updates) >= 100:
# #                 self._execute_mongo_batch(updates)
# #                 updates = []
        
# #         # Process remaining updates
# #         if updates:
# #             self._execute_mongo_batch(updates)
    
# #     def _execute_mongo_batch(self, updates: List[UpdateOne]):
# #         """Execute a batch of MongoDB updates."""
# #         try:
# #             collection = self.mongo_client.get_collection(self.collection_name)
# #             result = collection.bulk_write(updates, ordered=False)
# #             logger.info(f"MongoDB batch update: {result.modified_count} documents updated")
# #         except Exception as e:
# #             logger.error(f"Error in MongoDB batch update: {e}")
    
# #     def batch_upload_to_qdrant(self):
# #         """Upload chunks and keywords to Qdrant in batches."""
# #         # Process chunk points
# #         chunk_points = []
# #         while not self.qdrant_chunks_queue.empty():
# #             chunk = self.qdrant_chunks_queue.get()
            
# #             point = PointStruct(
# #                 id=str(uuid.uuid4()),
# #                 vector=[float(v) for v in chunk.embedding],
# #                 payload={
# #                     "company_name": chunk.company_name,
# #                     "type": chunk.type,
# #                     "sub_type": chunk.sub_type,
# #                     "title": chunk.title,
# #                     "pdf_link": chunk.pdf_link,
# #                     "chunk": chunk.text,
# #                     "keywords": chunk.keywords,
# #                     "chunk_id": chunk.chunk_id,
# #                     "chunk_index": chunk.chunk_index
# #                 }
# #             )
# #             chunk_points.append(point)
        
# #         if chunk_points:
# #             insert_points_to_collection(self.qdrant_client, self.chunk_collection, chunk_points)
        
# #         # Process keyword points
# #         keyword_points = []
# #         while not self.qdrant_keywords_queue.empty():
# #             keyword_data = self.qdrant_keywords_queue.get()
            
# #             point = PointStruct(
# #                 id=str(uuid.uuid4()),
# #                 vector=[float(v) for v in keyword_data['embedding']],
# #                 payload={
# #                     "company_name": keyword_data['company_name'],
# #                     "type": keyword_data['type'],
# #                     "sub_type": keyword_data['sub_type'],
# #                     "title": keyword_data['title'],
# #                     "pdf_link": keyword_data['pdf_link'],
# #                     "keyword": keyword_data['keyword']
# #                 }
# #             )
# #             keyword_points.append(point)
        
# #         if keyword_points:
# #             insert_points_to_collection(self.qdrant_client, self.keyword_collection, keyword_points)
    
# #     async def run_pipeline(self, limit: int = None):
# #         """Run the complete RAG ingestion pipeline with workers."""
# #         start_time = time.time()
# #         logger.info("Starting MongoDB RAG ingestion pipeline...")
        
# #         try:
# #             # Setup Qdrant collections
# #             self.setup_qdrant_collections()
            
# #             # Get documents to process
# #             documents = self.get_documents_to_process(limit)
# #             if not documents:
# #                 logger.info("No documents need processing")
# #                 return
            
# #             self.stats['total_documents'] = len(documents)
            
# #             # Process documents concurrently
# #             async with PDFDownloader() as downloader:
# #                 tasks = []
# #                 semaphore = asyncio.Semaphore(self.max_workers)
                
# #                 async def process_with_semaphore(doc):
# #                     async with semaphore:
# #                         return await self.process_single_document(doc, downloader)
                
# #                 # Create tasks for all documents
# #                 for document in documents:
# #                     task = asyncio.create_task(process_with_semaphore(document))
# #                     tasks.append(task)
                
# #                 # Process documents and collect results
# #                 results = []
# #                 for i, task in enumerate(asyncio.as_completed(tasks), 1):
# #                     result = await task
# #                     results.append(result)
                    
# #                     if result.success:
# #                         self.stats['processed_documents'] += 1
# #                         self.stats['total_chunks'] += result.chunks_count
# #                         self.stats['total_keywords'] += result.keywords_count
# #                     else:
# #                         self.stats['failed_documents'] += 1
# #                         logger.error(f"Failed to process document {result.document_id}: {result.error_message}")
                    
# #                     # Log progress
# #                     if i % 10 == 0 or i == len(tasks):
# #                         logger.info(f"Progress: {i}/{len(tasks)} documents processed")
                    
# #                     # Batch upload to databases every 50 documents
# #                     if i % 50 == 0:
# #                         self.batch_update_mongodb()
# #                         self.batch_upload_to_qdrant()
                
# #                 # Final batch uploads
# #                 self.batch_update_mongodb()
# #                 self.batch_upload_to_qdrant()
                
# #                 # Cleanup temporary files
# #                 downloader.cleanup()
            
# #             # Update statistics
# #             self.stats['processing_time'] = time.time() - start_time
            
# #             # Print final summary
# #             self.print_final_summary()
            
# #             logger.info("MongoDB RAG ingestion pipeline completed successfully!")
            
# #         except Exception as e:
# #             logger.error(f"Pipeline failed: {e}")
# #             raise
    
# #     def print_final_summary(self):
# #         """Print comprehensive summary of the ingestion process."""
# #         print("\n" + "="*80)
# #         print("MONGODB RAG INGESTION PIPELINE SUMMARY")
# #         print("="*80)
        
# #         print(f"Total Documents: {self.stats['total_documents']}")
# #         print(f"Successfully Processed: {self.stats['processed_documents']}")
# #         print(f"Failed: {self.stats['failed_documents']}")
# #         print(f"Success Rate: {(self.stats['processed_documents']/self.stats['total_documents']*100):.1f}%")
        
# #         print(f"\nData Generated:")
# #         print(f"  Total Chunks: {self.stats['total_chunks']:,}")
# #         print(f"  Total Keywords: {self.stats['total_keywords']:,}")
# #         print(f"  Avg Chunks per Document: {self.stats['total_chunks']/max(self.stats['processed_documents'], 1):.1f}")
        
# #         print(f"\nConfiguration:")
# #         print(f"  Max Workers: {self.max_workers}")
# #         print(f"  Chunk Size: {self.chunk_size}")
# #         print(f"  Chunk Overlap: {self.chunk_overlap}")
# #         print(f"  Embedding Model: {self.embedding_model}")
# #         print(f"  Embedding Dimensions: {self.embedding_dimensions}")
        
# #         print(f"\nPerformance:")
# #         print(f"  Total Processing Time: {self.stats['processing_time']:.2f} seconds")
# #         print(f"  Avg Time per Document: {self.stats['processing_time']/max(self.stats['processed_documents'], 1):.2f} seconds")
        
# #         # Token usage summary
# #         if hasattr(self.token_tracker, 'print_summary'):
# #             self.token_tracker.print_summary()
        
# #         print("="*80)

# # async def main():
# #     """Main function to run the MongoDB RAG ingestion pipeline."""
# #     try:
# #         # Configuration
# #         max_workers = int(os.getenv("MAX_WORKERS", "5"))
# #         limit = int(os.getenv("DOCUMENT_LIMIT", "0")) or None
        
# #         pipeline = MongoRAGIngestionPipeline(max_workers=max_workers)
# #         await pipeline.run_pipeline(limit=limit)
        
# #     except Exception as e:
# #         logger.error(f"Main pipeline execution failed: {e}")
# #         raise

# # if __name__ == "__main__":
# #     asyncio.run(main())




# import os
# import logging
# import asyncio
# import numpy as np
# import json
# import requests
# import tempfile
# import aiohttp
# import aiofiles
# from pathlib import Path
# from typing import List, Dict, Any, Optional, Tuple, Set
# from dataclasses import dataclass
# import hashlib
# import uuid
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import threading
# from queue import Queue
# import time
# from urllib.parse import urlparse
# import re
# from collections import defaultdict

# from qdrant_client import QdrantClient
# from qdrant_client.http import models
# from qdrant_client.models import Distance, VectorParams, PointStruct
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from openai import AsyncOpenAI
# from dotenv import load_dotenv
# import pymongo
# from pymongo import UpdateOne
# from bson import ObjectId

# from clients.parsing_client import PDFParsingClient
# from clients.mongo_client import MongoDBClient
# from clients.qdrant_client import get_qdrant_client, create_collection, insert_points_to_collection
# from clients.token_tracker import TokenTracker

# # Load environment variables
# load_dotenv()

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('rag_ingestion_mongo.log'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# @dataclass
# class DocumentChunk:
#     """Represents a document chunk with metadata and keywords."""
#     text: str
#     source_file: str
#     chunk_index: int
#     start_char: int
#     end_char: int
#     chunk_id: str
#     keywords: List[str]
#     embedding: Optional[List[float]] = None
#     company_name: str = ""
#     type: str = ""
#     sub_type: str = ""
#     title: str = ""
#     pdf_link: str = ""

# @dataclass
# class ProcessingResult:
#     """Result of processing a single document."""
#     document_id: str
#     success: bool
#     chunks_count: int
#     keywords_count: int
#     error_message: str = ""
#     processing_time: float = 0.0

# class KeywordExtractor:
#     """Advanced keyword extraction using multiple strategies."""
    
#     def __init__(self, openai_client: AsyncOpenAI):
#         self.openai_client = openai_client
        
#         # Insurance domain specific terms
#         self.insurance_terms = {
#             'coverage', 'premium', 'deductible', 'policy', 'claim', 'benefit', 'exclusion',
#             'copay', 'coinsurance', 'underwriting', 'actuarial', 'rider', 'annuity',
#             'endowment', 'term insurance', 'whole life', 'disability', 'critical illness',
#             'hospitalization', 'outpatient', 'inpatient', 'maternity', 'dental', 'vision',
#             'pharmaceutical', 'pre-existing', 'waiting period', 'sum insured', 'family floater',
#             'individual', 'group', 'corporate', 'health insurance', 'life insurance',
#             'motor insurance', 'travel insurance', 'home insurance', 'fire insurance'
#         }
        
#         # Common stop words to exclude
#         self.stop_words = {
#             'the', 'is', 'at', 'which', 'on', 'and', 'a', 'to', 'are', 'as', 'was',
#             'for', 'an', 'be', 'by', 'i', 'you', 'it', 'of', 'or', 'will', 'my',
#             'one', 'have', 'from', 'or', 'had', 'but', 'not', 'what', 'all', 'were',
#             'they', 'we', 'when', 'your', 'can', 'said', 'there', 'each', 'do',
#             'their', 'time', 'if', 'up', 'out', 'many', 'then', 'them', 'these',
#             'so', 'some', 'her', 'would', 'make', 'like', 'into', 'him', 'has',
#             'two', 'more', 'very', 'after', 'words', 'first', 'where', 'much',
#             'through', 'back', 'years', 'work', 'came', 'right', 'still', 'such',
#             'because', 'turn', 'here', 'why', 'asked', 'went', 'men', 'read',
#             'need', 'land', 'different', 'home', 'us', 'move', 'try', 'kind',
#             'hand', 'picture', 'again', 'change', 'off', 'play', 'spell', 'air',
#             'away', 'animal', 'house', 'point', 'page', 'letter', 'mother', 'answer',
#             'found', 'study', 'still', 'learn', 'should', 'america', 'world'
#         }
    
#     def extract_statistical_keywords(self, text: str, max_keywords: int = 15) -> List[str]:
#         """Extract keywords using statistical methods."""
#         # Clean and tokenize text
#         words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
#         # Filter out stop words and get word frequencies
#         word_freq = defaultdict(int)
#         for word in words:
#             if word not in self.stop_words and len(word) > 2:
#                 word_freq[word] += 1
        
#         # Prioritize insurance domain terms
#         scored_words = []
#         for word, freq in word_freq.items():
#             score = freq
#             if word in self.insurance_terms or any(term in word for term in self.insurance_terms):
#                 score *= 3  # Boost insurance-related terms
#             scored_words.append((word, score))
        
#         # Sort by score and return top keywords
#         scored_words.sort(key=lambda x: x[1], reverse=True)
#         return [word for word, _ in scored_words[:max_keywords]]
    
#     async def extract_llm_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
#         """Extract keywords using LLM for semantic understanding."""
#         try:
#             prompt = f"""Extract the most important {max_keywords} keywords from this insurance policy text. 
# Focus on:
# - Insurance terms and concepts
# - Coverage types and benefits
# - Policy conditions and exclusions
# - Financial terms
# - Medical conditions or treatments mentioned

# Text: {text[:2000]}...

# Return only the keywords as a comma-separated list, no explanations."""

#             response = await self.openai_client.chat.completions.create(
#                 model="gpt-4o-mini",
#                 messages=[{"role": "user", "content": prompt}],
#                 max_tokens=150,
#                 temperature=0.3
#             )
            
#             keywords_text = response.choices[0].message.content.strip()
#             keywords = [kw.strip().lower() for kw in keywords_text.split(',') if kw.strip()]
#             return keywords[:max_keywords]
            
#         except Exception as e:
#             logger.warning(f"LLM keyword extraction failed: {e}")
#             return []
    
#     async def extract_keywords(self, text: str, max_keywords: int = 20) -> List[str]:
#         """Extract keywords using hybrid approach."""
#         # Get statistical keywords
#         stat_keywords = self.extract_statistical_keywords(text, max_keywords // 2)
        
#         # Get LLM keywords for semantic understanding
#         llm_keywords = await self.extract_llm_keywords(text, max_keywords // 2)
        
#         # Combine and deduplicate
#         all_keywords = list(set(stat_keywords + llm_keywords))
        
#         # Ensure we don't exceed max_keywords
#         return all_keywords[:max_keywords]

# class PDFDownloader:
#     """Efficient PDF downloader with retry logic and caching."""
    
#     def __init__(self, temp_dir: str = None):
#         self.temp_dir = temp_dir or tempfile.mkdtemp()
#         Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
#         self.session = None
    
#     async def __aenter__(self):
#         self.session = aiohttp.ClientSession(
#             timeout=aiohttp.ClientTimeout(total=300),  # 5 minutes timeout
#             connector=aiohttp.TCPConnector(limit=10)
#         )
#         return self
    
#     async def __aexit__(self, exc_type, exc_val, exc_tb):
#         if self.session:
#             await self.session.close()
    
#     async def download_pdf(self, url: str, document_id: str, max_retries: int = 3) -> Optional[str]:
#         """Download PDF with retry logic."""
#         for attempt in range(max_retries):
#             try:
#                 # Generate filename from URL and document ID
#                 parsed_url = urlparse(url)
#                 filename = f"{document_id}_{hashlib.md5(url.encode()).hexdigest()[:8]}.pdf"
#                 file_path = os.path.join(self.temp_dir, filename)
                
#                 # Check if file already exists
#                 if os.path.exists(file_path):
#                     logger.info(f"PDF already exists: {filename}")
#                     return file_path
                
#                 logger.info(f"Downloading PDF from {url} (attempt {attempt + 1})")
                
#                 async with self.session.get(url) as response:
#                     if response.status == 200:
#                         content = await response.read()
                        
#                         # Validate it's a PDF
#                         if content.startswith(b'%PDF'):
#                             async with aiofiles.open(file_path, 'wb') as f:
#                                 await f.write(content)
                            
#                             logger.info(f"Successfully downloaded: {filename}")
#                             return file_path
#                         else:
#                             logger.warning(f"Downloaded content is not a valid PDF: {url}")
#                             return None
#                     else:
#                         logger.warning(f"HTTP {response.status} for {url}")
                        
#             except Exception as e:
#                 logger.warning(f"Download attempt {attempt + 1} failed for {url}: {e}")
#                 if attempt < max_retries - 1:
#                     await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
#         logger.error(f"Failed to download PDF after {max_retries} attempts: {url}")
#         return None
    
#     def cleanup(self):
#         """Clean up temporary files."""
#         import shutil
#         try:
#             shutil.rmtree(self.temp_dir)
#             logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
#         except Exception as e:
#             logger.warning(f"Failed to cleanup temp directory: {e}")

# class MongoRAGIngestionPipeline:
#     """High-performance RAG ingestion pipeline for MongoDB insurance policies."""
    
#     def __init__(self, max_workers: int = 5):
#         # Configuration
#         self.collection_name = "policies"
#         self.chunk_collection = "policy_chunks2"
#         self.keyword_collection = "policy_keywords2"
#         self.max_workers = max_workers
        
#         # Text splitter configuration
#         self.chunk_size = 1024
#         self.chunk_overlap = 200
        
#         # OpenAI configuration
#         self.embedding_model = "text-embedding-3-large"
#         self.embedding_dimensions = 1024
        
#         # Initialize clients
#         self.token_tracker = TokenTracker()
#         self.parsing_client = PDFParsingClient()
#         self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#         self.mongo_client = MongoDBClient()
#         self.qdrant_client = get_qdrant_client()
#         self.keyword_extractor = KeywordExtractor(self.openai_client)
        
#         # Initialize text splitter
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=self.chunk_size,
#             chunk_overlap=self.chunk_overlap,
#             length_function=len,
#             separators=["\n\n", "\n", ".", "!", "?", " ", ""]
#         )
        
#         # Processing statistics
#         self.stats = {
#             'total_documents': 0,
#             'processed_documents': 0,
#             'failed_documents': 0,
#             'total_chunks': 0,
#             'total_keywords': 0,
#             'processing_time': 0.0
#         }
        
#         # Thread locks for concurrent operations
#         self.stats_lock = threading.Lock()
        
#     def setup_qdrant_collections(self):
#         """Setup Qdrant collections for chunks and keywords."""
#         logger.info("Setting up Qdrant collections...")
        
#         try:
#             # Create chunk collection
#             if not create_collection(self.qdrant_client, self.chunk_collection, self.embedding_dimensions):
#                 logger.warning(f"Failed to create chunk collection: {self.chunk_collection}")
            
#             # Create keyword collection
#             if not create_collection(self.qdrant_client, self.keyword_collection, self.embedding_dimensions):
#                 logger.warning(f"Failed to create keyword collection: {self.keyword_collection}")
            
#             logger.info("Qdrant collections setup completed")
            
#         except Exception as e:
#             logger.error(f"Error setting up Qdrant collections: {e}")
#             raise
    
#     def get_documents_to_process(self, limit: int = None) -> List[Dict[str, Any]]:
#         """Get documents from MongoDB that need processing."""
#         query = {
#             "$or": [
#                 {"parsed_content": {"$exists": False}},
#                 {"chunks": {"$exists": False}},
#                 {"keywords": {"$exists": False}}
#             ]
#         }
        
#         try:
#             documents = self.mongo_client.find_documents(
#                 collection_name=self.collection_name,
#                 query=query,
#                 limit=limit or 0
#             )
            
#             logger.info(f"Found {len(documents)} documents to process")
#             return documents
            
#         except Exception as e:
#             logger.error(f"Error fetching documents from MongoDB: {e}")
#             return []
    
#     async def update_mongo_immediately(self, document_id, update_data: Dict[str, Any]):
#         """Update MongoDB document immediately after processing."""
#         try:
#             collection = self.mongo_client.get_collection(self.collection_name)
            
#             # Handle both ObjectId and string cases
#             if isinstance(document_id, str):
#                 try:
#                     query_id = ObjectId(document_id)
#                 except:
#                     query_id = document_id
#             else:
#                 query_id = document_id
            
#             result = collection.update_one(
#                 {"_id": query_id},
#                 {"$set": update_data}
#             )
            
#             if result.modified_count > 0:
#                 logger.info(f"✅ MongoDB updated for document {document_id}")
#             else:
#                 logger.warning(f"⚠️ No MongoDB update for document {document_id} - matched: {result.matched_count}, modified: {result.modified_count}")
                
#         except Exception as e:
#             logger.error(f"❌ Error updating MongoDB for document {document_id}: {e}")
    
#     async def upload_chunks_to_qdrant_immediately(self, chunks: List[DocumentChunk]):
#         """Upload chunk embeddings to Qdrant immediately."""
#         try:
#             chunk_points = []
            
#             for chunk in chunks:
#                 if chunk.embedding:
#                     point = PointStruct(
#                         id=str(uuid.uuid4()),
#                         vector=[float(v) for v in chunk.embedding],
#                         payload={
#                             "company_name": chunk.company_name,
#                             "type": chunk.type,
#                             "sub_type": chunk.sub_type,
#                             "title": chunk.title,
#                             "pdf_link": chunk.pdf_link,
#                             "chunk": chunk.text,
#                             "keywords": chunk.keywords,
#                             "chunk_id": chunk.chunk_id,
#                             "chunk_index": chunk.chunk_index
#                         }
#                     )
#                     chunk_points.append(point)
            
#             if chunk_points:
#                 success = insert_points_to_collection(
#                     self.qdrant_client, 
#                     self.chunk_collection, 
#                     chunk_points
#                 )
                
#                 if success:
#                     logger.info(f"Uploaded {len(chunk_points)} chunk points to Qdrant")
#                 else:
#                     logger.error(f"Failed to upload chunk points to Qdrant")
                    
#         except Exception as e:
#             logger.error(f"Error uploading chunks to Qdrant: {e}")
    
#     async def upload_keywords_to_qdrant_immediately(self, keyword_embeddings: List[Dict[str, Any]]):
#         """Upload keyword embeddings to Qdrant immediately."""
#         try:
#             keyword_points = []
            
#             for keyword_data in keyword_embeddings:
#                 point = PointStruct(
#                     id=str(uuid.uuid4()),
#                     vector=[float(v) for v in keyword_data['embedding']],
#                     payload={
#                         "company_name": keyword_data['company_name'],
#                         "type": keyword_data['type'],
#                         "sub_type": keyword_data['sub_type'],
#                         "title": keyword_data['title'],
#                         "pdf_link": keyword_data['pdf_link'],
#                         "keyword": keyword_data['keyword']
#                     }
#                 )
#                 keyword_points.append(point)
            
#             if keyword_points:
#                 success = insert_points_to_collection(
#                     self.qdrant_client, 
#                     self.keyword_collection, 
#                     keyword_points
#                 )
                
#                 if success:
#                     logger.info(f"Uploaded {len(keyword_points)} keyword points to Qdrant")
#                 else:
#                     logger.error(f"Failed to upload keyword points to Qdrant")
                    
#         except Exception as e:
#             logger.error(f"Error uploading keywords to Qdrant: {e}")
    
#     async def process_single_document(self, document: Dict[str, Any], downloader: PDFDownloader) -> ProcessingResult:
#         """Process a single document: download, parse, chunk, extract keywords, upload immediately."""
#         start_time = time.time()
#         original_doc_id = document['_id']  # Keep original ObjectId
#         doc_id = str(original_doc_id)      # String version for logging
        
#         try:
#             # Get PDF URL (check multiple possible field names)
#             pdf_url = document.get('pdf_link') or document.get('url') or document.get('pdf_url')
            
#             if not pdf_url:
#                 return ProcessingResult(
#                     document_id=doc_id,
#                     success=False,
#                     chunks_count=0,
#                     keywords_count=0,
#                     error_message="No PDF URL found in document"
#                 )
            
#             # Download PDF
#             pdf_path = await downloader.download_pdf(pdf_url, doc_id)
#             if not pdf_path:
#                 return ProcessingResult(
#                     document_id=doc_id,
#                     success=False,
#                     chunks_count=0,
#                     keywords_count=0,
#                     error_message="Failed to download PDF"
#                 )
            
#             # Parse PDF to markdown
#             try:
#                 parsed_content = self.parsing_client.parse_single_pdf(pdf_path)
#                 if not parsed_content or len(parsed_content.strip()) < 100:
#                     return ProcessingResult(
#                         document_id=doc_id,
#                         success=False,
#                         chunks_count=0,
#                         keywords_count=0,
#                         error_message="PDF parsing failed or content too short"
#                     )
#             except Exception as e:
#                 return ProcessingResult(
#                     document_id=doc_id,
#                     success=False,
#                     chunks_count=0,
#                     keywords_count=0,
#                     error_message=f"PDF parsing failed: {str(e)}"
#                 )
            
#             # Create chunks
#             text_chunks = self.text_splitter.split_text(parsed_content)
            
#             if not text_chunks:
#                 return ProcessingResult(
#                     document_id=doc_id,
#                     success=False,
#                     chunks_count=0,
#                     keywords_count=0,
#                     error_message="No chunks created from parsed content"
#                 )
            
#             # Extract metadata
#             company_name = document.get('company_name', '')
#             type_field = document.get('type', '')
#             sub_type = document.get('sub_type', '')
#             title = document.get('title', '')
            
#             # Process each chunk
#             document_chunks = []
#             all_keywords_set = set()
            
#             char_position = 0
#             for i, chunk_text in enumerate(text_chunks):
#                 # Find chunk position in original text
#                 chunk_start = parsed_content.find(chunk_text, char_position)
#                 if chunk_start == -1:
#                     chunk_start = char_position
#                 chunk_end = chunk_start + len(chunk_text)
                
#                 # Extract keywords for this chunk
#                 chunk_keywords = await self.keyword_extractor.extract_keywords(chunk_text, max_keywords=15)
#                 all_keywords_set.update(chunk_keywords)
                
#                 # Create DocumentChunk
#                 chunk_id = self._generate_chunk_id(doc_id, i, chunk_text)
#                 chunk = DocumentChunk(
#                     text=chunk_text.strip(),
#                     source_file=title,
#                     chunk_index=i,
#                     start_char=chunk_start,
#                     end_char=chunk_end,
#                     chunk_id=chunk_id,
#                     keywords=chunk_keywords,
#                     company_name=company_name,
#                     type=type_field,
#                     sub_type=sub_type,
#                     title=title,
#                     pdf_link=pdf_url
#                 )
                
#                 document_chunks.append(chunk)
#                 char_position = chunk_end
            
#             # Generate embeddings for chunks
#             chunks_with_embeddings = await self.generate_embeddings(document_chunks)
            
#             # Prepare data for MongoDB update
#             chunks_data = []
#             for chunk in chunks_with_embeddings:
#                 chunks_data.append({
#                     'chunk_id': chunk.chunk_id,
#                     'text': chunk.text,
#                     'chunk_index': chunk.chunk_index,
#                     'keywords': chunk.keywords,
#                     'start_char': chunk.start_char,
#                     'end_char': chunk.end_char
#                 })
            
#             all_keywords = list(all_keywords_set)
            
#             # Prepare MongoDB update data
#             mongo_update_data = {
#                 'parsed_content': parsed_content,
#                 'chunks': chunks_data,
#                 'keywords': all_keywords,
#                 'processing_timestamp': time.time()
#             }
            
#             # IMMEDIATE UPLOADS - Run all uploads concurrently
#             upload_tasks = []
            
#             # 1. Update MongoDB immediately - pass the original ObjectId
#             upload_tasks.append(
#                 self.update_mongo_immediately(original_doc_id, mongo_update_data)  # Pass ObjectId, not string
#             )
            
#             # 2. Upload chunks to Qdrant immediately
#             if chunks_with_embeddings:
#                 upload_tasks.append(
#                     self.upload_chunks_to_qdrant_immediately(chunks_with_embeddings)
#                 )
            
#             # 3. Generate and upload keyword embeddings immediately
#             if all_keywords:
#                 keyword_embeddings = await self.generate_keyword_embeddings(all_keywords, document)
#                 if keyword_embeddings:
#                     upload_tasks.append(
#                         self.upload_keywords_to_qdrant_immediately(keyword_embeddings)
#                     )
            
#             # Execute all uploads concurrently
#             if upload_tasks:
#                 await asyncio.gather(*upload_tasks, return_exceptions=True)
            
#             processing_time = time.time() - start_time
            
#             # Update stats thread-safely
#             with self.stats_lock:
#                 self.stats['processed_documents'] += 1
#                 self.stats['total_chunks'] += len(chunks_with_embeddings)
#                 self.stats['total_keywords'] += len(all_keywords)
            
#             logger.info(f"✅ COMPLETED document {doc_id}: {len(chunks_with_embeddings)} chunks, {len(all_keywords)} keywords, uploaded to MongoDB + Qdrant")
            
#             return ProcessingResult(
#                 document_id=doc_id,
#                 success=True,
#                 chunks_count=len(chunks_with_embeddings),
#                 keywords_count=len(all_keywords),
#                 processing_time=processing_time
#             )
            
#         except Exception as e:
#             processing_time = time.time() - start_time
            
#             # Update stats thread-safely
#             with self.stats_lock:
#                 self.stats['failed_documents'] += 1
            
#             logger.error(f"❌ FAILED document {doc_id}: {e}")
#             return ProcessingResult(
#                 document_id=doc_id,
#                 success=False,
#                 chunks_count=0,
#                 keywords_count=0,
#                 error_message=str(e),
#                 processing_time=processing_time
#             )
    
#     def _generate_chunk_id(self, doc_id: str, chunk_index: int, chunk_text: str) -> str:
#         """Generate a unique ID for a chunk."""
#         content_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:8]
#         return f"{doc_id}_{chunk_index:04d}_{content_hash}"
    
#     async def generate_embeddings(self, chunks: List[DocumentChunk], batch_size: int = 50) -> List[DocumentChunk]:
#         """Generate embeddings for chunks in batches."""
#         logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        
#         processed_chunks = []
        
#         for i in range(0, len(chunks), batch_size):
#             batch = chunks[i:i + batch_size]
            
#             try:
#                 # Prepare texts for embedding
#                 texts = [chunk.text for chunk in batch]
                
#                 # Generate embeddings
#                 response = await self.openai_client.embeddings.create(
#                     model=self.embedding_model,
#                     input=texts,
#                     dimensions=self.embedding_dimensions,
#                     encoding_format="float"
#                 )
                
#                 # Assign embeddings to chunks
#                 for chunk, embedding_data in zip(batch, response.data):
#                     chunk.embedding = embedding_data.embedding
#                     processed_chunks.append(chunk)
                
#                 # Track token usage
#                 for text in texts:
#                     self.token_tracker.track_openai_embedding(text, self.embedding_model)
                
#                 # Small delay to respect rate limits
#                 await asyncio.sleep(0.1)
                
#             except Exception as e:
#                 logger.error(f"Error generating embeddings for batch: {e}")
#                 # Add chunks without embeddings to avoid losing data
#                 for chunk in batch:
#                     chunk.embedding = None
#                     processed_chunks.append(chunk)
        
#         valid_chunks = [chunk for chunk in processed_chunks if chunk.embedding is not None]
#         logger.info(f"Generated embeddings for {len(valid_chunks)}/{len(chunks)} chunks")
#         return valid_chunks
    
#     async def generate_keyword_embeddings(self, keywords: List[str], document: Dict[str, Any]) -> List[Dict[str, Any]]:
#         """Generate embeddings for keywords."""
#         try:
#             # Create keyword phrases for better semantic representation
#             keyword_phrases = [f"insurance {keyword}" for keyword in keywords]
            
#             response = await self.openai_client.embeddings.create(
#                 model=self.embedding_model,
#                 input=keyword_phrases,
#                 dimensions=self.embedding_dimensions,
#                 encoding_format="float"
#             )
            
#             keyword_data = []
#             for keyword, embedding_data in zip(keywords, response.data):
#                 keyword_data.append({
#                     'keyword': keyword,
#                     'embedding': embedding_data.embedding,
#                     'company_name': document.get('company_name', ''),
#                     'type': document.get('type', ''),
#                     'sub_type': document.get('sub_type', ''),
#                     'title': document.get('title', ''),
#                     'pdf_link': document.get('pdf_link') or document.get('url') or document.get('pdf_url', '')
#                 })
            
#             return keyword_data
            
#         except Exception as e:
#             logger.error(f"Error generating keyword embeddings: {e}")
#             return []
    
#     async def run_pipeline(self, limit: int = None):
#         """Run the complete RAG ingestion pipeline with immediate uploads."""
#         start_time = time.time()
#         logger.info("🚀 Starting MongoDB RAG ingestion pipeline with IMMEDIATE UPLOADS...")
        
#         try:
#             # Setup Qdrant collections
#             self.setup_qdrant_collections()
            
#             # Get documents to process
#             documents = self.get_documents_to_process(limit)
#             if not documents:
#                 logger.info("No documents need processing")
#                 return
            
#             self.stats['total_documents'] = len(documents)
#             logger.info(f"📄 Processing {len(documents)} documents with {self.max_workers} workers")
            
#             # Process documents concurrently with immediate uploads
#             async with PDFDownloader() as downloader:
#                 tasks = []
#                 semaphore = asyncio.Semaphore(self.max_workers)
                
#                 async def process_with_semaphore(doc):
#                     async with semaphore:
#                         return await self.process_single_document(doc, downloader)
                
#                 # Create tasks for all documents
#                 for document in documents:
#                     task = asyncio.create_task(process_with_semaphore(document))
#                     tasks.append(task)
                
#                 # Process documents and collect results
#                 results = []
#                 completed_count = 0
                
#                 for task in asyncio.as_completed(tasks):
#                     result = await task
#                     results.append(result)
#                     completed_count += 1
                    
#                     # Log progress with real-time stats
#                     if completed_count % 5 == 0 or completed_count == len(tasks):
#                         success_rate = (self.stats['processed_documents'] / completed_count * 100) if completed_count > 0 else 0
#                         logger.info(f"📊 Progress: {completed_count}/{len(tasks)} | ✅ {self.stats['processed_documents']} success | ❌ {self.stats['failed_documents']} failed | 📈 {success_rate:.1f}% success rate")
                
#                 # Cleanup temporary files
#                 downloader.cleanup()
            
#             # Update final statistics
#             self.stats['processing_time'] = time.time() - start_time
            
#             # Print final summary
#             self.print_final_summary()
            
#             logger.info("🎉 MongoDB RAG ingestion pipeline completed successfully with IMMEDIATE UPLOADS!")
            
#         except Exception as e:
#             logger.error(f"Pipeline failed: {e}")
#             raise
    
#     def print_final_summary(self):
#         """Print comprehensive summary of the ingestion process."""
#         print("\n" + "="*80)
#         print("🚀 MONGODB RAG INGESTION PIPELINE SUMMARY (IMMEDIATE UPLOADS)")
#         print("="*80)
        
#         print(f"📄 Total Documents: {self.stats['total_documents']}")
#         print(f"✅ Successfully Processed: {self.stats['processed_documents']}")
#         print(f"❌ Failed: {self.stats['failed_documents']}")
#         success_rate = (self.stats['processed_documents']/max(self.stats['total_documents'], 1)*100)
#         print(f"📈 Success Rate: {success_rate:.1f}%")
        
#         print(f"\n📊 Data Generated & Uploaded:")
#         print(f"  🔗 Total Chunks: {self.stats['total_chunks']:,}")
#         print(f"  🏷️ Total Keywords: {self.stats['total_keywords']:,}")
#         avg_chunks = self.stats['total_chunks']/max(self.stats['processed_documents'], 1)
#         print(f"  📋 Avg Chunks per Document: {avg_chunks:.1f}")
        
#         print(f"\n⚙️ Configuration:")
#         print(f"  👥 Max Workers: {self.max_workers}")
#         print(f"  📏 Chunk Size: {self.chunk_size}")
#         print(f"  🔄 Chunk Overlap: {self.chunk_overlap}")
#         print(f"  🤖 Embedding Model: {self.embedding_model}")
#         print(f"  📐 Embedding Dimensions: {self.embedding_dimensions}")
        
#         print(f"\n⚡ Performance:")
#         print(f"  ⏱️ Total Processing Time: {self.stats['processing_time']:.2f} seconds")
#         avg_time = self.stats['processing_time']/max(self.stats['processed_documents'], 1)
#         print(f"  ⏰ Avg Time per Document: {avg_time:.2f} seconds")
        
#         # Token usage summary
#         if hasattr(self.token_tracker, 'print_summary'):
#             print(f"\n💰 Token Usage:")
#             self.token_tracker.print_summary()
        
#         print("\n🎯 Upload Strategy: IMMEDIATE - Each document uploaded to MongoDB + Qdrant as soon as processing completes")
#         print("="*80)

# async def main():
#     """Main function to run the MongoDB RAG ingestion pipeline."""
#     try:
#         # Configuration
#         max_workers = int(os.getenv("MAX_WORKERS", "8"))
#         limit = int(os.getenv("DOCUMENT_LIMIT", "0")) or None
        
#         logger.info(f"🔧 Configuration: {max_workers} workers, limit: {limit or 'no limit'}")
        
#         pipeline = MongoRAGIngestionPipeline(max_workers=max_workers)
#         await pipeline.run_pipeline(limit=limit)
        
#     except Exception as e:
#         logger.error(f"Main pipeline execution failed: {e}")
#         raise

# if __name__ == "__main__":
#     asyncio.run(main())



import os
import logging
import asyncio
import numpy as np
import json
import requests
import tempfile
import aiohttp
import aiofiles
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
import hashlib
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import time
from urllib.parse import urlparse
import re
from collections import defaultdict

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import AsyncOpenAI
from dotenv import load_dotenv
import pymongo
from pymongo import UpdateOne
from bson import ObjectId

from clients.parsing_client import PDFParsingClient
from clients.mongo_client import MongoDBClient
from clients.qdrant_client import get_qdrant_client, create_collection, insert_points_to_collection
from clients.token_tracker import TokenTracker

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_ingestion_mongo.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a document chunk with metadata and keywords."""
    text: str
    source_file: str
    chunk_index: int
    start_char: int
    end_char: int
    chunk_id: str
    keywords: List[str]
    embedding: Optional[List[float]] = None
    company_name: str = ""
    type: str = ""
    sub_type: str = ""
    title: str = ""
    pdf_link: str = ""

@dataclass
class ProcessingResult:
    """Result of processing a single document."""
    document_id: str
    success: bool
    chunks_count: int
    keywords_count: int
    error_message: str = ""
    processing_time: float = 0.0

class KeywordExtractor:
    """Advanced keyword extraction using multiple strategies."""
    
    def __init__(self, openai_client: AsyncOpenAI):
        self.openai_client = openai_client
        
        # Insurance domain specific terms
        self.insurance_terms = {
            'coverage', 'premium', 'deductible', 'policy', 'claim', 'benefit', 'exclusion',
            'copay', 'coinsurance', 'underwriting', 'actuarial', 'rider', 'annuity',
            'endowment', 'term insurance', 'whole life', 'disability', 'critical illness',
            'hospitalization', 'outpatient', 'inpatient', 'maternity', 'dental', 'vision',
            'pharmaceutical', 'pre-existing', 'waiting period', 'sum insured', 'family floater',
            'individual', 'group', 'corporate', 'health insurance', 'life insurance',
            'motor insurance', 'travel insurance', 'home insurance', 'fire insurance'
        }
        
        # Common stop words to exclude
        self.stop_words = {
            'the', 'is', 'at', 'which', 'on', 'and', 'a', 'to', 'are', 'as', 'was',
            'for', 'an', 'be', 'by', 'i', 'you', 'it', 'of', 'or', 'will', 'my',
            'one', 'have', 'from', 'or', 'had', 'but', 'not', 'what', 'all', 'were',
            'they', 'we', 'when', 'your', 'can', 'said', 'there', 'each', 'do',
            'their', 'time', 'if', 'up', 'out', 'many', 'then', 'them', 'these',
            'so', 'some', 'her', 'would', 'make', 'like', 'into', 'him', 'has',
            'two', 'more', 'very', 'after', 'words', 'first', 'where', 'much',
            'through', 'back', 'years', 'work', 'came', 'right', 'still', 'such',
            'because', 'turn', 'here', 'why', 'asked', 'went', 'men', 'read',
            'need', 'land', 'different', 'home', 'us', 'move', 'try', 'kind',
            'hand', 'picture', 'again', 'change', 'off', 'play', 'spell', 'air',
            'away', 'animal', 'house', 'point', 'page', 'letter', 'mother', 'answer',
            'found', 'study', 'still', 'learn', 'should', 'america', 'world'
        }
    
    def extract_statistical_keywords(self, text: str, max_keywords: int = 15) -> List[str]:
        """Extract keywords using statistical methods."""
        # Clean and tokenize text
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out stop words and get word frequencies
        word_freq = defaultdict(int)
        for word in words:
            if word not in self.stop_words and len(word) > 2:
                word_freq[word] += 1
        
        # Prioritize insurance domain terms
        scored_words = []
        for word, freq in word_freq.items():
            score = freq
            if word in self.insurance_terms or any(term in word for term in self.insurance_terms):
                score *= 3  # Boost insurance-related terms
            scored_words.append((word, score))
        
        # Sort by score and return top keywords
        scored_words.sort(key=lambda x: x[1], reverse=True)
        return [word for word, _ in scored_words[:max_keywords]]
    
    async def extract_llm_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords using LLM for semantic understanding."""
        try:
            prompt = f"""Extract the most important {max_keywords} keywords from this insurance policy text. 
Focus on:
- Insurance terms and concepts
- Coverage types and benefits
- Policy conditions and exclusions
- Financial terms
- Medical conditions or treatments mentioned

Text: {text[:2000]}...

Return only the keywords as a comma-separated list, no explanations."""

            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.3
            )
            
            keywords_text = response.choices[0].message.content.strip()
            keywords = [kw.strip().lower() for kw in keywords_text.split(',') if kw.strip()]
            return keywords[:max_keywords]
            
        except Exception as e:
            logger.warning(f"LLM keyword extraction failed: {e}")
            return []
    
    async def extract_keywords(self, text: str, max_keywords: int = 20) -> List[str]:
        """Extract keywords using hybrid approach."""
        # Get statistical keywords
        stat_keywords = self.extract_statistical_keywords(text, max_keywords // 2)
        
        # Get LLM keywords for semantic understanding
        llm_keywords = await self.extract_llm_keywords(text, max_keywords // 2)
        
        # Combine and deduplicate
        all_keywords = list(set(stat_keywords + llm_keywords))
        
        # Ensure we don't exceed max_keywords
        return all_keywords[:max_keywords]

class PDFDownloader:
    """Efficient PDF downloader with retry logic and caching."""
    
    def __init__(self, temp_dir: str = None):
        self.temp_dir = temp_dir or tempfile.mkdtemp()
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300),  # 5 minutes timeout
            connector=aiohttp.TCPConnector(limit=10)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def download_pdf(self, url: str, document_id: str, max_retries: int = 3) -> Optional[str]:
        """Download PDF with retry logic."""
        for attempt in range(max_retries):
            try:
                # Generate filename from URL and document ID
                parsed_url = urlparse(url)
                filename = f"{document_id}_{hashlib.md5(url.encode()).hexdigest()[:8]}.pdf"
                file_path = os.path.join(self.temp_dir, filename)
                
                # Check if file already exists
                if os.path.exists(file_path):
                    logger.info(f"PDF already exists: {filename}")
                    return file_path
                
                logger.info(f"Downloading PDF from {url} (attempt {attempt + 1})")
                
                async with self.session.get(url) as response:
                    if response.status == 200:
                        content = await response.read()
                        
                        # Validate it's a PDF
                        if content.startswith(b'%PDF'):
                            async with aiofiles.open(file_path, 'wb') as f:
                                await f.write(content)
                            
                            logger.info(f"Successfully downloaded: {filename}")
                            return file_path
                        else:
                            logger.warning(f"Downloaded content is not a valid PDF: {url}")
                            return None
                    else:
                        logger.warning(f"HTTP {response.status} for {url}")
                        
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1} failed for {url}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        logger.error(f"Failed to download PDF after {max_retries} attempts: {url}")
        return None
    
    def cleanup(self):
        """Clean up temporary files."""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directory: {e}")

class MongoRAGIngestionPipeline:
    """High-performance RAG ingestion pipeline for MongoDB insurance policies."""
    
    def __init__(self, max_workers: int = 5):
        # Configuration
        self.collection_name = "policies"
        self.chunk_collection = "policy_chunks2"
        self.keyword_collection = "policy_keywords2"
        self.max_workers = max_workers
        
        # Text splitter configuration
        self.chunk_size = 1024
        self.chunk_overlap = 200
        
        # OpenAI configuration
        self.embedding_model = "text-embedding-3-large"
        self.embedding_dimensions = 1024
        
        # Initialize clients
        self.token_tracker = TokenTracker()
        self.parsing_client = PDFParsingClient()
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.mongo_client = MongoDBClient()
        self.qdrant_client = get_qdrant_client()
        self.keyword_extractor = KeywordExtractor(self.openai_client)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )
        
        # Processing statistics
        self.stats = {
            'total_documents': 0,
            'processed_documents': 0,
            'failed_documents': 0,
            'total_chunks': 0,
            'total_keywords': 0,
            'processing_time': 0.0
        }
        
        # Thread locks for concurrent operations
        self.stats_lock = threading.Lock()
        
    def setup_qdrant_collections(self):
        """Setup Qdrant collections for chunks and keywords."""
        logger.info("Setting up Qdrant collections...")
        
        try:
            # Create chunk collection
            if not create_collection(self.qdrant_client, self.chunk_collection, self.embedding_dimensions):
                logger.warning(f"Failed to create chunk collection: {self.chunk_collection}")
            
            # Create keyword collection
            if not create_collection(self.qdrant_client, self.keyword_collection, self.embedding_dimensions):
                logger.warning(f"Failed to create keyword collection: {self.keyword_collection}")
            
            logger.info("Qdrant collections setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up Qdrant collections: {e}")
            raise
    
    def get_documents_to_process(self, limit: int = None) -> List[Dict[str, Any]]:
        """Get documents from MongoDB that need processing."""
        query = {
            "$or": [
                {"parsed_content": {"$exists": False}},
                {"chunks": {"$exists": False}},
                {"keywords": {"$exists": False}}
            ]
        }
        
        try:
            documents = self.mongo_client.find_documents(
                collection_name=self.collection_name,
                query=query,
                limit=limit or 0
            )
            
            logger.info(f"Found {len(documents)} documents to process")
            return documents
            
        except Exception as e:
            logger.error(f"Error fetching documents from MongoDB: {e}")
            return []
    
    async def update_mongo_immediately(self, document_id, update_data: Dict[str, Any]):
        """Update MongoDB document immediately after processing."""
        try:
            collection = self.mongo_client.get_collection(self.collection_name)
            
            # Handle both ObjectId and string cases
            if isinstance(document_id, str):
                try:
                    query_id = ObjectId(document_id)
                except:
                    query_id = document_id
            else:
                query_id = document_id
            
            result = collection.update_one(
                {"_id": query_id},
                {"$set": update_data}
            )
            
            if result.modified_count > 0:
                logger.info(f"✅ MongoDB updated for document {document_id}")
            else:
                logger.warning(f"⚠️ No MongoDB update for document {document_id} - matched: {result.matched_count}, modified: {result.modified_count}")
                
        except Exception as e:
            logger.error(f"❌ Error updating MongoDB for document {document_id}: {e}")
    
    async def upload_chunks_to_qdrant_immediately(self, chunks: List[DocumentChunk]):
        """Upload chunk embeddings to Qdrant immediately."""
        try:
            chunk_points = []
            
            for chunk in chunks:
                if chunk.embedding:
                    point = PointStruct(
                        id=str(uuid.uuid4()),
                        vector=[float(v) for v in chunk.embedding],
                        payload={
                            "company_name": chunk.company_name,
                            "type": chunk.type,
                            "sub_type": chunk.sub_type,
                            "title": chunk.title,
                            "pdf_link": chunk.pdf_link,
                            "chunk": chunk.text,
                            "keywords": chunk.keywords,
                            "chunk_id": chunk.chunk_id,
                            "chunk_index": chunk.chunk_index
                        }
                    )
                    chunk_points.append(point)
            
            if chunk_points:
                success = insert_points_to_collection(
                    self.qdrant_client, 
                    self.chunk_collection, 
                    chunk_points
                )
                
                if success:
                    logger.info(f"Uploaded {len(chunk_points)} chunk points to Qdrant")
                else:
                    logger.error(f"Failed to upload chunk points to Qdrant")
                    
        except Exception as e:
            logger.error(f"Error uploading chunks to Qdrant: {e}")
    
    async def upload_keywords_to_qdrant_immediately(self, keyword_embeddings: List[Dict[str, Any]]):
        """Upload keyword embeddings to Qdrant immediately."""
        try:
            keyword_points = []
            
            for keyword_data in keyword_embeddings:
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=[float(v) for v in keyword_data['embedding']],
                    payload={
                        "company_name": keyword_data['company_name'],
                        "type": keyword_data['type'],
                        "sub_type": keyword_data['sub_type'],
                        "title": keyword_data['title'],
                        "pdf_link": keyword_data['pdf_link'],
                        "keyword": keyword_data['keyword']
                    }
                )
                keyword_points.append(point)
            
            if keyword_points:
                success = insert_points_to_collection(
                    self.qdrant_client, 
                    self.keyword_collection, 
                    keyword_points
                )
                
                if success:
                    logger.info(f"Uploaded {len(keyword_points)} keyword points to Qdrant")
                else:
                    logger.error(f"Failed to upload keyword points to Qdrant")
                    
        except Exception as e:
            logger.error(f"Error uploading keywords to Qdrant: {e}")
    
    async def process_single_document(self, document: Dict[str, Any], downloader: PDFDownloader) -> ProcessingResult:
        """Process a single document: download, parse, chunk, extract keywords, upload immediately."""
        start_time = time.time()
        original_doc_id = document['_id']  # Keep original ObjectId
        doc_id = str(original_doc_id)      # String version for logging
        
        try:
            # Check if document is already processed
            if (document.get('parsed_content') and 
                document.get('chunks') and 
                document.get('keywords')):
                logger.info(f"⏭️ SKIPPED document {doc_id}: Already processed (has parsed_content, chunks, and keywords)")
                return ProcessingResult(
                    document_id=doc_id,
                    success=True,
                    chunks_count=len(document.get('chunks', [])),
                    keywords_count=len(document.get('keywords', [])),
                    error_message="Already processed - skipped"
                )
            
            # Get PDF URL (check multiple possible field names)
            pdf_url = document.get('pdf_link') or document.get('url') or document.get('pdf_url')
            
            if not pdf_url:
                return ProcessingResult(
                    document_id=doc_id,
                    success=False,
                    chunks_count=0,
                    keywords_count=0,
                    error_message="No PDF URL found in document"
                )
            
            # Download PDF
            pdf_path = await downloader.download_pdf(pdf_url, doc_id)
            if not pdf_path:
                return ProcessingResult(
                    document_id=doc_id,
                    success=False,
                    chunks_count=0,
                    keywords_count=0,
                    error_message="Failed to download PDF"
                )
            
            # Parse PDF to markdown
            try:
                parsed_content = self.parsing_client.parse_single_pdf(pdf_path)
                if not parsed_content or len(parsed_content.strip()) < 100:
                    return ProcessingResult(
                        document_id=doc_id,
                        success=False,
                        chunks_count=0,
                        keywords_count=0,
                        error_message="PDF parsing failed or content too short"
                    )
            except Exception as e:
                return ProcessingResult(
                    document_id=doc_id,
                    success=False,
                    chunks_count=0,
                    keywords_count=0,
                    error_message=f"PDF parsing failed: {str(e)}"
                )
            
            # Create chunks
            text_chunks = self.text_splitter.split_text(parsed_content)
            
            if not text_chunks:
                return ProcessingResult(
                    document_id=doc_id,
                    success=False,
                    chunks_count=0,
                    keywords_count=0,
                    error_message="No chunks created from parsed content"
                )
            
            # Extract metadata
            company_name = document.get('company_name', '')
            type_field = document.get('type', '')
            sub_type = document.get('sub_type', '')
            title = document.get('title', '')
            
            # Process each chunk
            document_chunks = []
            all_keywords_set = set()
            
            char_position = 0
            for i, chunk_text in enumerate(text_chunks):
                # Find chunk position in original text
                chunk_start = parsed_content.find(chunk_text, char_position)
                if chunk_start == -1:
                    chunk_start = char_position
                chunk_end = chunk_start + len(chunk_text)
                
                # Extract keywords for this chunk
                chunk_keywords = await self.keyword_extractor.extract_keywords(chunk_text, max_keywords=15)
                all_keywords_set.update(chunk_keywords)
                
                # Create DocumentChunk
                chunk_id = self._generate_chunk_id(doc_id, i, chunk_text)
                chunk = DocumentChunk(
                    text=chunk_text.strip(),
                    source_file=title,
                    chunk_index=i,
                    start_char=chunk_start,
                    end_char=chunk_end,
                    chunk_id=chunk_id,
                    keywords=chunk_keywords,
                    company_name=company_name,
                    type=type_field,
                    sub_type=sub_type,
                    title=title,
                    pdf_link=pdf_url
                )
                
                document_chunks.append(chunk)
                char_position = chunk_end
            
            # Generate embeddings for chunks
            chunks_with_embeddings = await self.generate_embeddings(document_chunks)
            
            # Prepare data for MongoDB update
            chunks_data = []
            for chunk in chunks_with_embeddings:
                chunks_data.append({
                    'chunk_id': chunk.chunk_id,
                    'text': chunk.text,
                    'chunk_index': chunk.chunk_index,
                    'keywords': chunk.keywords,
                    'start_char': chunk.start_char,
                    'end_char': chunk.end_char
                })
            
            all_keywords = list(all_keywords_set)
            
            # Prepare MongoDB update data
            mongo_update_data = {
                'parsed_content': parsed_content,
                'chunks': chunks_data,
                'keywords': all_keywords,
                'processing_timestamp': time.time()
            }
            
            # IMMEDIATE UPLOADS - Run all uploads concurrently
            upload_tasks = []
            
            # 1. Update MongoDB immediately - pass the original ObjectId
            upload_tasks.append(
                self.update_mongo_immediately(original_doc_id, mongo_update_data)  # Pass ObjectId, not string
            )
            
            # 2. Upload chunks to Qdrant immediately
            if chunks_with_embeddings:
                upload_tasks.append(
                    self.upload_chunks_to_qdrant_immediately(chunks_with_embeddings)
                )
            
            # 3. Generate and upload keyword embeddings immediately
            if all_keywords:
                keyword_embeddings = await self.generate_keyword_embeddings(all_keywords, document)
                if keyword_embeddings:
                    upload_tasks.append(
                        self.upload_keywords_to_qdrant_immediately(keyword_embeddings)
                    )
            
            # Execute all uploads concurrently
            if upload_tasks:
                await asyncio.gather(*upload_tasks, return_exceptions=True)
            
            processing_time = time.time() - start_time
            
            # Update stats thread-safely
            with self.stats_lock:
                self.stats['processed_documents'] += 1
                self.stats['total_chunks'] += len(chunks_with_embeddings)
                self.stats['total_keywords'] += len(all_keywords)
            
            logger.info(f"✅ COMPLETED document {doc_id}: {len(chunks_with_embeddings)} chunks, {len(all_keywords)} keywords, uploaded to MongoDB + Qdrant")
            
            return ProcessingResult(
                document_id=doc_id,
                success=True,
                chunks_count=len(chunks_with_embeddings),
                keywords_count=len(all_keywords),
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Update stats thread-safely
            with self.stats_lock:
                self.stats['failed_documents'] += 1
            
            logger.error(f"❌ FAILED document {doc_id}: {e}")
            return ProcessingResult(
                document_id=doc_id,
                success=False,
                chunks_count=0,
                keywords_count=0,
                error_message=str(e),
                processing_time=processing_time
            )
    
    def _generate_chunk_id(self, doc_id: str, chunk_index: int, chunk_text: str) -> str:
        """Generate a unique ID for a chunk."""
        content_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:8]
        return f"{doc_id}_{chunk_index:04d}_{content_hash}"
    
    async def generate_embeddings(self, chunks: List[DocumentChunk], batch_size: int = 50) -> List[DocumentChunk]:
        """Generate embeddings for chunks in batches."""
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        
        processed_chunks = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            try:
                # Prepare texts for embedding
                texts = [chunk.text for chunk in batch]
                
                # Generate embeddings
                response = await self.openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=texts,
                    dimensions=self.embedding_dimensions,
                    encoding_format="float"
                )
                
                # Assign embeddings to chunks
                for chunk, embedding_data in zip(batch, response.data):
                    chunk.embedding = embedding_data.embedding
                    processed_chunks.append(chunk)
                
                # Track token usage
                for text in texts:
                    self.token_tracker.track_openai_embedding(text, self.embedding_model)
                
                # Small delay to respect rate limits
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error generating embeddings for batch: {e}")
                # Add chunks without embeddings to avoid losing data
                for chunk in batch:
                    chunk.embedding = None
                    processed_chunks.append(chunk)
        
        valid_chunks = [chunk for chunk in processed_chunks if chunk.embedding is not None]
        logger.info(f"Generated embeddings for {len(valid_chunks)}/{len(chunks)} chunks")
        return valid_chunks
    
    async def generate_keyword_embeddings(self, keywords: List[str], document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate embeddings for keywords."""
        try:
            # Create keyword phrases for better semantic representation
            keyword_phrases = [f"insurance {keyword}" for keyword in keywords]
            
            response = await self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=keyword_phrases,
                dimensions=self.embedding_dimensions,
                encoding_format="float"
            )
            
            keyword_data = []
            for keyword, embedding_data in zip(keywords, response.data):
                keyword_data.append({
                    'keyword': keyword,
                    'embedding': embedding_data.embedding,
                    'company_name': document.get('company_name', ''),
                    'type': document.get('type', ''),
                    'sub_type': document.get('sub_type', ''),
                    'title': document.get('title', ''),
                    'pdf_link': document.get('pdf_link') or document.get('url') or document.get('pdf_url', '')
                })
            
            return keyword_data
            
        except Exception as e:
            logger.error(f"Error generating keyword embeddings: {e}")
            return []
    
    async def run_pipeline(self, limit: int = None):
        """Run the complete RAG ingestion pipeline with immediate uploads."""
        start_time = time.time()
        logger.info("🚀 Starting MongoDB RAG ingestion pipeline with IMMEDIATE UPLOADS...")
        
        try:
            # Setup Qdrant collections
            self.setup_qdrant_collections()
            
            # Get documents to process
            documents = self.get_documents_to_process(limit)
            if not documents:
                logger.info("No documents need processing")
                return
            
            self.stats['total_documents'] = len(documents)
            logger.info(f"📄 Processing {len(documents)} documents with {self.max_workers} workers")
            
            # Process documents concurrently with immediate uploads
            async with PDFDownloader() as downloader:
                tasks = []
                semaphore = asyncio.Semaphore(self.max_workers)
                
                async def process_with_semaphore(doc):
                    async with semaphore:
                        return await self.process_single_document(doc, downloader)
                
                # Create tasks for all documents
                for document in documents:
                    task = asyncio.create_task(process_with_semaphore(document))
                    tasks.append(task)
                
                # Process documents and collect results
                results = []
                completed_count = 0
                
                for task in asyncio.as_completed(tasks):
                    result = await task
                    results.append(result)
                    completed_count += 1
                    
                    # Log progress with real-time stats
                    if completed_count % 5 == 0 or completed_count == len(tasks):
                        success_rate = (self.stats['processed_documents'] / completed_count * 100) if completed_count > 0 else 0
                        logger.info(f"📊 Progress: {completed_count}/{len(tasks)} | ✅ {self.stats['processed_documents']} success | ❌ {self.stats['failed_documents']} failed | 📈 {success_rate:.1f}% success rate")
                
                # Cleanup temporary files
                downloader.cleanup()
            
            # Update final statistics
            self.stats['processing_time'] = time.time() - start_time
            
            # Print final summary
            self.print_final_summary()
            
            logger.info("🎉 MongoDB RAG ingestion pipeline completed successfully with IMMEDIATE UPLOADS!")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def print_final_summary(self):
        """Print comprehensive summary of the ingestion process."""
        print("\n" + "="*80)
        print("🚀 MONGODB RAG INGESTION PIPELINE SUMMARY (IMMEDIATE UPLOADS)")
        print("="*80)
        
        print(f"📄 Total Documents: {self.stats['total_documents']}")
        print(f"✅ Successfully Processed: {self.stats['processed_documents']}")
        print(f"❌ Failed: {self.stats['failed_documents']}")
        success_rate = (self.stats['processed_documents']/max(self.stats['total_documents'], 1)*100)
        print(f"📈 Success Rate: {success_rate:.1f}%")
        
        print(f"\n📊 Data Generated & Uploaded:")
        print(f"  🔗 Total Chunks: {self.stats['total_chunks']:,}")
        print(f"  🏷️ Total Keywords: {self.stats['total_keywords']:,}")
        avg_chunks = self.stats['total_chunks']/max(self.stats['processed_documents'], 1)
        print(f"  📋 Avg Chunks per Document: {avg_chunks:.1f}")
        
        print(f"\n⚙️ Configuration:")
        print(f"  👥 Max Workers: {self.max_workers}")
        print(f"  📏 Chunk Size: {self.chunk_size}")
        print(f"  🔄 Chunk Overlap: {self.chunk_overlap}")
        print(f"  🤖 Embedding Model: {self.embedding_model}")
        print(f"  📐 Embedding Dimensions: {self.embedding_dimensions}")
        
        print(f"\n⚡ Performance:")
        print(f"  ⏱️ Total Processing Time: {self.stats['processing_time']:.2f} seconds")
        avg_time = self.stats['processing_time']/max(self.stats['processed_documents'], 1)
        print(f"  ⏰ Avg Time per Document: {avg_time:.2f} seconds")
        
        # Token usage summary
        if hasattr(self.token_tracker, 'print_summary'):
            print(f"\n💰 Token Usage:")
            self.token_tracker.print_summary()
        
        print("\n🎯 Upload Strategy: IMMEDIATE - Each document uploaded to MongoDB + Qdrant as soon as processing completes")
        print("="*80)

async def main():
    """Main function to run the MongoDB RAG ingestion pipeline."""
    try:
        # Configuration
        max_workers = int(os.getenv("MAX_WORKERS", "8"))
        limit = int(os.getenv("DOCUMENT_LIMIT", "0")) or None
        
        logger.info(f"🔧 Configuration: {max_workers} workers, limit: {limit or 'no limit'}")
        
        pipeline = MongoRAGIngestionPipeline(max_workers=max_workers)
        await pipeline.run_pipeline(limit=limit)
        
    except Exception as e:
        logger.error(f"Main pipeline execution failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())