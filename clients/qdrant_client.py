import os
from dotenv import load_dotenv
from typing import List, Dict, Optional
import logging  
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, ScoredPoint
from qdrant_client.http import models

logger = logging.getLogger(__name__)

load_dotenv()

def get_qdrant_client() -> QdrantClient:
    """Get Qdrant client instance using environment variables."""
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if not qdrant_url:
        raise ValueError("QDRANT_URL not found in environment variables")
    
    try:
        # Try cloud/remote Qdrant first
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key if qdrant_api_key else None
        )
        # Test connection
        client.get_collections()
        logger.info("Connected to remote Qdrant instance")
        return client
    except Exception as e:
        logger.error(f"Could not connect to remote Qdrant: {e}")
        raise

def create_collection(client: QdrantClient, collection_name: str, vector_size: int = 1024) -> bool:
    """Create a Qdrant collection with optimized configuration."""
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size, 
                distance=Distance.COSINE
            ),
            shard_number=1,
            sharding_method=models.ShardingMethod.AUTO,
            hnsw_config=models.HnswConfigDiff(
                m=16,               # Reduced from 24 for better balance
                ef_construct=200,   # Adjusted for better build time
                full_scan_threshold=10000,  # Increased for better performance
                on_disk=True
            ),
            optimizers_config=models.OptimizersConfigDiff(
                deleted_threshold=0.2,  # Trigger optimization when 20% points are deleted
                vacuum_min_vector_number=1000,  # Minimum vectors to start vacuum
                default_segment_number=2  # Number of segments to store points
            ),
            on_disk_payload=True,  # Changed to True for better memory usage
            timeout=60  # Increased timeout for larger collections
        )
        logger.info(f"Collection {collection_name} created successfully.")
        return True
    except Exception as e:
        logger.error(f"Error creating collection {collection_name}: {e}")
        return False

def insert_points_to_collection(client: QdrantClient, collection_name: str, points: List[PointStruct], batch_size: int = 1000) -> bool:
    """Insert points into a Qdrant collection in batches."""
    try:
        logger.info(f"Inserting {len(points)} points to collection {collection_name}")
        
        # Insert in batches for better performance
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            logger.info(f"Inserting batch {i//batch_size + 1}/{(len(points) + batch_size - 1)//batch_size}")
            
            client.upsert(
                collection_name=collection_name,
                points=batch
            )
        
        logger.info(f"Successfully inserted {len(points)} points to {collection_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error inserting points into collection {collection_name}: {e}")
        return False

def search_points_in_collection(
    client: QdrantClient, 
    collection_name: str, 
    query_vector: List[float], 
    limit: int = 10,
    score_threshold: Optional[float] = None,
    with_payload: bool = True
) -> List[ScoredPoint]:
    """Search for similar points in a Qdrant collection."""
    try:
        logger.info(f"Searching for top {limit} similar points in {collection_name}")
        
        search_params = {
            "collection_name": collection_name,
            "query_vector": query_vector,
            "limit": limit,
            "with_payload": with_payload
        }
        
        if score_threshold is not None:
            search_params["score_threshold"] = score_threshold
        
        results = client.search(**search_params)
        
        logger.info(f"Found {len(results)} similar points")
        return results
        
    except Exception as e:
        logger.error(f"Error searching points in collection {collection_name}: {e}")
        return []

def get_collection_info(client: QdrantClient, collection_name: str) -> Optional[Dict]:
    """Get information about a collection."""
    try:
        collection_info = client.get_collection(collection_name)
        
        info = {
            "collection_name": collection_name,
            "total_points": collection_info.points_count,
            "vector_size": collection_info.config.params.vectors.size,
            "distance_metric": collection_info.config.params.vectors.distance
        }
        
        logger.info(f"Collection info: {info}")
        return info
        
    except Exception as e:
        logger.error(f"Error getting collection info for {collection_name}: {e}")
        return None

def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    """Check if a collection exists."""
    try:
        collections = client.get_collections().collections
        collection_names = [col.name for col in collections]
        return collection_name in collection_names
    except Exception as e:
        logger.error(f"Error checking if collection {collection_name} exists: {e}")
        return False

def delete_collection(client: QdrantClient, collection_name: str) -> bool:
    """Delete a collection."""
    try:
        client.delete_collection(collection_name)
        logger.info(f"Collection {collection_name} deleted successfully")
        return True
    except Exception as e:
        logger.error(f"Error deleting collection {collection_name}: {e}")
        return False

# Legacy function name for backward compatibility
def retrieve_points_from_collection(client: QdrantClient, collection_name: str, query_vector: List[float], limit: int = 10):
    """Legacy function - use search_points_in_collection instead."""
    logger.warning("retrieve_points_from_collection is deprecated, use search_points_in_collection instead")
    return search_points_in_collection(client, collection_name, query_vector, limit)