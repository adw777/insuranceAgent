# import os
# from dotenv import load_dotenv
# from typing import List, Dict, Optional
# import logging  
# from qdrant_client import QdrantClient
# from qdrant_client.models import Distance, VectorParams, PointStruct, ScoredPoint
# from qdrant_client.http import models

# logger = logging.getLogger(__name__)

# load_dotenv()

# def get_qdrant_client() -> QdrantClient:
#     """Get Qdrant client instance using environment variables."""
#     qdrant_url = os.getenv("QDRANT_URL")
#     qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
#     if not qdrant_url:
#         raise ValueError("QDRANT_URL not found in environment variables")
    
#     try:
#         # Try cloud/remote Qdrant first
#         client = QdrantClient(
#             url=qdrant_url,
#             api_key=qdrant_api_key if qdrant_api_key else None
#         )
#         # Test connection
#         client.get_collections()
#         logger.info("Connected to remote Qdrant instance")
#         return client
#     except Exception as e:
#         logger.error(f"Could not connect to remote Qdrant: {e}")
#         raise

# def create_collection(client: QdrantClient, collection_name: str, vector_size: int = 1024) -> bool:
#     """Create a Qdrant collection with optimized configuration."""
#     try:
#         client.create_collection(
#             collection_name=collection_name,
#             vectors_config=models.VectorParams(
#                 size=vector_size, 
#                 distance=Distance.COSINE
#             ),
#             shard_number=1,
#             sharding_method=models.ShardingMethod.AUTO,
#             hnsw_config=models.HnswConfigDiff(
#                 m=16,               # Reduced from 24 for better balance
#                 ef_construct=200,   # Adjusted for better build time
#                 full_scan_threshold=10000,  # Increased for better performance
#                 on_disk=True
#             ),
#             optimizers_config=models.OptimizersConfigDiff(
#                 deleted_threshold=0.2,  # Trigger optimization when 20% points are deleted
#                 vacuum_min_vector_number=1000,  # Minimum vectors to start vacuum
#                 default_segment_number=2  # Number of segments to store points
#             ),
#             on_disk_payload=True,  # Changed to True for better memory usage
#             timeout=60  # Increased timeout for larger collections
#         )
#         logger.info(f"Collection {collection_name} created successfully.")
#         return True
#     except Exception as e:
#         logger.error(f"Error creating collection {collection_name}: {e}")
#         return False

# def insert_points_to_collection(client: QdrantClient, collection_name: str, points: List[PointStruct], batch_size: int = 1000) -> bool:
#     """Insert points into a Qdrant collection in batches."""
#     try:
#         logger.info(f"Inserting {len(points)} points to collection {collection_name}")
        
#         # Insert in batches for better performance
#         for i in range(0, len(points), batch_size):
#             batch = points[i:i + batch_size]
#             logger.info(f"Inserting batch {i//batch_size + 1}/{(len(points) + batch_size - 1)//batch_size}")
            
#             client.upsert(
#                 collection_name=collection_name,
#                 points=batch
#             )
        
#         logger.info(f"Successfully inserted {len(points)} points to {collection_name}")
#         return True
        
#     except Exception as e:
#         logger.error(f"Error inserting points into collection {collection_name}: {e}")
#         return False

# def search_points_in_collection(
#     client: QdrantClient, 
#     collection_name: str, 
#     query_vector: List[float], 
#     limit: int = 10,
#     score_threshold: Optional[float] = None,
#     with_payload: bool = True
# ) -> List[ScoredPoint]:
#     """Search for similar points in a Qdrant collection."""
#     try:
#         logger.info(f"Searching for top {limit} similar points in {collection_name}")
        
#         search_params = {
#             "collection_name": collection_name,
#             "query_vector": query_vector,
#             "limit": limit,
#             "with_payload": with_payload
#         }
        
#         if score_threshold is not None:
#             search_params["score_threshold"] = score_threshold
        
#         results = client.search(**search_params)
        
#         logger.info(f"Found {len(results)} similar points")
#         return results
        
#     except Exception as e:
#         logger.error(f"Error searching points in collection {collection_name}: {e}")
#         return []

# def get_collection_info(client: QdrantClient, collection_name: str) -> Optional[Dict]:
#     """Get information about a collection."""
#     try:
#         collection_info = client.get_collection(collection_name)
        
#         info = {
#             "collection_name": collection_name,
#             "total_points": collection_info.points_count,
#             "vector_size": collection_info.config.params.vectors.size,
#             "distance_metric": collection_info.config.params.vectors.distance
#         }
        
#         logger.info(f"Collection info: {info}")
#         return info
        
#     except Exception as e:
#         logger.error(f"Error getting collection info for {collection_name}: {e}")
#         return None

# def collection_exists(client: QdrantClient, collection_name: str) -> bool:
#     """Check if a collection exists."""
#     try:
#         collections = client.get_collections().collections
#         collection_names = [col.name for col in collections]
#         return collection_name in collection_names
#     except Exception as e:
#         logger.error(f"Error checking if collection {collection_name} exists: {e}")
#         return False

# def delete_collection(client: QdrantClient, collection_name: str) -> bool:
#     """Delete a collection."""
#     try:
#         client.delete_collection(collection_name)
#         logger.info(f"Collection {collection_name} deleted successfully")
#         return True
#     except Exception as e:
#         logger.error(f"Error deleting collection {collection_name}: {e}")
#         return False

# # Legacy function name for backward compatibility
# def retrieve_points_from_collection(client: QdrantClient, collection_name: str, query_vector: List[float], limit: int = 10):
#     """Legacy function - use search_points_in_collection instead."""
#     logger.warning("retrieve_points_from_collection is deprecated, use search_points_in_collection instead")
#     return search_points_in_collection(client, collection_name, query_vector, limit)


import os
from dotenv import load_dotenv
from typing import List, Dict, Optional, Union, Any
import logging  
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, ScoredPoint, Record
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

def query_points_with_filter(
    client: QdrantClient,
    collection_name: str,
    query_vector: List[float],
    filters: Dict[str, Any],
    limit: int = 10,
    score_threshold: Optional[float] = None,
    hnsw_ef: int = 128,
    exact: bool = False,
    with_payload: bool = True,
    with_vectors: bool = False
) -> List[ScoredPoint]:
    """
    Query points in a collection with vector similarity and payload filters.
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection to query
        query_vector: Vector to search for similar points
        filters: Dictionary of field conditions {key: value} or {key: [value1, value2]} for multiple values
        limit: Maximum number of points to return
        score_threshold: Minimum similarity score threshold
        hnsw_ef: Size of the dynamic list for HNSW search (higher = more accurate but slower)
        exact: Whether to use exact search (slower but more accurate)
        with_payload: Whether to include payload in results
        with_vectors: Whether to include vectors in results
    
    Returns:
        List of ScoredPoint objects matching the criteria
    
    Example:
        # Single filter
        results = query_points_with_filter(client, "my_collection", [0.1, 0.2], {"city": "London"})
        
        # Multiple filters
        results = query_points_with_filter(client, "my_collection", [0.1, 0.2], {
            "city": "London", 
            "color": "red"
        })
    """
    try:
        logger.info(f"Querying points in {collection_name} with vector similarity and filters: {filters}")
        
        # Build filter conditions
        must_conditions = []
        
        for key, value in filters.items():
            if isinstance(value, list):
                # Handle multiple values for the same field (OR condition within the field)
                should_conditions = []
                for val in value:
                    should_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=val)
                        )
                    )
                # Wrap multiple values in a should condition
                must_conditions.append(
                    models.Filter(should=should_conditions)
                )
            else:
                # Single value condition
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )
        
        # Create the main filter
        query_filter = models.Filter(must=must_conditions)
        
        # Set up query parameters
        query_params = {
            "collection_name": collection_name,
            "query": query_vector,
            "query_filter": query_filter,
            "search_params": models.SearchParams(hnsw_ef=hnsw_ef, exact=exact),
            "limit": limit,
            "with_payload": with_payload,
            "with_vectors": with_vectors
        }
        
        if score_threshold is not None:
            query_params["score_threshold"] = score_threshold
        
        # Perform query operation
        results = client.query_points(**query_params)
        
        logger.info(f"Found {len(results.points)} points matching query and filters")
        return results.points
        
    except Exception as e:
        logger.error(f"Error querying points with filter in collection {collection_name}: {e}")
        return []

def create_payload_index(
    client: QdrantClient,
    collection_name: str,
    field_name: str,
    field_schema: str = "keyword"
) -> bool:
    """
    Create a payload index for a specific field to enable filtering.
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection
        field_name: Name of the field to index
        field_schema: Type of index - "keyword", "integer", "float", "bool", etc.
    
    Returns:
        True if index was created successfully, False otherwise
    """
    try:
        logger.info(f"Creating {field_schema} index for field '{field_name}' in collection '{collection_name}'")
        
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_schema
        )
        
        logger.info(f"Successfully created {field_schema} index for field '{field_name}'")
        return True
        
    except Exception as e:
        logger.error(f"Error creating index for field '{field_name}': {e}")
        return False

def scroll_filtered_points(
    client: QdrantClient,
    collection_name: str,
    filters: Dict[str, Any],
    limit: int = 100,
    offset: Optional[str] = None,
    with_payload: bool = True,
    with_vectors: bool = False
) -> List[Record]:
    """
    Scroll through points in a collection with filters.
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection to scroll
        filters: Dictionary of field conditions {key: value} or {key: [value1, value2]} for multiple values
        limit: Maximum number of points to return per scroll
        offset: Pagination offset (point ID to start from)
        with_payload: Whether to include payload in results
        with_vectors: Whether to include vectors in results
    
    Returns:
        List of Record objects matching the filter criteria
    
    Example:
        # Single filter
        results = scroll_filtered_points(client, "my_collection", {"city": "London"})
        
        # Multiple filters
        results = scroll_filtered_points(client, "my_collection", {
            "city": "London", 
            "color": "red"
        })
        
        # Multiple values for one field
        results = scroll_filtered_points(client, "my_collection", {
            "city": ["London", "Paris"]
        })
    """
    try:
        logger.info(f"Scrolling filtered points in {collection_name} with filters: {filters}")
        
        # Build filter conditions
        must_conditions = []
        
        for key, value in filters.items():
            if isinstance(value, list):
                # Handle multiple values for the same field (OR condition within the field)
                should_conditions = []
                for val in value:
                    should_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=val)
                        )
                    )
                # Wrap multiple values in a should condition
                must_conditions.append(
                    models.Filter(should=should_conditions)
                )
            else:
                # Single value condition
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )
        
        # Create the main filter
        scroll_filter = models.Filter(must=must_conditions)
        
        # Perform scroll operation
        scroll_params = {
            "collection_name": collection_name,
            "scroll_filter": scroll_filter,
            "limit": limit,
            "with_payload": with_payload,
            "with_vectors": with_vectors
        }
        
        if offset:
            scroll_params["offset"] = offset
        
        result = client.scroll(**scroll_params)
        
        # Extract points from the scroll result
        points = result[0] if result else []
        
        logger.info(f"Found {len(points)} filtered points")
        return points
        
    except Exception as e:
        logger.error(f"Error scrolling filtered points in collection {collection_name}: {e}")
        return []

def scroll_all_filtered_points(
    client: QdrantClient,
    collection_name: str,
    filters: Dict[str, Any],
    batch_size: int = 1000,
    with_payload: bool = True,
    with_vectors: bool = False
) -> List[Record]:
    """
    Scroll through ALL points in a collection that match the filters.
    This function handles pagination automatically.
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection to scroll
        filters: Dictionary of field conditions {key: value}
        batch_size: Number of points to fetch per batch
        with_payload: Whether to include payload in results
        with_vectors: Whether to include vectors in results
    
    Returns:
        List of all Record objects matching the filter criteria
    """
    try:
        logger.info(f"Scrolling ALL filtered points in {collection_name} with filters: {filters}")
        
        all_points = []
        offset = None
        
        while True:
            # Get a batch of points
            batch_points = scroll_filtered_points(
                client=client,
                collection_name=collection_name,
                filters=filters,
                limit=batch_size,
                offset=offset,
                with_payload=with_payload,
                with_vectors=with_vectors
            )
            
            if not batch_points:
                # No more points to fetch
                break
                
            all_points.extend(batch_points)
            
            # Get the last point ID for next iteration
            if len(batch_points) < batch_size:
                # This was the last batch
                break
            else:
                # Set offset to the last point ID for pagination
                offset = batch_points[-1].id
        
        logger.info(f"Retrieved total of {len(all_points)} filtered points")
        return all_points
        
    except Exception as e:
        logger.error(f"Error scrolling all filtered points in collection {collection_name}: {e}")
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