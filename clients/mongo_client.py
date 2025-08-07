import os
import logging
from typing import Optional, Dict, Any, List
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class MongoDBClient:
    """MongoDB client wrapper for database operations."""
    
    def __init__(self):
        self.mongo_uri = os.getenv("MONGO_URI")
        if not self.mongo_uri:
            raise ValueError("MONGO_URI not found in environment variables")
        
        self.client: Optional[MongoClient] = None
        self.database: Optional[Database] = None
        self._connect()
    
    def _connect(self):
        """Initialize MongoDB connection."""
        try:
            self.client = MongoClient(
                self.mongo_uri,
                serverSelectionTimeoutMS=5000,  # 5 second timeout
                connectTimeoutMS=5000,
                socketTimeoutMS=5000,
                maxPoolSize=50,
                minPoolSize=5
            )
            
            # Test connection
            self.client.admin.command('ping')
            logger.info("Successfully connected to MongoDB")
            
            # Get database name from URI or use default
            db_name = os.getenv("MONGO_DATABASE", "insurance")
            self.database = self.client[db_name]
            logger.info(f"Using database: {db_name}")
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {e}")
            raise
    
    def get_collection(self, collection_name: str) -> Collection:
        """Get a collection from the database."""
        if self.database is None:
            raise RuntimeError("Database not initialized")
        return self.database[collection_name]
    
    def insert_document(self, collection_name: str, document: Dict[str, Any]) -> str:
        """Insert a single document into a collection."""
        try:
            collection = self.get_collection(collection_name)
            result = collection.insert_one(document)
            logger.info(f"Inserted document with ID: {result.inserted_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error inserting document into {collection_name}: {e}")
            raise
    
    def insert_documents(self, collection_name: str, documents: List[Dict[str, Any]]) -> List[str]:
        """Insert multiple documents into a collection."""
        try:
            collection = self.get_collection(collection_name)
            result = collection.insert_many(documents)
            logger.info(f"Inserted {len(result.inserted_ids)} documents")
            return [str(doc_id) for doc_id in result.inserted_ids]
        except Exception as e:
            logger.error(f"Error inserting documents into {collection_name}: {e}")
            raise
    
    def find_documents(self, collection_name: str, query: Dict[str, Any] = None, limit: int = 0) -> List[Dict[str, Any]]:
        """Find documents in a collection."""
        try:
            collection = self.get_collection(collection_name)
            query = query or {}
            
            if limit > 0:
                cursor = collection.find(query).limit(limit)
            else:
                cursor = collection.find(query)
            
            documents = list(cursor)
            logger.info(f"Found {len(documents)} documents in {collection_name}")
            return documents
        except Exception as e:
            logger.error(f"Error finding documents in {collection_name}: {e}")
            raise
    
    def find_one_document(self, collection_name: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find a single document in a collection."""
        try:
            collection = self.get_collection(collection_name)
            document = collection.find_one(query)
            if document:
                logger.info(f"Found document in {collection_name}")
            else:
                logger.info(f"No document found in {collection_name} with query: {query}")
            return document
        except Exception as e:
            logger.error(f"Error finding document in {collection_name}: {e}")
            raise
    
    def update_document(self, collection_name: str, query: Dict[str, Any], update: Dict[str, Any]) -> int:
        """Update a single document in a collection."""
        try:
            collection = self.get_collection(collection_name)
            result = collection.update_one(query, {"$set": update})
            logger.info(f"Updated {result.modified_count} document(s) in {collection_name}")
            return result.modified_count
        except Exception as e:
            logger.error(f"Error updating document in {collection_name}: {e}")
            raise
    
    def update_documents(self, collection_name: str, query: Dict[str, Any], update: Dict[str, Any]) -> int:
        """Update multiple documents in a collection."""
        try:
            collection = self.get_collection(collection_name)
            result = collection.update_many(query, {"$set": update})
            logger.info(f"Updated {result.modified_count} document(s) in {collection_name}")
            return result.modified_count
        except Exception as e:
            logger.error(f"Error updating documents in {collection_name}: {e}")
            raise
    
    def delete_document(self, collection_name: str, query: Dict[str, Any]) -> int:
        """Delete a single document from a collection."""
        try:
            collection = self.get_collection(collection_name)
            result = collection.delete_one(query)
            logger.info(f"Deleted {result.deleted_count} document(s) from {collection_name}")
            return result.deleted_count
        except Exception as e:
            logger.error(f"Error deleting document from {collection_name}: {e}")
            raise
    
    def delete_documents(self, collection_name: str, query: Dict[str, Any]) -> int:
        """Delete multiple documents from a collection."""
        try:
            collection = self.get_collection(collection_name)
            result = collection.delete_many(query)
            logger.info(f"Deleted {result.deleted_count} document(s) from {collection_name}")
            return result.deleted_count
        except Exception as e:
            logger.error(f"Error deleting documents from {collection_name}: {e}")
            raise
    
    def count_documents(self, collection_name: str, query: Dict[str, Any] = None) -> int:
        """Count documents in a collection."""
        try:
            collection = self.get_collection(collection_name)
            query = query or {}
            count = collection.count_documents(query)
            logger.info(f"Found {count} documents in {collection_name}")
            return count
        except Exception as e:
            logger.error(f"Error counting documents in {collection_name}: {e}")
            raise
    
    def create_index(self, collection_name: str, index_spec: List[tuple], unique: bool = False) -> str:
        """Create an index on a collection."""
        try:
            collection = self.get_collection(collection_name)
            index_name = collection.create_index(index_spec, unique=unique)
            logger.info(f"Created index '{index_name}' on {collection_name}")
            return index_name
        except Exception as e:
            logger.error(f"Error creating index on {collection_name}: {e}")
            raise
    
    def drop_collection(self, collection_name: str):
        """Drop a collection."""
        try:
            collection = self.get_collection(collection_name)
            collection.drop()
            logger.info(f"Dropped collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error dropping collection {collection_name}: {e}")
            raise
    
    def list_collections(self) -> List[str]:
        """List all collections in the database."""
        try:
            if self.database is None:
                raise RuntimeError("Database not initialized")
            collections = self.database.list_collection_names()
            logger.info(f"Found {len(collections)} collections")
            return collections
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            raise
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            if self.database is None:
                raise RuntimeError("Database not initialized")
            stats = self.database.command("dbstats")
            logger.info("Retrieved database statistics")
            return stats
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            raise
    
    def close(self):
        """Close the MongoDB connection."""
        if self.client is not None:
            self.client.close()
            logger.info("MongoDB connection closed")

# Convenience function to get a MongoDB client instance
def get_mongo_client() -> MongoDBClient:
    """Get a MongoDB client instance."""
    return MongoDBClient()

# Example usage and testing
if __name__ == "__main__":
    try:
        # Test the MongoDB connection
        mongo_client = get_mongo_client()
        
        # Test basic operations
        print("Testing MongoDB connection...")
        
        # List collections
        collections = mongo_client.list_collections()
        print(f"Collections: {collections}")
        
        # Get database stats
        stats = mongo_client.get_database_stats()
        print(f"Database size: {stats.get('dataSize', 0)} bytes")
        
        # Test document operations
        test_collection = "test_collection"
        
        # Insert a test document
        test_doc = {"test": "data", "timestamp": "2024-01-01"}
        doc_id = mongo_client.insert_document(test_collection, test_doc)
        print(f"Inserted test document with ID: {doc_id}")
        
        # Find the document
        found_doc = mongo_client.find_one_document(test_collection, {"test": "data"})
        print(f"Found document: {found_doc}")
        
        # Count documents
        count = mongo_client.count_documents(test_collection)
        print(f"Document count: {count}")
        
        # Clean up test document
        deleted_count = mongo_client.delete_document(test_collection, {"test": "data"})
        print(f"Deleted {deleted_count} test document(s)")
        
        print("MongoDB client test completed successfully!")
        
    except Exception as e:
        print(f"Error testing MongoDB client: {e}")
    finally:
        if 'mongo_client' in locals():
            mongo_client.close()