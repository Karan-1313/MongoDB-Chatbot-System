"""MongoDB vector store implementation for document storage and retrieval."""

import hashlib
import logging
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from pymongo.collection import Collection
from pymongo.errors import PyMongoError, ConnectionFailure, ServerSelectionTimeoutError

from ..database.connection import get_mongodb_connection
from ..core.config import get_settings
from ..core.exceptions import (
    VectorStoreError,
    DatabaseError,
    TimeoutError,
    handle_external_api_error
)
from ..core.retry import retry_mongodb
from ..core.monitoring import DatabaseMonitor

logger = logging.getLogger(__name__)


class MongoVectorStore:
    """MongoDB vector store for document storage and similarity search."""
    
    def __init__(self, collection_name: Optional[str] = None):
        """Initialize MongoDB vector store.
        
        Args:
            collection_name: Name of the MongoDB collection to use
        """
        self.connection = get_mongodb_connection()
        self.settings = get_settings()
        self.collection_name = collection_name or self.settings.mongodb_collection
        self._collection: Optional[Collection] = None
    
    @property
    def collection(self) -> Collection:
        """Get MongoDB collection instance."""
        if self._collection is None:
            self._collection = self.connection.get_collection(self.collection_name)
        return self._collection
    
    @retry_mongodb
    def add_documents(self, documents: List[Document], embeddings: List[List[float]], 
                     skip_duplicates: bool = True) -> List[str]:
        """Add documents with embeddings to the vector store.
        
        Args:
            documents: List of LangChain Document objects
            embeddings: List of embedding vectors corresponding to documents
            skip_duplicates: Whether to skip duplicate documents
            
        Returns:
            List of inserted document IDs
        """
        with DatabaseMonitor("add_documents", self.collection_name):
            if len(documents) != len(embeddings):
                raise VectorStoreError(
                    "Number of documents must match number of embeddings",
                    operation="add_documents",
                    collection=self.collection_name
                )
            
            try:
                # Prepare documents for insertion
                docs_to_insert = []
                skipped_count = 0
                
                for doc, embedding in zip(documents, embeddings):
                    # Check for duplicates if requested
                    if skip_duplicates and self._is_duplicate(doc):
                        skipped_count += 1
                        logger.debug(f"Skipping duplicate document: {doc.metadata.get('source', 'unknown')}")
                        continue
                    
                    doc_dict = {
                        "content": doc.page_content,
                        "embedding": embedding,
                        "metadata": doc.metadata,
                        "text_length": len(doc.page_content),
                        "chunk_index": doc.metadata.get("chunk_id", 0),
                        "content_hash": self._generate_content_hash(doc.page_content)
                    }
                    docs_to_insert.append(doc_dict)
                
                if not docs_to_insert:
                    logger.info("No new documents to insert (all were duplicates)")
                    return []
                
                # Insert documents
                result = self.collection.insert_many(docs_to_insert)
                inserted_ids = [str(id_) for id_ in result.inserted_ids]
                
                logger.info(f"Added {len(inserted_ids)} documents to vector store (skipped {skipped_count} duplicates)")
                return inserted_ids
                
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                raise TimeoutError(
                    f"MongoDB connection timeout during document insertion: {e}",
                    operation="add_documents"
                )
            except PyMongoError as e:
                logger.error(f"Failed to add documents to vector store: {e}")
                raise VectorStoreError(
                    f"Failed to add documents: {e}",
                    operation="add_documents",
                    collection=self.collection_name
                )
    
    @retry_mongodb
    def similarity_search(self, query_embedding: List[float], k: int = 5, 
                         filter_dict: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Perform similarity search using vector embeddings.
        
        Args:
            query_embedding: Query vector embedding
            k: Number of similar documents to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of similar Document objects
        """
        with DatabaseMonitor("similarity_search", self.collection_name):
            try:
                # Build aggregation pipeline for vector search
                pipeline = []
                
                # Vector search stage (Atlas Vector Search)
                vector_search_stage = {
                    "$vectorSearch": {
                        "index": self.settings.vector_index_name,
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": k * 10,  # Search more candidates for better results
                        "limit": k
                    }
                }
                
                # Add filter if provided
                if filter_dict:
                    vector_search_stage["$vectorSearch"]["filter"] = filter_dict
                
                pipeline.append(vector_search_stage)
                
                # Add score projection
                pipeline.append({
                    "$project": {
                        "content": 1,
                        "metadata": 1,
                        "text_length": 1,
                        "chunk_index": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                })
                
                # Execute aggregation pipeline
                results = list(self.collection.aggregate(pipeline))
                
                # Convert results to LangChain Document objects
                documents = []
                for result in results:
                    # Add score to metadata
                    metadata = result.get("metadata", {})
                    metadata["score"] = result.get("score", 0.0)
                    metadata["_id"] = str(result["_id"])
                    
                    doc = Document(
                        page_content=result["content"],
                        metadata=metadata
                    )
                    documents.append(doc)
                
                logger.debug(f"Found {len(documents)} similar documents")
                return documents
                
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                raise TimeoutError(
                    f"MongoDB connection timeout during vector search: {e}",
                    operation="similarity_search"
                )
            except PyMongoError as e:
                logger.error(f"Vector search failed: {e}")
                # Try fallback search
                try:
                    return self._fallback_text_search(query_embedding, k, filter_dict)
                except Exception as fallback_error:
                    raise VectorStoreError(
                        f"Vector search and fallback both failed. Vector search error: {e}, Fallback error: {fallback_error}",
                        operation="similarity_search",
                        collection=self.collection_name
                    )
    
    def _fallback_text_search(self, query_embedding: List[float], k: int, 
                             filter_dict: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Fallback text-based search when vector search is not available.
        
        This method provides basic functionality when Atlas Vector Search is not configured.
        """
        try:
            # Simple text search as fallback
            query = {}
            if filter_dict:
                query.update(filter_dict)
            
            # Get recent documents as fallback
            results = list(self.collection.find(query).limit(k))
            
            documents = []
            for result in results:
                metadata = result.get("metadata", {})
                metadata["_id"] = str(result["_id"])
                metadata["score"] = 0.5  # Default score for fallback
                
                doc = Document(
                    page_content=result["content"],
                    metadata=metadata
                )
                documents.append(doc)
            
            logger.warning(f"Used fallback search, returned {len(documents)} documents")
            return documents
            
        except PyMongoError as e:
            logger.error(f"Fallback search failed: {e}")
            return []
    
    def similarity_search_by_text(self, query: str, k: int = 5, 
                                 filter_dict: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Perform similarity search using text query (requires embedding generation).
        
        Note: This method requires an embedding function to convert text to vectors.
        Use similarity_search() with pre-computed embeddings for better performance.
        
        Args:
            query: Text query to search for
            k: Number of similar documents to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of similar Document objects
        """
        # This method would require embedding generation
        # For now, we'll use a simple text search as placeholder
        logger.warning("similarity_search_by_text requires embedding generation - using text search fallback")
        return self._fallback_text_search([], k, filter_dict)
    
    def create_index(self) -> None:
        """Create vector search index for the collection.
        
        Note: Atlas Vector Search indexes must be created through MongoDB Atlas UI or CLI.
        This method provides the index definition for reference.
        """
        try:
            index_definition = {
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": 3072,  # text-embedding-3-large dimensions
                        "similarity": "cosine"
                    },
                    {
                        "type": "filter",
                        "path": "metadata.source"
                    },
                    {
                        "type": "filter", 
                        "path": "metadata.chunk_id"
                    }
                ]
            }
            
            logger.info(f"Vector index definition for collection '{self.collection_name}':")
            logger.info(f"Index name: {self.settings.vector_index_name}")
            logger.info(f"Index definition: {index_definition}")
            logger.info("Please create this index in MongoDB Atlas UI or using Atlas CLI")
            
            # Try to create regular indexes for metadata fields and content hash
            try:
                # Create compound index for duplicate detection
                self.collection.create_index([
                    ("content_hash", 1),
                    ("metadata.source", 1),
                    ("metadata.chunk_id", 1)
                ], unique=False, name="duplicate_detection_index")
                
                # Create separate indexes for common queries
                self.collection.create_index([("metadata.source", 1)])
                self.collection.create_index([("metadata.chunk_id", 1)])
                self.collection.create_index([("content_hash", 1)])
                
                logger.info("Created metadata and duplicate detection indexes successfully")
            except Exception as e:
                logger.warning(f"Could not create indexes: {e}")
                
        except Exception as e:
            logger.error(f"Failed to prepare index definition: {e}")
            raise
    
    def get_document_count(self) -> int:
        """Get total number of documents in the collection."""
        try:
            count = self.collection.count_documents({})
            logger.debug(f"Document count: {count}")
            return count
        except PyMongoError as e:
            logger.error(f"Failed to get document count: {e}")
            return 0
    
    def delete_documents(self, filter_dict: Dict[str, Any]) -> int:
        """Delete documents matching the filter.
        
        Args:
            filter_dict: MongoDB filter to match documents for deletion
            
        Returns:
            Number of deleted documents
        """
        try:
            result = self.collection.delete_many(filter_dict)
            logger.info(f"Deleted {result.deleted_count} documents")
            return result.deleted_count
        except PyMongoError as e:
            logger.error(f"Failed to delete documents: {e}")
            raise
    
    def clear_collection(self) -> int:
        """Clear all documents from the collection.
        
        Returns:
            Number of deleted documents
        """
        return self.delete_documents({})
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate a hash for document content to detect duplicates.
        
        Args:
            content: Document content to hash
            
        Returns:
            SHA-256 hash of the content
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _is_duplicate(self, document: Document) -> bool:
        """Check if a document is a duplicate based on content hash and metadata.
        
        Args:
            document: Document to check for duplicates
            
        Returns:
            True if document is a duplicate, False otherwise
        """
        try:
            content_hash = self._generate_content_hash(document.page_content)
            source = document.metadata.get("source", "")
            chunk_id = document.metadata.get("chunk_id", 0)
            
            # Check for exact content match
            existing_doc = self.collection.find_one({
                "content_hash": content_hash,
                "metadata.source": source,
                "metadata.chunk_id": chunk_id
            })
            
            return existing_doc is not None
            
        except Exception as e:
            logger.warning(f"Error checking for duplicates: {e}")
            return False
    
    def get_duplicate_count(self) -> Dict[str, int]:
        """Get statistics about duplicate documents in the collection.
        
        Returns:
            Dictionary with duplicate statistics
        """
        try:
            pipeline = [
                {
                    "$group": {
                        "_id": "$content_hash",
                        "count": {"$sum": 1},
                        "sources": {"$addToSet": "$metadata.source"}
                    }
                },
                {
                    "$match": {
                        "count": {"$gt": 1}
                    }
                }
            ]
            
            duplicates = list(self.collection.aggregate(pipeline))
            
            total_duplicates = sum(doc["count"] - 1 for doc in duplicates)  # Subtract 1 to count only extras
            unique_duplicate_groups = len(duplicates)
            
            return {
                "total_duplicate_documents": total_duplicates,
                "unique_duplicate_groups": unique_duplicate_groups,
                "duplicate_details": duplicates
            }
            
        except Exception as e:
            logger.error(f"Failed to get duplicate statistics: {e}")
            return {"total_duplicate_documents": 0, "unique_duplicate_groups": 0, "duplicate_details": []}
    
    def remove_duplicates(self) -> int:
        """Remove duplicate documents from the collection, keeping only the first occurrence.
        
        Returns:
            Number of duplicate documents removed
        """
        try:
            # Find all duplicate groups
            pipeline = [
                {
                    "$group": {
                        "_id": "$content_hash",
                        "docs": {"$push": {"id": "$_id", "source": "$metadata.source"}},
                        "count": {"$sum": 1}
                    }
                },
                {
                    "$match": {
                        "count": {"$gt": 1}
                    }
                }
            ]
            
            duplicate_groups = list(self.collection.aggregate(pipeline))
            
            removed_count = 0
            for group in duplicate_groups:
                # Keep the first document, remove the rest
                docs_to_remove = group["docs"][1:]  # Skip the first one
                ids_to_remove = [doc["id"] for doc in docs_to_remove]
                
                if ids_to_remove:
                    result = self.collection.delete_many({"_id": {"$in": ids_to_remove}})
                    removed_count += result.deleted_count
                    logger.debug(f"Removed {result.deleted_count} duplicates for content hash: {group['_id'][:8]}...")
            
            logger.info(f"Removed {removed_count} duplicate documents")
            return removed_count
            
        except Exception as e:
            logger.error(f"Failed to remove duplicates: {e}")
            raise