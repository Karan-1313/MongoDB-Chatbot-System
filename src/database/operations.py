"""Database operations and health checks."""

import logging
from typing import Dict, Any, Optional
from pymongo.errors import PyMongoError, ConnectionFailure, ServerSelectionTimeoutError

from .connection import get_mongodb_connection
from ..core.exceptions import DatabaseError, TimeoutError
from ..core.retry import retry_mongodb
from ..core.monitoring import DatabaseMonitor

logger = logging.getLogger(__name__)


class DatabaseOperations:
    """Database operations and utilities."""
    
    def __init__(self):
        """Initialize database operations."""
        self.connection = get_mongodb_connection()
    
    @retry_mongodb
    def health_check(self) -> bool:
        """Perform a health check on the database connection.
        
        Returns:
            True if database is healthy, False otherwise
        """
        with DatabaseMonitor("health_check", "system"):
            try:
                # Validate connection
                if not self.connection.validate_connection():
                    logger.warning("Database connection validation failed")
                    return False
                
                # Perform a simple operation
                db = self.connection.get_database()
                result = db.command("ping")
                
                if result.get("ok") == 1:
                    logger.debug("Database health check passed")
                    return True
                else:
                    logger.warning("Database ping returned unexpected result")
                    return False
                    
            except ServerSelectionTimeoutError as e:
                logger.error(f"Database health check timeout: {e}")
                return False
            except ConnectionFailure as e:
                logger.error(f"Database connection failure during health check: {e}")
                return False
            except PyMongoError as e:
                logger.error(f"Database error during health check: {e}")
                return False
            except Exception as e:
                logger.error(f"Unexpected error during database health check: {e}")
                return False
    
    @retry_mongodb
    def get_collection_stats(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for a collection.
        
        Args:
            collection_name: Name of the collection (uses default if None)
            
        Returns:
            Dictionary with collection statistics
        """
        with DatabaseMonitor("get_collection_stats", collection_name or "default"):
            try:
                collection = self.connection.get_collection(collection_name)
                
                # Get basic stats
                stats = {
                    "document_count": collection.count_documents({}),
                    "collection_name": collection.name,
                    "database_name": collection.database.name
                }
                
                # Try to get more detailed stats
                try:
                    db_stats = collection.database.command("collStats", collection.name)
                    stats.update({
                        "size_bytes": db_stats.get("size", 0),
                        "storage_size_bytes": db_stats.get("storageSize", 0),
                        "index_count": db_stats.get("nindexes", 0),
                        "avg_obj_size": db_stats.get("avgObjSize", 0)
                    })
                except Exception as e:
                    logger.warning(f"Could not get detailed collection stats: {e}")
                
                return stats
                
            except PyMongoError as e:
                logger.error(f"Failed to get collection stats: {e}")
                raise DatabaseError(
                    f"Failed to get collection statistics: {e}",
                    operation="get_collection_stats",
                    collection=collection_name
                )
    
    @retry_mongodb
    def get_database_info(self) -> Dict[str, Any]:
        """Get general database information.
        
        Returns:
            Dictionary with database information
        """
        try:
            db = self.connection.get_database()
            
            # Get database stats
            db_stats = db.command("dbStats")
            
            # Get list of collections
            collections = db.list_collection_names()
            
            info = {
                "database_name": db.name,
                "collections": collections,
                "collection_count": len(collections),
                "data_size_bytes": db_stats.get("dataSize", 0),
                "storage_size_bytes": db_stats.get("storageSize", 0),
                "index_count": db_stats.get("indexes", 0),
                "objects": db_stats.get("objects", 0)
            }
            
            return info
            
        except PyMongoError as e:
            logger.error(f"Failed to get database info: {e}")
            raise DatabaseError(
                f"Failed to get database information: {e}",
                operation="get_database_info"
            )
    
    @retry_mongodb
    def test_write_operation(self) -> bool:
        """Test if write operations are working.
        
        Returns:
            True if write operations work, False otherwise
        """
        with DatabaseMonitor("test_write_operation", "system"):
            try:
                collection = self.connection.get_collection()
                
                # Insert a test document
                test_doc = {
                    "_test": True,
                    "timestamp": "test_write_operation",
                    "data": "health_check"
                }
                
                result = collection.insert_one(test_doc)
                
                if result.inserted_id:
                    # Clean up the test document
                    collection.delete_one({"_id": result.inserted_id})
                    logger.debug("Database write operation test passed")
                    return True
                else:
                    logger.warning("Database write operation test failed - no ID returned")
                    return False
                    
            except PyMongoError as e:
                logger.error(f"Database write operation test failed: {e}")
                return False
            except Exception as e:
                logger.error(f"Unexpected error during write operation test: {e}")
                return False
    
    def get_comprehensive_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status of the database.
        
        Returns:
            Dictionary with detailed health information
        """
        health_status = {
            "overall_healthy": False,
            "connection_healthy": False,
            "read_operations": False,
            "write_operations": False,
            "error_details": []
        }
        
        try:
            # Test connection
            health_status["connection_healthy"] = self.health_check()
            
            # Test read operations
            try:
                stats = self.get_collection_stats()
                health_status["read_operations"] = True
                health_status["collection_stats"] = stats
            except Exception as e:
                health_status["error_details"].append(f"Read operations failed: {e}")
            
            # Test write operations
            try:
                health_status["write_operations"] = self.test_write_operation()
            except Exception as e:
                health_status["error_details"].append(f"Write operations failed: {e}")
            
            # Overall health
            health_status["overall_healthy"] = (
                health_status["connection_healthy"] and
                health_status["read_operations"] and
                health_status["write_operations"]
            )
            
        except Exception as e:
            health_status["error_details"].append(f"Health check failed: {e}")
            logger.error(f"Comprehensive health check failed: {e}")
        
        return health_status