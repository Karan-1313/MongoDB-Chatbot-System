"""MongoDB connection management."""

import logging
from typing import Optional
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, ConfigurationError

from ..core.config import get_settings
from ..core.exceptions import (
    DatabaseError,
    AuthenticationError,
    ConfigurationError as ChatbotConfigurationError,
    TimeoutError
)
from ..core.retry import retry_mongodb

logger = logging.getLogger(__name__)


class MongoDBConnection:
    """MongoDB connection manager with connection pooling."""
    
    def __init__(self):
        self.settings = get_settings()
        self._client: Optional[MongoClient] = None
        self._database: Optional[Database] = None
        
    @retry_mongodb
    def connect(self) -> None:
        """Establish connection to MongoDB."""
        try:
            self._client = MongoClient(
                self.settings.mongodb_uri,
                maxPoolSize=50,
                minPoolSize=5,
                maxIdleTimeMS=30000,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                socketTimeoutMS=20000
            )
            
            # Test connection
            self._client.admin.command('ping')
            self._database = self._client[self.settings.mongodb_database]
            
            logger.info(f"Connected to MongoDB database: {self.settings.mongodb_database}")
            
        except ServerSelectionTimeoutError as e:
            logger.error(f"MongoDB server selection timeout: {e}")
            raise TimeoutError(
                f"Failed to connect to MongoDB server within timeout: {e}",
                operation="connect"
            )
        except ConnectionFailure as e:
            logger.error(f"MongoDB connection failure: {e}")
            if "authentication failed" in str(e).lower():
                raise AuthenticationError(
                    f"MongoDB authentication failed: {e}",
                    service="mongodb"
                )
            else:
                raise DatabaseError(
                    f"Failed to connect to MongoDB: {e}",
                    operation="connect"
                )
        except ConfigurationError as e:
            logger.error(f"MongoDB configuration error: {e}")
            raise ChatbotConfigurationError(
                f"MongoDB configuration error: {e}",
                setting="mongodb_uri"
            )
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {e}")
            raise DatabaseError(
                f"Unexpected MongoDB connection error: {e}",
                operation="connect"
            )
    
    def disconnect(self) -> None:
        """Close MongoDB connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._database = None
            logger.info("Disconnected from MongoDB")
    
    def get_database(self) -> Database:
        """Get database instance."""
        if self._database is None:
            self.connect()
        return self._database
    
    def get_collection(self, collection_name: Optional[str] = None) -> Collection:
        """Get collection instance."""
        db = self.get_database()
        col_name = collection_name or self.settings.mongodb_collection
        return db[col_name]
    
    def validate_connection(self) -> bool:
        """Validate MongoDB connection."""
        try:
            if self._client is None:
                logger.warning("No MongoDB client available for validation")
                return False
            
            # Ping with timeout
            self._client.admin.command('ping', maxTimeMS=5000)
            logger.debug("MongoDB connection validation successful")
            return True
            
        except ServerSelectionTimeoutError as e:
            logger.error(f"MongoDB connection validation timeout: {e}")
            return False
        except ConnectionFailure as e:
            logger.error(f"MongoDB connection validation failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during MongoDB connection validation: {e}")
            return False


# Global connection instance
_connection: Optional[MongoDBConnection] = None


def get_mongodb_connection() -> MongoDBConnection:
    """Get global MongoDB connection instance."""
    global _connection
    if _connection is None:
        _connection = MongoDBConnection()
    return _connection