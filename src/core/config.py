"""Configuration management for the MongoDB Chatbot System."""

import os
from typing import Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # MongoDB Configuration
    mongodb_uri: str
    mongodb_database: str = "chatbot_db"
    mongodb_collection: str = "documents"
    
    # OpenAI Configuration
    openai_api_key: str
    embedding_model: str = "text-embedding-3-large"
    chat_model: str = "gpt-4"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    
    # Vector Search Configuration
    vector_index_name: str = "vector_index"
    similarity_threshold: float = 0.7
    max_retrieved_docs: int = 5
    
    @field_validator('mongodb_uri')
    @classmethod
    def validate_mongodb_uri(cls, v):
        if not v:
            raise ValueError('MongoDB URI is required')
        return v
    
    @field_validator('openai_api_key')
    @classmethod
    def validate_openai_api_key(cls, v):
        if not v:
            raise ValueError('OpenAI API key is required')
        return v
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of: {valid_levels}')
        return v.upper()
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False
    }


def get_settings() -> Settings:
    """Get application settings instance."""
    return Settings()


def validate_required_env_vars() -> None:
    """Validate that all required environment variables are set."""
    validation_errors = []
    
    try:
        settings = get_settings()
        
        # Validate MongoDB URI format
        if not settings.mongodb_uri.startswith(('mongodb://', 'mongodb+srv://')):
            validation_errors.append("MONGODB_URI must start with 'mongodb://' or 'mongodb+srv://'")
        
        # Validate OpenAI API key format
        if not settings.openai_api_key.startswith('sk-'):
            validation_errors.append("OPENAI_API_KEY must start with 'sk-'")
        
        # Validate port range
        if not (1 <= settings.api_port <= 65535):
            validation_errors.append("API_PORT must be between 1 and 65535")
        
        # Validate similarity threshold range
        if not (0.0 <= settings.similarity_threshold <= 1.0):
            validation_errors.append("SIMILARITY_THRESHOLD must be between 0.0 and 1.0")
        
        # Validate max retrieved docs
        if settings.max_retrieved_docs < 1:
            validation_errors.append("MAX_RETRIEVED_DOCS must be a positive integer")
        
        if validation_errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in validation_errors)
            print(f"✗ {error_msg}")
            raise SystemExit(1)
        
        print("✓ All required environment variables are properly configured")
        
    except Exception as e:
        if validation_errors:
            # Re-raise the validation errors we collected
            raise SystemExit(1)
        else:
            # Handle other configuration errors (missing variables, etc.)
            error_msg = str(e)
            if "field required" in error_msg.lower():
                missing_vars = []
                if "mongodb_uri" in error_msg.lower():
                    missing_vars.append("MONGODB_URI")
                if "openai_api_key" in error_msg.lower():
                    missing_vars.append("OPENAI_API_KEY")
                
                if missing_vars:
                    print(f"✗ Missing required environment variables: {', '.join(missing_vars)}")
                    print("  Please copy .env.example to .env and configure the required variables")
                else:
                    print(f"✗ Configuration validation failed: {e}")
            else:
                print(f"✗ Configuration validation failed: {e}")
            
            raise SystemExit(1)