"""Response models for the API."""

from datetime import datetime
from typing import List
from pydantic import BaseModel, Field


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    
    answer: str = Field(..., description="Generated answer")
    sources: List[str] = Field(default_factory=list, description="Source documents used")
    session_id: str = Field(..., description="Session ID")
    processing_time: float = Field(..., description="Processing time in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Based on the documents, the main topic is...",
                "sources": ["document1.pdf", "document2.txt"],
                "session_id": "user-123-session",
                "processing_time": 2.34
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid request format",
                "status_code": 400,
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }


class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")
    version: str = Field("1.0.0", description="Application version")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-01T12:00:00Z",
                "version": "1.0.0"
            }
        }