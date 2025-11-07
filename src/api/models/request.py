"""Request models for the API."""

from typing import Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    
    question: str = Field(..., min_length=1, max_length=2000, description="User question")
    session_id: Optional[str] = Field(None, description="Optional session ID for context")
    max_tokens: Optional[int] = Field(1000, ge=1, le=4000, description="Maximum tokens in response")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the main topic of the documents?",
                "session_id": "user-123-session",
                "max_tokens": 1000
            }
        }