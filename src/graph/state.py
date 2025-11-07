"""Graph state models for LangGraph workflow."""

import json
import uuid
from typing import List, Optional, TypedDict, Dict, Any
from langchain_core.documents import Document
from pydantic import BaseModel, Field, validator


class GraphState(TypedDict):
    """State model for the LangGraph workflow."""
    
    question: str
    retrieved_docs: List[Document]
    context: str
    answer: str
    sources: List[str]
    session_id: str
    error: Optional[str]


class GraphStateValidator(BaseModel):
    """Pydantic model for validating GraphState."""
    
    question: str = Field(..., min_length=1, max_length=2000, description="User question")
    retrieved_docs: List[Dict[str, Any]] = Field(default_factory=list, description="Retrieved documents")
    context: str = Field(default="", description="Formatted context from retrieved documents")
    answer: str = Field(default="", description="Generated answer")
    sources: List[str] = Field(default_factory=list, description="Source references")
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Session identifier")
    error: Optional[str] = Field(default=None, description="Error message if any")
    
    @validator('question')
    def validate_question(cls, v):
        if not v or not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()
    
    @validator('session_id')
    def validate_session_id(cls, v):
        if not v:
            return str(uuid.uuid4())
        return v
    
    @validator('sources')
    def validate_sources(cls, v):
        # Remove duplicates while preserving order
        seen = set()
        return [x for x in v if not (x in seen or seen.add(x))]


class GraphStateManager:
    """Manager for GraphState validation and serialization."""
    
    @staticmethod
    def validate_state(state: Dict[str, Any]) -> GraphState:
        """Validate and normalize a state dictionary."""
        try:
            # Convert Documents to dict format for validation
            if 'retrieved_docs' in state and state['retrieved_docs']:
                docs_as_dicts = []
                for doc in state['retrieved_docs']:
                    if isinstance(doc, Document):
                        docs_as_dicts.append({
                            'page_content': doc.page_content,
                            'metadata': doc.metadata
                        })
                    else:
                        docs_as_dicts.append(doc)
                state['retrieved_docs'] = docs_as_dicts
            
            # Validate using Pydantic
            validated = GraphStateValidator(**state)
            
            # Convert back to GraphState format
            result: GraphState = {
                'question': validated.question,
                'retrieved_docs': [
                    Document(page_content=doc['page_content'], metadata=doc['metadata'])
                    for doc in validated.retrieved_docs
                ],
                'context': validated.context,
                'answer': validated.answer,
                'sources': validated.sources,
                'session_id': validated.session_id,
                'error': validated.error
            }
            
            return result
            
        except Exception as e:
            raise ValueError(f"State validation failed: {str(e)}")
    
    @staticmethod
    def serialize_state(state: GraphState) -> Dict[str, Any]:
        """Serialize GraphState to a JSON-serializable dictionary."""
        try:
            serialized = {
                'question': state['question'],
                'retrieved_docs': [
                    {
                        'page_content': doc.page_content,
                        'metadata': doc.metadata
                    }
                    for doc in state['retrieved_docs']
                ],
                'context': state['context'],
                'answer': state['answer'],
                'sources': state['sources'],
                'session_id': state['session_id'],
                'error': state['error']
            }
            return serialized
        except Exception as e:
            raise ValueError(f"State serialization failed: {str(e)}")
    
    @staticmethod
    def deserialize_state(data: Dict[str, Any]) -> GraphState:
        """Deserialize a dictionary back to GraphState."""
        try:
            state: GraphState = {
                'question': data.get('question', ''),
                'retrieved_docs': [
                    Document(page_content=doc['page_content'], metadata=doc['metadata'])
                    for doc in data.get('retrieved_docs', [])
                ],
                'context': data.get('context', ''),
                'answer': data.get('answer', ''),
                'sources': data.get('sources', []),
                'session_id': data.get('session_id', str(uuid.uuid4())),
                'error': data.get('error')
            }
            return state
        except Exception as e:
            raise ValueError(f"State deserialization failed: {str(e)}")
    
    @staticmethod
    def create_initial_state(question: str, session_id: Optional[str] = None) -> GraphState:
        """Create an initial GraphState with just a question."""
        return GraphStateManager.validate_state({
            'question': question,
            'session_id': session_id or str(uuid.uuid4())
        })