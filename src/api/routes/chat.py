"""Chat endpoint implementation."""

import logging
import time
import uuid
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status, Request
from fastapi.responses import JSONResponse

from ..models import ChatRequest, ChatResponse, ErrorResponse
from ...graph.workflow import get_workflow
from ...core.config import get_settings
from ...core.logging import PerformanceMonitor
from ...core.exceptions import (
    ChatbotBaseException,
    ValidationError,
    AuthenticationError,
    RateLimitError,
    TimeoutError,
    WorkflowError,
    get_http_status_code,
    create_error_response
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Chat endpoint that processes user questions and returns AI-generated responses.
    
    This endpoint:
    1. Validates the incoming request
    2. Executes the LangGraph workflow (retrieval + reasoning)
    3. Returns the formatted response with sources and metadata
    
    Args:
        request: ChatRequest containing the user question and optional parameters
        
    Returns:
        ChatResponse with the generated answer, sources, and metadata
        
    Raises:
        HTTPException: For various error conditions (400, 500, etc.)
    """
    # Generate session ID if not provided
    session_id = request.session_id or f"session-{uuid.uuid4().hex[:8]}"
    
    with PerformanceMonitor(
        "chat_request",
        session_id=session_id,
        question_length=len(request.question),
        max_tokens=request.max_tokens
    ) as monitor:
        try:
            logger.info(
                f"Processing chat request for session {session_id}",
                extra={
                    'session_id': session_id,
                    'question_length': len(request.question),
                    'max_tokens': request.max_tokens
                }
            )
            logger.debug(f"Question: {request.question[:100]}...")
            
            # Additional validation
            if len(request.question.strip()) < 3:
                raise ValidationError(
                    "Question must be at least 3 characters long",
                    field="question",
                    value=request.question
                )
            
            # Get the workflow instance
            try:
                workflow = get_workflow()
            except Exception as e:
                logger.error(f"Failed to get workflow instance: {e}")
                raise WorkflowError(f"Failed to initialize workflow: {e}")
            
            # Prepare workflow configuration
            config = {
                "max_tokens": request.max_tokens,
                "configurable": {"thread_id": session_id}
            }
            
            # Execute the workflow
            logger.debug("Executing LangGraph workflow")
            try:
                result = await workflow.arun(
                    question=request.question,
                    session_id=session_id,
                    config=config
                )
            except Exception as e:
                logger.error(f"Workflow execution failed: {e}")
                raise WorkflowError(f"Workflow execution failed: {e}")
            
            # Check for workflow errors
            if result.get('error'):
                error_msg = result['error']
                logger.error(f"Workflow returned error: {error_msg}")
                raise WorkflowError(f"Workflow error: {error_msg}")
            
            # Extract response data
            answer = result.get('answer', '')
            sources = result.get('sources', [])
            
            # Validate that we have a meaningful response
            if not answer or answer.strip() == '':
                logger.warning("Empty answer generated")
                answer = "I apologize, but I couldn't generate a meaningful response to your question. Please try rephrasing your question or check if relevant documents are available."
            
            # Create response
            response = ChatResponse(
                answer=answer,
                sources=sources,
                session_id=session_id,
                processing_time=round(monitor.duration or 0, 3) if monitor.duration else 0
            )
            
            logger.info(
                f"Chat request completed successfully",
                extra={
                    'session_id': session_id,
                    'duration': monitor.duration,
                    'answer_length': len(answer),
                    'sources_count': len(sources)
                }
            )
            
            return response
            
        except ChatbotBaseException as e:
            # Handle our custom exceptions
            logger.error(f"Chatbot error in chat endpoint: {e}")
            
            status_code = get_http_status_code(e)
            error_response = create_error_response(e)
            
            raise HTTPException(
                status_code=status_code,
                detail=error_response["message"],
                headers={"Retry-After": str(e.retry_after)} if e.retry_after else None
            )
            
        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
            
        except Exception as e:
            logger.error(f"Unexpected error in chat endpoint: {e}", exc_info=True)
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred while processing your request. Please try again later."
            )


@router.get("/chat/sessions/{session_id}/history")
async def get_session_history(session_id: str) -> Dict[str, Any]:
    """
    Get the conversation history for a specific session.
    
    Args:
        session_id: The session ID to retrieve history for
        
    Returns:
        Dictionary containing the session history
        
    Raises:
        HTTPException: If session not found or other errors occur
    """
    try:
        logger.info(f"Retrieving history for session {session_id}")
        
        workflow = get_workflow()
        history = workflow.get_state_history(session_id)
        
        if not history:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No history found for session {session_id}"
            )
        
        # Format history for response
        formatted_history = []
        for state_snapshot in history:
            if hasattr(state_snapshot, 'values'):
                state_data = state_snapshot.values
                formatted_history.append({
                    "question": state_data.get('question', ''),
                    "answer": state_data.get('answer', ''),
                    "sources": state_data.get('sources', []),
                    "timestamp": getattr(state_snapshot, 'created_at', None)
                })
        
        return {
            "session_id": session_id,
            "history": formatted_history,
            "total_interactions": len(formatted_history)
        }
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Error retrieving session history: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve session history"
        )


@router.delete("/chat/sessions/{session_id}")
async def clear_session(session_id: str) -> Dict[str, str]:
    """
    Clear the conversation history for a specific session.
    
    Args:
        session_id: The session ID to clear
        
    Returns:
        Confirmation message
        
    Raises:
        HTTPException: If clearing fails
    """
    try:
        logger.info(f"Clearing session {session_id}")
        
        workflow = get_workflow()
        success = workflow.clear_session(session_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to clear session {session_id}"
            )
        
        return {
            "message": f"Session {session_id} cleared successfully",
            "session_id": session_id
        }
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Error clearing session: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear session"
        )