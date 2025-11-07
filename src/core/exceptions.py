"""Custom exception classes for the MongoDB Chatbot System."""

import logging
from typing import Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Categories of errors for better classification and handling."""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    DATABASE = "database"
    EXTERNAL_API = "external_api"
    WORKFLOW = "workflow"
    CONFIGURATION = "configuration"
    TIMEOUT = "timeout"
    RESOURCE = "resource"
    UNKNOWN = "unknown"


class ChatbotBaseException(Exception):
    """Base exception class for all chatbot-related errors."""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        details: Optional[Dict[str, Any]] = None,
        retry_after: Optional[int] = None,
        original_error: Optional[Exception] = None
    ):
        """Initialize base exception.
        
        Args:
            message: Human-readable error message
            category: Error category for classification
            details: Additional error details
            retry_after: Seconds to wait before retrying (if applicable)
            original_error: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.category = category
        self.details = details or {}
        self.retry_after = retry_after
        self.original_error = original_error
        
        # Log the exception
        logger.error(
            f"{self.__class__.__name__}: {message}",
            extra={
                "category": category.value,
                "details": details,
                "retry_after": retry_after,
                "original_error": str(original_error) if original_error else None
            }
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "category": self.category.value,
            "details": self.details,
            "retry_after": self.retry_after
        }


class ValidationError(ChatbotBaseException):
    """Exception raised for input validation errors."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)
        
        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION,
            details=details
        )


class AuthenticationError(ChatbotBaseException):
    """Exception raised for authentication and authorization errors."""
    
    def __init__(self, message: str, service: Optional[str] = None):
        details = {}
        if service:
            details["service"] = service
        
        super().__init__(
            message=message,
            category=ErrorCategory.AUTHENTICATION,
            details=details
        )


class RateLimitError(ChatbotBaseException):
    """Exception raised when rate limits are exceeded."""
    
    def __init__(self, message: str, service: str, retry_after: Optional[int] = None):
        super().__init__(
            message=message,
            category=ErrorCategory.RATE_LIMIT,
            details={"service": service},
            retry_after=retry_after
        )


class DatabaseError(ChatbotBaseException):
    """Exception raised for database-related errors."""
    
    def __init__(self, message: str, operation: Optional[str] = None, collection: Optional[str] = None):
        details = {}
        if operation:
            details["operation"] = operation
        if collection:
            details["collection"] = collection
        
        super().__init__(
            message=message,
            category=ErrorCategory.DATABASE,
            details=details
        )


class ExternalAPIError(ChatbotBaseException):
    """Exception raised for external API errors (OpenAI, etc.)."""
    
    def __init__(
        self,
        message: str,
        service: str,
        status_code: Optional[int] = None,
        retry_after: Optional[int] = None,
        original_error: Optional[Exception] = None
    ):
        details = {"service": service}
        if status_code:
            details["status_code"] = status_code
        
        super().__init__(
            message=message,
            category=ErrorCategory.EXTERNAL_API,
            details=details,
            retry_after=retry_after,
            original_error=original_error
        )


class WorkflowError(ChatbotBaseException):
    """Exception raised for LangGraph workflow errors."""
    
    def __init__(self, message: str, node: Optional[str] = None, state: Optional[Dict[str, Any]] = None):
        details = {}
        if node:
            details["node"] = node
        if state:
            # Only include safe state information
            safe_state = {
                "question_length": len(state.get("question", "")),
                "has_retrieved_docs": bool(state.get("retrieved_docs")),
                "has_answer": bool(state.get("answer")),
                "session_id": state.get("session_id", "")
            }
            details["state"] = safe_state
        
        super().__init__(
            message=message,
            category=ErrorCategory.WORKFLOW,
            details=details
        )


class ConfigurationError(ChatbotBaseException):
    """Exception raised for configuration-related errors."""
    
    def __init__(self, message: str, setting: Optional[str] = None):
        details = {}
        if setting:
            details["setting"] = setting
        
        super().__init__(
            message=message,
            category=ErrorCategory.CONFIGURATION,
            details=details
        )


class TimeoutError(ChatbotBaseException):
    """Exception raised for timeout errors."""
    
    def __init__(self, message: str, operation: str, timeout_seconds: Optional[float] = None):
        details = {"operation": operation}
        if timeout_seconds:
            details["timeout_seconds"] = timeout_seconds
        
        super().__init__(
            message=message,
            category=ErrorCategory.TIMEOUT,
            details=details
        )


class ResourceError(ChatbotBaseException):
    """Exception raised for resource-related errors (memory, disk, etc.)."""
    
    def __init__(self, message: str, resource_type: str, current_usage: Optional[str] = None):
        details = {"resource_type": resource_type}
        if current_usage:
            details["current_usage"] = current_usage
        
        super().__init__(
            message=message,
            category=ErrorCategory.RESOURCE,
            details=details
        )


class EmbeddingError(ExternalAPIError):
    """Exception raised for embedding generation errors."""
    
    def __init__(self, message: str, model: Optional[str] = None, text_length: Optional[int] = None):
        details = {"service": "openai_embeddings"}
        if model:
            details["model"] = model
        if text_length:
            details["text_length"] = text_length
        
        super().__init__(
            message=message,
            service="openai_embeddings",
            original_error=None
        )
        self.details.update(details)


class VectorStoreError(DatabaseError):
    """Exception raised for vector store operations."""
    
    def __init__(self, message: str, operation: str, collection: Optional[str] = None):
        super().__init__(
            message=message,
            operation=operation,
            collection=collection
        )


def handle_external_api_error(error: Exception, service: str) -> ExternalAPIError:
    """Convert external API errors to standardized ExternalAPIError.
    
    Args:
        error: Original exception from external API
        service: Name of the external service
        
    Returns:
        Standardized ExternalAPIError
    """
    error_message = str(error)
    status_code = None
    retry_after = None
    
    # Handle OpenAI specific errors
    if service == "openai":
        if "rate limit" in error_message.lower():
            return RateLimitError(
                message=f"OpenAI rate limit exceeded: {error_message}",
                service=service,
                retry_after=60  # Default retry after 60 seconds
            )
        elif "authentication" in error_message.lower() or "api key" in error_message.lower():
            return AuthenticationError(
                message=f"OpenAI authentication failed: {error_message}",
                service=service
            )
        elif "timeout" in error_message.lower():
            return TimeoutError(
                message=f"OpenAI request timeout: {error_message}",
                operation="api_request"
            )
    
    # Handle MongoDB specific errors
    elif service == "mongodb":
        if "timeout" in error_message.lower():
            return TimeoutError(
                message=f"MongoDB operation timeout: {error_message}",
                operation="database_operation"
            )
        elif "authentication" in error_message.lower():
            return AuthenticationError(
                message=f"MongoDB authentication failed: {error_message}",
                service=service
            )
        elif "connection" in error_message.lower():
            return DatabaseError(
                message=f"MongoDB connection error: {error_message}",
                operation="connection"
            )
    
    # Generic external API error
    return ExternalAPIError(
        message=f"{service} API error: {error_message}",
        service=service,
        status_code=status_code,
        retry_after=retry_after,
        original_error=error
    )


def get_http_status_code(exception: ChatbotBaseException) -> int:
    """Get appropriate HTTP status code for an exception.
    
    Args:
        exception: Chatbot exception
        
    Returns:
        HTTP status code
    """
    status_mapping = {
        ErrorCategory.VALIDATION: 400,
        ErrorCategory.AUTHENTICATION: 401,
        ErrorCategory.RATE_LIMIT: 429,
        ErrorCategory.DATABASE: 503,
        ErrorCategory.EXTERNAL_API: 502,
        ErrorCategory.WORKFLOW: 500,
        ErrorCategory.CONFIGURATION: 500,
        ErrorCategory.TIMEOUT: 408,
        ErrorCategory.RESOURCE: 507,
        ErrorCategory.UNKNOWN: 500
    }
    
    return status_mapping.get(exception.category, 500)


def create_error_response(exception: ChatbotBaseException) -> Dict[str, Any]:
    """Create standardized error response from exception.
    
    Args:
        exception: Chatbot exception
        
    Returns:
        Error response dictionary
    """
    response = exception.to_dict()
    response["status_code"] = get_http_status_code(exception)
    
    # Add retry information if available
    if exception.retry_after:
        response["retry_after"] = exception.retry_after
    
    return response