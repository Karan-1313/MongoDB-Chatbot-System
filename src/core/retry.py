"""Retry utilities for external API calls with exponential backoff."""

import asyncio
import logging
import time
from functools import wraps
from typing import Callable, Any, Optional, Type, Tuple, Union
from .exceptions import (
    ChatbotBaseException, 
    ExternalAPIError, 
    RateLimitError, 
    TimeoutError,
    handle_external_api_error
)

logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        timeout: Optional[float] = None
    ):
        """Initialize retry configuration.
        
        Args:
            max_attempts: Maximum number of retry attempts
            base_delay: Base delay in seconds before first retry
            max_delay: Maximum delay in seconds between retries
            exponential_base: Base for exponential backoff calculation
            jitter: Whether to add random jitter to delays
            timeout: Total timeout for all attempts in seconds
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.timeout = timeout
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number.
        
        Args:
            attempt: Current attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            import random
            # Add ±25% jitter
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
            delay = max(0.1, delay)  # Ensure minimum delay
        
        return delay


# Predefined retry configurations for different services
OPENAI_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay=2.0,
    max_delay=30.0,
    exponential_base=2.0,
    jitter=True,
    timeout=120.0
)

MONGODB_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=10.0,
    exponential_base=2.0,
    jitter=True,
    timeout=60.0
)

EMBEDDING_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay=3.0,
    max_delay=45.0,
    exponential_base=2.0,
    jitter=True,
    timeout=180.0
)


def should_retry(exception: Exception, service: str) -> bool:
    """Determine if an exception should trigger a retry.
    
    Args:
        exception: Exception that occurred
        service: Service name that caused the exception
        
    Returns:
        True if should retry, False otherwise
    """
    # Convert to standardized exception if needed
    if not isinstance(exception, ChatbotBaseException):
        exception = handle_external_api_error(exception, service)
    
    # Don't retry validation or authentication errors
    if isinstance(exception, (ValidationError, AuthenticationError)):
        return False
    
    # Retry rate limit errors (with delay)
    if isinstance(exception, RateLimitError):
        return True
    
    # Retry timeout errors
    if isinstance(exception, TimeoutError):
        return True
    
    # Retry external API errors (but not authentication)
    if isinstance(exception, ExternalAPIError):
        return not isinstance(exception, AuthenticationError)
    
    # Retry database connection errors
    if isinstance(exception, DatabaseError):
        error_msg = str(exception).lower()
        return any(keyword in error_msg for keyword in [
            "connection", "timeout", "network", "temporary"
        ])
    
    # Don't retry workflow errors by default
    if isinstance(exception, WorkflowError):
        return False
    
    # Retry unknown errors (might be temporary)
    return True


def retry_with_backoff(
    config: RetryConfig,
    service: str,
    retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None
):
    """Decorator for retrying functions with exponential backoff.
    
    Args:
        config: Retry configuration
        service: Service name for error handling
        retryable_exceptions: Tuple of exception types to retry on
        
    Returns:
        Decorated function
    """
    if retryable_exceptions is None:
        retryable_exceptions = (Exception,)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    # Check total timeout
                    if config.timeout and (time.time() - start_time) > config.timeout:
                        raise TimeoutError(
                            f"Total timeout exceeded ({config.timeout}s) for {service}",
                            operation=func.__name__
                        )
                    
                    # Execute the function
                    result = func(*args, **kwargs)
                    
                    # Log successful retry if this wasn't the first attempt
                    if attempt > 0:
                        logger.info(
                            f"Function {func.__name__} succeeded on attempt {attempt + 1} "
                            f"for service {service}"
                        )
                    
                    return result
                    
                except retryable_exceptions as e:
                    last_exception = e
                    
                    # Check if we should retry this exception
                    if not should_retry(e, service):
                        logger.warning(
                            f"Non-retryable error in {func.__name__} for {service}: {e}"
                        )
                        raise
                    
                    # Don't retry on the last attempt
                    if attempt == config.max_attempts - 1:
                        break
                    
                    # Calculate delay
                    delay = config.calculate_delay(attempt)
                    
                    # Handle rate limit specific delays
                    if isinstance(e, RateLimitError) and e.retry_after:
                        delay = max(delay, e.retry_after)
                    
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__} ({service}): {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    
                    time.sleep(delay)
            
            # All attempts failed
            logger.error(
                f"All {config.max_attempts} attempts failed for {func.__name__} ({service}). "
                f"Last error: {last_exception}"
            )
            
            # Raise the last exception, converting to standardized format if needed
            if isinstance(last_exception, ChatbotBaseException):
                raise last_exception
            else:
                raise handle_external_api_error(last_exception, service)
        
        return wrapper
    return decorator


def async_retry_with_backoff(
    config: RetryConfig,
    service: str,
    retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None
):
    """Async decorator for retrying functions with exponential backoff.
    
    Args:
        config: Retry configuration
        service: Service name for error handling
        retryable_exceptions: Tuple of exception types to retry on
        
    Returns:
        Decorated async function
    """
    if retryable_exceptions is None:
        retryable_exceptions = (Exception,)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    # Check total timeout
                    if config.timeout and (time.time() - start_time) > config.timeout:
                        raise TimeoutError(
                            f"Total timeout exceeded ({config.timeout}s) for {service}",
                            operation=func.__name__
                        )
                    
                    # Execute the async function
                    result = await func(*args, **kwargs)
                    
                    # Log successful retry if this wasn't the first attempt
                    if attempt > 0:
                        logger.info(
                            f"Async function {func.__name__} succeeded on attempt {attempt + 1} "
                            f"for service {service}"
                        )
                    
                    return result
                    
                except retryable_exceptions as e:
                    last_exception = e
                    
                    # Check if we should retry this exception
                    if not should_retry(e, service):
                        logger.warning(
                            f"Non-retryable error in async {func.__name__} for {service}: {e}"
                        )
                        raise
                    
                    # Don't retry on the last attempt
                    if attempt == config.max_attempts - 1:
                        break
                    
                    # Calculate delay
                    delay = config.calculate_delay(attempt)
                    
                    # Handle rate limit specific delays
                    if isinstance(e, RateLimitError) and e.retry_after:
                        delay = max(delay, e.retry_after)
                    
                    logger.warning(
                        f"Async attempt {attempt + 1} failed for {func.__name__} ({service}): {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    
                    await asyncio.sleep(delay)
            
            # All attempts failed
            logger.error(
                f"All {config.max_attempts} async attempts failed for {func.__name__} ({service}). "
                f"Last error: {last_exception}"
            )
            
            # Raise the last exception, converting to standardized format if needed
            if isinstance(last_exception, ChatbotBaseException):
                raise last_exception
            else:
                raise handle_external_api_error(last_exception, service)
        
        return wrapper
    return decorator


# Convenience decorators for common services
def retry_openai(func: Callable) -> Callable:
    """Decorator for retrying OpenAI API calls."""
    return retry_with_backoff(OPENAI_RETRY_CONFIG, "openai")(func)


def retry_mongodb(func: Callable) -> Callable:
    """Decorator for retrying MongoDB operations."""
    return retry_with_backoff(MONGODB_RETRY_CONFIG, "mongodb")(func)


def retry_embeddings(func: Callable) -> Callable:
    """Decorator for retrying embedding generation."""
    return retry_with_backoff(EMBEDDING_RETRY_CONFIG, "openai_embeddings")(func)


def async_retry_openai(func: Callable) -> Callable:
    """Async decorator for retrying OpenAI API calls."""
    return async_retry_with_backoff(OPENAI_RETRY_CONFIG, "openai")(func)


def async_retry_mongodb(func: Callable) -> Callable:
    """Async decorator for retrying MongoDB operations."""
    return async_retry_with_backoff(MONGODB_RETRY_CONFIG, "mongodb")(func)


def async_retry_embeddings(func: Callable) -> Callable:
    """Async decorator for retrying embedding generation."""
    return async_retry_with_backoff(EMBEDDING_RETRY_CONFIG, "openai_embeddings")(func)


# Import the exceptions here to avoid circular imports
from .exceptions import ValidationError, AuthenticationError, DatabaseError, WorkflowError