# Comprehensive Error Handling Guide

This document describes the comprehensive error handling system implemented for the MongoDB Chatbot System.

## Overview

The error handling system provides:
- **Custom exception classes** for different error types
- **Retry logic** with exponential backoff for external API calls
- **Comprehensive try-catch blocks** with appropriate error responses
- **Standardized error responses** across all API endpoints
- **Proper logging** and monitoring of errors

## Custom Exception Classes

### Base Exception: `ChatbotBaseException`

All custom exceptions inherit from `ChatbotBaseException`, which provides:
- Error categorization
- Structured error details
- Retry-after information
- Original error tracking
- Automatic logging

### Exception Categories

1. **ValidationError**: Input validation failures
2. **AuthenticationError**: Authentication/authorization issues
3. **RateLimitError**: Rate limit exceeded errors
4. **DatabaseError**: Database operation failures
5. **ExternalAPIError**: External API call failures
6. **WorkflowError**: LangGraph workflow errors
7. **ConfigurationError**: Configuration-related errors
8. **TimeoutError**: Operation timeout errors
9. **ResourceError**: Resource exhaustion errors
10. **EmbeddingError**: Embedding generation failures
11. **VectorStoreError**: Vector store operation failures

### Usage Example

```python
from src.core.exceptions import ValidationError, DatabaseError

# Raise a validation error
if not user_input:
    raise ValidationError(
        "Input cannot be empty",
        field="question",
        value=user_input
    )

# Raise a database error
try:
    collection.insert_one(document)
except PyMongoError as e:
    raise DatabaseError(
        f"Failed to insert document: {e}",
        operation="insert_one",
        collection="documents"
    )
```

## Retry Logic

### Retry Decorators

The system provides decorators for automatic retry with exponential backoff:

- `@retry_openai`: For OpenAI API calls
- `@retry_mongodb`: For MongoDB operations
- `@retry_embeddings`: For embedding generation
- `@async_retry_openai`: Async version for OpenAI
- `@async_retry_mongodb`: Async version for MongoDB
- `@async_retry_embeddings`: Async version for embeddings

### Retry Configuration

Each service has its own retry configuration:

```python
OPENAI_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay=2.0,
    max_delay=30.0,
    exponential_base=2.0,
    jitter=True,
    timeout=120.0
)
```

### Usage Example

```python
from src.core.retry import retry_openai

@retry_openai
def call_openai_api(prompt):
    return client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
```

## Error Handling in Components

### API Layer

The FastAPI application includes:
- Global exception handlers for all custom exceptions
- HTTP status code mapping
- Structured error responses
- Request/response logging

### Database Layer

MongoDB operations include:
- Connection retry logic
- Timeout handling
- Authentication error detection
- Connection validation

### External API Layer

OpenAI API calls include:
- Rate limit handling
- Authentication error detection
- Timeout management
- Response validation

### Workflow Layer

LangGraph workflows include:
- Node-level error handling
- State validation
- Retry logic for individual nodes
- Graceful degradation

## Error Response Format

All API errors return a standardized format:

```json
{
    "error": "ValidationError",
    "message": "Question must be at least 3 characters long",
    "status_code": 400,
    "timestamp": "2024-01-01T12:00:00Z",
    "category": "validation",
    "details": {
        "field": "question",
        "value": "hi"
    },
    "retry_after": null
}
```

## HTTP Status Code Mapping

- **ValidationError**: 400 Bad Request
- **AuthenticationError**: 401 Unauthorized
- **RateLimitError**: 429 Too Many Requests
- **DatabaseError**: 503 Service Unavailable
- **ExternalAPIError**: 502 Bad Gateway
- **WorkflowError**: 500 Internal Server Error
- **ConfigurationError**: 500 Internal Server Error
- **TimeoutError**: 408 Request Timeout
- **ResourceError**: 507 Insufficient Storage

## Logging and Monitoring

### Log Levels

- **ERROR**: All exceptions and failures
- **WARNING**: Retry attempts and degraded performance
- **INFO**: Successful operations and state changes
- **DEBUG**: Detailed execution information

### Log Format

All error logs include:
- Exception type and message
- Error category
- Additional details
- Stack trace (for unexpected errors)
- Request context

### Example Log Entry

```
2024-01-01 12:00:00 ERROR [chat.py:45] ValidationError: Question must be at least 3 characters long
  category: validation
  details: {"field": "question", "value": "hi"}
  request_id: req-123456
```

## Best Practices

### 1. Use Specific Exceptions

Always use the most specific exception type:

```python
# Good
raise ValidationError("Invalid email format", field="email", value=email)

# Bad
raise Exception("Invalid email")
```

### 2. Include Context

Provide relevant context in exception details:

```python
raise DatabaseError(
    "Failed to insert document",
    operation="insert_one",
    collection="documents"
)
```

### 3. Handle Retries Appropriately

Use retry decorators for transient failures:

```python
@retry_mongodb
def insert_document(doc):
    return collection.insert_one(doc)
```

### 4. Log at Appropriate Levels

- Use ERROR for actual problems
- Use WARNING for recoverable issues
- Use INFO for normal operations
- Use DEBUG for detailed tracing

### 5. Provide User-Friendly Messages

Convert technical errors to user-friendly messages:

```python
try:
    result = complex_operation()
except DatabaseError as e:
    logger.error(f"Database error: {e}")
    raise HTTPException(
        status_code=503,
        detail="The service is temporarily unavailable. Please try again later."
    )
```

## Testing Error Handling

### Unit Tests

Test error conditions explicitly:

```python
def test_empty_question_validation():
    with pytest.raises(ValidationError) as exc_info:
        process_question("")
    
    assert exc_info.value.category == ErrorCategory.VALIDATION
    assert "empty" in str(exc_info.value).lower()
```

### Integration Tests

Test error propagation through the system:

```python
def test_database_error_handling():
    # Mock database failure
    with patch('src.database.connection.MongoClient') as mock_client:
        mock_client.side_effect = ConnectionFailure("Connection failed")
        
        response = client.post("/api/v1/chat", json={"question": "test"})
        assert response.status_code == 503
        assert "service unavailable" in response.json()["message"].lower()
```

## Monitoring and Alerting

### Metrics to Monitor

1. **Error Rates**: Track error rates by type and endpoint
2. **Retry Rates**: Monitor retry attempts and success rates
3. **Response Times**: Track API response times
4. **Database Health**: Monitor connection status and query performance
5. **External API Health**: Track OpenAI API response times and errors

### Alert Conditions

- Error rate > 5% for any endpoint
- Database connection failures
- OpenAI API rate limit exceeded
- High retry rates (> 20%)
- Response times > 30 seconds

## Configuration

### Environment Variables

Error handling behavior can be configured via environment variables:

```bash
# Retry configuration
MAX_RETRY_ATTEMPTS=3
RETRY_BASE_DELAY=1.0
RETRY_MAX_DELAY=60.0

# Timeout configuration
API_TIMEOUT=30
DATABASE_TIMEOUT=10
OPENAI_TIMEOUT=120

# Logging configuration
LOG_LEVEL=INFO
ERROR_LOG_FILE=/var/log/chatbot/errors.log
```

### Runtime Configuration

Some error handling behavior can be adjusted at runtime:

```python
# Adjust retry configuration
from src.core.retry import OPENAI_RETRY_CONFIG
OPENAI_RETRY_CONFIG.max_attempts = 5

# Adjust logging level
import logging
logging.getLogger('src').setLevel(logging.DEBUG)
```

This comprehensive error handling system ensures robust operation of the MongoDB Chatbot System with proper error recovery, user feedback, and system monitoring.