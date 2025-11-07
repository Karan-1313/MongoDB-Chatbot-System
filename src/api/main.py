"""FastAPI application initialization."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

from ..core.config import get_settings
from ..core.logging import setup_logging
from ..core.monitoring import MonitoringMiddleware, get_metrics, get_health_status
from ..database.connection import get_mongodb_connection
from .models.response import ErrorResponse, HealthResponse
from ..core.exceptions import (
    ChatbotBaseException,
    get_http_status_code,
    create_error_response
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting up MongoDB Chatbot API...")
    
    # Setup logging
    setup_logging()
    logger.info("Logging system initialized")
    
    # Initialize MongoDB connection
    try:
        connection = get_mongodb_connection()
        connection.connect()
        logger.info("MongoDB connection established")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down MongoDB Chatbot API...")
    try:
        connection = get_mongodb_connection()
        connection.disconnect()
        logger.info("MongoDB connection closed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI app
app = FastAPI(
    title="MongoDB Chatbot API",
    description="A chatbot system using MongoDB vector search and LangGraph",
    version="1.0.0",
    lifespan=lifespan
)

# Get settings
settings = get_settings()

# Add monitoring middleware (before CORS)
app.add_middleware(MonitoringMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handlers
@app.exception_handler(ChatbotBaseException)
async def chatbot_exception_handler(request: Request, exc: ChatbotBaseException):
    """Handle chatbot-specific exceptions globally."""
    logger.error(f"Chatbot exception in {request.url.path}: {exc}")
    
    status_code = get_http_status_code(exc)
    error_response = create_error_response(exc)
    
    # Create ErrorResponse model
    error_model = ErrorResponse(
        error=error_response["error"],
        message=error_response["message"],
        status_code=status_code
    )
    
    headers = {}
    if exc.retry_after:
        headers["Retry-After"] = str(exc.retry_after)
    
    return JSONResponse(
        status_code=status_code,
        content=error_model.dict(),
        headers=headers
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions globally."""
    logger.warning(f"HTTP exception in {request.url.path}: {exc.detail}")
    
    error_response = ErrorResponse(
        error=exc.__class__.__name__,
        message=exc.detail,
        status_code=exc.status_code
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.dict()
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    logger.error(f"Unhandled exception in {request.url.path}: {exc}", exc_info=True)
    
    error_response = ErrorResponse(
        error="InternalServerError",
        message="An unexpected error occurred. Please try again later.",
        status_code=500
    )
    
    return JSONResponse(
        status_code=500,
        content=error_response.dict()
    )


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with comprehensive status."""
    try:
        # Get health status from monitoring
        health_data = get_health_status()
        
        # Check MongoDB connection
        from ..database.operations import DatabaseOperations
        db_ops = DatabaseOperations()
        db_healthy = db_ops.health_check()
        
        # Determine overall status
        if not db_healthy:
            status = "unhealthy"
            health_data['issues'].append("Database connection failed")
        elif health_data['status'] == "unhealthy":
            status = "unhealthy"
        elif health_data['status'] == "degraded" or not db_healthy:
            status = "degraded"
        else:
            status = "healthy"
        
        if status != "healthy":
            return JSONResponse(
                status_code=503,
                content={
                    "status": status,
                    "issues": health_data['issues'],
                    "database_healthy": db_healthy,
                    "metrics": health_data['metrics']
                }
            )
        
        return HealthResponse(status=status)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(
                error="ServiceUnavailable",
                message="Health check failed",
                status_code=503
            ).dict()
        )


# Metrics endpoint
@app.get("/metrics")
async def metrics_endpoint():
    """Get application metrics."""
    try:
        metrics = get_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="InternalServerError",
                message="Failed to retrieve metrics",
                status_code=500
            ).dict()
        )


# Include routers
from .routes import chat_router
app.include_router(chat_router, prefix="/api/v1", tags=["chat"])

# Status endpoint
@app.get("/status")
async def status_check():
    """Detailed status endpoint."""
    try:
        from ..database.operations import DatabaseOperations
        from ..graph.workflow import get_workflow
        
        # Check database
        db_ops = DatabaseOperations()
        db_status = "healthy" if db_ops.health_check() else "unhealthy"
        
        # Check workflow
        try:
            workflow = get_workflow()
            workflow_status = "healthy"
        except Exception:
            workflow_status = "unhealthy"
        
        # Check OpenAI (basic check)
        openai_status = "unknown"  # Would need actual API call to verify
        
        overall_status = "healthy" if all([
            db_status == "healthy",
            workflow_status == "healthy"
        ]) else "degraded"
        
        return {
            "status": overall_status,
            "components": {
                "database": db_status,
                "workflow": workflow_status,
                "openai": openai_status
            },
            "version": "1.0.0",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
        )


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "MongoDB Chatbot API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "status": "/status"
    }