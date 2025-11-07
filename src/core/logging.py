"""Logging configuration for the MongoDB Chatbot System."""

import json
import logging
import logging.handlers
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .config import get_settings


class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for machine-readable logs."""
    
    def format(self, record):
        # Base log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


class ColoredConsoleFormatter(logging.Formatter):
    """Custom formatter with colors and structured format for console output."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Add timestamp
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create base format with performance metrics if available
        extra_info = ""
        if hasattr(record, 'duration') and record.duration is not None:
            extra_info += f" | {record.duration:.3f}s"
        if hasattr(record, 'request_id') and record.request_id:
            extra_info += f" | req:{record.request_id[:8]}"
        if hasattr(record, 'session_id') and record.session_id:
            extra_info += f" | session:{record.session_id[:8]}"
        
        log_format = (
            f"{timestamp} | {{levelname:8}} | {{name:20}} | {{message}}{extra_info}"
        )
        
        # Add color for console output
        if hasattr(record, 'levelname') and record.levelname in self.COLORS:
            color = self.COLORS[record.levelname]
            reset = self.COLORS['RESET']
            log_format = f"{color}{log_format}{reset}"
        
        formatter = logging.Formatter(log_format, style='{')
        return formatter.format(record)


def setup_logging(log_level: Optional[str] = None, enable_file_logging: bool = True) -> None:
    """Set up logging configuration for the application."""
    settings = get_settings()
    level = log_level or settings.log_level
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level))
    console_handler.setFormatter(ColoredConsoleFormatter())
    root_logger.addHandler(console_handler)
    
    # File handlers for structured logging
    if enable_file_logging:
        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Application log file (structured JSON)
        app_handler = logging.handlers.RotatingFileHandler(
            logs_dir / "app.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        app_handler.setLevel(getattr(logging, level))
        app_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(app_handler)
        
        # Error log file (errors and above only)
        error_handler = logging.handlers.RotatingFileHandler(
            logs_dir / "error.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=10
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(error_handler)
        
        # Performance log file (for metrics)
        perf_handler = logging.handlers.RotatingFileHandler(
            logs_dir / "performance.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(StructuredFormatter())
        
        # Create performance logger
        perf_logger = logging.getLogger("performance")
        perf_logger.addHandler(perf_handler)
        perf_logger.setLevel(logging.INFO)
        perf_logger.propagate = False
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("pymongo").setLevel(logging.WARNING)
    logging.getLogger("motor").setLevel(logging.WARNING)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level: {level}")
    if enable_file_logging:
        logger.info("File logging enabled: app.log, error.log, performance.log")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given name."""
    return logging.getLogger(name)


def get_performance_logger() -> logging.Logger:
    """Get the performance logger instance."""
    return logging.getLogger("performance")


class PerformanceMonitor:
    """Context manager for monitoring performance metrics."""
    
    def __init__(self, operation: str, logger: Optional[logging.Logger] = None, **kwargs):
        self.operation = operation
        self.logger = logger or get_performance_logger()
        self.extra_data = kwargs
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(
            f"Starting {self.operation}",
            extra={
                "operation": self.operation,
                "event": "start",
                **self.extra_data
            }
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        if exc_type is None:
            self.logger.info(
                f"Completed {self.operation}",
                extra={
                    "operation": self.operation,
                    "event": "complete",
                    "duration": duration,
                    "success": True,
                    **self.extra_data
                }
            )
        else:
            self.logger.error(
                f"Failed {self.operation}",
                extra={
                    "operation": self.operation,
                    "event": "error",
                    "duration": duration,
                    "success": False,
                    "error_type": exc_type.__name__ if exc_type else None,
                    "error_message": str(exc_val) if exc_val else None,
                    **self.extra_data
                }
            )
    
    @property
    def duration(self) -> Optional[float]:
        """Get the duration of the operation."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


def log_performance_metric(
    operation: str,
    duration: float,
    success: bool = True,
    logger: Optional[logging.Logger] = None,
    **kwargs
) -> None:
    """Log a performance metric."""
    perf_logger = logger or get_performance_logger()
    
    perf_logger.info(
        f"Performance metric: {operation}",
        extra={
            "operation": operation,
            "duration": duration,
            "success": success,
            "metric_type": "performance",
            **kwargs
        }
    )


def log_api_request(
    method: str,
    path: str,
    status_code: int,
    duration: float,
    request_id: str,
    user_agent: Optional[str] = None,
    ip_address: Optional[str] = None,
    **kwargs
) -> None:
    """Log API request details."""
    logger = get_logger("api.requests")
    
    logger.info(
        f"{method} {path} - {status_code}",
        extra={
            "event_type": "api_request",
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration": duration,
            "request_id": request_id,
            "user_agent": user_agent,
            "ip_address": ip_address,
            **kwargs
        }
    )


def log_workflow_execution(
    workflow_name: str,
    session_id: str,
    duration: float,
    success: bool,
    nodes_executed: list,
    error: Optional[str] = None,
    **kwargs
) -> None:
    """Log workflow execution details."""
    logger = get_logger("workflow.execution")
    
    logger.info(
        f"Workflow {workflow_name} {'completed' if success else 'failed'}",
        extra={
            "event_type": "workflow_execution",
            "workflow_name": workflow_name,
            "session_id": session_id,
            "duration": duration,
            "success": success,
            "nodes_executed": nodes_executed,
            "error": error,
            **kwargs
        }
    )


def log_database_operation(
    operation: str,
    collection: str,
    duration: float,
    success: bool,
    documents_affected: Optional[int] = None,
    error: Optional[str] = None,
    **kwargs
) -> None:
    """Log database operation details."""
    logger = get_logger("database.operations")
    
    logger.info(
        f"Database {operation} on {collection}",
        extra={
            "event_type": "database_operation",
            "operation": operation,
            "collection": collection,
            "duration": duration,
            "success": success,
            "documents_affected": documents_affected,
            "error": error,
            **kwargs
        }
    )