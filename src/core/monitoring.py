"""Monitoring and metrics collection for the MongoDB Chatbot System."""

import time
import uuid
from typing import Dict, Any, Optional
from collections import defaultdict, deque
from datetime import datetime, timedelta
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from .logging import (
    get_logger,
    get_performance_logger,
    log_api_request,
    log_performance_metric
)


class MetricsCollector:
    """Collects and stores application metrics."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.request_times = deque(maxlen=max_history)
        self.request_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.endpoint_metrics = defaultdict(lambda: {
            'count': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'errors': 0
        })
        self.start_time = datetime.utcnow()
    
    def record_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration: float,
        error: Optional[str] = None
    ):
        """Record a request metric."""
        now = datetime.utcnow()
        
        # Record request time
        self.request_times.append({
            'timestamp': now,
            'duration': duration,
            'status_code': status_code,
            'method': method,
            'path': path
        })
        
        # Update counters
        self.request_counts[f"{method} {path}"] += 1
        if status_code >= 400:
            self.error_counts[f"{status_code}"] += 1
        
        # Update endpoint metrics
        endpoint_key = f"{method} {path}"
        metrics = self.endpoint_metrics[endpoint_key]
        metrics['count'] += 1
        metrics['total_time'] += duration
        metrics['avg_time'] = metrics['total_time'] / metrics['count']
        metrics['min_time'] = min(metrics['min_time'], duration)
        metrics['max_time'] = max(metrics['max_time'], duration)
        
        if status_code >= 400:
            metrics['errors'] += 1
        
        # Log performance metric
        log_performance_metric(
            operation=f"api_request_{method.lower()}",
            duration=duration,
            success=status_code < 400,
            method=method,
            path=path,
            status_code=status_code,
            error=error
        )
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of collected metrics."""
        now = datetime.utcnow()
        uptime = (now - self.start_time).total_seconds()
        
        # Calculate recent metrics (last 5 minutes)
        recent_cutoff = now - timedelta(minutes=5)
        recent_requests = [
            req for req in self.request_times
            if req['timestamp'] > recent_cutoff
        ]
        
        # Calculate averages
        total_requests = len(self.request_times)
        recent_count = len(recent_requests)
        
        avg_response_time = (
            sum(req['duration'] for req in self.request_times) / total_requests
            if total_requests > 0 else 0
        )
        
        recent_avg_response_time = (
            sum(req['duration'] for req in recent_requests) / recent_count
            if recent_count > 0 else 0
        )
        
        # Error rates
        total_errors = sum(self.error_counts.values())
        error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0
        
        recent_errors = sum(1 for req in recent_requests if req['status_code'] >= 400)
        recent_error_rate = (recent_errors / recent_count * 100) if recent_count > 0 else 0
        
        return {
            'uptime_seconds': uptime,
            'total_requests': total_requests,
            'recent_requests_5min': recent_count,
            'avg_response_time_ms': round(avg_response_time * 1000, 2),
            'recent_avg_response_time_ms': round(recent_avg_response_time * 1000, 2),
            'error_rate_percent': round(error_rate, 2),
            'recent_error_rate_percent': round(recent_error_rate, 2),
            'requests_per_minute': round(recent_count / 5, 2) if recent_count > 0 else 0,
            'endpoint_metrics': dict(self.endpoint_metrics),
            'error_counts': dict(self.error_counts),
            'timestamp': now.isoformat()
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status based on metrics."""
        metrics = self.get_metrics_summary()
        
        # Determine health status
        status = "healthy"
        issues = []
        
        # Check error rate
        if metrics['recent_error_rate_percent'] > 10:
            status = "degraded"
            issues.append(f"High error rate: {metrics['recent_error_rate_percent']}%")
        elif metrics['recent_error_rate_percent'] > 25:
            status = "unhealthy"
            issues.append(f"Critical error rate: {metrics['recent_error_rate_percent']}%")
        
        # Check response time
        if metrics['recent_avg_response_time_ms'] > 5000:
            status = "degraded" if status == "healthy" else status
            issues.append(f"Slow response time: {metrics['recent_avg_response_time_ms']}ms")
        elif metrics['recent_avg_response_time_ms'] > 10000:
            status = "unhealthy"
            issues.append(f"Critical response time: {metrics['recent_avg_response_time_ms']}ms")
        
        return {
            'status': status,
            'issues': issues,
            'metrics': metrics
        }


# Global metrics collector instance
metrics_collector = MetricsCollector()


class MonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for monitoring API requests and responses."""
    
    def __init__(self, app, exclude_paths: Optional[list] = None):
        super().__init__(app)
        self.logger = get_logger(__name__)
        self.perf_logger = get_performance_logger()
        self.exclude_paths = exclude_paths or ['/health', '/metrics', '/docs', '/openapi.json']
    
    async def dispatch(self, request: Request, call_next):
        """Process request and response with monitoring."""
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Skip monitoring for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
        
        # Start timing
        start_time = time.time()
        
        # Extract request info
        method = request.method
        path = request.url.path
        user_agent = request.headers.get('user-agent', 'Unknown')
        ip_address = self._get_client_ip(request)
        
        # Log request start
        self.logger.info(
            f"Request started: {method} {path}",
            extra={
                'request_id': request_id,
                'method': method,
                'path': path,
                'user_agent': user_agent,
                'ip_address': ip_address,
                'event': 'request_start'
            }
        )
        
        # Process request
        error_message = None
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            # Handle exceptions
            status_code = 500
            error_message = str(e)
            self.logger.error(
                f"Request failed with exception: {method} {path}",
                extra={
                    'request_id': request_id,
                    'method': method,
                    'path': path,
                    'error': error_message,
                    'event': 'request_exception'
                },
                exc_info=True
            )
            # Create error response
            from fastapi.responses import JSONResponse
            response = JSONResponse(
                status_code=500,
                content={
                    'error': 'InternalServerError',
                    'message': 'An unexpected error occurred',
                    'request_id': request_id
                }
            )
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Add response headers
        response.headers['X-Request-ID'] = request_id
        response.headers['X-Response-Time'] = f"{duration:.3f}s"
        
        # Log request completion
        log_api_request(
            method=method,
            path=path,
            status_code=status_code,
            duration=duration,
            request_id=request_id,
            user_agent=user_agent,
            ip_address=ip_address,
            error=error_message
        )
        
        # Record metrics
        metrics_collector.record_request(
            method=method,
            path=path,
            status_code=status_code,
            duration=duration,
            error=error_message
        )
        
        # Log slow requests
        if duration > 1.0:  # Log requests taking more than 1 second
            self.logger.warning(
                f"Slow request: {method} {path} took {duration:.3f}s",
                extra={
                    'request_id': request_id,
                    'method': method,
                    'path': path,
                    'duration': duration,
                    'status_code': status_code,
                    'event': 'slow_request'
                }
            )
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers (common in reverse proxy setups)
        forwarded_for = request.headers.get('x-forwarded-for')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        real_ip = request.headers.get('x-real-ip')
        if real_ip:
            return real_ip
        
        # Fallback to direct client IP
        if hasattr(request, 'client') and request.client:
            return request.client.host
        
        return 'unknown'


def get_metrics() -> Dict[str, Any]:
    """Get current application metrics."""
    return metrics_collector.get_metrics_summary()


def get_health_status() -> Dict[str, Any]:
    """Get application health status."""
    return metrics_collector.get_health_status()


class WorkflowMonitor:
    """Monitor workflow execution performance."""
    
    def __init__(self, workflow_name: str, session_id: str):
        self.workflow_name = workflow_name
        self.session_id = session_id
        self.start_time = None
        self.nodes_executed = []
        self.logger = get_logger(f"workflow.{workflow_name}")
    
    def start(self):
        """Start monitoring workflow execution."""
        self.start_time = time.time()
        self.logger.info(
            f"Starting workflow: {self.workflow_name}",
            extra={
                'workflow_name': self.workflow_name,
                'session_id': self.session_id,
                'event': 'workflow_start'
            }
        )
    
    def node_executed(self, node_name: str, duration: float, success: bool, error: Optional[str] = None):
        """Record node execution."""
        self.nodes_executed.append({
            'node': node_name,
            'duration': duration,
            'success': success,
            'error': error
        })
        
        self.logger.info(
            f"Node executed: {node_name}",
            extra={
                'workflow_name': self.workflow_name,
                'session_id': self.session_id,
                'node_name': node_name,
                'duration': duration,
                'success': success,
                'error': error,
                'event': 'node_execution'
            }
        )
    
    def complete(self, success: bool, error: Optional[str] = None):
        """Complete workflow monitoring."""
        if not self.start_time:
            return
        
        duration = time.time() - self.start_time
        
        from .logging import log_workflow_execution
        log_workflow_execution(
            workflow_name=self.workflow_name,
            session_id=self.session_id,
            duration=duration,
            success=success,
            nodes_executed=[node['node'] for node in self.nodes_executed],
            error=error,
            total_nodes=len(self.nodes_executed),
            failed_nodes=sum(1 for node in self.nodes_executed if not node['success'])
        )


class DatabaseMonitor:
    """Monitor database operations."""
    
    def __init__(self, operation: str, collection: str):
        self.operation = operation
        self.collection = collection
        self.start_time = None
        self.logger = get_logger("database.monitor")
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.start_time:
            return
        
        duration = time.time() - self.start_time
        success = exc_type is None
        error = str(exc_val) if exc_val else None
        
        from .logging import log_database_operation
        log_database_operation(
            operation=self.operation,
            collection=self.collection,
            duration=duration,
            success=success,
            error=error
        )