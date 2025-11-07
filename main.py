"""Main entry point for the MongoDB Chatbot System."""

import signal
import sys
import asyncio
from pathlib import Path
from typing import Optional

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.core.config import validate_required_env_vars
from src.core.logging import setup_logging, get_logger


class GracefulShutdown:
    """Handle graceful shutdown of the application."""
    
    def __init__(self):
        self.shutdown_event = asyncio.Event()
        self.logger = get_logger(__name__)
        
    def setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.shutdown_event.set()
        
        # Handle SIGINT (Ctrl+C) and SIGTERM
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # On Windows, also handle SIGBREAK
        if sys.platform == "win32":
            signal.signal(signal.SIGBREAK, signal_handler)


def validate_environment():
    """Validate environment configuration and dependencies."""
    logger = get_logger(__name__)
    
    try:
        # Validate environment variables
        logger.info("Validating environment configuration...")
        validate_required_env_vars()
        logger.info("✓ Environment configuration validated successfully")
        
        # Test MongoDB connection
        logger.info("Testing MongoDB connection...")
        from src.database.connection import get_mongodb_connection
        connection = get_mongodb_connection()
        connection.connect()
        connection.disconnect()
        logger.info("✓ MongoDB connection test successful")
        
        # Test OpenAI API key (basic validation)
        logger.info("Validating OpenAI configuration...")
        from src.core.config import get_settings
        settings = get_settings()
        if not settings.openai_api_key.startswith('sk-'):
            logger.warning("OpenAI API key format may be invalid")
        else:
            logger.info("✓ OpenAI configuration validated")
            
    except Exception as e:
        logger.error(f"Environment validation failed: {e}")
        raise


async def run_server():
    """Run the FastAPI server with graceful shutdown support."""
    logger = get_logger(__name__)
    
    try:
        # Import FastAPI components
        from src.api.main import app
        from src.core.config import get_settings
        import uvicorn
        
        settings = get_settings()
        
        # Set up graceful shutdown
        shutdown_handler = GracefulShutdown()
        shutdown_handler.setup_signal_handlers()
        
        logger.info(f"Starting server on {settings.api_host}:{settings.api_port}")
        
        # Create uvicorn config
        config = uvicorn.Config(
            app,
            host=settings.api_host,
            port=settings.api_port,
            log_config=None,  # Use our custom logging
            access_log=False,  # We handle access logging in middleware
        )
        
        # Create and start server
        server = uvicorn.Server(config)
        
        # Start server in background task
        server_task = asyncio.create_task(server.serve())
        
        # Wait for shutdown signal or server completion
        done, pending = await asyncio.wait(
            [server_task, asyncio.create_task(shutdown_handler.shutdown_event.wait())],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # If shutdown was requested, gracefully stop the server
        if shutdown_handler.shutdown_event.is_set():
            logger.info("Shutdown signal received, stopping server...")
            server.should_exit = True
            
            # Wait for server to stop gracefully (with timeout)
            try:
                await asyncio.wait_for(server_task, timeout=30.0)
                logger.info("Server stopped gracefully")
            except asyncio.TimeoutError:
                logger.warning("Server shutdown timeout, forcing exit")
                
        # Cancel any pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
                
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise


def main():
    """Main application entry point."""
    logger: Optional[object] = None
    
    try:
        # Setup logging first
        setup_logging()
        logger = get_logger(__name__)
        
        logger.info("=" * 60)
        logger.info("MongoDB Chatbot System Starting Up")
        logger.info("=" * 60)
        
        # Validate environment
        validate_environment()
        
        # Run the server
        asyncio.run(run_server())
        
    except KeyboardInterrupt:
        if logger:
            logger.info("Application interrupted by user")
    except Exception as e:
        error_msg = f"Failed to start application: {e}"
        if logger:
            logger.error(error_msg, exc_info=True)
        else:
            print(error_msg)
        sys.exit(1)
    finally:
        if logger:
            logger.info("MongoDB Chatbot System shutdown complete")


if __name__ == "__main__":
    main()