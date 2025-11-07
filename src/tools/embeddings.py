"""OpenAI embedding utilities for the MongoDB Chatbot System."""

import asyncio
import logging
from typing import List, Optional, Union
import time
from openai import OpenAI, AsyncOpenAI
from openai.types import CreateEmbeddingResponse

from ..core.config import get_settings
from ..core.exceptions import (
    EmbeddingError, 
    RateLimitError, 
    AuthenticationError,
    TimeoutError,
    handle_external_api_error
)
from ..core.retry import retry_embeddings, async_retry_embeddings

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """OpenAI embedding generator with batch processing and error handling."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize the embedding generator.
        
        Args:
            api_key: OpenAI API key. If None, uses config.
            model: Embedding model name. If None, uses config.
        """
        settings = get_settings()
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.embedding_model
        
        # Initialize OpenAI clients
        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)
        
        # Rate limiting configuration
        self.max_tokens_per_minute = 1000000  # text-embedding-3-large limit
        self.max_requests_per_minute = 3000
        self.tokens_used = 0
        self.requests_made = 0
        self.minute_start = time.time()
        
        logger.info(f"Initialized EmbeddingGenerator with model: {self.model}")
    
    def _reset_rate_limits_if_needed(self) -> None:
        """Reset rate limit counters if a minute has passed."""
        current_time = time.time()
        if current_time - self.minute_start >= 60:
            self.tokens_used = 0
            self.requests_made = 0
            self.minute_start = current_time
    
    def _check_rate_limits(self, estimated_tokens: int) -> None:
        """Check if request would exceed rate limits.
        
        Args:
            estimated_tokens: Estimated tokens for the request.
            
        Raises:
            RateLimitError: If rate limits would be exceeded.
        """
        self._reset_rate_limits_if_needed()
        
        if (self.tokens_used + estimated_tokens > self.max_tokens_per_minute or
            self.requests_made >= self.max_requests_per_minute):
            raise RateLimitError(
                f"Rate limit would be exceeded. "
                f"Tokens: {self.tokens_used}/{self.max_tokens_per_minute}, "
                f"Requests: {self.requests_made}/{self.max_requests_per_minute}"
            )
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        return len(text) // 4  # Rough estimate: 1 token ≈ 4 characters
    
    @retry_embeddings
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed.
            
        Returns:
            List of embedding values.
            
        Raises:
            EmbeddingError: If embedding generation fails.
        """
        if not text or not text.strip():
            raise EmbeddingError("Text cannot be empty")
        
        estimated_tokens = self._estimate_tokens(text)
        self._check_rate_limits(estimated_tokens)
        
        try:
            logger.debug(f"Generating embedding for text of length {len(text)}")
            
            response: CreateEmbeddingResponse = self.client.embeddings.create(
                model=self.model,
                input=text.strip(),
                encoding_format="float"
            )
            
            # Update rate limiting counters
            self.tokens_used += response.usage.total_tokens
            self.requests_made += 1
            
            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding with {len(embedding)} dimensions")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            # Convert to standardized exception
            standardized_error = handle_external_api_error(e, "openai_embeddings")
            if isinstance(standardized_error, EmbeddingError):
                raise standardized_error
            else:
                raise EmbeddingError(
                    f"Embedding generation failed: {e}",
                    model=self.model,
                    text_length=len(text)
                )   
 
    def generate_embeddings_batch(
        self, 
        texts: List[str], 
        batch_size: int = 100
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed.
            batch_size: Number of texts to process in each batch.
            
        Returns:
            List of embedding lists.
            
        Raises:
            EmbeddingError: If batch processing fails.
        """
        if not texts:
            return []
        
        # Filter out empty texts
        valid_texts = [text.strip() for text in texts if text and text.strip()]
        if not valid_texts:
            raise EmbeddingError("No valid texts provided")
        
        logger.info(f"Processing {len(valid_texts)} texts in batches of {batch_size}")
        
        all_embeddings = []
        
        for i in range(0, len(valid_texts), batch_size):
            batch = valid_texts[i:i + batch_size]
            batch_embeddings = self._process_batch(batch)
            all_embeddings.extend(batch_embeddings)
            
            logger.debug(f"Processed batch {i//batch_size + 1}/{(len(valid_texts)-1)//batch_size + 1}")
            
            # Small delay between batches to respect rate limits
            if i + batch_size < len(valid_texts):
                time.sleep(0.1)
        
        logger.info(f"Successfully generated {len(all_embeddings)} embeddings")
        return all_embeddings
    
    @retry_embeddings
    def _process_batch(self, texts: List[str]) -> List[List[float]]:
        """Process a single batch of texts.
        
        Args:
            texts: Batch of texts to embed.
            
        Returns:
            List of embeddings for the batch.
        """
        total_estimated_tokens = sum(self._estimate_tokens(text) for text in texts)
        self._check_rate_limits(total_estimated_tokens)
        
        try:
            response: CreateEmbeddingResponse = self.client.embeddings.create(
                model=self.model,
                input=texts,
                encoding_format="float"
            )
            
            # Update rate limiting counters
            self.tokens_used += response.usage.total_tokens
            self.requests_made += 1
            
            embeddings = [item.embedding for item in response.data]
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to process batch: {e}")
            # Convert to standardized exception
            standardized_error = handle_external_api_error(e, "openai_embeddings")
            if isinstance(standardized_error, EmbeddingError):
                raise standardized_error
            else:
                raise EmbeddingError(
                    f"Batch processing failed: {e}",
                    model=self.model,
                    text_length=sum(len(text) for text in texts)
                )
    
    @async_retry_embeddings
    async def generate_embedding_async(self, text: str) -> List[float]:
        """Generate embedding asynchronously for a single text.
        
        Args:
            text: Text to embed.
            
        Returns:
            List of embedding values.
        """
        if not text or not text.strip():
            raise EmbeddingError("Text cannot be empty")
        
        estimated_tokens = self._estimate_tokens(text)
        self._check_rate_limits(estimated_tokens)
        
        try:
            logger.debug(f"Generating async embedding for text of length {len(text)}")
            
            response = await self.async_client.embeddings.create(
                model=self.model,
                input=text.strip(),
                encoding_format="float"
            )
            
            # Update rate limiting counters
            self.tokens_used += response.usage.total_tokens
            self.requests_made += 1
            
            embedding = response.data[0].embedding
            logger.debug(f"Generated async embedding with {len(embedding)} dimensions")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate async embedding: {e}")
            # Convert to standardized exception
            standardized_error = handle_external_api_error(e, "openai_embeddings")
            if isinstance(standardized_error, EmbeddingError):
                raise standardized_error
            else:
                raise EmbeddingError(
                    f"Async embedding generation failed: {e}",
                    model=self.model,
                    text_length=len(text)
                )
    
    async def generate_embeddings_batch_async(
        self, 
        texts: List[str], 
        batch_size: int = 100,
        max_concurrent: int = 5
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts asynchronously.
        
        Args:
            texts: List of texts to embed.
            batch_size: Number of texts to process in each batch.
            max_concurrent: Maximum concurrent batch requests.
            
        Returns:
            List of embedding lists.
        """
        if not texts:
            return []
        
        # Filter out empty texts
        valid_texts = [text.strip() for text in texts if text and text.strip()]
        if not valid_texts:
            raise EmbeddingError("No valid texts provided")
        
        logger.info(f"Processing {len(valid_texts)} texts asynchronously in batches of {batch_size}")
        
        # Create batches
        batches = [
            valid_texts[i:i + batch_size] 
            for i in range(0, len(valid_texts), batch_size)
        ]
        
        # Process batches with concurrency limit
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_batch_with_semaphore(batch):
            async with semaphore:
                return await self._process_batch_async(batch)
        
        # Execute all batches
        batch_results = await asyncio.gather(
            *[process_batch_with_semaphore(batch) for batch in batches],
            return_exceptions=True
        )
        
        # Collect results and handle exceptions
        all_embeddings = []
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"Batch {i} failed: {result}")
                raise EmbeddingError(f"Batch processing failed: {result}")
            all_embeddings.extend(result)
        
        logger.info(f"Successfully generated {len(all_embeddings)} embeddings asynchronously")
        return all_embeddings
    
    @async_retry_embeddings
    async def _process_batch_async(self, texts: List[str]) -> List[List[float]]:
        """Process a single batch of texts asynchronously."""
        total_estimated_tokens = sum(self._estimate_tokens(text) for text in texts)
        self._check_rate_limits(total_estimated_tokens)
        
        try:
            response = await self.async_client.embeddings.create(
                model=self.model,
                input=texts,
                encoding_format="float"
            )
            
            # Update rate limiting counters
            self.tokens_used += response.usage.total_tokens
            self.requests_made += 1
            
            embeddings = [item.embedding for item in response.data]
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to process async batch: {e}")
            # Convert to standardized exception
            standardized_error = handle_external_api_error(e, "openai_embeddings")
            if isinstance(standardized_error, EmbeddingError):
                raise standardized_error
            else:
                raise EmbeddingError(
                    f"Async batch processing failed: {e}",
                    model=self.model,
                    text_length=sum(len(text) for text in texts)
                )


def create_embedding_generator(
    api_key: Optional[str] = None, 
    model: Optional[str] = None
) -> EmbeddingGenerator:
    """Factory function to create an EmbeddingGenerator instance.
    
    Args:
        api_key: OpenAI API key. If None, uses config.
        model: Embedding model name. If None, uses config.
        
    Returns:
        Configured EmbeddingGenerator instance.
    """
    return EmbeddingGenerator(api_key=api_key, model=model)