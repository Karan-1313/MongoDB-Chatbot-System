"""Retrieval node for LangGraph workflow."""

import logging
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document

from ..graph.state import GraphState, GraphStateManager
from ..tools.vector_store import MongoVectorStore
from ..tools.embeddings import EmbeddingGenerator
from ..core.config import get_settings
from ..core.exceptions import (
    WorkflowError,
    VectorStoreError,
    EmbeddingError,
    ValidationError
)

logger = logging.getLogger(__name__)


class RetrievalNode:
    """Node for retrieving relevant documents from MongoDB vector store."""
    
    def __init__(self, vector_store: Optional[MongoVectorStore] = None, 
                 embedding_generator: Optional[EmbeddingGenerator] = None):
        """Initialize retrieval node.
        
        Args:
            vector_store: MongoDB vector store instance
            embedding_generator: Embedding generator instance
        """
        self.vector_store = vector_store or MongoVectorStore()
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.settings = get_settings()
    
    def __call__(self, state: GraphState) -> GraphState:
        """Execute retrieval node.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated graph state with retrieved documents
        """
        try:
            logger.info(f"Starting document retrieval for question: {state['question'][:100]}...")
            
            # Validate input state
            try:
                validated_state = GraphStateManager.validate_state(state)
                question = validated_state['question']
            except Exception as e:
                raise WorkflowError(
                    f"Invalid state provided to retrieval node: {e}",
                    node="retrieval",
                    state=state
                )
            
            if not question or not question.strip():
                raise ValidationError(
                    "Question cannot be empty for document retrieval",
                    field="question",
                    value=question
                )
            
            # Generate embedding for the question
            try:
                query_embedding = self.embedding_generator.generate_embedding(question)
                logger.debug(f"Generated embedding for question (dimension: {len(query_embedding)})")
            except EmbeddingError as e:
                # Re-raise embedding errors as-is
                raise e
            except Exception as e:
                logger.error(f"Unexpected error generating embedding: {e}")
                raise WorkflowError(
                    f"Unexpected embedding generation error: {e}",
                    node="retrieval",
                    state=validated_state
                )
            
            # Perform similarity search
            try:
                max_docs = getattr(self.settings, 'max_retrieved_docs', 5)
                similarity_threshold = getattr(self.settings, 'similarity_threshold', 0.7)
                
                # Get similar documents
                retrieved_docs = self.vector_store.similarity_search(
                    query_embedding=query_embedding,
                    k=max_docs
                )
                
                # Filter by similarity threshold if scores are available
                filtered_docs = []
                for doc in retrieved_docs:
                    score = doc.metadata.get('score', 1.0)
                    if score >= similarity_threshold:
                        filtered_docs.append(doc)
                    else:
                        logger.debug(f"Filtered out document with score {score} (threshold: {similarity_threshold})")
                
                # If no documents meet threshold, take top documents anyway
                if not filtered_docs and retrieved_docs:
                    logger.warning(f"No documents met similarity threshold {similarity_threshold}, using top documents")
                    filtered_docs = retrieved_docs[:min(3, len(retrieved_docs))]
                
                logger.info(f"Retrieved {len(filtered_docs)} relevant documents")
                
                # Update state with retrieved documents
                validated_state['retrieved_docs'] = filtered_docs
                
                # Generate sources list
                sources = []
                for doc in filtered_docs:
                    source = doc.metadata.get('source', 'Unknown')
                    if source not in sources:
                        sources.append(source)
                
                validated_state['sources'] = sources
                
                # Clear any previous errors
                validated_state['error'] = None
                
                logger.info(f"Retrieval completed successfully. Found {len(filtered_docs)} documents from {len(sources)} sources")
                return validated_state
                
            except VectorStoreError as e:
                # Re-raise vector store errors as-is
                raise e
            except Exception as e:
                logger.error(f"Unexpected error during vector search: {e}")
                raise WorkflowError(
                    f"Unexpected vector search error: {e}",
                    node="retrieval",
                    state=validated_state
                )
                
        except (ValidationError, EmbeddingError, VectorStoreError, WorkflowError):
            # Re-raise known exceptions as-is
            raise
        except Exception as e:
            logger.error(f"Critical retrieval node execution failed: {e}")
            raise WorkflowError(
                f"Critical retrieval node failure: {e}",
                node="retrieval",
                state=state
            )


def create_retrieval_node(vector_store: Optional[MongoVectorStore] = None,
                         embedding_generator: Optional[EmbeddingGenerator] = None) -> RetrievalNode:
    """Factory function to create a retrieval node.
    
    Args:
        vector_store: Optional vector store instance
        embedding_generator: Optional embedding generator instance
        
    Returns:
        Configured retrieval node
    """
    return RetrievalNode(vector_store=vector_store, embedding_generator=embedding_generator)


# LangGraph node function
def retrieval_node(state: GraphState) -> GraphState:
    """LangGraph node function for document retrieval.
    
    This function creates a new RetrievalNode instance and executes it.
    Use this function when defining LangGraph workflows.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated graph state with retrieved documents
    """
    node = RetrievalNode()
    return node(state)