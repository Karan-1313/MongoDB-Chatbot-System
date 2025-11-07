"""Reasoning node for LangGraph workflow."""

import logging
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from openai import OpenAI
from openai.types.chat import ChatCompletion

from ..graph.state import GraphState, GraphStateManager
from ..core.config import get_settings
from ..core.exceptions import (
    WorkflowError,
    ExternalAPIError,
    AuthenticationError,
    RateLimitError,
    TimeoutError,
    handle_external_api_error
)
from ..core.retry import retry_openai

logger = logging.getLogger(__name__)


class ReasoningNode:
    """Node for generating responses using OpenAI GPT and retrieved context."""
    
    def __init__(self, openai_client: Optional[OpenAI] = None):
        """Initialize reasoning node.
        
        Args:
            openai_client: Optional OpenAI client instance
        """
        self.settings = get_settings()
        self.client = openai_client or OpenAI(api_key=self.settings.openai_api_key)
    
    def __call__(self, state: GraphState) -> GraphState:
        """Execute reasoning node.
        
        Args:
            state: Current graph state with retrieved documents
            
        Returns:
            Updated graph state with generated answer
        """
        try:
            logger.info("Starting response generation...")
            
            # Validate input state
            validated_state = GraphStateManager.validate_state(state)
            question = validated_state['question']
            retrieved_docs = validated_state['retrieved_docs']
            
            if not question or not question.strip():
                logger.error("Empty question provided for reasoning")
                validated_state['error'] = "Question cannot be empty"
                return validated_state
            
            # Format context from retrieved documents
            context = self._format_context(retrieved_docs)
            validated_state['context'] = context
            
            # Generate response using OpenAI
            try:
                answer = self._generate_response(question, context, retrieved_docs)
                validated_state['answer'] = answer
                
                # Update sources with proper citations
                sources = self._extract_sources(retrieved_docs)
                validated_state['sources'] = sources
                
                # Clear any previous errors
                validated_state['error'] = None
                
                logger.info("Response generation completed successfully")
                return validated_state
                
            except Exception as e:
                logger.error(f"OpenAI API call failed: {e}")
                validated_state['error'] = f"Response generation failed: {str(e)}"
                return validated_state
                
        except Exception as e:
            logger.error(f"Reasoning node execution failed: {e}")
            # Ensure we return a valid state even on error
            try:
                error_state = GraphStateManager.validate_state(state)
                error_state['error'] = f"Reasoning node failed: {str(e)}"
                return error_state
            except:
                # Last resort - return minimal valid state
                return {
                    'question': state.get('question', ''),
                    'retrieved_docs': state.get('retrieved_docs', []),
                    'context': '',
                    'answer': '',
                    'sources': [],
                    'session_id': state.get('session_id', ''),
                    'error': f"Critical reasoning failure: {str(e)}"
                }
    
    def _format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into context string.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant documents found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Unknown')
            score = doc.metadata.get('score', 0.0)
            chunk_id = doc.metadata.get('chunk_id', 0)
            
            # Truncate very long content
            content = doc.page_content
            if len(content) > 1000:
                content = content[:1000] + "..."
            
            context_part = f"[Document {i}] (Source: {source}, Chunk: {chunk_id}, Score: {score:.3f})\n{content}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    @retry_openai
    def _generate_response(self, question: str, context: str, documents: List[Document]) -> str:
        """Generate response using OpenAI GPT.
        
        Args:
            question: User question
            context: Formatted context from retrieved documents
            documents: Original retrieved documents for citation
            
        Returns:
            Generated response
        """
        # Create system prompt
        system_prompt = """You are a helpful AI assistant that answers questions based on provided context documents. 

Instructions:
1. Use ONLY the information provided in the context documents to answer questions
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Always cite your sources by referencing the document numbers (e.g., "According to Document 1...")
4. Be concise but comprehensive in your answers
5. If multiple documents contain relevant information, synthesize the information appropriately
6. Maintain a professional and helpful tone

Context Documents:
{context}"""

        # Create user prompt
        user_prompt = f"Question: {question}\n\nPlease provide a comprehensive answer based on the context documents above."
        
        try:
            # Make API call to OpenAI
            response: ChatCompletion = self.client.chat.completions.create(
                model=self.settings.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt.format(context=context)},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.1,  # Low temperature for more consistent responses
                top_p=0.9
            )
            
            if not response.choices or not response.choices[0].message.content:
                raise ExternalAPIError(
                    "OpenAI returned empty response",
                    service="openai"
                )
            
            answer = response.choices[0].message.content.strip()
            
            # Add source citations if not already present
            if documents and not any(f"Document {i}" in answer for i in range(1, len(documents) + 1)):
                sources_text = self._format_source_citations(documents)
                answer += f"\n\nSources: {sources_text}"
            
            logger.debug(f"Generated response length: {len(answer)} characters")
            return answer
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            
            # Convert to standardized exception and re-raise for retry logic
            standardized_error = handle_external_api_error(e, "openai")
            
            # If this is the final attempt (after retries), provide fallback
            if isinstance(standardized_error, (AuthenticationError, RateLimitError)):
                # Don't provide fallback for auth/rate limit errors
                raise standardized_error
            
            # For other errors, we might want to provide a fallback response
            # But since this method is decorated with retry, let the retry logic handle it first
            raise standardized_error
    
    def _extract_sources(self, documents: List[Document]) -> List[str]:
        """Extract unique source references from documents.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            List of unique source references
        """
        sources = []
        seen_sources = set()
        
        for doc in documents:
            source = doc.metadata.get('source', 'Unknown')
            if source not in seen_sources:
                sources.append(source)
                seen_sources.add(source)
        
        return sources
    
    def _format_source_citations(self, documents: List[Document]) -> str:
        """Format source citations for the response.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted source citations string
        """
        sources = self._extract_sources(documents)
        if not sources:
            return "No sources available"
        
        return ", ".join(sources)


def create_reasoning_node(openai_client: Optional[OpenAI] = None) -> ReasoningNode:
    """Factory function to create a reasoning node.
    
    Args:
        openai_client: Optional OpenAI client instance
        
    Returns:
        Configured reasoning node
    """
    return ReasoningNode(openai_client=openai_client)


# LangGraph node function
def reasoning_node(state: GraphState) -> GraphState:
    """LangGraph node function for response generation.
    
    This function creates a new ReasoningNode instance and executes it.
    Use this function when defining LangGraph workflows.
    
    Args:
        state: Current graph state with retrieved documents
        
    Returns:
        Updated graph state with generated answer
    """
    node = ReasoningNode()
    return node(state)