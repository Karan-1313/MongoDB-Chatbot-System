"""LangGraph workflow orchestration for MongoDB chatbot."""

import logging
import time
from typing import Dict, Any, Optional, Callable
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import GraphState, GraphStateManager
from ..nodes.retrieval import retrieval_node
from ..nodes.reasoning import reasoning_node
from ..core.monitoring import WorkflowMonitor

logger = logging.getLogger(__name__)


class ChatbotWorkflow:
    """LangGraph workflow orchestrator for the chatbot system."""
    
    def __init__(self, checkpointer: Optional[MemorySaver] = None):
        """Initialize the chatbot workflow.
        
        Args:
            checkpointer: Optional checkpointer for state persistence
        """
        self.checkpointer = checkpointer or MemorySaver()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow.
        
        Returns:
            Compiled StateGraph instance
        """
        # Create the state graph
        workflow = StateGraph(GraphState)
        
        # Add nodes to the graph
        workflow.add_node("retrieval", self._safe_retrieval_node)
        workflow.add_node("reasoning", self._safe_reasoning_node)
        
        # Define the workflow edges
        workflow.set_entry_point("retrieval")
        workflow.add_edge("retrieval", "reasoning")
        workflow.add_edge("reasoning", END)
        
        # Add conditional edges for error handling
        workflow.add_conditional_edges(
            "retrieval",
            self._should_continue_after_retrieval,
            {
                "continue": "reasoning",
                "end": END
            }
        )
        
        # Compile the graph
        compiled_graph = workflow.compile(checkpointer=self.checkpointer)
        
        logger.info("LangGraph workflow compiled successfully")
        return compiled_graph
    
    def _safe_retrieval_node(self, state: GraphState) -> GraphState:
        """Wrapper for retrieval node with error handling and retry logic.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated graph state
        """
        max_retries = 2
        retry_count = 0
        start_time = time.time()
        
        while retry_count <= max_retries:
            try:
                logger.debug(f"Executing retrieval node (attempt {retry_count + 1})")
                node_start = time.time()
                result = retrieval_node(state)
                node_duration = time.time() - node_start
                
                # Check if retrieval was successful
                if result.get('error'):
                    logger.warning(f"Retrieval node returned error: {result['error']}")
                    if retry_count < max_retries:
                        retry_count += 1
                        logger.info(f"Retrying retrieval (attempt {retry_count + 1})")
                        continue
                    else:
                        logger.error("Max retries reached for retrieval node")
                        # Log node execution with error
                        if hasattr(state, 'session_id'):
                            monitor = WorkflowMonitor("chatbot", state.get('session_id', 'unknown'))
                            monitor.node_executed("retrieval", node_duration, False, result['error'])
                        return result
                
                logger.info(f"Retrieval successful: {len(result.get('retrieved_docs', []))} documents retrieved")
                
                # Log successful node execution
                if hasattr(state, 'session_id'):
                    monitor = WorkflowMonitor("chatbot", state.get('session_id', 'unknown'))
                    monitor.node_executed("retrieval", node_duration, True)
                
                return result
                
            except Exception as e:
                node_duration = time.time() - node_start if 'node_start' in locals() else 0
                logger.error(f"Retrieval node exception (attempt {retry_count + 1}): {e}")
                
                if retry_count < max_retries:
                    retry_count += 1
                    continue
                else:
                    # Log failed node execution
                    if hasattr(state, 'session_id'):
                        monitor = WorkflowMonitor("chatbot", state.get('session_id', 'unknown'))
                        monitor.node_executed("retrieval", node_duration, False, str(e))
                    
                    # Return error state
                    error_state = GraphStateManager.validate_state(state)
                    error_state['error'] = f"Retrieval failed after {max_retries + 1} attempts: {str(e)}"
                    return error_state
        
        # This should never be reached, but just in case
        error_state = GraphStateManager.validate_state(state)
        error_state['error'] = "Unexpected error in retrieval retry logic"
        return error_state
    
    def _safe_reasoning_node(self, state: GraphState) -> GraphState:
        """Wrapper for reasoning node with error handling and retry logic.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated graph state
        """
        max_retries = 2
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                logger.debug(f"Executing reasoning node (attempt {retry_count + 1})")
                node_start = time.time()
                result = reasoning_node(state)
                node_duration = time.time() - node_start
                
                # Check if reasoning was successful
                if result.get('error'):
                    logger.warning(f"Reasoning node returned error: {result['error']}")
                    if retry_count < max_retries:
                        retry_count += 1
                        logger.info(f"Retrying reasoning (attempt {retry_count + 1})")
                        continue
                    else:
                        logger.error("Max retries reached for reasoning node")
                        # Log node execution with error
                        if hasattr(state, 'session_id'):
                            monitor = WorkflowMonitor("chatbot", state.get('session_id', 'unknown'))
                            monitor.node_executed("reasoning", node_duration, False, result['error'])
                        return result
                
                logger.info("Reasoning successful: response generated")
                
                # Log successful node execution
                if hasattr(state, 'session_id'):
                    monitor = WorkflowMonitor("chatbot", state.get('session_id', 'unknown'))
                    monitor.node_executed("reasoning", node_duration, True)
                
                return result
                
            except Exception as e:
                node_duration = time.time() - node_start if 'node_start' in locals() else 0
                logger.error(f"Reasoning node exception (attempt {retry_count + 1}): {e}")
                
                if retry_count < max_retries:
                    retry_count += 1
                    continue
                else:
                    # Log failed node execution
                    if hasattr(state, 'session_id'):
                        monitor = WorkflowMonitor("chatbot", state.get('session_id', 'unknown'))
                        monitor.node_executed("reasoning", node_duration, False, str(e))
                    
                    # Return error state
                    error_state = GraphStateManager.validate_state(state)
                    error_state['error'] = f"Reasoning failed after {max_retries + 1} attempts: {str(e)}"
                    return error_state
        
        # This should never be reached, but just in case
        error_state = GraphStateManager.validate_state(state)
        error_state['error'] = "Unexpected error in reasoning retry logic"
        return error_state
    
    def _should_continue_after_retrieval(self, state: GraphState) -> str:
        """Determine whether to continue to reasoning after retrieval.
        
        Args:
            state: Current graph state
            
        Returns:
            Next step: "continue" or "end"
        """
        # Check for critical errors that should stop the workflow
        error = state.get('error')
        if error and 'Critical' in error:
            logger.error(f"Critical error detected, stopping workflow: {error}")
            return "end"
        
        # Check if we have retrieved documents or if we should continue anyway
        retrieved_docs = state.get('retrieved_docs', [])
        if not retrieved_docs and not error:
            logger.warning("No documents retrieved, but no error - continuing to reasoning")
        
        return "continue"
    
    async def arun(self, question: str, session_id: Optional[str] = None, 
                   config: Optional[Dict[str, Any]] = None) -> GraphState:
        """Run the workflow asynchronously.
        
        Args:
            question: User question
            session_id: Optional session ID for conversation tracking
            config: Optional configuration for the workflow
            
        Returns:
            Final graph state with answer
        """
        from ..core.exceptions import WorkflowError, ValidationError
        
        # Initialize workflow monitor
        monitor = WorkflowMonitor("chatbot", session_id or "unknown")
        monitor.start()
        
        try:
            # Validate inputs
            if not question or not question.strip():
                raise ValidationError("Question cannot be empty", field="question", value=question)
            
            # Create initial state
            try:
                initial_state = GraphStateManager.create_initial_state(question, session_id)
            except Exception as e:
                raise WorkflowError(f"Failed to create initial state: {e}")
            
            # Set up configuration
            workflow_config = config or {}
            if session_id:
                workflow_config["configurable"] = {"thread_id": session_id}
            
            logger.info(f"Starting workflow for session {session_id}")
            
            # Run the workflow with timeout
            try:
                final_state = await self.graph.ainvoke(initial_state, config=workflow_config)
            except Exception as e:
                logger.error(f"Graph execution failed: {e}")
                monitor.complete(False, str(e))
                raise WorkflowError(f"Graph execution failed: {e}")
            
            # Validate final state
            if not final_state:
                monitor.complete(False, "Workflow returned empty state")
                raise WorkflowError("Workflow returned empty state")
            
            # Check for errors in final state
            error = final_state.get('error')
            if error:
                logger.warning(f"Workflow completed with error: {error}")
                monitor.complete(False, error)
            else:
                monitor.complete(True)
            
            logger.info(f"Workflow completed for session {session_id}")
            return final_state
            
        except (ValidationError, WorkflowError):
            # Re-raise our custom exceptions
            monitor.complete(False, str(e) if 'e' in locals() else "Validation or workflow error")
            raise
        except Exception as e:
            logger.error(f"Unexpected workflow execution error: {e}", exc_info=True)
            monitor.complete(False, str(e))
            raise WorkflowError(f"Unexpected workflow error: {e}")
    
    def run(self, question: str, session_id: Optional[str] = None, 
            config: Optional[Dict[str, Any]] = None) -> GraphState:
        """Run the workflow synchronously.
        
        Args:
            question: User question
            session_id: Optional session ID for conversation tracking
            config: Optional configuration for the workflow
            
        Returns:
            Final graph state with answer
        """
        try:
            # Create initial state
            initial_state = GraphStateManager.create_initial_state(question, session_id)
            
            # Set up configuration
            workflow_config = config or {}
            if session_id:
                workflow_config["configurable"] = {"thread_id": session_id}
            
            logger.info(f"Starting workflow for session {session_id}")
            
            # Run the workflow
            final_state = self.graph.invoke(initial_state, config=workflow_config)
            
            logger.info(f"Workflow completed for session {session_id}")
            return final_state
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            # Return error state
            error_state = GraphStateManager.create_initial_state(question, session_id)
            error_state['error'] = f"Workflow execution failed: {str(e)}"
            return error_state
    
    def get_state_history(self, session_id: str) -> list:
        """Get the state history for a session.
        
        Args:
            session_id: Session ID to get history for
            
        Returns:
            List of state snapshots
        """
        try:
            config = {"configurable": {"thread_id": session_id}}
            history = list(self.graph.get_state_history(config))
            return history
        except Exception as e:
            logger.error(f"Failed to get state history: {e}")
            return []
    
    def clear_session(self, session_id: str) -> bool:
        """Clear the state for a specific session.
        
        Args:
            session_id: Session ID to clear
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # This would depend on the checkpointer implementation
            # For MemorySaver, we might need to implement custom clearing
            logger.info(f"Session {session_id} cleared (implementation depends on checkpointer)")
            return True
        except Exception as e:
            logger.error(f"Failed to clear session {session_id}: {e}")
            return False


def create_chatbot_workflow(checkpointer: Optional[MemorySaver] = None) -> ChatbotWorkflow:
    """Factory function to create a chatbot workflow.
    
    Args:
        checkpointer: Optional checkpointer for state persistence
        
    Returns:
        Configured ChatbotWorkflow instance
    """
    return ChatbotWorkflow(checkpointer=checkpointer)


# Global workflow instance for easy access
_workflow_instance: Optional[ChatbotWorkflow] = None


def get_workflow() -> ChatbotWorkflow:
    """Get the global workflow instance.
    
    Returns:
        Global ChatbotWorkflow instance
    """
    global _workflow_instance
    if _workflow_instance is None:
        _workflow_instance = create_chatbot_workflow()
    return _workflow_instance


def reset_workflow() -> None:
    """Reset the global workflow instance."""
    global _workflow_instance
    _workflow_instance = None