"""
Semantic search engine for tool discovery.

This module provides semantic search capabilities over tool catalogs using
embedding-based similarity matching.
"""
from typing import Optional
import logging

from mcp_router.core.models import ToolMetadata, SearchResult
from mcp_router.embedding.engine import EmbeddingEngine
from mcp_router.search.sanitize import combine_query_and_context
from mcp_router.search.similarity import compute_similarities


logger = logging.getLogger(__name__)


class SemanticSearchEngine:
    """
    Performs semantic search over tool catalog using embeddings.
    
    Uses cosine similarity to rank tools by relevance to a query.
    """
    
    def __init__(self, embedding_engine: EmbeddingEngine) -> None:
        """
        Initialize semantic search engine.
        
        Args:
            embedding_engine: Initialized embedding engine for generating query embeddings
        """
        self.embedding_engine = embedding_engine
        self._tools: list[ToolMetadata] = []
    
    def set_tools(self, tools: list[ToolMetadata]) -> None:
        """
        Set the tool catalog to search over.
        
        All tools must have embeddings already generated.
        
        Args:
            tools: List of tools with embeddings
            
        Raises:
            ValueError: If any tool is missing an embedding
        """
        # Validate all tools have embeddings
        for tool in tools:
            if tool.embedding is None:
                raise ValueError(
                    f"Tool '{tool.name}' is missing embedding. "
                    f"Generate embeddings before setting tools."
                )
        
        self._tools = tools
        logger.info(f"Semantic search engine loaded with {len(tools)} tools")
    
    def add_tools(self, tools: list[ToolMetadata]) -> None:
        """
        Add tools to the existing catalog.
        
        All tools must have embeddings already generated.
        
        Args:
            tools: List of tools with embeddings to add
            
        Raises:
            ValueError: If any tool is missing an embedding or has duplicate name
        """
        if not tools:
            return
        
        # Validate all tools have embeddings
        for tool in tools:
            if tool.embedding is None:
                raise ValueError(
                    f"Tool '{tool.name}' is missing embedding. "
                    f"Generate embeddings before adding tools."
                )
        
        # Check for duplicate tool names
        existing_names = {tool.name for tool in self._tools}
        for tool in tools:
            if tool.name in existing_names:
                raise ValueError(
                    f"Tool '{tool.name}' already exists in catalog. "
                    f"Cannot add duplicate tools."
                )
        
        # Add tools to catalog
        self._tools.extend(tools)
        logger.info(
            f"Added {len(tools)} tools to search engine. "
            f"Total tools: {len(self._tools)}"
        )
    
    def remove_tools(self, namespace: str) -> None:
        """
        Remove all tools with the specified namespace.
        
        Args:
            namespace: Namespace prefix to remove (e.g., 'playwright')
        """
        if not namespace:
            raise ValueError("Namespace cannot be empty")
        
        # Find tools to remove
        tools_to_remove = [
            tool for tool in self._tools
            if tool.name.startswith(f"{namespace}.")
        ]
        
        if not tools_to_remove:
            logger.info(f"No tools found with namespace '{namespace}'")
            return
        
        # Remove tools from catalog
        self._tools = [
            tool for tool in self._tools
            if not tool.name.startswith(f"{namespace}.")
        ]
        
        logger.info(
            f"Removed {len(tools_to_remove)} tools with namespace '{namespace}'. "
            f"Remaining tools: {len(self._tools)}"
        )
    
    async def search_tools(
        self,
        query: str,
        context: Optional[list[str]] = None,
        top_k: int = 10
    ) -> list[SearchResult]:
        """
        Search for tools using semantic similarity.
        
        Args:
            query: User query describing desired functionality
            context: Optional context strings to enhance query
            top_k: Number of results to return (default: 10)
            
        Returns:
            Ranked tools with similarity scores (descending order)
            
        Raises:
            ValueError: If query is empty or missing
            RuntimeError: If no tools are loaded
        """
        # Validate query
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        # Check if tools are loaded
        if not self._tools:
            raise RuntimeError("No tools loaded. Call set_tools() first.")
        
        # Combine query and context
        combined_text = combine_query_and_context(query, context)
        
        logger.info(
            f"Searching for tools with query: '{query}' "
            f"(context: {len(context) if context else 0} items)"
        )
        
        # Generate query embedding
        query_embedding = self.embedding_engine.generate_embedding(combined_text)
        
        # Compute similarities with all tool embeddings
        tool_embeddings = [tool.embedding for tool in self._tools]
        similarities = compute_similarities(query_embedding, tool_embeddings)
        
        # Create search results with tools and scores
        results = [
            SearchResult(tool=tool, similarity=score)
            for tool, score in zip(self._tools, similarities)
        ]
        
        # Sort by similarity (descending)
        results.sort(key=lambda r: r.similarity, reverse=True)
        
        # Return top-k results
        top_results = results[:top_k]
        
        logger.info(
            f"Search complete. Top result: '{top_results[0].tool.name}' "
            f"(similarity: {top_results[0].similarity:.4f})"
        )
        
        return top_results
    
    def get_tool_count(self) -> int:
        """Get the number of tools in the catalog."""
        return len(self._tools)
