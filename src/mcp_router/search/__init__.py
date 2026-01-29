"""
Search module for semantic tool discovery.

This module provides semantic search capabilities using embeddings and cosine similarity.
"""
from mcp_router.search.engine import SemanticSearchEngine
from mcp_router.search.similarity import cosine_similarity, compute_similarities
from mcp_router.search.sanitize import sanitize_query, combine_query_and_context
from mcp_router.search.default_subset import select_default_tool_subset

__all__ = [
    'SemanticSearchEngine',
    'cosine_similarity',
    'compute_similarities',
    'sanitize_query',
    'combine_query_and_context',
    'select_default_tool_subset',
]
