"""
Embedding module for semantic search.

This module provides embedding generation using sentence-transformers.
"""
from mcp_router.embedding.engine import EmbeddingEngine
from mcp_router.embedding.utils import generate_tool_embedding_text

__all__ = ['EmbeddingEngine', 'generate_tool_embedding_text']
