"""
Query sanitization utilities for semantic search.

This module provides functions to clean and prepare queries for embedding generation.
"""
import re
from typing import Optional


def sanitize_query(query: str) -> str:
    """
    Remove special characters that could interfere with embedding.
    
    Keeps alphanumeric characters, spaces, and common punctuation (.,!?-).
    
    Args:
        query: Raw query string
        
    Returns:
        Sanitized query string
        
    Examples:
        >>> sanitize_query("test@#$%web page")
        'test web page'
        
        >>> sanitize_query("navigate   to    URL")
        'navigate to URL'
        
        >>> sanitize_query("click! button?")
        'click! button?'
    """
    # Remove special characters (keep alphanumeric, spaces, and basic punctuation)
    sanitized = re.sub(r'[^\w\s.,!?-]', ' ', query)
    
    # Collapse multiple spaces
    sanitized = re.sub(r'\s+', ' ', sanitized)
    
    return sanitized.strip()


def combine_query_and_context(query: str, context: Optional[list[str]] = None) -> str:
    """
    Combine query and context for embedding generation.
    
    Sanitizes both query and context strings, then combines them with a space.
    
    Args:
        query: User query
        context: Optional context strings
        
    Returns:
        Combined text for embedding
        
    Examples:
        >>> combine_query_and_context("test web page")
        'test web page'
        
        >>> combine_query_and_context("test", ["browser automation", "testing"])
        'test browser automation testing'
        
        >>> combine_query_and_context("test@#$", ["context!"])
        'test context'
    """
    sanitized_query = sanitize_query(query)
    
    if not context:
        return sanitized_query
    
    # Sanitize and combine all context strings
    sanitized_context = ' '.join(sanitize_query(c) for c in context if c)
    
    # Combine query and context
    if sanitized_context:
        return f"{sanitized_query} {sanitized_context}"
    
    return sanitized_query
