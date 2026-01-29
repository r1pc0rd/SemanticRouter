"""
Utility functions for embedding generation.

This module provides helper functions for generating embedding text from tool metadata.
"""
from mcp_router.core.models import ToolMetadata


def generate_tool_embedding_text(tool: ToolMetadata) -> str:
    """
    Generate text for embedding from tool metadata.
    
    Combines tool name, description, category description, and parameter names
    into a single text string optimized for semantic search.
    
    Args:
        tool: Tool metadata to generate embedding text from
        
    Returns:
        Combined text string for embedding
        
    Example:
        >>> tool = ToolMetadata(
        ...     name='browser.navigate',
        ...     original_name='navigate',
        ...     description='Navigate to a URL',
        ...     input_schema=JSONSchema(
        ...         type='object',
        ...         properties={'url': {'type': 'string'}}
        ...     ),
        ...     upstream_id='playwright',
        ...     category_description='Web browser automation'
        ... )
        >>> generate_tool_embedding_text(tool)
        'navigate | Navigate to a URL | Web browser automation | Parameters: url'
    """
    parts: list[str] = []
    
    # Include tool name (without namespace prefix)
    parts.append(tool.original_name)
    
    # Include description
    if tool.description:
        parts.append(tool.description)
    
    # Include category description if available
    if tool.category_description:
        parts.append(tool.category_description)
    
    # Include parameter names from input schema
    if tool.input_schema.properties:
        param_names = list(tool.input_schema.properties.keys())
        if param_names:
            parts.append(f"Parameters: {', '.join(param_names)}")
    
    return ' | '.join(parts)
