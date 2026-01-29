"""
Tool namespace generation utilities.

This module handles the creation of namespaced tool names from upstream tools.
"""
from mcp_router.core.config import UpstreamConfig


def generate_tool_namespace(
    upstream_id: str,
    original_tool_name: str,
    upstream_config: UpstreamConfig
) -> str:
    """
    Generate namespaced tool name for a tool from an upstream MCP server.
    
    The namespace format is: {prefix}.{original_tool_name}
    
    Where prefix is:
    - upstream_config.semantic_prefix if configured
    - upstream_id otherwise
    
    Args:
        upstream_id: ID of the upstream server
        original_tool_name: Original tool name from upstream
        upstream_config: Configuration for the upstream server
        
    Returns:
        Namespaced tool name (e.g., "browser.navigate" or "playwright.navigate")
        
    Examples:
        >>> config = UpstreamConfig(transport='stdio', command='test', semantic_prefix='browser')
        >>> generate_tool_namespace('playwright', 'navigate', config)
        'browser.navigate'
        
        >>> config = UpstreamConfig(transport='stdio', command='test')
        >>> generate_tool_namespace('playwright', 'navigate', config)
        'playwright.navigate'
    """
    # Use semantic_prefix if configured, otherwise use upstream_id
    prefix = upstream_config.semantic_prefix if upstream_config.semantic_prefix else upstream_id
    
    return f"{prefix}.{original_tool_name}"


def parse_tool_namespace(namespaced_name: str) -> tuple[str, str]:
    """
    Parse a namespaced tool name into prefix and tool name.
    
    Args:
        namespaced_name: Namespaced tool name (e.g., "browser.navigate")
        
    Returns:
        Tuple of (prefix, tool_name)
        
    Raises:
        ValueError: If the name is not properly namespaced (no dot separator)
        
    Examples:
        >>> parse_tool_namespace('browser.navigate')
        ('browser', 'navigate')
        
        >>> parse_tool_namespace('playwright.click')
        ('playwright', 'click')
        
        >>> parse_tool_namespace('invalid')
        Traceback (most recent call last):
            ...
        ValueError: Tool name must be namespaced with format 'prefix.toolname'
    """
    if '.' not in namespaced_name:
        raise ValueError(
            f"Tool name must be namespaced with format 'prefix.toolname', "
            f"got: '{namespaced_name}'"
        )
    
    # Split on first dot only (tool names might contain dots)
    parts = namespaced_name.split('.', 1)
    prefix = parts[0]
    tool_name = parts[1]
    
    # Validate that both parts are non-empty and not just whitespace or dots
    # Strip whitespace and check if anything meaningful remains
    prefix_stripped = prefix.strip()
    tool_name_stripped = tool_name.strip()
    
    # Check if prefix is empty or only whitespace
    if not prefix or not prefix_stripped:
        raise ValueError(
            f"Tool name must be namespaced with format 'prefix.toolname', "
            f"got: '{namespaced_name}'"
        )
    
    # Check if tool_name is empty, only whitespace, or only dots
    if not tool_name or not tool_name_stripped or tool_name_stripped.replace('.', '') == '':
        raise ValueError(
            f"Tool name must be namespaced with format 'prefix.toolname', "
            f"got: '{namespaced_name}'"
        )
    
    return prefix, tool_name


def match_upstream_by_prefix(
    prefix: str,
    upstream_configs: dict[str, UpstreamConfig]
) -> str | None:
    """
    Find the upstream ID that matches the given prefix.
    
    The prefix can match either:
    - The upstream_id directly
    - The semantic_prefix of an upstream
    
    Args:
        prefix: Tool namespace prefix to match
        upstream_configs: Dictionary of upstream configurations
        
    Returns:
        Upstream ID if found, None otherwise
        
    Examples:
        >>> configs = {
        ...     'playwright': UpstreamConfig(
        ...         transport='stdio',
        ...         command='test',
        ...         semantic_prefix='browser'
        ...     )
        ... }
        >>> match_upstream_by_prefix('browser', configs)
        'playwright'
        >>> match_upstream_by_prefix('playwright', configs)
        'playwright'
        >>> match_upstream_by_prefix('unknown', configs)
        None
    """
    # First, check if prefix matches an upstream_id directly
    if prefix in upstream_configs:
        return prefix
    
    # Then, check if prefix matches any semantic_prefix
    for upstream_id, config in upstream_configs.items():
        if config.semantic_prefix == prefix:
            return upstream_id
    
    return None
