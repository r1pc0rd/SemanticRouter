"""
Default tool subset selection for tools/list.

This module provides functions to select a diverse subset of tools
for the standard tools/list response.
"""
from mcp_router.core.models import ToolMetadata


def select_default_tool_subset(tools: list[ToolMetadata], max_tools: int = 20) -> list[ToolMetadata]:
    """
    Select a diverse subset of tools with coverage across upstreams.
    
    Strategy:
    1. Group tools by upstream
    2. Select proportionally from each upstream
    3. Within each upstream, select alphabetically first tools
    4. Return up to max_tools tools
    
    This ensures that the default tool list provides representation from
    all upstream servers rather than being dominated by a single upstream.
    
    Args:
        tools: All available tools
        max_tools: Maximum number of tools to return (default: 20)
        
    Returns:
        List of up to max_tools tools with diverse upstream coverage
        
    Examples:
        >>> tools = [
        ...     ToolMetadata(name='a.tool1', original_name='tool1', ...),
        ...     ToolMetadata(name='a.tool2', original_name='tool2', ...),
        ...     ToolMetadata(name='b.tool1', original_name='tool1', ...),
        ... ]
        >>> subset = select_default_tool_subset(tools, max_tools=2)
        >>> len(subset) == 2
        True
    """
    if not tools:
        return []
    
    # Group tools by upstream
    tools_by_upstream: dict[str, list[ToolMetadata]] = {}
    
    for tool in tools:
        if tool.upstream_id not in tools_by_upstream:
            tools_by_upstream[tool.upstream_id] = []
        tools_by_upstream[tool.upstream_id].append(tool)
    
    # Sort tools within each upstream alphabetically by original name
    for upstream_id in tools_by_upstream:
        tools_by_upstream[upstream_id].sort(key=lambda t: t.original_name)
    
    # Calculate how many tools to take from each upstream
    num_upstreams = len(tools_by_upstream)
    tools_per_upstream = max(1, max_tools // num_upstreams)
    
    # Select tools proportionally from each upstream
    default_tools: list[ToolMetadata] = []
    upstream_ids = sorted(tools_by_upstream.keys())  # Sort for deterministic order
    
    for upstream_id in upstream_ids:
        upstream_tools = tools_by_upstream[upstream_id]
        # Take up to tools_per_upstream from this upstream
        selected = upstream_tools[:tools_per_upstream]
        default_tools.extend(selected)
    
    # If we haven't reached max_tools yet, add more tools round-robin
    if len(default_tools) < max_tools:
        # Track how many we've taken from each upstream
        taken_counts = {uid: tools_per_upstream for uid in upstream_ids}
        
        # Round-robin through upstreams to fill remaining slots
        while len(default_tools) < max_tools:
            added_any = False
            
            for upstream_id in upstream_ids:
                if len(default_tools) >= max_tools:
                    break
                
                upstream_tools = tools_by_upstream[upstream_id]
                taken = taken_counts[upstream_id]
                
                # If this upstream has more tools available
                if taken < len(upstream_tools):
                    default_tools.append(upstream_tools[taken])
                    taken_counts[upstream_id] += 1
                    added_any = True
            
            # If we couldn't add any more tools, we're done
            if not added_any:
                break
    
    # Return exactly max_tools (or fewer if not enough tools available)
    return default_tools[:max_tools]
