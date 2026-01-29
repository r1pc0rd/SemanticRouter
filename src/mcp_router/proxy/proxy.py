"""
Tool call proxy for forwarding tool calls to upstream MCP servers.

This module handles proxying tool calls from clients to the appropriate
upstream MCP server based on the tool's namespace.
"""
import asyncio
import logging
from typing import Optional

from mcp_router.core.models import ToolCallResult, ContentItem
from mcp_router.core.namespace import parse_tool_namespace, match_upstream_by_prefix
from mcp_router.core.config import RouterConfig, UpstreamConfig
from mcp_router.discovery.upstream import UpstreamConnection


logger = logging.getLogger(__name__)


class ToolCallProxy:
    """
    Proxies tool calls to upstream MCP servers.
    
    Handles:
    - Parsing tool namespace to identify upstream
    - Forwarding tool calls to correct upstream
    - Returning upstream responses without modification
    - Error handling for invalid namespaces and upstream failures
    - Connection reuse
    """
    
    def __init__(
        self,
        upstreams: dict[str, UpstreamConnection],
        upstream_configs: dict[str, UpstreamConfig]
    ) -> None:
        """
        Initialize tool call proxy.
        
        Args:
            upstreams: Dictionary of upstream connections (upstream_id -> connection)
            upstream_configs: Dictionary of upstream configurations (upstream_id -> config)
        """
        self.upstreams = upstreams
        self.upstream_configs = upstream_configs
    
    async def call_tool(
        self,
        namespaced_tool_name: str,
        arguments: dict,
        timeout: float = 30.0
    ) -> ToolCallResult:
        """
        Execute a tool call by proxying to the appropriate upstream.
        
        Args:
            namespaced_tool_name: Full tool name with namespace (e.g., "browser.navigate")
            arguments: Tool arguments to pass to upstream
            timeout: Timeout in seconds (default: 30.0)
            
        Returns:
            ToolCallResult with content from upstream
            
        Raises:
            ValueError: If tool namespace is invalid
            RuntimeError: If upstream is not found or connection fails
            asyncio.TimeoutError: If tool call exceeds timeout
        """
        # Parse namespace to get prefix and original tool name
        try:
            prefix, original_tool_name = parse_tool_namespace(namespaced_tool_name)
        except ValueError as e:
            logger.error(f"Invalid tool namespace '{namespaced_tool_name}': {e}")
            raise ValueError(f"Invalid tool namespace: {e}") from e
        
        # Find upstream by prefix
        upstream_id = match_upstream_by_prefix(prefix, self.upstream_configs)
        
        if upstream_id is None:
            error_msg = (
                f"No upstream found for prefix '{prefix}' in tool '{namespaced_tool_name}'"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Get upstream connection
        upstream = self.upstreams.get(upstream_id)
        
        if upstream is None:
            error_msg = (
                f"Upstream '{upstream_id}' not connected for tool '{namespaced_tool_name}'"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Check if upstream is connected
        if not upstream.is_connected:
            error_msg = (
                f"Upstream '{upstream_id}' is not connected for tool '{namespaced_tool_name}'"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Forward tool call to upstream with timeout
        try:
            logger.info(
                f"Forwarding tool call '{original_tool_name}' to upstream '{upstream_id}'"
            )
            
            # Execute with timeout
            result_dict = await asyncio.wait_for(
                upstream.call_tool(original_tool_name, arguments),
                timeout=timeout
            )
            
            # Convert dict result to ToolCallResult
            content_items = []
            for item_dict in result_dict.get('content', []):
                content_item = ContentItem(
                    type=item_dict['type'],
                    text=item_dict.get('text'),
                    data=item_dict.get('data'),
                    mime_type=item_dict.get('mimeType'),
                    uri=item_dict.get('uri')
                )
                content_items.append(content_item)
            
            result = ToolCallResult(
                content=content_items,
                is_error=result_dict.get('isError', False)
            )
            
            logger.info(
                f"Tool call '{original_tool_name}' completed on upstream '{upstream_id}' "
                f"(error={result.is_error})"
            )
            
            return result
            
        except asyncio.TimeoutError:
            error_msg = (
                f"Tool call '{original_tool_name}' on upstream '{upstream_id}' "
                f"timed out after {timeout} seconds"
            )
            logger.error(error_msg)
            raise asyncio.TimeoutError(error_msg)
            
        except RuntimeError as e:
            # Forward upstream errors
            logger.error(
                f"Tool call '{original_tool_name}' failed on upstream '{upstream_id}': {e}"
            )
            raise RuntimeError(f"Upstream error: {e}") from e
            
        except Exception as e:
            # Catch any other errors
            error_msg = (
                f"Unexpected error calling tool '{original_tool_name}' "
                f"on upstream '{upstream_id}': {e}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
