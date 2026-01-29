"""
Discovery module for connecting to upstream MCP servers.

This module handles tool discovery from multiple upstream MCP servers.
"""
from mcp_router.discovery.manager import ToolDiscoveryManager
from mcp_router.discovery.upstream import UpstreamConnection

__all__ = ['ToolDiscoveryManager', 'UpstreamConnection']
