"""
Upstream MCP server connection management.

This module handles connections to individual upstream MCP servers.
"""
from typing import Optional
import asyncio
import logging

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from mcp_router.core.config import UpstreamConfig
from mcp_router.core.models import ToolMetadata, JSONSchema


logger = logging.getLogger(__name__)


class UpstreamConnection:
    """
    Manages connection to a single upstream MCP server.
    
    Handles establishing connection, fetching tools, and maintaining
    the connection for tool calls.
    """
    
    def __init__(self, upstream_id: str, config: UpstreamConfig) -> None:
        """
        Initialize upstream connection.
        
        Args:
            upstream_id: Unique identifier for this upstream
            config: Configuration for the upstream server
        """
        self.upstream_id = upstream_id
        self.config = config
        self.tools: list[ToolMetadata] = []
        self._session: Optional[ClientSession] = None
        self._read_stream = None
        self._write_stream = None
        self._stdio_context = None
    
    async def connect(self) -> None:
        """
        Establish connection to the upstream MCP server.
        
        Currently only supports stdio transport.
        
        Raises:
            RuntimeError: If connection fails
            ValueError: If transport type is not supported
        """
        if self.config.transport != 'stdio':
            raise ValueError(
                f"Unsupported transport type: {self.config.transport}. "
                f"Only 'stdio' is currently supported."
            )
        
        if not self.config.command:
            raise ValueError(f"Missing 'command' for stdio transport in upstream '{self.upstream_id}'")
        
        try:
            # Create server parameters
            server_params = StdioServerParameters(
                command=self.config.command,
                args=self.config.args or [],
                env=None  # Use current environment
            )
            
            # Establish stdio connection
            self._stdio_context = stdio_client(server_params)
            self._read_stream, self._write_stream = await self._stdio_context.__aenter__()
            
            # Create client session
            self._session = ClientSession(self._read_stream, self._write_stream)
            await self._session.__aenter__()
            
            # Initialize the connection
            await self._session.initialize()
            
            logger.info(f"Connected to upstream '{self.upstream_id}'")
            
        except Exception as e:
            logger.error(f"Failed to connect to upstream '{self.upstream_id}': {e}")
            raise RuntimeError(f"Connection failed for upstream '{self.upstream_id}': {e}") from e
    
    async def fetch_tools(self) -> list[ToolMetadata]:
        """
        Fetch tool catalog from the upstream server.
        
        Returns:
            List of tool metadata objects
            
        Raises:
            RuntimeError: If not connected or fetch fails
        """
        if self._session is None:
            raise RuntimeError(f"Not connected to upstream '{self.upstream_id}'")
        
        try:
            # Call tools/list on the upstream
            response = await self._session.list_tools()
            
            # Convert to our ToolMetadata format
            tools = []
            for tool in response.tools:
                # Convert input schema using from_dict to handle all fields
                input_schema = JSONSchema.from_dict(tool.inputSchema)
                
                # Create tool metadata
                tool_metadata = ToolMetadata(
                    name=f"{self.config.semantic_prefix or self.upstream_id}.{tool.name}",
                    original_name=tool.name,
                    description=tool.description or "",
                    input_schema=input_schema,
                    upstream_id=self.upstream_id,
                    category_description=self.config.category_description
                )
                
                tools.append(tool_metadata)
            
            self.tools = tools
            logger.info(f"Fetched {len(tools)} tools from upstream '{self.upstream_id}'")
            return tools
            
        except Exception as e:
            logger.error(f"Failed to fetch tools from upstream '{self.upstream_id}': {e}")
            raise RuntimeError(f"Tool fetch failed for upstream '{self.upstream_id}': {e}") from e
    
    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """
        Execute a tool call on the upstream server.
        
        Args:
            tool_name: Original tool name (without namespace prefix)
            arguments: Tool arguments
            
        Returns:
            Tool execution result
            
        Raises:
            RuntimeError: If not connected or call fails
        """
        if self._session is None:
            raise RuntimeError(f"Not connected to upstream '{self.upstream_id}'")
        
        try:
            # Call the tool on the upstream
            result = await self._session.call_tool(tool_name, arguments)
            
            # Convert result to dict format
            return {
                'content': [
                    {
                        'type': item.type,
                        'text': getattr(item, 'text', None),
                        'data': getattr(item, 'data', None),
                        'mimeType': getattr(item, 'mimeType', None),
                    }
                    for item in result.content
                ],
                'isError': result.isError if hasattr(result, 'isError') else False
            }
            
        except Exception as e:
            logger.error(f"Tool call failed for '{tool_name}' on upstream '{self.upstream_id}': {e}")
            raise RuntimeError(f"Tool call failed: {e}") from e
    
    async def disconnect(self) -> None:
        """Close the connection to the upstream server."""
        try:
            if self._session:
                await self._session.__aexit__(None, None, None)
                self._session = None
            
            if self._stdio_context:
                await self._stdio_context.__aexit__(None, None, None)
                self._stdio_context = None
            
            logger.info(f"Disconnected from upstream '{self.upstream_id}'")
            
        except Exception as e:
            logger.warning(f"Error during disconnect from upstream '{self.upstream_id}': {e}")
    
    @property
    def is_connected(self) -> bool:
        """Check if currently connected to the upstream."""
        return self._session is not None
