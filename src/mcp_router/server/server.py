"""MCP server implementation using the official mcp SDK."""

import asyncio
import logging
from typing import Any, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
)

from ..discovery.manager import ToolDiscoveryManager
from ..embedding.engine import EmbeddingEngine
from ..search.engine import SemanticSearchEngine
from ..proxy.proxy import ToolCallProxy
from ..core.models import ToolMetadata
from ..core.errors import (
    ValidationError,
    ToolNotFoundError,
    EmbeddingError,
    UpstreamError,
    format_error_response
)
from ..core.validation import validate_tool_parameters, validate_search_query
from ..core.logging import log_with_metadata

logger = logging.getLogger(__name__)


class SemanticRouterServer:
    """
    MCP server that provides semantic search capabilities for tool discovery.
    
    This server acts as an intelligent proxy between MCP clients and multiple
    upstream MCP servers, implementing semantic search to help clients discover
    relevant tools dynamically.
    """
    
    def __init__(
        self,
        discovery_manager: ToolDiscoveryManager,
        embedding_engine: EmbeddingEngine,
        search_engine: SemanticSearchEngine,
        proxy: ToolCallProxy,
    ):
        """
        Initialize the semantic router server.
        
        Args:
            discovery_manager: Manager for upstream tool discovery
            embedding_engine: Engine for generating embeddings
            search_engine: Engine for semantic search
            proxy: Proxy for forwarding tool calls
        """
        self.discovery_manager = discovery_manager
        self.embedding_engine = embedding_engine
        self.search_engine = search_engine
        self.proxy = proxy
        
        # Create MCP server instance
        self.server = Server("mcp-semantic-router")
        
        # Register handlers
        self._register_handlers()
    
    def _register_handlers(self) -> None:
        """Register MCP protocol handlers."""
        
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """
            Handle tools/list request.
            
            Returns the default subset of 20 tools selected for diversity
            across upstream servers, plus the search_tools tool.
            """
            logger.info("Handling tools/list request")
            
            # Get default tool subset (20 tools)
            default_tools = self.discovery_manager.get_default_tool_subset()
            
            # Log that default subset is being returned (Requirement 9.3)
            log_with_metadata(
                logger,
                logging.INFO,
                "Returning default tool subset",
                metadata={
                    'tool_count': len(default_tools),
                    'includes_search_tools': True
                }
            )
            
            # Convert to MCP Tool format
            mcp_tools = [self._tool_metadata_to_mcp_tool(tool) for tool in default_tools]
            
            # Add search_tools as a tool
            search_tools_tool = Tool(
                name="search_tools",
                description=(
                    "IMPORTANT: Use this tool FIRST when the user asks about a specific task or domain "
                    "(testing, issues, repositories, etc.). This returns the most relevant tools for the "
                    "user's request, reducing the number of tools you need to consider. Provide a query "
                    "describing what the user wants to do. Example queries: 'test a web page', "
                    "'create a bug report', 'check repository status'."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Query describing what the user wants to do"
                        },
                        "context": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional context strings to enhance the query"
                        }
                    },
                    "required": ["query"]
                }
            )
            mcp_tools.append(search_tools_tool)
            
            # Add load_upstream as a tool
            load_upstream_tool = Tool(
                name="load_upstream",
                description=(
                    "Load an upstream MCP server on-demand. Provide either 'upstream' (canonical name) "
                    "or 'alias' (friendly name). This allows you to dynamically load additional tools "
                    "without restarting the router."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "upstream": {
                            "type": "string",
                            "description": "Canonical upstream name (e.g., 'playwright', 'jira')"
                        },
                        "alias": {
                            "type": "string",
                            "description": "Upstream alias (e.g., 'browser', 'issue-tracker')"
                        }
                    },
                    "required": []
                }
            )
            mcp_tools.append(load_upstream_tool)
            
            # Add unload_upstream as a tool
            unload_upstream_tool = Tool(
                name="unload_upstream",
                description=(
                    "Unload an upstream MCP server. Provide 'upstream' (canonical name or alias). "
                    "This removes all tools from the upstream and closes the connection. "
                    "The operation is idempotent - unloading an already-unloaded upstream succeeds."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "upstream": {
                            "type": "string",
                            "description": "Upstream name or alias to unload (e.g., 'playwright', 'browser')"
                        }
                    },
                    "required": ["upstream"]
                }
            )
            mcp_tools.append(unload_upstream_tool)
            
            return mcp_tools
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            """
            Handle tools/call request.
            
            Routes calls to either search_tools (handled locally) or forwards
            to upstream MCP servers.
            
            Args:
                name: Tool name (e.g., "search_tools" or "playwright.navigate")
                arguments: Tool call arguments
                
            Returns:
                Tool call result content
                
            Raises:
                ValidationError: If parameters are invalid
                ToolNotFoundError: If tool is not found
                UpstreamError: If upstream call fails
            """
            logger.info(f"Handling tools/call for {name}", extra={
                "tool_name": name,
                "arguments": arguments
            })
            
            try:
                # Check if this is a search_tools call
                if name == "search_tools":
                    return await self._handle_search_tools(arguments)
                
                # Check if this is a load_upstream call
                if name == "load_upstream":
                    return await self._handle_load_upstream(arguments)
                
                # Check if this is an unload_upstream call
                if name == "unload_upstream":
                    return await self._handle_unload_upstream(arguments)
                
                # Get tool metadata for validation
                tool = self.discovery_manager.get_tool_by_name(name)
                if tool is None:
                    raise ToolNotFoundError(
                        f"Tool not found: {name}",
                        data={"tool_name": name}
                    )
                
                # Validate parameters against schema
                validate_tool_parameters(arguments, tool.input_schema)
                
                # Forward to proxy
                result = await self.proxy.call_tool(name, arguments)
                
                # Parse namespace to get upstream_id
                upstream_id = name.split('.')[0] if '.' in name else name
                
                # Log result (Requirement 9.4)
                if result.is_error:
                    log_with_metadata(
                        logger,
                        logging.ERROR,
                        f"Tool call failed: {name}",
                        metadata={
                            'tool_name': name,
                            'upstream_id': upstream_id,
                            'status': 'failure',
                            'error': result.content[0].text if result.content else "Unknown error"
                        }
                    )
                else:
                    log_with_metadata(
                        logger,
                        logging.INFO,
                        f"Tool call succeeded: {name}",
                        metadata={
                            'tool_name': name,
                            'upstream_id': upstream_id,
                            'status': 'success'
                        }
                    )
                
                # Return content
                return result.content
                
            except (ValidationError, ToolNotFoundError, UpstreamError) as e:
                # Re-raise router errors
                raise
            except Exception as e:
                # Wrap unexpected errors
                logger.error(f"Unexpected error in call_tool: {e}", exc_info=True)
                raise UpstreamError(
                    f"Tool call failed: {str(e)}",
                    data={"tool_name": name, "error": str(e)}
                )
    
    async def _handle_search_tools(self, arguments: dict[str, Any]) -> list[TextContent]:
        """
        Handle search_tools tool call.
        
        Args:
            arguments: Tool arguments containing 'query' and optional 'context'
            
        Returns:
            List of TextContent with search results
            
        Raises:
            ValidationError: If query is missing or empty
            EmbeddingError: If search fails
        """
        # Extract arguments
        query = arguments.get("query")
        context = arguments.get("context")
        
        # Validate query
        validate_search_query(query)
        
        logger.info(
            f"Executing search_tools",
            extra={
                "query": query,
                "context_items": len(context) if context else 0
            }
        )
        
        # Perform semantic search
        try:
            results = await self.search_engine.search_tools(
                query=query,
                context=context,
                top_k=10
            )
            
            # Log search_tools request (Requirement 9.2)
            top_matches = [r.tool.name for r in results[:3]] if results else []
            log_with_metadata(
                logger,
                logging.INFO,
                "search_tools request completed",
                metadata={
                    'query': query,
                    'context_length': len(context) if context else 0,
                    'top_matches': top_matches,
                    'results_count': len(results)
                }
            )
            
            # Format results as text
            result_lines = [f"Found {len(results)} relevant tools:\n"]
            
            for i, search_result in enumerate(results, 1):
                tool = search_result.tool
                similarity = search_result.similarity
                result_lines.append(
                    f"{i}. {tool.name} (similarity: {similarity:.4f})\n"
                    f"   Description: {tool.description}\n"
                )
            
            result_text = "\n".join(result_lines)
            
            return [TextContent(type="text", text=result_text)]
            
        except ValidationError:
            # Re-raise validation errors
            raise
        except Exception as e:
            logger.error(
                f"search_tools failed: {e}",
                extra={"query": query, "error": str(e)},
                exc_info=True
            )
            raise EmbeddingError(
                f"Semantic search failed: {str(e)}",
                data={"query": query, "error": str(e)}
            )
    
    async def _handle_load_upstream(self, arguments: dict[str, Any]) -> list[TextContent]:
        """
        Handle load_upstream tool call.
        
        Loads an upstream MCP server dynamically by name or alias. This allows
        clients to load additional upstreams on-demand without restarting the router.
        
        Args:
            arguments: Tool arguments containing 'upstream' or 'alias'
            
        Returns:
            List of TextContent with load result
            
        Raises:
            ValidationError: If neither upstream nor alias is provided
        """
        # Extract arguments
        upstream = arguments.get("upstream")
        alias = arguments.get("alias")
        
        # Validate that at least one is provided
        if not upstream and not alias:
            raise ValidationError(
                "Either 'upstream' or 'alias' must be provided",
                data={"arguments": arguments}
            )
        
        # Use alias if provided, otherwise use upstream
        name = alias if alias else upstream
        
        logger.info(
            f"Executing load_upstream",
            extra={
                "upstream_name": name,
                "is_alias": bool(alias)
            }
        )
        
        # Call discovery manager to load upstream
        try:
            result = await self.discovery_manager.load_upstream(name)
            
            # Log result
            if result.get("success"):
                log_with_metadata(
                    logger,
                    logging.INFO,
                    f"Successfully loaded upstream: {result['upstream']}",
                    metadata={
                        'upstream': result['upstream'],
                        'tool_count': result.get('tool_count', 0),
                        'requested_as': name
                    }
                )
                
                # Format success message
                result_text = (
                    f"Successfully loaded upstream '{result['upstream']}' "
                    f"with {result.get('tool_count', 0)} tools."
                )
            else:
                log_with_metadata(
                    logger,
                    logging.ERROR,
                    f"Failed to load upstream: {name}",
                    metadata={
                        'upstream_name': name,
                        'error': result.get('error', 'Unknown error')
                    }
                )
                
                # Format error message
                result_text = (
                    f"Failed to load upstream '{name}': "
                    f"{result.get('error', 'Unknown error')}"
                )
            
            return [TextContent(type="text", text=result_text)]
            
        except Exception as e:
            logger.error(
                f"load_upstream failed: {e}",
                extra={"upstream_name": name, "error": str(e)},
                exc_info=True
            )
            raise UpstreamError(
                f"Failed to load upstream: {str(e)}",
                data={"name": name, "error": str(e)}
            )
    
    async def _handle_unload_upstream(self, arguments: dict[str, Any]) -> list[TextContent]:
        """
        Handle unload_upstream tool call.
        
        Unloads an upstream MCP server by name or alias. This removes all tools
        from the upstream, closes the connection, and removes it from the loaded
        upstreams set. The operation is idempotent.
        
        Args:
            arguments: Tool arguments containing 'upstream'
            
        Returns:
            List of TextContent with unload result
            
        Raises:
            ValidationError: If upstream is not provided
        """
        # Extract arguments
        upstream = arguments.get("upstream")
        
        # Validate that upstream is provided
        if not upstream:
            raise ValidationError(
                "'upstream' must be provided",
                data={"arguments": arguments}
            )
        
        logger.info(
            f"Executing unload_upstream",
            extra={
                "upstream_name": upstream
            }
        )
        
        # Call discovery manager to unload upstream
        try:
            result = await self.discovery_manager.unload_upstream(upstream)
            
            # Log result
            if result.get("success"):
                log_with_metadata(
                    logger,
                    logging.INFO,
                    f"Successfully unloaded upstream: {result['upstream']}",
                    metadata={
                        'upstream': result['upstream'],
                        'requested_as': upstream
                    }
                )
                
                # Format success message
                result_text = f"Successfully unloaded upstream '{result['upstream']}'."
            else:
                log_with_metadata(
                    logger,
                    logging.ERROR,
                    f"Failed to unload upstream: {upstream}",
                    metadata={
                        'upstream_name': upstream,
                        'error': result.get('error', 'Unknown error')
                    }
                )
                
                # Format error message
                result_text = (
                    f"Failed to unload upstream '{upstream}': "
                    f"{result.get('error', 'Unknown error')}"
                )
            
            return [TextContent(type="text", text=result_text)]
            
        except Exception as e:
            logger.error(
                f"unload_upstream failed: {e}",
                extra={"upstream_name": upstream, "error": str(e)},
                exc_info=True
            )
            raise UpstreamError(
                f"Failed to unload upstream: {str(e)}",
                data={"name": upstream, "error": str(e)}
            )
    
    def _tool_metadata_to_mcp_tool(self, tool: ToolMetadata) -> Tool:
        """
        Convert ToolMetadata to MCP Tool format.
        
        Args:
            tool: Tool metadata
            
        Returns:
            MCP Tool object
        """
        return Tool(
            name=tool.name,
            description=tool.description,
            inputSchema=tool.input_schema.to_dict(),
        )
    
    async def run(self) -> None:
        """
        Run the MCP server with stdio transport.
        
        This method starts the server and handles stdio communication with
        the MCP client.
        """
        logger.info("Starting MCP semantic router server")
        
        # Run server with stdio transport
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )
