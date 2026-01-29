"""
Tool discovery manager for connecting to upstream MCP servers.

This module manages connections to multiple upstream MCP servers and
aggregates their tool catalogs.
"""
import asyncio
import logging
from typing import Dict, List, Optional, TYPE_CHECKING

from mcp_router.core.config import RouterConfig
from mcp_router.core.config_parser import load_config
from mcp_router.core.models import ToolMetadata
from mcp_router.core.logging import log_with_metadata
from mcp_router.discovery.upstream import UpstreamConnection
from mcp_router.discovery.alias_resolver import AliasResolver

if TYPE_CHECKING:
    from mcp_router.embedding.engine import EmbeddingEngine
    from mcp_router.search.engine import SemanticSearchEngine


logger = logging.getLogger(__name__)


class ToolDiscoveryManager:
    """
    Manages tool discovery across multiple upstream MCP servers.
    
    Handles connecting to upstreams, fetching tool catalogs, and maintaining
    the aggregated tool list. Supports dynamic loading and unloading of upstreams.
    """
    
    def __init__(
        self,
        config: RouterConfig,
        embedding_engine: 'EmbeddingEngine',
        search_engine: 'SemanticSearchEngine'
    ) -> None:
        """Initialize the tool discovery manager with configuration.
        
        Args:
            config: Router configuration containing upstream definitions
            embedding_engine: Engine for generating tool embeddings
            search_engine: Engine for semantic search over tools
        """
        self._config: RouterConfig = config
        self.embedding_engine = embedding_engine
        self.search_engine = search_engine
        self.upstreams: Dict[str, UpstreamConnection] = {}
        self._loaded_upstreams: Dict[str, UpstreamConnection] = {}
        self.all_tools: List[ToolMetadata] = []
        self._alias_resolver: AliasResolver = AliasResolver(config)
    
    async def initialize(self) -> None:
        """
        Initialize tool discovery by loading upstreams based on auto_load configuration.
        
        This method:
        1. Parses auto_load configuration
        2. Loads specified upstreams (or all if auto_load=["all"])
        3. Handles failures gracefully (logs but continues)
        
        If auto_load is empty, no upstreams are loaded on startup.
        """
        # Get auto_load configuration (default to ["all"] if loading config is None)
        auto_load = self._config.loading.auto_load if self._config.loading else ["all"]
        
        # Handle special case: ["all"] means load all upstreams
        if auto_load == ["all"]:
            auto_load = list(self._config.mcp_servers.keys())
            logger.info(f"auto_load=['all'] - loading all {len(auto_load)} upstreams")
        elif not auto_load:
            logger.info("auto_load=[] - no upstreams will be loaded on startup")
            return
        else:
            logger.info(f"auto_load={auto_load} - loading {len(auto_load)} specified upstreams")
        
        # Load each upstream in auto_load list
        successful_loads = 0
        failed_loads = []
        
        for upstream_name in auto_load:
            try:
                # Use load_upstream() method (stub for now, will be implemented in Task 6)
                result = await self.load_upstream(upstream_name)
                if result.get("success"):
                    successful_loads += 1
                else:
                    failed_loads.append(upstream_name)
                    logger.warning(f"Failed to load upstream '{upstream_name}': {result.get('error')}")
            except Exception as e:
                failed_loads.append(upstream_name)
                logger.warning(f"Failed to load upstream '{upstream_name}': {e}")
        
        logger.info(
            f"Initialization complete: {successful_loads} upstreams loaded successfully"
        )
        
        if failed_loads:
            logger.warning(f"Failed to load upstreams: {', '.join(failed_loads)}")
    
    async def load_upstream(self, name: str) -> dict:
        """
        Load an upstream MCP server dynamically.
        
        This method:
        1. Resolves aliases to canonical upstream names
        2. Checks if already loaded (idempotent)
        3. Validates upstream exists in configuration
        4. Creates connection and fetches tools
        5. Generates embeddings for tools
        6. Adds tools to search engine
        7. Stores connection for future tool calls
        
        Args:
            name: Upstream name or alias
            
        Returns:
            Dictionary with:
            - success: bool - Whether load succeeded
            - upstream: str - Canonical upstream name
            - tool_count: int - Number of tools loaded (if successful)
            - error: str - Error message (if failed)
        """
        try:
            # Resolve alias to canonical name
            canonical_name = self._alias_resolver.resolve(name)
            logger.info(f"Loading upstream '{canonical_name}' (requested as '{name}')")
            
            # Check if already loaded (idempotent)
            if canonical_name in self._loaded_upstreams:
                tool_count = len([t for t in self.all_tools if t.upstream_id == canonical_name])
                logger.info(f"Upstream '{canonical_name}' already loaded with {tool_count} tools")
                return {
                    "success": True,
                    "upstream": canonical_name,
                    "tool_count": tool_count
                }
            
            # Validate upstream exists in config
            if canonical_name not in self._config.mcp_servers:
                error_msg = f"Upstream '{canonical_name}' not found in configuration"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
            
            # Get upstream config
            upstream_config = self._config.mcp_servers[canonical_name]
            
            # Create connection with timeout
            connection = UpstreamConnection(canonical_name, upstream_config)
            timeout = self._config.loading.connection_timeout if self._config.loading else 30
            
            try:
                await asyncio.wait_for(connection.connect(), timeout=timeout)
            except asyncio.TimeoutError:
                error_msg = f"Connection timeout after {timeout}s"
                logger.error(f"Failed to connect to '{canonical_name}': {error_msg}")
                return {"success": False, "error": error_msg}
            except Exception as e:
                error_msg = f"Connection failed: {str(e)}"
                logger.error(f"Failed to connect to '{canonical_name}': {error_msg}")
                return {"success": False, "error": error_msg}
            
            # Fetch tools from upstream
            try:
                tools = await connection.fetch_tools()
            except Exception as e:
                error_msg = f"Failed to fetch tools: {str(e)}"
                logger.error(f"Failed to fetch tools from '{canonical_name}': {error_msg}")
                await connection.disconnect()
                return {"success": False, "error": error_msg}
            
            # Generate embeddings for tools
            try:
                await self.embedding_engine.generate_tool_embeddings(tools)
            except Exception as e:
                error_msg = f"Failed to generate embeddings: {str(e)}"
                logger.error(f"Failed to generate embeddings for '{canonical_name}': {error_msg}")
                await connection.disconnect()
                return {"success": False, "error": error_msg}
            
            # Add tools to search engine
            try:
                self.search_engine.add_tools(tools)
            except Exception as e:
                error_msg = f"Failed to add tools to search engine: {str(e)}"
                logger.error(f"Failed to add tools from '{canonical_name}': {error_msg}")
                await connection.disconnect()
                return {"success": False, "error": error_msg}
            
            # Store connection and update all_tools
            self._loaded_upstreams[canonical_name] = connection
            self.upstreams[canonical_name] = connection  # Also store in upstreams for backward compatibility
            self.all_tools.extend(tools)
            
            logger.info(f"Successfully loaded upstream '{canonical_name}' with {len(tools)} tools")
            return {
                "success": True,
                "upstream": canonical_name,
                "tool_count": len(tools)
            }
            
        except ValueError as e:
            # Alias resolution error
            error_msg = str(e)
            logger.error(f"Failed to load upstream '{name}': {error_msg}")
            return {"success": False, "error": error_msg}
        except Exception as e:
            # Unexpected error
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(f"Failed to load upstream '{name}': {error_msg}")
            return {"success": False, "error": error_msg}
    
    async def load_multiple_upstreams(self, names: List[str]) -> dict:
        """
        Load multiple upstream MCP servers concurrently.
        
        This method:
        1. Resolves all aliases to canonical names
        2. Loads upstreams concurrently using asyncio.gather()
        3. Tracks successful and failed loads
        4. Returns summary of results
        
        Args:
            names: List of upstream names or aliases to load
            
        Returns:
            Dictionary with:
            - loaded: List[str] - Successfully loaded upstream names
            - failed: List[dict] - Failed loads with format:
                [{"name": str, "error": str}, ...]
        """
        logger.info(f"Loading {len(names)} upstreams: {names}")
        
        # Resolve all aliases first
        resolved_names = []
        failed = []
        
        for name in names:
            try:
                canonical_name = self._alias_resolver.resolve(name)
                resolved_names.append(canonical_name)
            except ValueError as e:
                # Alias resolution failed
                failed.append({
                    "name": name,
                    "error": str(e)
                })
                logger.warning(f"Failed to resolve upstream '{name}': {e}")
        
        # Load all resolved upstreams concurrently
        if resolved_names:
            # Create tasks for all loads
            load_tasks = [self.load_upstream(name) for name in resolved_names]
            
            # Execute concurrently
            results = await asyncio.gather(*load_tasks, return_exceptions=False)
            
            # Process results
            loaded = []
            for name, result in zip(resolved_names, results):
                if result.get("success"):
                    loaded.append(name)
                else:
                    failed.append({
                        "name": name,
                        "error": result.get("error", "Unknown error")
                    })
        else:
            loaded = []
        
        logger.info(
            f"Batch load complete: {len(loaded)} succeeded, {len(failed)} failed"
        )
        
        return {
            "loaded": loaded,
            "failed": failed
        }
    
    async def unload_upstream(self, name: str) -> dict:
        """
        Unload an upstream MCP server.
        
        Removes all tools from the upstream, closes the connection, and removes
        it from the loaded upstreams set. This operation is idempotent - unloading
        an already-unloaded upstream returns success.
        
        Args:
            name: Upstream name or alias to unload
            
        Returns:
            dict: Success dict with format:
                {"success": True, "upstream": canonical_name}
            Or error dict with format:
                {"success": False, "error": error_message}
        """
        try:
            # Resolve alias to canonical name
            canonical_name = self._alias_resolver.resolve(name)
            
            # Check if upstream is loaded
            if canonical_name not in self._loaded_upstreams:
                # Already unloaded - return success (idempotent)
                logger.info(f"Upstream '{canonical_name}' is not loaded, nothing to unload")
                return {
                    "success": True,
                    "upstream": canonical_name
                }
            
            logger.info(f"Unloading upstream '{canonical_name}'...")
            
            # Get namespace for this upstream
            namespace = canonical_name
            
            # Remove tools from search engine
            try:
                self.search_engine.remove_tools(namespace)
                logger.info(f"Removed tools for upstream '{canonical_name}' from search engine")
            except Exception as e:
                error_msg = f"Failed to remove tools from search engine: {str(e)}"
                logger.error(f"Failed to unload upstream '{canonical_name}': {error_msg}")
                return {"success": False, "error": error_msg}
            
            # Close upstream connection
            connection = self._loaded_upstreams.get(canonical_name)
            if connection:
                try:
                    await connection.disconnect()
                    logger.info(f"Disconnected from upstream '{canonical_name}'")
                except Exception as e:
                    # Log but don't fail - connection might already be closed
                    logger.warning(f"Error disconnecting from upstream '{canonical_name}': {str(e)}")
                
                # Remove from loaded upstreams dict
                del self._loaded_upstreams[canonical_name]
                # Also remove from upstreams dict for backward compatibility
                if canonical_name in self.upstreams:
                    del self.upstreams[canonical_name]
            
            # Remove tools from all_tools list
            self.all_tools = [
                tool for tool in self.all_tools
                if tool.upstream_id != canonical_name
            ]
            
            logger.info(f"Successfully unloaded upstream '{canonical_name}'")
            return {
                "success": True,
                "upstream": canonical_name
            }
            
        except ValueError as e:
            # Alias resolution error
            error_msg = str(e)
            logger.error(f"Failed to unload upstream '{name}': {error_msg}")
            return {"success": False, "error": error_msg}
        except Exception as e:
            # Unexpected error
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(f"Failed to unload upstream '{name}': {error_msg}")
            return {"success": False, "error": error_msg}
    
    def get_loaded_upstreams(self) -> List[str]:
        """
        Get list of currently loaded upstream names.
        
        Returns:
            List of canonical upstream names that are currently loaded
        """
        return list(self._loaded_upstreams.keys())
    
    def is_loaded(self, name: str) -> bool:
        """
        Check if an upstream is currently loaded.
        
        Resolves aliases to canonical names before checking. Returns False
        if the name is invalid or not found in configuration.
        
        Args:
            name: Upstream name or alias to check
            
        Returns:
            True if upstream is loaded, False otherwise
        """
        try:
            canonical_name = self._alias_resolver.resolve(name)
            return canonical_name in self._loaded_upstreams
        except ValueError:
            # Invalid name or alias
            return False
    
    def get_available_upstreams(self) -> List[str]:
        """
        Get list of all available upstream names from configuration.
        
        Returns:
            List of all upstream names defined in the router configuration
        """
        return list(self._config.mcp_servers.keys())
    
    def get_all_tools(self) -> list[ToolMetadata]:
        """
        Get all discovered tools from all upstreams.
        
        Returns:
            List of all tool metadata objects
        """
        return self.all_tools.copy()
    
    def get_upstream(self, upstream_id: str) -> Optional[UpstreamConnection]:
        """
        Get upstream connection by ID.
        
        Args:
            upstream_id: Upstream identifier
            
        Returns:
            UpstreamConnection if found, None otherwise
        """
        return self.upstreams.get(upstream_id)
    
    def get_tool_by_name(self, namespaced_name: str) -> Optional[ToolMetadata]:
        """
        Find a tool by its namespaced name.
        
        Args:
            namespaced_name: Full tool name with namespace (e.g., "browser.navigate")
            
        Returns:
            ToolMetadata if found, None otherwise
        """
        for tool in self.all_tools:
            if tool.name == namespaced_name:
                return tool
        return None
    
    async def shutdown(self) -> None:
        """
        Shutdown all upstream connections gracefully.
        """
        logger.info("Shutting down tool discovery manager...")
        
        for upstream_id, upstream in self.upstreams.items():
            try:
                await upstream.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting from upstream '{upstream_id}': {e}")
        
        self.upstreams.clear()
        self.all_tools.clear()
        
        logger.info("Tool discovery manager shutdown complete")
    
    def get_default_tool_subset(self, max_tools: int = 20) -> list[ToolMetadata]:
        """
        Get a diverse subset of tools for the default tools/list response.
        
        Selects tools proportionally from each upstream to ensure diversity.
        
        Args:
            max_tools: Maximum number of tools to return (default: 20)
            
        Returns:
            List of up to max_tools tools with diverse upstream coverage
        """
        from mcp_router.search.default_subset import select_default_tool_subset
        return select_default_tool_subset(self.all_tools, max_tools)
