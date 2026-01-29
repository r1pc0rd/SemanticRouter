"""
Main entry point for the MCP Semantic Router.

This module provides the startup sequence for the router:
1. Load configuration from config.json
2. Initialize embedding engine
3. Initialize tool discovery manager
4. Generate tool embeddings
5. Start MCP server with stdio transport
"""

import asyncio
import sys
import logging
from pathlib import Path
from typing import Optional

from mcp_router.core.config_parser import load_config, ConfigurationError
from mcp_router.core.logging import setup_logging, log_with_metadata
from mcp_router.embedding.engine import EmbeddingEngine
from mcp_router.discovery.manager import ToolDiscoveryManager
from mcp_router.search.engine import SemanticSearchEngine
from mcp_router.proxy.proxy import ToolCallProxy
from mcp_router.server.server import SemanticRouterServer


# Set up logging
logger = logging.getLogger(__name__)


async def startup_sequence(config_path: str = "config.json") -> None:
    """
    Execute the startup sequence for the MCP Semantic Router.
    
    Args:
        config_path: Path to the configuration file (default: config.json)
        
    Raises:
        ConfigurationError: If configuration is invalid
        RuntimeError: If startup fails
    """
    try:
        # Step 1: Load configuration
        logger.info("Loading configuration from %s", config_path)
        config = load_config(config_path)
        log_with_metadata(
            logger,
            logging.INFO,
            "Configuration loaded successfully",
            {"upstream_count": len(config.mcp_servers)}
        )
        
        # Step 2: Initialize embedding engine
        logger.info("Initializing embedding engine")
        embedding_engine = EmbeddingEngine()
        await embedding_engine.initialize()
        log_with_metadata(
            logger,
            logging.INFO,
            "Embedding engine initialized",
            {"model": "all-MiniLM-L6-v2", "dimensions": 384}
        )
        
        # Step 3: Initialize search engine (before discovery manager)
        logger.info("Initializing search engine")
        search_engine = SemanticSearchEngine(embedding_engine)
        
        # Step 4: Initialize tool discovery manager
        logger.info("Discovering tools from upstream MCP servers")
        discovery_manager = ToolDiscoveryManager(config, embedding_engine, search_engine)
        await discovery_manager.initialize()
        
        tool_count = len(discovery_manager.get_all_tools())
        log_with_metadata(
            logger,
            logging.INFO,
            "Tool discovery completed",
            {"total_tools": tool_count}
        )
        
        # Step 5: Generate tool embeddings and set tools in search engine
        logger.info("Generating embeddings for %d tools", tool_count)
        all_tools = discovery_manager.get_all_tools()
        await embedding_engine.generate_tool_embeddings(all_tools)
        search_engine.set_tools(all_tools)
        log_with_metadata(
            logger,
            logging.INFO,
            "Tool embeddings generated",
            {"tool_count": tool_count}
        )
        
        # Step 6: Initialize proxy
        logger.info("Initializing proxy")
        proxy = ToolCallProxy(
            upstreams=discovery_manager.upstreams,
            upstream_configs=config.mcp_servers
        )
        
        # Step 7: Create and start MCP server
        logger.info("Starting MCP server with stdio transport")
        server = SemanticRouterServer(
            discovery_manager=discovery_manager,
            embedding_engine=embedding_engine,
            search_engine=search_engine,
            proxy=proxy
        )
        
        log_with_metadata(
            logger,
            logging.INFO,
            "MCP server started",
            {"transport": "stdio", "tools_available": tool_count}
        )
        
        # Run the server (handles stdio internally)
        await server.run()
            
    except ConfigurationError as e:
        log_with_metadata(
            logger,
            logging.ERROR,
            "Configuration error during startup",
            {"error": str(e)}
        )
        raise
    except Exception as e:
        log_with_metadata(
            logger,
            logging.ERROR,
            "Unexpected error during startup",
            {"error": str(e), "error_type": type(e).__name__}
        )
        raise RuntimeError(f"Startup failed: {e}") from e


async def shutdown_sequence(
    discovery_manager: Optional[ToolDiscoveryManager] = None
) -> None:
    """
    Execute graceful shutdown sequence.
    
    Args:
        discovery_manager: Tool discovery manager to clean up
    """
    try:
        logger.info("Starting graceful shutdown")
        
        # Close upstream connections
        if discovery_manager:
            await discovery_manager.shutdown()
            log_with_metadata(
                logger,
                logging.INFO,
                "Upstream connections closed",
                {}
            )
        
        logger.info("Shutdown complete")
        
    except Exception as e:
        log_with_metadata(
            logger,
            logging.ERROR,
            "Error during shutdown",
            {"error": str(e)}
        )


def main() -> None:
    """
    Main entry point for the MCP Semantic Router.
    
    Usage:
        python -m mcp_router [config_path]
        
    Args:
        config_path: Optional path to configuration file (default: config.json)
    """
    # Set up logging
    setup_logging()
    
    # Get config path from command line or use default
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    
    # Check if config file exists
    if not Path(config_path).exists():
        logger.error("Configuration file not found: %s", config_path)
        sys.exit(1)
    
    # Run startup sequence
    try:
        asyncio.run(startup_sequence(config_path))
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down")
        sys.exit(0)
    except Exception as e:
        logger.error("Fatal error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
