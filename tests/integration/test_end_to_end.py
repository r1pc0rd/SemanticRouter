"""
Integration tests for end-to-end MCP Semantic Router functionality.

These tests validate the complete startup sequence, tool discovery,
semantic search, and tool call proxying with mock upstream servers.
"""

import asyncio
import json
import pytest
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_router.__main__ import startup_sequence, shutdown_sequence
from mcp_router.core.config import RouterConfig, UpstreamConfig
from mcp_router.core.models import ToolMetadata, JSONSchema
from mcp_router.discovery.manager import ToolDiscoveryManager
from mcp_router.embedding.engine import EmbeddingEngine


# Mock upstream tools for testing
MOCK_TOOLS = [
    {
        "name": "search_web",
        "description": "Search the web for information",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "read_file",
        "description": "Read contents of a file",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"}
            },
            "required": ["path"]
        }
    }
]


@pytest.fixture
def mock_config_file(tmp_path: Path) -> Path:
    """Create a temporary configuration file for testing."""
    config = {
        "mcpServers": {
            "test_upstream": {
                "transport": "stdio",
                "command": "python",
                "args": ["-m", "test_mcp_server"]
            }
        }
    }
    
    config_path = tmp_path / "test_config.json"
    config_path.write_text(json.dumps(config))
    return config_path


@pytest.fixture
def mock_upstream_connection():
    """Create a mock upstream connection that returns test tools."""
    mock_conn = AsyncMock()
    mock_conn.connect = AsyncMock()
    mock_conn.fetch_tools = AsyncMock(return_value=[
        ToolMetadata(
            name="test_upstream.search_web",
            original_name="search_web",
            description="Search the web for information",
            input_schema=JSONSchema(
                type="object",
                properties={"query": {"type": "string"}},
                required=["query"]
            ),
            upstream_id="test_upstream"
        ),
        ToolMetadata(
            name="test_upstream.read_file",
            original_name="read_file",
            description="Read contents of a file",
            input_schema=JSONSchema(
                type="object",
                properties={"path": {"type": "string"}},
                required=["path"]
            ),
            upstream_id="test_upstream"
        )
    ])
    mock_conn.disconnect = AsyncMock()
    return mock_conn


@pytest.mark.asyncio
async def test_startup_sequence_success(
    mock_config_file: Path,
    mock_upstream_connection: AsyncMock
):
    """
    Test successful startup sequence with mock upstream.
    
    Validates:
    - Configuration loading
    - Embedding engine initialization
    - Tool discovery
    - Tool embedding generation
    """
    with patch('mcp_router.__main__.SemanticRouterServer') as mock_server_class, \
         patch('mcp_router.discovery.manager.UpstreamConnection', return_value=mock_upstream_connection):
        
        # Mock the server instance and its run method
        mock_server_instance = AsyncMock()
        mock_server_instance.run = AsyncMock()
        mock_server_class.return_value = mock_server_instance
        
        # Run startup sequence
        try:
            await asyncio.wait_for(
                startup_sequence(str(mock_config_file)),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            pytest.fail("Startup sequence timed out")
        
        # Verify upstream connection was established
        mock_upstream_connection.connect.assert_called_once()
        
        # Verify tools were fetched
        mock_upstream_connection.fetch_tools.assert_called_once()
        
        # Verify server was created with correct arguments
        mock_server_class.assert_called_once()
        call_kwargs = mock_server_class.call_args.kwargs
        assert 'discovery_manager' in call_kwargs
        assert 'embedding_engine' in call_kwargs
        assert 'search_engine' in call_kwargs
        assert 'proxy' in call_kwargs
        
        # Verify server.run() was called
        mock_server_instance.run.assert_called_once()


@pytest.mark.asyncio
async def test_startup_sequence_invalid_config(tmp_path: Path):
    """
    Test startup sequence with invalid configuration file.
    
    Validates:
    - Configuration error is raised
    - Startup fails gracefully
    """
    # Create invalid config (missing required fields)
    invalid_config = tmp_path / "invalid_config.json"
    invalid_config.write_text('{"upstreams": {}}')
    
    with pytest.raises(Exception):  # Should raise ConfigurationError or RuntimeError
        await startup_sequence(str(invalid_config))


@pytest.mark.asyncio
async def test_startup_sequence_missing_config():
    """
    Test startup sequence with missing configuration file.
    
    Validates:
    - FileNotFoundError is raised
    - Startup fails gracefully
    """
    with pytest.raises(Exception):
        await startup_sequence("nonexistent_config.json")


@pytest.mark.asyncio
async def test_shutdown_sequence_success(mock_upstream_connection: AsyncMock):
    """
    Test graceful shutdown sequence.
    
    Validates:
    - Upstream connections are closed
    - Resources are cleaned up
    """
    # Create a discovery manager with mock upstream
    discovery_manager = ToolDiscoveryManager()
    discovery_manager.upstreams = {"test_upstream": mock_upstream_connection}
    
    # Run shutdown sequence
    await shutdown_sequence(discovery_manager)
    
    # Verify upstream was disconnected
    mock_upstream_connection.disconnect.assert_called_once()


@pytest.mark.asyncio
async def test_shutdown_sequence_with_error(mock_upstream_connection: AsyncMock):
    """
    Test shutdown sequence when upstream disconnect fails.
    
    Validates:
    - Errors during shutdown are logged but don't crash
    - Shutdown completes despite errors
    """
    # Make disconnect raise an error
    mock_upstream_connection.disconnect = AsyncMock(side_effect=Exception("Disconnect failed"))
    
    # Create a discovery manager with mock upstream
    discovery_manager = ToolDiscoveryManager()
    discovery_manager.upstreams = {"test_upstream": mock_upstream_connection}
    
    # Run shutdown sequence - should not raise
    await shutdown_sequence(discovery_manager)
    
    # Verify disconnect was attempted
    mock_upstream_connection.disconnect.assert_called_once()


@pytest.mark.asyncio
async def test_tool_discovery_with_failed_upstream(
    mock_config_file: Path,
    mock_upstream_connection: AsyncMock
):
    """
    Test tool discovery when one upstream fails.
    
    Validates:
    - Failed upstream doesn't prevent startup
    - Other upstreams are still discovered
    - Graceful failure handling
    """
    # Make the first connection attempt fail, second succeed
    failing_conn = AsyncMock()
    failing_conn.connect = AsyncMock(side_effect=Exception("Connection failed"))
    
    connections = [failing_conn, mock_upstream_connection]
    connection_index = 0
    
    def get_connection(*args, **kwargs):
        nonlocal connection_index
        conn = connections[connection_index]
        connection_index += 1
        return conn
    
    # Create config with two upstreams
    config = {
        "mcpServers": {
            "failing_upstream": {
                "transport": "stdio",
                "command": "python",
                "args": ["-m", "failing_server"]
            },
            "working_upstream": {
                "transport": "stdio",
                "command": "python",
                "args": ["-m", "working_server"]
            }
        }
    }
    
    config_path = mock_config_file.parent / "multi_upstream_config.json"
    config_path.write_text(json.dumps(config))
    
    with patch('mcp_router.discovery.manager.UpstreamConnection', side_effect=get_connection):
        discovery_manager = ToolDiscoveryManager()
        await discovery_manager.initialize(str(config_path))
        
        # Verify at least one upstream succeeded
        assert len(discovery_manager.upstreams) >= 1
        
        # Verify tools were discovered from working upstream
        tools = discovery_manager.get_all_tools()
        assert len(tools) > 0


@pytest.mark.asyncio
async def test_end_to_end_tool_search_and_call(
    mock_config_file: Path,
    mock_upstream_connection: AsyncMock
):
    """
    Test end-to-end flow: startup -> search tools -> call tool.
    
    Validates:
    - Complete workflow from startup to tool execution
    - Semantic search returns relevant tools
    - Tool calls are proxied correctly
    """
    with patch('mcp_router.discovery.manager.UpstreamConnection', return_value=mock_upstream_connection):
        # Initialize discovery manager
        discovery_manager = ToolDiscoveryManager()
        await discovery_manager.initialize(str(mock_config_file))
        
        # Initialize embedding engine
        embedding_engine = EmbeddingEngine()
        await embedding_engine.initialize()
        
        # Generate tool embeddings
        tools = discovery_manager.get_all_tools()
        await embedding_engine.generate_tool_embeddings(tools)
        
        # Test semantic search
        from mcp_router.search.engine import SemanticSearchEngine
        search_engine = SemanticSearchEngine(embedding_engine)
        search_engine.set_tools(tools)
        
        results = await search_engine.search_tools("search the internet", context=[])
        
        # Verify search returns results
        assert len(results) > 0
        assert any("search" in r.tool.name.lower() for r in results)
        
        # Test tool call proxy
        from mcp_router.proxy.proxy import ToolCallProxy
        proxy = ToolCallProxy(
            upstreams=discovery_manager.upstreams,
            upstream_configs=discovery_manager._config.mcp_servers
        )
        
        # Mock the upstream call_tool method
        mock_upstream_connection.call_tool = AsyncMock(return_value={
            "content": [{"type": "text", "text": "Search results"}],
            "isError": False
        })
        
        result = await proxy.call_tool(
            "test_upstream.search_web",
            {"query": "test query"}
        )
        
        # Verify tool was called
        assert result is not None
        assert not result.is_error
        mock_upstream_connection.call_tool.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
