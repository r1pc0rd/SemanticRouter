"""Unit tests for server loading handlers (load_upstream, unload_upstream, list_upstreams)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from mcp.types import TextContent

from src.mcp_router.server.server import SemanticRouterServer
from src.mcp_router.core.errors import ValidationError, UpstreamError


class TestLoadUpstreamHandler:
    """Test _handle_load_upstream() method."""
    
    @pytest.fixture
    def mock_server(self):
        """Create a mock server with mocked dependencies."""
        server = MagicMock(spec=SemanticRouterServer)
        server.discovery_manager = MagicMock()
        server.embedding_engine = MagicMock()
        server.search_engine = MagicMock()
        server.proxy = MagicMock()
        return server
    
    @pytest.mark.asyncio
    async def test_load_by_upstream_name_success(self, mock_server):
        """Test loading upstream by canonical name - success case."""
        # Setup
        mock_server.discovery_manager.load_upstream = AsyncMock(return_value={
            "success": True,
            "upstream": "playwright",
            "tool_count": 15
        })
        
        # Execute
        from src.mcp_router.server.server import SemanticRouterServer
        result = await SemanticRouterServer._handle_load_upstream(
            mock_server,
            {"upstream": "playwright"}
        )
        
        # Verify
        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert "Successfully loaded upstream 'playwright'" in result[0].text
        assert "15 tools" in result[0].text
        mock_server.discovery_manager.load_upstream.assert_called_once_with("playwright")
    
    @pytest.mark.asyncio
    async def test_load_by_alias_success(self, mock_server):
        """Test loading upstream by alias - success case."""
        # Setup
        mock_server.discovery_manager.load_upstream = AsyncMock(return_value={
            "success": True,
            "upstream": "playwright",
            "tool_count": 15
        })
        
        # Execute
        from src.mcp_router.server.server import SemanticRouterServer
        result = await SemanticRouterServer._handle_load_upstream(
            mock_server,
            {"alias": "browser"}
        )
        
        # Verify
        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert "Successfully loaded upstream 'playwright'" in result[0].text
        assert "15 tools" in result[0].text
        mock_server.discovery_manager.load_upstream.assert_called_once_with("browser")
    
    @pytest.mark.asyncio
    async def test_load_missing_arguments(self, mock_server):
        """Test loading with neither upstream nor alias provided."""
        # Execute & Verify
        from src.mcp_router.server.server import SemanticRouterServer
        with pytest.raises(ValidationError) as exc_info:
            await SemanticRouterServer._handle_load_upstream(
                mock_server,
                {}
            )
        
        assert "Either 'upstream' or 'alias' must be provided" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_load_invalid_upstream(self, mock_server):
        """Test loading with invalid upstream name."""
        # Setup
        mock_server.discovery_manager.load_upstream = AsyncMock(return_value={
            "success": False,
            "error": "Upstream 'invalid' not found in configuration"
        })
        
        # Execute
        from src.mcp_router.server.server import SemanticRouterServer
        result = await SemanticRouterServer._handle_load_upstream(
            mock_server,
            {"upstream": "invalid"}
        )
        
        # Verify
        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert "Failed to load upstream 'invalid'" in result[0].text
        assert "not found in configuration" in result[0].text
    
    @pytest.mark.asyncio
    async def test_load_connection_failure(self, mock_server):
        """Test loading with connection failure."""
        # Setup
        mock_server.discovery_manager.load_upstream = AsyncMock(return_value={
            "success": False,
            "error": "Connection timeout after 30 seconds"
        })
        
        # Execute
        from src.mcp_router.server.server import SemanticRouterServer
        result = await SemanticRouterServer._handle_load_upstream(
            mock_server,
            {"upstream": "playwright"}
        )
        
        # Verify
        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert "Failed to load upstream 'playwright'" in result[0].text
        assert "Connection timeout" in result[0].text
    
    @pytest.mark.asyncio
    async def test_load_already_loaded(self, mock_server):
        """Test loading upstream that is already loaded (idempotent)."""
        # Setup
        mock_server.discovery_manager.load_upstream = AsyncMock(return_value={
            "success": True,
            "upstream": "playwright",
            "tool_count": 15,
            "already_loaded": True
        })
        
        # Execute
        from src.mcp_router.server.server import SemanticRouterServer
        result = await SemanticRouterServer._handle_load_upstream(
            mock_server,
            {"upstream": "playwright"}
        )
        
        # Verify
        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert "Successfully loaded upstream 'playwright'" in result[0].text
    
    @pytest.mark.asyncio
    async def test_load_exception_handling(self, mock_server):
        """Test exception handling in load_upstream."""
        # Setup
        mock_server.discovery_manager.load_upstream = AsyncMock(
            side_effect=Exception("Unexpected error")
        )
        
        # Execute & Verify
        from src.mcp_router.server.server import SemanticRouterServer
        with pytest.raises(UpstreamError) as exc_info:
            await SemanticRouterServer._handle_load_upstream(
                mock_server,
                {"upstream": "playwright"}
            )
        
        assert "Failed to load upstream" in str(exc_info.value)
        assert "Unexpected error" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_load_prefers_alias_over_upstream(self, mock_server):
        """Test that alias is preferred when both upstream and alias are provided."""
        # Setup
        mock_server.discovery_manager.load_upstream = AsyncMock(return_value={
            "success": True,
            "upstream": "playwright",
            "tool_count": 15
        })
        
        # Execute
        from src.mcp_router.server.server import SemanticRouterServer
        result = await SemanticRouterServer._handle_load_upstream(
            mock_server,
            {"upstream": "jira", "alias": "browser"}
        )
        
        # Verify - should use alias "browser", not upstream "jira"
        mock_server.discovery_manager.load_upstream.assert_called_once_with("browser")


class TestLoadUpstreamIntegration:
    """Integration tests for load_upstream tool in server."""
    
    @pytest.mark.asyncio
    async def test_load_upstream_tool_registered(self):
        """Test that load_upstream tool is registered in list_tools()."""
        # Setup
        from src.mcp_router.discovery.manager import ToolDiscoveryManager
        from src.mcp_router.embedding.engine import EmbeddingEngine
        from src.mcp_router.search.engine import SemanticSearchEngine
        from src.mcp_router.proxy.proxy import ToolCallProxy
        from src.mcp_router.core.config import RouterConfig, UpstreamConfig
        
        config = RouterConfig(
            mcp_servers={
                "test": UpstreamConfig(
                    transport="stdio",
                    command="test",
                    args=[]
                )
            }
        )
        
        # Create mocked components
        discovery_manager = MagicMock(spec=ToolDiscoveryManager)
        discovery_manager.get_default_tool_subset = MagicMock(return_value=[])
        
        embedding_engine = MagicMock(spec=EmbeddingEngine)
        search_engine = MagicMock(spec=SemanticSearchEngine)
        proxy = MagicMock(spec=ToolCallProxy)
        
        # Create server
        server = SemanticRouterServer(
            discovery_manager=discovery_manager,
            embedding_engine=embedding_engine,
            search_engine=search_engine,
            proxy=proxy
        )
        
        # Get list_tools handler
        list_tools_handler = None
        for handler in server.server.request_handlers.values():
            if hasattr(handler, '__name__') and 'list_tools' in handler.__name__:
                list_tools_handler = handler
                break
        
        # Execute
        if list_tools_handler:
            tools = await list_tools_handler()
            
            # Verify load_upstream tool is present
            tool_names = [tool.name for tool in tools]
            assert "load_upstream" in tool_names
            
            # Find load_upstream tool
            load_upstream_tool = next(t for t in tools if t.name == "load_upstream")
            
            # Verify schema
            assert "upstream" in load_upstream_tool.inputSchema["properties"]
            assert "alias" in load_upstream_tool.inputSchema["properties"]
            assert load_upstream_tool.inputSchema["required"] == []
    
    @pytest.mark.asyncio
    async def test_load_upstream_tool_call_routing(self):
        """Test that load_upstream tool calls are routed correctly."""
        # Setup
        from src.mcp_router.discovery.manager import ToolDiscoveryManager
        from src.mcp_router.embedding.engine import EmbeddingEngine
        from src.mcp_router.search.engine import SemanticSearchEngine
        from src.mcp_router.proxy.proxy import ToolCallProxy
        from src.mcp_router.core.config import RouterConfig, UpstreamConfig
        
        config = RouterConfig(
            mcp_servers={
                "test": UpstreamConfig(
                    transport="stdio",
                    command="test",
                    args=[]
                )
            }
        )
        
        # Create mocked components
        discovery_manager = MagicMock(spec=ToolDiscoveryManager)
        discovery_manager.load_upstream = AsyncMock(return_value={
            "success": True,
            "upstream": "test",
            "tool_count": 5
        })
        
        embedding_engine = MagicMock(spec=EmbeddingEngine)
        search_engine = MagicMock(spec=SemanticSearchEngine)
        proxy = MagicMock(spec=ToolCallProxy)
        
        # Create server
        server = SemanticRouterServer(
            discovery_manager=discovery_manager,
            embedding_engine=embedding_engine,
            search_engine=search_engine,
            proxy=proxy
        )
        
        # Get call_tool handler
        call_tool_handler = None
        for handler in server.server.request_handlers.values():
            if hasattr(handler, '__name__') and 'call_tool' in handler.__name__:
                call_tool_handler = handler
                break
        
        # Execute
        if call_tool_handler:
            result = await call_tool_handler(
                name="load_upstream",
                arguments={"upstream": "test"}
            )
            
            # Verify
            assert len(result) == 1
            assert isinstance(result[0], TextContent)
            assert "Successfully loaded upstream 'test'" in result[0].text
            discovery_manager.load_upstream.assert_called_once_with("test")
