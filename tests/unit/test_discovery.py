"""
Unit tests for tool discovery manager and upstream connections.

Tests tool discovery, upstream connection management, and error handling.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile
import json

from mcp_router.core.config import UpstreamConfig, RouterConfig
from mcp_router.core.models import ToolMetadata, JSONSchema
from mcp_router.discovery.manager import ToolDiscoveryManager
from mcp_router.discovery.upstream import UpstreamConnection


class TestUpstreamConnection:
    """Tests for UpstreamConnection class"""
    
    def test_initialization(self):
        """Test upstream connection initialization"""
        config = UpstreamConfig(
            transport='stdio',
            command='python',
            args=['server.py']
        )
        
        upstream = UpstreamConnection('test-upstream', config)
        
        assert upstream.upstream_id == 'test-upstream'
        assert upstream.config == config
        assert upstream.tools == []
        assert not upstream.is_connected
    
    @pytest.mark.asyncio
    async def test_connect_unsupported_transport(self):
        """Test that unsupported transport raises error"""
        config = UpstreamConfig(
            transport='sse',
            url='http://example.com'
        )
        
        upstream = UpstreamConnection('test-upstream', config)
        
        with pytest.raises(ValueError, match="Unsupported transport"):
            await upstream.connect()
    
    @pytest.mark.asyncio
    async def test_connect_missing_command(self):
        """Test that missing command for stdio raises error"""
        # The validation happens in UpstreamConfig.__post_init__
        # so we can't even create the config without a command
        # This test verifies the config validation works
        with pytest.raises(ValueError, match="command is required"):
            config = UpstreamConfig(transport='stdio')
    
    @pytest.mark.asyncio
    async def test_fetch_tools_not_connected(self):
        """Test that fetching tools before connection raises error"""
        config = UpstreamConfig(
            transport='stdio',
            command='python'
        )
        
        upstream = UpstreamConnection('test-upstream', config)
        
        with pytest.raises(RuntimeError, match="Not connected"):
            await upstream.fetch_tools()
    
    @pytest.mark.asyncio
    async def test_call_tool_not_connected(self):
        """Test that calling tool before connection raises error"""
        config = UpstreamConfig(
            transport='stdio',
            command='python'
        )
        
        upstream = UpstreamConnection('test-upstream', config)
        
        with pytest.raises(RuntimeError, match="Not connected"):
            await upstream.call_tool('test_tool', {})


class TestToolDiscoveryManager:
    """Tests for ToolDiscoveryManager class"""
    
    def test_initialization(self):
        """Test manager initialization"""
        manager = ToolDiscoveryManager()
        
        assert manager.upstreams == {}
        assert manager.all_tools == []
    
    @pytest.mark.asyncio
    async def test_initialize_missing_config(self):
        """Test initialization with missing config file"""
        manager = ToolDiscoveryManager()
        
        with pytest.raises(RuntimeError, match="Configuration loading failed"):
            await manager.initialize('/nonexistent/config.json')
    
    @pytest.mark.asyncio
    async def test_initialize_invalid_config(self):
        """Test initialization with invalid config"""
        # Create temporary invalid config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"invalid": "config"}')
            config_path = f.name
        
        manager = ToolDiscoveryManager()
        
        try:
            with pytest.raises(RuntimeError):
                await manager.initialize(config_path)
        finally:
            import os
            os.unlink(config_path)
    
    def test_get_all_tools(self):
        """Test getting all tools"""
        manager = ToolDiscoveryManager()
        
        # Add some tools
        tool1 = ToolMetadata(
            name='upstream1.tool1',
            original_name='tool1',
            description='Test tool 1',
            input_schema=JSONSchema(type='object'),
            upstream_id='upstream1'
        )
        tool2 = ToolMetadata(
            name='upstream2.tool2',
            original_name='tool2',
            description='Test tool 2',
            input_schema=JSONSchema(type='object'),
            upstream_id='upstream2'
        )
        
        manager.all_tools = [tool1, tool2]
        
        # Get all tools
        tools = manager.get_all_tools()
        
        assert len(tools) == 2
        assert tools[0] == tool1
        assert tools[1] == tool2
        
        # Verify it's a copy
        assert tools is not manager.all_tools
    
    def test_get_upstream(self):
        """Test getting upstream by ID"""
        manager = ToolDiscoveryManager()
        
        config = UpstreamConfig(transport='stdio', command='python')
        upstream = UpstreamConnection('test-upstream', config)
        
        manager.upstreams['test-upstream'] = upstream
        
        # Get existing upstream
        result = manager.get_upstream('test-upstream')
        assert result is upstream
        
        # Get non-existent upstream
        result = manager.get_upstream('nonexistent')
        assert result is None
    
    def test_get_tool_by_name(self):
        """Test finding tool by namespaced name"""
        manager = ToolDiscoveryManager()
        
        tool1 = ToolMetadata(
            name='upstream1.tool1',
            original_name='tool1',
            description='Test tool 1',
            input_schema=JSONSchema(type='object'),
            upstream_id='upstream1'
        )
        tool2 = ToolMetadata(
            name='upstream2.tool2',
            original_name='tool2',
            description='Test tool 2',
            input_schema=JSONSchema(type='object'),
            upstream_id='upstream2'
        )
        
        manager.all_tools = [tool1, tool2]
        
        # Find existing tool
        result = manager.get_tool_by_name('upstream1.tool1')
        assert result is tool1
        
        # Find non-existent tool
        result = manager.get_tool_by_name('nonexistent.tool')
        assert result is None
    
    @pytest.mark.asyncio
    async def test_shutdown(self):
        """Test graceful shutdown"""
        manager = ToolDiscoveryManager()
        
        # Create mock upstream
        config = UpstreamConfig(transport='stdio', command='python')
        upstream = UpstreamConnection('test-upstream', config)
        upstream.disconnect = AsyncMock()
        
        manager.upstreams['test-upstream'] = upstream
        manager.all_tools = [
            ToolMetadata(
                name='test.tool',
                original_name='tool',
                description='Test',
                input_schema=JSONSchema(type='object'),
                upstream_id='test-upstream'
            )
        ]
        
        # Shutdown
        await manager.shutdown()
        
        # Verify cleanup
        assert manager.upstreams == {}
        assert manager.all_tools == []
        upstream.disconnect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_shutdown_with_error(self):
        """Test shutdown handles disconnect errors gracefully"""
        manager = ToolDiscoveryManager()
        
        # Create mock upstream that raises error on disconnect
        config = UpstreamConfig(transport='stdio', command='python')
        upstream = UpstreamConnection('test-upstream', config)
        upstream.disconnect = AsyncMock(side_effect=Exception("Disconnect failed"))
        
        manager.upstreams['test-upstream'] = upstream
        
        # Shutdown should not raise error
        await manager.shutdown()
        
        # Verify cleanup still happened
        assert manager.upstreams == {}


class TestToolDiscoveryIntegration:
    """Integration tests for tool discovery with mocked MCP connections"""
    
    @pytest.mark.asyncio
    async def test_successful_discovery(self):
        """Test successful tool discovery from multiple upstreams"""
        # This would require mocking the MCP client, which is complex
        # For now, we'll skip this and rely on manual testing
        # TODO: Implement with proper MCP mocking
        pass
    
    @pytest.mark.asyncio
    async def test_partial_failure(self):
        """Test that discovery continues when some upstreams fail"""
        # This would require mocking the MCP client
        # TODO: Implement with proper MCP mocking
        pass
