"""
Unit tests for tool call proxy.

Tests tool call forwarding, error handling, and connection reuse.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
import asyncio

from mcp_router.core.config import UpstreamConfig
from mcp_router.core.models import ToolCallResult, ContentItem
from mcp_router.proxy.proxy import ToolCallProxy
from mcp_router.discovery.upstream import UpstreamConnection


class TestToolCallProxy:
    """Tests for ToolCallProxy class"""
    
    def test_initialization(self):
        """Test proxy initialization"""
        config = UpstreamConfig(transport='stdio', command='test')
        upstream = UpstreamConnection('test-upstream', config)
        
        proxy = ToolCallProxy(
            upstreams={'test-upstream': upstream},
            upstream_configs={'test-upstream': config}
        )
        
        assert proxy.upstreams == {'test-upstream': upstream}
        assert proxy.upstream_configs == {'test-upstream': config}
    
    @pytest.mark.asyncio
    async def test_successful_tool_call(self):
        """Test successful tool call forwarding"""
        # Create mock upstream
        config = UpstreamConfig(transport='stdio', command='test')
        upstream = UpstreamConnection('test-upstream', config)
        upstream._session = MagicMock()  # Mark as connected
        
        # Mock call_tool to return a result
        upstream.call_tool = AsyncMock(return_value={
            'content': [
                {'type': 'text', 'text': 'Hello, world!'}
            ],
            'isError': False
        })
        
        # Create proxy
        proxy = ToolCallProxy(
            upstreams={'test-upstream': upstream},
            upstream_configs={'test-upstream': config}
        )
        
        # Call tool
        result = await proxy.call_tool(
            'test-upstream.greet',
            {'name': 'Alice'}
        )
        
        # Verify call was forwarded
        upstream.call_tool.assert_called_once_with('greet', {'name': 'Alice'})
        
        # Verify result
        assert isinstance(result, ToolCallResult)
        assert len(result.content) == 1
        assert result.content[0].type == 'text'
        assert result.content[0].text == 'Hello, world!'
        assert result.is_error is False
    
    @pytest.mark.asyncio
    async def test_tool_call_with_semantic_prefix(self):
        """Test tool call with semantic prefix"""
        # Create config with semantic prefix
        config = UpstreamConfig(
            transport='stdio',
            command='test',
            semantic_prefix='browser'
        )
        upstream = UpstreamConnection('test-upstream', config)
        upstream._session = MagicMock()
        upstream.call_tool = AsyncMock(return_value={
            'content': [{'type': 'text', 'text': 'Navigated'}],
            'isError': False
        })
        
        # Create proxy
        proxy = ToolCallProxy(
            upstreams={'test-upstream': upstream},
            upstream_configs={'test-upstream': config}
        )
        
        # Call tool using semantic prefix
        result = await proxy.call_tool('browser.navigate', {'url': 'https://example.com'})
        
        # Verify correct upstream was called
        upstream.call_tool.assert_called_once_with('navigate', {'url': 'https://example.com'})
        assert isinstance(result, ToolCallResult)
    
    @pytest.mark.asyncio
    async def test_invalid_namespace_no_dot(self):
        """Test error handling for namespace without dot"""
        proxy = ToolCallProxy(upstreams={}, upstream_configs={})
        
        with pytest.raises(ValueError, match="Invalid tool namespace"):
            await proxy.call_tool('invalid_tool_name', {})
    
    @pytest.mark.asyncio
    async def test_invalid_namespace_empty_prefix(self):
        """Test error handling for namespace with empty prefix"""
        proxy = ToolCallProxy(upstreams={}, upstream_configs={})
        
        with pytest.raises(ValueError, match="Invalid tool namespace"):
            await proxy.call_tool('.tool_name', {})
    
    @pytest.mark.asyncio
    async def test_invalid_namespace_empty_tool_name(self):
        """Test error handling for namespace with empty tool name"""
        proxy = ToolCallProxy(upstreams={}, upstream_configs={})
        
        with pytest.raises(ValueError, match="Invalid tool namespace"):
            await proxy.call_tool('prefix.', {})
    
    @pytest.mark.asyncio
    async def test_unknown_upstream_prefix(self):
        """Test error handling for unknown upstream prefix"""
        config = UpstreamConfig(transport='stdio', command='test')
        upstream = UpstreamConnection('known-upstream', config)
        
        proxy = ToolCallProxy(
            upstreams={'known-upstream': upstream},
            upstream_configs={'known-upstream': config}
        )
        
        with pytest.raises(ValueError, match="No upstream found"):
            await proxy.call_tool('unknown.tool', {})
    
    @pytest.mark.asyncio
    async def test_upstream_not_connected(self):
        """Test error handling when upstream is not connected"""
        config = UpstreamConfig(transport='stdio', command='test')
        upstream = UpstreamConnection('test-upstream', config)
        # Don't set _session, so is_connected will be False
        
        proxy = ToolCallProxy(
            upstreams={'test-upstream': upstream},
            upstream_configs={'test-upstream': config}
        )
        
        with pytest.raises(RuntimeError, match="not connected"):
            await proxy.call_tool('test-upstream.tool', {})
    
    @pytest.mark.asyncio
    async def test_upstream_missing_from_connections(self):
        """Test error handling when upstream is in config but not connected"""
        config = UpstreamConfig(transport='stdio', command='test')
        
        # Upstream in config but not in upstreams dict
        proxy = ToolCallProxy(
            upstreams={},
            upstream_configs={'test-upstream': config}
        )
        
        with pytest.raises(RuntimeError, match="not connected"):
            await proxy.call_tool('test-upstream.tool', {})
    
    @pytest.mark.asyncio
    async def test_upstream_error_forwarding(self):
        """Test that upstream errors are forwarded to client"""
        config = UpstreamConfig(transport='stdio', command='test')
        upstream = UpstreamConnection('test-upstream', config)
        upstream._session = MagicMock()
        
        # Mock call_tool to raise error
        upstream.call_tool = AsyncMock(
            side_effect=RuntimeError("Tool execution failed")
        )
        
        proxy = ToolCallProxy(
            upstreams={'test-upstream': upstream},
            upstream_configs={'test-upstream': config}
        )
        
        with pytest.raises(RuntimeError, match="Upstream error"):
            await proxy.call_tool('test-upstream.failing_tool', {})
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout handling for slow tool calls"""
        config = UpstreamConfig(transport='stdio', command='test')
        upstream = UpstreamConnection('test-upstream', config)
        upstream._session = MagicMock()
        
        # Mock call_tool to be slow
        async def slow_call(*args, **kwargs):
            await asyncio.sleep(10)
            return {'content': [], 'isError': False}
        
        upstream.call_tool = AsyncMock(side_effect=slow_call)
        
        proxy = ToolCallProxy(
            upstreams={'test-upstream': upstream},
            upstream_configs={'test-upstream': config}
        )
        
        # Call with short timeout
        with pytest.raises(asyncio.TimeoutError, match="timed out"):
            await proxy.call_tool('test-upstream.slow_tool', {}, timeout=0.1)
    
    @pytest.mark.asyncio
    async def test_connection_reuse(self):
        """Test that connections are reused for multiple calls"""
        config = UpstreamConfig(transport='stdio', command='test')
        upstream = UpstreamConnection('test-upstream', config)
        upstream._session = MagicMock()
        upstream.call_tool = AsyncMock(return_value={
            'content': [{'type': 'text', 'text': 'result'}],
            'isError': False
        })
        
        proxy = ToolCallProxy(
            upstreams={'test-upstream': upstream},
            upstream_configs={'test-upstream': config}
        )
        
        # Make multiple calls
        await proxy.call_tool('test-upstream.tool1', {})
        await proxy.call_tool('test-upstream.tool2', {})
        await proxy.call_tool('test-upstream.tool3', {})
        
        # Verify all calls used the same upstream
        assert upstream.call_tool.call_count == 3
        
        # Verify the same upstream instance is still in proxy
        assert proxy.upstreams['test-upstream'] is upstream
    
    @pytest.mark.asyncio
    async def test_multiple_content_items(self):
        """Test handling of multiple content items in response"""
        config = UpstreamConfig(transport='stdio', command='test')
        upstream = UpstreamConnection('test-upstream', config)
        upstream._session = MagicMock()
        
        # Mock call_tool to return multiple content items
        upstream.call_tool = AsyncMock(return_value={
            'content': [
                {'type': 'text', 'text': 'First item'},
                {'type': 'text', 'text': 'Second item'},
                {'type': 'resource', 'uri': 'file:///test.txt', 'mimeType': 'text/plain'}
            ],
            'isError': False
        })
        
        proxy = ToolCallProxy(
            upstreams={'test-upstream': upstream},
            upstream_configs={'test-upstream': config}
        )
        
        result = await proxy.call_tool('test-upstream.multi_content', {})
        
        # Verify all content items are present
        assert len(result.content) == 3
        assert result.content[0].type == 'text'
        assert result.content[0].text == 'First item'
        assert result.content[1].type == 'text'
        assert result.content[1].text == 'Second item'
        assert result.content[2].type == 'resource'
        assert result.content[2].uri == 'file:///test.txt'
        assert result.content[2].mime_type == 'text/plain'
    
    @pytest.mark.asyncio
    async def test_error_result_from_upstream(self):
        """Test handling of error results from upstream"""
        config = UpstreamConfig(transport='stdio', command='test')
        upstream = UpstreamConnection('test-upstream', config)
        upstream._session = MagicMock()
        
        # Mock call_tool to return error result
        upstream.call_tool = AsyncMock(return_value={
            'content': [{'type': 'text', 'text': 'Error: Invalid parameter'}],
            'isError': True
        })
        
        proxy = ToolCallProxy(
            upstreams={'test-upstream': upstream},
            upstream_configs={'test-upstream': config}
        )
        
        result = await proxy.call_tool('test-upstream.error_tool', {})
        
        # Verify error flag is preserved
        assert result.is_error is True
        assert len(result.content) == 1
        assert result.content[0].text == 'Error: Invalid parameter'
    
    @pytest.mark.asyncio
    async def test_empty_arguments(self):
        """Test tool call with empty arguments"""
        config = UpstreamConfig(transport='stdio', command='test')
        upstream = UpstreamConnection('test-upstream', config)
        upstream._session = MagicMock()
        upstream.call_tool = AsyncMock(return_value={
            'content': [{'type': 'text', 'text': 'No args needed'}],
            'isError': False
        })
        
        proxy = ToolCallProxy(
            upstreams={'test-upstream': upstream},
            upstream_configs={'test-upstream': config}
        )
        
        result = await proxy.call_tool('test-upstream.no_args_tool', {})
        
        # Verify empty dict was passed
        upstream.call_tool.assert_called_once_with('no_args_tool', {})
        assert isinstance(result, ToolCallResult)
    
    @pytest.mark.asyncio
    async def test_complex_arguments(self):
        """Test tool call with complex nested arguments"""
        config = UpstreamConfig(transport='stdio', command='test')
        upstream = UpstreamConnection('test-upstream', config)
        upstream._session = MagicMock()
        upstream.call_tool = AsyncMock(return_value={
            'content': [{'type': 'text', 'text': 'Processed'}],
            'isError': False
        })
        
        proxy = ToolCallProxy(
            upstreams={'test-upstream': upstream},
            upstream_configs={'test-upstream': config}
        )
        
        # Complex nested arguments
        args = {
            'user': {
                'name': 'Alice',
                'age': 30,
                'preferences': ['reading', 'coding']
            },
            'options': {
                'verbose': True,
                'timeout': 60
            }
        }
        
        result = await proxy.call_tool('test-upstream.complex_tool', args)
        
        # Verify arguments were passed without modification
        upstream.call_tool.assert_called_once_with('complex_tool', args)
        assert isinstance(result, ToolCallResult)
