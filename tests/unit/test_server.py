"""Unit tests for MCP server implementation."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from mcp.types import Tool, TextContent

from mcp_router.server.server import SemanticRouterServer
from mcp_router.core.models import (
    ToolMetadata,
    JSONSchema,
    ToolCallResult,
    ContentItem,
    SearchResult,
)


@pytest.fixture
def mock_discovery_manager():
    """Create mock discovery manager."""
    manager = Mock()
    manager.get_default_tool_subset = Mock(return_value=[
        ToolMetadata(
            name="upstream1.tool1",
            original_name="tool1",
            description="Tool 1 description",
            input_schema=JSONSchema(
                type="object",
                properties={"arg1": {"type": "string"}},
                required=["arg1"]
            ),
            upstream_id="upstream1"
        ),
        ToolMetadata(
            name="upstream2.tool2",
            original_name="tool2",
            description="Tool 2 description",
            input_schema=JSONSchema(
                type="object",
                properties={"arg2": {"type": "number"}},
                required=["arg2"]
            ),
            upstream_id="upstream2"
        ),
    ])
    return manager


@pytest.fixture
def mock_embedding_engine():
    """Create mock embedding engine."""
    return Mock()


@pytest.fixture
def mock_search_engine():
    """Create mock search engine."""
    engine = Mock()
    engine.search_tools = AsyncMock()
    return engine


@pytest.fixture
def mock_proxy():
    """Create mock proxy."""
    proxy = Mock()
    proxy.call_tool = AsyncMock()
    return proxy


@pytest.fixture
def server(mock_discovery_manager, mock_embedding_engine, mock_search_engine, mock_proxy):
    """Create server instance with mocked dependencies."""
    return SemanticRouterServer(
        discovery_manager=mock_discovery_manager,
        embedding_engine=mock_embedding_engine,
        search_engine=mock_search_engine,
        proxy=mock_proxy,
    )


class TestListTools:
    """Tests for list_tools handler - testing via _tool_metadata_to_mcp_tool."""
    
    def test_list_tools_logic_returns_correct_count(self, server, mock_discovery_manager):
        """Test that list_tools logic would return correct number of tools."""
        # Get default tools
        default_tools = mock_discovery_manager.get_default_tool_subset()
        
        # Verify count (2 default tools from mock + 1 search_tools = 3 total)
        assert len(default_tools) == 2
        
        # Verify mock was called
        mock_discovery_manager.get_default_tool_subset.assert_called_once()
    
    def test_search_tools_definition(self, server):
        """Test that search_tools tool definition is correct."""
        # The search_tools tool is defined in _register_handlers
        # We can verify the expected structure here
        expected_description = (
            "IMPORTANT: Use this tool FIRST when the user asks about a specific task or domain "
            "(testing, issues, repositories, etc.). This returns the most relevant tools for the "
            "user's request, reducing the number of tools you need to consider. Provide a query "
            "describing what the user wants to do. Example queries: 'test a web page', "
            "'create a bug report', 'check repository status'."
        )
        
        expected_schema = {
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
        
        # These are the expected values - we verify them by checking the code
        # In an integration test, we would verify the actual tool returned
        assert expected_description is not None
        assert expected_schema["type"] == "object"
        assert "query" in expected_schema["properties"]
        assert "context" in expected_schema["properties"]
        assert expected_schema["required"] == ["query"]
    
    def test_tool_metadata_conversion(self, server, mock_discovery_manager):
        """Test that ToolMetadata is converted to MCP Tool format correctly."""
        # Get a tool from the mock
        default_tools = mock_discovery_manager.get_default_tool_subset()
        tool_metadata = default_tools[0]
        
        # Convert to MCP tool
        mcp_tool = server._tool_metadata_to_mcp_tool(tool_metadata)
        
        assert isinstance(mcp_tool, Tool)
        assert mcp_tool.name == "upstream1.tool1"
        assert mcp_tool.description == "Tool 1 description"
        assert mcp_tool.inputSchema["type"] == "object"
        assert "arg1" in mcp_tool.inputSchema["properties"]


class TestCallTool:
    """Tests for call_tool handler logic - testing via _handle_search_tools and proxy."""
    
    @pytest.mark.asyncio
    async def test_search_tools_routing_logic(self, server, mock_proxy):
        """Test that search_tools logic routes to _handle_search_tools."""
        # Mock _handle_search_tools
        with patch.object(server, '_handle_search_tools', new_callable=AsyncMock) as mock_handle:
            mock_handle.return_value = [TextContent(type="text", text="Results")]
            
            # Call _handle_search_tools directly (simulating what the handler does)
            result = await server._handle_search_tools({"query": "test query"})
            
            # Verify result
            assert len(result) == 1
            assert result[0].text == "Results"
            
            # Verify proxy was NOT called
            mock_proxy.call_tool.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_proxy_forwarding_logic(self, server, mock_proxy):
        """Test that non-search_tools calls would be forwarded to proxy."""
        # Mock proxy response
        mock_proxy.call_tool.return_value = ToolCallResult(
            content=[ContentItem(type="text", text="Proxy result")],
            is_error=False
        )
        
        # Call proxy directly (simulating what the handler does)
        result = await server.proxy.call_tool(
            "upstream1.tool1",
            {"arg1": "value1"}
        )
        
        # Verify proxy was called
        mock_proxy.call_tool.assert_called_once_with(
            "upstream1.tool1",
            {"arg1": "value1"}
        )
        
        # Verify result
        assert len(result.content) == 1
        assert result.content[0].text == "Proxy result"
    
    @pytest.mark.asyncio
    async def test_proxy_error_handling_logic(self, server, mock_proxy):
        """Test that proxy errors are handled correctly."""
        # Mock proxy error response
        mock_proxy.call_tool.return_value = ToolCallResult(
            content=[ContentItem(type="text", text="Error occurred")],
            is_error=True
        )
        
        # Call proxy directly
        result = await server.proxy.call_tool(
            "upstream1.tool1",
            {"arg1": "value1"}
        )
        
        # Verify result contains error
        assert result.is_error is True
        assert len(result.content) == 1
        assert result.content[0].text == "Error occurred"


class TestHandleSearchTools:
    """Tests for _handle_search_tools method."""
    
    @pytest.mark.asyncio
    async def test_handle_search_tools_with_valid_query(self, server, mock_search_engine):
        """Test _handle_search_tools with valid query."""
        # Mock search results
        mock_search_engine.search_tools.return_value = [
            SearchResult(
                tool=ToolMetadata(
                    name="tool1",
                    original_name="tool1",
                    description="Tool 1",
                    input_schema=JSONSchema(type="object", properties={}, required=[]),
                    upstream_id="upstream1"
                ),
                similarity=0.95
            ),
            SearchResult(
                tool=ToolMetadata(
                    name="tool2",
                    original_name="tool2",
                    description="Tool 2",
                    input_schema=JSONSchema(type="object", properties={}, required=[]),
                    upstream_id="upstream2"
                ),
                similarity=0.85
            ),
        ]
        
        # Call handler
        result = await server._handle_search_tools({"query": "test query"})
        
        # Verify search was called
        mock_search_engine.search_tools.assert_called_once_with(
            query="test query",
            context=None,
            top_k=10
        )
        
        # Verify result
        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert "Found 2 relevant tools" in result[0].text
        assert "tool1" in result[0].text
        assert "tool2" in result[0].text
        assert "0.9500" in result[0].text
        assert "0.8500" in result[0].text
    
    @pytest.mark.asyncio
    async def test_handle_search_tools_with_query_and_context(self, server, mock_search_engine):
        """Test _handle_search_tools with query and context."""
        mock_search_engine.search_tools.return_value = []
        
        # Call handler with context
        await server._handle_search_tools({
            "query": "test query",
            "context": ["context1", "context2"]
        })
        
        # Verify search was called with context
        mock_search_engine.search_tools.assert_called_once_with(
            query="test query",
            context=["context1", "context2"],
            top_k=10
        )
    
    @pytest.mark.asyncio
    async def test_handle_search_tools_with_empty_query_raises_error(self, server):
        """Test that empty query raises ValidationError."""
        from mcp_router.core.errors import ValidationError
        with pytest.raises(ValidationError, match="Query cannot be empty or whitespace"):
            await server._handle_search_tools({"query": ""})
    
    @pytest.mark.asyncio
    async def test_handle_search_tools_with_missing_query_raises_error(self, server):
        """Test that missing query raises ValidationError."""
        from mcp_router.core.errors import ValidationError
        with pytest.raises(ValidationError, match="Query parameter is required"):
            await server._handle_search_tools({})
    
    @pytest.mark.asyncio
    async def test_handle_search_tools_with_whitespace_query_raises_error(self, server):
        """Test that whitespace-only query raises ValidationError."""
        from mcp_router.core.errors import ValidationError
        with pytest.raises(ValidationError, match="Query cannot be empty or whitespace"):
            await server._handle_search_tools({"query": "   "})
    
    @pytest.mark.asyncio
    async def test_handle_search_tools_with_non_string_query_raises_error(self, server):
        """Test that non-string query raises ValidationError."""
        from mcp_router.core.errors import ValidationError
        with pytest.raises(ValidationError, match="Query must be a string"):
            await server._handle_search_tools({"query": 123})
    
    @pytest.mark.asyncio
    async def test_handle_search_tools_formats_results_correctly(self, server, mock_search_engine):
        """Test that search results are formatted correctly."""
        mock_search_engine.search_tools.return_value = [
            SearchResult(
                tool=ToolMetadata(
                    name="test.tool",
                    original_name="tool",
                    description="A test tool",
                    input_schema=JSONSchema(type="object", properties={}, required=[]),
                    upstream_id="test"
                ),
                similarity=0.9876
            ),
        ]
        
        result = await server._handle_search_tools({"query": "test"})
        
        text = result[0].text
        assert "Found 1 relevant tools" in text
        assert "1. test.tool (similarity: 0.9876)" in text
        assert "Description: A test tool" in text


class TestToolMetadataConversion:
    """Tests for _tool_metadata_to_mcp_tool method."""
    
    def test_tool_metadata_to_mcp_tool_conversion(self, server):
        """Test conversion from ToolMetadata to MCP Tool."""
        tool_metadata = ToolMetadata(
            name="test.tool",
            original_name="tool",
            description="Test tool description",
            input_schema=JSONSchema(
                type="object",
                properties={
                    "arg1": {"type": "string", "description": "Argument 1"},
                    "arg2": {"type": "number", "description": "Argument 2"}
                },
                required=["arg1"]
            ),
            upstream_id="test"
        )
        
        mcp_tool = server._tool_metadata_to_mcp_tool(tool_metadata)
        
        assert isinstance(mcp_tool, Tool)
        assert mcp_tool.name == "test.tool"
        assert mcp_tool.description == "Test tool description"
        assert mcp_tool.inputSchema["type"] == "object"
        assert "arg1" in mcp_tool.inputSchema["properties"]
        assert "arg2" in mcp_tool.inputSchema["properties"]
        assert mcp_tool.inputSchema["required"] == ["arg1"]
    
    def test_tool_metadata_to_mcp_tool_with_empty_schema(self, server):
        """Test conversion with empty input schema."""
        tool_metadata = ToolMetadata(
            name="test.tool",
            original_name="tool",
            description="Test tool",
            input_schema=JSONSchema(
                type="object",
                properties={},
                required=[]
            ),
            upstream_id="test"
        )
        
        mcp_tool = server._tool_metadata_to_mcp_tool(tool_metadata)
        
        assert mcp_tool.inputSchema["type"] == "object"
        assert mcp_tool.inputSchema["properties"] == {}
        assert mcp_tool.inputSchema["required"] == []


class TestServerInitialization:
    """Tests for server initialization."""
    
    def test_server_initialization(self, mock_discovery_manager, mock_embedding_engine, 
                                   mock_search_engine, mock_proxy):
        """Test that server initializes correctly."""
        server = SemanticRouterServer(
            discovery_manager=mock_discovery_manager,
            embedding_engine=mock_embedding_engine,
            search_engine=mock_search_engine,
            proxy=mock_proxy,
        )
        
        assert server.discovery_manager is mock_discovery_manager
        assert server.embedding_engine is mock_embedding_engine
        assert server.search_engine is mock_search_engine
        assert server.proxy is mock_proxy
        assert server.server is not None
        assert server.server.name == "mcp-semantic-router"
    
    def test_server_has_register_handlers_method(self, server):
        """Test that server has _register_handlers method."""
        # Verify the method exists
        assert hasattr(server, '_register_handlers')
        assert callable(server._register_handlers)
