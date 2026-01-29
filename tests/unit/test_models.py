"""
Unit tests for tool metadata and schema models.
"""
import pytest
import numpy as np

from mcp_router.core.models import (
    JSONSchema,
    ToolMetadata,
    ContentItem,
    ToolCallResult,
    SearchResult,
)
from mcp_router.core.namespace import (
    generate_tool_namespace,
    parse_tool_namespace,
    match_upstream_by_prefix,
)
from mcp_router.core.config import UpstreamConfig


class TestJSONSchema:
    """Tests for JSONSchema dataclass."""
    
    def test_basic_schema(self) -> None:
        """Test basic schema creation."""
        schema = JSONSchema(type='object')
        assert schema.type == 'object'
        assert schema.properties is None
        assert schema.required is None
    
    def test_schema_with_properties(self) -> None:
        """Test schema with properties."""
        schema = JSONSchema(
            type='object',
            properties={'url': {'type': 'string'}},
            required=['url']
        )
        assert schema.type == 'object'
        assert schema.properties == {'url': {'type': 'string'}}
        assert schema.required == ['url']
    
    def test_schema_to_dict(self) -> None:
        """Test schema conversion to dictionary."""
        schema = JSONSchema(
            type='object',
            properties={'url': {'type': 'string'}},
            required=['url']
        )
        schema_dict = schema.to_dict()
        
        assert schema_dict['type'] == 'object'
        assert schema_dict['properties'] == {'url': {'type': 'string'}}
        assert schema_dict['required'] == ['url']
    
    def test_schema_from_dict(self) -> None:
        """Test schema creation from dictionary."""
        schema_dict = {
            'type': 'object',
            'properties': {'url': {'type': 'string'}},
            'required': ['url']
        }
        schema = JSONSchema.from_dict(schema_dict)
        
        assert schema.type == 'object'
        assert schema.properties == {'url': {'type': 'string'}}
        assert schema.required == ['url']
    
    def test_schema_with_additional_fields(self) -> None:
        """Test schema with additional JSON Schema fields."""
        schema_dict = {
            'type': 'object',
            'properties': {'url': {'type': 'string'}},
            'required': ['url'],
            'additionalProperties': False,
            'description': 'Test schema',
            'items': {'type': 'string'},
            'enum': ['value1', 'value2'],
            'default': 'value1'
        }
        schema = JSONSchema.from_dict(schema_dict)
        
        # Check explicitly modeled fields
        assert schema.additional_properties is False
        assert schema.description == 'Test schema'
        assert schema.items == {'type': 'string'}
        assert schema.enum == ['value1', 'value2']
        assert schema.default == 'value1'
        
        # Verify fields are included in to_dict with correct naming
        result = schema.to_dict()
        assert result['additionalProperties'] is False  # camelCase in output
        assert result['description'] == 'Test schema'
        assert result['items'] == {'type': 'string'}
        assert result['enum'] == ['value1', 'value2']
        assert result['default'] == 'value1'
    
    def test_schema_with_unknown_fields(self) -> None:
        """Test schema with unknown additional fields."""
        schema_dict = {
            'type': 'object',
            'properties': {'url': {'type': 'string'}},
            'customField': 'custom value',
            'anotherField': 123
        }
        schema = JSONSchema.from_dict(schema_dict)
        
        # Unknown fields should be in additional_fields
        assert schema.additional_fields['customField'] == 'custom value'
        assert schema.additional_fields['anotherField'] == 123
        
        # Verify they're included in to_dict
        result = schema.to_dict()
        assert result['customField'] == 'custom value'
        assert result['anotherField'] == 123
    
    def test_schema_playwright_style(self) -> None:
        """Test schema in Playwright MCP style (real-world example)."""
        # This is the actual format returned by Playwright MCP
        schema_dict = {
            'type': 'object',
            'properties': {
                'url': {
                    'type': 'string',
                    'description': 'URL to navigate to'
                }
            },
            'required': ['url'],
            'additionalProperties': False
        }
        schema = JSONSchema.from_dict(schema_dict)
        
        assert schema.type == 'object'
        assert schema.properties is not None
        assert 'url' in schema.properties
        assert schema.required == ['url']
        assert schema.additional_properties is False
        
        # Verify round-trip conversion
        result = schema.to_dict()
        assert result['type'] == 'object'
        assert result['additionalProperties'] is False
        assert result['required'] == ['url']


class TestToolMetadata:
    """Tests for ToolMetadata dataclass."""
    
    def test_basic_tool_metadata(self) -> None:
        """Test basic tool metadata creation."""
        schema = JSONSchema(type='object')
        tool = ToolMetadata(
            name='browser.navigate',
            original_name='navigate',
            description='Navigate to a URL',
            input_schema=schema,
            upstream_id='playwright'
        )
        
        assert tool.name == 'browser.navigate'
        assert tool.original_name == 'navigate'
        assert tool.description == 'Navigate to a URL'
        assert tool.input_schema == schema
        assert tool.upstream_id == 'playwright'
        assert tool.embedding is None
        assert tool.category_description is None
    
    def test_tool_with_embedding(self) -> None:
        """Test tool metadata with embedding."""
        schema = JSONSchema(type='object')
        embedding = np.random.rand(384).astype(np.float32)
        
        tool = ToolMetadata(
            name='browser.navigate',
            original_name='navigate',
            description='Navigate to a URL',
            input_schema=schema,
            upstream_id='playwright',
            embedding=embedding
        )
        
        assert tool.has_embedding()
        assert np.array_equal(tool.embedding, embedding)
    
    def test_tool_without_embedding(self) -> None:
        """Test tool metadata without embedding."""
        schema = JSONSchema(type='object')
        tool = ToolMetadata(
            name='browser.navigate',
            original_name='navigate',
            description='Navigate to a URL',
            input_schema=schema,
            upstream_id='playwright'
        )
        
        assert not tool.has_embedding()
    
    def test_tool_to_dict(self) -> None:
        """Test tool metadata conversion to dictionary."""
        schema = JSONSchema(
            type='object',
            properties={'url': {'type': 'string'}},
            required=['url']
        )
        tool = ToolMetadata(
            name='browser.navigate',
            original_name='navigate',
            description='Navigate to a URL',
            input_schema=schema,
            upstream_id='playwright'
        )
        
        tool_dict = tool.to_dict()
        
        assert tool_dict['name'] == 'browser.navigate'
        assert tool_dict['description'] == 'Navigate to a URL'
        assert 'inputSchema' in tool_dict
        assert tool_dict['inputSchema']['type'] == 'object'
        # Embedding should not be in dict
        assert 'embedding' not in tool_dict
    
    def test_get_parameter_names(self) -> None:
        """Test getting parameter names from schema."""
        schema = JSONSchema(
            type='object',
            properties={
                'url': {'type': 'string'},
                'timeout': {'type': 'number'}
            }
        )
        tool = ToolMetadata(
            name='browser.navigate',
            original_name='navigate',
            description='Navigate to a URL',
            input_schema=schema,
            upstream_id='playwright'
        )
        
        param_names = tool.get_parameter_names()
        assert set(param_names) == {'url', 'timeout'}
    
    def test_get_parameter_names_empty(self) -> None:
        """Test getting parameter names when schema has no properties."""
        schema = JSONSchema(type='object')
        tool = ToolMetadata(
            name='browser.navigate',
            original_name='navigate',
            description='Navigate to a URL',
            input_schema=schema,
            upstream_id='playwright'
        )
        
        param_names = tool.get_parameter_names()
        assert param_names == []


class TestNamespaceGeneration:
    """Tests for namespace generation functions."""
    
    def test_generate_namespace_without_prefix(self) -> None:
        """Test namespace generation without semantic prefix."""
        config = UpstreamConfig(transport='stdio', command='test')
        namespace = generate_tool_namespace('playwright', 'navigate', config)
        
        assert namespace == 'playwright.navigate'
    
    def test_generate_namespace_with_prefix(self) -> None:
        """Test namespace generation with semantic prefix."""
        config = UpstreamConfig(
            transport='stdio',
            command='test',
            semantic_prefix='browser'
        )
        namespace = generate_tool_namespace('playwright', 'navigate', config)
        
        assert namespace == 'browser.navigate'
    
    def test_parse_namespace_valid(self) -> None:
        """Test parsing valid namespaced name."""
        prefix, tool_name = parse_tool_namespace('browser.navigate')
        
        assert prefix == 'browser'
        assert tool_name == 'navigate'
    
    def test_parse_namespace_with_dots_in_tool_name(self) -> None:
        """Test parsing namespace where tool name contains dots."""
        prefix, tool_name = parse_tool_namespace('browser.navigate.to.url')
        
        assert prefix == 'browser'
        assert tool_name == 'navigate.to.url'
    
    def test_parse_namespace_invalid_no_dot(self) -> None:
        """Test parsing invalid namespace without dot."""
        with pytest.raises(ValueError, match="must be namespaced"):
            parse_tool_namespace('navigate')
    
    def test_parse_namespace_invalid_empty_prefix(self) -> None:
        """Test parsing invalid namespace with empty prefix."""
        with pytest.raises(ValueError, match="must be namespaced"):
            parse_tool_namespace('.navigate')
    
    def test_parse_namespace_invalid_empty_tool_name(self) -> None:
        """Test parsing invalid namespace with empty tool name."""
        with pytest.raises(ValueError, match="must be namespaced"):
            parse_tool_namespace('browser.')
    
    def test_match_upstream_by_id(self) -> None:
        """Test matching upstream by upstream_id."""
        config = UpstreamConfig(transport='stdio', command='test')
        configs = {'playwright': config}
        
        matched = match_upstream_by_prefix('playwright', configs)
        assert matched == 'playwright'
    
    def test_match_upstream_by_semantic_prefix(self) -> None:
        """Test matching upstream by semantic_prefix."""
        config = UpstreamConfig(
            transport='stdio',
            command='test',
            semantic_prefix='browser'
        )
        configs = {'playwright': config}
        
        matched = match_upstream_by_prefix('browser', configs)
        assert matched == 'playwright'
    
    def test_match_upstream_not_found(self) -> None:
        """Test matching upstream that doesn't exist."""
        config = UpstreamConfig(transport='stdio', command='test')
        configs = {'playwright': config}
        
        matched = match_upstream_by_prefix('unknown', configs)
        assert matched is None


class TestContentItem:
    """Tests for ContentItem dataclass."""
    
    def test_text_content(self) -> None:
        """Test text content item."""
        item = ContentItem(type='text', text='Hello, world!')
        
        assert item.type == 'text'
        assert item.text == 'Hello, world!'
        
        item_dict = item.to_dict()
        assert item_dict['type'] == 'text'
        assert item_dict['text'] == 'Hello, world!'
    
    def test_image_content(self) -> None:
        """Test image content item."""
        item = ContentItem(
            type='image',
            data='base64data',
            mime_type='image/png'
        )
        
        assert item.type == 'image'
        assert item.data == 'base64data'
        assert item.mime_type == 'image/png'
    
    def test_resource_content(self) -> None:
        """Test resource content item."""
        item = ContentItem(type='resource', uri='file:///path/to/resource')
        
        assert item.type == 'resource'
        assert item.uri == 'file:///path/to/resource'


class TestToolCallResult:
    """Tests for ToolCallResult dataclass."""
    
    def test_successful_result(self) -> None:
        """Test successful tool call result."""
        content = [ContentItem(type='text', text='Success')]
        result = ToolCallResult(content=content)
        
        assert result.content == content
        assert not result.is_error
    
    def test_error_result(self) -> None:
        """Test error tool call result."""
        content = [ContentItem(type='text', text='Error occurred')]
        result = ToolCallResult(content=content, is_error=True)
        
        assert result.content == content
        assert result.is_error
    
    def test_result_to_dict(self) -> None:
        """Test tool call result conversion to dictionary."""
        content = [ContentItem(type='text', text='Success')]
        result = ToolCallResult(content=content)
        
        result_dict = result.to_dict()
        assert 'content' in result_dict
        assert len(result_dict['content']) == 1
        assert result_dict['content'][0]['type'] == 'text'
        assert result_dict['isError'] is False


class TestSearchResult:
    """Tests for SearchResult dataclass."""
    
    def test_search_result(self) -> None:
        """Test search result creation."""
        schema = JSONSchema(type='object')
        tool = ToolMetadata(
            name='browser.navigate',
            original_name='navigate',
            description='Navigate to a URL',
            input_schema=schema,
            upstream_id='playwright'
        )
        
        result = SearchResult(tool=tool, similarity=0.85)
        
        assert result.tool == tool
        assert result.similarity == 0.85
    
    def test_search_result_to_dict(self) -> None:
        """Test search result conversion to dictionary."""
        schema = JSONSchema(type='object')
        tool = ToolMetadata(
            name='browser.navigate',
            original_name='navigate',
            description='Navigate to a URL',
            input_schema=schema,
            upstream_id='playwright'
        )
        
        result = SearchResult(tool=tool, similarity=0.85)
        result_dict = result.to_dict()
        
        assert result_dict['name'] == 'browser.navigate'
        assert result_dict['description'] == 'Navigate to a URL'
        assert result_dict['similarity'] == 0.85
