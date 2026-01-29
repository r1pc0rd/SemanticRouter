"""
Tool metadata and schema models for MCP Semantic Router.

This module defines the data models for tools discovered from upstream MCP servers.
"""
from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np
import numpy.typing as npt


@dataclass
class JSONSchema:
    """JSON schema for tool input parameters.
    
    Supports common JSON Schema fields used by MCP servers.
    
    Attributes:
        type: Schema type (e.g., 'object', 'string', 'number')
        properties: Dictionary of property schemas
        required: List of required property names
        additional_properties: Whether additional properties are allowed (JSON Schema field)
        description: Schema description
        items: Schema for array items (for type='array')
        enum: Allowed values (for enum types)
        default: Default value
        additional_fields: Any other schema fields not explicitly modeled
    """
    type: str
    properties: Optional[dict[str, Any]] = None
    required: Optional[list[str]] = None
    additional_properties: Optional[bool] = None
    description: Optional[str] = None
    items: Optional[dict[str, Any]] = None
    enum: Optional[list[Any]] = None
    default: Optional[Any] = None
    additional_fields: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert schema to dictionary representation.
        
        Converts Python field names to JSON Schema field names (e.g., additional_properties -> additionalProperties).
        """
        result: dict[str, Any] = {'type': self.type}
        
        if self.properties is not None:
            result['properties'] = self.properties
        
        if self.required is not None:
            result['required'] = self.required
        
        if self.additional_properties is not None:
            result['additionalProperties'] = self.additional_properties
        
        if self.description is not None:
            result['description'] = self.description
        
        if self.items is not None:
            result['items'] = self.items
        
        if self.enum is not None:
            result['enum'] = self.enum
        
        if self.default is not None:
            result['default'] = self.default
        
        # Add any additional fields
        result.update(self.additional_fields)
        
        return result
    
    @classmethod
    def from_dict(cls, schema_dict: dict[str, Any]) -> 'JSONSchema':
        """Create JSONSchema from dictionary.
        
        Converts JSON Schema field names to Python field names (e.g., additionalProperties -> additional_properties).
        
        Args:
            schema_dict: Dictionary containing schema definition
            
        Returns:
            JSONSchema instance
        """
        # Extract known fields with camelCase to snake_case conversion
        type_val = schema_dict.get('type', 'object')
        properties = schema_dict.get('properties')
        required = schema_dict.get('required')
        additional_properties = schema_dict.get('additionalProperties')
        description = schema_dict.get('description')
        items = schema_dict.get('items')
        enum = schema_dict.get('enum')
        default = schema_dict.get('default')
        
        # Collect additional fields (anything not explicitly handled)
        known_fields = {
            'type', 'properties', 'required', 'additionalProperties',
            'description', 'items', 'enum', 'default'
        }
        additional_fields = {
            k: v for k, v in schema_dict.items()
            if k not in known_fields
        }
        
        return cls(
            type=type_val,
            properties=properties,
            required=required,
            additional_properties=additional_properties,
            description=description,
            items=items,
            enum=enum,
            default=default,
            additional_fields=additional_fields
        )


@dataclass
class ToolMetadata:
    """Metadata for a discovered tool from an upstream MCP server.
    
    Attributes:
        name: Namespaced tool name ({upstream_id}.{original_tool_name})
        original_name: Original tool name from upstream
        description: Tool description from upstream
        input_schema: JSON schema for tool input parameters
        upstream_id: ID of the upstream server providing this tool
        embedding: Pre-computed 384-dimensional embedding vector (set after initialization)
        category_description: Optional category description from config
    """
    name: str
    original_name: str
    description: str
    input_schema: JSONSchema
    upstream_id: str
    embedding: Optional[npt.NDArray[np.float32]] = None
    category_description: Optional[str] = None
    
    def has_embedding(self) -> bool:
        """Check if tool has an embedding."""
        return self.embedding is not None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert tool metadata to dictionary (for serialization).
        
        Note: Embedding is excluded from serialization.
        """
        return {
            'name': self.name,
            'description': self.description,
            'inputSchema': self.input_schema.to_dict()
        }
    
    def get_parameter_names(self) -> list[str]:
        """Get list of parameter names from input schema.
        
        Returns:
            List of parameter names, or empty list if no properties
        """
        if self.input_schema.properties:
            return list(self.input_schema.properties.keys())
        return []


@dataclass
class ContentItem:
    """Content item in tool call result.
    
    Attributes:
        type: Content type ('text', 'image', or 'resource')
        text: Text content (for type='text')
        data: Base64-encoded data (for type='image')
        mime_type: MIME type (for type='image')
        uri: Resource URI (for type='resource')
    """
    type: str  # 'text', 'image', or 'resource'
    text: Optional[str] = None
    data: Optional[str] = None
    mime_type: Optional[str] = None
    uri: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert content item to dictionary."""
        result: dict[str, Any] = {'type': self.type}
        
        if self.text is not None:
            result['text'] = self.text
        if self.data is not None:
            result['data'] = self.data
        if self.mime_type is not None:
            result['mimeType'] = self.mime_type
        if self.uri is not None:
            result['uri'] = self.uri
        
        return result


@dataclass
class ToolCallResult:
    """Result from tool execution.
    
    Attributes:
        content: List of content items in the result
        is_error: Whether the result represents an error
    """
    content: list[ContentItem]
    is_error: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        """Convert tool call result to dictionary."""
        return {
            'content': [item.to_dict() for item in self.content],
            'isError': self.is_error
        }


@dataclass
class SearchResult:
    """Result from semantic search.
    
    Attributes:
        tool: The matched tool metadata
        similarity: Cosine similarity score (0-1)
    """
    tool: ToolMetadata
    similarity: float
    
    def to_dict(self) -> dict[str, Any]:
        """Convert search result to dictionary."""
        tool_dict = self.tool.to_dict()
        tool_dict['similarity'] = self.similarity
        return tool_dict
