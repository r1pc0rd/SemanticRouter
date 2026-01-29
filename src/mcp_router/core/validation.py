"""
Parameter validation for MCP Semantic Router.

This module provides validation functions for tool parameters and search queries.
"""
from typing import Any, Optional

from .errors import ValidationError
from .models import JSONSchema


def validate_tool_parameters(params: dict[str, Any], schema: JSONSchema) -> None:
    """
    Validate tool call parameters against JSON schema.
    
    Args:
        params: Parameters to validate
        schema: JSON schema to validate against
        
    Raises:
        ValidationError: If parameters are invalid
    """
    # Check if schema has properties
    if schema.properties is None:
        # No properties defined - accept any parameters
        return
    
    # Check required fields
    # Convert to list to handle hypothesis LazyStrategy in property tests
    required_fields = list(schema.required) if schema.required is not None else []
    if required_fields:
        for required_field in required_fields:
            if required_field not in params:
                raise ValidationError(
                    f"Missing required parameter: {required_field}",
                    data={"missing_field": required_field}
                )
    
    # Check parameter types
    for param_name, param_value in params.items():
        # Check if parameter is in schema
        if param_name not in schema.properties:
            raise ValidationError(
                f"Unknown parameter: {param_name}",
                data={"unknown_field": param_name}
            )
        
        # Get expected type from schema
        param_schema = schema.properties[param_name]
        expected_type = param_schema.get("type") if isinstance(param_schema, dict) else None
        
        if expected_type:
            # Validate type
            if not _validate_type(param_value, expected_type):
                raise ValidationError(
                    f"Invalid type for parameter '{param_name}': expected {expected_type}, got {type(param_value).__name__}",
                    data={
                        "parameter": param_name,
                        "expected_type": expected_type,
                        "actual_type": type(param_value).__name__
                    }
                )


def validate_search_query(query: Optional[str]) -> None:
    """
    Validate search_tools query parameter.
    
    Args:
        query: Query string to validate
        
    Raises:
        ValidationError: If query is invalid
    """
    # Check if query is missing
    if query is None:
        raise ValidationError(
            "Query parameter is required",
            data={"error": "missing_query"}
        )
    
    # Check if query is not a string
    if not isinstance(query, str):
        raise ValidationError(
            f"Query must be a string, got {type(query).__name__}",
            data={"error": "invalid_type", "type": type(query).__name__}
        )
    
    # Check if query is empty or whitespace
    if not query.strip():
        raise ValidationError(
            "Query cannot be empty or whitespace",
            data={"error": "empty_query"}
        )


def _validate_type(value: Any, expected_type: str) -> bool:
    """
    Validate that a value matches the expected JSON schema type.
    
    Args:
        value: Value to validate
        expected_type: Expected JSON schema type
        
    Returns:
        True if value matches expected type, False otherwise
    """
    # Map JSON schema types to Python types
    type_mapping = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None)
    }
    
    expected_python_type = type_mapping.get(expected_type)
    
    if expected_python_type is None:
        # Unknown type - accept any value
        return True
    
    return isinstance(value, expected_python_type)
