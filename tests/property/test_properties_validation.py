"""
Property-based tests for parameter validation and error handling.

Tests Property 25: Invalid tool parameters produce validation error
Tests Property 26: All errors include code and message
Validates: Requirements 10.4, 10.5, 10.6
"""
import pytest
from hypothesis import given, strategies as st, settings, HealthCheck

from mcp_router.core.validation import validate_tool_parameters, validate_search_query
from mcp_router.core.errors import (
    ValidationError,
    ConfigurationError,
    ToolNotFoundError,
    EmbeddingError,
    UpstreamError,
    format_error_response
)
from mcp_router.core.models import JSONSchema


# Strategy for generating valid JSON schema types
json_type_strategy = st.sampled_from(['string', 'number', 'integer', 'boolean', 'array', 'object'])

# Strategy for generating property names
property_name_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='_'),
    min_size=1,
    max_size=30
)


@st.composite
def json_schema_with_required_fields(draw: st.DrawFn) -> tuple[JSONSchema, list[str]]:
    """Generate a JSON schema with required fields."""
    # Generate 1-5 properties
    num_properties = draw(st.integers(min_value=1, max_value=5))
    
    properties = {}
    property_names = []
    
    for _ in range(num_properties):
        prop_name = draw(property_name_strategy)
        # Ensure unique property names
        while prop_name in properties:
            prop_name = draw(property_name_strategy)
        
        prop_type = draw(json_type_strategy)
        properties[prop_name] = {'type': prop_type}
        property_names.append(prop_name)
    
    # Select 1-3 required fields
    num_required = draw(st.integers(min_value=1, max_value=min(3, len(property_names))))
    required_fields = draw(st.lists(
        st.sampled_from(property_names),
        min_size=num_required,
        max_size=num_required,
        unique=True
    ))
    
    schema = JSONSchema(
        type='object',
        properties=properties,
        required=required_fields
    )
    
    return schema, required_fields


@st.composite
def params_missing_required_field(draw: st.DrawFn, schema: JSONSchema, required_fields: list[str]) -> dict:
    """Generate parameters missing at least one required field."""
    # Choose a required field to omit
    omitted_field = draw(st.sampled_from(required_fields))
    
    # Generate valid values for other fields
    params = {}
    for prop_name, prop_schema in schema.properties.items():
        if prop_name == omitted_field:
            continue  # Skip the omitted field
        
        # Generate value based on type
        prop_type = prop_schema.get('type', 'string')
        if prop_type == 'string':
            params[prop_name] = draw(st.text())
        elif prop_type == 'number':
            params[prop_name] = draw(st.floats(allow_nan=False, allow_infinity=False))
        elif prop_type == 'integer':
            params[prop_name] = draw(st.integers())
        elif prop_type == 'boolean':
            params[prop_name] = draw(st.booleans())
        elif prop_type == 'array':
            params[prop_name] = draw(st.lists(st.text(), max_size=5))
        elif prop_type == 'object':
            params[prop_name] = draw(st.dictionaries(st.text(), st.text(), max_size=3))
    
    return params


@st.composite
def params_with_wrong_type(draw: st.DrawFn, schema: JSONSchema) -> dict:
    """Generate parameters with at least one wrong type."""
    params = {}
    
    # Choose a field to give wrong type
    property_names = list(schema.properties.keys())
    wrong_type_field = draw(st.sampled_from(property_names))
    
    for prop_name, prop_schema in schema.properties.items():
        prop_type = prop_schema.get('type', 'string')
        
        if prop_name == wrong_type_field:
            # Generate wrong type
            if prop_type == 'string':
                params[prop_name] = draw(st.integers())
            elif prop_type in ('number', 'integer'):
                params[prop_name] = draw(st.text())
            elif prop_type == 'boolean':
                params[prop_name] = draw(st.text())
            elif prop_type == 'array':
                params[prop_name] = draw(st.text())
            elif prop_type == 'object':
                params[prop_name] = draw(st.text())
        else:
            # Generate correct type
            if prop_type == 'string':
                params[prop_name] = draw(st.text())
            elif prop_type == 'number':
                params[prop_name] = draw(st.floats(allow_nan=False, allow_infinity=False))
            elif prop_type == 'integer':
                params[prop_name] = draw(st.integers())
            elif prop_type == 'boolean':
                params[prop_name] = draw(st.booleans())
            elif prop_type == 'array':
                params[prop_name] = draw(st.lists(st.text(), max_size=5))
            elif prop_type == 'object':
                params[prop_name] = draw(st.dictionaries(st.text(), st.text(), max_size=3))
    
    return params


@st.composite
def params_with_extra_field(draw: st.DrawFn, schema: JSONSchema) -> dict:
    """Generate parameters with an extra field not in schema."""
    params = {}
    
    # Generate valid values for all fields
    for prop_name, prop_schema in schema.properties.items():
        prop_type = prop_schema.get('type', 'string')
        
        if prop_type == 'string':
            params[prop_name] = draw(st.text())
        elif prop_type == 'number':
            params[prop_name] = draw(st.floats(allow_nan=False, allow_infinity=False))
        elif prop_type == 'integer':
            params[prop_name] = draw(st.integers())
        elif prop_type == 'boolean':
            params[prop_name] = draw(st.booleans())
        elif prop_type == 'array':
            params[prop_name] = draw(st.lists(st.text(), max_size=5))
        elif prop_type == 'object':
            params[prop_name] = draw(st.dictionaries(st.text(), st.text(), max_size=3))
    
    # Add an extra field
    extra_field_name = draw(property_name_strategy)
    # Ensure it's not already in schema
    while extra_field_name in schema.properties:
        extra_field_name = draw(property_name_strategy)
    
    params[extra_field_name] = draw(st.text())
    
    return params


@st.composite
def schema_and_params_missing_required(draw: st.DrawFn) -> tuple[JSONSchema, dict]:
    """Generate schema and params missing a required field."""
    schema, required_fields = draw(json_schema_with_required_fields())
    params = draw(params_missing_required_field(schema, required_fields))
    return schema, params


@pytest.mark.property
@given(schema_and_params=schema_and_params_missing_required())
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_property_25_missing_required_field_produces_validation_error(
    schema_and_params: tuple[JSONSchema, dict]
) -> None:
    """
    Property 25: Invalid tool parameters produce validation error.
    
    For any tool call with parameters that are missing required fields, have
    incorrect types, or include extra fields not in the schema, the router SHALL
    return a validation error with code -32602 and a descriptive message.
    
    Validates: Requirements 10.4
    """
    schema, params = schema_and_params
    
    # Validation should raise ValidationError
    with pytest.raises(ValidationError) as exc_info:
        validate_tool_parameters(params, schema)
    
    # Verify error has correct code
    assert exc_info.value.code == -32602
    
    # Verify error has message
    assert exc_info.value.message
    assert isinstance(exc_info.value.message, str)
    assert len(exc_info.value.message) > 0


@st.composite
def schema_and_params_with_wrong_type(draw: st.DrawFn) -> tuple[JSONSchema, dict]:
    """Generate schema and params with wrong type."""
    schema, _ = draw(json_schema_with_required_fields())
    params = draw(params_with_wrong_type(schema))
    return schema, params


@pytest.mark.property
@given(schema_and_params=schema_and_params_with_wrong_type())
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_property_25_wrong_type_produces_validation_error(
    schema_and_params: tuple[JSONSchema, dict]
) -> None:
    """
    Property 25: Invalid tool parameters produce validation error (wrong type).
    
    Validates: Requirements 10.4
    """
    schema, params = schema_and_params
    
    # Validation should raise ValidationError
    with pytest.raises(ValidationError) as exc_info:
        validate_tool_parameters(params, schema)
    
    # Verify error has correct code
    assert exc_info.value.code == -32602
    
    # Verify error has message
    assert exc_info.value.message
    assert isinstance(exc_info.value.message, str)
    assert len(exc_info.value.message) > 0


@st.composite
def schema_and_params_with_extra_field(draw: st.DrawFn) -> tuple[JSONSchema, dict]:
    """Generate schema and params with extra field."""
    schema, _ = draw(json_schema_with_required_fields())
    params = draw(params_with_extra_field(schema))
    return schema, params


@pytest.mark.property
@given(schema_and_params=schema_and_params_with_extra_field())
@settings(max_examples=50, suppress_health_check=[HealthCheck.large_base_example, HealthCheck.too_slow])
def test_property_25_extra_field_produces_validation_error(
    schema_and_params: tuple[JSONSchema, dict]
) -> None:
    """
    Property 25: Invalid tool parameters produce validation error (extra field).
    
    Validates: Requirements 10.4
    """
    schema, params = schema_and_params
    
    # Validation should raise ValidationError
    with pytest.raises(ValidationError) as exc_info:
        validate_tool_parameters(params, schema)
    
    # Verify error has correct code
    assert exc_info.value.code == -32602
    
    # Verify error has message
    assert exc_info.value.message
    assert isinstance(exc_info.value.message, str)
    assert len(exc_info.value.message) > 0


@pytest.mark.property
@given(
    query=st.one_of(
        st.none(),  # Missing query
        st.just(''),  # Empty string
        st.just('   '),  # Whitespace only
        st.just('\t\n'),  # Whitespace characters
    )
)
def test_property_25_invalid_search_query_produces_validation_error(query: str | None) -> None:
    """
    Property 25: Invalid search query produces validation error.
    
    For any search_tools call with a missing, empty, or whitespace-only query,
    the router SHALL return a validation error with code -32602.
    
    Validates: Requirements 10.5
    """
    # Validation should raise ValidationError
    with pytest.raises(ValidationError) as exc_info:
        validate_search_query(query)
    
    # Verify error has correct code
    assert exc_info.value.code == -32602
    
    # Verify error has message
    assert exc_info.value.message
    assert isinstance(exc_info.value.message, str)
    assert len(exc_info.value.message) > 0


@pytest.mark.property
@given(
    error_class=st.sampled_from([
        ValidationError,
        ConfigurationError,
        ToolNotFoundError,
        EmbeddingError,
        UpstreamError
    ]),
    message=st.text(min_size=1, max_size=200),
    request_id=st.one_of(st.none(), st.integers(), st.text(min_size=1, max_size=50))
)
def test_property_26_all_errors_include_code_and_message(
    error_class: type,
    message: str,
    request_id: int | str | None
) -> None:
    """
    Property 26: All errors include code and message.
    
    For any error that occurs during router operation, the error response SHALL
    include a JSON-RPC 2.0 error code and a descriptive message. The response
    SHALL conform to the JSON-RPC 2.0 error format.
    
    Validates: Requirements 10.6
    """
    # Create error instance
    error = error_class(message)
    
    # Format error response
    response = format_error_response(error, request_id)
    
    # Verify response structure
    assert 'jsonrpc' in response
    assert response['jsonrpc'] == '2.0'
    
    assert 'id' in response
    assert response['id'] == request_id
    
    assert 'error' in response
    error_obj = response['error']
    
    # Verify error object has code and message
    assert 'code' in error_obj
    assert isinstance(error_obj['code'], int)
    
    assert 'message' in error_obj
    assert isinstance(error_obj['message'], str)
    assert len(error_obj['message']) > 0


@pytest.mark.property
@given(
    message=st.text(min_size=1, max_size=200),
    request_id=st.one_of(st.none(), st.integers(), st.text(min_size=1, max_size=50))
)
def test_property_26_unknown_errors_formatted_correctly(
    message: str,
    request_id: int | str | None
) -> None:
    """
    Property 26: Unknown errors are formatted correctly.
    
    Even for unknown exception types, the error response SHALL include
    code and message in JSON-RPC 2.0 format.
    
    Validates: Requirements 10.6
    """
    # Create unknown error type
    error = Exception(message)
    
    # Format error response
    response = format_error_response(error, request_id)
    
    # Verify response structure
    assert 'jsonrpc' in response
    assert response['jsonrpc'] == '2.0'
    
    assert 'id' in response
    assert response['id'] == request_id
    
    assert 'error' in response
    error_obj = response['error']
    
    # Verify error object has code and message
    assert 'code' in error_obj
    assert isinstance(error_obj['code'], int)
    assert error_obj['code'] == -32603  # Internal error for unknown exceptions
    
    assert 'message' in error_obj
    assert isinstance(error_obj['message'], str)
    assert len(error_obj['message']) > 0
