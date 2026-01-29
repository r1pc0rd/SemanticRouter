"""
Unit tests for error handling and validation.

Tests error handling across all components:
- Configuration errors (10.1)
- Upstream errors (10.2)
- Embedding errors (10.3)
- Validation errors (10.4, 10.5)
- Error format (10.6)
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock

from mcp_router.core.errors import (
    ValidationError,
    ConfigurationError,
    ToolNotFoundError,
    EmbeddingError,
    UpstreamError,
    format_error_response,
    ERROR_CODE_PARSE_ERROR,
    ERROR_CODE_INVALID_REQUEST,
    ERROR_CODE_METHOD_NOT_FOUND,
    ERROR_CODE_INVALID_PARAMS,
    ERROR_CODE_INTERNAL_ERROR,
    ERROR_CODE_SERVER_ERROR,
)
from mcp_router.core.validation import validate_tool_parameters, validate_search_query
from mcp_router.core.models import JSONSchema


# ============================================================================
# Configuration Error Tests (10.1)
# ============================================================================

def test_configuration_error_has_correct_code():
    """Configuration errors should have code -32600 (Invalid Request)."""
    error = ConfigurationError("Missing required field: upstreams")
    
    assert error.code == ERROR_CODE_INVALID_REQUEST
    assert error.message == "Missing required field: upstreams"


def test_configuration_error_with_data():
    """Configuration errors can include additional data."""
    error = ConfigurationError(
        "Invalid upstream config",
        data={"upstream_id": "test", "field": "command"}
    )
    
    assert error.code == ERROR_CODE_INVALID_REQUEST
    assert error.data == {"upstream_id": "test", "field": "command"}


def test_configuration_error_formatted_correctly():
    """Configuration errors should format as JSON-RPC 2.0 errors."""
    error = ConfigurationError("Config file not found")
    response = format_error_response(error, request_id=1)
    
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 1
    assert response["error"]["code"] == ERROR_CODE_INVALID_REQUEST
    assert response["error"]["message"] == "Config file not found"


# ============================================================================
# Upstream Error Tests (10.2)
# ============================================================================

def test_upstream_error_has_correct_code():
    """Upstream errors should have code -32000 (Server Error)."""
    error = UpstreamError("Connection timeout")
    
    assert error.code == ERROR_CODE_SERVER_ERROR
    assert error.message == "Connection timeout"


def test_upstream_error_with_upstream_details():
    """Upstream errors can include upstream server details."""
    error = UpstreamError(
        "Upstream server returned error",
        data={
            "upstream_id": "playwright",
            "original_error": "Browser not installed"
        }
    )
    
    assert error.code == ERROR_CODE_SERVER_ERROR
    assert error.data["upstream_id"] == "playwright"
    assert error.data["original_error"] == "Browser not installed"


def test_upstream_error_formatted_correctly():
    """Upstream errors should format as JSON-RPC 2.0 errors."""
    error = UpstreamError("Connection lost", data={"upstream_id": "test"})
    response = format_error_response(error, request_id="abc-123")
    
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == "abc-123"
    assert response["error"]["code"] == ERROR_CODE_SERVER_ERROR
    assert response["error"]["message"] == "Connection lost"
    assert response["error"]["data"]["upstream_id"] == "test"


# ============================================================================
# Embedding Error Tests (10.3)
# ============================================================================

def test_embedding_error_has_correct_code():
    """Embedding errors should have code -32603 (Internal Error)."""
    error = EmbeddingError("Model failed to load")
    
    assert error.code == ERROR_CODE_INTERNAL_ERROR
    assert error.message == "Model failed to load"


def test_embedding_error_with_details():
    """Embedding errors can include error details."""
    error = EmbeddingError(
        "Embedding generation failed",
        data={"query": "test query", "error": "Out of memory"}
    )
    
    assert error.code == ERROR_CODE_INTERNAL_ERROR
    assert error.data["query"] == "test query"
    assert error.data["error"] == "Out of memory"


def test_embedding_error_formatted_correctly():
    """Embedding errors should format as JSON-RPC 2.0 errors."""
    error = EmbeddingError("Cosine similarity failed")
    response = format_error_response(error, request_id=42)
    
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 42
    assert response["error"]["code"] == ERROR_CODE_INTERNAL_ERROR
    assert response["error"]["message"] == "Cosine similarity failed"


# ============================================================================
# Validation Error Tests - Tool Parameters (10.4)
# ============================================================================

def test_validation_error_has_correct_code():
    """Validation errors should have code -32602 (Invalid Params)."""
    error = ValidationError("Missing required parameter: url")
    
    assert error.code == ERROR_CODE_INVALID_PARAMS
    assert error.message == "Missing required parameter: url"


def test_validate_tool_parameters_missing_required_field():
    """Validation should fail when required field is missing."""
    schema = JSONSchema(
        type="object",
        properties={
            "url": {"type": "string"},
            "timeout": {"type": "number"}
        },
        required=["url"]
    )
    
    params = {"timeout": 30}  # Missing 'url'
    
    with pytest.raises(ValidationError) as exc_info:
        validate_tool_parameters(params, schema)
    
    assert exc_info.value.code == ERROR_CODE_INVALID_PARAMS
    assert "url" in exc_info.value.message.lower()
    assert exc_info.value.data["missing_field"] == "url"


def test_validate_tool_parameters_wrong_type():
    """Validation should fail when parameter has wrong type."""
    schema = JSONSchema(
        type="object",
        properties={
            "count": {"type": "integer"},
            "name": {"type": "string"}
        },
        required=["count"]
    )
    
    params = {"count": "not a number", "name": "test"}  # Wrong type for 'count'
    
    with pytest.raises(ValidationError) as exc_info:
        validate_tool_parameters(params, schema)
    
    assert exc_info.value.code == ERROR_CODE_INVALID_PARAMS
    assert "count" in exc_info.value.message.lower()
    assert exc_info.value.data["parameter"] == "count"
    assert exc_info.value.data["expected_type"] == "integer"


def test_validate_tool_parameters_extra_field():
    """Validation should fail when extra field is provided."""
    schema = JSONSchema(
        type="object",
        properties={
            "url": {"type": "string"}
        },
        required=["url"]
    )
    
    params = {"url": "https://example.com", "extra": "field"}  # Extra field
    
    with pytest.raises(ValidationError) as exc_info:
        validate_tool_parameters(params, schema)
    
    assert exc_info.value.code == ERROR_CODE_INVALID_PARAMS
    assert "extra" in exc_info.value.message.lower()
    assert exc_info.value.data["unknown_field"] == "extra"


def test_validate_tool_parameters_accepts_valid_params():
    """Validation should pass for valid parameters."""
    schema = JSONSchema(
        type="object",
        properties={
            "url": {"type": "string"},
            "timeout": {"type": "number"}
        },
        required=["url"]
    )
    
    params = {"url": "https://example.com", "timeout": 30}
    
    # Should not raise
    validate_tool_parameters(params, schema)


def test_validate_tool_parameters_no_properties_accepts_any():
    """Validation should accept any parameters when schema has no properties."""
    schema = JSONSchema(type="object", properties=None, required=None)
    
    params = {"anything": "goes", "foo": 123}
    
    # Should not raise
    validate_tool_parameters(params, schema)


# ============================================================================
# Validation Error Tests - Search Query (10.5)
# ============================================================================

def test_validate_search_query_missing():
    """Validation should fail when query is None."""
    with pytest.raises(ValidationError) as exc_info:
        validate_search_query(None)
    
    assert exc_info.value.code == ERROR_CODE_INVALID_PARAMS
    assert "required" in exc_info.value.message.lower()
    assert exc_info.value.data["error"] == "missing_query"


def test_validate_search_query_empty_string():
    """Validation should fail when query is empty string."""
    with pytest.raises(ValidationError) as exc_info:
        validate_search_query("")
    
    assert exc_info.value.code == ERROR_CODE_INVALID_PARAMS
    assert "empty" in exc_info.value.message.lower()
    assert exc_info.value.data["error"] == "empty_query"


def test_validate_search_query_whitespace_only():
    """Validation should fail when query is only whitespace."""
    with pytest.raises(ValidationError) as exc_info:
        validate_search_query("   \t\n  ")
    
    assert exc_info.value.code == ERROR_CODE_INVALID_PARAMS
    assert "empty" in exc_info.value.message.lower()
    assert exc_info.value.data["error"] == "empty_query"


def test_validate_search_query_wrong_type():
    """Validation should fail when query is not a string."""
    with pytest.raises(ValidationError) as exc_info:
        validate_search_query(123)  # type: ignore
    
    assert exc_info.value.code == ERROR_CODE_INVALID_PARAMS
    assert "string" in exc_info.value.message.lower()
    assert exc_info.value.data["error"] == "invalid_type"


def test_validate_search_query_accepts_valid_query():
    """Validation should pass for valid query."""
    # Should not raise
    validate_search_query("test a web page")
    validate_search_query("  valid query with spaces  ")


# ============================================================================
# Tool Not Found Error Tests
# ============================================================================

def test_tool_not_found_error_has_correct_code():
    """Tool not found errors should have code -32601 (Method Not Found)."""
    error = ToolNotFoundError("Tool not found: unknown.tool")
    
    assert error.code == ERROR_CODE_METHOD_NOT_FOUND
    assert error.message == "Tool not found: unknown.tool"


def test_tool_not_found_error_with_tool_name():
    """Tool not found errors can include tool name."""
    error = ToolNotFoundError(
        "Tool not found",
        data={"tool_name": "playwright.navigate"}
    )
    
    assert error.code == ERROR_CODE_METHOD_NOT_FOUND
    assert error.data["tool_name"] == "playwright.navigate"


def test_tool_not_found_error_formatted_correctly():
    """Tool not found errors should format as JSON-RPC 2.0 errors."""
    error = ToolNotFoundError("Tool not found: test.tool")
    response = format_error_response(error, request_id="req-1")
    
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == "req-1"
    assert response["error"]["code"] == ERROR_CODE_METHOD_NOT_FOUND
    assert response["error"]["message"] == "Tool not found: test.tool"


# ============================================================================
# Error Format Tests (10.6)
# ============================================================================

def test_format_error_response_includes_jsonrpc_version():
    """All error responses should include jsonrpc version 2.0."""
    error = ValidationError("Test error")
    response = format_error_response(error, request_id=1)
    
    assert "jsonrpc" in response
    assert response["jsonrpc"] == "2.0"


def test_format_error_response_includes_request_id():
    """All error responses should include request ID."""
    error = ValidationError("Test error")
    
    # Test with integer ID
    response = format_error_response(error, request_id=42)
    assert response["id"] == 42
    
    # Test with string ID
    response = format_error_response(error, request_id="abc-123")
    assert response["id"] == "abc-123"
    
    # Test with None ID
    response = format_error_response(error, request_id=None)
    assert response["id"] is None


def test_format_error_response_includes_error_object():
    """All error responses should include error object with code and message."""
    error = ValidationError("Test error")
    response = format_error_response(error, request_id=1)
    
    assert "error" in response
    error_obj = response["error"]
    
    assert "code" in error_obj
    assert isinstance(error_obj["code"], int)
    
    assert "message" in error_obj
    assert isinstance(error_obj["message"], str)
    assert len(error_obj["message"]) > 0


def test_format_error_response_includes_data_when_present():
    """Error responses should include data field when error has data."""
    error = ValidationError(
        "Invalid parameter",
        data={"parameter": "url", "reason": "malformed"}
    )
    response = format_error_response(error, request_id=1)
    
    assert "data" in response["error"]
    assert response["error"]["data"]["parameter"] == "url"
    assert response["error"]["data"]["reason"] == "malformed"


def test_format_error_response_omits_data_when_absent():
    """Error responses should omit data field when error has no data."""
    error = ValidationError("Invalid parameter")
    response = format_error_response(error, request_id=1)
    
    # Data field should not be present
    assert "data" not in response["error"]


def test_format_error_response_handles_unknown_exceptions():
    """Unknown exceptions should be formatted as internal errors."""
    error = RuntimeError("Something went wrong")
    response = format_error_response(error, request_id=1)
    
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 1
    assert response["error"]["code"] == ERROR_CODE_INTERNAL_ERROR
    assert response["error"]["message"] == "Something went wrong"
    assert response["error"]["data"]["type"] == "RuntimeError"


def test_format_error_response_preserves_error_codes():
    """Error formatter should preserve error codes from RouterError subclasses."""
    test_cases = [
        (ConfigurationError("Config error"), ERROR_CODE_INVALID_REQUEST),
        (ValidationError("Validation error"), ERROR_CODE_INVALID_PARAMS),
        (ToolNotFoundError("Tool not found"), ERROR_CODE_METHOD_NOT_FOUND),
        (EmbeddingError("Embedding error"), ERROR_CODE_INTERNAL_ERROR),
        (UpstreamError("Upstream error"), ERROR_CODE_SERVER_ERROR),
    ]
    
    for error, expected_code in test_cases:
        response = format_error_response(error, request_id=1)
        assert response["error"]["code"] == expected_code, \
            f"Expected code {expected_code} for {type(error).__name__}"


# ============================================================================
# Integration Tests - Error Handling in Context
# ============================================================================

def test_validation_error_formatted_as_json_rpc_error():
    """Validation errors should be formatted as JSON-RPC 2.0 errors."""
    schema = JSONSchema(
        type="object",
        properties={"url": {"type": "string"}},
        required=["url"]
    )
    
    params = {}  # Missing required field
    
    try:
        validate_tool_parameters(params, schema)
        pytest.fail("Expected ValidationError")
    except ValidationError as e:
        response = format_error_response(e, request_id=1)
        
        # Verify JSON-RPC 2.0 format
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert response["error"]["code"] == ERROR_CODE_INVALID_PARAMS
        assert "url" in response["error"]["message"].lower()


def test_search_query_validation_error_formatted_correctly():
    """Search query validation errors should be formatted correctly."""
    try:
        validate_search_query("")
        pytest.fail("Expected ValidationError")
    except ValidationError as e:
        response = format_error_response(e, request_id="search-1")
        
        # Verify JSON-RPC 2.0 format
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == "search-1"
        assert response["error"]["code"] == ERROR_CODE_INVALID_PARAMS
        assert "empty" in response["error"]["message"].lower()
