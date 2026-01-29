"""
Error handling and JSON-RPC error formatting for MCP Semantic Router.

This module defines custom exception classes and utilities for formatting
errors as JSON-RPC 2.0 error responses.
"""
from typing import Any, Optional


# JSON-RPC 2.0 Error Codes
# Standard codes from JSON-RPC 2.0 specification
ERROR_CODE_PARSE_ERROR = -32700  # Invalid JSON
ERROR_CODE_INVALID_REQUEST = -32600  # Not JSON-RPC 2.0
ERROR_CODE_METHOD_NOT_FOUND = -32601  # Method/tool not found
ERROR_CODE_INVALID_PARAMS = -32602  # Invalid parameters
ERROR_CODE_INTERNAL_ERROR = -32603  # Internal errors (embedding, etc.)
ERROR_CODE_SERVER_ERROR = -32000  # Server errors (upstream timeout, connection lost)


class RouterError(Exception):
    """Base exception for all router errors.
    
    Attributes:
        message: Error message
        code: JSON-RPC error code
        data: Optional additional error data
    """
    
    def __init__(self, message: str, code: int, data: Optional[dict[str, Any]] = None):
        """Initialize router error.
        
        Args:
            message: Error message
            code: JSON-RPC error code
            data: Optional additional error data
        """
        super().__init__(message)
        self.message = message
        self.code = code
        self.data = data


class ConfigurationError(RouterError):
    """Configuration error (startup phase).
    
    These errors occur during configuration parsing and validation.
    They cause the router to fail to start.
    """
    
    def __init__(self, message: str, data: Optional[dict[str, Any]] = None):
        """Initialize configuration error.
        
        Args:
            message: Error message
            data: Optional additional error data
        """
        super().__init__(message, ERROR_CODE_INVALID_REQUEST, data)


class ValidationError(RouterError):
    """Parameter validation error (runtime).
    
    These errors occur when tool call parameters or search queries
    fail validation.
    """
    
    def __init__(self, message: str, data: Optional[dict[str, Any]] = None):
        """Initialize validation error.
        
        Args:
            message: Error message
            data: Optional additional error data
        """
        super().__init__(message, ERROR_CODE_INVALID_PARAMS, data)


class ToolNotFoundError(RouterError):
    """Tool not found error (runtime).
    
    These errors occur when a tool call references an unknown tool
    or invalid namespace.
    """
    
    def __init__(self, message: str, data: Optional[dict[str, Any]] = None):
        """Initialize tool not found error.
        
        Args:
            message: Error message
            data: Optional additional error data
        """
        super().__init__(message, ERROR_CODE_METHOD_NOT_FOUND, data)


class EmbeddingError(RouterError):
    """Embedding generation error (runtime).
    
    These errors occur when embedding generation fails during
    semantic search.
    """
    
    def __init__(self, message: str, data: Optional[dict[str, Any]] = None):
        """Initialize embedding error.
        
        Args:
            message: Error message
            data: Optional additional error data
        """
        super().__init__(message, ERROR_CODE_INTERNAL_ERROR, data)


class UpstreamError(RouterError):
    """Upstream server error (runtime).
    
    These errors occur when upstream connections fail or timeout.
    """
    
    def __init__(self, message: str, data: Optional[dict[str, Any]] = None):
        """Initialize upstream error.
        
        Args:
            message: Error message
            data: Optional additional error data
        """
        super().__init__(message, ERROR_CODE_SERVER_ERROR, data)


def format_error_response(
    error: Exception,
    request_id: Optional[Any] = None
) -> dict[str, Any]:
    """
    Format an exception as a JSON-RPC 2.0 error response.
    
    Args:
        error: Exception to format
        request_id: Optional request ID from JSON-RPC request
        
    Returns:
        JSON-RPC 2.0 error response dictionary
    """
    # Extract error details
    if isinstance(error, RouterError):
        code = error.code
        message = error.message
        data = error.data
    else:
        # Unknown error - treat as internal error
        code = ERROR_CODE_INTERNAL_ERROR
        message = str(error)
        data = {"type": type(error).__name__}
    
    # Build error response
    error_response: dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": code,
            "message": message
        }
    }
    
    # Add data if present
    if data is not None:
        error_response["error"]["data"] = data
    
    return error_response
