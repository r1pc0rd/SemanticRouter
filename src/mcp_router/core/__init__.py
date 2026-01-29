"""Core data models and schemas for MCP Semantic Router."""

from .config import RouterConfig, UpstreamConfig
from .config_parser import load_config, parse_config, ConfigurationError
from .models import (
    JSONSchema,
    ToolMetadata,
    ContentItem,
    ToolCallResult,
    SearchResult,
)
from .namespace import (
    generate_tool_namespace,
    parse_tool_namespace,
    match_upstream_by_prefix,
)
from .logging import (
    LogEntry,
    JSONFormatter,
    setup_logging,
    log_with_metadata,
)

__all__ = [
    'RouterConfig',
    'UpstreamConfig',
    'load_config',
    'parse_config',
    'ConfigurationError',
    'JSONSchema',
    'ToolMetadata',
    'ContentItem',
    'ToolCallResult',
    'SearchResult',
    'generate_tool_namespace',
    'parse_tool_namespace',
    'match_upstream_by_prefix',
    'LogEntry',
    'JSONFormatter',
    'setup_logging',
    'log_with_metadata',
]
