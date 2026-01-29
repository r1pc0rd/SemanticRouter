"""
Configuration schemas for MCP Semantic Router.

This module defines the configuration dataclasses for upstream MCP servers
and the router itself.
"""
from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class LoadingConfig:
    """Configuration for dynamic upstream loading behavior.
    
    Attributes:
        auto_load: List of upstream names to load on startup. 
                   Use ["all"] to load all upstreams, [] to load none.
        lazy_load: If True, automatically load upstreams when tools are called
        cache_embeddings: If True, cache embeddings to disk for faster loading
        connection_timeout: Timeout in seconds for upstream connections
        max_concurrent_upstreams: Maximum number of upstreams that can be loaded simultaneously
        rate_limit: Maximum number of load/unload operations per second
    """
    auto_load: List[str] = field(default_factory=lambda: ["all"])
    lazy_load: bool = True
    cache_embeddings: bool = True
    connection_timeout: int = 30
    max_concurrent_upstreams: int = 10
    rate_limit: int = 5
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.connection_timeout <= 0:
            raise ValueError("connection_timeout must be positive")
        if self.max_concurrent_upstreams <= 0:
            raise ValueError("max_concurrent_upstreams must be positive")
        if self.rate_limit <= 0:
            raise ValueError("rate_limit must be positive")
        if not isinstance(self.auto_load, list):
            raise ValueError("auto_load must be a list")


@dataclass
class UpstreamConfig:
    """Configuration for an upstream MCP server.
    
    Attributes:
        transport: Communication protocol (stdio, sse, or http)
        command: Command to start the upstream server (for stdio)
        args: Arguments for the command (for stdio)
        url: URL for the upstream server (for sse/http)
        semantic_prefix: Optional prefix for tool namespacing (defaults to upstream ID)
        category_description: Optional description for embedding context
        aliases: List of alternative names for this upstream (for natural language loading)
    """
    transport: Literal['stdio', 'sse', 'http']
    command: Optional[str] = None
    args: Optional[List[str]] = None
    url: Optional[str] = None
    semantic_prefix: Optional[str] = None
    category_description: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.transport == 'stdio':
            if not self.command:
                raise ValueError("command is required for stdio transport")
            if self.args is None:
                self.args = []
        elif self.transport in ('sse', 'http'):
            if not self.url:
                raise ValueError(f"url is required for {self.transport} transport")
        
        # Validate aliases are alphanumeric + spaces only
        for alias in self.aliases:
            if not alias:
                raise ValueError("Alias cannot be empty")
            if not all(c.isalnum() or c.isspace() or c == '-' or c == '_' for c in alias):
                raise ValueError(
                    f"Invalid alias '{alias}': aliases must contain only "
                    f"alphanumeric characters, spaces, hyphens, and underscores"
                )


@dataclass
class RouterConfig:
    """Router configuration with upstream MCP servers.
    
    Attributes:
        mcp_servers: Dictionary mapping upstream IDs to their configurations
        loading: Configuration for dynamic upstream loading behavior
    """
    mcp_servers: dict[str, UpstreamConfig] = field(default_factory=dict)
    loading: LoadingConfig = field(default_factory=LoadingConfig)
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.mcp_servers:
            raise ValueError("At least one upstream MCP server must be configured")
