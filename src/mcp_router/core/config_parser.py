"""
Configuration file parser for MCP Semantic Router.

This module handles reading and parsing JSON configuration files.
"""
import json
from pathlib import Path
from typing import Any

from mcp_router.core.config import LoadingConfig, RouterConfig, UpstreamConfig


class ConfigurationError(Exception):
    """Raised when configuration is invalid or cannot be loaded."""
    pass


def load_config(config_path: str | Path) -> RouterConfig:
    """
    Load and parse configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Parsed RouterConfig object
        
    Raises:
        ConfigurationError: If config file is missing, malformed, or invalid
    """
    config_path = Path(config_path)
    
    # Check if file exists
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    
    # Read and parse JSON
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Invalid JSON in configuration file: {e}")
    except Exception as e:
        raise ConfigurationError(f"Failed to read configuration file: {e}")
    
    # Parse configuration
    try:
        return parse_config(config_dict)
    except (ValueError, KeyError, TypeError) as e:
        raise ConfigurationError(f"Invalid configuration structure: {e}")


def parse_config(config_dict: dict[str, Any]) -> RouterConfig:
    """
    Parse configuration dictionary into RouterConfig object.
    
    Args:
        config_dict: Configuration dictionary from JSON
        
    Returns:
        Parsed RouterConfig object
        
    Raises:
        ValueError: If configuration structure is invalid
        KeyError: If required fields are missing
        TypeError: If field types are incorrect
    """
    # Extract mcpServers section
    if 'mcpServers' not in config_dict:
        raise KeyError("Configuration must contain 'mcpServers' key")
    
    mcp_servers_dict = config_dict['mcpServers']
    
    if not isinstance(mcp_servers_dict, dict):
        raise TypeError("'mcpServers' must be a dictionary")
    
    if not mcp_servers_dict:
        raise ValueError("'mcpServers' must contain at least one server")
    
    # Parse loading configuration (optional, use defaults if missing)
    loading_config = LoadingConfig()
    if 'loading' in config_dict:
        loading_dict = config_dict['loading']
        if not isinstance(loading_dict, dict):
            raise TypeError("'loading' must be a dictionary")
        
        loading_config = LoadingConfig(
            auto_load=loading_dict.get('auto_load', ["all"]),
            lazy_load=loading_dict.get('lazy_load', True),
            cache_embeddings=loading_dict.get('cache_embeddings', True),
            connection_timeout=loading_dict.get('connection_timeout', 30),
            max_concurrent_upstreams=loading_dict.get('max_concurrent_upstreams', 10),
            rate_limit=loading_dict.get('rate_limit', 5)
        )
    
    # Parse each upstream configuration
    mcp_servers: dict[str, UpstreamConfig] = {}
    
    for upstream_id, upstream_dict in mcp_servers_dict.items():
        if not isinstance(upstream_dict, dict):
            raise TypeError(f"Configuration for '{upstream_id}' must be a dictionary")
        
        # Extract required and optional fields
        transport = upstream_dict.get('transport')
        if not transport:
            raise ValueError(f"'transport' is required for upstream '{upstream_id}'")
        
        if transport not in ('stdio', 'sse', 'http'):
            raise ValueError(
                f"Invalid transport '{transport}' for upstream '{upstream_id}'. "
                f"Must be 'stdio', 'sse', or 'http'"
            )
        
        # Parse aliases (optional)
        aliases = upstream_dict.get('aliases', [])
        if not isinstance(aliases, list):
            raise TypeError(f"'aliases' for upstream '{upstream_id}' must be a list")
        
        # Create UpstreamConfig (validation happens in __post_init__)
        upstream_config = UpstreamConfig(
            transport=transport,  # type: ignore
            command=upstream_dict.get('command'),
            args=upstream_dict.get('args'),
            url=upstream_dict.get('url'),
            semantic_prefix=upstream_dict.get('semantic_prefix'),
            category_description=upstream_dict.get('category_description'),
            aliases=aliases
        )
        
        mcp_servers[upstream_id] = upstream_config
    
    return RouterConfig(mcp_servers=mcp_servers, loading=loading_config)
