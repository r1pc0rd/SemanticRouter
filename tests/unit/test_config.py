"""
Unit tests for configuration management.

Tests specific examples and edge cases for configuration parsing.
"""
import json
import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile

from mcp_router.core.config import UpstreamConfig, RouterConfig
from mcp_router.core.config_parser import load_config, parse_config, ConfigurationError


class TestUpstreamConfig:
    """Tests for UpstreamConfig dataclass."""
    
    def test_stdio_config_valid(self) -> None:
        """Test valid stdio configuration."""
        config = UpstreamConfig(
            transport='stdio',
            command='npx',
            args=['-y', '@modelcontextprotocol/server-playwright']
        )
        assert config.transport == 'stdio'
        assert config.command == 'npx'
        assert config.args == ['-y', '@modelcontextprotocol/server-playwright']
    
    def test_stdio_config_missing_command(self) -> None:
        """Test stdio configuration without command raises error."""
        with pytest.raises(ValueError, match="command is required"):
            UpstreamConfig(transport='stdio')
    
    def test_stdio_config_empty_args(self) -> None:
        """Test stdio configuration with no args defaults to empty list."""
        config = UpstreamConfig(transport='stdio', command='test')
        assert config.args == []
    
    def test_sse_config_valid(self) -> None:
        """Test valid SSE configuration."""
        config = UpstreamConfig(
            transport='sse',
            url='https://example.com/mcp'
        )
        assert config.transport == 'sse'
        assert config.url == 'https://example.com/mcp'
    
    def test_sse_config_missing_url(self) -> None:
        """Test SSE configuration without URL raises error."""
        with pytest.raises(ValueError, match="url is required"):
            UpstreamConfig(transport='sse')
    
    def test_http_config_valid(self) -> None:
        """Test valid HTTP configuration."""
        config = UpstreamConfig(
            transport='http',
            url='http://localhost:8080'
        )
        assert config.transport == 'http'
        assert config.url == 'http://localhost:8080'
    
    def test_optional_fields(self) -> None:
        """Test optional semantic_prefix and category_description."""
        config = UpstreamConfig(
            transport='stdio',
            command='test',
            semantic_prefix='browser',
            category_description='Web testing tools'
        )
        assert config.semantic_prefix == 'browser'
        assert config.category_description == 'Web testing tools'
    
    def test_aliases_default_empty_list(self) -> None:
        """Test that aliases defaults to empty list."""
        config = UpstreamConfig(transport='stdio', command='test')
        assert config.aliases == []
    
    def test_aliases_valid_alphanumeric(self) -> None:
        """Test valid aliases with alphanumeric characters."""
        config = UpstreamConfig(
            transport='stdio',
            command='test',
            aliases=['browser', 'web', 'playwright']
        )
        assert config.aliases == ['browser', 'web', 'playwright']
    
    def test_aliases_valid_with_spaces(self) -> None:
        """Test valid aliases with spaces."""
        config = UpstreamConfig(
            transport='stdio',
            command='test',
            aliases=['web browser', 'browser tools']
        )
        assert config.aliases == ['web browser', 'browser tools']
    
    def test_aliases_valid_with_hyphens(self) -> None:
        """Test valid aliases with hyphens."""
        config = UpstreamConfig(
            transport='stdio',
            command='test',
            aliases=['web-browser', 'browser-tools']
        )
        assert config.aliases == ['web-browser', 'browser-tools']
    
    def test_aliases_valid_with_underscores(self) -> None:
        """Test valid aliases with underscores."""
        config = UpstreamConfig(
            transport='stdio',
            command='test',
            aliases=['web_browser', 'browser_tools']
        )
        assert config.aliases == ['web_browser', 'browser_tools']
    
    def test_aliases_valid_mixed_characters(self) -> None:
        """Test valid aliases with mixed alphanumeric, spaces, hyphens, and underscores."""
        config = UpstreamConfig(
            transport='stdio',
            command='test',
            aliases=['Web Browser 2024', 'browser-tools_v1']
        )
        assert config.aliases == ['Web Browser 2024', 'browser-tools_v1']
    
    def test_aliases_empty_string_raises_error(self) -> None:
        """Test that empty string alias raises ValueError."""
        with pytest.raises(ValueError, match="Alias cannot be empty"):
            UpstreamConfig(
                transport='stdio',
                command='test',
                aliases=['']
            )
    
    def test_aliases_special_characters_raises_error(self) -> None:
        """Test that aliases with special characters raise ValueError."""
        with pytest.raises(ValueError, match="Invalid alias.*alphanumeric"):
            UpstreamConfig(
                transport='stdio',
                command='test',
                aliases=['browser@tools']
            )
    
    def test_aliases_with_dots_raises_error(self) -> None:
        """Test that aliases with dots raise ValueError."""
        with pytest.raises(ValueError, match="Invalid alias.*alphanumeric"):
            UpstreamConfig(
                transport='stdio',
                command='test',
                aliases=['browser.tools']
            )
    
    def test_aliases_with_slashes_raises_error(self) -> None:
        """Test that aliases with slashes raise ValueError."""
        with pytest.raises(ValueError, match="Invalid alias.*alphanumeric"):
            UpstreamConfig(
                transport='stdio',
                command='test',
                aliases=['browser/tools']
            )


class TestRouterConfig:
    """Tests for RouterConfig dataclass."""
    
    def test_valid_config(self) -> None:
        """Test valid router configuration."""
        upstream = UpstreamConfig(transport='stdio', command='test')
        config = RouterConfig(mcp_servers={'test': upstream})
        assert len(config.mcp_servers) == 1
        assert 'test' in config.mcp_servers
    
    def test_empty_servers_raises_error(self) -> None:
        """Test router configuration with no servers raises error."""
        with pytest.raises(ValueError, match="At least one upstream"):
            RouterConfig(mcp_servers={})


class TestParseConfig:
    """Tests for parse_config function."""
    
    def test_parse_valid_config(self, sample_config_dict: dict) -> None:
        """Test parsing valid configuration dictionary."""
        config = parse_config(sample_config_dict)
        
        assert len(config.mcp_servers) == 2
        assert 'playwright' in config.mcp_servers
        assert 'atlassian' in config.mcp_servers
        
        playwright = config.mcp_servers['playwright']
        assert playwright.transport == 'stdio'
        assert playwright.command == 'npx'
        assert playwright.semantic_prefix == 'browser'
        assert playwright.category_description == 'Web browser automation and testing tools'
    
    def test_parse_missing_mcpservers_key(self) -> None:
        """Test parsing config without mcpServers key raises error."""
        with pytest.raises(KeyError, match="mcpServers"):
            parse_config({})
    
    def test_parse_mcpservers_not_dict(self) -> None:
        """Test parsing config where mcpServers is not a dict raises error."""
        with pytest.raises(TypeError, match="must be a dictionary"):
            parse_config({'mcpServers': 'not a dict'})
    
    def test_parse_empty_mcpservers(self) -> None:
        """Test parsing config with empty mcpServers raises error."""
        with pytest.raises(ValueError, match="at least one server"):
            parse_config({'mcpServers': {}})
    
    def test_parse_missing_transport(self) -> None:
        """Test parsing upstream without transport raises error."""
        config_dict = {
            'mcpServers': {
                'test': {
                    'command': 'test'
                }
            }
        }
        with pytest.raises(ValueError, match="transport.*required"):
            parse_config(config_dict)
    
    def test_parse_invalid_transport(self) -> None:
        """Test parsing upstream with invalid transport raises error."""
        config_dict = {
            'mcpServers': {
                'test': {
                    'transport': 'invalid',
                    'command': 'test'
                }
            }
        }
        with pytest.raises(ValueError, match="Invalid transport"):
            parse_config(config_dict)
    
    def test_parse_stdio_missing_command(self) -> None:
        """Test parsing stdio upstream without command raises error."""
        config_dict = {
            'mcpServers': {
                'test': {
                    'transport': 'stdio'
                }
            }
        }
        with pytest.raises(ValueError, match="command is required"):
            parse_config(config_dict)
    
    def test_parse_sse_missing_url(self) -> None:
        """Test parsing SSE upstream without URL raises error."""
        config_dict = {
            'mcpServers': {
                'test': {
                    'transport': 'sse'
                }
            }
        }
        with pytest.raises(ValueError, match="url is required"):
            parse_config(config_dict)
    
    def test_parse_upstream_config_not_dict(self) -> None:
        """Test parsing upstream config that is not a dict raises error."""
        config_dict = {
            'mcpServers': {
                'test': 'not a dict'
            }
        }
        with pytest.raises(TypeError, match="must be a dictionary"):
            parse_config(config_dict)


class TestLoadConfig:
    """Tests for load_config function."""
    
    def test_load_valid_config_file(self, sample_config_dict: dict) -> None:
        """Test loading valid configuration from file."""
        with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_config_dict, f)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            assert len(config.mcp_servers) == 2
            assert 'playwright' in config.mcp_servers
        finally:
            Path(temp_path).unlink()
    
    def test_load_missing_file(self) -> None:
        """Test loading non-existent file raises error."""
        with pytest.raises(ConfigurationError, match="not found"):
            load_config('/nonexistent/path/config.json')
    
    def test_load_malformed_json(self) -> None:
        """Test loading file with invalid JSON raises error."""
        with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{ invalid json }')
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigurationError, match="Invalid JSON"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_load_invalid_structure(self) -> None:
        """Test loading file with invalid structure raises error."""
        with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({'invalid': 'structure'}, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigurationError, match="Invalid configuration structure"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_load_config_with_path_object(self, sample_config_dict: dict) -> None:
        """Test loading configuration using Path object."""
        with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_config_dict, f)
            temp_path = Path(f.name)
        
        try:
            config = load_config(temp_path)
            assert len(config.mcp_servers) == 2
        finally:
            temp_path.unlink()
