"""
Unit tests for configuration parser with loading section and aliases support.
"""
import pytest

from mcp_router.core.config import LoadingConfig, RouterConfig
from mcp_router.core.config_parser import parse_config


class TestParseLoadingSection:
    """Tests for parsing the loading section of configuration."""
    
    def test_parse_config_without_loading_section(self) -> None:
        """Test parsing config without loading section uses defaults."""
        config_dict = {
            'mcpServers': {
                'test': {
                    'transport': 'stdio',
                    'command': 'test'
                }
            }
        }
        
        config = parse_config(config_dict)
        
        # Should use default LoadingConfig values
        assert config.loading.auto_load == ["all"]
        assert config.loading.lazy_load is True
        assert config.loading.cache_embeddings is True
        assert config.loading.connection_timeout == 30
        assert config.loading.max_concurrent_upstreams == 10
        assert config.loading.rate_limit == 5
    
    def test_parse_config_with_empty_loading_section(self) -> None:
        """Test parsing config with empty loading section uses defaults."""
        config_dict = {
            'mcpServers': {
                'test': {
                    'transport': 'stdio',
                    'command': 'test'
                }
            },
            'loading': {}
        }
        
        config = parse_config(config_dict)
        
        # Should use default LoadingConfig values
        assert config.loading.auto_load == ["all"]
        assert config.loading.lazy_load is True
        assert config.loading.cache_embeddings is True
        assert config.loading.connection_timeout == 30
        assert config.loading.max_concurrent_upstreams == 10
        assert config.loading.rate_limit == 5
    
    def test_parse_config_with_partial_loading_section(self) -> None:
        """Test parsing config with partial loading section uses defaults for missing fields."""
        config_dict = {
            'mcpServers': {
                'test': {
                    'transport': 'stdio',
                    'command': 'test'
                }
            },
            'loading': {
                'auto_load': ['upstream1'],
                'connection_timeout': 60
            }
        }
        
        config = parse_config(config_dict)
        
        # Should use provided values
        assert config.loading.auto_load == ['upstream1']
        assert config.loading.connection_timeout == 60
        
        # Should use defaults for missing fields
        assert config.loading.lazy_load is True
        assert config.loading.cache_embeddings is True
        assert config.loading.max_concurrent_upstreams == 10
        assert config.loading.rate_limit == 5
    
    def test_parse_config_with_complete_loading_section(self) -> None:
        """Test parsing config with complete loading section."""
        config_dict = {
            'mcpServers': {
                'test': {
                    'transport': 'stdio',
                    'command': 'test'
                }
            },
            'loading': {
                'auto_load': ['upstream1', 'upstream2'],
                'lazy_load': False,
                'cache_embeddings': False,
                'connection_timeout': 60,
                'max_concurrent_upstreams': 5,
                'rate_limit': 10
            }
        }
        
        config = parse_config(config_dict)
        
        assert config.loading.auto_load == ['upstream1', 'upstream2']
        assert config.loading.lazy_load is False
        assert config.loading.cache_embeddings is False
        assert config.loading.connection_timeout == 60
        assert config.loading.max_concurrent_upstreams == 5
        assert config.loading.rate_limit == 10
    
    def test_parse_config_with_auto_load_all(self) -> None:
        """Test parsing config with auto_load set to ['all']."""
        config_dict = {
            'mcpServers': {
                'test': {
                    'transport': 'stdio',
                    'command': 'test'
                }
            },
            'loading': {
                'auto_load': ['all']
            }
        }
        
        config = parse_config(config_dict)
        assert config.loading.auto_load == ['all']
    
    def test_parse_config_with_auto_load_empty(self) -> None:
        """Test parsing config with auto_load set to empty list."""
        config_dict = {
            'mcpServers': {
                'test': {
                    'transport': 'stdio',
                    'command': 'test'
                }
            },
            'loading': {
                'auto_load': []
            }
        }
        
        config = parse_config(config_dict)
        assert config.loading.auto_load == []
    
    def test_parse_config_loading_not_dict_raises_error(self) -> None:
        """Test that loading section that is not a dict raises TypeError."""
        config_dict = {
            'mcpServers': {
                'test': {
                    'transport': 'stdio',
                    'command': 'test'
                }
            },
            'loading': 'not a dict'
        }
        
        with pytest.raises(TypeError, match="'loading' must be a dictionary"):
            parse_config(config_dict)
    
    def test_parse_config_loading_invalid_values_raises_error(self) -> None:
        """Test that invalid loading values raise ValueError."""
        config_dict = {
            'mcpServers': {
                'test': {
                    'transport': 'stdio',
                    'command': 'test'
                }
            },
            'loading': {
                'connection_timeout': -1
            }
        }
        
        with pytest.raises(ValueError, match="connection_timeout must be positive"):
            parse_config(config_dict)


class TestParseAliases:
    """Tests for parsing aliases in upstream configurations."""
    
    def test_parse_config_without_aliases(self) -> None:
        """Test parsing upstream config without aliases field."""
        config_dict = {
            'mcpServers': {
                'test': {
                    'transport': 'stdio',
                    'command': 'test'
                }
            }
        }
        
        config = parse_config(config_dict)
        assert config.mcp_servers['test'].aliases == []
    
    def test_parse_config_with_empty_aliases(self) -> None:
        """Test parsing upstream config with empty aliases list."""
        config_dict = {
            'mcpServers': {
                'test': {
                    'transport': 'stdio',
                    'command': 'test',
                    'aliases': []
                }
            }
        }
        
        config = parse_config(config_dict)
        assert config.mcp_servers['test'].aliases == []
    
    def test_parse_config_with_single_alias(self) -> None:
        """Test parsing upstream config with single alias."""
        config_dict = {
            'mcpServers': {
                'test': {
                    'transport': 'stdio',
                    'command': 'test',
                    'aliases': ['browser']
                }
            }
        }
        
        config = parse_config(config_dict)
        assert config.mcp_servers['test'].aliases == ['browser']
    
    def test_parse_config_with_multiple_aliases(self) -> None:
        """Test parsing upstream config with multiple aliases."""
        config_dict = {
            'mcpServers': {
                'test': {
                    'transport': 'stdio',
                    'command': 'test',
                    'aliases': ['browser', 'web', 'playwright']
                }
            }
        }
        
        config = parse_config(config_dict)
        assert config.mcp_servers['test'].aliases == ['browser', 'web', 'playwright']
    
    def test_parse_config_with_aliases_containing_spaces(self) -> None:
        """Test parsing upstream config with aliases containing spaces."""
        config_dict = {
            'mcpServers': {
                'test': {
                    'transport': 'stdio',
                    'command': 'test',
                    'aliases': ['web browser', 'browser tools']
                }
            }
        }
        
        config = parse_config(config_dict)
        assert config.mcp_servers['test'].aliases == ['web browser', 'browser tools']
    
    def test_parse_config_with_aliases_containing_hyphens(self) -> None:
        """Test parsing upstream config with aliases containing hyphens."""
        config_dict = {
            'mcpServers': {
                'test': {
                    'transport': 'stdio',
                    'command': 'test',
                    'aliases': ['web-browser', 'browser-tools']
                }
            }
        }
        
        config = parse_config(config_dict)
        assert config.mcp_servers['test'].aliases == ['web-browser', 'browser-tools']
    
    def test_parse_config_with_aliases_containing_underscores(self) -> None:
        """Test parsing upstream config with aliases containing underscores."""
        config_dict = {
            'mcpServers': {
                'test': {
                    'transport': 'stdio',
                    'command': 'test',
                    'aliases': ['web_browser', 'browser_tools']
                }
            }
        }
        
        config = parse_config(config_dict)
        assert config.mcp_servers['test'].aliases == ['web_browser', 'browser_tools']
    
    def test_parse_config_aliases_not_list_raises_error(self) -> None:
        """Test that aliases field that is not a list raises TypeError."""
        config_dict = {
            'mcpServers': {
                'test': {
                    'transport': 'stdio',
                    'command': 'test',
                    'aliases': 'not a list'
                }
            }
        }
        
        with pytest.raises(TypeError, match="'aliases'.*must be a list"):
            parse_config(config_dict)
    
    def test_parse_config_with_invalid_alias_raises_error(self) -> None:
        """Test that invalid alias characters raise ValueError."""
        config_dict = {
            'mcpServers': {
                'test': {
                    'transport': 'stdio',
                    'command': 'test',
                    'aliases': ['browser@tools']
                }
            }
        }
        
        with pytest.raises(ValueError, match="Invalid alias.*alphanumeric"):
            parse_config(config_dict)
    
    def test_parse_config_with_empty_alias_raises_error(self) -> None:
        """Test that empty string alias raises ValueError."""
        config_dict = {
            'mcpServers': {
                'test': {
                    'transport': 'stdio',
                    'command': 'test',
                    'aliases': ['']
                }
            }
        }
        
        with pytest.raises(ValueError, match="Alias cannot be empty"):
            parse_config(config_dict)


class TestBackwardsCompatibility:
    """Tests for backwards compatibility with old configuration format."""
    
    def test_old_config_without_loading_or_aliases(self) -> None:
        """Test that old config format without loading or aliases still works."""
        config_dict = {
            'mcpServers': {
                'playwright': {
                    'transport': 'stdio',
                    'command': 'npx',
                    'args': ['-y', '@modelcontextprotocol/server-playwright'],
                    'semantic_prefix': 'browser',
                    'category_description': 'Web browser automation'
                },
                'atlassian': {
                    'transport': 'stdio',
                    'command': 'npx',
                    'args': ['-y', '@modelcontextprotocol/server-atlassian']
                }
            }
        }
        
        config = parse_config(config_dict)
        
        # Should parse successfully
        assert len(config.mcp_servers) == 2
        assert 'playwright' in config.mcp_servers
        assert 'atlassian' in config.mcp_servers
        
        # Should use default loading config
        assert config.loading.auto_load == ["all"]
        assert config.loading.lazy_load is True
        
        # Should have empty aliases
        assert config.mcp_servers['playwright'].aliases == []
        assert config.mcp_servers['atlassian'].aliases == []
    
    def test_mixed_config_some_with_aliases_some_without(self) -> None:
        """Test config where some upstreams have aliases and some don't."""
        config_dict = {
            'mcpServers': {
                'playwright': {
                    'transport': 'stdio',
                    'command': 'npx',
                    'args': ['-y', '@modelcontextprotocol/server-playwright'],
                    'aliases': ['browser', 'web']
                },
                'atlassian': {
                    'transport': 'stdio',
                    'command': 'npx',
                    'args': ['-y', '@modelcontextprotocol/server-atlassian']
                }
            }
        }
        
        config = parse_config(config_dict)
        
        # Playwright should have aliases
        assert config.mcp_servers['playwright'].aliases == ['browser', 'web']
        
        # Atlassian should have empty aliases
        assert config.mcp_servers['atlassian'].aliases == []


class TestCompleteConfiguration:
    """Tests for complete configuration with all new features."""
    
    def test_parse_complete_config_with_loading_and_aliases(self) -> None:
        """Test parsing complete config with loading section and aliases."""
        config_dict = {
            'mcpServers': {
                'playwright': {
                    'transport': 'stdio',
                    'command': 'npx',
                    'args': ['-y', '@modelcontextprotocol/server-playwright'],
                    'semantic_prefix': 'browser',
                    'category_description': 'Web browser automation',
                    'aliases': ['browser', 'web', 'playwright']
                },
                'atlassian': {
                    'transport': 'stdio',
                    'command': 'npx',
                    'args': ['-y', '@modelcontextprotocol/server-atlassian'],
                    'aliases': ['jira', 'confluence', 'atlassian tools']
                }
            },
            'loading': {
                'auto_load': ['playwright'],
                'lazy_load': True,
                'cache_embeddings': True,
                'connection_timeout': 45,
                'max_concurrent_upstreams': 8,
                'rate_limit': 3
            }
        }
        
        config = parse_config(config_dict)
        
        # Verify upstreams
        assert len(config.mcp_servers) == 2
        assert config.mcp_servers['playwright'].aliases == ['browser', 'web', 'playwright']
        assert config.mcp_servers['atlassian'].aliases == ['jira', 'confluence', 'atlassian tools']
        
        # Verify loading config
        assert config.loading.auto_load == ['playwright']
        assert config.loading.lazy_load is True
        assert config.loading.cache_embeddings is True
        assert config.loading.connection_timeout == 45
        assert config.loading.max_concurrent_upstreams == 8
        assert config.loading.rate_limit == 3
