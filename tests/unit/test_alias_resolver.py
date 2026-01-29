"""Unit tests for AliasResolver."""

import pytest
from src.mcp_router.discovery.alias_resolver import AliasResolver
from src.mcp_router.core.config import RouterConfig, UpstreamConfig


class TestAliasResolverInitialization:
    """Test AliasResolver initialization and alias map building."""
    
    def test_empty_config(self):
        """Test resolver with no upstreams - should fail during config creation."""
        # RouterConfig requires at least one upstream
        with pytest.raises(ValueError, match="At least one upstream MCP server must be configured"):
            config = RouterConfig(mcp_servers={})
    
    def test_single_upstream_no_aliases(self):
        """Test resolver with one upstream and no aliases."""
        config = RouterConfig(mcp_servers={
            "test-server": UpstreamConfig(
                transport="stdio",
                command="test",
                args=[],
                aliases=[]
            )
        })
        resolver = AliasResolver(config)
        
        # Should resolve exact upstream name
        assert resolver.resolve("test-server") == "test-server"
        
        # Should fail for unknown names
        with pytest.raises(ValueError):
            resolver.resolve("unknown")
    
    def test_single_upstream_with_aliases(self):
        """Test resolver with one upstream and multiple aliases."""
        config = RouterConfig(mcp_servers={
            "playwright": UpstreamConfig(
                transport="stdio",
                command="test",
                args=[],
                aliases=["browser", "web", "playwright-tools"]
            )
        })
        resolver = AliasResolver(config)
        
        # Should resolve all aliases to upstream name
        assert resolver.resolve("browser") == "playwright"
        assert resolver.resolve("web") == "playwright"
        assert resolver.resolve("playwright-tools") == "playwright"
        assert resolver.resolve("playwright") == "playwright"
    
    def test_multiple_upstreams_with_aliases(self):
        """Test resolver with multiple upstreams."""
        config = RouterConfig(mcp_servers={
            "playwright": UpstreamConfig(
                transport="stdio",
                command="test1",
                args=[],
                aliases=["browser", "web"]
            ),
            "jira": UpstreamConfig(
                transport="stdio",
                command="test2",
                args=[],
                aliases=["tickets", "issues"]
            )
        })
        resolver = AliasResolver(config)
        
        # Should resolve each alias to correct upstream
        assert resolver.resolve("browser") == "playwright"
        assert resolver.resolve("web") == "playwright"
        assert resolver.resolve("tickets") == "jira"
        assert resolver.resolve("issues") == "jira"
        assert resolver.resolve("playwright") == "playwright"
        assert resolver.resolve("jira") == "jira"


class TestAliasResolverCaseInsensitivity:
    """Test case-insensitive alias resolution."""
    
    def test_alias_case_insensitive(self):
        """Test that aliases are resolved case-insensitively."""
        config = RouterConfig(mcp_servers={
            "playwright": UpstreamConfig(
                transport="stdio",
                command="test",
                args=[],
                aliases=["Browser", "WEB"]
            )
        })
        resolver = AliasResolver(config)
        
        # All case variations should resolve
        assert resolver.resolve("browser") == "playwright"
        assert resolver.resolve("Browser") == "playwright"
        assert resolver.resolve("BROWSER") == "playwright"
        assert resolver.resolve("web") == "playwright"
        assert resolver.resolve("WEB") == "playwright"
        assert resolver.resolve("Web") == "playwright"
    
    def test_upstream_name_case_sensitive(self):
        """Test that upstream names are case-sensitive."""
        config = RouterConfig(mcp_servers={
            "Playwright": UpstreamConfig(
                transport="stdio",
                command="test",
                args=[],
                aliases=[]
            )
        })
        resolver = AliasResolver(config)
        
        # Exact match should work
        assert resolver.resolve("Playwright") == "Playwright"
        
        # Different case should fail
        with pytest.raises(ValueError):
            resolver.resolve("playwright")
        with pytest.raises(ValueError):
            resolver.resolve("PLAYWRIGHT")


class TestAliasResolverErrorHandling:
    """Test error handling and error messages."""
    
    def test_unknown_name_error_message(self):
        """Test that error message includes available options."""
        config = RouterConfig(mcp_servers={
            "playwright": UpstreamConfig(
                transport="stdio",
                command="test",
                args=[],
                aliases=["browser", "web"]
            )
        })
        resolver = AliasResolver(config)
        
        with pytest.raises(ValueError) as exc_info:
            resolver.resolve("unknown")
        
        error_msg = str(exc_info.value)
        assert "unknown" in error_msg.lower()
        assert "browser" in error_msg
        assert "web" in error_msg
        assert "playwright" in error_msg
    
    def test_error_message_no_aliases(self):
        """Test error message when no aliases are defined."""
        config = RouterConfig(mcp_servers={
            "test-server": UpstreamConfig(
                transport="stdio",
                command="test",
                args=[],
                aliases=[]
            )
        })
        resolver = AliasResolver(config)
        
        with pytest.raises(ValueError) as exc_info:
            resolver.resolve("unknown")
        
        error_msg = str(exc_info.value)
        assert "test-server" in error_msg
        assert "unknown" in error_msg.lower()


class TestResolveMultiple:
    """Test resolve_multiple() method."""
    
    def test_resolve_multiple_all_valid(self):
        """Test resolving multiple valid names."""
        config = RouterConfig(mcp_servers={
            "playwright": UpstreamConfig(
                transport="stdio",
                command="test1",
                args=[],
                aliases=["browser"]
            ),
            "jira": UpstreamConfig(
                transport="stdio",
                command="test2",
                args=[],
                aliases=["tickets"]
            )
        })
        resolver = AliasResolver(config)
        
        result = resolver.resolve_multiple(["browser", "jira", "tickets", "playwright"])
        assert result == ["playwright", "jira", "jira", "playwright"]
    
    def test_resolve_multiple_empty_list(self):
        """Test resolving empty list."""
        config = RouterConfig(mcp_servers={
            "test": UpstreamConfig(
                transport="stdio",
                command="test",
                args=[],
                aliases=[]
            )
        })
        resolver = AliasResolver(config)
        
        result = resolver.resolve_multiple([])
        assert result == []
    
    def test_resolve_multiple_with_errors(self):
        """Test that resolve_multiple fails if any name is invalid."""
        config = RouterConfig(mcp_servers={
            "playwright": UpstreamConfig(
                transport="stdio",
                command="test",
                args=[],
                aliases=["browser"]
            )
        })
        resolver = AliasResolver(config)
        
        with pytest.raises(ValueError) as exc_info:
            resolver.resolve_multiple(["browser", "unknown1", "playwright", "unknown2"])
        
        error_msg = str(exc_info.value)
        assert "unknown1" in error_msg
        assert "unknown2" in error_msg
        assert "Failed to resolve" in error_msg
    
    def test_resolve_multiple_preserves_order(self):
        """Test that resolve_multiple preserves input order."""
        config = RouterConfig(mcp_servers={
            "a": UpstreamConfig(transport="stdio", command="test", args=[], aliases=["alias-a"]),
            "b": UpstreamConfig(transport="stdio", command="test", args=[], aliases=["alias-b"]),
            "c": UpstreamConfig(transport="stdio", command="test", args=[], aliases=["alias-c"])
        })
        resolver = AliasResolver(config)
        
        result = resolver.resolve_multiple(["c", "alias-a", "b", "alias-c", "a"])
        assert result == ["c", "a", "b", "c", "a"]


class TestAliasResolverEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_duplicate_aliases_across_upstreams(self):
        """Test that duplicate aliases are handled (last one wins)."""
        config = RouterConfig(mcp_servers={
            "upstream1": UpstreamConfig(
                transport="stdio",
                command="test1",
                args=[],
                aliases=["shared"]
            ),
            "upstream2": UpstreamConfig(
                transport="stdio",
                command="test2",
                args=[],
                aliases=["shared"]
            )
        })
        resolver = AliasResolver(config)
        
        # Should resolve to one of them (implementation-dependent)
        result = resolver.resolve("shared")
        assert result in ["upstream1", "upstream2"]
    
    def test_alias_same_as_upstream_name(self):
        """Test alias that matches another upstream's name."""
        config = RouterConfig(mcp_servers={
            "upstream1": UpstreamConfig(
                transport="stdio",
                command="test1",
                args=[],
                aliases=["upstream2"]  # Alias matches another upstream name
            ),
            "upstream2": UpstreamConfig(
                transport="stdio",
                command="test2",
                args=[],
                aliases=[]
            )
        })
        resolver = AliasResolver(config)
        
        # Upstream name should take precedence over alias
        assert resolver.resolve("upstream2") == "upstream2"
    
    def test_whitespace_in_aliases(self):
        """Test aliases with spaces and special characters."""
        config = RouterConfig(mcp_servers={
            "playwright": UpstreamConfig(
                transport="stdio",
                command="test",
                args=[],
                aliases=["browser tools", "web-automation", "playwright_mcp"]
            )
        })
        resolver = AliasResolver(config)
        
        assert resolver.resolve("browser tools") == "playwright"
        assert resolver.resolve("web-automation") == "playwright"
        assert resolver.resolve("playwright_mcp") == "playwright"
    
    def test_resolve_with_leading_trailing_spaces(self):
        """Test that leading/trailing spaces don't affect resolution."""
        config = RouterConfig(mcp_servers={
            "playwright": UpstreamConfig(
                transport="stdio",
                command="test",
                args=[],
                aliases=["browser"]
            )
        })
        resolver = AliasResolver(config)
        
        # Note: Current implementation doesn't strip spaces
        # This test documents current behavior
        with pytest.raises(ValueError):
            resolver.resolve(" browser ")
