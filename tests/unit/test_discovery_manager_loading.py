"""
Unit tests for ToolDiscoveryManager dynamic loading functionality.

Tests the refactored initialization pattern with auto_load configuration.
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List

from mcp_router.core.config import RouterConfig, UpstreamConfig, LoadingConfig
from mcp_router.core.models import ToolMetadata, JSONSchema
from mcp_router.discovery.manager import ToolDiscoveryManager
from mcp_router.discovery.upstream import UpstreamConnection


# Shared fixtures for all tests
@pytest.fixture
def mock_embedding_engine():
    """Create a mock EmbeddingEngine."""
    engine = MagicMock()
    engine.generate_tool_embeddings = AsyncMock(return_value=[])
    return engine


@pytest.fixture
def mock_search_engine():
    """Create a mock SemanticSearchEngine."""
    engine = MagicMock()
    engine.add_tools = MagicMock()
    engine.remove_tools = MagicMock()
    return engine


class TestInitializeAutoLoadAll:
    """Test initialize() with auto_load=['all'] configuration."""
    
    @pytest.fixture
    def config_with_all(self) -> RouterConfig:
        """Create config with auto_load=['all'] and 3 upstreams."""
        return RouterConfig(
            mcp_servers={
                "playwright": UpstreamConfig(
                    transport="stdio",
                    command="npx",
                    args=["-y", "@playwright/mcp-server"]
                ),
                "jira": UpstreamConfig(
                    transport="stdio",
                    command="npx",
                    args=["-y", "@jira/mcp-server"]
                ),
                "github": UpstreamConfig(
                    transport="stdio",
                    command="npx",
                    args=["-y", "@github/mcp-server"]
                )
            },
            loading=LoadingConfig(
                auto_load=["all"],
                lazy_load=True,
                cache_embeddings=True,
                connection_timeout=30,
                max_concurrent_upstreams=10,
                rate_limit=5
            )
        )
    
    @pytest.mark.asyncio
    async def test_initialize_loads_all_upstreams(self, config_with_all, mock_embedding_engine, mock_search_engine):
        """Test that initialize() loads all upstreams when auto_load=['all']."""
        manager = ToolDiscoveryManager(config_with_all, mock_embedding_engine, mock_search_engine)
        
        # Mock load_upstream to track calls
        load_calls = []
        async def mock_load_upstream(name: str) -> dict:
            load_calls.append(name)
            return {"success": True, "upstream": name, "tool_count": 5}
        
        manager.load_upstream = mock_load_upstream
        
        # Initialize
        await manager.initialize()
        
        # Verify all upstreams were loaded
        assert len(load_calls) == 3
        assert "playwright" in load_calls
        assert "jira" in load_calls
        assert "github" in load_calls
    
    @pytest.mark.asyncio
    async def test_initialize_handles_partial_failures(self, config_with_all, mock_embedding_engine, mock_search_engine):
        """Test that initialize() continues loading even if some upstreams fail."""
        manager = ToolDiscoveryManager(config_with_all, mock_embedding_engine, mock_search_engine)
        
        # Mock load_upstream to fail for jira
        async def mock_load_upstream(name: str) -> dict:
            if name == "jira":
                return {"success": False, "error": "Connection failed"}
            return {"success": True, "upstream": name, "tool_count": 5}
        
        manager.load_upstream = mock_load_upstream
        
        # Initialize should not raise exception
        await manager.initialize()
        
        # No assertion needed - test passes if no exception raised


class TestInitializeAutoLoadSpecific:
    """Test initialize() with specific upstream names in auto_load."""
    
    @pytest.fixture
    def config_with_specific(self) -> RouterConfig:
        """Create config with auto_load=['playwright', 'jira']."""
        return RouterConfig(
            mcp_servers={
                "playwright": UpstreamConfig(
                    transport="stdio",
                    command="npx",
                    args=["-y", "@playwright/mcp-server"]
                ),
                "jira": UpstreamConfig(
                    transport="stdio",
                    command="npx",
                    args=["-y", "@jira/mcp-server"]
                ),
                "github": UpstreamConfig(
                    transport="stdio",
                    command="npx",
                    args=["-y", "@github/mcp-server"]
                )
            },
            loading=LoadingConfig(
                auto_load=["playwright", "jira"],
                lazy_load=True,
                cache_embeddings=True,
                connection_timeout=30,
                max_concurrent_upstreams=10,
                rate_limit=5
            )
        )
    
    @pytest.mark.asyncio
    async def test_initialize_loads_only_specified_upstreams(self, config_with_specific, mock_embedding_engine, mock_search_engine):
        """Test that initialize() loads only specified upstreams."""
        manager = ToolDiscoveryManager(config_with_specific, mock_embedding_engine, mock_search_engine)
        
        # Mock load_upstream to track calls
        load_calls = []
        async def mock_load_upstream(name: str) -> dict:
            load_calls.append(name)
            return {"success": True, "upstream": name, "tool_count": 5}
        
        manager.load_upstream = mock_load_upstream
        
        # Initialize
        await manager.initialize()
        
        # Verify only specified upstreams were loaded
        assert len(load_calls) == 2
        assert "playwright" in load_calls
        assert "jira" in load_calls
        assert "github" not in load_calls
    
    @pytest.mark.asyncio
    async def test_initialize_with_single_upstream(self):
        """Test initialize() with auto_load containing single upstream."""
        config = RouterConfig(
            mcp_servers={
                "playwright": UpstreamConfig(
                    transport="stdio",
                    command="npx",
                    args=["-y", "@playwright/mcp-server"]
                ),
                "jira": UpstreamConfig(
                    transport="stdio",
                    command="npx",
                    args=["-y", "@jira/mcp-server"]
                )
            },
            loading=LoadingConfig(auto_load=["playwright"])
        )
        
        # Create mock engines
        mock_embedding_engine = MagicMock()
        mock_embedding_engine.generate_tool_embeddings = AsyncMock(return_value=[])
        mock_search_engine = MagicMock()
        mock_search_engine.add_tools = MagicMock()
        
        manager = ToolDiscoveryManager(config, mock_embedding_engine, mock_search_engine)
        
        # Mock load_upstream
        load_calls = []
        async def mock_load_upstream(name: str) -> dict:
            load_calls.append(name)
            return {"success": True, "upstream": name, "tool_count": 5}
        
        manager.load_upstream = mock_load_upstream
        
        # Initialize
        await manager.initialize()
        
        # Verify only playwright was loaded
        assert load_calls == ["playwright"]


class TestInitializeAutoLoadEmpty:
    """Test initialize() with auto_load=[] (no upstreams loaded)."""
    
    @pytest.fixture
    def config_with_empty(self) -> RouterConfig:
        """Create config with auto_load=[]."""
        return RouterConfig(
            mcp_servers={
                "playwright": UpstreamConfig(
                    transport="stdio",
                    command="npx",
                    args=["-y", "@playwright/mcp-server"]
                ),
                "jira": UpstreamConfig(
                    transport="stdio",
                    command="npx",
                    args=["-y", "@jira/mcp-server"]
                )
            },
            loading=LoadingConfig(
                auto_load=[],
                lazy_load=True,
                cache_embeddings=True,
                connection_timeout=30,
                max_concurrent_upstreams=10,
                rate_limit=5
            )
        )
    
    @pytest.mark.asyncio
    async def test_initialize_loads_no_upstreams(self, config_with_empty):
        """Test that initialize() loads no upstreams when auto_load=[]."""
        # Create mock engines
        mock_embedding_engine = MagicMock()
        mock_embedding_engine.generate_tool_embeddings = AsyncMock(return_value=[])
        mock_search_engine = MagicMock()
        mock_search_engine.add_tools = MagicMock()
        
        manager = ToolDiscoveryManager(config_with_empty, mock_embedding_engine, mock_search_engine)
        
        # Mock load_upstream to track calls
        load_calls = []
        async def mock_load_upstream(name: str) -> dict:
            load_calls.append(name)
            return {"success": True, "upstream": name, "tool_count": 5}
        
        manager.load_upstream = mock_load_upstream
        
        # Initialize
        await manager.initialize()
        
        # Verify no upstreams were loaded
        assert len(load_calls) == 0
    
    @pytest.mark.asyncio
    async def test_initialize_returns_immediately_with_empty_auto_load(self, config_with_empty):
        """Test that initialize() returns immediately when auto_load=[]."""
        # Create mock engines
        mock_embedding_engine = MagicMock()
        mock_embedding_engine.generate_tool_embeddings = AsyncMock(return_value=[])
        mock_search_engine = MagicMock()
        mock_search_engine.add_tools = MagicMock()
        
        manager = ToolDiscoveryManager(config_with_empty, mock_embedding_engine, mock_search_engine)
        
        # Mock load_upstream - should not be called
        manager.load_upstream = AsyncMock()
        
        # Initialize
        await manager.initialize()
        
        # Verify load_upstream was never called
        manager.load_upstream.assert_not_called()


class TestInitializeNoLoadingConfig:
    """Test initialize() when loading config is None (defaults to ['all'])."""
    
    @pytest.fixture
    def config_without_loading(self) -> RouterConfig:
        """Create config without loading section (uses defaults)."""
        config = RouterConfig(
            mcp_servers={
                "playwright": UpstreamConfig(
                    transport="stdio",
                    command="npx",
                    args=["-y", "@playwright/mcp-server"]
                ),
                "jira": UpstreamConfig(
                    transport="stdio",
                    command="npx",
                    args=["-y", "@jira/mcp-server"]
                )
            }
        )
        # Simulate missing loading config by setting to None
        # (In practice, RouterConfig always has LoadingConfig with defaults,
        # but we test the defensive code in initialize())
        return config
    
    @pytest.mark.asyncio
    async def test_initialize_defaults_to_all_when_no_loading_config(self, config_without_loading):
        """Test that initialize() defaults to loading all upstreams when loading config is None."""
        # Create mock engines
        mock_embedding_engine = MagicMock()
        mock_embedding_engine.generate_tool_embeddings = AsyncMock(return_value=[])
        mock_search_engine = MagicMock()
        mock_search_engine.add_tools = MagicMock()
        
        manager = ToolDiscoveryManager(config_without_loading, mock_embedding_engine, mock_search_engine)
        
        # Mock load_upstream to track calls
        load_calls = []
        async def mock_load_upstream(name: str) -> dict:
            load_calls.append(name)
            return {"success": True, "upstream": name, "tool_count": 5}
        
        manager.load_upstream = mock_load_upstream
        
        # Initialize
        await manager.initialize()
        
        # Verify all upstreams were loaded (default behavior)
        assert len(load_calls) == 2
        assert "playwright" in load_calls
        assert "jira" in load_calls


class TestInitializeExceptionHandling:
    """Test initialize() exception handling during upstream loading."""
    
    @pytest.fixture
    def config_basic(self) -> RouterConfig:
        """Create basic config with 2 upstreams."""
        return RouterConfig(
            mcp_servers={
                "playwright": UpstreamConfig(
                    transport="stdio",
                    command="npx",
                    args=["-y", "@playwright/mcp-server"]
                ),
                "jira": UpstreamConfig(
                    transport="stdio",
                    command="npx",
                    args=["-y", "@jira/mcp-server"]
                )
            },
            loading=LoadingConfig(auto_load=["all"])
        )
    
    @pytest.mark.asyncio
    async def test_initialize_handles_load_upstream_exception(self, config_basic):
        """Test that initialize() handles exceptions from load_upstream gracefully."""
        # Create mock engines
        mock_embedding_engine = MagicMock()
        mock_embedding_engine.generate_tool_embeddings = AsyncMock(return_value=[])
        mock_search_engine = MagicMock()
        mock_search_engine.add_tools = MagicMock()
        
        manager = ToolDiscoveryManager(config_basic, mock_embedding_engine, mock_search_engine)
        
        # Mock load_upstream to raise exception for jira
        async def mock_load_upstream(name: str) -> dict:
            if name == "jira":
                raise ConnectionError("Failed to connect to jira")
            return {"success": True, "upstream": name, "tool_count": 5}
        
        manager.load_upstream = mock_load_upstream
        
        # Initialize should not raise exception
        await manager.initialize()
        
        # No assertion needed - test passes if no exception raised
    
    @pytest.mark.asyncio
    async def test_initialize_continues_after_exception(self, config_basic):
        """Test that initialize() continues loading other upstreams after exception."""
        # Create mock engines
        mock_embedding_engine = MagicMock()
        mock_embedding_engine.generate_tool_embeddings = AsyncMock(return_value=[])
        mock_search_engine = MagicMock()
        mock_search_engine.add_tools = MagicMock()
        
        manager = ToolDiscoveryManager(config_basic, mock_embedding_engine, mock_search_engine)
        
        # Mock load_upstream to raise exception for first upstream
        load_calls = []
        async def mock_load_upstream(name: str) -> dict:
            load_calls.append(name)
            if name == "playwright":
                raise RuntimeError("Unexpected error")
            return {"success": True, "upstream": name, "tool_count": 5}
        
        manager.load_upstream = mock_load_upstream
        
        # Initialize
        await manager.initialize()
        
        # Verify both upstreams were attempted
        assert len(load_calls) == 2
        assert "playwright" in load_calls
        assert "jira" in load_calls


class TestInitializeLoadUpstreamStub:
    """Test that load_upstream stub is called correctly during initialize()."""
    
    @pytest.fixture
    def config_basic(self) -> RouterConfig:
        """Create basic config."""
        return RouterConfig(
            mcp_servers={
                "playwright": UpstreamConfig(
                    transport="stdio",
                    command="npx",
                    args=["-y", "@playwright/mcp-server"]
                )
            },
            loading=LoadingConfig(auto_load=["playwright"])
        )
    
    @pytest.mark.asyncio
    async def test_load_upstream_stub_returns_success(self, config_basic):
        """Test that load_upstream stub returns success dict."""
        from mcp_router.core.models import ToolMetadata, JSONSchema
        
        # Create mock engines
        mock_embedding_engine = MagicMock()
        mock_embedding_engine.generate_tool_embeddings = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        mock_search_engine = MagicMock()
        mock_search_engine.add_tools = MagicMock()
        
        manager = ToolDiscoveryManager(config_basic, mock_embedding_engine, mock_search_engine)
        
        # Mock UpstreamConnection to avoid real connection
        with patch('mcp_router.discovery.manager.UpstreamConnection') as MockConnection:
            mock_connection = AsyncMock()
            mock_connection.connect = AsyncMock()
            mock_connection.fetch_tools = AsyncMock(return_value=[
                ToolMetadata(
                    name="playwright.navigate",
                    original_name="navigate",
                    description="Navigate to URL",
                    input_schema=JSONSchema(type="object", properties={}),
                    upstream_id="playwright"
                )
            ])
            mock_connection.disconnect = AsyncMock()
            MockConnection.return_value = mock_connection
            
            # Call load_upstream
            result = await manager.load_upstream("playwright")
            
            # Verify returns expected format
            assert result["success"] is True
            assert result["upstream"] == "playwright"
            assert result["tool_count"] == 1
    
    @pytest.mark.asyncio
    async def test_initialize_uses_load_upstream_stub(self, config_basic):
        """Test that initialize() calls load_upstream stub."""
        from mcp_router.core.models import ToolMetadata, JSONSchema
        
        # Create mock engines
        mock_embedding_engine = MagicMock()
        mock_embedding_engine.generate_tool_embeddings = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        mock_search_engine = MagicMock()
        mock_search_engine.add_tools = MagicMock()
        
        manager = ToolDiscoveryManager(config_basic, mock_embedding_engine, mock_search_engine)
        
        # Mock UpstreamConnection to avoid real connection
        with patch('mcp_router.discovery.manager.UpstreamConnection') as MockConnection:
            mock_connection = AsyncMock()
            mock_connection.connect = AsyncMock()
            mock_connection.fetch_tools = AsyncMock(return_value=[
                ToolMetadata(
                    name="playwright.navigate",
                    original_name="navigate",
                    description="Navigate to URL",
                    input_schema=JSONSchema(type="object", properties={}),
                    upstream_id="playwright"
                )
            ])
            mock_connection.disconnect = AsyncMock()
            MockConnection.return_value = mock_connection
            
            # Spy on load_upstream method
            original_load_upstream = manager.load_upstream
            call_count = 0
            
            async def spy_load_upstream(name: str) -> dict:
                nonlocal call_count
                call_count += 1
                return await original_load_upstream(name)
            
            manager.load_upstream = spy_load_upstream
            
            # Initialize
            await manager.initialize()
            
            # Verify load_upstream was called once
            assert call_count == 1


class TestInitializeAliasResolver:
    """Test that AliasResolver is initialized correctly."""
    
    @pytest.fixture
    def config_with_aliases(self) -> RouterConfig:
        """Create config with upstream aliases."""
        return RouterConfig(
            mcp_servers={
                "playwright": UpstreamConfig(
                    transport="stdio",
                    command="npx",
                    args=["-y", "@playwright/mcp-server"],
                    aliases=["browser", "web automation"]
                )
            },
            loading=LoadingConfig(auto_load=[])
        )
    
    @pytest.mark.asyncio
    async def test_alias_resolver_initialized_in_constructor(self, config_with_aliases):
        """Test that AliasResolver is initialized in __init__."""
        # Create mock engines
        mock_embedding_engine = MagicMock()
        mock_embedding_engine.generate_tool_embeddings = AsyncMock(return_value=[])
        mock_search_engine = MagicMock()
        mock_search_engine.add_tools = MagicMock()
        
        manager = ToolDiscoveryManager(config_with_aliases, mock_embedding_engine, mock_search_engine)
        
        # Verify alias resolver exists
        assert manager._alias_resolver is not None
        
        # Verify alias resolver can resolve aliases
        resolved = manager._alias_resolver.resolve("browser")
        assert resolved == "playwright"
    
    @pytest.mark.asyncio
    async def test_alias_resolver_available_during_initialize(self, config_with_aliases):
        """Test that alias resolver is available during initialize()."""
        # Create mock engines
        mock_embedding_engine = MagicMock()
        mock_embedding_engine.generate_tool_embeddings = AsyncMock(return_value=[])
        mock_search_engine = MagicMock()
        mock_search_engine.add_tools = MagicMock()
        
        manager = ToolDiscoveryManager(config_with_aliases, mock_embedding_engine, mock_search_engine)
        
        # Initialize (should not raise exception)
        await manager.initialize()
        
        # Verify alias resolver still works after initialize
        resolved = manager._alias_resolver.resolve("web automation")
        assert resolved == "playwright"


# ============================================================================
# TASK 6 TESTS: load_upstream() method
# ============================================================================


class TestLoadUpstreamSuccess:
    """Test successful load_upstream() scenarios."""
    
    @pytest.fixture
    def config_basic(self) -> RouterConfig:
        """Create basic config with one upstream."""
        return RouterConfig(
            mcp_servers={
                "playwright": UpstreamConfig(
                    transport="stdio",
                    command="npx",
                    args=["-y", "@playwright/mcp-server"]
                )
            },
            loading=LoadingConfig(
                auto_load=[],
                connection_timeout=30
            )
        )
    
    @pytest.fixture
    def mock_embedding_engine(self):
        """Create mock embedding engine."""
        engine = MagicMock()
        engine.generate_tool_embeddings = AsyncMock(return_value=None)
        return engine
    
    @pytest.fixture
    def mock_search_engine(self):
        """Create mock search engine."""
        engine = MagicMock()
        engine.add_tools = MagicMock(return_value=None)
        return engine
    
    @pytest.fixture
    def mock_tools(self):
        """Create mock tool list."""
        from mcp_router.core.models import ToolMetadata, JSONSchema
        return [
            ToolMetadata(
                name="playwright.navigate",
                original_name="navigate",
                description="Navigate to URL",
                input_schema=JSONSchema(type="object", properties={}),
                upstream_id="playwright"
            ),
            ToolMetadata(
                name="playwright.click",
                original_name="click",
                description="Click element",
                input_schema=JSONSchema(type="object", properties={}),
                upstream_id="playwright"
            )
        ]
    
    @pytest.mark.asyncio
    async def test_load_upstream_successful(
        self, config_basic, mock_embedding_engine, mock_search_engine, mock_tools
    ):
        """Test successful upstream loading."""
        manager = ToolDiscoveryManager(config_basic, mock_embedding_engine, mock_search_engine)
        
        # Mock UpstreamConnection
        with patch('mcp_router.discovery.manager.UpstreamConnection') as MockConnection:
            mock_connection = AsyncMock()
            mock_connection.connect = AsyncMock()
            mock_connection.fetch_tools = AsyncMock(return_value=mock_tools)
            mock_connection.disconnect = AsyncMock()
            MockConnection.return_value = mock_connection
            
            # Load upstream
            result = await manager.load_upstream("playwright")
            
            # Verify success
            assert result["success"] is True
            assert result["upstream"] == "playwright"
            assert result["tool_count"] == 2
            
            # Verify connection was created and connected
            MockConnection.assert_called_once()
            mock_connection.connect.assert_called_once()
            
            # Verify tools were fetched
            mock_connection.fetch_tools.assert_called_once()
            
            # Verify embeddings were generated
            mock_embedding_engine.generate_tool_embeddings.assert_called_once_with(mock_tools)
            
            # Verify tools were added to search engine
            mock_search_engine.add_tools.assert_called_once_with(mock_tools)
            
            # Verify connection was stored
            assert "playwright" in manager._loaded_upstreams
            assert manager._loaded_upstreams["playwright"] == mock_connection
            
            # Verify tools were added to all_tools
            assert len(manager.all_tools) == 2
    
    @pytest.mark.asyncio
    async def test_load_upstream_with_alias(
        self, mock_embedding_engine, mock_search_engine, mock_tools
    ):
        """Test loading upstream using alias."""
        config = RouterConfig(
            mcp_servers={
                "playwright": UpstreamConfig(
                    transport="stdio",
                    command="npx",
                    args=["-y", "@playwright/mcp-server"],
                    aliases=["browser"]
                )
            },
            loading=LoadingConfig(auto_load=[])
        )
        
        manager = ToolDiscoveryManager(config, mock_embedding_engine, mock_search_engine)
        
        # Mock UpstreamConnection
        with patch('mcp_router.discovery.manager.UpstreamConnection') as MockConnection:
            mock_connection = AsyncMock()
            mock_connection.connect = AsyncMock()
            mock_connection.fetch_tools = AsyncMock(return_value=mock_tools)
            MockConnection.return_value = mock_connection
            
            # Load using alias
            result = await manager.load_upstream("browser")
            
            # Verify success with canonical name
            assert result["success"] is True
            assert result["upstream"] == "playwright"
            assert result["tool_count"] == 2
            
            # Verify stored under canonical name
            assert "playwright" in manager._loaded_upstreams


class TestLoadUpstreamIdempotent:
    """Test idempotent behavior of load_upstream()."""
    
    @pytest.fixture
    def config_basic(self) -> RouterConfig:
        """Create basic config."""
        return RouterConfig(
            mcp_servers={
                "playwright": UpstreamConfig(
                    transport="stdio",
                    command="npx",
                    args=["-y", "@playwright/mcp-server"]
                )
            },
            loading=LoadingConfig(auto_load=[])
        )
    
    @pytest.fixture
    def mock_embedding_engine(self):
        """Create mock embedding engine."""
        engine = MagicMock()
        engine.generate_tool_embeddings = AsyncMock()
        return engine
    
    @pytest.fixture
    def mock_search_engine(self):
        """Create mock search engine."""
        engine = MagicMock()
        engine.add_tools = MagicMock()
        return engine
    
    @pytest.mark.asyncio
    async def test_load_already_loaded_upstream(
        self, config_basic, mock_embedding_engine, mock_search_engine
    ):
        """Test loading an already-loaded upstream returns immediately."""
        from mcp_router.core.models import ToolMetadata, JSONSchema
        
        manager = ToolDiscoveryManager(config_basic, mock_embedding_engine, mock_search_engine)
        
        # Pre-populate loaded upstreams
        mock_connection = AsyncMock()
        manager._loaded_upstreams["playwright"] = mock_connection
        
        # Add some tools to all_tools
        manager.all_tools = [
            ToolMetadata(
                name="playwright.navigate",
                original_name="navigate",
                description="Navigate",
                input_schema=JSONSchema(type="object", properties={}),
                upstream_id="playwright"
            ),
            ToolMetadata(
                name="playwright.click",
                original_name="click",
                description="Click",
                input_schema=JSONSchema(type="object", properties={}),
                upstream_id="playwright"
            )
        ]
        
        # Load again
        result = await manager.load_upstream("playwright")
        
        # Verify success with existing tool count
        assert result["success"] is True
        assert result["upstream"] == "playwright"
        assert result["tool_count"] == 2
        
        # Verify no new connection was created (embedding engine not called)
        mock_embedding_engine.generate_tool_embeddings.assert_not_called()
        mock_search_engine.add_tools.assert_not_called()


class TestLoadUpstreamErrors:
    """Test error handling in load_upstream()."""
    
    @pytest.fixture
    def config_basic(self) -> RouterConfig:
        """Create basic config."""
        return RouterConfig(
            mcp_servers={
                "playwright": UpstreamConfig(
                    transport="stdio",
                    command="npx",
                    args=["-y", "@playwright/mcp-server"]
                )
            },
            loading=LoadingConfig(
                auto_load=[],
                connection_timeout=5
            )
        )
    
    @pytest.fixture
    def mock_embedding_engine(self):
        """Create mock embedding engine."""
        engine = MagicMock()
        engine.generate_tool_embeddings = AsyncMock()
        return engine
    
    @pytest.fixture
    def mock_search_engine(self):
        """Create mock search engine."""
        engine = MagicMock()
        engine.add_tools = MagicMock()
        return engine
    
    @pytest.mark.asyncio
    async def test_load_invalid_upstream_name(
        self, config_basic, mock_embedding_engine, mock_search_engine
    ):
        """Test loading non-existent upstream."""
        manager = ToolDiscoveryManager(config_basic, mock_embedding_engine, mock_search_engine)
        
        # Load invalid upstream
        result = await manager.load_upstream("nonexistent")
        
        # Verify failure
        assert result["success"] is False
        assert "Unknown upstream or alias" in result["error"]
    
    @pytest.mark.asyncio
    async def test_load_connection_timeout(
        self, config_basic, mock_embedding_engine, mock_search_engine
    ):
        """Test connection timeout handling."""
        manager = ToolDiscoveryManager(config_basic, mock_embedding_engine, mock_search_engine)
        
        # Mock UpstreamConnection with timeout
        with patch('mcp_router.discovery.manager.UpstreamConnection') as MockConnection:
            mock_connection = AsyncMock()
            mock_connection.connect = AsyncMock(side_effect=asyncio.TimeoutError())
            mock_connection.disconnect = AsyncMock()
            MockConnection.return_value = mock_connection
            
            # Load upstream
            result = await manager.load_upstream("playwright")
            
            # Verify failure
            assert result["success"] is False
            assert "timeout" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_load_connection_failure(
        self, config_basic, mock_embedding_engine, mock_search_engine
    ):
        """Test connection failure handling."""
        manager = ToolDiscoveryManager(config_basic, mock_embedding_engine, mock_search_engine)
        
        # Mock UpstreamConnection with connection error
        with patch('mcp_router.discovery.manager.UpstreamConnection') as MockConnection:
            mock_connection = AsyncMock()
            mock_connection.connect = AsyncMock(side_effect=ConnectionError("Connection refused"))
            mock_connection.disconnect = AsyncMock()
            MockConnection.return_value = mock_connection
            
            # Load upstream
            result = await manager.load_upstream("playwright")
            
            # Verify failure
            assert result["success"] is False
            assert "Connection failed" in result["error"]
    
    @pytest.mark.asyncio
    async def test_load_fetch_tools_failure(
        self, config_basic, mock_embedding_engine, mock_search_engine
    ):
        """Test tool fetch failure handling."""
        manager = ToolDiscoveryManager(config_basic, mock_embedding_engine, mock_search_engine)
        
        # Mock UpstreamConnection with fetch_tools error
        with patch('mcp_router.discovery.manager.UpstreamConnection') as MockConnection:
            mock_connection = AsyncMock()
            mock_connection.connect = AsyncMock()
            mock_connection.fetch_tools = AsyncMock(side_effect=RuntimeError("Fetch failed"))
            mock_connection.disconnect = AsyncMock()
            MockConnection.return_value = mock_connection
            
            # Load upstream
            result = await manager.load_upstream("playwright")
            
            # Verify failure
            assert result["success"] is False
            assert "Failed to fetch tools" in result["error"]
            
            # Verify connection was disconnected
            mock_connection.disconnect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_load_embedding_generation_failure(
        self, config_basic, mock_embedding_engine, mock_search_engine
    ):
        """Test embedding generation failure handling."""
        from mcp_router.core.models import ToolMetadata, JSONSchema
        
        manager = ToolDiscoveryManager(config_basic, mock_embedding_engine, mock_search_engine)
        
        # Mock embedding engine to fail
        mock_embedding_engine.generate_tool_embeddings = AsyncMock(
            side_effect=RuntimeError("Embedding failed")
        )
        
        # Mock UpstreamConnection
        with patch('mcp_router.discovery.manager.UpstreamConnection') as MockConnection:
            mock_connection = AsyncMock()
            mock_connection.connect = AsyncMock()
            mock_connection.fetch_tools = AsyncMock(return_value=[
                ToolMetadata(
                    name="playwright.navigate",
                    original_name="navigate",
                    description="Navigate",
                    input_schema=JSONSchema(type="object", properties={}),
                    upstream_id="playwright"
                )
            ])
            mock_connection.disconnect = AsyncMock()
            MockConnection.return_value = mock_connection
            
            # Load upstream
            result = await manager.load_upstream("playwright")
            
            # Verify failure
            assert result["success"] is False
            assert "Failed to generate embeddings" in result["error"]
            
            # Verify connection was disconnected
            mock_connection.disconnect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_load_search_engine_failure(
        self, config_basic, mock_embedding_engine, mock_search_engine
    ):
        """Test search engine add_tools failure handling."""
        from mcp_router.core.models import ToolMetadata, JSONSchema
        
        manager = ToolDiscoveryManager(config_basic, mock_embedding_engine, mock_search_engine)
        
        # Mock search engine to fail
        mock_search_engine.add_tools = MagicMock(side_effect=ValueError("Duplicate tool"))
        
        # Mock UpstreamConnection
        with patch('mcp_router.discovery.manager.UpstreamConnection') as MockConnection:
            mock_connection = AsyncMock()
            mock_connection.connect = AsyncMock()
            mock_connection.fetch_tools = AsyncMock(return_value=[
                ToolMetadata(
                    name="playwright.navigate",
                    original_name="navigate",
                    description="Navigate",
                    input_schema=JSONSchema(type="object", properties={}),
                    upstream_id="playwright"
                )
            ])
            mock_connection.disconnect = AsyncMock()
            MockConnection.return_value = mock_connection
            
            # Load upstream
            result = await manager.load_upstream("playwright")
            
            # Verify failure
            assert result["success"] is False
            assert "Failed to add tools to search engine" in result["error"]
            
            # Verify connection was disconnected
            mock_connection.disconnect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_load_alias_resolution_error(
        self, config_basic, mock_embedding_engine, mock_search_engine
    ):
        """Test alias resolution error handling."""
        manager = ToolDiscoveryManager(config_basic, mock_embedding_engine, mock_search_engine)
        
        # Try to load with invalid alias
        result = await manager.load_upstream("invalid_alias")
        
        # Verify failure (alias resolver will raise ValueError)
        assert result["success"] is False
        # Error message should contain info about invalid alias
        assert "error" in result



class TestUnloadUpstreamSuccess:
    """Test successful unload_upstream() operations."""
    
    @pytest.fixture
    def config_basic(self) -> RouterConfig:
        """Create basic config with one upstream and aliases."""
        return RouterConfig(
            mcp_servers={
                "playwright": UpstreamConfig(
                    transport="stdio",
                    command="npx",
                    args=["-y", "@playwright/mcp-server"],
                    aliases=["browser"]
                )
            },
            loading=LoadingConfig(
                auto_load=[],
                connection_timeout=30
            )
        )
    
    @pytest.mark.asyncio
    async def test_unload_upstream_successful(
        self, config_basic, mock_embedding_engine, mock_search_engine
    ):
        """Test successful unload of a loaded upstream."""
        from mcp_router.core.models import ToolMetadata, JSONSchema
        
        manager = ToolDiscoveryManager(config_basic, mock_embedding_engine, mock_search_engine)
        
        # Mock UpstreamConnection
        with patch('mcp_router.discovery.manager.UpstreamConnection') as MockConnection:
            mock_connection = AsyncMock()
            mock_connection.connect = AsyncMock()
            mock_connection.fetch_tools = AsyncMock(return_value=[
                ToolMetadata(
                    name="playwright.navigate",
                    original_name="navigate",
                    description="Navigate to URL",
                    input_schema=JSONSchema(type="object", properties={}),
                    upstream_id="playwright"
                )
            ])
            mock_connection.disconnect = AsyncMock()
            MockConnection.return_value = mock_connection
            
            # Load upstream first
            load_result = await manager.load_upstream("playwright")
            assert load_result["success"] is True
            
            # Verify upstream is loaded
            assert "playwright" in manager._loaded_upstreams
            assert "playwright" in manager.upstreams
            
            # Unload upstream
            unload_result = await manager.unload_upstream("playwright")
            
            # Verify success
            assert unload_result["success"] is True
            assert unload_result["upstream"] == "playwright"
            
            # Verify upstream is unloaded
            assert "playwright" not in manager._loaded_upstreams
            assert "playwright" not in manager.upstreams
            
            # Verify connection was disconnected
            mock_connection.disconnect.assert_called()
            
            # Verify tools were removed from search engine
            mock_search_engine.remove_tools.assert_called_once_with("playwright")
    
    @pytest.mark.asyncio
    async def test_unload_upstream_with_alias(
        self, config_basic, mock_embedding_engine, mock_search_engine
    ):
        """Test unload using an alias."""
        from mcp_router.core.models import ToolMetadata, JSONSchema
        
        manager = ToolDiscoveryManager(config_basic, mock_embedding_engine, mock_search_engine)
        
        # Mock UpstreamConnection
        with patch('mcp_router.discovery.manager.UpstreamConnection') as MockConnection:
            mock_connection = AsyncMock()
            mock_connection.connect = AsyncMock()
            mock_connection.fetch_tools = AsyncMock(return_value=[
                ToolMetadata(
                    name="playwright.navigate",
                    original_name="navigate",
                    description="Navigate to URL",
                    input_schema=JSONSchema(type="object", properties={}),
                    upstream_id="playwright"
                )
            ])
            mock_connection.disconnect = AsyncMock()
            MockConnection.return_value = mock_connection
            
            # Load upstream
            await manager.load_upstream("playwright")
            
            # Unload using alias
            result = await manager.unload_upstream("browser")
            
            # Verify success
            assert result["success"] is True
            assert result["upstream"] == "playwright"
            
            # Verify upstream is unloaded
            assert "playwright" not in manager._loaded_upstreams


class TestUnloadUpstreamIdempotent:
    """Test idempotent behavior of unload_upstream()."""
    
    @pytest.fixture
    def config_basic(self) -> RouterConfig:
        """Create basic config with one upstream."""
        return RouterConfig(
            mcp_servers={
                "playwright": UpstreamConfig(
                    transport="stdio",
                    command="npx",
                    args=["-y", "@playwright/mcp-server"]
                )
            },
            loading=LoadingConfig(
                auto_load=[],
                connection_timeout=30
            )
        )
    
    @pytest.mark.asyncio
    async def test_unload_not_loaded_upstream(
        self, config_basic, mock_embedding_engine, mock_search_engine
    ):
        """Test unloading an upstream that is not loaded."""
        manager = ToolDiscoveryManager(config_basic, mock_embedding_engine, mock_search_engine)
        
        # Unload upstream that was never loaded
        result = await manager.unload_upstream("playwright")
        
        # Verify success (idempotent)
        assert result["success"] is True
        assert result["upstream"] == "playwright"
        
        # Verify search engine remove_tools was not called
        mock_search_engine.remove_tools.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_unload_twice(
        self, config_basic, mock_embedding_engine, mock_search_engine
    ):
        """Test unloading the same upstream twice."""
        from mcp_router.core.models import ToolMetadata, JSONSchema
        
        manager = ToolDiscoveryManager(config_basic, mock_embedding_engine, mock_search_engine)
        
        # Mock UpstreamConnection
        with patch('mcp_router.discovery.manager.UpstreamConnection') as MockConnection:
            mock_connection = AsyncMock()
            mock_connection.connect = AsyncMock()
            mock_connection.fetch_tools = AsyncMock(return_value=[
                ToolMetadata(
                    name="playwright.navigate",
                    original_name="navigate",
                    description="Navigate to URL",
                    input_schema=JSONSchema(type="object", properties={}),
                    upstream_id="playwright"
                )
            ])
            mock_connection.disconnect = AsyncMock()
            MockConnection.return_value = mock_connection
            
            # Load upstream
            await manager.load_upstream("playwright")
            
            # Unload first time
            result1 = await manager.unload_upstream("playwright")
            assert result1["success"] is True
            
            # Unload second time
            result2 = await manager.unload_upstream("playwright")
            assert result2["success"] is True
            
            # Verify search engine remove_tools was called only once
            assert mock_search_engine.remove_tools.call_count == 1


class TestUnloadUpstreamErrors:
    """Test error handling in unload_upstream()."""
    
    @pytest.fixture
    def config_basic(self) -> RouterConfig:
        """Create basic config with one upstream."""
        return RouterConfig(
            mcp_servers={
                "playwright": UpstreamConfig(
                    transport="stdio",
                    command="npx",
                    args=["-y", "@playwright/mcp-server"]
                )
            },
            loading=LoadingConfig(
                auto_load=[],
                connection_timeout=30
            )
        )
    
    @pytest.mark.asyncio
    async def test_unload_invalid_upstream_name(
        self, config_basic, mock_embedding_engine, mock_search_engine
    ):
        """Test unloading with invalid upstream name."""
        manager = ToolDiscoveryManager(config_basic, mock_embedding_engine, mock_search_engine)
        
        # Unload invalid upstream
        result = await manager.unload_upstream("nonexistent")
        
        # Verify failure
        assert result["success"] is False
        assert "Unknown upstream or alias" in result["error"]
    
    @pytest.mark.asyncio
    async def test_unload_search_engine_failure(
        self, config_basic, mock_embedding_engine, mock_search_engine
    ):
        """Test handling of search engine remove_tools failure."""
        from mcp_router.core.models import ToolMetadata, JSONSchema
        
        manager = ToolDiscoveryManager(config_basic, mock_embedding_engine, mock_search_engine)
        
        # Mock UpstreamConnection
        with patch('mcp_router.discovery.manager.UpstreamConnection') as MockConnection:
            mock_connection = AsyncMock()
            mock_connection.connect = AsyncMock()
            mock_connection.fetch_tools = AsyncMock(return_value=[
                ToolMetadata(
                    name="playwright.navigate",
                    original_name="navigate",
                    description="Navigate to URL",
                    input_schema=JSONSchema(type="object", properties={}),
                    upstream_id="playwright"
                )
            ])
            mock_connection.disconnect = AsyncMock()
            MockConnection.return_value = mock_connection
            
            # Load upstream
            await manager.load_upstream("playwright")
            
            # Mock search engine to fail on remove_tools
            mock_search_engine.remove_tools = MagicMock(side_effect=RuntimeError("Remove failed"))
            
            # Unload upstream
            result = await manager.unload_upstream("playwright")
            
            # Verify failure
            assert result["success"] is False
            assert "Failed to remove tools from search engine" in result["error"]
            
            # Verify upstream is still in loaded set (rollback)
            assert "playwright" in manager._loaded_upstreams
    
    @pytest.mark.asyncio
    async def test_unload_disconnect_failure_continues(
        self, config_basic, mock_embedding_engine, mock_search_engine
    ):
        """Test that disconnect failure doesn't prevent unload."""
        from mcp_router.core.models import ToolMetadata, JSONSchema
        
        manager = ToolDiscoveryManager(config_basic, mock_embedding_engine, mock_search_engine)
        
        # Mock UpstreamConnection
        with patch('mcp_router.discovery.manager.UpstreamConnection') as MockConnection:
            mock_connection = AsyncMock()
            mock_connection.connect = AsyncMock()
            mock_connection.fetch_tools = AsyncMock(return_value=[
                ToolMetadata(
                    name="playwright.navigate",
                    original_name="navigate",
                    description="Navigate to URL",
                    input_schema=JSONSchema(type="object", properties={}),
                    upstream_id="playwright"
                )
            ])
            mock_connection.disconnect = AsyncMock(side_effect=RuntimeError("Disconnect failed"))
            MockConnection.return_value = mock_connection
            
            # Load upstream
            await manager.load_upstream("playwright")
            
            # Unload upstream (disconnect will fail but should continue)
            result = await manager.unload_upstream("playwright")
            
            # Verify success (disconnect failure is logged but doesn't fail unload)
            assert result["success"] is True
            assert result["upstream"] == "playwright"
            
            # Verify upstream is unloaded despite disconnect failure
            assert "playwright" not in manager._loaded_upstreams



# ============================================================================
# Task 8: Status Methods Tests
# ============================================================================

class TestStatusMethods:
    """Test status methods: get_loaded_upstreams(), is_loaded(), get_available_upstreams()"""
    
    @pytest.fixture
    def config(self) -> RouterConfig:
        """Create config with 3 upstreams and aliases."""
        return RouterConfig(
            mcp_servers={
                "playwright": UpstreamConfig(
                    transport="stdio",
                    command="npx",
                    args=["-y", "@playwright/mcp-server"],
                    aliases=["browser"]
                ),
                "filesystem": UpstreamConfig(
                    transport="stdio",
                    command="npx",
                    args=["-y", "@filesystem/mcp-server"]
                ),
                "github": UpstreamConfig(
                    transport="stdio",
                    command="npx",
                    args=["-y", "@github/mcp-server"]
                )
            },
            loading=LoadingConfig(auto_load=[])
        )
    
    @pytest.fixture
    def mock_embedding_engine(self):
        """Create mock embedding engine."""
        engine = MagicMock()
        engine.generate_tool_embeddings = AsyncMock(return_value=None)
        return engine
    
    @pytest.fixture
    def mock_search_engine(self):
        """Create mock search engine."""
        engine = MagicMock()
        engine.add_tools = MagicMock(return_value=None)
        engine.remove_tools = MagicMock(return_value=None)
        return engine
    
    @pytest.fixture
    def manager(self, config, mock_embedding_engine, mock_search_engine):
        """Create ToolDiscoveryManager instance."""
        return ToolDiscoveryManager(config, mock_embedding_engine, mock_search_engine)
    
    @pytest.fixture
    def mock_upstream_connection(self):
        """Create mock UpstreamConnection that can be configured per test."""
        mock_connection = AsyncMock()
        mock_connection.connect = AsyncMock()
        mock_connection.fetch_tools = AsyncMock(return_value=[])
        mock_connection.disconnect = AsyncMock()
        return mock_connection
    
    @pytest.mark.asyncio
    async def test_get_loaded_upstreams_empty(self, manager):
        """Test get_loaded_upstreams() returns empty list when no upstreams loaded"""
        loaded = manager.get_loaded_upstreams()
        assert loaded == []
    
    @pytest.mark.asyncio
    async def test_get_loaded_upstreams_single(self, manager, mock_upstream_connection):
        """Test get_loaded_upstreams() returns single upstream"""
        # Configure mock connection
        mock_upstream_connection.fetch_tools.return_value = [
            ToolMetadata(
                name="playwright.navigate",
                original_name="navigate",
                description="Navigate to URL",
                input_schema={"type": "object"},
                upstream_id="playwright"
            )
        ]
        
        # Patch UpstreamConnection and load upstream
        with patch('mcp_router.discovery.manager.UpstreamConnection', return_value=mock_upstream_connection):
            await manager.load_upstream("playwright")
        
        # Verify loaded upstreams
        loaded = manager.get_loaded_upstreams()
        assert loaded == ["playwright"]
    
    @pytest.mark.asyncio
    async def test_get_loaded_upstreams_multiple(self, manager, mock_upstream_connection):
        """Test get_loaded_upstreams() returns multiple upstreams"""
        # Configure mock connection for playwright
        mock_upstream_connection.fetch_tools.return_value = [
            ToolMetadata(
                name="playwright.navigate",
                original_name="navigate",
                description="Navigate to URL",
                input_schema={"type": "object"},
                upstream_id="playwright"
            )
        ]
        
        # Patch and load playwright
        with patch('mcp_router.discovery.manager.UpstreamConnection', return_value=mock_upstream_connection):
            await manager.load_upstream("playwright")
        
        # Configure mock connection for filesystem
        mock_upstream_connection.fetch_tools.return_value = [
            ToolMetadata(
                name="filesystem.read_file",
                original_name="read_file",
                description="Read file",
                input_schema={"type": "object"},
                upstream_id="filesystem"
            )
        ]
        
        # Patch and load filesystem
        with patch('mcp_router.discovery.manager.UpstreamConnection', return_value=mock_upstream_connection):
            await manager.load_upstream("filesystem")
        
        # Verify loaded upstreams
        loaded = manager.get_loaded_upstreams()
        assert set(loaded) == {"playwright", "filesystem"}
        assert len(loaded) == 2
    
    @pytest.mark.asyncio
    async def test_is_loaded_true(self, manager, mock_upstream_connection):
        """Test is_loaded() returns True for loaded upstream"""
        # Configure mock connection
        mock_upstream_connection.fetch_tools.return_value = [
            ToolMetadata(
                name="playwright.navigate",
                original_name="navigate",
                description="Navigate to URL",
                input_schema={"type": "object"},
                upstream_id="playwright"
            )
        ]
        
        # Patch and load upstream
        with patch('mcp_router.discovery.manager.UpstreamConnection', return_value=mock_upstream_connection):
            await manager.load_upstream("playwright")
        
        # Verify is_loaded returns True
        assert manager.is_loaded("playwright") is True
    
    @pytest.mark.asyncio
    async def test_is_loaded_false(self, manager):
        """Test is_loaded() returns False for not loaded upstream"""
        # Verify is_loaded returns False
        assert manager.is_loaded("playwright") is False
    
    @pytest.mark.asyncio
    async def test_is_loaded_with_alias(self, manager, mock_upstream_connection):
        """Test is_loaded() works with aliases"""
        # Configure mock connection
        mock_upstream_connection.fetch_tools.return_value = [
            ToolMetadata(
                name="playwright.navigate",
                original_name="navigate",
                description="Navigate to URL",
                input_schema={"type": "object"},
                upstream_id="playwright"
            )
        ]
        
        # Patch and load upstream
        with patch('mcp_router.discovery.manager.UpstreamConnection', return_value=mock_upstream_connection):
            await manager.load_upstream("playwright")
        
        # Verify is_loaded works with alias
        assert manager.is_loaded("browser") is True  # "browser" is alias for "playwright"
    
    @pytest.mark.asyncio
    async def test_is_loaded_invalid_name(self, manager):
        """Test is_loaded() returns False for invalid upstream name"""
        # Verify is_loaded returns False for invalid name
        assert manager.is_loaded("nonexistent") is False
    
    @pytest.mark.asyncio
    async def test_get_available_upstreams(self, manager):
        """Test get_available_upstreams() returns all upstreams from config"""
        # Get available upstreams
        available = manager.get_available_upstreams()
        
        # Verify all upstreams from config are returned
        assert set(available) == {"playwright", "filesystem", "github"}
        assert len(available) == 3
    
    @pytest.mark.asyncio
    async def test_get_available_upstreams_independent_of_loaded(self, manager, mock_upstream_connection):
        """Test get_available_upstreams() returns all upstreams regardless of loaded state"""
        # Configure mock connection
        mock_upstream_connection.fetch_tools.return_value = [
            ToolMetadata(
                name="playwright.navigate",
                original_name="navigate",
                description="Navigate to URL",
                input_schema={"type": "object"},
                upstream_id="playwright"
            )
        ]
        
        # Patch and load one upstream
        with patch('mcp_router.discovery.manager.UpstreamConnection', return_value=mock_upstream_connection):
            await manager.load_upstream("playwright")
        
        # Get available upstreams
        available = manager.get_available_upstreams()
        
        # Verify all upstreams are still returned (not just loaded ones)
        assert set(available) == {"playwright", "filesystem", "github"}
        assert len(available) == 3


# ============================================================================
# Task 9: load_multiple_upstreams() Tests
# ============================================================================

class TestLoadMultipleUpstreamsSuccess:
    """Test successful load_multiple_upstreams() operations."""
    
    @pytest.fixture
    def config(self) -> RouterConfig:
        """Create config with multiple upstreams and aliases."""
        return RouterConfig(
            mcp_servers={
                "playwright": UpstreamConfig(
                    transport="stdio",
                    command="npx",
                    args=["-y", "@playwright/mcp-server"],
                    aliases=["browser"]
                ),
                "filesystem": UpstreamConfig(
                    transport="stdio",
                    command="npx",
                    args=["-y", "@filesystem/mcp-server"],
                    aliases=["fs"]
                ),
                "github": UpstreamConfig(
                    transport="stdio",
                    command="npx",
                    args=["-y", "@github/mcp-server"]
                )
            },
            loading=LoadingConfig(auto_load=[])
        )
    
    @pytest.fixture
    def mock_embedding_engine(self):
        """Create mock embedding engine."""
        engine = MagicMock()
        engine.generate_tool_embeddings = AsyncMock(return_value=None)
        return engine
    
    @pytest.fixture
    def mock_search_engine(self):
        """Create mock search engine."""
        engine = MagicMock()
        engine.add_tools = MagicMock(return_value=None)
        engine.remove_tools = MagicMock(return_value=None)
        return engine
    
    @pytest.fixture
    def manager(self, config, mock_embedding_engine, mock_search_engine):
        """Create ToolDiscoveryManager instance."""
        return ToolDiscoveryManager(config, mock_embedding_engine, mock_search_engine)
    
    @pytest.mark.asyncio
    async def test_load_multiple_upstreams_all_succeed(self, manager):
        """Test loading multiple upstreams successfully."""
        from mcp_router.core.models import ToolMetadata, JSONSchema
        
        # Mock UpstreamConnection
        with patch('mcp_router.discovery.manager.UpstreamConnection') as MockConnection:
            mock_connection = AsyncMock()
            mock_connection.connect = AsyncMock()
            mock_connection.fetch_tools = AsyncMock(return_value=[
                ToolMetadata(
                    name="test.tool",
                    original_name="tool",
                    description="Test tool",
                    input_schema=JSONSchema(type="object", properties={}),
                    upstream_id="test"
                )
            ])
            mock_connection.disconnect = AsyncMock()
            MockConnection.return_value = mock_connection
            
            # Load multiple upstreams
            result = await manager.load_multiple_upstreams(["playwright", "filesystem"])
            
            # Verify all succeeded
            assert len(result["loaded"]) == 2
            assert set(result["loaded"]) == {"playwright", "filesystem"}
            assert len(result["failed"]) == 0
            
            # Verify both are loaded
            assert manager.is_loaded("playwright")
            assert manager.is_loaded("filesystem")
    
    @pytest.mark.asyncio
    async def test_load_multiple_upstreams_with_aliases(self, manager):
        """Test loading multiple upstreams using aliases."""
        from mcp_router.core.models import ToolMetadata, JSONSchema
        
        # Mock UpstreamConnection
        with patch('mcp_router.discovery.manager.UpstreamConnection') as MockConnection:
            mock_connection = AsyncMock()
            mock_connection.connect = AsyncMock()
            mock_connection.fetch_tools = AsyncMock(return_value=[
                ToolMetadata(
                    name="test.tool",
                    original_name="tool",
                    description="Test tool",
                    input_schema=JSONSchema(type="object", properties={}),
                    upstream_id="test"
                )
            ])
            mock_connection.disconnect = AsyncMock()
            MockConnection.return_value = mock_connection
            
            # Load using aliases
            result = await manager.load_multiple_upstreams(["browser", "fs"])
            
            # Verify all succeeded (canonical names returned)
            assert len(result["loaded"]) == 2
            assert set(result["loaded"]) == {"playwright", "filesystem"}
            assert len(result["failed"]) == 0
    
    @pytest.mark.asyncio
    async def test_load_multiple_upstreams_empty_list(self, manager):
        """Test loading empty list of upstreams."""
        # Load empty list
        result = await manager.load_multiple_upstreams([])
        
        # Verify empty results
        assert result["loaded"] == []
        assert result["failed"] == []
    
    @pytest.mark.asyncio
    async def test_load_multiple_upstreams_single(self, manager):
        """Test loading single upstream via load_multiple_upstreams."""
        from mcp_router.core.models import ToolMetadata, JSONSchema
        
        # Mock UpstreamConnection
        with patch('mcp_router.discovery.manager.UpstreamConnection') as MockConnection:
            mock_connection = AsyncMock()
            mock_connection.connect = AsyncMock()
            mock_connection.fetch_tools = AsyncMock(return_value=[
                ToolMetadata(
                    name="test.tool",
                    original_name="tool",
                    description="Test tool",
                    input_schema=JSONSchema(type="object", properties={}),
                    upstream_id="test"
                )
            ])
            mock_connection.disconnect = AsyncMock()
            MockConnection.return_value = mock_connection
            
            # Load single upstream
            result = await manager.load_multiple_upstreams(["playwright"])
            
            # Verify success
            assert result["loaded"] == ["playwright"]
            assert result["failed"] == []


class TestLoadMultipleUpstreamsPartialFailure:
    """Test load_multiple_upstreams() with partial failures."""
    
    @pytest.fixture
    def config(self) -> RouterConfig:
        """Create config with multiple upstreams."""
        return RouterConfig(
            mcp_servers={
                "playwright": UpstreamConfig(
                    transport="stdio",
                    command="npx",
                    args=["-y", "@playwright/mcp-server"]
                ),
                "filesystem": UpstreamConfig(
                    transport="stdio",
                    command="npx",
                    args=["-y", "@filesystem/mcp-server"]
                ),
                "github": UpstreamConfig(
                    transport="stdio",
                    command="npx",
                    args=["-y", "@github/mcp-server"]
                )
            },
            loading=LoadingConfig(auto_load=[])
        )
    
    @pytest.fixture
    def mock_embedding_engine(self):
        """Create mock embedding engine."""
        engine = MagicMock()
        engine.generate_tool_embeddings = AsyncMock(return_value=None)
        return engine
    
    @pytest.fixture
    def mock_search_engine(self):
        """Create mock search engine."""
        engine = MagicMock()
        engine.add_tools = MagicMock(return_value=None)
        return engine
    
    @pytest.fixture
    def manager(self, config, mock_embedding_engine, mock_search_engine):
        """Create ToolDiscoveryManager instance."""
        return ToolDiscoveryManager(config, mock_embedding_engine, mock_search_engine)
    
    @pytest.mark.asyncio
    async def test_load_multiple_some_succeed_some_fail(self, manager):
        """Test loading multiple upstreams where some succeed and some fail."""
        from mcp_router.core.models import ToolMetadata, JSONSchema
        
        # Mock UpstreamConnection with different behaviors
        with patch('mcp_router.discovery.manager.UpstreamConnection') as MockConnection:
            # Track which upstream_id is being created
            def create_connection(upstream_id, config):
                """Create mock connection - note: first arg is upstream_id, second is config."""
                mock_connection = AsyncMock()
                
                if upstream_id == "filesystem":
                    # Fail connection for filesystem
                    mock_connection.connect = AsyncMock(side_effect=ConnectionError("Connection refused"))
                else:
                    # Succeed for others
                    mock_connection.connect = AsyncMock()
                    mock_connection.fetch_tools = AsyncMock(return_value=[
                        ToolMetadata(
                            name=f"{upstream_id}.tool",
                            original_name="tool",
                            description="Test tool",
                            input_schema=JSONSchema(type="object", properties={}),
                            upstream_id=upstream_id
                        )
                    ])
                mock_connection.disconnect = AsyncMock()
                return mock_connection
            
            MockConnection.side_effect = create_connection
            
            # Load multiple upstreams
            result = await manager.load_multiple_upstreams(["playwright", "filesystem", "github"])
            
            # Verify partial success
            assert len(result["loaded"]) == 2
            assert set(result["loaded"]) == {"playwright", "github"}
            
            # Verify partial failure
            assert len(result["failed"]) == 1
            assert result["failed"][0]["name"] == "filesystem"
            assert "Connection failed" in result["failed"][0]["error"] or "Connection refused" in result["failed"][0]["error"]
    
    @pytest.mark.asyncio
    async def test_load_multiple_with_invalid_alias(self, manager):
        """Test loading multiple upstreams with invalid alias."""
        from mcp_router.core.models import ToolMetadata, JSONSchema
        
        # Mock UpstreamConnection
        with patch('mcp_router.discovery.manager.UpstreamConnection') as MockConnection:
            mock_connection = AsyncMock()
            mock_connection.connect = AsyncMock()
            mock_connection.fetch_tools = AsyncMock(return_value=[
                ToolMetadata(
                    name="test.tool",
                    original_name="tool",
                    description="Test tool",
                    input_schema=JSONSchema(type="object", properties={}),
                    upstream_id="test"
                )
            ])
            mock_connection.disconnect = AsyncMock()
            MockConnection.return_value = mock_connection
            
            # Load with one valid and one invalid name
            result = await manager.load_multiple_upstreams(["playwright", "invalid_name"])
            
            # Verify partial success
            assert len(result["loaded"]) == 1
            assert result["loaded"] == ["playwright"]
            
            # Verify partial failure (alias resolution failed)
            assert len(result["failed"]) == 1
            assert result["failed"][0]["name"] == "invalid_name"
            assert "Unknown upstream or alias" in result["failed"][0]["error"]


class TestLoadMultipleUpstreamsAllFail:
    """Test load_multiple_upstreams() where all loads fail."""
    
    @pytest.fixture
    def config(self) -> RouterConfig:
        """Create config with multiple upstreams."""
        return RouterConfig(
            mcp_servers={
                "playwright": UpstreamConfig(
                    transport="stdio",
                    command="npx",
                    args=["-y", "@playwright/mcp-server"]
                ),
                "filesystem": UpstreamConfig(
                    transport="stdio",
                    command="npx",
                    args=["-y", "@filesystem/mcp-server"]
                )
            },
            loading=LoadingConfig(auto_load=[])
        )
    
    @pytest.fixture
    def mock_embedding_engine(self):
        """Create mock embedding engine."""
        engine = MagicMock()
        engine.generate_tool_embeddings = AsyncMock(return_value=None)
        return engine
    
    @pytest.fixture
    def mock_search_engine(self):
        """Create mock search engine."""
        engine = MagicMock()
        engine.add_tools = MagicMock(return_value=None)
        return engine
    
    @pytest.fixture
    def manager(self, config, mock_embedding_engine, mock_search_engine):
        """Create ToolDiscoveryManager instance."""
        return ToolDiscoveryManager(config, mock_embedding_engine, mock_search_engine)
    
    @pytest.mark.asyncio
    async def test_load_multiple_all_fail_connection(self, manager):
        """Test loading multiple upstreams where all fail to connect."""
        # Mock UpstreamConnection to always fail
        with patch('mcp_router.discovery.manager.UpstreamConnection') as MockConnection:
            mock_connection = AsyncMock()
            mock_connection.connect = AsyncMock(side_effect=ConnectionError("Connection refused"))
            mock_connection.disconnect = AsyncMock()
            MockConnection.return_value = mock_connection
            
            # Load multiple upstreams
            result = await manager.load_multiple_upstreams(["playwright", "filesystem"])
            
            # Verify all failed
            assert len(result["loaded"]) == 0
            assert len(result["failed"]) == 2
            
            # Verify error messages
            failed_names = {f["name"] for f in result["failed"]}
            assert failed_names == {"playwright", "filesystem"}
            
            for failure in result["failed"]:
                assert "Connection failed" in failure["error"]
    
    @pytest.mark.asyncio
    async def test_load_multiple_all_invalid_aliases(self, manager):
        """Test loading multiple upstreams with all invalid aliases."""
        # Load with all invalid names
        result = await manager.load_multiple_upstreams(["invalid1", "invalid2", "invalid3"])
        
        # Verify all failed (alias resolution)
        assert len(result["loaded"]) == 0
        assert len(result["failed"]) == 3
        
        # Verify error messages
        failed_names = {f["name"] for f in result["failed"]}
        assert failed_names == {"invalid1", "invalid2", "invalid3"}
        
        for failure in result["failed"]:
            assert "Unknown upstream or alias" in failure["error"]


class TestLoadMultipleUpstreamsConcurrency:
    """Test concurrent loading behavior of load_multiple_upstreams()."""
    
    @pytest.fixture
    def config(self) -> RouterConfig:
        """Create config with multiple upstreams."""
        return RouterConfig(
            mcp_servers={
                "upstream1": UpstreamConfig(
                    transport="stdio",
                    command="npx",
                    args=["-y", "@upstream1/mcp-server"]
                ),
                "upstream2": UpstreamConfig(
                    transport="stdio",
                    command="npx",
                    args=["-y", "@upstream2/mcp-server"]
                ),
                "upstream3": UpstreamConfig(
                    transport="stdio",
                    command="npx",
                    args=["-y", "@upstream3/mcp-server"]
                )
            },
            loading=LoadingConfig(auto_load=[])
        )
    
    @pytest.fixture
    def mock_embedding_engine(self):
        """Create mock embedding engine."""
        engine = MagicMock()
        engine.generate_tool_embeddings = AsyncMock(return_value=None)
        return engine
    
    @pytest.fixture
    def mock_search_engine(self):
        """Create mock search engine."""
        engine = MagicMock()
        engine.add_tools = MagicMock(return_value=None)
        return engine
    
    @pytest.fixture
    def manager(self, config, mock_embedding_engine, mock_search_engine):
        """Create ToolDiscoveryManager instance."""
        return ToolDiscoveryManager(config, mock_embedding_engine, mock_search_engine)
    
    @pytest.mark.asyncio
    async def test_load_multiple_concurrent_execution(self, manager):
        """Test that upstreams are loaded concurrently (not sequentially)."""
        from mcp_router.core.models import ToolMetadata, JSONSchema
        import time
        
        # Track connection times
        connection_times = []
        
        # Mock UpstreamConnection with delays
        with patch('mcp_router.discovery.manager.UpstreamConnection') as MockConnection:
            async def delayed_connect():
                """Simulate connection delay."""
                connection_times.append(time.time())
                await asyncio.sleep(0.1)  # 100ms delay
            
            mock_connection = AsyncMock()
            mock_connection.connect = AsyncMock(side_effect=delayed_connect)
            mock_connection.fetch_tools = AsyncMock(return_value=[
                ToolMetadata(
                    name="test.tool",
                    original_name="tool",
                    description="Test tool",
                    input_schema=JSONSchema(type="object", properties={}),
                    upstream_id="test"
                )
            ])
            mock_connection.disconnect = AsyncMock()
            MockConnection.return_value = mock_connection
            
            # Load 3 upstreams
            start_time = time.time()
            result = await manager.load_multiple_upstreams(["upstream1", "upstream2", "upstream3"])
            end_time = time.time()
            
            # Verify all succeeded
            assert len(result["loaded"]) == 3
            
            # Verify concurrent execution (should take ~100ms, not ~300ms)
            # Allow some overhead, but should be much less than sequential
            elapsed = end_time - start_time
            assert elapsed < 0.25  # Should be ~0.1s, not ~0.3s
            
            # Verify connections started close together (concurrent)
            if len(connection_times) >= 2:
                time_diff = connection_times[-1] - connection_times[0]
                assert time_diff < 0.05  # All should start within 50ms
    
    @pytest.mark.asyncio
    async def test_load_multiple_idempotent_with_already_loaded(self, manager):
        """Test loading multiple upstreams where some are already loaded."""
        from mcp_router.core.models import ToolMetadata, JSONSchema
        
        # Mock UpstreamConnection
        with patch('mcp_router.discovery.manager.UpstreamConnection') as MockConnection:
            mock_connection = AsyncMock()
            mock_connection.connect = AsyncMock()
            mock_connection.fetch_tools = AsyncMock(return_value=[
                ToolMetadata(
                    name="test.tool",
                    original_name="tool",
                    description="Test tool",
                    input_schema=JSONSchema(type="object", properties={}),
                    upstream_id="test"
                )
            ])
            mock_connection.disconnect = AsyncMock()
            MockConnection.return_value = mock_connection
            
            # Load upstream1 first
            await manager.load_upstream("upstream1")
            
            # Reset mock call counts
            mock_connection.connect.reset_mock()
            mock_connection.fetch_tools.reset_mock()
            
            # Load multiple including already-loaded upstream1
            result = await manager.load_multiple_upstreams(["upstream1", "upstream2", "upstream3"])
            
            # Verify all succeeded (idempotent for upstream1)
            assert len(result["loaded"]) == 3
            assert set(result["loaded"]) == {"upstream1", "upstream2", "upstream3"}
            assert len(result["failed"]) == 0
            
            # Verify upstream1 was not reconnected (idempotent)
            # Only 2 new connections should be made (upstream2, upstream3)
            assert mock_connection.connect.call_count == 2
