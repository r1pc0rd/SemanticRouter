"""
Unit tests for dynamic tool management in SemanticSearchEngine.

Tests the add_tools() and remove_tools() methods added for dynamic upstream loading.
"""
import pytest
import numpy as np
from unittest.mock import Mock

from mcp_router.search.engine import SemanticSearchEngine
from mcp_router.embedding.engine import EmbeddingEngine
from mcp_router.core.models import ToolMetadata, JSONSchema


@pytest.fixture
def mock_embedding_engine() -> Mock:
    """Create a mock embedding engine."""
    engine = Mock(spec=EmbeddingEngine)
    # Mock generate_embedding to return a random 384-dim vector
    engine.generate_embedding.return_value = np.random.rand(384).astype(np.float32)
    return engine


@pytest.fixture
def search_engine(mock_embedding_engine: Mock) -> SemanticSearchEngine:
    """Create a semantic search engine with mock embedding engine."""
    return SemanticSearchEngine(embedding_engine=mock_embedding_engine)


def create_test_tool(
    name: str,
    upstream_id: str,
    description: str = "Test tool",
    with_embedding: bool = True
) -> ToolMetadata:
    """Helper function to create a test tool with optional embedding."""
    schema = JSONSchema(
        type='object',
        properties={'param': {'type': 'string'}},
        required=['param']
    )
    
    embedding = None
    if with_embedding:
        embedding = np.random.rand(384).astype(np.float32)
    
    return ToolMetadata(
        name=name,
        original_name=name.split('.')[-1],
        description=description,
        input_schema=schema,
        upstream_id=upstream_id,
        embedding=embedding
    )


class TestAddTools:
    """Tests for add_tools() method."""
    
    def test_add_tools_to_empty_catalog(self, search_engine: SemanticSearchEngine) -> None:
        """Test successfully adding tools to empty catalog."""
        tools = [
            create_test_tool('playwright.navigate', 'playwright'),
            create_test_tool('playwright.click', 'playwright'),
        ]
        
        search_engine.add_tools(tools)
        
        assert search_engine.get_tool_count() == 2
    
    def test_add_tools_to_existing_catalog(self, search_engine: SemanticSearchEngine) -> None:
        """Test successfully adding tools to existing catalog."""
        # Set initial tools
        initial_tools = [
            create_test_tool('jira.create_issue', 'jira'),
        ]
        search_engine.set_tools(initial_tools)
        
        # Add more tools
        new_tools = [
            create_test_tool('playwright.navigate', 'playwright'),
            create_test_tool('playwright.click', 'playwright'),
        ]
        search_engine.add_tools(new_tools)
        
        assert search_engine.get_tool_count() == 3
    
    def test_add_tools_raises_error_for_missing_embedding(
        self,
        search_engine: SemanticSearchEngine
    ) -> None:
        """Test that add_tools raises ValueError for tools missing embeddings."""
        tools = [
            create_test_tool('playwright.navigate', 'playwright', with_embedding=True),
            create_test_tool('playwright.click', 'playwright', with_embedding=False),
        ]
        
        with pytest.raises(ValueError, match="missing embedding"):
            search_engine.add_tools(tools)
    
    def test_add_tools_raises_error_for_duplicate_names(
        self,
        search_engine: SemanticSearchEngine
    ) -> None:
        """Test that add_tools raises ValueError for duplicate tool names."""
        # Set initial tools
        initial_tools = [
            create_test_tool('playwright.navigate', 'playwright'),
        ]
        search_engine.set_tools(initial_tools)
        
        # Try to add tool with duplicate name
        new_tools = [
            create_test_tool('playwright.navigate', 'playwright'),  # Duplicate!
        ]
        
        with pytest.raises(ValueError, match="already exists"):
            search_engine.add_tools(new_tools)
    
    def test_add_tools_with_empty_list(self, search_engine: SemanticSearchEngine) -> None:
        """Test that add_tools with empty list is a no-op."""
        # Set initial tools
        initial_tools = [
            create_test_tool('jira.create_issue', 'jira'),
        ]
        search_engine.set_tools(initial_tools)
        
        # Add empty list (should be no-op)
        search_engine.add_tools([])
        
        assert search_engine.get_tool_count() == 1
    
    def test_add_tools_preserves_existing_tools(
        self,
        search_engine: SemanticSearchEngine
    ) -> None:
        """Test that add_tools preserves existing tools."""
        # Set initial tools
        initial_tools = [
            create_test_tool('jira.create_issue', 'jira'),
            create_test_tool('jira.update_issue', 'jira'),
        ]
        search_engine.set_tools(initial_tools)
        
        # Add new tools
        new_tools = [
            create_test_tool('playwright.navigate', 'playwright'),
        ]
        search_engine.add_tools(new_tools)
        
        # Verify all tools are present
        assert search_engine.get_tool_count() == 3


class TestRemoveTools:
    """Tests for remove_tools() method."""
    
    def test_remove_tools_by_namespace(self, search_engine: SemanticSearchEngine) -> None:
        """Test successfully removing tools by namespace."""
        tools = [
            create_test_tool('playwright.navigate', 'playwright'),
            create_test_tool('playwright.click', 'playwright'),
            create_test_tool('jira.create_issue', 'jira'),
        ]
        search_engine.set_tools(tools)
        
        # Remove playwright tools
        search_engine.remove_tools('playwright')
        
        assert search_engine.get_tool_count() == 1
    
    def test_remove_tools_with_nonexistent_namespace(
        self,
        search_engine: SemanticSearchEngine
    ) -> None:
        """Test removing tools with non-existent namespace (no-op)."""
        tools = [
            create_test_tool('jira.create_issue', 'jira'),
        ]
        search_engine.set_tools(tools)
        
        # Remove non-existent namespace (should be no-op)
        search_engine.remove_tools('playwright')
        
        assert search_engine.get_tool_count() == 1
    
    def test_remove_tools_raises_error_for_empty_namespace(
        self,
        search_engine: SemanticSearchEngine
    ) -> None:
        """Test that remove_tools raises ValueError for empty namespace."""
        tools = [
            create_test_tool('jira.create_issue', 'jira'),
        ]
        search_engine.set_tools(tools)
        
        with pytest.raises(ValueError, match="cannot be empty"):
            search_engine.remove_tools('')
    
    def test_remove_tools_only_removes_matching_namespace(
        self,
        search_engine: SemanticSearchEngine
    ) -> None:
        """Test that remove_tools only removes tools with matching namespace."""
        tools = [
            create_test_tool('playwright.navigate', 'playwright'),
            create_test_tool('playwright.click', 'playwright'),
            create_test_tool('jira.create_issue', 'jira'),
            create_test_tool('jira.update_issue', 'jira'),
        ]
        search_engine.set_tools(tools)
        
        # Remove only playwright tools
        search_engine.remove_tools('playwright')
        
        # Verify only jira tools remain
        assert search_engine.get_tool_count() == 2
    
    def test_remove_all_tools(self, search_engine: SemanticSearchEngine) -> None:
        """Test removing all tools from catalog."""
        tools = [
            create_test_tool('playwright.navigate', 'playwright'),
            create_test_tool('playwright.click', 'playwright'),
        ]
        search_engine.set_tools(tools)
        
        # Remove all playwright tools
        search_engine.remove_tools('playwright')
        
        assert search_engine.get_tool_count() == 0


class TestIntegrationScenarios:
    """Integration tests for add_tools and remove_tools."""
    
    @pytest.mark.asyncio
    async def test_search_works_after_adding_tools(
        self,
        search_engine: SemanticSearchEngine
    ) -> None:
        """Test that search still works after adding tools."""
        # Set initial tools
        initial_tools = [
            create_test_tool('jira.create_issue', 'jira', 'Create a new Jira issue'),
        ]
        search_engine.set_tools(initial_tools)
        
        # Add more tools
        new_tools = [
            create_test_tool('playwright.navigate', 'playwright', 'Navigate to a URL'),
        ]
        search_engine.add_tools(new_tools)
        
        # Search should work
        results = await search_engine.search_tools('navigate to website', top_k=5)
        
        assert len(results) > 0
        assert results[0].tool.name in ['playwright.navigate', 'jira.create_issue']
    
    @pytest.mark.asyncio
    async def test_search_works_after_removing_tools(
        self,
        search_engine: SemanticSearchEngine
    ) -> None:
        """Test that search still works after removing tools."""
        # Set initial tools
        tools = [
            create_test_tool('playwright.navigate', 'playwright', 'Navigate to a URL'),
            create_test_tool('jira.create_issue', 'jira', 'Create a new Jira issue'),
        ]
        search_engine.set_tools(tools)
        
        # Remove playwright tools
        search_engine.remove_tools('playwright')
        
        # Search should still work with remaining tools
        results = await search_engine.search_tools('create issue', top_k=5)
        
        assert len(results) == 1
        assert results[0].tool.name == 'jira.create_issue'
    
    def test_add_then_remove_workflow(self, search_engine: SemanticSearchEngine) -> None:
        """Test add then remove workflow."""
        # Start with jira tools
        initial_tools = [
            create_test_tool('jira.create_issue', 'jira'),
            create_test_tool('jira.update_issue', 'jira'),
        ]
        search_engine.set_tools(initial_tools)
        assert search_engine.get_tool_count() == 2
        
        # Add playwright tools
        playwright_tools = [
            create_test_tool('playwright.navigate', 'playwright'),
            create_test_tool('playwright.click', 'playwright'),
        ]
        search_engine.add_tools(playwright_tools)
        assert search_engine.get_tool_count() == 4
        
        # Remove playwright tools
        search_engine.remove_tools('playwright')
        assert search_engine.get_tool_count() == 2
        
        # Add playwright tools again
        search_engine.add_tools(playwright_tools)
        assert search_engine.get_tool_count() == 4
    
    def test_multiple_add_operations(self, search_engine: SemanticSearchEngine) -> None:
        """Test multiple add operations in sequence."""
        # Add jira tools
        jira_tools = [
            create_test_tool('jira.create_issue', 'jira'),
        ]
        search_engine.add_tools(jira_tools)
        assert search_engine.get_tool_count() == 1
        
        # Add playwright tools
        playwright_tools = [
            create_test_tool('playwright.navigate', 'playwright'),
        ]
        search_engine.add_tools(playwright_tools)
        assert search_engine.get_tool_count() == 2
        
        # Add github tools
        github_tools = [
            create_test_tool('github.create_pr', 'github'),
        ]
        search_engine.add_tools(github_tools)
        assert search_engine.get_tool_count() == 3
    
    def test_multiple_remove_operations(self, search_engine: SemanticSearchEngine) -> None:
        """Test multiple remove operations in sequence."""
        # Set initial tools from multiple upstreams
        tools = [
            create_test_tool('jira.create_issue', 'jira'),
            create_test_tool('playwright.navigate', 'playwright'),
            create_test_tool('github.create_pr', 'github'),
        ]
        search_engine.set_tools(tools)
        assert search_engine.get_tool_count() == 3
        
        # Remove jira tools
        search_engine.remove_tools('jira')
        assert search_engine.get_tool_count() == 2
        
        # Remove playwright tools
        search_engine.remove_tools('playwright')
        assert search_engine.get_tool_count() == 1
        
        # Remove github tools
        search_engine.remove_tools('github')
        assert search_engine.get_tool_count() == 0
    
    @pytest.mark.asyncio
    async def test_search_returns_only_loaded_tools(
        self,
        search_engine: SemanticSearchEngine
    ) -> None:
        """Test that search only returns tools from loaded upstreams."""
        # Set initial tools
        tools = [
            create_test_tool('playwright.navigate', 'playwright', 'Navigate to a URL'),
            create_test_tool('jira.create_issue', 'jira', 'Create a new Jira issue'),
        ]
        search_engine.set_tools(tools)
        
        # Remove playwright tools
        search_engine.remove_tools('playwright')
        
        # Search for navigation (should not find playwright.navigate)
        results = await search_engine.search_tools('navigate to website', top_k=5)
        
        # Should only return jira tool (no playwright tools)
        assert len(results) == 1
        assert results[0].tool.upstream_id == 'jira'
        assert 'playwright' not in results[0].tool.name
