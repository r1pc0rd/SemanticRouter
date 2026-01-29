"""
Unit tests for semantic search engine.

Tests search functionality, query sanitization, and similarity computation.
"""
import pytest
import numpy as np

from mcp_router.core.models import ToolMetadata, JSONSchema
from mcp_router.embedding.engine import EmbeddingEngine
from mcp_router.search.engine import SemanticSearchEngine
from mcp_router.search.sanitize import sanitize_query, combine_query_and_context
from mcp_router.search.similarity import cosine_similarity, compute_similarities


class TestQuerySanitization:
    """Tests for query sanitization functions"""
    
    def test_sanitize_basic_query(self):
        """Test sanitization of basic query"""
        result = sanitize_query("test web page")
        assert result == "test web page"
    
    def test_sanitize_special_characters(self):
        """Test removal of special characters"""
        result = sanitize_query("test@#$%web^&*page")
        assert result == "test web page"
    
    def test_sanitize_multiple_spaces(self):
        """Test collapsing multiple spaces"""
        result = sanitize_query("test   web    page")
        assert result == "test web page"
    
    def test_sanitize_keeps_punctuation(self):
        """Test that basic punctuation is kept"""
        result = sanitize_query("test! web? page.")
        assert result == "test! web? page."
    
    def test_sanitize_empty_string(self):
        """Test sanitization of empty string"""
        result = sanitize_query("")
        assert result == ""
    
    def test_sanitize_whitespace_only(self):
        """Test sanitization of whitespace-only string"""
        result = sanitize_query("   \t\n  ")
        assert result == ""
    
    def test_combine_query_without_context(self):
        """Test combining query without context"""
        result = combine_query_and_context("test query")
        assert result == "test query"
    
    def test_combine_query_with_context(self):
        """Test combining query with context"""
        result = combine_query_and_context("test", ["context1", "context2"])
        assert result == "test context1 context2"
    
    def test_combine_query_with_empty_context(self):
        """Test combining query with empty context list"""
        result = combine_query_and_context("test", [])
        assert result == "test"
    
    def test_combine_sanitizes_both(self):
        """Test that both query and context are sanitized"""
        result = combine_query_and_context("test@#$", ["context@#$"])
        assert result == "test context"


class TestCosineSimilarity:
    """Tests for cosine similarity computation"""
    
    def test_identical_vectors(self):
        """Test similarity of identical vectors"""
        vec = np.array([1.0, 2.0, 3.0])
        similarity = cosine_similarity(vec, vec)
        assert abs(similarity - 1.0) < 1e-6
    
    def test_orthogonal_vectors(self):
        """Test similarity of orthogonal vectors"""
        vec_a = np.array([1.0, 0.0, 0.0])
        vec_b = np.array([0.0, 1.0, 0.0])
        similarity = cosine_similarity(vec_a, vec_b)
        assert abs(similarity - 0.0) < 1e-6
    
    def test_opposite_vectors(self):
        """Test similarity of opposite vectors"""
        vec_a = np.array([1.0, 0.0, 0.0])
        vec_b = np.array([-1.0, 0.0, 0.0])
        similarity = cosine_similarity(vec_a, vec_b)
        assert abs(similarity - (-1.0)) < 1e-6
    
    def test_zero_vector(self):
        """Test similarity with zero vector"""
        vec_a = np.array([1.0, 2.0, 3.0])
        vec_b = np.array([0.0, 0.0, 0.0])
        similarity = cosine_similarity(vec_a, vec_b)
        assert similarity == 0.0
    
    def test_different_dimensions_raises_error(self):
        """Test that different dimensions raise error"""
        vec_a = np.array([1.0, 2.0])
        vec_b = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError, match="same dimensionality"):
            cosine_similarity(vec_a, vec_b)
    
    def test_compute_similarities_batch(self):
        """Test computing similarities for multiple vectors"""
        query = np.array([1.0, 0.0, 0.0])
        tools = [
            np.array([1.0, 0.0, 0.0]),  # Same direction
            np.array([0.0, 1.0, 0.0]),  # Orthogonal
            np.array([-1.0, 0.0, 0.0]), # Opposite
        ]
        
        similarities = compute_similarities(query, tools)
        
        assert len(similarities) == 3
        assert abs(similarities[0] - 1.0) < 1e-6
        assert abs(similarities[1] - 0.0) < 1e-6
        assert abs(similarities[2] - (-1.0)) < 1e-6


class TestSemanticSearchEngine:
    """Tests for SemanticSearchEngine class"""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test search engine initialization"""
        engine = EmbeddingEngine()
        await engine.initialize()
        
        search_engine = SemanticSearchEngine(engine)
        assert search_engine.get_tool_count() == 0
    
    @pytest.mark.asyncio
    async def test_set_tools(self):
        """Test setting tools in search engine"""
        engine = EmbeddingEngine()
        await engine.initialize()
        
        # Create tools with embeddings
        tools = [
            ToolMetadata(
                name="test.tool1",
                original_name="tool1",
                description="Test tool 1",
                input_schema=JSONSchema(type='object'),
                upstream_id="test"
            ),
            ToolMetadata(
                name="test.tool2",
                original_name="tool2",
                description="Test tool 2",
                input_schema=JSONSchema(type='object'),
                upstream_id="test"
            )
        ]
        
        await engine.generate_tool_embeddings(tools)
        
        search_engine = SemanticSearchEngine(engine)
        search_engine.set_tools(tools)
        
        assert search_engine.get_tool_count() == 2
    
    @pytest.mark.asyncio
    async def test_set_tools_without_embeddings_raises_error(self):
        """Test that setting tools without embeddings raises error"""
        engine = EmbeddingEngine()
        await engine.initialize()
        
        # Create tool without embedding
        tool = ToolMetadata(
            name="test.tool",
            original_name="tool",
            description="Test",
            input_schema=JSONSchema(type='object'),
            upstream_id="test"
        )
        
        search_engine = SemanticSearchEngine(engine)
        
        with pytest.raises(ValueError, match="missing embedding"):
            search_engine.set_tools([tool])
    
    @pytest.mark.asyncio
    async def test_search_basic(self):
        """Test basic search functionality"""
        engine = EmbeddingEngine()
        await engine.initialize()
        
        # Create tools with different descriptions
        tools = [
            ToolMetadata(
                name="browser.navigate",
                original_name="navigate",
                description="Navigate to a URL in the browser",
                input_schema=JSONSchema(type='object'),
                upstream_id="browser"
            ),
            ToolMetadata(
                name="browser.click",
                original_name="click",
                description="Click on an element",
                input_schema=JSONSchema(type='object'),
                upstream_id="browser"
            ),
            ToolMetadata(
                name="file.read",
                original_name="read",
                description="Read a file from disk",
                input_schema=JSONSchema(type='object'),
                upstream_id="file"
            )
        ]
        
        await engine.generate_tool_embeddings(tools)
        
        search_engine = SemanticSearchEngine(engine)
        search_engine.set_tools(tools)
        
        # Search for browser-related tools
        results = await search_engine.search_tools("navigate to website")
        
        assert len(results) <= 10
        assert all(r.similarity is not None for r in results)
        # First result should be navigate (most relevant)
        assert results[0].tool.name == "browser.navigate"
    
    @pytest.mark.asyncio
    async def test_search_with_context(self):
        """Test search with context"""
        engine = EmbeddingEngine()
        await engine.initialize()
        
        tool = ToolMetadata(
            name="test.tool",
            original_name="tool",
            description="Test tool",
            input_schema=JSONSchema(type='object'),
            upstream_id="test"
        )
        
        await engine.generate_tool_embeddings([tool])
        
        search_engine = SemanticSearchEngine(engine)
        search_engine.set_tools([tool])
        
        # Search with context
        results = await search_engine.search_tools(
            "test",
            context=["additional context", "more info"]
        )
        
        assert len(results) == 1
        assert results[0].tool.name == "test.tool"
    
    @pytest.mark.asyncio
    async def test_search_empty_query_raises_error(self):
        """Test that empty query raises error"""
        engine = EmbeddingEngine()
        await engine.initialize()
        
        tool = ToolMetadata(
            name="test.tool",
            original_name="tool",
            description="Test",
            input_schema=JSONSchema(type='object'),
            upstream_id="test"
        )
        
        await engine.generate_tool_embeddings([tool])
        
        search_engine = SemanticSearchEngine(engine)
        search_engine.set_tools([tool])
        
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await search_engine.search_tools("")
    
    @pytest.mark.asyncio
    async def test_search_without_tools_raises_error(self):
        """Test that searching without tools raises error"""
        engine = EmbeddingEngine()
        await engine.initialize()
        
        search_engine = SemanticSearchEngine(engine)
        
        with pytest.raises(RuntimeError, match="No tools loaded"):
            await search_engine.search_tools("test query")
    
    @pytest.mark.asyncio
    async def test_search_top_k(self):
        """Test that search respects top_k parameter"""
        engine = EmbeddingEngine()
        await engine.initialize()
        
        # Create 20 tools
        tools = [
            ToolMetadata(
                name=f"test.tool{i}",
                original_name=f"tool{i}",
                description=f"Test tool {i}",
                input_schema=JSONSchema(type='object'),
                upstream_id="test"
            )
            for i in range(20)
        ]
        
        await engine.generate_tool_embeddings(tools)
        
        search_engine = SemanticSearchEngine(engine)
        search_engine.set_tools(tools)
        
        # Search with top_k=5
        results = await search_engine.search_tools("test", top_k=5)
        
        assert len(results) == 5
    
    @pytest.mark.asyncio
    async def test_search_results_sorted(self):
        """Test that search results are sorted by similarity"""
        engine = EmbeddingEngine()
        await engine.initialize()
        
        tools = [
            ToolMetadata(
                name=f"test.tool{i}",
                original_name=f"tool{i}",
                description=f"Test tool {i}",
                input_schema=JSONSchema(type='object'),
                upstream_id="test"
            )
            for i in range(10)
        ]
        
        await engine.generate_tool_embeddings(tools)
        
        search_engine = SemanticSearchEngine(engine)
        search_engine.set_tools(tools)
        
        results = await search_engine.search_tools("test")
        
        # Verify results are sorted in descending order
        for i in range(len(results) - 1):
            assert results[i].similarity >= results[i + 1].similarity


class TestAddTools:
    """Tests for add_tools() method"""
    
    @pytest.mark.asyncio
    async def test_add_tools_to_empty_catalog(self):
        """Test adding tools to empty catalog"""
        engine = EmbeddingEngine()
        await engine.initialize()
        
        tools = [
            ToolMetadata(
                name="test.tool1",
                original_name="tool1",
                description="Test tool 1",
                input_schema=JSONSchema(type='object'),
                upstream_id="test"
            ),
            ToolMetadata(
                name="test.tool2",
                original_name="tool2",
                description="Test tool 2",
                input_schema=JSONSchema(type='object'),
                upstream_id="test"
            )
        ]
        
        await engine.generate_tool_embeddings(tools)
        
        search_engine = SemanticSearchEngine(engine)
        assert search_engine.get_tool_count() == 0
        
        search_engine.add_tools(tools)
        assert search_engine.get_tool_count() == 2
    
    @pytest.mark.asyncio
    async def test_add_tools_to_existing_catalog(self):
        """Test adding tools to existing catalog"""
        engine = EmbeddingEngine()
        await engine.initialize()
        
        # Initial tools
        initial_tools = [
            ToolMetadata(
                name="browser.navigate",
                original_name="navigate",
                description="Navigate to URL",
                input_schema=JSONSchema(type='object'),
                upstream_id="browser"
            )
        ]
        
        # New tools to add
        new_tools = [
            ToolMetadata(
                name="file.read",
                original_name="read",
                description="Read file",
                input_schema=JSONSchema(type='object'),
                upstream_id="file"
            ),
            ToolMetadata(
                name="file.write",
                original_name="write",
                description="Write file",
                input_schema=JSONSchema(type='object'),
                upstream_id="file"
            )
        ]
        
        await engine.generate_tool_embeddings(initial_tools + new_tools)
        
        search_engine = SemanticSearchEngine(engine)
        search_engine.set_tools(initial_tools)
        assert search_engine.get_tool_count() == 1
        
        search_engine.add_tools(new_tools)
        assert search_engine.get_tool_count() == 3
    
    @pytest.mark.asyncio
    async def test_add_tools_without_embeddings_raises_error(self):
        """Test that adding tools without embeddings raises error"""
        engine = EmbeddingEngine()
        await engine.initialize()
        
        # Tool without embedding
        tool = ToolMetadata(
            name="test.tool",
            original_name="tool",
            description="Test",
            input_schema=JSONSchema(type='object'),
            upstream_id="test"
        )
        
        search_engine = SemanticSearchEngine(engine)
        
        with pytest.raises(ValueError, match="missing embedding"):
            search_engine.add_tools([tool])
    
    @pytest.mark.asyncio
    async def test_add_tools_with_duplicate_name_raises_error(self):
        """Test that adding duplicate tool names raises error"""
        engine = EmbeddingEngine()
        await engine.initialize()
        
        # Initial tool
        initial_tool = ToolMetadata(
            name="test.tool1",
            original_name="tool1",
            description="Test tool 1",
            input_schema=JSONSchema(type='object'),
            upstream_id="test"
        )
        
        # Duplicate tool (same name)
        duplicate_tool = ToolMetadata(
            name="test.tool1",
            original_name="tool1",
            description="Different description",
            input_schema=JSONSchema(type='object'),
            upstream_id="test"
        )
        
        await engine.generate_tool_embeddings([initial_tool, duplicate_tool])
        
        search_engine = SemanticSearchEngine(engine)
        search_engine.set_tools([initial_tool])
        
        with pytest.raises(ValueError, match="already exists"):
            search_engine.add_tools([duplicate_tool])
    
    @pytest.mark.asyncio
    async def test_add_empty_list_does_nothing(self):
        """Test that adding empty list does nothing"""
        engine = EmbeddingEngine()
        await engine.initialize()
        
        search_engine = SemanticSearchEngine(engine)
        assert search_engine.get_tool_count() == 0
        
        search_engine.add_tools([])
        assert search_engine.get_tool_count() == 0
    
    @pytest.mark.asyncio
    async def test_add_tools_preserves_search_functionality(self):
        """Test that added tools are searchable"""
        engine = EmbeddingEngine()
        await engine.initialize()
        
        # Initial tools
        initial_tools = [
            ToolMetadata(
                name="browser.navigate",
                original_name="navigate",
                description="Navigate to a URL in the browser",
                input_schema=JSONSchema(type='object'),
                upstream_id="browser"
            )
        ]
        
        # New tools
        new_tools = [
            ToolMetadata(
                name="file.read",
                original_name="read",
                description="Read a file from disk",
                input_schema=JSONSchema(type='object'),
                upstream_id="file"
            )
        ]
        
        await engine.generate_tool_embeddings(initial_tools + new_tools)
        
        search_engine = SemanticSearchEngine(engine)
        search_engine.set_tools(initial_tools)
        search_engine.add_tools(new_tools)
        
        # Search for file-related tools
        results = await search_engine.search_tools("read file")
        
        assert len(results) == 2
        # File tool should be most relevant
        assert results[0].tool.name == "file.read"
    
    @pytest.mark.asyncio
    async def test_add_tools_multiple_times(self):
        """Test adding tools in multiple batches"""
        engine = EmbeddingEngine()
        await engine.initialize()
        
        batch1 = [
            ToolMetadata(
                name="test.tool1",
                original_name="tool1",
                description="Test 1",
                input_schema=JSONSchema(type='object'),
                upstream_id="test"
            )
        ]
        
        batch2 = [
            ToolMetadata(
                name="test.tool2",
                original_name="tool2",
                description="Test 2",
                input_schema=JSONSchema(type='object'),
                upstream_id="test"
            )
        ]
        
        batch3 = [
            ToolMetadata(
                name="test.tool3",
                original_name="tool3",
                description="Test 3",
                input_schema=JSONSchema(type='object'),
                upstream_id="test"
            )
        ]
        
        await engine.generate_tool_embeddings(batch1 + batch2 + batch3)
        
        search_engine = SemanticSearchEngine(engine)
        
        search_engine.add_tools(batch1)
        assert search_engine.get_tool_count() == 1
        
        search_engine.add_tools(batch2)
        assert search_engine.get_tool_count() == 2
        
        search_engine.add_tools(batch3)
        assert search_engine.get_tool_count() == 3


class TestRemoveTools:
    """Tests for remove_tools() method"""
    
    @pytest.mark.asyncio
    async def test_remove_tools_by_namespace(self):
        """Test removing tools by namespace"""
        engine = EmbeddingEngine()
        await engine.initialize()
        
        tools = [
            ToolMetadata(
                name="browser.navigate",
                original_name="navigate",
                description="Navigate",
                input_schema=JSONSchema(type='object'),
                upstream_id="browser"
            ),
            ToolMetadata(
                name="browser.click",
                original_name="click",
                description="Click",
                input_schema=JSONSchema(type='object'),
                upstream_id="browser"
            ),
            ToolMetadata(
                name="file.read",
                original_name="read",
                description="Read",
                input_schema=JSONSchema(type='object'),
                upstream_id="file"
            )
        ]
        
        await engine.generate_tool_embeddings(tools)
        
        search_engine = SemanticSearchEngine(engine)
        search_engine.set_tools(tools)
        assert search_engine.get_tool_count() == 3
        
        # Remove browser namespace
        search_engine.remove_tools("browser")
        assert search_engine.get_tool_count() == 1
    
    @pytest.mark.asyncio
    async def test_remove_tools_nonexistent_namespace(self):
        """Test removing tools with nonexistent namespace does nothing"""
        engine = EmbeddingEngine()
        await engine.initialize()
        
        tools = [
            ToolMetadata(
                name="browser.navigate",
                original_name="navigate",
                description="Navigate",
                input_schema=JSONSchema(type='object'),
                upstream_id="browser"
            )
        ]
        
        await engine.generate_tool_embeddings(tools)
        
        search_engine = SemanticSearchEngine(engine)
        search_engine.set_tools(tools)
        
        # Remove nonexistent namespace
        search_engine.remove_tools("nonexistent")
        assert search_engine.get_tool_count() == 1
    
    @pytest.mark.asyncio
    async def test_remove_tools_empty_namespace_raises_error(self):
        """Test that empty namespace raises error"""
        engine = EmbeddingEngine()
        await engine.initialize()
        
        search_engine = SemanticSearchEngine(engine)
        
        with pytest.raises(ValueError, match="Namespace cannot be empty"):
            search_engine.remove_tools("")
    
    @pytest.mark.asyncio
    async def test_remove_tools_from_empty_catalog(self):
        """Test removing tools from empty catalog does nothing"""
        engine = EmbeddingEngine()
        await engine.initialize()
        
        search_engine = SemanticSearchEngine(engine)
        assert search_engine.get_tool_count() == 0
        
        search_engine.remove_tools("test")
        assert search_engine.get_tool_count() == 0
    
    @pytest.mark.asyncio
    async def test_remove_tools_preserves_other_namespaces(self):
        """Test that removing one namespace preserves others"""
        engine = EmbeddingEngine()
        await engine.initialize()
        
        tools = [
            ToolMetadata(
                name="browser.navigate",
                original_name="navigate",
                description="Navigate to URL",
                input_schema=JSONSchema(type='object'),
                upstream_id="browser"
            ),
            ToolMetadata(
                name="file.read",
                original_name="read",
                description="Read file",
                input_schema=JSONSchema(type='object'),
                upstream_id="file"
            ),
            ToolMetadata(
                name="file.write",
                original_name="write",
                description="Write file",
                input_schema=JSONSchema(type='object'),
                upstream_id="file"
            )
        ]
        
        await engine.generate_tool_embeddings(tools)
        
        search_engine = SemanticSearchEngine(engine)
        search_engine.set_tools(tools)
        
        # Remove browser namespace
        search_engine.remove_tools("browser")
        
        # File tools should remain
        assert search_engine.get_tool_count() == 2
        
        # Search should still work with remaining tools
        results = await search_engine.search_tools("file")
        assert len(results) == 2
        assert all("file." in r.tool.name for r in results)
    
    @pytest.mark.asyncio
    async def test_remove_all_tools(self):
        """Test removing all tools leaves empty catalog"""
        engine = EmbeddingEngine()
        await engine.initialize()
        
        tools = [
            ToolMetadata(
                name="test.tool1",
                original_name="tool1",
                description="Test 1",
                input_schema=JSONSchema(type='object'),
                upstream_id="test"
            ),
            ToolMetadata(
                name="test.tool2",
                original_name="tool2",
                description="Test 2",
                input_schema=JSONSchema(type='object'),
                upstream_id="test"
            )
        ]
        
        await engine.generate_tool_embeddings(tools)
        
        search_engine = SemanticSearchEngine(engine)
        search_engine.set_tools(tools)
        
        search_engine.remove_tools("test")
        assert search_engine.get_tool_count() == 0
    
    @pytest.mark.asyncio
    async def test_remove_tools_case_sensitive(self):
        """Test that namespace removal is case-sensitive"""
        engine = EmbeddingEngine()
        await engine.initialize()
        
        tools = [
            ToolMetadata(
                name="Browser.navigate",
                original_name="navigate",
                description="Navigate",
                input_schema=JSONSchema(type='object'),
                upstream_id="Browser"
            )
        ]
        
        await engine.generate_tool_embeddings(tools)
        
        search_engine = SemanticSearchEngine(engine)
        search_engine.set_tools(tools)
        
        # Try to remove with lowercase (should not match)
        search_engine.remove_tools("browser")
        assert search_engine.get_tool_count() == 1
        
        # Remove with correct case
        search_engine.remove_tools("Browser")
        assert search_engine.get_tool_count() == 0


class TestAddAndRemoveIntegration:
    """Integration tests for add_tools() and remove_tools() together"""
    
    @pytest.mark.asyncio
    async def test_add_then_remove(self):
        """Test adding tools then removing them"""
        engine = EmbeddingEngine()
        await engine.initialize()
        
        tools = [
            ToolMetadata(
                name="test.tool1",
                original_name="tool1",
                description="Test 1",
                input_schema=JSONSchema(type='object'),
                upstream_id="test"
            )
        ]
        
        await engine.generate_tool_embeddings(tools)
        
        search_engine = SemanticSearchEngine(engine)
        
        # Add tools
        search_engine.add_tools(tools)
        assert search_engine.get_tool_count() == 1
        
        # Remove tools
        search_engine.remove_tools("test")
        assert search_engine.get_tool_count() == 0
    
    @pytest.mark.asyncio
    async def test_remove_then_add_same_namespace(self):
        """Test removing then re-adding tools with same namespace"""
        engine = EmbeddingEngine()
        await engine.initialize()
        
        tools_v1 = [
            ToolMetadata(
                name="test.tool1",
                original_name="tool1",
                description="Version 1",
                input_schema=JSONSchema(type='object'),
                upstream_id="test"
            )
        ]
        
        tools_v2 = [
            ToolMetadata(
                name="test.tool2",
                original_name="tool2",
                description="Version 2",
                input_schema=JSONSchema(type='object'),
                upstream_id="test"
            )
        ]
        
        await engine.generate_tool_embeddings(tools_v1 + tools_v2)
        
        search_engine = SemanticSearchEngine(engine)
        
        # Add v1
        search_engine.add_tools(tools_v1)
        assert search_engine.get_tool_count() == 1
        
        # Remove test namespace
        search_engine.remove_tools("test")
        assert search_engine.get_tool_count() == 0
        
        # Add v2 (same namespace, different tools)
        search_engine.add_tools(tools_v2)
        assert search_engine.get_tool_count() == 1
    
    @pytest.mark.asyncio
    async def test_dynamic_catalog_management(self):
        """Test dynamic loading/unloading scenario"""
        engine = EmbeddingEngine()
        await engine.initialize()
        
        # Create tools for multiple upstreams
        browser_tools = [
            ToolMetadata(
                name="browser.navigate",
                original_name="navigate",
                description="Navigate",
                input_schema=JSONSchema(type='object'),
                upstream_id="browser"
            )
        ]
        
        file_tools = [
            ToolMetadata(
                name="file.read",
                original_name="read",
                description="Read",
                input_schema=JSONSchema(type='object'),
                upstream_id="file"
            )
        ]
        
        db_tools = [
            ToolMetadata(
                name="db.query",
                original_name="query",
                description="Query",
                input_schema=JSONSchema(type='object'),
                upstream_id="db"
            )
        ]
        
        await engine.generate_tool_embeddings(browser_tools + file_tools + db_tools)
        
        search_engine = SemanticSearchEngine(engine)
        
        # Start with browser tools
        search_engine.add_tools(browser_tools)
        assert search_engine.get_tool_count() == 1
        
        # Add file tools
        search_engine.add_tools(file_tools)
        assert search_engine.get_tool_count() == 2
        
        # Add db tools
        search_engine.add_tools(db_tools)
        assert search_engine.get_tool_count() == 3
        
        # Remove browser tools
        search_engine.remove_tools("browser")
        assert search_engine.get_tool_count() == 2
        
        # Remove file tools
        search_engine.remove_tools("file")
        assert search_engine.get_tool_count() == 1
        
        # Only db tools remain
        results = await search_engine.search_tools("query")
        assert len(results) == 1
        assert results[0].tool.name == "db.query"
