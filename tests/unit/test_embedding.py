"""
Unit tests for embedding engine.

Tests embedding generation, model initialization, and error handling.
"""
import pytest
import numpy as np

from mcp_router.core.models import ToolMetadata, JSONSchema
from mcp_router.embedding.engine import EmbeddingEngine
from mcp_router.embedding.utils import generate_tool_embedding_text


class TestEmbeddingEngine:
    """Tests for EmbeddingEngine class"""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test that engine initializes successfully"""
        engine = EmbeddingEngine()
        assert not engine.is_initialized
        
        await engine.initialize()
        assert engine.is_initialized
    
    @pytest.mark.asyncio
    async def test_generate_embedding_success(self):
        """Test successful embedding generation"""
        engine = EmbeddingEngine()
        await engine.initialize()
        
        text = "Navigate to a URL"
        embedding = engine.generate_embedding(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
    
    @pytest.mark.asyncio
    async def test_generate_embedding_not_initialized(self):
        """Test that generating embedding before initialization raises error"""
        engine = EmbeddingEngine()
        
        with pytest.raises(RuntimeError, match="not initialized"):
            engine.generate_embedding("test")
    
    @pytest.mark.asyncio
    async def test_generate_embedding_empty_text(self):
        """Test that empty text raises ValueError"""
        engine = EmbeddingEngine()
        await engine.initialize()
        
        with pytest.raises(ValueError, match="empty text"):
            engine.generate_embedding("")
        
        with pytest.raises(ValueError, match="empty text"):
            engine.generate_embedding("   ")
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_batch(self):
        """Test batch embedding generation"""
        engine = EmbeddingEngine()
        await engine.initialize()
        
        texts = [
            "Navigate to a URL",
            "Click on an element",
            "Take a screenshot"
        ]
        
        embeddings = engine.generate_embeddings_batch(texts)
        
        assert len(embeddings) == 3
        for embedding in embeddings:
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (384,)
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_empty_list(self):
        """Test batch generation with empty list"""
        engine = EmbeddingEngine()
        await engine.initialize()
        
        embeddings = engine.generate_embeddings_batch([])
        assert embeddings == []
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_with_empty_text(self):
        """Test batch generation with empty text raises error"""
        engine = EmbeddingEngine()
        await engine.initialize()
        
        texts = ["valid text", "", "another valid text"]
        
        with pytest.raises(ValueError, match="empty text"):
            engine.generate_embeddings_batch(texts)
    
    @pytest.mark.asyncio
    async def test_generate_tool_embeddings(self):
        """Test generating embeddings for tools"""
        engine = EmbeddingEngine()
        await engine.initialize()
        
        tools = [
            ToolMetadata(
                name="browser.navigate",
                original_name="navigate",
                description="Navigate to a URL",
                input_schema=JSONSchema(type='object', properties={'url': {'type': 'string'}}),
                upstream_id="playwright"
            ),
            ToolMetadata(
                name="browser.click",
                original_name="click",
                description="Click on an element",
                input_schema=JSONSchema(type='object'),
                upstream_id="playwright"
            )
        ]
        
        # Verify no embeddings initially
        for tool in tools:
            assert tool.embedding is None
        
        # Generate embeddings
        await engine.generate_tool_embeddings(tools)
        
        # Verify embeddings exist
        for tool in tools:
            assert tool.embedding is not None
            assert isinstance(tool.embedding, np.ndarray)
            assert tool.embedding.shape == (384,)
    
    @pytest.mark.asyncio
    async def test_generate_tool_embeddings_empty_list(self):
        """Test generating embeddings for empty tool list"""
        engine = EmbeddingEngine()
        await engine.initialize()
        
        # Should not raise error
        await engine.generate_tool_embeddings([])
    
    @pytest.mark.asyncio
    async def test_embeddings_are_deterministic(self):
        """Test that same text produces same embedding"""
        engine = EmbeddingEngine()
        await engine.initialize()
        
        text = "Navigate to a URL"
        embedding1 = engine.generate_embedding(text)
        embedding2 = engine.generate_embedding(text)
        
        # Should be very close (allowing for floating point precision)
        np.testing.assert_array_almost_equal(embedding1, embedding2, decimal=5)


class TestGenerateToolEmbeddingText:
    """Tests for generate_tool_embedding_text utility function"""
    
    def test_basic_tool(self):
        """Test embedding text generation for basic tool"""
        tool = ToolMetadata(
            name="browser.navigate",
            original_name="navigate",
            description="Navigate to a URL",
            input_schema=JSONSchema(type='object'),
            upstream_id="playwright"
        )
        
        text = generate_tool_embedding_text(tool)
        assert "navigate" in text
        assert "Navigate to a URL" in text
    
    def test_tool_with_parameters(self):
        """Test embedding text includes parameter names"""
        tool = ToolMetadata(
            name="browser.navigate",
            original_name="navigate",
            description="Navigate to a URL",
            input_schema=JSONSchema(
                type='object',
                properties={
                    'url': {'type': 'string'},
                    'timeout': {'type': 'number'}
                }
            ),
            upstream_id="playwright"
        )
        
        text = generate_tool_embedding_text(tool)
        assert "Parameters:" in text
        assert "url" in text
        assert "timeout" in text
    
    def test_tool_with_category_description(self):
        """Test embedding text includes category description"""
        tool = ToolMetadata(
            name="browser.navigate",
            original_name="navigate",
            description="Navigate to a URL",
            input_schema=JSONSchema(type='object'),
            upstream_id="playwright",
            category_description="Web browser automation tools"
        )
        
        text = generate_tool_embedding_text(tool)
        assert "Web browser automation tools" in text
    
    def test_tool_with_all_fields(self):
        """Test embedding text with all fields populated"""
        tool = ToolMetadata(
            name="browser.navigate",
            original_name="navigate",
            description="Navigate to a URL",
            input_schema=JSONSchema(
                type='object',
                properties={'url': {'type': 'string'}}
            ),
            upstream_id="playwright",
            category_description="Web browser automation"
        )
        
        text = generate_tool_embedding_text(tool)
        
        # Verify all parts are included
        assert "navigate" in text
        assert "Navigate to a URL" in text
        assert "Web browser automation" in text
        assert "Parameters: url" in text
        
        # Verify separator is used
        assert "|" in text
    
    def test_tool_without_parameters(self):
        """Test embedding text when tool has no parameters"""
        tool = ToolMetadata(
            name="browser.refresh",
            original_name="refresh",
            description="Refresh the page",
            input_schema=JSONSchema(type='object'),
            upstream_id="playwright"
        )
        
        text = generate_tool_embedding_text(tool)
        assert "Parameters:" not in text
    
    def test_tool_without_category(self):
        """Test embedding text when tool has no category description"""
        tool = ToolMetadata(
            name="browser.navigate",
            original_name="navigate",
            description="Navigate to a URL",
            input_schema=JSONSchema(type='object'),
            upstream_id="playwright"
        )
        
        text = generate_tool_embedding_text(tool)
        # Should still work, just without category
        assert "navigate" in text
        assert "Navigate to a URL" in text
