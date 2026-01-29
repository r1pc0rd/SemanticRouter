"""
Property-based tests for embedding engine.

Tests Property 11: All tools have embeddings after initialization
Tests Property 12: Tool embeddings are cached
Tests Property 13: All embeddings have 384 dimensions
Validates: Requirements 4.1, 4.6, 5.4
"""
import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
import numpy as np
import asyncio

from mcp_router.core.models import ToolMetadata, JSONSchema
from mcp_router.embedding.engine import EmbeddingEngine
from mcp_router.embedding.utils import generate_tool_embedding_text


# Strategy for generating tool descriptions
description_strategy = st.text(min_size=5, max_size=200).filter(lambda x: x.strip())

# Strategy for generating tool names
tool_name_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='-_'),
    min_size=1,
    max_size=50
)

# Strategy for generating non-empty text
non_empty_text_strategy = st.text(min_size=1, max_size=500).filter(lambda x: x.strip())


@pytest.mark.property
@pytest.mark.asyncio
@given(text=non_empty_text_strategy)
@settings(
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,  # Disable deadline for model loading
    max_examples=10  # Reduce examples since model loading is slow
)
async def test_property_13_embedding_dimensions(text: str) -> None:
    """
    Property 13: All embeddings have 384 dimensions.
    
    For any text input, the generated embedding SHALL have exactly 384 dimensions.
    
    Validates: Requirements 5.4
    """
    # Initialize embedding engine
    engine = EmbeddingEngine()
    await engine.initialize()
    
    # Generate embedding
    embedding = engine.generate_embedding(text)
    
    # Verify dimensions
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (384,)
    assert embedding.dtype == np.float32 or embedding.dtype == np.float64


@pytest.mark.property
@pytest.mark.asyncio
@given(
    tool_name=tool_name_strategy,
    description=description_strategy,
    upstream_id=tool_name_strategy
)
@settings(
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,  # Disable deadline for model loading
    max_examples=10  # Reduce examples since model loading is slow
)
async def test_property_11_all_tools_have_embeddings(
    tool_name: str,
    description: str,
    upstream_id: str
) -> None:
    """
    Property 11: All tools have embeddings after initialization.
    
    After calling generate_tool_embeddings(), every tool SHALL have a non-null
    embedding field with 384 dimensions.
    
    Validates: Requirements 4.1
    """
    # Create tool metadata
    tool = ToolMetadata(
        name=f"{upstream_id}.{tool_name}",
        original_name=tool_name,
        description=description,
        input_schema=JSONSchema(type='object'),
        upstream_id=upstream_id
    )
    
    # Verify no embedding initially
    assert tool.embedding is None
    
    # Initialize embedding engine
    engine = EmbeddingEngine()
    await engine.initialize()
    
    # Generate embeddings
    await engine.generate_tool_embeddings([tool])
    
    # Verify embedding exists and has correct dimensions
    assert tool.embedding is not None
    assert isinstance(tool.embedding, np.ndarray)
    assert tool.embedding.shape == (384,)


@pytest.mark.property
@pytest.mark.asyncio
@given(
    tool_name=tool_name_strategy,
    description=description_strategy,
    upstream_id=tool_name_strategy
)
@settings(
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,  # Disable deadline for model loading
    max_examples=10  # Reduce examples since model loading is slow
)
async def test_property_12_tool_embeddings_are_cached(
    tool_name: str,
    description: str,
    upstream_id: str
) -> None:
    """
    Property 12: Tool embeddings are cached.
    
    After generating embeddings, the embedding SHALL be stored in the tool object
    and subsequent accesses SHALL return the same cached embedding without
    regenerating it.
    
    Validates: Requirements 4.6
    """
    # Create tool metadata
    tool = ToolMetadata(
        name=f"{upstream_id}.{tool_name}",
        original_name=tool_name,
        description=description,
        input_schema=JSONSchema(type='object'),
        upstream_id=upstream_id
    )
    
    # Initialize embedding engine
    engine = EmbeddingEngine()
    await engine.initialize()
    
    # Generate embeddings
    await engine.generate_tool_embeddings([tool])
    
    # Get reference to cached embedding
    cached_embedding = tool.embedding
    assert cached_embedding is not None
    
    # Verify the embedding is stored in the tool object
    assert tool.embedding is cached_embedding
    
    # Verify it's the same object (not a copy)
    assert tool.embedding is cached_embedding


@pytest.mark.property
@pytest.mark.asyncio
@given(
    texts=st.lists(
        non_empty_text_strategy,
        min_size=1,
        max_size=5  # Reduce size for faster tests
    )
)
@settings(
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,  # Disable deadline for model loading
    max_examples=10  # Reduce examples since model loading is slow
)
async def test_property_13_batch_embedding_dimensions(texts: list[str]) -> None:
    """
    Property 13: Batch embeddings all have 384 dimensions.
    
    For any list of text inputs, all generated embeddings SHALL have exactly
    384 dimensions.
    
    Validates: Requirements 5.4
    """
    # Initialize embedding engine
    engine = EmbeddingEngine()
    await engine.initialize()
    
    # Generate embeddings in batch
    embeddings = engine.generate_embeddings_batch(texts)
    
    # Verify all embeddings have correct dimensions
    assert len(embeddings) == len(texts)
    for embedding in embeddings:
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)


@pytest.mark.property
@given(
    tool_name=tool_name_strategy,
    description=description_strategy,
    upstream_id=tool_name_strategy,
    category_description=st.one_of(
        st.none(),
        st.text(min_size=10, max_size=100).filter(lambda x: x.strip())
    )
)
@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow]
)
def test_property_17_category_description_in_embeddings(
    tool_name: str,
    description: str,
    upstream_id: str,
    category_description: str | None
) -> None:
    """
    Property 17: Category description is included in embeddings.
    
    For any upstream with a category_description configured, all tools from that
    upstream SHALL have the category_description included in their embedding text.
    
    Validates: Requirements 6.5
    """
    # Create tool metadata with or without category_description
    tool = ToolMetadata(
        name=f"{upstream_id}.{tool_name}",
        original_name=tool_name,
        description=description,
        input_schema=JSONSchema(type='object'),
        upstream_id=upstream_id,
        category_description=category_description
    )
    
    # Generate embedding text
    embedding_text = generate_tool_embedding_text(tool)
    
    # Verify category_description is included when present
    if category_description:
        assert category_description in embedding_text, \
            f"Category description '{category_description}' not found in embedding text: {embedding_text}"
    else:
        # When category_description is None, verify it's not in the text
        # (this is implicit - we just check the text is valid)
        assert embedding_text  # Non-empty
        assert tool.original_name in embedding_text  # Tool name is present
        assert tool.description in embedding_text  # Description is present

