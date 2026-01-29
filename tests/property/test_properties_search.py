"""
Property-based tests for semantic search engine.

Tests Property 6: Query embeddings combine query and context
Tests Property 7: Semantic search ranks by cosine similarity
Tests Property 8: Search results include top 10 tools with scores
Tests Property 9: Empty or missing query produces error
Tests Property 10: Query sanitization removes special characters
Validates: Requirements 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10, 4.2, 4.4, 4.5, 4.7
"""
import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
import numpy as np

from mcp_router.core.models import ToolMetadata, JSONSchema
from mcp_router.embedding.engine import EmbeddingEngine
from mcp_router.search.engine import SemanticSearchEngine
from mcp_router.search.sanitize import sanitize_query, combine_query_and_context
from mcp_router.search.similarity import cosine_similarity


# Strategy for generating non-empty text
non_empty_text_strategy = st.text(min_size=1, max_size=200).filter(lambda x: x.strip())


@pytest.mark.property
@given(
    query=non_empty_text_strategy,
    context=st.one_of(
        st.none(),
        st.lists(non_empty_text_strategy, min_size=1, max_size=3)
    )
)
def test_property_6_query_embeddings_combine_query_and_context(
    query: str,
    context: list[str] | None
) -> None:
    """
    Property 6: Query embeddings combine query and context.
    
    For any query Q and context C, the combined text SHALL include both Q and C.
    
    Validates: Requirements 3.4, 3.5
    """
    combined = combine_query_and_context(query, context)
    
    # Verify query is included (after sanitization)
    sanitized_query = sanitize_query(query)
    assert sanitized_query in combined or not sanitized_query
    
    # Verify context is included (if provided)
    if context:
        for ctx in context:
            sanitized_ctx = sanitize_query(ctx)
            if sanitized_ctx:  # Only check non-empty sanitized context
                assert sanitized_ctx in combined


@pytest.mark.property
@given(
    vec_a=st.lists(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False), min_size=10, max_size=10),
    vec_b=st.lists(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False), min_size=10, max_size=10)
)
def test_property_7_cosine_similarity_properties(
    vec_a: list[float],
    vec_b: list[float]
) -> None:
    """
    Property 7: Cosine similarity has expected mathematical properties.
    
    For any vectors A and B:
    - Similarity is between -1 and 1 (with floating point tolerance)
    - Similarity of A with itself is 1 (if non-zero)
    - Similarity is symmetric: sim(A, B) == sim(B, A)
    
    Validates: Requirements 3.6, 4.3
    """
    arr_a = np.array(vec_a, dtype=np.float32)
    arr_b = np.array(vec_b, dtype=np.float32)
    
    # Compute similarity
    sim_ab = cosine_similarity(arr_a, arr_b)
    sim_ba = cosine_similarity(arr_b, arr_a)
    
    # Property: Similarity is between -1 and 1 (with floating point tolerance)
    assert -1.0 - 1e-6 <= sim_ab <= 1.0 + 1e-6
    assert -1.0 - 1e-6 <= sim_ba <= 1.0 + 1e-6
    
    # Property: Similarity is symmetric
    assert abs(sim_ab - sim_ba) < 1e-6
    
    # Property: Self-similarity is 1 (for non-zero vectors)
    if np.linalg.norm(arr_a) > 0:
        sim_aa = cosine_similarity(arr_a, arr_a)
        assert abs(sim_aa - 1.0) < 1e-6


@pytest.mark.property
@pytest.mark.asyncio
@given(
    num_tools=st.integers(min_value=15, max_value=30),
    top_k=st.integers(min_value=5, max_value=15)
)
@settings(
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
    max_examples=5
)
async def test_property_8_search_results_include_top_k_tools(
    num_tools: int,
    top_k: int
) -> None:
    """
    Property 8: Search results include top-k tools with scores.
    
    For any query and catalog of N tools, search SHALL return min(k, N) results
    ranked by descending similarity score.
    
    Validates: Requirements 3.8, 4.7
    """
    # Initialize embedding engine
    engine = EmbeddingEngine()
    await engine.initialize()
    
    # Create tools with embeddings
    tools = []
    for i in range(num_tools):
        tool = ToolMetadata(
            name=f"upstream{i % 3}.tool{i}",
            original_name=f"tool{i}",
            description=f"Test tool {i} for testing",
            input_schema=JSONSchema(type='object'),
            upstream_id=f"upstream{i % 3}"
        )
        tools.append(tool)
    
    # Generate embeddings
    await engine.generate_tool_embeddings(tools)
    
    # Create search engine
    search_engine = SemanticSearchEngine(engine)
    search_engine.set_tools(tools)
    
    # Perform search
    results = await search_engine.search_tools("test tool", top_k=top_k)
    
    # Verify result count
    expected_count = min(top_k, num_tools)
    assert len(results) == expected_count
    
    # Verify all results have scores
    for result in results:
        assert result.similarity is not None
        assert -1.0 <= result.similarity <= 1.0
    
    # Verify results are sorted by descending similarity
    for i in range(len(results) - 1):
        assert results[i].similarity >= results[i + 1].similarity


@pytest.mark.property
@pytest.mark.asyncio
@given(
    empty_query=st.one_of(
        st.just(""),
        st.just("   "),
        st.just("\t\n"),
    )
)
@settings(deadline=None, max_examples=5)
async def test_property_9_empty_query_produces_error(empty_query: str) -> None:
    """
    Property 9: Empty or missing query produces error.
    
    For any empty or whitespace-only query, search SHALL raise ValueError.
    
    Validates: Requirements 3.9
    """
    # Initialize engines
    engine = EmbeddingEngine()
    await engine.initialize()
    
    search_engine = SemanticSearchEngine(engine)
    
    # Create a dummy tool
    tool = ToolMetadata(
        name="test.tool",
        original_name="tool",
        description="Test",
        input_schema=JSONSchema(type='object'),
        upstream_id="test"
    )
    await engine.generate_tool_embeddings([tool])
    search_engine.set_tools([tool])
    
    # Verify empty query raises error
    with pytest.raises(ValueError, match="Query cannot be empty"):
        await search_engine.search_tools(empty_query)


@pytest.mark.property
@given(
    query=st.text(min_size=1, max_size=100)
)
def test_property_10_query_sanitization_removes_special_characters(query: str) -> None:
    """
    Property 10: Query sanitization removes special characters.
    
    For any query Q, sanitized query SHALL only contain alphanumeric characters,
    spaces, and basic punctuation (.,!?-).
    
    Validates: Requirements 3.10
    """
    sanitized = sanitize_query(query)
    
    # Verify only allowed characters remain
    # Allowed: alphanumeric, spaces, and .,!?-
    import re
    allowed_pattern = r'^[\w\s.,!?-]*$'
    assert re.match(allowed_pattern, sanitized), f"Sanitized query contains invalid characters: '{sanitized}'"
    
    # Verify no multiple consecutive spaces
    assert '  ' not in sanitized
