"""
Property-based tests for default tool subset selection.

Tests Property 22: Default tool subset provides upstream diversity
Tests Property 23: Non-empty catalog produces non-empty tool list
Validates: Requirements 8.1, 8.2, 8.3, 8.4
"""
import pytest
from hypothesis import given, strategies as st, settings

from mcp_router.core.models import ToolMetadata, JSONSchema
from mcp_router.search.default_subset import select_default_tool_subset


# Strategy for generating tool names
tool_name_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='-_'),
    min_size=1,
    max_size=30
)


@pytest.mark.property
@given(
    num_upstreams=st.integers(min_value=2, max_value=10),
    tools_per_upstream=st.integers(min_value=5, max_value=30)
)
def test_property_22_default_subset_provides_upstream_diversity(
    num_upstreams: int,
    tools_per_upstream: int
) -> None:
    """
    Property 22: Default tool subset provides upstream diversity.
    
    For any catalog with N upstreams, the default subset SHALL include
    tools from multiple upstreams (not dominated by a single upstream).
    
    Validates: Requirements 8.2, 8.3
    """
    # Create tools from multiple upstreams
    tools = []
    for upstream_idx in range(num_upstreams):
        upstream_id = f"upstream{upstream_idx}"
        for tool_idx in range(tools_per_upstream):
            tool = ToolMetadata(
                name=f"{upstream_id}.tool{tool_idx}",
                original_name=f"tool{tool_idx}",
                description=f"Test tool {tool_idx}",
                input_schema=JSONSchema(type='object'),
                upstream_id=upstream_id
            )
            tools.append(tool)
    
    # Select default subset
    subset = select_default_tool_subset(tools, max_tools=20)
    
    # Verify subset size
    assert len(subset) <= 20
    assert len(subset) > 0
    
    # Count tools from each upstream in the subset
    upstream_counts = {}
    for tool in subset:
        upstream_counts[tool.upstream_id] = upstream_counts.get(tool.upstream_id, 0) + 1
    
    # Property: Multiple upstreams should be represented
    # (unless there's only 1 upstream or very few tools)
    if num_upstreams > 1 and len(subset) >= 2:
        # At least 2 upstreams should be represented
        assert len(upstream_counts) >= min(2, num_upstreams)
    
    # Property: No single upstream should dominate excessively
    # (unless there's only 1 upstream)
    if num_upstreams > 1:
        max_count = max(upstream_counts.values())
        # No upstream should have more than 80% of the subset
        # (allowing some flexibility for small subsets)
        assert max_count <= len(subset) * 0.8 + 1


@pytest.mark.property
@given(
    num_tools=st.integers(min_value=1, max_value=100)
)
def test_property_23_non_empty_catalog_produces_non_empty_list(
    num_tools: int
) -> None:
    """
    Property 23: Non-empty catalog produces non-empty tool list.
    
    For any non-empty tool catalog, the default subset SHALL be non-empty.
    
    Validates: Requirements 8.4
    """
    # Create tools
    tools = [
        ToolMetadata(
            name=f"upstream.tool{i}",
            original_name=f"tool{i}",
            description=f"Test tool {i}",
            input_schema=JSONSchema(type='object'),
            upstream_id="upstream"
        )
        for i in range(num_tools)
    ]
    
    # Select default subset
    subset = select_default_tool_subset(tools, max_tools=20)
    
    # Property: Non-empty input produces non-empty output
    assert len(subset) > 0
    
    # Property: Subset size is min(num_tools, 20)
    assert len(subset) == min(num_tools, 20)


@pytest.mark.property
def test_property_23_empty_catalog_produces_empty_list() -> None:
    """
    Property 23: Empty catalog produces empty tool list.
    
    For an empty tool catalog, the default subset SHALL be empty.
    
    Validates: Requirements 8.4
    """
    # Select from empty catalog
    subset = select_default_tool_subset([], max_tools=20)
    
    # Property: Empty input produces empty output
    assert len(subset) == 0


@pytest.mark.property
@given(
    num_upstreams=st.integers(min_value=1, max_value=5),
    tools_per_upstream=st.integers(min_value=1, max_value=10)
)
def test_property_22_alphabetical_selection_within_upstream(
    num_upstreams: int,
    tools_per_upstream: int
) -> None:
    """
    Property 22: Tools are selected alphabetically within each upstream.
    
    For any upstream, tools SHALL be selected in alphabetical order by
    original name.
    
    Validates: Requirements 8.3
    """
    # Create tools with specific names to test alphabetical ordering
    tools = []
    for upstream_idx in range(num_upstreams):
        upstream_id = f"upstream{upstream_idx}"
        # Create tools with names that will sort alphabetically
        tool_names = [f"tool_{chr(97 + i)}" for i in range(tools_per_upstream)]
        for tool_name in tool_names:
            tool = ToolMetadata(
                name=f"{upstream_id}.{tool_name}",
                original_name=tool_name,
                description=f"Test {tool_name}",
                input_schema=JSONSchema(type='object'),
                upstream_id=upstream_id
            )
            tools.append(tool)
    
    # Select default subset
    subset = select_default_tool_subset(tools, max_tools=20)
    
    # Group subset by upstream
    subset_by_upstream = {}
    for tool in subset:
        if tool.upstream_id not in subset_by_upstream:
            subset_by_upstream[tool.upstream_id] = []
        subset_by_upstream[tool.upstream_id].append(tool)
    
    # Verify alphabetical order within each upstream
    for upstream_id, upstream_tools in subset_by_upstream.items():
        tool_names = [t.original_name for t in upstream_tools]
        # Verify tools are in alphabetical order
        assert tool_names == sorted(tool_names)


@pytest.mark.property
@given(
    max_tools=st.integers(min_value=1, max_value=50)
)
def test_property_22_respects_max_tools_limit(max_tools: int) -> None:
    """
    Property 22: Default subset respects max_tools limit.
    
    For any max_tools value, the subset SHALL contain at most max_tools tools.
    
    Validates: Requirements 8.1
    """
    # Create many tools
    tools = [
        ToolMetadata(
            name=f"upstream{i % 3}.tool{i}",
            original_name=f"tool{i}",
            description=f"Test tool {i}",
            input_schema=JSONSchema(type='object'),
            upstream_id=f"upstream{i % 3}"
        )
        for i in range(100)
    ]
    
    # Select subset with custom max_tools
    subset = select_default_tool_subset(tools, max_tools=max_tools)
    
    # Property: Subset size <= max_tools
    assert len(subset) <= max_tools
