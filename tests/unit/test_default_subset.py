"""
Unit tests for default tool subset selection.

Tests specific scenarios and edge cases for the default subset algorithm.
"""
import pytest

from mcp_router.core.models import ToolMetadata, JSONSchema
from mcp_router.search.default_subset import select_default_tool_subset


def test_select_default_subset_with_multiple_upstreams():
    """Test default subset selection with multiple upstreams."""
    # Create tools from 3 upstreams
    tools = []
    for upstream_idx in range(3):
        upstream_id = f"upstream{upstream_idx}"
        for tool_idx in range(10):
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
    
    # Should return 20 tools
    assert len(subset) == 20
    
    # Should have tools from all 3 upstreams
    upstream_ids = {tool.upstream_id for tool in subset}
    assert len(upstream_ids) == 3
    
    # Each upstream should have approximately equal representation
    # With 20 tools and 3 upstreams, expect 6-7 tools per upstream
    for upstream_id in upstream_ids:
        count = sum(1 for tool in subset if tool.upstream_id == upstream_id)
        assert 6 <= count <= 7


def test_select_default_subset_with_single_upstream():
    """Test default subset selection with a single upstream."""
    # Create 30 tools from one upstream
    tools = [
        ToolMetadata(
            name=f"upstream.tool{i}",
            original_name=f"tool{i}",
            description=f"Test tool {i}",
            input_schema=JSONSchema(type='object'),
            upstream_id="upstream"
        )
        for i in range(30)
    ]
    
    # Select default subset
    subset = select_default_tool_subset(tools, max_tools=20)
    
    # Should return 20 tools
    assert len(subset) == 20
    
    # All from the same upstream
    assert all(tool.upstream_id == "upstream" for tool in subset)
    
    # Should be the first 20 alphabetically (string sorting)
    # Alphabetical order: tool0, tool1, tool10, tool11, ..., tool19, tool2, tool20, ..., tool29, tool3, ...
    expected_names = sorted([f"tool{i}" for i in range(30)])[:20]
    actual_names = [tool.original_name for tool in subset]
    assert actual_names == expected_names


def test_select_default_subset_with_fewer_than_20_tools():
    """Test default subset selection when total tools < 20."""
    # Create only 10 tools
    tools = [
        ToolMetadata(
            name=f"upstream.tool{i}",
            original_name=f"tool{i}",
            description=f"Test tool {i}",
            input_schema=JSONSchema(type='object'),
            upstream_id="upstream"
        )
        for i in range(10)
    ]
    
    # Select default subset
    subset = select_default_tool_subset(tools, max_tools=20)
    
    # Should return all 10 tools
    assert len(subset) == 10
    
    # Should be all the tools
    assert set(tool.name for tool in subset) == set(tool.name for tool in tools)


def test_select_default_subset_empty_catalog():
    """Test default subset selection with empty catalog."""
    # Select from empty catalog
    subset = select_default_tool_subset([], max_tools=20)
    
    # Should return empty list
    assert len(subset) == 0


def test_select_default_subset_alphabetical_order():
    """Test that tools are selected alphabetically within each upstream."""
    # Create tools with specific names to test ordering
    tools = [
        ToolMetadata(
            name="upstream.zebra",
            original_name="zebra",
            description="Z tool",
            input_schema=JSONSchema(type='object'),
            upstream_id="upstream"
        ),
        ToolMetadata(
            name="upstream.apple",
            original_name="apple",
            description="A tool",
            input_schema=JSONSchema(type='object'),
            upstream_id="upstream"
        ),
        ToolMetadata(
            name="upstream.banana",
            original_name="banana",
            description="B tool",
            input_schema=JSONSchema(type='object'),
            upstream_id="upstream"
        ),
    ]
    
    # Select default subset
    subset = select_default_tool_subset(tools, max_tools=20)
    
    # Should return all 3 tools in alphabetical order
    assert len(subset) == 3
    assert subset[0].original_name == "apple"
    assert subset[1].original_name == "banana"
    assert subset[2].original_name == "zebra"


def test_select_default_subset_proportional_distribution():
    """Test that tools are distributed proportionally across upstreams."""
    # Create 2 upstreams with different numbers of tools
    tools = []
    
    # Upstream 1: 20 tools
    for i in range(20):
        tools.append(ToolMetadata(
            name=f"upstream1.tool{i}",
            original_name=f"tool{i}",
            description=f"Tool {i}",
            input_schema=JSONSchema(type='object'),
            upstream_id="upstream1"
        ))
    
    # Upstream 2: 20 tools
    for i in range(20):
        tools.append(ToolMetadata(
            name=f"upstream2.tool{i}",
            original_name=f"tool{i}",
            description=f"Tool {i}",
            input_schema=JSONSchema(type='object'),
            upstream_id="upstream2"
        ))
    
    # Select default subset
    subset = select_default_tool_subset(tools, max_tools=20)
    
    # Should return 20 tools
    assert len(subset) == 20
    
    # Should have 10 from each upstream (equal distribution)
    upstream1_count = sum(1 for tool in subset if tool.upstream_id == "upstream1")
    upstream2_count = sum(1 for tool in subset if tool.upstream_id == "upstream2")
    
    assert upstream1_count == 10
    assert upstream2_count == 10


def test_select_default_subset_custom_max_tools():
    """Test default subset selection with custom max_tools value."""
    # Create 30 tools
    tools = [
        ToolMetadata(
            name=f"upstream.tool{i}",
            original_name=f"tool{i}",
            description=f"Test tool {i}",
            input_schema=JSONSchema(type='object'),
            upstream_id="upstream"
        )
        for i in range(30)
    ]
    
    # Select subset with max_tools=10
    subset = select_default_tool_subset(tools, max_tools=10)
    
    # Should return exactly 10 tools
    assert len(subset) == 10
    
    # Should be the first 10 alphabetically (string sorting)
    # Alphabetical order: tool0, tool1, tool10, tool11, ..., tool19, tool2, ...
    expected_names = sorted([f"tool{i}" for i in range(30)])[:10]
    actual_names = [tool.original_name for tool in subset]
    assert actual_names == expected_names


def test_select_default_subset_uneven_upstream_sizes():
    """Test default subset with upstreams having very different tool counts."""
    tools = []
    
    # Upstream 1: 2 tools
    for i in range(2):
        tools.append(ToolMetadata(
            name=f"upstream1.tool{i}",
            original_name=f"tool{i}",
            description=f"Tool {i}",
            input_schema=JSONSchema(type='object'),
            upstream_id="upstream1"
        ))
    
    # Upstream 2: 50 tools
    for i in range(50):
        tools.append(ToolMetadata(
            name=f"upstream2.tool{i}",
            original_name=f"tool{i}",
            description=f"Tool {i}",
            input_schema=JSONSchema(type='object'),
            upstream_id="upstream2"
        ))
    
    # Select default subset
    subset = select_default_tool_subset(tools, max_tools=20)
    
    # Should return 20 tools
    assert len(subset) == 20
    
    # Both upstreams should be represented
    upstream_ids = {tool.upstream_id for tool in subset}
    assert len(upstream_ids) == 2
    
    # Upstream 1 should have at most 2 tools (all it has)
    upstream1_count = sum(1 for tool in subset if tool.upstream_id == "upstream1")
    assert upstream1_count == 2
    
    # Upstream 2 should have the rest
    upstream2_count = sum(1 for tool in subset if tool.upstream_id == "upstream2")
    assert upstream2_count == 18


def test_select_default_subset_deterministic():
    """Test that subset selection is deterministic for the same input."""
    # Create tools
    tools = []
    for upstream_idx in range(3):
        upstream_id = f"upstream{upstream_idx}"
        for tool_idx in range(10):
            tool = ToolMetadata(
                name=f"{upstream_id}.tool{tool_idx}",
                original_name=f"tool{tool_idx}",
                description=f"Test tool {tool_idx}",
                input_schema=JSONSchema(type='object'),
                upstream_id=upstream_id
            )
            tools.append(tool)
    
    # Select subset twice
    subset1 = select_default_tool_subset(tools, max_tools=20)
    subset2 = select_default_tool_subset(tools, max_tools=20)
    
    # Should be identical
    assert len(subset1) == len(subset2)
    assert [t.name for t in subset1] == [t.name for t in subset2]
