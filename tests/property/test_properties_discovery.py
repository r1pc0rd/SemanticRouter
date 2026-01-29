"""
Property-based tests for tool discovery and namespace generation.

Tests Property 4: Tool discovery creates correct namespaces and metadata
Tests Property 16: Semantic prefix overrides default namespace
Validates: Requirements 2.4, 2.5, 6.4
"""
import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
import numpy as np

from mcp_router.core.config import UpstreamConfig
from mcp_router.core.models import ToolMetadata, JSONSchema
from mcp_router.core.namespace import (
    generate_tool_namespace,
    parse_tool_namespace,
    match_upstream_by_prefix,
)


# Strategy for generating valid upstream IDs
upstream_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='-_'),
    min_size=1,
    max_size=50
)

# Strategy for generating tool names
tool_name_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='-_'),
    min_size=1,
    max_size=50
)

# Strategy for generating optional semantic prefixes
semantic_prefix_strategy = st.one_of(
    st.none(),
    st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='-_'),
        min_size=1,
        max_size=30
    )
)


@pytest.mark.property
@given(
    upstream_id=upstream_id_strategy,
    tool_name=tool_name_strategy,
    semantic_prefix=semantic_prefix_strategy
)
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_property_4_namespace_generation(
    upstream_id: str,
    tool_name: str,
    semantic_prefix: str | None
) -> None:
    """
    Property 4: Tool discovery creates correct namespaces and metadata.
    
    For any tool discovered from an upstream MCP, the tool SHALL be namespaced
    as {upstream_id}.{tool_name} (or {semantic_prefix}.{tool_name} if configured).
    
    Validates: Requirements 2.4, 2.5
    """
    # Create upstream config
    config = UpstreamConfig(
        transport='stdio',
        command='test',
        semantic_prefix=semantic_prefix
    )
    
    # Generate namespace
    namespaced_name = generate_tool_namespace(upstream_id, tool_name, config)
    
    # Verify namespace format
    assert '.' in namespaced_name
    prefix, parsed_tool_name = namespaced_name.split('.', 1)
    
    # Verify tool name is preserved
    assert parsed_tool_name == tool_name
    
    # Verify prefix is correct
    if semantic_prefix:
        assert prefix == semantic_prefix
    else:
        assert prefix == upstream_id


@pytest.mark.property
@given(
    upstream_id=upstream_id_strategy,
    tool_name=tool_name_strategy,
    semantic_prefix=st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='-_'),
        min_size=1,
        max_size=30
    )
)
def test_property_16_semantic_prefix_overrides_namespace(
    upstream_id: str,
    tool_name: str,
    semantic_prefix: str
) -> None:
    """
    Property 16: Semantic prefix overrides default namespace.
    
    For any upstream with a semantic_prefix configured, all tools from that
    upstream SHALL use the semantic_prefix instead of the upstream_id in their namespace.
    
    Validates: Requirements 6.4
    """
    # Create config with semantic prefix
    config = UpstreamConfig(
        transport='stdio',
        command='test',
        semantic_prefix=semantic_prefix
    )
    
    # Generate namespace
    namespaced_name = generate_tool_namespace(upstream_id, tool_name, config)
    
    # Verify semantic prefix is used
    prefix, _ = namespaced_name.split('.', 1)
    assert prefix == semantic_prefix
    assert prefix != upstream_id or semantic_prefix == upstream_id


@pytest.mark.property
@given(
    prefix=st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='-_'),
        min_size=1,
        max_size=30
    ),
    tool_name=tool_name_strategy
)
def test_property_4_namespace_parsing(prefix: str, tool_name: str) -> None:
    """
    Property 4: Namespaced tool names can be parsed correctly.
    
    Validates: Requirements 2.4
    """
    # Create namespaced name
    namespaced_name = f"{prefix}.{tool_name}"
    
    # Parse it
    parsed_prefix, parsed_tool_name = parse_tool_namespace(namespaced_name)
    
    # Verify parsing is correct
    assert parsed_prefix == prefix
    assert parsed_tool_name == tool_name


@pytest.mark.property
@given(
    invalid_name=st.one_of(
        # Names without dots
        st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='-_'),
            min_size=1,
            max_size=50
        ).filter(lambda x: '.' not in x),
        # Names with empty prefix (starts with dot)
        st.text(min_size=1, max_size=50).map(lambda x: f".{x}"),
        # Names with empty tool name (ends with dot, no dots in prefix)
        st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='-_'),
            min_size=1,
            max_size=50
        ).map(lambda x: f"{x}."),
        # Names with only whitespace after dot
        st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='-_'),
            min_size=1,
            max_size=30
        ).flatmap(lambda prefix: st.text(alphabet=' \t\n', min_size=1, max_size=10).map(lambda ws: f"{prefix}.{ws}")),
        # Names with only dots after first dot
        st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='-_'),
            min_size=1,
            max_size=30
        ).flatmap(lambda prefix: st.text(alphabet='.', min_size=1, max_size=10).map(lambda dots: f"{prefix}.{dots}")),
    )
)
def test_property_4_invalid_namespace_raises_error(invalid_name: str) -> None:
    """
    Property 4: Invalid namespaced names raise errors.
    
    Validates: Requirements 2.4
    """
    with pytest.raises(ValueError, match="must be namespaced"):
        parse_tool_namespace(invalid_name)


@pytest.mark.property
@given(
    upstream_id=upstream_id_strategy,
    semantic_prefix=semantic_prefix_strategy,
    query_prefix=st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='-_'),
        min_size=1,
        max_size=30
    )
)
def test_property_16_upstream_matching_by_prefix(
    upstream_id: str,
    semantic_prefix: str | None,
    query_prefix: str
) -> None:
    """
    Property 16: Upstream can be matched by either upstream_id or semantic_prefix.
    
    Validates: Requirements 6.4
    """
    # Create upstream config
    config = UpstreamConfig(
        transport='stdio',
        command='test',
        semantic_prefix=semantic_prefix
    )
    
    upstream_configs = {upstream_id: config}
    
    # Test matching by upstream_id
    matched = match_upstream_by_prefix(upstream_id, upstream_configs)
    assert matched == upstream_id
    
    # Test matching by semantic_prefix (if configured)
    if semantic_prefix:
        matched = match_upstream_by_prefix(semantic_prefix, upstream_configs)
        assert matched == upstream_id
    
    # Test non-matching prefix
    if query_prefix != upstream_id and query_prefix != semantic_prefix:
        matched = match_upstream_by_prefix(query_prefix, upstream_configs)
        assert matched is None


@pytest.mark.property
@given(
    upstream_id=upstream_id_strategy,
    tool_name=tool_name_strategy,
    description=st.text(min_size=1, max_size=200),
    category_description=st.one_of(st.none(), st.text(min_size=1, max_size=200))
)
def test_property_4_tool_metadata_creation(
    upstream_id: str,
    tool_name: str,
    description: str,
    category_description: str | None
) -> None:
    """
    Property 4: Tool metadata includes all required fields.
    
    Validates: Requirements 2.5
    """
    # Create tool metadata
    config = UpstreamConfig(
        transport='stdio',
        command='test',
        category_description=category_description
    )
    
    namespaced_name = generate_tool_namespace(upstream_id, tool_name, config)
    
    input_schema = JSONSchema(type='object')
    
    tool = ToolMetadata(
        name=namespaced_name,
        original_name=tool_name,
        description=description,
        input_schema=input_schema,
        upstream_id=upstream_id,
        category_description=category_description
    )
    
    # Verify all required fields are present
    assert tool.name == namespaced_name
    assert tool.original_name == tool_name
    assert tool.description == description
    assert tool.input_schema == input_schema
    assert tool.upstream_id == upstream_id
    assert tool.category_description == category_description
    
    # Verify namespace format
    assert '.' in tool.name
    prefix, parsed_tool_name = tool.name.split('.', 1)
    assert parsed_tool_name == tool_name
