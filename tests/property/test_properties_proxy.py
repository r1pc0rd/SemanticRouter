"""
Property-based tests for tool call proxy.

Tests Property 2: Tool calls are proxied transparently
Tests Property 18: Tool namespace parsing identifies correct upstream
Tests Property 19: Invalid tool namespace produces error
Tests Property 20: Upstream errors are forwarded to client
Tests Property 21: Timeout errors are reported after 30 seconds
Tests Property 27: Upstream connections are reused
Validates: Requirements 2.6, 3.1, 3.2, 3.3, 3.4, 5.1, 5.2, 5.3
"""
import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from unittest.mock import AsyncMock, MagicMock
import asyncio

from mcp_router.core.config import UpstreamConfig
from mcp_router.core.models import ToolCallResult, ContentItem
from mcp_router.proxy.proxy import ToolCallProxy
from mcp_router.discovery.upstream import UpstreamConnection


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

# Strategy for generating semantic prefixes
semantic_prefix_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='-_'),
    min_size=1,
    max_size=30
)

# Strategy for generating tool arguments
arguments_strategy = st.dictionaries(
    keys=st.text(min_size=1, max_size=20),
    values=st.one_of(
        st.text(max_size=100),
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.booleans(),
        st.none()
    ),
    max_size=10
)


@pytest.mark.property
@pytest.mark.asyncio
@given(
    upstream_id=upstream_id_strategy,
    tool_name=tool_name_strategy,
    arguments=arguments_strategy,
    result_text=st.text(min_size=1, max_size=200)
)
@settings(suppress_health_check=[HealthCheck.too_slow])
async def test_property_2_transparent_proxying(
    upstream_id: str,
    tool_name: str,
    arguments: dict,
    result_text: str
) -> None:
    """
    Property 2: Tool calls are proxied transparently.
    
    For any tool call, the router SHALL forward the call to the upstream
    and return the result without modification.
    
    Validates: Requirements 2.6, 3.1
    """
    # Create mock upstream connection
    config = UpstreamConfig(transport='stdio', command='test')
    upstream = UpstreamConnection(upstream_id, config)
    upstream._session = MagicMock()  # Mark as connected
    
    # Mock the call_tool method to return a result
    expected_result = {
        'content': [{'type': 'text', 'text': result_text}],
        'isError': False
    }
    upstream.call_tool = AsyncMock(return_value=expected_result)
    
    # Create proxy
    proxy = ToolCallProxy(
        upstreams={upstream_id: upstream},
        upstream_configs={upstream_id: config}
    )
    
    # Call tool through proxy
    namespaced_name = f"{upstream_id}.{tool_name}"
    result = await proxy.call_tool(namespaced_name, arguments)
    
    # Verify call was forwarded to upstream
    upstream.call_tool.assert_called_once_with(tool_name, arguments)
    
    # Verify result is returned without modification
    assert isinstance(result, ToolCallResult)
    assert len(result.content) == 1
    assert result.content[0].type == 'text'
    assert result.content[0].text == result_text
    assert result.is_error is False


@pytest.mark.property
@pytest.mark.asyncio
@given(
    upstream_id=upstream_id_strategy,
    semantic_prefix=semantic_prefix_strategy,
    tool_name=tool_name_strategy,
    use_semantic_prefix=st.booleans()
)
async def test_property_18_namespace_parsing_identifies_upstream(
    upstream_id: str,
    semantic_prefix: str,
    tool_name: str,
    use_semantic_prefix: bool
) -> None:
    """
    Property 18: Tool namespace parsing identifies correct upstream.
    
    For any namespaced tool name, the proxy SHALL correctly identify
    the upstream server by parsing the namespace prefix.
    
    Validates: Requirements 3.2
    """
    # Create config with optional semantic prefix
    config = UpstreamConfig(
        transport='stdio',
        command='test',
        semantic_prefix=semantic_prefix if use_semantic_prefix else None
    )
    
    # Create mock upstream
    upstream = UpstreamConnection(upstream_id, config)
    upstream._session = MagicMock()  # Mark as connected
    upstream.call_tool = AsyncMock(return_value={
        'content': [{'type': 'text', 'text': 'result'}],
        'isError': False
    })
    
    # Create proxy
    proxy = ToolCallProxy(
        upstreams={upstream_id: upstream},
        upstream_configs={upstream_id: config}
    )
    
    # Build namespaced name using appropriate prefix
    prefix = semantic_prefix if use_semantic_prefix else upstream_id
    namespaced_name = f"{prefix}.{tool_name}"
    
    # Call tool
    result = await proxy.call_tool(namespaced_name, {})
    
    # Verify correct upstream was called
    upstream.call_tool.assert_called_once_with(tool_name, {})
    assert isinstance(result, ToolCallResult)


@pytest.mark.property
@pytest.mark.asyncio
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
        # Names with empty tool name (ends with dot)
        st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='-_'),
            min_size=1,
            max_size=50
        ).map(lambda x: f"{x}."),
    )
)
async def test_property_19_invalid_namespace_produces_error(
    invalid_name: str
) -> None:
    """
    Property 19: Invalid tool namespace produces error.
    
    For any tool call with an invalid namespace format, the proxy SHALL
    return an error without attempting to forward the call.
    
    Validates: Requirements 3.3
    """
    # Create proxy with empty upstreams
    proxy = ToolCallProxy(upstreams={}, upstream_configs={})
    
    # Attempt to call tool with invalid namespace
    with pytest.raises(ValueError, match="Invalid tool namespace"):
        await proxy.call_tool(invalid_name, {})


@pytest.mark.property
@pytest.mark.asyncio
@given(
    upstream_id=upstream_id_strategy,
    tool_name=tool_name_strategy,
    error_message=st.text(min_size=1, max_size=200)
)
async def test_property_20_upstream_errors_forwarded(
    upstream_id: str,
    tool_name: str,
    error_message: str
) -> None:
    """
    Property 20: Upstream errors are forwarded to client.
    
    For any tool call that fails on the upstream, the proxy SHALL
    forward the error to the client without modification.
    
    Validates: Requirements 3.4
    """
    # Create mock upstream that raises error
    config = UpstreamConfig(transport='stdio', command='test')
    upstream = UpstreamConnection(upstream_id, config)
    upstream._session = MagicMock()  # Mark as connected
    upstream.call_tool = AsyncMock(side_effect=RuntimeError(error_message))
    
    # Create proxy
    proxy = ToolCallProxy(
        upstreams={upstream_id: upstream},
        upstream_configs={upstream_id: config}
    )
    
    # Call tool through proxy
    namespaced_name = f"{upstream_id}.{tool_name}"
    
    # Verify error is forwarded
    with pytest.raises(RuntimeError, match="Upstream error"):
        await proxy.call_tool(namespaced_name, {})


@pytest.mark.property
@pytest.mark.asyncio
@given(
    upstream_id=upstream_id_strategy,
    tool_name=tool_name_strategy
)
@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
async def test_property_21_timeout_errors_reported(
    upstream_id: str,
    tool_name: str
) -> None:
    """
    Property 21: Timeout errors are reported after 30 seconds.
    
    For any tool call that exceeds the timeout, the proxy SHALL
    return a timeout error.
    
    Validates: Requirements 5.1
    """
    # Create mock upstream that hangs
    config = UpstreamConfig(transport='stdio', command='test')
    upstream = UpstreamConnection(upstream_id, config)
    upstream._session = MagicMock()  # Mark as connected
    
    async def slow_call(*args, **kwargs):
        await asyncio.sleep(100)  # Simulate slow call
        return {'content': [], 'isError': False}
    
    upstream.call_tool = AsyncMock(side_effect=slow_call)
    
    # Create proxy
    proxy = ToolCallProxy(
        upstreams={upstream_id: upstream},
        upstream_configs={upstream_id: config}
    )
    
    # Call tool with short timeout
    namespaced_name = f"{upstream_id}.{tool_name}"
    
    # Verify timeout error is raised
    with pytest.raises(asyncio.TimeoutError, match="timed out"):
        await proxy.call_tool(namespaced_name, {}, timeout=0.1)


@pytest.mark.property
@pytest.mark.asyncio
@given(
    upstream_id=upstream_id_strategy,
    tool_names=st.lists(tool_name_strategy, min_size=2, max_size=5, unique=True)
)
async def test_property_27_connection_reuse(
    upstream_id: str,
    tool_names: list[str]
) -> None:
    """
    Property 27: Upstream connections are reused.
    
    For multiple tool calls to the same upstream, the proxy SHALL
    reuse the existing connection rather than creating new connections.
    
    Validates: Requirements 5.2, 5.3
    """
    # Create mock upstream
    config = UpstreamConfig(transport='stdio', command='test')
    upstream = UpstreamConnection(upstream_id, config)
    upstream._session = MagicMock()  # Mark as connected
    upstream.call_tool = AsyncMock(return_value={
        'content': [{'type': 'text', 'text': 'result'}],
        'isError': False
    })
    
    # Create proxy
    proxy = ToolCallProxy(
        upstreams={upstream_id: upstream},
        upstream_configs={upstream_id: config}
    )
    
    # Call multiple tools
    for tool_name in tool_names:
        namespaced_name = f"{upstream_id}.{tool_name}"
        await proxy.call_tool(namespaced_name, {})
    
    # Verify all calls used the same upstream connection
    assert upstream.call_tool.call_count == len(tool_names)
    
    # Verify the same upstream instance was used
    # (proxy.upstreams should still contain the same upstream object)
    assert proxy.upstreams[upstream_id] is upstream


@pytest.mark.property
@pytest.mark.asyncio
@given(
    upstream_id=upstream_id_strategy,
    tool_name=tool_name_strategy,
    unknown_prefix=st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='-_'),
        min_size=1,
        max_size=30
    )
)
async def test_property_19_unknown_upstream_produces_error(
    upstream_id: str,
    tool_name: str,
    unknown_prefix: str
) -> None:
    """
    Property 19: Unknown upstream prefix produces error.
    
    For any tool call with a namespace prefix that doesn't match any
    configured upstream, the proxy SHALL return an error.
    
    Validates: Requirements 3.3
    """
    # Ensure unknown_prefix is different from upstream_id
    if unknown_prefix == upstream_id:
        unknown_prefix = f"{unknown_prefix}_different"
    
    # Create proxy with one upstream
    config = UpstreamConfig(transport='stdio', command='test')
    upstream = UpstreamConnection(upstream_id, config)
    upstream._session = MagicMock()
    
    proxy = ToolCallProxy(
        upstreams={upstream_id: upstream},
        upstream_configs={upstream_id: config}
    )
    
    # Attempt to call tool with unknown prefix
    namespaced_name = f"{unknown_prefix}.{tool_name}"
    
    with pytest.raises(ValueError, match="No upstream found"):
        await proxy.call_tool(namespaced_name, {})
