"""
Property-based tests for configuration parsing.

Tests Property 14: Config parsing extracts all upstream fields
Tests Property 15: Invalid config causes startup failure
Validates: Requirements 6.2, 6.3, 6.6
"""
import pytest
from hypothesis import given, strategies as st, settings, HealthCheck

from mcp_router.core.config import UpstreamConfig
from mcp_router.core.config_parser import parse_config


# Strategy for generating valid transport types
transport_strategy = st.sampled_from(['stdio', 'sse', 'http'])

# Strategy for generating valid upstream IDs
upstream_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='-_'),
    min_size=1,
    max_size=50
)

# Strategy for generating command strings
command_strategy = st.text(min_size=1, max_size=100)

# Strategy for generating args lists
args_strategy = st.lists(st.text(min_size=1, max_size=50), min_size=0, max_size=10)

# Strategy for generating URLs
url_strategy = st.from_regex(r'https?://[a-z0-9.-]+(?::[0-9]+)?(?:/[a-z0-9._-]*)*', fullmatch=True)

# Strategy for generating optional strings
optional_string_strategy = st.one_of(st.none(), st.text(min_size=1, max_size=200))


@st.composite
def stdio_upstream_config(draw: st.DrawFn) -> dict:
    """Generate a valid stdio upstream configuration."""
    return {
        'transport': 'stdio',
        'command': draw(command_strategy),
        'args': draw(args_strategy),
        'semantic_prefix': draw(optional_string_strategy),
        'category_description': draw(optional_string_strategy)
    }


@st.composite
def sse_upstream_config(draw: st.DrawFn) -> dict:
    """Generate a valid SSE upstream configuration."""
    return {
        'transport': 'sse',
        'url': draw(url_strategy),
        'semantic_prefix': draw(optional_string_strategy),
        'category_description': draw(optional_string_strategy)
    }


@st.composite
def http_upstream_config(draw: st.DrawFn) -> dict:
    """Generate a valid HTTP upstream configuration."""
    return {
        'transport': 'http',
        'url': draw(url_strategy),
        'semantic_prefix': draw(optional_string_strategy),
        'category_description': draw(optional_string_strategy)
    }


@st.composite
def valid_config_dict(draw: st.DrawFn) -> dict:
    """Generate a valid configuration dictionary."""
    # Generate 1-5 upstream servers
    num_upstreams = draw(st.integers(min_value=1, max_value=5))
    
    mcp_servers = {}
    for i in range(num_upstreams):
        upstream_id = draw(upstream_id_strategy)
        # Ensure unique IDs
        while upstream_id in mcp_servers:
            upstream_id = draw(upstream_id_strategy)
        
        # Randomly choose transport type
        transport_type = draw(st.sampled_from(['stdio', 'sse', 'http']))
        
        if transport_type == 'stdio':
            upstream_config = draw(stdio_upstream_config())
        elif transport_type == 'sse':
            upstream_config = draw(sse_upstream_config())
        else:  # http
            upstream_config = draw(http_upstream_config())
        
        mcp_servers[upstream_id] = upstream_config
    
    return {'mcpServers': mcp_servers}


@pytest.mark.property
@given(config_dict=valid_config_dict())
@settings(suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
def test_property_14_config_parsing_extracts_all_fields(config_dict: dict) -> None:
    """
    Property 14: Config parsing extracts all upstream fields.
    
    For any valid configuration file with upstream definitions, the router SHALL
    extract all required fields (id, transport, command/args for stdio) and
    optional fields (semantic_prefix, category_description) when present.
    
    Validates: Requirements 6.2
    """
    # Parse the configuration
    router_config = parse_config(config_dict)
    
    # Verify all upstreams were parsed
    assert len(router_config.mcp_servers) == len(config_dict['mcpServers'])
    
    # Verify each upstream configuration
    for upstream_id, expected_config in config_dict['mcpServers'].items():
        # Upstream should exist in parsed config
        assert upstream_id in router_config.mcp_servers
        
        actual_config = router_config.mcp_servers[upstream_id]
        
        # Verify required fields
        assert actual_config.transport == expected_config['transport']
        
        # Verify transport-specific required fields
        if expected_config['transport'] == 'stdio':
            assert actual_config.command == expected_config['command']
            assert actual_config.args == expected_config.get('args', [])
        elif expected_config['transport'] in ('sse', 'http'):
            assert actual_config.url == expected_config['url']
        
        # Verify optional fields when present
        if 'semantic_prefix' in expected_config:
            assert actual_config.semantic_prefix == expected_config['semantic_prefix']
        
        if 'category_description' in expected_config:
            assert actual_config.category_description == expected_config['category_description']


@pytest.mark.property
@given(
    upstream_id=upstream_id_strategy,
    command=command_strategy,
    args=args_strategy,
    semantic_prefix=optional_string_strategy,
    category_description=optional_string_strategy
)
def test_property_14_stdio_config_fields(
    upstream_id: str,
    command: str,
    args: list[str],
    semantic_prefix: str | None,
    category_description: str | None
) -> None:
    """
    Property 14: Stdio configuration extracts all fields correctly.
    
    Validates: Requirements 6.2
    """
    config_dict = {
        'mcpServers': {
            upstream_id: {
                'transport': 'stdio',
                'command': command,
                'args': args,
                'semantic_prefix': semantic_prefix,
                'category_description': category_description
            }
        }
    }
    
    router_config = parse_config(config_dict)
    upstream_config = router_config.mcp_servers[upstream_id]
    
    assert upstream_config.transport == 'stdio'
    assert upstream_config.command == command
    assert upstream_config.args == args
    assert upstream_config.semantic_prefix == semantic_prefix
    assert upstream_config.category_description == category_description


@pytest.mark.property
@given(
    upstream_id=upstream_id_strategy,
    url=url_strategy,
    semantic_prefix=optional_string_strategy,
    category_description=optional_string_strategy
)
def test_property_14_sse_config_fields(
    upstream_id: str,
    url: str,
    semantic_prefix: str | None,
    category_description: str | None
) -> None:
    """
    Property 14: SSE configuration extracts all fields correctly.
    
    Validates: Requirements 6.2
    """
    config_dict = {
        'mcpServers': {
            upstream_id: {
                'transport': 'sse',
                'url': url,
                'semantic_prefix': semantic_prefix,
                'category_description': category_description
            }
        }
    }
    
    router_config = parse_config(config_dict)
    upstream_config = router_config.mcp_servers[upstream_id]
    
    assert upstream_config.transport == 'sse'
    assert upstream_config.url == url
    assert upstream_config.semantic_prefix == semantic_prefix
    assert upstream_config.category_description == category_description



@pytest.mark.property
@given(
    config_dict=st.one_of(
        # Missing mcpServers key
        st.fixed_dictionaries({}),
        st.fixed_dictionaries({'other_key': st.text()}),
        # mcpServers is not a dict
        st.fixed_dictionaries({'mcpServers': st.one_of(st.text(), st.integers(), st.lists(st.text()))}),
        # mcpServers is empty
        st.fixed_dictionaries({'mcpServers': st.just({})}),
    )
)
def test_property_15_invalid_config_causes_failure(config_dict: dict) -> None:
    """
    Property 15: Invalid config causes startup failure.
    
    For any configuration file that is malformed (invalid JSON), missing required
    fields, or has invalid structure, the router SHALL fail to start and SHALL
    log a descriptive error.
    
    Validates: Requirements 6.3, 6.6
    """
    # Invalid configurations should raise an error
    with pytest.raises((ValueError, KeyError, TypeError)):
        parse_config(config_dict)


@pytest.mark.property
@given(
    upstream_id=upstream_id_strategy,
    transport=st.sampled_from(['stdio', 'sse', 'http'])
)
def test_property_15_missing_required_fields_causes_failure(
    upstream_id: str,
    transport: str
) -> None:
    """
    Property 15: Missing required fields causes startup failure.
    
    Validates: Requirements 6.3, 6.6
    """
    # Create config missing required fields for the transport type
    if transport == 'stdio':
        # Missing command
        config_dict = {
            'mcpServers': {
                upstream_id: {
                    'transport': 'stdio',
                    # command is missing
                }
            }
        }
    else:  # sse or http
        # Missing url
        config_dict = {
            'mcpServers': {
                upstream_id: {
                    'transport': transport,
                    # url is missing
                }
            }
        }
    
    # Should raise ValueError due to missing required fields
    with pytest.raises(ValueError):
        parse_config(config_dict)


@pytest.mark.property
@given(
    upstream_id=upstream_id_strategy,
    invalid_transport=st.text().filter(lambda x: x not in ('stdio', 'sse', 'http'))
)
def test_property_15_invalid_transport_causes_failure(
    upstream_id: str,
    invalid_transport: str
) -> None:
    """
    Property 15: Invalid transport type causes startup failure.
    
    Validates: Requirements 6.3, 6.6
    """
    config_dict = {
        'mcpServers': {
            upstream_id: {
                'transport': invalid_transport,
                'command': 'test'
            }
        }
    }
    
    # Should raise ValueError due to invalid transport
    with pytest.raises(ValueError):
        parse_config(config_dict)
