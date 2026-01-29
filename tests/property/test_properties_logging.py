"""
Property-based tests for structured logging.

Tests Property 24: All logs are valid structured JSON.
"""
import json
import logging
from io import StringIO
from hypothesis import given, strategies as st, settings, HealthCheck
from mcp_router.core.logging import (
    LogEntry,
    JSONFormatter,
    setup_logging,
    log_with_metadata
)


# Property 24: All logs are valid structured JSON (Requirements 9.5, 9.6)
@given(
    level=st.sampled_from(['info', 'warn', 'error']),
    component=st.text(min_size=1, max_size=50),
    message=st.text(min_size=1, max_size=200),
    metadata=st.one_of(
        st.none(),
        st.dictionaries(
            keys=st.text(min_size=1, max_size=20),
            values=st.one_of(
                st.text(max_size=50),
                st.integers(),
                st.floats(allow_nan=False, allow_infinity=False),
                st.booleans(),
                st.lists(st.text(max_size=20), max_size=5)
            ),
            max_size=5
        )
    )
)
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_property_24_log_entries_are_valid_json(level, component, message, metadata):
    """
    Property 24: All logs are valid structured JSON.
    
    Validates Requirements 9.5, 9.6:
    - All log entries can be parsed as JSON
    - All log entries have required fields (timestamp, level, component, message)
    - Timestamps are in ISO 8601 format
    - Log levels are valid (info, warn, error)
    """
    # Create log entry
    entry = LogEntry(
        level=level,
        component=component,
        message=message,
        metadata=metadata
    )
    
    # Convert to JSON
    json_str = entry.to_json()
    
    # Parse JSON - should not raise exception
    parsed = json.loads(json_str)
    
    # Verify required fields exist
    assert 'timestamp' in parsed
    assert 'level' in parsed
    assert 'component' in parsed
    assert 'message' in parsed
    
    # Verify field values
    assert parsed['level'] == level
    assert parsed['component'] == component
    assert parsed['message'] == message
    
    # Verify timestamp is ISO 8601 format (basic check)
    assert 'T' in parsed['timestamp']
    assert len(parsed['timestamp']) > 10
    
    # Verify metadata if present
    if metadata is not None:
        assert 'metadata' in parsed
        assert parsed['metadata'] == metadata
    else:
        # metadata field should be null or absent
        assert parsed.get('metadata') is None


@given(
    message=st.text(min_size=1, max_size=100),
    metadata=st.dictionaries(
        keys=st.text(min_size=1, max_size=20),
        values=st.text(max_size=50),
        max_size=3
    )
)
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_property_24_json_formatter_produces_valid_json(message, metadata):
    """
    Property 24: JSONFormatter produces valid JSON logs.
    
    Validates that the JSONFormatter class produces parseable JSON output.
    """
    # Set up logger with JSON formatter
    logger = logging.getLogger('test_logger')
    logger.setLevel(logging.INFO)
    
    # Create string buffer to capture output
    buffer = StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    
    try:
        # Log with metadata
        log_with_metadata(logger, logging.INFO, message, metadata=metadata)
        
        # Get logged output
        output = buffer.getvalue().strip()
        
        # Parse JSON - should not raise exception
        parsed = json.loads(output)
        
        # Verify required fields
        assert 'timestamp' in parsed
        assert 'level' in parsed
        assert 'component' in parsed
        assert 'message' in parsed
        
        # Verify message
        assert parsed['message'] == message
        
        # Verify metadata
        assert 'metadata' in parsed
        assert parsed['metadata'] == metadata
        
    finally:
        # Clean up
        logger.removeHandler(handler)


@given(
    messages=st.lists(
        st.tuples(
            st.sampled_from(['info', 'warn', 'error']),
            st.text(min_size=1, max_size=100)
        ),
        min_size=1,
        max_size=10
    )
)
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_property_24_multiple_log_entries_are_valid_json(messages):
    """
    Property 24: Multiple log entries are all valid JSON.
    
    Validates that a sequence of log entries all produce valid JSON.
    """
    # Set up logger with JSON formatter
    logger = logging.getLogger('test_multi_logger')
    logger.setLevel(logging.DEBUG)
    
    # Create string buffer to capture output
    buffer = StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    
    try:
        # Log multiple messages
        for level_str, message in messages:
            level = getattr(logging, level_str.upper())
            logger.log(level, message)
        
        # Get logged output
        output = buffer.getvalue()
        
        # Split into lines and parse each
        lines = [line.strip() for line in output.split('\n') if line.strip()]
        
        assert len(lines) == len(messages)
        
        for line, (expected_level, expected_message) in zip(lines, messages):
            # Parse JSON - should not raise exception
            parsed = json.loads(line)
            
            # Verify required fields
            assert 'timestamp' in parsed
            assert 'level' in parsed
            assert 'component' in parsed
            assert 'message' in parsed
            
            # Verify values
            assert parsed['level'] == expected_level
            assert parsed['message'] == expected_message
            
    finally:
        # Clean up
        logger.removeHandler(handler)


@given(
    component=st.text(min_size=1, max_size=50),
    message=st.text(min_size=1, max_size=100)
)
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_property_24_log_levels_are_valid(component, message):
    """
    Property 24: Log levels are valid (info, warn, error).
    
    Validates Requirement 9.6: Log levels are one of info, warn, error.
    """
    valid_levels = ['info', 'warn', 'error']
    
    for level in valid_levels:
        entry = LogEntry(
            level=level,
            component=component,
            message=message
        )
        
        json_str = entry.to_json()
        parsed = json.loads(json_str)
        
        assert parsed['level'] in valid_levels
        assert parsed['level'] == level


@given(
    metadata_keys=st.lists(
        st.text(min_size=1, max_size=20),
        min_size=1,
        max_size=10,
        unique=True
    ),
    metadata_values=st.lists(
        st.one_of(
            st.text(max_size=50),
            st.integers(),
            st.booleans()
        ),
        min_size=1,
        max_size=10
    )
)
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_property_24_metadata_is_preserved_in_json(metadata_keys, metadata_values):
    """
    Property 24: Metadata is correctly preserved in JSON output.
    
    Validates that metadata dictionaries are correctly serialized to JSON.
    """
    # Create metadata dict (handle length mismatch)
    min_len = min(len(metadata_keys), len(metadata_values))
    metadata = dict(zip(metadata_keys[:min_len], metadata_values[:min_len]))
    
    if not metadata:
        # Skip if empty
        return
    
    entry = LogEntry(
        level='info',
        component='test',
        message='test message',
        metadata=metadata
    )
    
    json_str = entry.to_json()
    parsed = json.loads(json_str)
    
    assert 'metadata' in parsed
    assert parsed['metadata'] == metadata
    
    # Verify all keys and values are preserved
    for key, value in metadata.items():
        assert key in parsed['metadata']
        assert parsed['metadata'][key] == value
