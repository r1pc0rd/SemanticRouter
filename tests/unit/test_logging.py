"""
Unit tests for structured logging.

Tests logging infrastructure and log generation for all operations.
"""
import json
import logging
from io import StringIO
from datetime import datetime
import pytest

from mcp_router.core.logging import (
    LogEntry,
    JSONFormatter,
    setup_logging,
    log_with_metadata
)


class TestLogEntry:
    """Tests for LogEntry dataclass."""
    
    def test_log_entry_creation(self):
        """Test creating a log entry."""
        entry = LogEntry(
            level='info',
            component='test_component',
            message='test message'
        )
        
        assert entry.level == 'info'
        assert entry.component == 'test_component'
        assert entry.message == 'test message'
        assert entry.metadata is None
        assert isinstance(entry.timestamp, str)
    
    def test_log_entry_with_metadata(self):
        """Test creating a log entry with metadata."""
        metadata = {'key': 'value', 'count': 42}
        entry = LogEntry(
            level='warn',
            component='test_component',
            message='test message',
            metadata=metadata
        )
        
        assert entry.metadata == metadata
    
    def test_log_entry_to_json(self):
        """Test converting log entry to JSON."""
        entry = LogEntry(
            level='error',
            component='test_component',
            message='test message',
            metadata={'error': 'something went wrong'}
        )
        
        json_str = entry.to_json()
        parsed = json.loads(json_str)
        
        assert parsed['level'] == 'error'
        assert parsed['component'] == 'test_component'
        assert parsed['message'] == 'test message'
        assert parsed['metadata'] == {'error': 'something went wrong'}
        assert 'timestamp' in parsed
    
    def test_log_entry_timestamp_format(self):
        """Test that timestamp is in ISO 8601 format."""
        entry = LogEntry(
            level='info',
            component='test',
            message='test'
        )
        
        # Parse timestamp to verify format
        timestamp = datetime.fromisoformat(entry.timestamp.replace('Z', '+00:00'))
        assert isinstance(timestamp, datetime)
    
    def test_log_entry_without_metadata(self):
        """Test log entry without metadata."""
        entry = LogEntry(
            level='info',
            component='test',
            message='test'
        )
        
        json_str = entry.to_json()
        parsed = json.loads(json_str)
        
        assert parsed.get('metadata') is None


class TestJSONFormatter:
    """Tests for JSONFormatter class."""
    
    def test_json_formatter_formats_record(self):
        """Test that JSONFormatter formats log records as JSON."""
        formatter = JSONFormatter()
        
        # Create log record
        record = logging.LogRecord(
            name='test_logger',
            level=logging.INFO,
            pathname='test.py',
            lineno=1,
            msg='test message',
            args=(),
            exc_info=None
        )
        
        # Format record
        formatted = formatter.format(record)
        
        # Parse JSON
        parsed = json.loads(formatted)
        
        assert 'timestamp' in parsed
        assert 'level' in parsed
        assert 'component' in parsed
        assert 'message' in parsed
        assert parsed['message'] == 'test message'
    
    def test_json_formatter_with_metadata(self):
        """Test JSONFormatter with metadata in record."""
        formatter = JSONFormatter()
        
        # Create log record with metadata
        record = logging.LogRecord(
            name='test_logger',
            level=logging.WARNING,
            pathname='test.py',
            lineno=1,
            msg='test message',
            args=(),
            exc_info=None
        )
        record.metadata = {'key': 'value', 'count': 42}
        
        # Format record
        formatted = formatter.format(record)
        
        # Parse JSON
        parsed = json.loads(formatted)
        
        assert 'metadata' in parsed
        assert parsed['metadata'] == {'key': 'value', 'count': 42}
    
    def test_json_formatter_log_levels(self):
        """Test JSONFormatter with different log levels."""
        formatter = JSONFormatter()
        
        levels = [
            (logging.INFO, 'info'),
            (logging.WARNING, 'warn'),
            (logging.ERROR, 'error')
        ]
        
        for level_int, level_str in levels:
            record = logging.LogRecord(
                name='test_logger',
                level=level_int,
                pathname='test.py',
                lineno=1,
                msg='test message',
                args=(),
                exc_info=None
            )
            
            formatted = formatter.format(record)
            parsed = json.loads(formatted)
            
            assert parsed['level'] == level_str


class TestSetupLogging:
    """Tests for setup_logging function."""
    
    def test_setup_logging_configures_logger(self):
        """Test that setup_logging configures the root logger."""
        # Set up logging
        setup_logging()
        
        # Get root logger
        logger = logging.getLogger()
        
        # Verify logger is configured
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0
        
        # Verify handler has JSON formatter
        handler = logger.handlers[0]
        assert isinstance(handler.formatter, JSONFormatter)
    
    def test_setup_logging_with_custom_level(self):
        """Test setup_logging with custom log level."""
        setup_logging(level=logging.DEBUG)
        
        logger = logging.getLogger()
        assert logger.level == logging.DEBUG


class TestLogWithMetadata:
    """Tests for log_with_metadata helper function."""
    
    def test_log_with_metadata_attaches_metadata(self):
        """Test that log_with_metadata attaches metadata to log record."""
        # Set up logger with JSON formatter
        logger = logging.getLogger('test_metadata_logger')
        logger.setLevel(logging.INFO)
        
        buffer = StringIO()
        handler = logging.StreamHandler(buffer)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        
        try:
            # Log with metadata
            metadata = {'tool_count': 5, 'upstream_id': 'test'}
            log_with_metadata(
                logger,
                logging.INFO,
                'Test message',
                metadata=metadata
            )
            
            # Get output
            output = buffer.getvalue().strip()
            parsed = json.loads(output)
            
            assert parsed['message'] == 'Test message'
            assert parsed['metadata'] == metadata
            
        finally:
            logger.removeHandler(handler)
    
    def test_log_with_metadata_different_levels(self):
        """Test log_with_metadata with different log levels."""
        logger = logging.getLogger('test_levels_logger')
        logger.setLevel(logging.DEBUG)
        
        buffer = StringIO()
        handler = logging.StreamHandler(buffer)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        
        try:
            # Log at different levels
            log_with_metadata(logger, logging.INFO, 'Info message', {'type': 'info'})
            log_with_metadata(logger, logging.WARNING, 'Warning message', {'type': 'warn'})
            log_with_metadata(logger, logging.ERROR, 'Error message', {'type': 'error'})
            
            # Get output
            output = buffer.getvalue()
            lines = [line.strip() for line in output.split('\n') if line.strip()]
            
            assert len(lines) == 3
            
            # Parse and verify each line
            info_log = json.loads(lines[0])
            assert info_log['level'] == 'info'
            assert info_log['metadata']['type'] == 'info'
            
            warn_log = json.loads(lines[1])
            assert warn_log['level'] == 'warn'
            assert warn_log['metadata']['type'] == 'warn'
            
            error_log = json.loads(lines[2])
            assert error_log['level'] == 'error'
            assert error_log['metadata']['type'] == 'error'
            
        finally:
            logger.removeHandler(handler)


class TestLoggingRequirements:
    """Tests for specific logging requirements."""
    
    def test_requirement_9_1_tool_discovery_logging(self):
        """
        Test Requirement 9.1: Log tool discovery at startup.
        
        Verifies that tool discovery logs include:
        - upstream_id
        - tool_count
        """
        logger = logging.getLogger('test_discovery_logger')
        logger.setLevel(logging.INFO)
        
        buffer = StringIO()
        handler = logging.StreamHandler(buffer)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        
        try:
            # Simulate tool discovery logging
            log_with_metadata(
                logger,
                logging.INFO,
                "Discovered tools from upstream 'playwright'",
                metadata={
                    'upstream_id': 'playwright',
                    'tool_count': 15
                }
            )
            
            output = buffer.getvalue().strip()
            parsed = json.loads(output)
            
            assert 'upstream_id' in parsed['metadata']
            assert 'tool_count' in parsed['metadata']
            assert parsed['metadata']['upstream_id'] == 'playwright'
            assert parsed['metadata']['tool_count'] == 15
            
        finally:
            logger.removeHandler(handler)
    
    def test_requirement_9_2_search_tools_logging(self):
        """
        Test Requirement 9.2: Log search_tools requests.
        
        Verifies that search_tools logs include:
        - query
        - context_length
        - top_matches (list of tool names)
        """
        logger = logging.getLogger('test_search_logger')
        logger.setLevel(logging.INFO)
        
        buffer = StringIO()
        handler = logging.StreamHandler(buffer)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        
        try:
            # Simulate search_tools logging
            log_with_metadata(
                logger,
                logging.INFO,
                "search_tools request completed",
                metadata={
                    'query': 'test a web page',
                    'context_length': 2,
                    'top_matches': ['playwright.navigate', 'playwright.click', 'playwright.screenshot'],
                    'results_count': 10
                }
            )
            
            output = buffer.getvalue().strip()
            parsed = json.loads(output)
            
            assert 'query' in parsed['metadata']
            assert 'context_length' in parsed['metadata']
            assert 'top_matches' in parsed['metadata']
            assert parsed['metadata']['query'] == 'test a web page'
            assert parsed['metadata']['context_length'] == 2
            assert len(parsed['metadata']['top_matches']) == 3
            
        finally:
            logger.removeHandler(handler)
    
    def test_requirement_9_3_tools_list_logging(self):
        """
        Test Requirement 9.3: Log tools/list requests.
        
        Verifies that tools/list logs include:
        - tool_count (number of tools in default subset)
        """
        logger = logging.getLogger('test_list_logger')
        logger.setLevel(logging.INFO)
        
        buffer = StringIO()
        handler = logging.StreamHandler(buffer)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        
        try:
            # Simulate tools/list logging
            log_with_metadata(
                logger,
                logging.INFO,
                "Returning default tool subset",
                metadata={
                    'tool_count': 20,
                    'includes_search_tools': True
                }
            )
            
            output = buffer.getvalue().strip()
            parsed = json.loads(output)
            
            assert 'tool_count' in parsed['metadata']
            assert parsed['metadata']['tool_count'] == 20
            
        finally:
            logger.removeHandler(handler)
    
    def test_requirement_9_4_tool_call_logging_success(self):
        """
        Test Requirement 9.4: Log tool calls (success).
        
        Verifies that tool call logs include:
        - tool_name
        - upstream_id
        - status (success/failure)
        """
        logger = logging.getLogger('test_call_logger')
        logger.setLevel(logging.INFO)
        
        buffer = StringIO()
        handler = logging.StreamHandler(buffer)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        
        try:
            # Simulate successful tool call logging
            log_with_metadata(
                logger,
                logging.INFO,
                "Tool call succeeded: playwright.navigate",
                metadata={
                    'tool_name': 'playwright.navigate',
                    'upstream_id': 'playwright',
                    'status': 'success'
                }
            )
            
            output = buffer.getvalue().strip()
            parsed = json.loads(output)
            
            assert 'tool_name' in parsed['metadata']
            assert 'upstream_id' in parsed['metadata']
            assert 'status' in parsed['metadata']
            assert parsed['metadata']['status'] == 'success'
            
        finally:
            logger.removeHandler(handler)
    
    def test_requirement_9_4_tool_call_logging_failure(self):
        """
        Test Requirement 9.4: Log tool calls (failure).
        
        Verifies that failed tool call logs include error information.
        """
        logger = logging.getLogger('test_call_fail_logger')
        logger.setLevel(logging.ERROR)
        
        buffer = StringIO()
        handler = logging.StreamHandler(buffer)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        
        try:
            # Simulate failed tool call logging
            log_with_metadata(
                logger,
                logging.ERROR,
                "Tool call failed: playwright.navigate",
                metadata={
                    'tool_name': 'playwright.navigate',
                    'upstream_id': 'playwright',
                    'status': 'failure',
                    'error': 'Connection timeout'
                }
            )
            
            output = buffer.getvalue().strip()
            parsed = json.loads(output)
            
            assert parsed['metadata']['status'] == 'failure'
            assert 'error' in parsed['metadata']
            assert parsed['metadata']['error'] == 'Connection timeout'
            
        finally:
            logger.removeHandler(handler)
    
    def test_requirement_9_5_structured_json_format(self):
        """
        Test Requirement 9.5: Use structured JSON logging format.
        
        Verifies that all logs are valid JSON with consistent structure.
        """
        logger = logging.getLogger('test_json_logger')
        logger.setLevel(logging.INFO)
        
        buffer = StringIO()
        handler = logging.StreamHandler(buffer)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        
        try:
            # Log multiple messages
            logger.info('Simple message')
            log_with_metadata(logger, logging.INFO, 'Message with metadata', {'key': 'value'})
            
            output = buffer.getvalue()
            lines = [line.strip() for line in output.split('\n') if line.strip()]
            
            # All lines should be valid JSON
            for line in lines:
                parsed = json.loads(line)
                assert 'timestamp' in parsed
                assert 'level' in parsed
                assert 'component' in parsed
                assert 'message' in parsed
            
        finally:
            logger.removeHandler(handler)
    
    def test_requirement_9_6_timestamps_and_levels(self):
        """
        Test Requirement 9.6: Include timestamps and log levels.
        
        Verifies that all logs include:
        - timestamp (ISO 8601 format)
        - level (info, warn, error)
        """
        logger = logging.getLogger('test_timestamp_logger')
        logger.setLevel(logging.INFO)
        
        buffer = StringIO()
        handler = logging.StreamHandler(buffer)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        
        try:
            # Log at different levels
            logger.info('Info message')
            logger.warning('Warning message')
            logger.error('Error message')
            
            output = buffer.getvalue()
            lines = [line.strip() for line in output.split('\n') if line.strip()]
            
            expected_levels = ['info', 'warn', 'error']
            
            for line, expected_level in zip(lines, expected_levels):
                parsed = json.loads(line)
                
                # Verify timestamp exists and is valid ISO 8601
                assert 'timestamp' in parsed
                timestamp = datetime.fromisoformat(parsed['timestamp'].replace('Z', '+00:00'))
                assert isinstance(timestamp, datetime)
                
                # Verify level
                assert 'level' in parsed
                assert parsed['level'] == expected_level
            
        finally:
            logger.removeHandler(handler)
