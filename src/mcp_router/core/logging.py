"""Structured logging infrastructure for MCP Semantic Router.

This module provides JSON-formatted structured logging with timestamps,
log levels, and component identification.
"""

from typing import Literal, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import logging


@dataclass
class LogEntry:
    """Structured log entry with JSON serialization.
    
    Attributes:
        level: Log level (info, warn, error)
        component: Component name that generated the log
        message: Human-readable log message
        metadata: Optional additional structured data
        timestamp: ISO 8601 formatted timestamp (auto-generated if not provided)
    """
    level: Literal['info', 'warn', 'error']
    component: str
    message: str
    metadata: Optional[dict[str, Any]] = None
    timestamp: str = None  # ISO 8601, auto-generated
    
    def __post_init__(self):
        """Auto-generate timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()
    
    def to_json(self) -> str:
        """Convert log entry to JSON string.
        
        Returns:
            JSON string representation of the log entry
        """
        return json.dumps(asdict(self))


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs structured JSON logs.
    
    Converts Python logging records into structured JSON format
    using the LogEntry dataclass.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as JSON.
        
        Args:
            record: Python logging record
            
        Returns:
            JSON-formatted log string
        """
        # Map Python log levels to our log levels
        level_map = {
            logging.INFO: 'info',
            logging.WARNING: 'warn',
            logging.ERROR: 'error',
            logging.CRITICAL: 'error',
            logging.DEBUG: 'info'
        }
        
        level = level_map.get(record.levelno, 'info')
        
        # Extract metadata from record if present
        metadata = None
        if hasattr(record, 'metadata'):
            metadata = record.metadata
        
        # Create structured log entry
        log_entry = LogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=level,
            component=record.name,
            message=record.getMessage(),
            metadata=metadata
        )
        
        return log_entry.to_json()


def setup_logging(level: int = logging.INFO) -> None:
    """Configure structured JSON logging for the application.
    
    Sets up the root logger with JSON formatting and the specified
    log level. All loggers in the application will inherit this
    configuration.
    
    Args:
        level: Python logging level (default: logging.INFO)
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler with JSON formatter
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(JSONFormatter())
    
    # Add handler to root logger
    root_logger.addHandler(console_handler)


def log_with_metadata(
    logger: logging.Logger,
    level: int,
    message: str,
    metadata: Optional[dict[str, Any]] = None
) -> None:
    """Log a message with structured metadata.
    
    Helper function to attach metadata to log records for JSON formatting.
    
    Args:
        logger: Logger instance to use
        level: Python logging level
        message: Log message
        metadata: Optional structured metadata dictionary
    """
    extra = {'metadata': metadata} if metadata is not None else {}
    logger.log(level, message, extra=extra)
