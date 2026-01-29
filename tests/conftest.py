"""
Pytest configuration and shared fixtures for MCP Semantic Router tests.
"""
import pytest
import asyncio
from typing import Generator


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_config_dict() -> dict:
    """Sample configuration dictionary for testing."""
    return {
        "mcpServers": {
            "playwright": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-playwright"],
                "semantic_prefix": "browser",
                "category_description": "Web browser automation and testing tools"
            },
            "atlassian": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-atlassian"],
                "semantic_prefix": "jira",
                "category_description": "Issue tracking and project management tools"
            }
        }
    }
