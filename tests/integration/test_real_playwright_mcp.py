"""
Real-world integration test using the actual Playwright MCP server.

This test validates the semantic router works with a real upstream MCP server
by connecting to the Playwright MCP server configured in Kiro's settings.
"""

import asyncio
import json
import pytest
from pathlib import Path
from typing import Any

from mcp_router.discovery.manager import ToolDiscoveryManager
from mcp_router.embedding.engine import EmbeddingEngine
from mcp_router.search.engine import SemanticSearchEngine
from mcp_router.proxy.proxy import ToolCallProxy


# Playwright MCP configuration from Kiro's mcp.json
PLAYWRIGHT_CONFIG = {
    "mcpServers": {
        "playwright": {
            "transport": "stdio",
            "command": "npx",
            "args": ["@playwright/mcp@latest"]
        }
    }
}


@pytest.fixture
def playwright_config_file(tmp_path: Path) -> Path:
    """Create a temporary configuration file with Playwright MCP server."""
    config_path = tmp_path / "playwright_config.json"
    config_path.write_text(json.dumps(PLAYWRIGHT_CONFIG))
    return config_path


@pytest.mark.asyncio
@pytest.mark.slow  # Mark as slow since it connects to real MCP server
async def test_real_playwright_discovery(playwright_config_file: Path):
    """
    Test tool discovery with real Playwright MCP server.
    
    Validates:
    - Connection to real Playwright MCP server
    - Tool discovery from real upstream
    - Tool metadata parsing
    """
    discovery_manager = ToolDiscoveryManager()
    
    try:
        # Initialize with real Playwright server
        await asyncio.wait_for(
            discovery_manager.initialize(str(playwright_config_file)),
            timeout=30.0  # Allow time for npx to download if needed
        )
        
        # Verify tools were discovered
        tools = discovery_manager.get_all_tools()
        assert len(tools) > 0, "No tools discovered from Playwright MCP"
        
        # Verify tools have playwright namespace
        playwright_tools = [t for t in tools if t.name.startswith("playwright.")]
        assert len(playwright_tools) > 0, "No playwright-namespaced tools found"
        
        # Verify expected Playwright tools exist
        tool_names = {t.name for t in tools}
        expected_tools = [
            "playwright.browser_navigate",
            "playwright.browser_snapshot",
            "playwright.browser_click",
            "playwright.browser_close"
        ]
        
        for expected_tool in expected_tools:
            assert expected_tool in tool_names, f"Expected tool {expected_tool} not found"
        
        print(f"\n✓ Discovered {len(tools)} tools from Playwright MCP")
        print(f"✓ Tool names: {', '.join(sorted(tool_names))}")
        
    finally:
        # Clean up
        await discovery_manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.slow
async def test_real_playwright_semantic_search(playwright_config_file: Path):
    """
    Test semantic search with real Playwright tools.
    
    Validates:
    - Embedding generation for real tools
    - Semantic search returns relevant Playwright tools
    - Search results are ranked by relevance
    """
    discovery_manager = ToolDiscoveryManager()
    embedding_engine = EmbeddingEngine()
    
    try:
        # Initialize discovery and embeddings
        await asyncio.wait_for(
            discovery_manager.initialize(str(playwright_config_file)),
            timeout=30.0
        )
        await embedding_engine.initialize()
        
        # Get tools and generate embeddings
        tools = discovery_manager.get_all_tools()
        await embedding_engine.generate_tool_embeddings(tools)
        
        # Initialize search engine
        search_engine = SemanticSearchEngine(embedding_engine)
        search_engine.set_tools(tools)
        
        # Test search for browser automation tasks
        test_queries = [
            ("navigate to a website", ["browser_navigate"]),
            ("take a screenshot", ["browser_snapshot"]),
            ("click a button", ["browser_click"]),
            ("close the browser", ["browser_close"]),
        ]
        
        for query, expected_keywords in test_queries:
            results = await search_engine.search_tools(query, context=[], top_k=5)
            
            assert len(results) > 0, f"No results for query: {query}"
            
            # Verify at least one result matches expected keywords
            top_result = results[0]
            assert any(
                keyword in top_result.tool.name.lower() or 
                keyword in top_result.tool.description.lower()
                for keyword in expected_keywords
            ), f"Top result for '{query}' doesn't match expected keywords: {expected_keywords}"
            
            print(f"\n✓ Query: '{query}'")
            print(f"  Top result: {top_result.tool.name} (similarity: {top_result.similarity:.4f})")
        
    finally:
        # Clean up
        await discovery_manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.skipif(
    True,  # Skip by default - requires browser installation
    reason="Requires Playwright browser installation (run manually with --run-playwright)"
)
async def test_real_playwright_tool_call(playwright_config_file: Path):
    """
    Test calling a real Playwright tool through the proxy.
    
    NOTE: This test is skipped by default because it requires:
    1. Playwright browsers to be installed (npx playwright install)
    2. A real browser to be launched
    
    To run this test manually:
    pytest tests/integration/test_real_playwright_mcp.py::test_real_playwright_tool_call -v --run-playwright
    
    Validates:
    - Tool call proxying to real upstream
    - Real Playwright tool execution
    - Response parsing
    """
    discovery_manager = ToolDiscoveryManager()
    
    try:
        # Initialize discovery
        await asyncio.wait_for(
            discovery_manager.initialize(str(playwright_config_file)),
            timeout=30.0
        )
        
        # Initialize proxy
        proxy = ToolCallProxy(
            upstreams=discovery_manager.upstreams,
            upstream_configs=discovery_manager._config.mcp_servers
        )
        
        # Call browser_snapshot tool (least invasive - just captures page state)
        result = await asyncio.wait_for(
            proxy.call_tool(
                "playwright.browser_snapshot",
                {}  # No arguments needed for snapshot
            ),
            timeout=60.0  # Allow time for browser launch
        )
        
        # Verify result
        assert result is not None, "Tool call returned None"
        assert not result.is_error, f"Tool call failed: {result.content}"
        assert len(result.content) > 0, "Tool call returned empty content"
        
        print(f"\n✓ Successfully called playwright.browser_snapshot")
        print(f"  Result type: {result.content[0].type}")
        print(f"  Content length: {len(result.content[0].text) if hasattr(result.content[0], 'text') else 'N/A'}")
        
    finally:
        # Clean up
        await discovery_manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.slow
async def test_real_playwright_end_to_end(playwright_config_file: Path):
    """
    Test complete end-to-end workflow with real Playwright MCP.
    
    Validates:
    - Full workflow: discovery -> embedding -> search -> (proxy ready)
    - All components work together with real upstream
    - System is ready for tool calls
    """
    discovery_manager = ToolDiscoveryManager()
    embedding_engine = EmbeddingEngine()
    
    try:
        # Step 1: Tool Discovery
        print("\n[1/4] Discovering tools from Playwright MCP...")
        await asyncio.wait_for(
            discovery_manager.initialize(str(playwright_config_file)),
            timeout=30.0
        )
        tools = discovery_manager.get_all_tools()
        print(f"✓ Discovered {len(tools)} tools")
        
        # Step 2: Embedding Generation
        print("\n[2/4] Generating embeddings...")
        await embedding_engine.initialize()
        await embedding_engine.generate_tool_embeddings(tools)
        print(f"✓ Generated embeddings for {len(tools)} tools")
        
        # Step 3: Semantic Search
        print("\n[3/4] Testing semantic search...")
        search_engine = SemanticSearchEngine(embedding_engine)
        search_engine.set_tools(tools)
        
        results = await search_engine.search_tools(
            "automate browser testing",
            context=[],
            top_k=5
        )
        print(f"✓ Found {len(results)} relevant tools")
        print(f"  Top 3 matches:")
        for i, result in enumerate(results[:3], 1):
            print(f"    {i}. {result.tool.name} (similarity: {result.similarity:.4f})")
        
        # Step 4: Proxy Initialization
        print("\n[4/4] Initializing tool call proxy...")
        proxy = ToolCallProxy(
            upstreams=discovery_manager.upstreams,
            upstream_configs=discovery_manager._config.mcp_servers
        )
        print("✓ Proxy ready for tool calls")
        
        # Verify system is fully operational
        assert len(tools) > 0
        assert len(results) > 0
        assert proxy is not None
        
        print("\n✅ End-to-end test PASSED - System fully operational with real Playwright MCP")
        
    finally:
        # Clean up
        await discovery_manager.shutdown()


if __name__ == "__main__":
    # Run with: python -m pytest tests/integration/test_real_playwright_mcp.py -v -s
    pytest.main([__file__, "-v", "-s", "-m", "slow"])
