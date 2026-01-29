# MCP Semantic Router

A semantic search-powered MCP router that intelligently discovers and ranks tools from multiple upstream MCP servers.

## Overview

The MCP Semantic Router solves the tool overload problem when working with multiple MCP servers (40-50+ tools). It uses local embeddings and semantic search to dynamically filter and rank tools based on query relevance, exposing only the most relevant 5-10 tools per request.

### Key Features

- **Semantic Tool Discovery**: Custom `search_tools` method for intelligent tool ranking
- **Local Embeddings**: Uses sentence-transformers (all-MiniLM-L6-v2) - no external API calls
- **Generic & Portable**: Works with any MCP client (Claude Desktop, Cursor, Kiro, etc.)
- **Transparent Proxying**: Tool calls are forwarded to upstream servers without modification
- **Dual Discovery**: Standard `tools/list` (20 default tools) + semantic `search_tools` (top 10 ranked)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd mcp-router

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e ".[dev]"
```

## Quick Start

1. **Create a configuration file** (`config.json`):

```json
{
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
```

2. **Run the router**:

```bash
python -m mcp_router config.json
```

3. **Connect your MCP client** to the router via stdio.

## Usage

### Standard Tool Discovery

The router exposes a standard `tools/list` method that returns 20 default tools:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list",
  "params": {}
}
```

### Semantic Tool Search

Use the custom `search_tools` method for intelligent tool discovery:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "search_tools",
  "params": {
    "query": "test a web page",
    "context": ["user wants to automate browser testing"]
  }
}
```

Returns top 10 semantically matched tools with similarity scores.

### Tool Execution

Call tools through the router using namespaced names:

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "browser.navigate",
    "arguments": {
      "url": "https://example.com"
    }
  }
}
```

## Architecture

```
MCP Client → MCP Router → Upstream MCPs
                ↓
         Semantic Search
         (Local Embeddings)
```

The router:
1. Discovers tools from configured upstream MCPs at startup
2. Generates embeddings for all tools using sentence-transformers
3. Provides semantic search via `search_tools` method
4. Proxies tool calls to appropriate upstream servers

## Development

### Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# Property tests only
pytest tests/property/

# With coverage
pytest --cov=mcp_router --cov-report=html
```

### Code Quality

```bash
# Format code
black src/ tests/

# Type checking
mypy src/

# Linting
ruff check src/ tests/
```

## Project Structure

```
mcp-router/
├── src/mcp_router/     # Source code
│   ├── core/           # Config, models, logging
│   ├── discovery/      # Tool discovery
│   ├── embedding/      # Embedding engine
│   ├── search/         # Semantic search
│   ├── proxy/          # Tool call proxying
│   └── server/         # MCP server
├── tests/              # Test suite
│   ├── unit/           # Unit tests
│   ├── integration/    # Integration tests
│   └── property/       # Property-based tests
├── docs/               # Documentation
└── temp/               # Temporary work files
```

## Configuration

See `config.example.json` for a complete configuration example.

### Configuration Options

- `transport`: Communication protocol (`stdio`, `sse`, `http`)
- `command`: Command to start upstream MCP server
- `args`: Arguments for the command
- `semantic_prefix`: Optional prefix for tool namespacing
- `category_description`: Optional description for embedding context

## Requirements

- Python 3.10+
- sentence-transformers
- numpy
- mcp (official MCP SDK)

## License

MIT

## Contributing

Contributions are welcome! Please see the spec documents in `.kiro/specs/mcp-semantic-router/` for design details and implementation tasks.
