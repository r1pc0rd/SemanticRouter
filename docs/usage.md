# Usage Guide

This guide explains how to use the MCP Semantic Router to discover and execute tools from multiple upstream MCP servers.

## Starting the Router

### Basic Usage

```bash
python -m mcp_router config.json
```

The router will:
1. Load the configuration file
2. Connect to all configured upstream MCP servers
3. Discover tools from each upstream
4. Generate embeddings for semantic search
5. Start the MCP server on stdio

### Startup Logs

You'll see structured JSON logs during startup:

```json
{"timestamp": "2025-01-15T10:00:00.000Z", "level": "info", "component": "EmbeddingEngine", "message": "Loading embedding model", "metadata": {"model": "sentence-transformers/all-MiniLM-L6-v2"}}
{"timestamp": "2025-01-15T10:00:02.000Z", "level": "info", "component": "ToolDiscoveryManager", "message": "Connecting to upstream", "metadata": {"upstreamId": "playwright"}}
{"timestamp": "2025-01-15T10:00:03.000Z", "level": "info", "component": "ToolDiscoveryManager", "message": "Discovered tools from upstream", "metadata": {"upstreamId": "playwright", "toolCount": 15}}
{"timestamp": "2025-01-15T10:00:05.000Z", "level": "info", "component": "SemanticRouterServer", "message": "MCP server started", "metadata": {"toolCount": 42}}
```

## Tool Discovery Methods

The router provides two methods for discovering tools:

### 1. Standard Tool List (`tools/list`)

Returns a default subset of 20 tools, selected to provide diversity across all upstream servers.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list",
  "params": {}
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "tools": [
      {
        "name": "browser.navigate",
        "description": "Navigate to a URL",
        "inputSchema": {
          "type": "object",
          "properties": {
            "url": {
              "type": "string",
              "description": "The URL to navigate to"
            }
          },
          "required": ["url"]
        }
      },
      // ... 19 more tools
    ]
  }
}
```

**Use case:** Initial tool discovery when you don't have a specific query.

### 2. Semantic Search (`search_tools`)

Returns the top 10 tools most relevant to your query, ranked by semantic similarity.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "search_tools",
    "arguments": {
      "query": "test a web page",
      "context": ["user wants to automate browser testing"]
    }
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "[{\"name\": \"browser.navigate\", \"description\": \"Navigate to a URL\", \"similarity\": 0.87}, {\"name\": \"browser.click\", \"description\": \"Click an element\", \"similarity\": 0.82}, ...]"
      }
    ]
  }
}
```

**Parameters:**
- `query` (required): The search query describing what you want to do
- `context` (optional): Array of context strings to refine the search

**Use case:** Finding specific tools when you know what task you want to accomplish.

## Executing Tools

### Basic Tool Call

Call a tool using its namespaced name:

**Request:**
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

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Navigated to https://example.com"
      }
    ]
  }
}
```

### Tool Namespacing

Tools are namespaced using the format: `<prefix>.<tool_name>`

The prefix is determined by:
1. The `semantic_prefix` in the configuration (if provided)
2. The upstream server ID (if no semantic_prefix)

**Example:**
```json
{
  "mcpServers": {
    "playwright": {
      "semantic_prefix": "browser",
      ...
    }
  }
}
```

- Tool `navigate` becomes `browser.navigate`
- Tool `click` becomes `browser.click`

## Semantic Search Examples

### Example 1: Web Automation

**Query:** "test a web page"

**Context:** ["user wants to automate browser testing"]

**Expected Results:**
1. `browser.navigate` - Navigate to a URL
2. `browser.click` - Click an element
3. `browser.screenshot` - Take a screenshot
4. `browser.fill` - Fill a form field
5. `browser.evaluate` - Execute JavaScript

### Example 2: Issue Tracking

**Query:** "create a bug report"

**Context:** ["user found a bug in the application"]

**Expected Results:**
1. `jira.create_issue` - Create a new issue
2. `jira.add_comment` - Add a comment to an issue
3. `jira.update_issue` - Update an issue
4. `jira.search_issues` - Search for issues
5. `jira.get_issue` - Get issue details

### Example 3: File Operations

**Query:** "read a configuration file"

**Context:** ["user needs to parse JSON config"]

**Expected Results:**
1. `filesystem.read_file` - Read file contents
2. `filesystem.list_directory` - List directory contents
3. `filesystem.get_file_info` - Get file metadata
4. `filesystem.search_files` - Search for files
5. `filesystem.read_multiple_files` - Read multiple files

## Error Handling

### Invalid Tool Name

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "tools/call",
  "params": {
    "name": "unknown.tool",
    "arguments": {}
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "error": {
    "code": -32601,
    "message": "Tool not found: unknown.tool",
    "data": {
      "toolName": "unknown.tool",
      "reason": "Unknown upstream prefix 'unknown'"
    }
  }
}
```

### Missing Required Parameter

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "method": "tools/call",
  "params": {
    "name": "search_tools",
    "arguments": {}
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "error": {
    "code": -32602,
    "message": "Invalid params: query parameter is required",
    "data": {
      "parameter": "query",
      "received": null
    }
  }
}
```

### Upstream Timeout

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 6,
  "method": "tools/call",
  "params": {
    "name": "browser.navigate",
    "arguments": {
      "url": "https://slow-site.com"
    }
  }
}
```

**Response (after 30 seconds):**
```json
{
  "jsonrpc": "2.0",
  "id": 6,
  "error": {
    "code": -32000,
    "message": "Upstream timeout after 30 seconds",
    "data": {
      "toolName": "browser.navigate",
      "upstreamId": "playwright"
    }
  }
}
```

### Upstream Error

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "method": "tools/call",
  "params": {
    "name": "browser.navigate",
    "arguments": {
      "url": "invalid-url"
    }
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "error": {
    "code": -32603,
    "message": "Invalid URL format",
    "data": {
      "toolName": "browser.navigate",
      "upstreamId": "playwright",
      "upstreamError": "URL must start with http:// or https://"
    }
  }
}
```

## Integration with MCP Clients

### Claude Desktop

Add the router to your Claude Desktop configuration:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`

**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "semantic-router": {
      "command": "python",
      "args": ["-m", "mcp_router", "/path/to/config.json"]
    }
  }
}
```

### Cursor

Add the router to your Cursor MCP configuration:

```json
{
  "mcpServers": {
    "semantic-router": {
      "command": "python",
      "args": ["-m", "mcp_router", "/path/to/config.json"]
    }
  }
}
```

### Kiro

Add the router to your Kiro MCP configuration:

`.kiro/settings/mcp.json`:

```json
{
  "mcpServers": {
    "semantic-router": {
      "command": "python",
      "args": ["-m", "mcp_router", "/path/to/config.json"]
    }
  }
}
```

### Custom MCP Client

Connect to the router via stdio:

```python
import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "mcp_router", "config.json"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # List tools
            tools = await session.list_tools()
            print(f"Available tools: {len(tools.tools)}")
            
            # Search for tools
            result = await session.call_tool(
                "search_tools",
                arguments={"query": "test a web page"}
            )
            print(f"Search results: {result.content}")
            
            # Call a tool
            result = await session.call_tool(
                "browser.navigate",
                arguments={"url": "https://example.com"}
            )
            print(f"Tool result: {result.content}")

asyncio.run(main())
```

## Best Practices

### 1. Use Semantic Search for Specific Tasks

When you know what you want to do, use `search_tools` instead of `tools/list`:

✅ Good:
```json
{
  "method": "tools/call",
  "params": {
    "name": "search_tools",
    "arguments": {
      "query": "create a Jira issue for a bug"
    }
  }
}
```

❌ Less efficient:
```json
{
  "method": "tools/list"
}
// Then manually search through 20 tools
```

### 2. Provide Context for Better Results

Include context to improve semantic search accuracy:

✅ Good:
```json
{
  "query": "test a web page",
  "context": [
    "user wants to automate browser testing",
    "need to verify login functionality"
  ]
}
```

❌ Less accurate:
```json
{
  "query": "test"
}
```

### 3. Use Descriptive Queries

Write queries that describe the task, not just keywords:

✅ Good:
```json
{
  "query": "navigate to a website and click the login button"
}
```

❌ Too vague:
```json
{
  "query": "click"
}
```

### 4. Handle Errors Gracefully

Always check for errors in responses:

```python
response = await session.call_tool(name, arguments)

if "error" in response:
    error = response["error"]
    print(f"Error {error['code']}: {error['message']}")
    if "data" in error:
        print(f"Details: {error['data']}")
else:
    result = response["result"]
    # Process result
```

### 5. Monitor Logs

Watch the router's logs to understand what's happening:

```bash
python -m mcp_router config.json 2>&1 | tee router.log
```

Then analyze the logs:
```bash
# Search queries
grep "search_tools" router.log

# Tool calls
grep "tools/call" router.log

# Errors
grep "error" router.log
```

## Troubleshooting

### No Tools Returned

**Symptom:** `tools/list` returns an empty array

**Causes:**
1. No upstream servers configured
2. All upstream connections failed
3. Configuration file not found

**Solution:**
Check the logs for connection errors and verify your configuration.

### Search Returns Irrelevant Tools

**Symptom:** `search_tools` returns tools that don't match the query

**Causes:**
1. Query is too vague
2. Missing context
3. Category descriptions not provided in configuration

**Solution:**
1. Make your query more specific
2. Add context strings
3. Add `category_description` to your upstream configurations

### Tool Call Fails

**Symptom:** `tools/call` returns an error

**Causes:**
1. Invalid tool name or namespace
2. Missing required parameters
3. Upstream server error
4. Upstream timeout

**Solution:**
1. Verify the tool name using `tools/list` or `search_tools`
2. Check the tool's `inputSchema` for required parameters
3. Check upstream server logs
4. Increase timeout if needed (requires code modification)

## See Also

- [Configuration Guide](configuration.md) - How to configure the router
- [README.md](../README.md) - Quick start guide
- [Architecture](architecture.md) - System design and components
