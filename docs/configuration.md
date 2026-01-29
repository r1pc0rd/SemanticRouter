# Configuration Guide

This guide explains how to configure the MCP Semantic Router to connect to upstream MCP servers.

## Configuration File Format

The router uses a JSON configuration file that defines upstream MCP servers. The file must contain a top-level `mcpServers` object with one or more server configurations.

### Basic Structure

```json
{
  "mcpServers": {
    "server-id": {
      "transport": "stdio",
      "command": "command-to-start-server",
      "args": ["arg1", "arg2"],
      "semantic_prefix": "optional-prefix",
      "category_description": "optional description"
    }
  }
}
```

## Configuration Fields

### Required Fields

#### `transport` (string)
The communication protocol for the upstream server.

**Supported values:**
- `"stdio"` - Standard input/output (most common)
- `"sse"` - Server-Sent Events
- `"http"` - HTTP transport

**Example:**
```json
"transport": "stdio"
```

#### `command` (string)
The command to execute to start the upstream MCP server.

**Examples:**
```json
"command": "npx"
"command": "python"
"command": "node"
"command": "/usr/local/bin/my-mcp-server"
```

#### `args` (array of strings)
Arguments to pass to the command.

**Examples:**
```json
"args": ["-y", "@modelcontextprotocol/server-playwright"]
"args": ["server.py", "--port", "8080"]
"args": ["/path/to/allowed/directory"]
```

### Optional Fields

#### `semantic_prefix` (string)
Override the default namespace prefix for tools from this upstream.

**Default behavior:** If not specified, the server ID is used as the prefix.

**Use case:** Provide more intuitive or descriptive tool names.

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

With this configuration:
- Tool `navigate` becomes `browser.navigate` (instead of `playwright.navigate`)
- Tool `click` becomes `browser.click` (instead of `playwright.click`)

#### `category_description` (string)
A description of the tool category that is included in the embedding text for semantic search.

**Use case:** Improve semantic search accuracy by providing context about what the tools do.

**Example:**
```json
"category_description": "Web browser automation and testing tools for navigating pages, clicking elements, filling forms, and capturing screenshots"
```

This description is combined with each tool's name, description, and parameters when generating embeddings, helping the semantic search engine better understand the tool's purpose.

## Complete Examples

### Example 1: Playwright Browser Automation

```json
{
  "mcpServers": {
    "playwright": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-playwright"],
      "semantic_prefix": "browser",
      "category_description": "Web browser automation and testing tools for navigating pages, clicking elements, filling forms, and capturing screenshots"
    }
  }
}
```

### Example 2: Multiple Upstreams

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
    },
    "filesystem": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/documents"],
      "category_description": "File system operations for reading and writing files"
    }
  }
}
```

### Example 3: Custom Python Server

```json
{
  "mcpServers": {
    "my-custom-server": {
      "transport": "stdio",
      "command": "python",
      "args": ["/path/to/my_server.py"],
      "semantic_prefix": "custom",
      "category_description": "Custom tools for specific business logic"
    }
  }
}
```

## Configuration Validation

The router validates the configuration at startup and will fail with an error if:

1. **Missing required fields**: `transport`, `command`, or `args` are not provided
2. **Invalid JSON**: The configuration file is not valid JSON
3. **File not found**: The configuration file path is incorrect
4. **Invalid transport**: The transport type is not supported

### Error Examples

**Missing required field:**
```
ConfigurationError: Missing required field 'command' for upstream 'playwright'
```

**Invalid JSON:**
```
ConfigurationError: Failed to parse config.json: Expecting property name enclosed in double quotes
```

**File not found:**
```
ConfigurationError: Configuration file not found: config.json
```

## Best Practices

### 1. Use Descriptive Server IDs
Choose server IDs that clearly indicate the upstream's purpose:

✅ Good:
```json
"playwright": { ... }
"atlassian": { ... }
"filesystem": { ... }
```

❌ Avoid:
```json
"server1": { ... }
"mcp-server": { ... }
"upstream": { ... }
```

### 2. Provide Semantic Prefixes
Use semantic prefixes to make tool names more intuitive:

✅ Good:
```json
"playwright": {
  "semantic_prefix": "browser",
  ...
}
```

Result: `browser.navigate`, `browser.click`

❌ Without prefix:
Result: `playwright.navigate`, `playwright.click`

### 3. Write Detailed Category Descriptions
Include comprehensive category descriptions to improve semantic search:

✅ Good:
```json
"category_description": "Web browser automation and testing tools for navigating pages, clicking elements, filling forms, capturing screenshots, and testing web applications"
```

❌ Too brief:
```json
"category_description": "Browser tools"
```

### 4. Use Absolute Paths for Local Servers
When running local MCP servers, use absolute paths:

✅ Good:
```json
"command": "/usr/local/bin/my-mcp-server"
"args": ["/home/user/data"]
```

❌ Avoid relative paths:
```json
"command": "./my-mcp-server"
"args": ["../data"]
```

### 5. Test Configuration Before Deployment
Always test your configuration with a single upstream first:

```json
{
  "mcpServers": {
    "test-server": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-playwright"]
    }
  }
}
```

Then gradually add more upstreams.

## Troubleshooting

### Upstream Connection Failures

If an upstream fails to connect, the router will:
1. Log an error with details
2. Continue with remaining upstreams
3. Exclude failed upstream's tools from the catalog

**Check the logs:**
```json
{
  "timestamp": "2025-01-15T10:00:05.000Z",
  "level": "error",
  "component": "ToolDiscoveryManager",
  "message": "Failed to connect to upstream",
  "metadata": {
    "upstreamId": "playwright",
    "error": "Command not found: npx"
  }
}
```

**Common causes:**
- Command not installed (`npx`, `python`, etc.)
- Incorrect command path
- Missing dependencies for the upstream server
- Incorrect arguments

### Configuration File Not Found

**Error:**
```
ConfigurationError: Configuration file not found: config.json
```

**Solution:**
Provide the full path to the configuration file:
```bash
python -m mcp_router /path/to/config.json
```

### Invalid JSON Syntax

**Error:**
```
ConfigurationError: Failed to parse config.json: Expecting property name enclosed in double quotes
```

**Solution:**
Validate your JSON using a JSON validator or linter. Common issues:
- Missing commas between fields
- Trailing commas (not allowed in JSON)
- Unquoted keys or values
- Incorrect quote types (use `"` not `'`)

## Environment-Specific Configurations

### Development
```json
{
  "mcpServers": {
    "local-test": {
      "transport": "stdio",
      "command": "python",
      "args": ["./dev/test_server.py"],
      "semantic_prefix": "test"
    }
  }
}
```

### Production
```json
{
  "mcpServers": {
    "playwright": {
      "transport": "stdio",
      "command": "/usr/local/bin/npx",
      "args": ["-y", "@modelcontextprotocol/server-playwright"],
      "semantic_prefix": "browser",
      "category_description": "Web browser automation and testing tools"
    },
    "atlassian": {
      "transport": "stdio",
      "command": "/usr/local/bin/npx",
      "args": ["-y", "@modelcontextprotocol/server-atlassian"],
      "semantic_prefix": "jira",
      "category_description": "Issue tracking and project management tools"
    }
  }
}
```

## See Also

- [README.md](../README.md) - Quick start guide
- [Usage Guide](usage.md) - How to use the router
- [Architecture](architecture.md) - System design and components
