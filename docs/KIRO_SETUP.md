# MCP Client Configuration for Semantic Router

This file shows how to configure an MCP client (like Kiro, Claude Desktop, Cursor, etc.) to use the MCP Semantic Router.

## Configuration

Add this to your MCP client's configuration file:

```json
{
  "mcpServers": {
    "semantic-router": {
      "command": "python",
      "args": [
        "-m",
        "mcp_router",
        "/path/to/mcp-router/config.json"
      ],
      "env": {},
      "disabled": false,
      "autoApprove": [
        "search_tools"
      ]
    }
  }
}
```

## Configuration Locations by Client

### Kiro
- **Location**: `~/.kiro/settings/mcp.json`
- **Windows**: `C:\Users\<username>\.kiro\settings\mcp.json`
- **macOS/Linux**: `~/.kiro/settings/mcp.json`

### Claude Desktop
- **Location**: `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

### Cursor
- **Location**: `~/.cursor/mcp.json`

## Setup Steps

1. **Install the router**:
   ```bash
   cd /path/to/mcp-router
   pip install -r requirements.txt
   ```

2. **Create router config** (`config.json`):
   ```json
   {
     "mcpServers": {
       "playwright": {
         "transport": "stdio",
         "command": "npx",
         "args": ["-y", "@playwright/mcp@latest"],
         "semantic_prefix": "browser",
         "category_description": "Browser automation tools"
       }
     }
   }
   ```

3. **Update client config** with the JSON above (adjust path to your config.json)

4. **Restart your MCP client**

## How It Works

```
┌──────────────┐
│  MCP Client  │
│ (Kiro/Claude)│
└──────┬───────┘
       │
       │ (connects to semantic-router)
       │
       ▼
┌─────────────────────────┐
│  MCP Semantic Router    │
│  (your proxy)           │
└────┬──────────────┬─────┘
     │              │
     │              │ (routes tool calls)
     │              │
     ▼              ▼
┌──────────┐  ┌──────────┐
│Upstream  │  │Upstream  │
│  MCP 1   │  │  MCP 2   │
└──────────┘  └──────────┘
```

## Tool Namespacing

All tools are namespaced by their upstream server:

- `<prefix>.<tool_name>` - Tools from upstream servers
- `search_tools` - Semantic search tool (provided by router)

Example:
- `browser.browser_navigate` - Navigate to URL (from Playwright)
- `jira.searchIssues` - Search Jira issues (from Atlassian)

## New Semantic Search Tool

The router adds: **`search_tools`**

Use natural language to find relevant tools:

**Example queries:**
- "find issues in Jira"
- "navigate to a website"
- "take a screenshot"

## Auto-Approved Tools

The `autoApprove` array lists tools that don't require user confirmation.

**Recommended minimum**:
```json
"autoApprove": [
  "search_tools"
]
```

**Optional** (add specific tools you trust):
```json
"autoApprove": [
  "search_tools",
  "browser.browser_navigate",
  "jira.searchIssues"
]
```

## Testing the Configuration

1. **Restart your MCP client**
2. **Check server status** - Verify `semantic-router` is connected
3. **List tools** - Should see tools from all upstream servers
4. **Try semantic search** - Use natural language to find tools

## Troubleshooting

### Router Not Starting

Check:
1. Python is installed and in PATH
2. Dependencies installed: `pip install -r requirements.txt`
3. Config file path is correct
4. Router config (`config.json`) is valid JSON

### Tools Not Appearing

Check:
1. Upstream servers are accessible
2. Router logs for connection errors
3. Router config has valid upstream definitions

### Test Manually

```bash
cd /path/to/mcp-router
python -m mcp_router config.json
```

Should see: "MCP Semantic Router started successfully"

## Benefits

1. **Semantic Search** - Find tools using natural language
2. **Unified Interface** - All tools through one MCP server
3. **Smart Routing** - Automatic forwarding to correct upstream
4. **Namespacing** - Clear prefixes prevent conflicts
5. **Extensibility** - Easy to add more upstream servers

## Adding More Upstream Servers

Edit your router's `config.json`:

```json
{
  "mcpServers": {
    "existing-server": { ... },
    "new-server": {
      "transport": "stdio",
      "command": "your-command",
      "args": ["arg1", "arg2"],
      "semantic_prefix": "prefix",
      "category_description": "Description"
    }
  }
}
```

Restart your MCP client to reload.

## Support

For issues:
1. Check router logs
2. Run tests: `pytest tests/ -v`
3. See README.md for detailed documentation
