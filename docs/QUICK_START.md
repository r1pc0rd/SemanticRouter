# MCP Semantic Router - Quick Start

## 🚀 Getting Started

Configure your MCP client to use the MCP Semantic Router and unlock semantic tool search!

## What You Get

### 🔍 Semantic Tool Search
Find tools using natural language instead of exact names:
```
"Find tools for searching Jira"
"How do I navigate to a webpage?"
"Show me screenshot tools"
```

### 🎯 Namespaced Tools
All tools have clear prefixes to avoid conflicts:
- `<prefix>.*` - Tools from upstream MCP servers
- `search_tools` - Semantic search (provided by router)

### 🔧 Example Configuration

The router can proxy any MCP servers. Here's an example with Atlassian and Playwright:

**Atlassian Tools** (prefix: `atlassian.`)
- `atlassian.searchJiraIssuesUsingJql` - Search Jira with JQL
- `atlassian.getJiraIssue` - Get issue details
- `atlassian.addCommentToJiraIssue` - Add comments
- `atlassian.getVisibleJiraProjects` - List projects

**Browser Tools** (prefix: `browser.`)
- `browser.browser_navigate` - Navigate to URL
- `browser.browser_click` - Click elements
- `browser.browser_snapshot` - Take screenshots
- `browser.browser_close` - Close browser

**Search Tool**
- `search_tools` - Semantic search for tools

## Quick Test

After configuring and restarting your MCP client:

1. **List tools**: Ask your client to show available tools
2. **Search**: "Find tools for Jira" or "Show me browser tools"
3. **Execute**: Try calling a tool through the router

## Configuration Files

| File | Purpose |
|------|---------|
| `config.json` | Router configuration (upstream servers) |
| `config.example.json` | Example router configuration |
| `kiro-mcp-config.json` | Example MCP client configuration |

## Setup Steps

1. **Install dependencies**:
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

3. **Configure your MCP client** (see `KIRO_SETUP.md` for details)

4. **Restart your MCP client**

## Troubleshooting

### Router not starting?
```bash
# Test manually
cd /path/to/mcp-router
python -m mcp_router config.json
```

Should output: "MCP Semantic Router started successfully"

### Need to check configuration?
- Verify `config.json` is valid JSON
- Check that Python and dependencies are installed
- Review logs for connection errors

## Documentation

- **Client Setup Guide**: `KIRO_SETUP.md` - How to configure MCP clients
- **Project README**: `README.md` - Complete documentation
- **Configuration**: `docs/configuration.md` - Detailed config options

## Architecture

```
MCP Client → Semantic Router → [Upstream MCP 1, Upstream MCP 2, ...]
```

The router acts as a proxy, adding semantic search and namespacing to all upstream tools.

## Next Steps

1. ✅ Install dependencies
2. ✅ Create router configuration
3. ✅ Configure MCP client
4. 🔄 Restart MCP client
5. ✅ Test tool discovery
6. ✅ Try semantic search
7. ✅ Execute tools

---

**Ready to enhance your MCP experience!** 🎉
