#!/bin/bash
# Run the hallucination detector MCP server
#
# Usage (OpenAI API - default):
#   export OPENAI_API_KEY=sk-...
#   ./run_mcp_server.sh
#
# Usage (Azure OpenAI Pool):
#   ./run_mcp_server.sh /path/to/aoai_pool.json
#
# Or set the environment variable:
#   export AOAI_POOL_JSON=/path/to/aoai_pool.json
#   ./run_mcp_server.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH:-}"

# Use first argument as pool JSON path if provided
if [ -n "$1" ]; then
    export AOAI_POOL_JSON="$1"
fi

# If pool JSON is set, validate it exists
if [ -n "$AOAI_POOL_JSON" ]; then
    if [ ! -f "$AOAI_POOL_JSON" ]; then
        echo "Error: Pool config file not found: $AOAI_POOL_JSON" >&2
        exit 1
    fi
    exec python3 -m strawberry.mcp_server "$AOAI_POOL_JSON"
else
    # No pool config - will use OpenAI API (requires OPENAI_API_KEY)
    exec python3 -m strawberry.mcp_server
fi
