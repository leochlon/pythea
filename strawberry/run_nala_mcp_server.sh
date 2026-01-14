#!/usr/bin/env bash
set -euo pipefail

# Convenience launcher for the research MCP server.
#
# Usage:
#   ./run_nala_mcp_server.sh                 # OpenAI backend (OPENAI_API_KEY)
#   ./run_nala_mcp_server.sh /path/pool.json # Azure OpenAI pool config

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python -m strawberry.nala_mcp_server "${1:-}" 
