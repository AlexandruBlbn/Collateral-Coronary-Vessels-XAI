#!/usr/bin/env bash
# Launch opencode with websearch enabled for this project
export OPENCODE_ENABLE_EXA=1
exec opencode --mcp mcp.json "$@"