#!/bin/sh
# PreToolUse hook: fires when Claude attempts to run git commit.
# Injects a system message reminding about the pre-land review checklist.
#
# This is a safety net for long conversations where the skill trigger
# might not fire due to context compression.
#
# Input: JSON on stdin with {tool_name, command, ...}
# Output: JSON on stdout with {systemMessage} or {}

# Read stdin (tool call details).
input=$(cat)

# Extract the bash command being run.
command=$(printf '%s' "$input" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('command', ''))
except Exception:
    pass
" 2>/dev/null)

# Only trigger on git commit commands.
case "$command" in
    *"git commit"*)
        cat <<'HOOK_EOF'
{"systemMessage": "MANDATORY: Before creating this commit, you must have completed the /pre-land-review checklist. If you have not run it in this session, invoke the pre-land-review skill NOW before proceeding with the commit. The checklist includes: (1) all deterministic checks pass, (2) AI performance analysis for kernel/pipeline changes, (3) AI zero-copy analysis for runtime changes, (4) AI maintainability analysis for non-test changes. Test-only changes need only deterministic checks."}
HOOK_EOF
        ;;
    *)
        printf '{}'
        ;;
esac
