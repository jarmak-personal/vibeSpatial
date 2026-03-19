#!/bin/sh
# PreToolUse hook (Edit/Write tools): protects enforcement-critical files
# from modification by AI agents.
#
# Protected paths:
#   .githooks/*                              (git hook scripts)
#   .claude/hooks/*                          (Claude hook scripts)
#   .claude/settings.json                    (hook registrations)
#   .claude/settings.local.json              (local hook overrides)
#   .claude/skills/pre-land-review/SKILL.md  (review skill definition)
#
# Returns JSON: {"decision":"block","reason":"..."} | {}

input=$(cat)
if ! command -v python3 >/dev/null 2>&1; then
    printf '{"systemMessage": "WARNING: file-guard hook requires python3. Enforcement file protection may be inactive."}'
    exit 0
fi

printf '%s' "$input" | python3 -c '
import sys, json, os, subprocess


def main():
    try:
        file_path = json.load(sys.stdin).get("file_path", "")
    except Exception:
        return "{}"

    if not file_path:
        return "{}"

    # Normalize to repo-relative path.
    try:
        repo_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        repo_root = os.getcwd()

    rel = file_path
    if file_path.startswith(repo_root + "/"):
        rel = file_path[len(repo_root) + 1:]

    PROTECTED = {
        ".githooks/": (
            "Git hooks (.githooks/) are enforcement infrastructure and "
            "cannot be modified by AI agents. Edit manually if needed."
        ),
        ".claude/hooks/": (
            "Claude hooks (.claude/hooks/) are enforcement infrastructure "
            "and cannot be modified by AI agents. Edit manually if needed."
        ),
        ".claude/settings.json": (
            ".claude/settings.json contains mandatory hook registrations "
            "and cannot be modified by AI agents. Edit manually if needed."
        ),
        ".claude/settings.local.json": (
            ".claude/settings.local.json may contain hook overrides and "
            "cannot be modified by AI agents. Edit manually if needed."
        ),
        ".claude/skills/pre-land-review/SKILL.md": (
            "The pre-land-review skill definition cannot be modified by "
            "AI agents. Edit manually if needed."
        ),
    }

    for prefix, reason in PROTECTED.items():
        if rel == prefix.rstrip("/") or rel.startswith(prefix):
            return json.dumps({"decision": "block", "reason": "BLOCKED: " + reason})

    return "{}"


try:
    print(main())
except Exception:
    print("{}")
'
