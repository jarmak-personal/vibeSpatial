#!/bin/sh
# PreToolUse hook (Bash tool): blocks enforcement bypasses and reminds
# about /pre-land-review before commits.
#
# Hard-blocks:
#   - git commit --no-verify / -n            (hook bypass)
#   - git config core.hooksPath              (hook redirect)
#   - git -c core.hooksPath=...              (inline hook bypass)
#   - Destructive ops on .githooks/, .claude/hooks/, .claude/settings*.json
#
# Soft gate:
#   - Any git commit -> systemMessage reminder about /pre-land-review
#
# Input:  JSON on stdin  {command: "..."}
# Output: JSON on stdout {"decision":"block","reason":"..."} |
#                         {"systemMessage":"..."} | {}

input=$(cat)
if ! command -v python3 >/dev/null 2>&1; then
    printf '{"systemMessage": "WARNING: pre-land-gate hook requires python3. Enforcement checks may be inactive."}'
    exit 0
fi

printf '%s' "$input" | python3 -c '
import sys, re, json


def block(reason):
    print(json.dumps({"decision": "block", "reason": "BLOCKED: " + reason}))
    sys.exit(0)


def remind(msg):
    print(json.dumps({"systemMessage": msg}))
    sys.exit(0)


def main():
    try:
        cmd = json.load(sys.stdin).get("command", "")
    except Exception:
        return "{}"

    is_git_commit = bool(re.search(r"\bgit\b.*\bcommit\b", cmd))

    # ---- Hard blocks -------------------------------------------------------

    # 1. git commit --no-verify / -n  (skips all hooks)
    if is_git_commit:
        if re.search(r"--no-verify", cmd):
            block(
                "--no-verify is prohibited. Pre-commit and commit-msg hooks "
                "are mandatory. Run /pre-land-review first, then commit "
                "without --no-verify."
            )
        # Detect short -n flag.  Strip -m/--message args first so we do not
        # match a literal "-n" inside a commit message string.
        SQ = "\x27"  # single-quote (avoids shell quoting issues)
        stripped = re.sub(
            r"-[a-zA-Z]*m\s*(?:\"[^\"]*\"|"
            + SQ + "[^" + SQ + "]*" + SQ +
            r"|\S+)",
            "", cmd,
        )
        stripped = re.sub(
            r"--message\s*=?\s*(?:\"[^\"]*\"|"
            + SQ + "[^" + SQ + "]*" + SQ +
            r"|\S+)",
            "", stripped,
        )
        if re.search(r"(?:^|\s)-n(?:\s|$)", stripped):
            block(
                "-n (--no-verify) is prohibited. Hooks are mandatory. "
                "Run /pre-land-review first, then commit without -n."
            )

    # 2. git config core.hooksPath  (redirects hooks away from .githooks/)
    if re.search(r"\bgit\b.*\bconfig\b.*\bcore\.hookspath\b", cmd, re.I):
        block(
            "Changing core.hooksPath is prohibited. "
            "Hooks must run from .githooks/."
        )

    # 3. Inline -c core.hooksPath override
    if re.search(r"\bgit\b.*-c\s.*\bcore\.hookspath\b", cmd, re.I):
        block(
            "Inline core.hooksPath override (-c core.hooksPath) "
            "is prohibited."
        )

    # 4. Destructive operations on enforcement infrastructure
    PROTECTED = {
        r"\.githooks/": ".githooks/",
        r"\.claude/hooks/": ".claude/hooks/",
        r"\.claude/settings\.json": ".claude/settings.json",
        r"\.claude/settings\.local\.json": ".claude/settings.local.json",
    }
    DESTRUCTIVE = (
        r"\b(rm|mv|cp|chmod|chown|truncate|sed|perl|awk"
        r"|tee|dd|ln|install|patch)\b"
    )
    for pat, label in PROTECTED.items():
        if re.search(pat, cmd):
            if re.search(DESTRUCTIVE, cmd):
                block(
                    f"Modifying enforcement file ({label}) via shell "
                    "is prohibited. Edit manually if needed."
                )
            if re.search(r">\s*\S*" + pat, cmd):
                block(
                    f"Redirect to enforcement file ({label}) "
                    "is prohibited."
                )

    # ---- Soft gate ----------------------------------------------------------

    # 5. Any git commit -> remind about /pre-land-review
    if is_git_commit:
        remind(
            "MANDATORY: Before creating this commit, you must have "
            "completed the /pre-land-review checklist. If you have not "
            "run it in this session, invoke the pre-land-review skill "
            "NOW before proceeding. The checklist includes: "
            "(1) all deterministic checks pass, "
            "(2) AI performance analysis for kernel/pipeline changes, "
            "(3) AI zero-copy analysis for runtime changes, "
            "(4) AI maintainability analysis for non-test changes. "
            "Test-only changes need only deterministic checks."
        )

    return "{}"


try:
    print(main())
except Exception:
    print("{}")
'
