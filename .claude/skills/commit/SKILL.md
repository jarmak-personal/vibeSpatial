---
name: commit
description: >
  PROACTIVELY USE THIS SKILL when the user says "commit", "land", "land this",
  "ship it", "done", "wrap up", "let's finish", or any intent to commit work.
  This is the ONLY user entrypoint for the commit workflow — do not invoke
  /pre-land-review directly for commits. Orchestrates the full landing flow:
  pre-land review, staging, review marker, and git commit.
user-invocable: true
argument-hint: "[optional commit message override]"
---

# Commit — Full Landing Flow

You are landing work. Follow these steps exactly in order. Do not skip any
step. Do not create a git commit without completing the review.

## Step 1: Run /pre-land-review

Invoke the `pre-land-review` skill. This runs:
- All deterministic checks (ruff, check_docs, architecture lints, zero-copy,
  perf patterns, maintainability)
- AI-powered sub-agent reviews (GPU code review, zero-copy enforcer,
  performance analysis, maintainability enforcer) as applicable

If the review finds BLOCKING issues, **stop here**. Fix them and re-run
`/commit`. Do not proceed to Step 2 with blocking findings.

## Step 2: Stage changes

After the review passes with verdict LAND:

1. Run `git status` to see what needs staging.
2. Run `git diff --cached --name-only` to see what is already staged.
3. Stage the appropriate files. Prefer staging specific files by name over
   `git add -A`. Never stage `.env`, credentials, or large binaries.
4. If the user specified which files to commit, stage only those.

## Step 3: Write the review marker

The `commit-msg` hook requires a content-addressable review marker. Write it:

```bash
printf '{\n  "timestamp": "%s",\n  "staged_hash": "%s",\n  "files": [%s],\n  "verdict": "LAND"\n}\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "$(git diff --cached | sha256sum | cut -d' ' -f1)" \
  "$(git diff --cached --name-only | sed 's/.*/"&"/' | paste -sd,)" \
  > .claude/.review-completed
```

**IMPORTANT**: If you stage any additional files AFTER writing the marker,
the hash will no longer match and the commit will be rejected. Always write
the marker as the LAST step before `git commit`.

## Step 4: Create the commit

1. Analyze the staged diff to draft a concise commit message.
2. If the user provided `$ARGUMENTS`, use that as the commit message (or
   incorporate it).
3. Create the commit using a HEREDOC for proper formatting:

```bash
git commit -m "$(cat <<'EOF'
<commit message here>

Co-Authored-By: Claude <co-author tag>
EOF
)"
```

4. Run `git status` after the commit to verify success.

## Step 5: Report

Tell the user the commit was created. Show the commit hash and summary.

## Rules

- NEVER skip the pre-land review. The commit-msg hook will reject the commit
  anyway if the marker is missing or stale.
- NEVER use `--no-verify` or `-n` flags. The pre-land-gate hook blocks these.
- If the pre-commit hook fails (ruff, docs, etc.), fix the issues and retry.
  Do NOT amend — create a new commit.
- The review marker is single-use: the commit-msg hook deletes it after a
  successful commit.
- If the commit fails for any reason, re-run `/commit` from the top.
