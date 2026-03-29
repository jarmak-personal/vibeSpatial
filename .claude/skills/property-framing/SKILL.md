---
name: property-framing
description: "PROACTIVELY USE THIS SKILL before delegating any work to engineer subagents (cuda-engineer, python-engineer) or before starting implementation yourself. Translates goal-oriented user requests ('improve performance of X', 'fix bug in Y', 'add feature Z') into property-convergence briefs that ground the work in measurable codebase properties. Trigger on: any implementation task, bug fix, feature request, performance improvement, refactor — essentially any time you are about to write code or delegate to an engineer. Do NOT trigger on: pure questions, exploration, brainstorming, or commit/review workflows."
user-invocable: true
argument-hint: <the user's goal-oriented request, e.g. "improve spatial_join performance">
---

# Property Framing — Translation Layer

You are translating a goal-oriented request into a property-convergence brief.
The user speaks in goals ("fix X", "improve Y"). Engineers should receive work
framed as: "here is the current property landscape, here is the goal, here is
what progress looks like."

User's request: **$ARGUMENTS**

---

## Step 1: Snapshot the property dashboard

Run the property dashboard to get the current state of all codebase properties:

```bash
uv run python scripts/property_dashboard.py --json
```

Record the output. This is the **before** snapshot.

## Step 2: Identify relevant properties

Map the user's request to the properties it should advance. Use this mapping:

| Request type | Primary properties | Secondary properties |
|---|---|---|
| Performance improvement | VPAT, ZCOPY | IGRD |
| Bug fix | ARCH | ZCOPY, VPAT |
| New GPU kernel/feature | ARCH, IGRD, VPAT | ZCOPY, MAINT |
| Refactor | ZCOPY, VPAT, IGRD | MAINT |
| API addition | ARCH, MAINT | ZCOPY |
| Test improvement | ARCH | MAINT |

If the request doesn't map to any existing property, note that in the brief —
the engineer may need to propose a new property.

## Step 3: Run intake routing

Use the intake router to find the relevant files and documentation:

```bash
uv run python scripts/intake.py "<user's request>"
```

## Step 4: Produce the property-convergence brief

Output the brief in this exact format. This is what you (or the engineer
subagent) will work from:

```
## Property-Convergence Brief

### Goal
<1-2 sentence restatement of what the user wants, in terms of what
should be TRUE about the codebase when we're done>

### Current Property State
<paste the human-readable dashboard output>

### Target Properties
<list the specific properties this work should advance, with current
distance and what "closer" looks like>

Primary:
- PROPERTY_CODE property_name (d=X.XX) — what advancing this looks like for this task

Secondary:
- PROPERTY_CODE property_name (d=X.XX) — what advancing this looks like for this task

### Relevant Files
<from intake routing>

### Success Criteria
This work is successful when:
1. The user's stated goal is achieved
2. At least one primary property distance decreases
3. No property distance increases
4. `uv run python scripts/property_dashboard.py` shows improvement

### What Does NOT Count as Success
- A fix that passes tests but advances no property (workaround)
- A change that advances one property but regresses another
- A change that "works" but adds new violations to any ratchet baseline
```

## Step 5: Save the before snapshot

Write the before snapshot to `.claude/.property-before.json`:

```bash
uv run python scripts/property_dashboard.py --json > .claude/.property-before.json
```

This will be compared to the after snapshot during `/pre-land-review`.

---

## How This Fits the Workflow

```
User: "improve spatial_join performance"
  │
  ▼
/property-framing    ◄── YOU ARE HERE: translate goal → property brief
  │
  ▼
/intake-router       ◄── find relevant files
  │
  ▼
Engineer subagent    ◄── receives the property-convergence brief, not just "improve X"
  │
  ▼
/commit              ◄── /pre-land-review compares before/after property state
```

The main agent should invoke this skill BEFORE spawning any engineer subagent.
The brief becomes the engineer's task description.

## Rules

- ALWAYS run the dashboard. Never skip the snapshot.
- ALWAYS identify which properties the work should advance. If none map,
  say so explicitly — this means either the request is too vague or a new
  property is needed.
- NEVER frame work as "complete this task." Frame it as "advance these
  properties while achieving the user's goal."
- The brief must be CONCRETE — "reduce VPAT violations in spatial_join.py"
  not "improve performance."
