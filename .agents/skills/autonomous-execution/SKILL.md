---
name: autonomous-execution
description: "PROACTIVELY USE THIS SKILL when the user provides a PRD, spec, requirements doc, or tasklist and wants end-to-end execution with minimal check-ins. Trigger on: \"autonomous mode\", \"own this\", \"own this end-to-end\", \"enact this PRD\", \"implement all of this\", \"here are the requirements\", \"no check-ins unless blocked\", or similar. Treat the provided PRD or tasklist as the mandate. Execute through implementation, verification, profiling, docs, and landing when requested. Only interrupt for true external blockers such as missing secrets, required sandbox or network approval, destructive irreversible actions not already authorized, or contradictory requirements."
---

# Autonomous Execution

Follow this skill when the user wants execution, not supervision.

## Default Stance

- Treat the provided PRD, spec, or tasklist as authoritative.
- Own the work end-to-end instead of pausing for approval on local choices.
- Use repo priorities to break ties: correctness is non-negotiable,
  performance is king, UX is queen.

## Communication

- Do not stop to present plans, option menus, or preference questions.
- Send concise milestone updates only when they materially help the user
  track progress.
- If multiple valid implementations exist, choose the one with the best
  long-term architecture and performance shape.

## Only Interrupt For

- Missing credentials, secrets, or external access the agent cannot infer.
- Required sandbox or network approval.
- Destructive or irreversible actions the user did not explicitly request.
- Contradictory requirements that cannot be reconciled from the PRD and
  repo policy.

## Completion Standard

- Execute through implementation, verification, profiling, docs, and
  cleanup as needed.
- If the user asks to land the work, use `$commit` and finish the full
  landing flow, including push.
