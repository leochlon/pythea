# nala-mcp Usage Guide

## What is nala-mcp?

**nala-mcp** (registered as `pythea` in Claude Code) is an MCP server for **autonomous research and Lean proof work** where Strawberry's hallucination/citation checker is the **enforcement layer**, not a "nice to have".

## Key Difference from Regular Workflow

| Regular Workflow | nala-mcp Workflow |
|-----------------|-------------------|
| think -> run tool -> justify later | claim -> audit -> permission -> action -> **check-in** -> evidence |
| Plan invalidated: Never | Plan invalidated: After EVERY tool run |
| Verification: Optional | Verification: REQUIRED before action |
| Check-in: Not required | Check-in: REQUIRED after every execution |

## When to Use nala-mcp

Use nala-mcp for:
- **Research workflows** requiring verified evidence chains
- **Lean/formal proof work** with mathematical rigor
- **Literature review** where citations must be grounded
- **High-stakes debugging** requiring evidence discipline

Do NOT use nala-mcp for:
- Regular coding tasks
- Quick file edits
- Simple questions
- Tasks where evidence overhead isn't justified

## Core Guarantees

### 1. Evidence Spans are the Unit of Truth
Everything the agent learns is stored as a **span** `S0`, `S1`, ... with provenance (where it came from).

### 2. No Plan, No Action
All evidence-generation tools require a `plan_step_idx`, and that step must come from an approved microplan.

### 3. Re-audit After Every Attempt
After any tool run (Lean build, command, fetch, PDF extract, etc.), the server **clears the microplan** and logs the attempt. The agent must create and audit a new microplan to continue.

### 4. Check-in Required After Execution
After executing any tool, you MUST call `checkin_last_action` before creating a new microplan. This acknowledges the execution result and unlocks the state machine. Skipping check-in causes a RuntimeError.

## Quick Start

### 1. Start a Research Run

```
Use mcp__pythea__start_run with:
- problem_statement: "Your research goal"
- slug: "short-name-for-run"
```

This creates a run directory at `./research_runs/<timestamp>-<slug>/` with:
- `state.json` - Run metadata, spans, attempts
- `spans/S#.txt` - Evidence span files
- `files/` - Downloaded files

### 2. Add Evidence Spans

```
Use mcp__pythea__add_span with:
- text: "The evidence text"
- kind: "note" | "file" | "cmd" | "web"
- source: "Where this came from"
```

Or capture file content:
```
Use mcp__pythea__add_file_span with:
- path: "/path/to/file"
- start_line: 10
- end_line: 50
```

### 3. Create an Audited Microplan

```
Use mcp__pythea__set_microplan with steps:
[
  {
    "idx": 0,
    "claim": "Search arXiv for papers on X",
    "cites": ["S0"],
    "confidence": 0.95
  },
  {
    "idx": 1,
    "claim": "Download and extract key findings",
    "cites": ["S0"]
  }
]
```

If Strawberry flags any claim (insufficient evidence), the plan is **rejected**.

### 4. Execute Plan Steps

Every execution tool requires `plan_step_idx`:

```
Use mcp__pythea__arxiv_search with:
- query: "your search query"
- plan_step_idx: 0
```

**After execution:**
- Output stored as new span (e.g., `S5`)
- Microplan is CLEARED
- **MUST call `checkin_last_action` before continuing**
- Then create new microplan to continue

## Available Tools

### Detection (No Plan Required)
- `audit_trace_budget` - Verify claims against spans
- `detect_hallucination` - Auto-extract and verify claims

### Run Lifecycle
- `start_run` - Create new research run
- `load_run` - Load existing run

### Evidence Spans
- `list_spans` - List span metadata
- `get_span` - Get span content
- `add_span` - Add text span
- `add_file_span` - Capture file excerpt
- `search_spans` - TF-IDF search over spans

### Planning
- `set_microplan` - Create audited microplan
- `get_microplan` - Get current plan
- `clear_microplan` - Clear plan manually

### Check-in (Required After Execution)
- `checkin_last_action` - **CRITICAL**: Acknowledge execution result, unlocks state machine
- `get_pending_checkin` - Check if check-in is pending (for debugging)
- `list_attempts` - Review execution history and attempt logs

### Execution (Requires `plan_step_idx`)
- `run_cmd` - Run shell command
- `rg_search` - Ripgrep search
- `python_run` - Execute Python script
- `cpp_run` - Compile and run C++
- `lean_build` - Run `lake build`
- `lean_query` - Run `lake env lean`
- `arxiv_search` - Search arXiv
- `fetch_url` - Fetch web content
- `download_url` - Download file
- `pdf_extract` - Extract PDF text
- `arxiv_download_source` - Download arXiv source

## Example: Literature Research Workflow

### Step 1: Start Run
```
mcp__pythea__start_run:
  problem_statement: "Survey computational verification approaches for Erdos-Straus"
  slug: "erdos-straus-survey"
```

### Step 2: Add Context Span
```
mcp__pythea__add_span:
  text: "Looking for papers on sieve methods and computational bounds"
  kind: "note"
  source: "research goal"
```

### Step 3: Create Microplan
```
mcp__pythea__set_microplan:
  steps: [
    {"idx": 0, "claim": "Search arXiv for computational verification papers", "cites": ["S0", "S1"]}
  ]
```

### Step 4: Execute Search
```
mcp__pythea__arxiv_search:
  query: "Erdos Straus computational verification"
  plan_step_idx: 0
```

### Step 5: Check-in (REQUIRED)
**After execution completes, you MUST check-in before continuing:**
```
mcp__pythea__checkin_last_action:
  notes: "Found 15 results, top match is paper on sieve methods"
```

This acknowledges the execution result and unlocks the state machine.

### Step 6: Plan Invalidated - Create New Plan
After check-in, the plan is cleared. You now have new evidence (S2 with search results). Create a new plan:

```
mcp__pythea__set_microplan:
  steps: [
    {"idx": 0, "claim": "Download top result PDF for detailed review", "cites": ["S2"]}
  ]
```

### Step 7: Continue...

This forces verification at every step - you can't "run ahead" without citing your evidence.

## Troubleshooting

### "No active microplan" Error
You tried to run an execution tool without an approved plan. Create and verify a microplan first with `set_microplan`.

### "Pending check-in" Error
You tried to create a new microplan without checking in after the last execution. Call `checkin_last_action` first to acknowledge the previous result.

### "Plan step index out of range" Error
The `plan_step_idx` doesn't match a valid step in your microplan.

### "Microplan verification failed" Error
Strawberry flagged one or more claims. Either:
1. Add more evidence spans to support the claims
2. Revise claims to match available evidence
3. Downgrade claims to hypotheses

### Server Not Responding
```bash
# Re-register the server (from project root)
claude mcp add pythea \
  -e OPENAI_API_KEY=$OPENAI_API_KEY -- \
  $(pwd)/.venv/bin/python -m strawberry.nala_mcp_server
```

## Integration with Existing Workflow

nala-mcp is **separate** from the regular Claude Code workflow:

- **Regular work**: Use standard agents (planner-agent, task-md-manager, etc.)
- **Research mode**: Explicitly start a nala-mcp run when you need evidence discipline

The two workflows don't interfere - nala-mcp runs are isolated in `./research_runs/` directories.
