# nala-mcp

An MCP server for **autonomous research + Lean proof work** where Strawberry’s hallucination/citation checker is the **enforcement layer**, not a “nice to have”.

This server is intended to replace prompt-only “skills” with a **server-side state machine**:

- The agent must first create an **audited microplan** (atomic claims + evidence span citations).
- Only then can it invoke execution / fetching / ingestion tools.
- **After every execution**, the plan is invalidated, forcing the agent to re-audit its next decision using updated evidence.

---

## Why this exists

Most agent setups look like:

> think → run tool → justify later

`nala-mcp` forces:

> **claim → audit → permission → action → evidence → (plan invalidated) → claim**

So “hallucination detection” is not optional—it's the gatekeeper for any meaningful action.

---

## Core guarantees

### Evidence spans are the unit of truth
Everything the agent learns is stored as a **span** `S0`, `S1`, … with provenance (where it came from).

### No plan, no action
All evidence-generation tools require a `plan_step_idx`, and that step must come from an approved microplan.

### Re-audit after every attempt
After any tool run (Lean build, command, fetch, PDF extract, etc.), the server **clears the microplan** and logs the attempt. The agent must create and audit a new microplan to continue.

---

## Quickstart

### Requirements
- Python 3.10+ recommended
- `mcp` SDK installed (the server imports `mcp.server.fastmcp.FastMCP`)
- Strawberry dependencies (OpenAI SDK, requests, pypdf, etc.)

### Install (repo / editable)
From the `strawberry/` directory:

```bash
pip install -e ".[mcp]"
```

or (if you prefer `uv`):

```bash
uv pip install -e ".[mcp]"
```

### Backend configuration

The server supports two verifier backends (same as Strawberry):

#### Option A: OpenAI API
Set:

```bash
export OPENAI_API_KEY="..."
```

#### Option B: Azure OpenAI pool
Set:

```bash
export AOAI_POOL_JSON="/path/to/pool.json"
```

or pass the pool path as argv[1] to the server entrypoint.

### Run the server

From `strawberry/`:

```bash
python -m strawberry.nala_mcp_server
```

or use the included helper script:

```bash
./run_nala_mcp_server.sh
# or:
./run_nala_mcp_server.sh /path/to/pool.json
```

---

## Claude Code configuration (tool-locked)

You “force Claude Code to use only this server” by:

1) Exposing only this MCP server in `.mcp.json` (or managed config)
2) Denying all built-in tools in `.claude/settings.json`, while allowing only `mcp__<server>__*`

### `.mcp.json` (project-scoped)

Create this at your repo root:

```json
{
  "mcpServers": {
    "pythea": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "strawberry.nala_mcp_server"],
      "env": {
        "OPENAI_API_KEY": "${OPENAI_API_KEY}"
      }
    }
  }
}
```

Notes:
- The key `"pythea"` is the MCP server name **as Claude Code will see it**.
- Tool names will appear as `mcp__pythea__start_run`, `mcp__pythea__set_microplan`, etc.

### `.claude/settings.json` (project-scoped permissions)

Create:

```json
{
  "permissions": {
    "defaultMode": "dontAsk",
    "allow": [
      "mcp__pythea__*"
    ],
    "deny": [
      "Bash",
      "Read",
      "Edit",
      "Write",
      "Grep",
      "Glob",
      "WebFetch",
      "WebSearch",
      "Task"
    ]
  },
  "enableAllProjectMcpServers": true
}
```

This makes Claude Code:
- auto-deny any built-in tools
- only allow MCP tools from your server

> If you need machine-wide enforcement (so users can’t add other MCP servers), use **managed** configuration (`managed-mcp.json`, `managed-settings.json`).

---

## Run directories & persistence

By default `start_run` creates a new run under:

```
./research_runs/<timestamp>-<slug>/
```

Layout:

- `state.json` — run metadata, span list, attempt ledger, microplan, plan_version
- `spans/S#.txt` — each evidence span as text
- `files/` — downloaded files (PDFs, tarballs, etc.)
- `scripts/` — generated Python/C++ scripts (from `python_run` / `cpp_run`)
- `tmp_lean/` — temporary Lean query files (from `lean_query`)

---

## The workflow you should use

### 1) Start a run

Call:

- `start_run(problem_statement=..., slug=...)`

This creates the run directory and seeds `S0` with a **Project Charter** span describing the rules.

### 2) Add evidence

Use:
- `add_span` for manual notes / extracted snippets
- `add_file_span` to capture a slice of a local file into a span
- `fetch_url`, `download_url`, `pdf_extract`, `arxiv_search`, `arxiv_download_source` for literature ingestion (gated)

### 3) Propose an audited microplan

Call:
- `set_microplan(steps=[...])`

Each step is an object with:

- `idx` (int) — step number (sorted by idx)
- `claim` (string) — atomic next action / hypothesis
- `cites` (list of span ids) — e.g. `["S0", "S3"]`
- `confidence` (float, optional) — default target confidence for that step

Example:

```json
[
  {
    "idx": 0,
    "claim": "Search arXiv for recent computational verification and sieve/filter papers on Erdős–Straus and save the results as evidence.",
    "cites": ["S0"],
    "confidence": 0.95
  },
  {
    "idx": 1,
    "claim": "Download the most relevant PDF from the search results and extract pages containing the main theorem/verification bound.",
    "cites": ["S0"]
  }
]
```

If the hallucination checker flags the plan (insufficient evidence), `set_microplan` raises and nothing can proceed.

### 4) Execute exactly one plan step

Execution tools require `plan_step_idx`:

- `arxiv_search(..., plan_step_idx=0)`
- `python_run(..., plan_step_idx=1)`
- `lean_build(..., plan_step_idx=2)`

**After the tool runs:**
- output is stored as a new span (e.g., `S7`)
- the plan is invalidated (cleared)
- an attempt record is appended to `state.json`

### 5) Re-plan and repeat
Because the plan is cleared after every run, the agent must call `set_microplan` again before doing anything else.

---

## Security model

### Command allowlist
`run_cmd` restricts which binaries can be invoked unless you explicitly allow unsafe commands.

Allowlisted binaries include (see code for full list):

- `rg`, `grep`, `sed`, `awk`, `cat`, `ls`, `find`
- `python`, `python3`
- `git`
- `lake`, `lean`
- `g++`, `clang++`, `make`, `cmake`, `ninja`, `cargo`
- `wget`, `curl`
- `bash`, `sh` (⚠️ very powerful — see note below)

To allow non-allowlisted commands:
- set `RESEARCH_MCP_ALLOW_UNSAFE=1` **or**
- pass `allow_unsafe=True` to `run_cmd`

**Important caution:** `bash`/`sh` are allowlisted, which means a user could run arbitrary shell via `bash -lc ...`.
If you want tighter security for untrusted environments, remove `bash` and `sh` from the allowlist in `_require_allowed_cmd`.

---

## Tool reference

### Detection

#### `audit_trace_budget(steps, ...) -> report`
Runs Strawberry’s trace-budget audit on a list of plan steps using the **current** span set.

#### `detect_hallucination(answer, ...) -> report`
Runs Strawberry’s claim-level hallucination detector on a narrative answer, using current spans.

---

### Run lifecycle

#### `start_run(problem_statement, root_dir="./research_runs", slug=None, seed_charter=True)`
Creates a new run directory and seeds `S0`.

Returns: `{ run_dir, seed_span }`

#### `load_run(run_dir)`
Loads an existing run (reads `state.json`).

---

### Evidence spans

#### `list_spans(limit=50, offset=0)`
Returns span metadata only.

#### `get_span(sid, max_chars=20000)`
Returns metadata + text excerpt.

#### `add_span(text, kind="note", source="manual")`
Adds a text span.

#### `add_file_span(path, start_line, end_line, kind="file", source=None)`
Captures a local file excerpt as a span.

#### `search_spans(query, top_k=5)`
Lightweight TF‑IDF search over span texts. Returns span ids + scores.

---

### Planning

#### `set_microplan(steps, verifier_model="gpt-4o-mini", default_target=0.95, units="bits")`
Audits and stores a microplan. Required before any evidence-generation tools.

#### `get_microplan()`
Returns current plan (if any).

#### `clear_microplan()`
Manually clears the plan.

---

### Evidence generation (all require `plan_step_idx`)

#### `run_cmd(cmd: [str], plan_step_idx: int, cwd=None, timeout_s=60, allow_unsafe=False)`
Runs a local command, stores stdout/stderr + exit code as a `cmd` span, invalidates plan.

#### `rg_search(pattern, path, plan_step_idx, timeout_s=30)`
Convenience wrapper for ripgrep: `rg -n pattern path`.

#### `python_run(code, plan_step_idx, args=None, timeout_s=120)`
Writes a script under `scripts/`, runs it, stores output as evidence.

#### `cpp_run(code, plan_step_idx, args=None, cxx="g++", cxxflags=["-O2","-std=c++20"], timeout_s=120)`
Compiles and runs C++ once, storing source + build + run logs as one evidence span.

---

### Lean tools

#### `lean_build(project_dir, plan_step_idx, timeout_s=600)`
Runs `lake build` in a Lean project.

#### `lean_query(project_dir, imports, commands, plan_step_idx, timeout_s=120)`
Creates a temporary `.lean` file under `tmp_lean/` and runs `lake env lean <file>`.
Useful for `#check`, `#find`, `#print`, and small proof-state experiments.

---

### Web + papers

#### `arxiv_search(query, plan_step_idx, max_results=10)`
Searches arXiv via Atom API. Stores results JSON as a span.

#### `fetch_url(url, plan_step_idx, strip_html=True, timeout_s=30, max_bytes=5_000_000)`
Fetches a URL and stores text (optionally stripped of HTML tags).

#### `download_url(url, plan_step_idx, filename=None, timeout_s=60, max_bytes=50_000_000)`
Downloads a file into `files/` and stores a metadata span.

#### `pdf_extract(path, plan_step_idx, page_start=1, page_end=1, max_chars=200_000)`
Extracts text from a PDF using `pypdf` and stores it as a span.

#### `arxiv_download_source(arxiv_id, plan_step_idx)`
Downloads arXiv “e-print” source tarball, extracts into `files/`, stores metadata as a span.

---

## Demo recipes

### Demo A: “Zero-trust literature brief”
1. `start_run("Erdos–Straus: modern computational verification + sieve/filter approach", slug="es-brief")`
2. `set_microplan` with steps:
   - search arXiv for key sieve/verification papers
   - download 1–2 PDFs
   - extract theorem/result pages
3. `arxiv_search` (plan_step_idx=0)
4. `set_microplan` again (because plan cleared)
5. `download_url` + `pdf_extract`
6. `detect_hallucination` on the generated brief to show it’s grounded

### Demo B: “Lean check loop (plan-gated)”
1. `start_run("Lean micro-proof demo", slug="lean-demo")`
2. `set_microplan`: “Run lean_query with #check of candidate lemma names; adjust based on output.”
3. `lean_query` (plan_step_idx=0)
4. New microplan based on the Lean output span.

---

## Notes / limitations

- `pdf_extract` is **text extraction**, not OCR. Scanned PDFs may extract poorly.
- `fetch_url` uses a minimal HTML stripper; it won’t preserve math formatting like MathJax.
- This server does not automatically judge *truth*—it enforces **evidence discipline** so the agent cannot outrun what it can cite.

---

## Versioning

- `strawberry` package: `0.2.0`
- MCP server name: `"nala-mcp"` (client-facing name is whatever you set in `.mcp.json`, e.g. `"pythea"`)

