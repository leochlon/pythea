# Strawberry Toolkit - Updates

## Overview

This document describes the updates made to integrate the strawberry-toolkit with an Azure OpenAI pool backend and expose it as an MCP server for Claude Code.

---

## New Files Added

### `src/strawberry/aoai_pool_backend.py`

A new backend adapter that wraps the Azure OpenAI pool (`AoaiPool`) to work with the toolkit's interface.

**Key features:**
- Load balances across multiple Azure OpenAI endpoints
- Supports logprobs extraction from Azure Chat Completions API
- Thread-safe connection pooling
- Compatible with the existing `BackendConfig` / `make_backend()` pattern

**Usage:**
```python
from strawberry.backend import BackendConfig, make_backend

cfg = BackendConfig(
    kind="aoai_pool",
    aoai_pool_json_path="/path/to/aoai_pool.json",
    max_concurrency=8,
)
backend = make_backend(cfg)
```

### `src/strawberry/aoai_pool_python.py`

The Azure OpenAI pool client (copied from external source). Provides:
- Weighted round-robin load balancing
- Automatic failover and cooldown
- Rate limit tracking (RPM/TPM)
- Key Vault integration for secrets

### `src/strawberry/mcp_server.py`

MCP server exposing two tools for hallucination detection:

1. **`detect_hallucination(answer, spans, ...)`**
   - Input: Answer text with citations + span definitions
   - Automatically splits answer into claims
   - Returns per-claim budget analysis

2. **`audit_trace_budget(steps, spans, ...)`**
   - Input: Pre-parsed claims with explicit citations
   - More reliable than auto-splitting
   - Recommended for production use

**Running the server:**
```bash
# Using the wrapper script
./run_mcp_server.sh /path/to/aoai_pool.json

# Or directly
python -m strawberry.mcp_server /path/to/aoai_pool.json
```

### `run_mcp_server.sh`

Convenience wrapper script that:
- Sets up PYTHONPATH correctly
- Validates pool config exists
- Launches the MCP server with stdio transport

---

## Modified Files

### `src/strawberry/backend.py`

Added `aoai_pool` as a new backend type:

```python
@dataclass
class BackendConfig:
    kind: str = "openai"  # "openai", "vllm", or "aoai_pool"
    # ... existing fields ...
    # New AOAI pool fields:
    aoai_pool_json_path: Optional[str] = None
    aoai_pool_max_attempts: Optional[int] = None
```

The `make_backend()` function now handles `kind="aoai_pool"`.

### `src/strawberry/__init__.py`

Updated exports to include new modules:
- `aoai_pool_backend`
- `aoai_pool_python`
- `mcp_server`

### `pyproject.toml`

Added optional dependencies:

```toml
[project.optional-dependencies]
mcp = ["mcp[cli]>=1.0.0; python_version>='3.10'"]
aoai = ["requests>=2.28"]

[project.scripts]
hallucination-detector-mcp = "strawberry.mcp_server:main"
```

Changed Python requirement from `>=3.10` to `>=3.9` (core toolkit works on 3.9, MCP requires 3.10+).

---

## Installation

```bash
# Create Python 3.12 virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install with MCP support
pip install -e ".[mcp]"

# If using Azure OpenAI pool, also install:
pip install -e ".[mcp,aoai]"
```

## Claude Code Registration

### Option 1: OpenAI API (Recommended for most users)

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=sk-...

# Register with Claude Code
claude mcp add hallucination-detector \
  -e OPENAI_API_KEY=$OPENAI_API_KEY -- \
  /path/to/.venv/bin/python -m strawberry.mcp_server
```

### Option 2: Azure OpenAI Pool

```bash
claude mcp add hallucination-detector -- \
  /path/to/.venv/bin/python -m strawberry.mcp_server \
  /path/to/aoai_pool.json
```

### Verify Installation

```bash
claude mcp list
# Should show: hallucination-detector: ... ✓ Connected
```

### Running Standalone (without Claude Code)

```bash
# OpenAI API
export OPENAI_API_KEY=sk-...
python -m strawberry.mcp_server

# Azure OpenAI Pool
python -m strawberry.mcp_server /path/to/aoai_pool.json
```

---

## Usage

Once the MCP server is registered, Claude Code can use two tools:

### `detect_hallucination`

Automatically splits an answer into claims and checks each against cited sources.

**Example prompt to Claude:**
```
Use detect_hallucination to verify this answer:

Answer: "The function returns 42 [S0] and handles errors gracefully [S1]."

Spans:
- S0: "def calculate(): return 42"
- S1: "try: ... except: raise"
```

**Input format:**
```json
{
  "answer": "The function returns 42 [S0] and handles errors gracefully [S1].",
  "spans": [
    {"sid": "S0", "text": "def calculate(): return 42"},
    {"sid": "S1", "text": "try: ... except: raise"}
  ]
}
```

### `audit_trace_budget`

More reliable - you provide pre-parsed claims with explicit citations.

**Example prompt to Claude:**
```
Use audit_trace_budget to verify these claims:

Steps:
1. "The function returns 42" citing S0
2. "Errors are re-raised, not handled" citing S1

Spans:
- S0: "def calculate(): return 42"
- S1: "try: ... except: raise"
```

**Input format:**
```json
{
  "steps": [
    {"idx": 0, "claim": "The function returns 42", "cites": ["S0"]},
    {"idx": 1, "claim": "Errors are re-raised, not handled", "cites": ["S1"]}
  ],
  "spans": [
    {"sid": "S0", "text": "def calculate(): return 42"},
    {"sid": "S1", "text": "try: ... except: raise"}
  ]
}
```

### Understanding Results

**Response structure:**
```json
{
  "flagged": true,
  "summary": {
    "claims_scored": 2,
    "flagged_claims": 1,
    "flagged_idxs": [1],
    "backend": "openai"
  },
  "details": [
    {
      "idx": 0,
      "claim": "The function returns 42",
      "cites": ["S0"],
      "flagged": false,
      "budget_gap": {"min": -2.1, "max": -1.5}
    },
    {
      "idx": 1,
      "claim": "Errors are re-raised, not handled",
      "cites": ["S1"],
      "flagged": true,
      "budget_gap": {"min": 8.3, "max": 12.1}
    }
  ]
}
```

**Interpreting `budget_gap`:**
| Gap (bits) | Meaning |
|------------|---------|
| < 0 | Claim well-supported by evidence |
| 0-2 | Probably fine, minor extrapolation |
| 2-10 | Suspicious - verify manually |
| > 10 | Likely hallucination/confabulation |

**Key fields:**
- `flagged`: `true` if ANY claim exceeds the information budget
- `flagged_idxs`: Which claims failed verification
- `budget_gap`: How many "bits" of unsupported information the claim contains

### Optional Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `verifier_model` | `gpt-4o-mini` | Model for verification |
| `default_target` | `0.95` | Confidence threshold (lower = more permissive) |
| `max_claims` | `25` | Max claims to process |
| `claim_split` | `sentences` | Split mode: `sentences` or `lines` |
| `units` | `bits` | Output units: `bits` or `nats` |
| `max_concurrency` | `8` | Parallel requests |

---

## Test Results

### Phase 1-4: Core Functionality ✅

| Test | Status |
|------|--------|
| Python 3.12 environment | ✅ |
| Toolkit imports | ✅ |
| MCP SDK imports | ✅ |
| KL divergence calculation | ✅ |
| Span scrubbing | ✅ |
| Prompt generation | ✅ |
| Logprob extraction | ✅ |
| AOAI pool text API | ✅ |
| AOAI pool JSON API | ✅ |
| Backend factory | ✅ |
| Entailment detection (should pass) | ✅ |
| Confabulation detection (should flag) | ✅ |

### Phase 5-6: MCP Server ✅

| Test | Status |
|------|--------|
| Server imports | ✅ |
| Span normalization | ✅ |
| Claim splitting | ✅ |
| Citation extraction | ✅ |
| Citation ID mapping | ✅ |
| Server starts without error | ✅ |
| Direct tool call | ✅ |
| Empty spans handling | ✅ |
| audit_trace_budget | ✅ |

### Phase 7: Claude Code Integration ✅

| Test | Status |
|------|--------|
| MCP registration | ✅ |
| Server connected | ✅ |

---

## Practical Scenario Results

| Scenario | Expected | Actual | Status |
|----------|----------|--------|--------|
| Phantom Citation | [❌, ✅] | [❌, ✅] | ✅ PASS |
| Wrong Reasoning | [❌, ✅] | [✅, ✅] | ⚠️ PARTIAL |
| Doc Confabulation | [❌, ✅, ✅] | [❌, ✅, ✅] | ✅ PASS |
| Stack Trace Misread | [❌, ✅] | [❌, ✅] | ✅ PASS |
| Test Output Misinterpret | [✅, ✅] | [❌, ✅] | ⚠️ PARTIAL |
| Evidence-Independent | A:❌, B:✅ | A:❌, B:✅ | ✅ PASS |
| Config Misread | [✅, ❌, ✅] | [✅, ❌, ✅] | ✅ PASS |
| Negation Blindness | [✅, ✅] | [✅, ✅] | ✅ PASS |

**Legend:** ❌ = NOT flagged, ✅ = FLAGGED

---

## Realistic Scenarios Part 2 Results

These scenarios test harder semantic and language-specific edge cases.

| # | Scenario | Description | Expected | Actual | Status |
|---|----------|-------------|----------|--------|--------|
| 23 | Lying Comment | Comment says "never returns null" but code does | Flag | Flagged | ✅ PASS |
| 24 | Test Mock Assumption | Claim based on mock behavior, not real API | Flag | Passed | ❌ FAIL |
| 25 | Incomplete Error Handling | Missing error path in catch block | Flag | Flagged | ✅ PASS |
| 26 | Docker Port Confusion | Internal vs exposed port mixup | Flag | Flagged | ✅ PASS |
| 27 | Silent Mutation | JS `.sort()` mutates array in place | Flag | Passed | ❌ FAIL |
| 28 | Stale Package Lock | lockfile version mismatch with package.json | Flag | Flagged | ✅ PASS |
| 29 | Premature Notification | Transaction timing issue (notify before commit) | Flag | Passed | ❌ FAIL |
| 30 | Inherited Default | Correct base class default value | Pass | Passed | ✅ PASS |
| 31 | Environment Check Inversion | `!==` logic inverted in conditional | Flag | Passed | ❌ FAIL |
| 32 | Floating Point Comparison | `0.1 + 0.2 !== 0.3` in JavaScript | Flag | Passed | ❌ FAIL |

**Result: 5/10 (50%)**

### Failure Analysis

The Part 2 failures reveal fundamental limitations in semantic understanding:

| Failure Type | Scenarios | Root Cause |
|--------------|-----------|------------|
| Language Semantics | 27, 32 | Detector doesn't know JS `.sort()` mutates or floating point quirks |
| Mock vs Reality | 24 | Can't distinguish test mock behavior from production API |
| Temporal Logic | 29 | Doesn't catch transaction timing/race conditions |
| Logic Inversion | 31 | Requires understanding conditional flow, not text matching |

### Combined Results Summary

| Test Suite | Pass Rate | Notes |
|------------|-----------|-------|
| Core Functionality | 100% | All imports, APIs, integrations working |
| Practical Scenarios (Part 1) | 75% | Good on textual/citation issues |
| Realistic Scenarios (Part 2) | 50% | Struggles with semantic/language issues |

---

## Critical Review

### Overall Assessment: B (Updated after Part 2)

### Strengths

| Capability | Rating |
|------------|--------|
| Catches phantom citations | ✅ Excellent |
| Detects confabulated details | ✅ Excellent |
| Flags external knowledge bleed | ✅ Excellent |
| Handles negation | ✅ Excellent |
| Accepts reasonable paraphrasing | ✅ Good |
| Distinguishes precision levels | ✅ Excellent |
| Catches lying comments | ✅ Good |
| Detects config/port mismatches | ✅ Good |

### Limitations

| Issue | Severity |
|-------|----------|
| Language-specific semantics (mutability, floats) | High |
| Mock vs production behavior distinction | High |
| Temporal/transaction logic | High |
| Logic inversion detection | Medium |
| Sometimes too strict on inference | Medium |
| Partial truth detection is weak | Medium |
| Claim splitting creates artifacts | Low |
| Latency (~2 calls per claim) | Low |

### User Paradigm Suitability

| Use Case | Rating | Notes |
|----------|--------|-------|
| Developer Debugging | ████████░░ Good | May be strict on obvious inferences |
| Code Reviewer | █████████░ Excellent | Great at catching unsupported claims |
| Architect/Designer | ████████░░ Good | Catches inflated numbers |
| Junior Dev Learning | ██████████ Excellent | Flags training-data answers |
| DevOps Postmortem | ██████░░░░ Moderate | Too strict on timestamp math |
| Security Analyst | █████████░ Excellent | High-value for verification |

### Recommendations

1. **Use `audit_trace_budget()` over `detect_hallucination()`** - Pre-parse claims for better accuracy

2. **Trust flags more than passes** - Low false positive rate, but significant false negatives on semantic issues

3. **Tune `default_target` for use case:**
   - Security: 0.95 (strict)
   - Debugging: 0.85 (permissive)

4. **Interpret `budget_gap`:**
   - `> 10 bits`: Strong confabulation signal
   - `2-10 bits`: Possible issue, verify
   - `< 2 bits`: Probably fine

5. **Know the blind spots** - The detector excels at textual/citation verification but cannot catch:
   - Language-specific behaviors (JS array mutation, floating point quirks)
   - Mock vs production API differences
   - Race conditions and transaction timing
   - Subtle logic inversions (`===` vs `!==`)

6. **Best use cases:**
   - Verifying quoted facts, numbers, and citations
   - Catching phantom/fabricated references
   - Detecting confabulated documentation details

   **Not recommended for:**
   - Code semantic analysis
   - Verifying runtime behavior claims
   - Detecting subtle logic bugs

---

## API Reference

### `detect_hallucination()`

```python
detect_hallucination(
    answer: str,                    # Answer text with citations like [S0]
    spans: List[Dict[str, str]],    # [{"sid": "S0", "text": "..."}]
    verifier_model: str = "gpt-4o-mini",
    default_target: float = 0.95,   # Confidence threshold
    max_claims: int = 25,
    claim_split: str = "sentences", # or "lines"
    units: str = "bits",            # or "nats"
) -> Dict
```

**Returns:**
```python
{
    "flagged": bool,
    "summary": {
        "claims_scored": int,
        "flagged_claims": int,
        "flagged_idxs": List[int],
    },
    "details": [
        {
            "idx": int,
            "claim": str,
            "cites": List[str],
            "flagged": bool,
            "required": {"min": float, "max": float},
            "observed": {"min": float, "max": float},
            "budget_gap": {"min": float, "max": float},
        }
    ]
}
```

### `audit_trace_budget()`

```python
audit_trace_budget(
    steps: List[Dict],    # [{"idx": 0, "claim": "...", "cites": ["S0"]}]
    spans: List[Dict],    # [{"sid": "S0", "text": "..."}]
    verifier_model: str = "gpt-4o-mini",
    default_target: float = 0.95,
    units: str = "bits",
) -> Dict
```

Same return format as `detect_hallucination()`.

---

## Changelog

### v0.2.0 (2026-01-10)

**Added:**
- Azure OpenAI pool backend (`aoai_pool`)
- MCP server for Claude Code integration
- `detect_hallucination` tool
- `audit_trace_budget` tool
- `run_mcp_server.sh` convenience script

**Changed:**
- Python requirement relaxed to >=3.9 (MCP still requires 3.10+)
- `BackendConfig` extended with AOAI pool options

**Dependencies:**
- Added optional `mcp` extras
- Added optional `aoai` extras

**Test Results:**
- Core functionality: 100% pass
- Practical scenarios (Part 1): 75% pass
- Realistic scenarios (Part 2): 50% pass
- Overall grade: B (strong on textual verification, weak on semantic analysis)
