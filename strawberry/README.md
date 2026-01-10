# Strawberry Toolkit

**Detect procedural hallucinations before they ship.**

LLMs hallucinate because they *know but don't use*—the answer is in the context, the model just doesn't route to it. This toolkit detects those failures mathematically, using only API outputs and logprobs.

---

## Quick Start

```bash
# Install
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[mcp]"

# Set API key
export OPENAI_API_KEY=sk-...

# Register with Claude Code
claude mcp add hallucination-detector \
  -e OPENAI_API_KEY=$OPENAI_API_KEY -- \
  $(pwd)/.venv/bin/python -m strawberry.mcp_server

# Verify
claude mcp list
```

Then ask Claude: *"Use detect_hallucination on the root cause you just generated"*

---

## Usage

### `detect_hallucination`

Automatically splits an answer into claims and checks each against cited sources.

**Example prompt:**
```
Use detect_hallucination to verify this answer:

Answer: "The function returns 42 [S0] and handles errors gracefully [S1]."

Spans:
- S0: "def calculate(): return 42"
- S1: "try: ... except: raise"
```

### `audit_trace_budget`

More reliable - you provide pre-parsed claims with explicit citations.

**Example prompt:**
```
Use audit_trace_budget to verify these claims:

Steps:
1. "The function returns 42" citing S0
2. "Errors are re-raised, not handled" citing S1

Spans:
- S0: "def calculate(): return 42"
- S1: "try: ... except: raise"
```

### Understanding Results

```json
{
  "flagged": true,
  "summary": {"claims_scored": 2, "flagged_claims": 1, "flagged_idxs": [1]},
  "details": [
    {"idx": 0, "claim": "...", "flagged": false, "budget_gap": {"min": -2.1, "max": -1.5}},
    {"idx": 1, "claim": "...", "flagged": true, "budget_gap": {"min": 8.3, "max": 12.1}}
  ]
}
```

| `budget_gap` (bits) | Meaning |
|---------------------|---------|
| < 0 | Claim well-supported |
| 0-2 | Minor extrapolation |
| 2-10 | Suspicious - verify |
| > 10 | Likely hallucination |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `verifier_model` | `gpt-4o-mini` | Model for verification |
| `default_target` | `0.95` | Confidence threshold |
| `max_claims` | `25` | Max claims to process |
| `units` | `bits` | Output units: `bits` or `nats` |

---

## What It Catches

**Citation & Evidence Failures:**
- Phantom citations (made-up references)
- Confabulated documentation details
- Evidence-independent answers (training data bleed)
- Partial evidence (claims exceed what's supported)
- Multi-source conflation (inventing connections between sources)

**Code Reading Failures:**
- Stack trace misreads
- Config value misreads
- Negation blindness ("NOT" missed)
- Lying comments (code contradicts comment)
- SQL join/schema misreads
- Git merge conflict residue
- Environment variable override misses (.env.local)
- Middleware order bugs

**Root Cause Analysis Failures:**
- Correlation claimed as causation
- Interpretive leaps stated as fact
- Prescriptive claims disguised as observations
- Unverified causal chains

**DevOps & Config Failures:**
- Docker port confusion (host:container)
- Stale package lock mismatches
- Version mismatch attribution
- Git misattribution (wrong author/commit)

**Test & Verification Failures:**
- Test output misinterpretation
- Test mock assumptions (mock ≠ production)
- Incomplete error handling assumptions

---

## The Theory

1. Compute `p1` = P(correct | full context)
2. Scrub evidence → compute `p0` = P(correct | scrubbed)
3. **ObservedBits** = KL(Ber(p1) || Ber(p0))
4. **RequiredBits** = KL(Ber(τ) || Ber(p0))

If ObservedBits < RequiredBits → evidence doesn't justify confidence → **flag**.

---

## CLI Tools

### Factual Recall Audit

```bash
python -m strawberry.factual_recall \
  --backend openai \
  --generator_model gpt-4o-mini \
  --question "Which US senators from Minnesota graduated from Princeton" \
  --out report.json
```

### Synthetic Binding Evaluation

```bash
strawberry run \
  --backend openai \
  --model gpt-4o-2024-08-06 \
  --n 200 --M 10 --distance 512 \
  --query FIRST --null SCRUB_FIRST
```

### Chain-of-Thought Auditing

```bash
strawberry cot \
  --backend openai \
  --generator_model gpt-4o-mini \
  --verifier_model gpt-4o-mini \
  --synthetic --M 10 --distance 256
```

---

## Python API

```python
from strawberry.tasks import generate_dataset
from strawberry.eval import run_eval

items = generate_dataset(n=50, distance_tokens=512, M=10, query_rule="FIRST", seed=0)
results = run_eval(items=items, model="gpt-4o-2024-08-06", null_mode="SCRUB_FIRST")
```

---

## Repo Layout

```
src/strawberry/
├── mcp_server.py       # MCP server for Claude Code
├── trace_budget.py     # Scrub + p0/p1 + budget calculations
├── factual_recall.py   # Factual recall auditor
├── cot_detector.py     # Trace hallucination detection
├── backend.py          # Backend abstraction
├── openai_backend.py   # OpenAI API implementation
├── tasks.py            # Synthetic prompt generation
├── eval.py             # Evaluation runner
└── metrics.py          # KL, bits-to-trust, Fano bounds
```

---

## Changelog

### v0.2.0 (2026-01-10)
- Added MCP server for Claude Code integration
- Added `detect_hallucination` and `audit_trace_budget` tools

---

## Citation

```bibtex
@article{chlon2026procedural,
  title={An Information-Theoretic and Causal Theory of Procedural Hallucinations},
  author={Chlon, Leon},
  institution={Hassana Labs},
  journal={arXiv preprint},
  year={2026}
}
```

---

## License

MIT
