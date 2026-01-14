# Proof Repair Agent

The **Proof Repair Agent** is an evidence-first workflow for repairing (or synthesizing) mathematical proofs.

It is designed for situations like:

- "This `.tex` proof doesn't prove theorem **Y** — fix it."
- "This proof has a gap — find the exact missing lemma and repair it."
- "Formalize this theorem in Lean and ensure the LaTeX statement matches."

The workflow is implemented as a **repo-scoped OpenAI Codex skill**:

- `.codex/skills/proof-repair-agent/SKILL.md` (the full, authoritative specification)

This document is a **human-friendly guide** to help people understand and use it.

Related guide:
- `PROOF_ATTACK_AGENT.md` (focused brute-force loop for a single stuck goal)

---

## What makes it different

Most "proof assistants" in LLM-land stop when the model is *confident*.

This workflow does **not** allow that.

You may stop only when you can produce **primary evidence**:

- a proof assistant **accepts** theorem **Y** (Lean/Coq/Isabelle/…)

…and you have passed verification gates using Strawberry's tools (`audit_trace_budget` / `detect_hallucination`).

If you cannot produce enough evidence, the agent must:
- gather more evidence (new logs, excerpts, counterexamples), **or**
- downgrade claims to hypotheses, **or**
- remove claims.

---

## How the new architecture avoids “Lean step hallucinations”

This repo now splits the work into two cooperating skills:

1) `$proof-attack-agent`
   - A focused brute-force loop for a **single stuck goal**.
   - Maintains an Attempt Ledger (anti-repeat).
   - Runs Strawberry verification as a heartbeat after **every** micro-plan update.

2) `$proof-repair-agent`
   - Owns the overall repair, but delegates stuck gaps to `$proof-attack-agent`.
   - Starts by writing a **plan-of-record** (Success Spec + Step Plan) using the `planning-agent` style,
     then checks plan adherence throughout the repair.

The key behavioral change: hallucination detection is no longer a “final report step”—it happens *before*
you commit to new Lean tactic sequences.

---

## Prerequisites

### Required

- **OpenAI Codex** (CLI or IDE extension) configured for this repo.
- The **Strawberry MCP server** registered in Codex.

### Recommended toolchains

- **LaTeX** toolchain (`latexmk` recommended; `pdflatex` also works).
- **Lean + Lake** (if you want a real "machine-checked proof" stop condition).

If you don't have a formal proof assistant available, you can still use the workflow to produce:

- corrected statements,
- partial results (lemmas / special cases),
- counterexamples to false sub-claims,

…but you **must not** claim "theorem Y is proven" without a formal backstop.

---

## Quickstart

### 1) Register Strawberry MCP with Codex

From `pythea/strawberry/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e "..[dev]"  # optional (tests)
pip install -e ".[mcp]"   # required for MCP server

export OPENAI_API_KEY=sk-...

codex mcp add hallucination-detector \
  --env OPENAI_API_KEY=$OPENAI_API_KEY -- \
  $(pwd)/.venv/bin/python -m strawberry.mcp_server

codex mcp list
```

### 2) Ensure proof toolchains are installed

- LaTeX: `latexmk` or `pdflatex` on `PATH`
- Lean: `lean` / `lake` on `PATH` (if you want the formal backstop)

### 3) Start Codex in the repo and invoke the skill

In the repo root:

```text
$proof-repair-agent

This .tex proof doesn't prove theorem Y.
- tex_path: path/to/paper.tex
- theorem label (if known): thm:some_label
- lean project dir (if any): path/to/lake/project
- lean theorem name (if any): Some.theorem

Repro commands:
- LaTeX: (use compile_latex)
- Lean: (use lean_check)
```

If you want the agent to do internet lookup (e.g., to find library lemmas / references), start Codex with web search enabled (CLI):

```bash
codex --search
```

---

## Core concepts

### Evidence Pack (mandatory)

The workflow requires an **Evidence Pack** with spans named `S0`, `S1`, …

Each span is a small, raw excerpt of evidence with provenance, for example:

- `path/to/file.tex:L10-L25`
- `command (cwd=..., exit=...)`
- `URL (date)`

A template lives at:

- `.codex/skills/proof-repair-agent/assets/evidence_pack_template.md`

Recommended run folder layout:

```
proof_agent_runs/<timestamp>-<slug>/
  plan.md
  evidence.md
  attempts.md        # optional but recommended (anti-repeat)
  report.md
```

### Strawberry verification gates

The agent should verify key claims using Strawberry:

- `audit_trace_budget(steps, spans, ...)` (preferred: you provide atomic claims + citations)
- `detect_hallucination(answer, spans, ...)` (more automatic)

If Strawberry abstains/flags, the agent must **not** proceed as if the claim were true.

### Theorem drift

A common failure mode is to "fix the proof" by quietly changing theorem **Y**.

The workflow treats this as a first-class failure mode:

- extract theorem statement from `.tex`
- extract theorem statement from Lean (if present)
- run `tex_vs_lean_drift_check`
- treat drift as a **test failure** unless explicitly allowed

---

## Workflow overview (7 phases)

This is the "state machine" the skill enforces.

| Phase | Goal | Typical evidence | Output |
|------:|------|------------------|--------|
| 0 | Pin down theorem **Y** | extracted statement blocks | canonical record of Y |
| 1 | Baseline reproduction | LaTeX logs; Lean build logs | "current failure" spans |
| 2 | Hypotheses + experiments | counterexamples; isolated sub-lemmas | ranked hypotheses |
| 3 | ROOT_CAUSE gate | cited steps + evidence | Strawberry-checked primary claim |
| 4 | Fix plan | planned edits + failure-mode checks | plan + test plan |
| 5 | Implement + test | LaTeX build; Lean build; drift check | passing checks or new evidence |
| 6 | Verification pass | Strawberry verification of report | claims constrained to evidence |
| 7 | Deliverables | patch + report + commands | reproducible final output |

---

## What "done" means

You may stop only when all are true:

1) The formal backstop accepts theorem **Y** (e.g., `lean_check` succeeds).
2) LaTeX compiles (or at minimum the document is consistent and builds).
3) The theorem drift check is not flagged.
4) Strawberry does **not** flag the minimum claim set.

If any condition fails, the workflow loops back to Phase 2.

---

## Output: what you should deliver

Use the template:

- `.codex/skills/proof-repair-agent/assets/proof_repair_report_template.md`

A good final report includes:

- the exact theorem **Y** (as written)
- baseline reproduction logs (what failed originally)
- hypotheses considered (including refuted ones)
- the **ROOT_CAUSE** claim, with citations
- a patch summary
- the test plan and outputs (with citations)
- theorem drift status
- remaining uncertainties (explicitly labeled)

---

## Using the workflow without Lean

You can still use the Proof Repair Agent to:

- identify and document proof gaps,
- propose missing lemmas and reduce them to smaller sub-claims,
- find counterexamples to false steps,
- produce correct partial theorems (e.g., special cases, density bounds).

But you must treat "theorem proven" as **out of scope** without a proof assistant.

A useful alternative stop condition in the no-Lean setting is:

- "We found a concrete counterexample" (disproves a sub-claim), or
- "We produced a corrected weaker theorem and proved it fully," or
- "We produced a repair plan and a checklist of missing lemmas to formalize later."

---

## Troubleshooting

### Strawberry keeps abstaining

That usually means the report claims are more specific than the evidence.

Fix:

- add the missing raw evidence spans (logs / code excerpts), or
- rewrite the claims to be narrower (what is actually shown), or
- isolate the disputed claim into smaller atomic steps and retry `audit_trace_budget`.

### LaTeX builds but the proof is still wrong

LaTeX success is only a *syntax* check.

You still need:

- proof assistant acceptance, or
- a validated counterexample for a false claim.

### Lean builds fail due to imports / mathlib

Treat this like any other root-cause problem:

- capture the build error log as evidence,
- hypothesize missing dependency / wrong import / version mismatch,
- run discriminating experiments (minimal Lean file, `lake update`, etc.),
- iterate.

---

## File locations

- Skill spec (authoritative): `.codex/skills/proof-repair-agent/SKILL.md`
- Evidence Pack template: `.codex/skills/proof-repair-agent/assets/evidence_pack_template.md`
- Report template: `.codex/skills/proof-repair-agent/assets/proof_repair_report_template.md`
- Web search notes: `.codex/skills/proof-repair-agent/references/codex_web_search.md`

---

## Suggested next improvements

If you use this heavily, consider adding:

- a dedicated "no formal backstop" variant skill (experiment-first validation)
- integrations for Coq/Isabelle (analogous to `lean_check`)
- a theorem-statement canonicalizer (for stronger drift detection)
- CI checks that fail PRs if proof-agent deliverables are missing for proof changes
