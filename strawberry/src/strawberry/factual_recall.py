#!/usr/bin/env python3
"""factual_recall.py

Closed-system factual recall: answer -> cite -> validate (trace-budget).

This script implements a *programmatic* version of:
  1) Answer a general factual recall question.
  2) Provide citations (as short evidence spans).
  3) Validate the answer by checking whether those cited spans actually entail the claim,
     using a *pseudo-prior* defined by scrubbing the cited spans (p0) vs full context (p1).

It reuses the toolkit's trace-budget machinery:
  - scrub cited spans
  - estimate p0 and p1 via YES/NO/UNSURE logprobs
  - compute RequiredBits and ObservedBits (Bernoulli KL)

Optional procedural-hallucination tell:
  - Re-answer after scrubbing the cited spans.
  - If the model keeps the same answer with high confidence, that indicates evidence-insensitive
    routing ("making it up as it goes").

Limitations (by design):
  - No external retrieval, no external verifier, no web.
  - Therefore this can *not* prove the citations exist in the real world; it can only test
    internal textual entailment and evidence dependence.

Run (OpenAI):
  python factual_recall.py \
    --backend openai \
    --generator_model gpt-4o-mini \
    --question "Which US senators from Minnesota graduated from Princeton" \
    --out report.json

Run (vLLM):
  python factual_recall.py \
    --backend vllm \
    --generator_model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --question "..." \
    --vllm_tensor_parallel 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


# -------------------------
# Import toolkit (standalone-friendly)
# -------------------------

def _find_toolkit_src(start_dir: str) -> Optional[str]:
    """Find a nearby `src/` directory that contains `strawberry/`."""
    cur = os.path.abspath(start_dir)
    for _ in range(8):
        cand = os.path.join(cur, "src")
        if os.path.isdir(os.path.join(cand, "strawberry")):
            return cand
        cur = os.path.dirname(cur)
    return None


def _ensure_toolkit_on_path() -> None:
    try:
        import strawberry  # noqa: F401
        return
    except Exception:
        pass

    src = _find_toolkit_src(os.path.dirname(os.path.abspath(__file__))) or _find_toolkit_src(os.getcwd())
    if src and src not in sys.path:
        sys.path.insert(0, src)


_ensure_toolkit_on_path()

try:
    from strawberry.backend import BackendConfig, make_backend
    from strawberry.trace_budget import BudgetResult, score_trace_budget, scrub_spans_by_id
except Exception as e:
    raise SystemExit(
        "Failed to import strawberry_toolkit.\n"
        "- If you're running from the repo root: `pip install -e .`\n"
        "- Or ensure `src/` is on PYTHONPATH.\n"
        f"Import error: {e}"
    )


# -------------------------
# Local data models (minimal)
# -------------------------

@dataclass
class EvidenceSpan:
    sid: str
    excerpt: str
    source: str = ""


@dataclass
class SimpleSpan:
    sid: str
    text: str


@dataclass
class SimpleStep:
    idx: int
    claim: str
    cites: List[str]
    confidence: float


@dataclass
class SimpleTrace:
    steps: List[SimpleStep]
    spans: List[SimpleSpan]


# -------------------------
# JSON schemas for Structured Outputs
# -------------------------


def _evidence_schema(max_spans: int) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "spans": {
                "type": "array",
                "maxItems": int(max_spans),
                "items": {
                    "type": "object",
                    "properties": {
                        "sid": {"type": "string"},
                        "source": {"type": "string"},
                        "excerpt": {"type": "string"},
                    },
                    "required": ["sid", "excerpt"],
                    "additionalProperties": False,
                },
            },
            "notes": {"type": "string"},
        },
        "required": ["spans"],
        "additionalProperties": False,
    }


def _answer_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "claim": {"type": "string"},
            "cites": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "abstain": {"type": "boolean"},
            "notes": {"type": "string"},
        },
        "required": ["answer", "claim", "cites", "abstain"],
        "additionalProperties": False,
    }


# -------------------------
# Core procedure
# -------------------------


def _normalize_spans(raw_spans: List[Dict[str, Any]], *, max_spans: int) -> List[EvidenceSpan]:
    """Ensure unique sequential S0.. ids and clamp span lengths."""
    out: List[EvidenceSpan] = []
    for i, s in enumerate(raw_spans[: int(max_spans)]):
        excerpt = str(s.get("excerpt", "") or "").strip()
        source = str(s.get("source", "") or "").strip()
        # Force sequential IDs for downstream consistency
        sid = f"S{i}"
        if excerpt:
            out.append(EvidenceSpan(sid=sid, excerpt=excerpt, source=source))
    return out


def retrieve_evidence(
    *,
    backend: Any,
    model: str,
    question: str,
    max_spans: int,
    max_words_per_span: int,
    temperature: float,
    reasoning: Optional[Dict[str, Any]],
) -> List[EvidenceSpan]:
    """Ask the model to produce evidence excerpts (closed-system)."""
    schema = _evidence_schema(max_spans=max_spans)

    prompt = f"""
QUESTION:
{question.strip()}

TASK:
Provide up to {max_spans} EVIDENCE SPANS that would support answering the QUESTION.

RULES:
- Each EVIDENCE SPAN must be a *declarative assertion* (not a question, not an instruction).
- Each excerpt must be <= {max_words_per_span} words.
- Include a best-effort SOURCE string (author/title/site/year), but do not invent URLs.
- If you cannot provide trustworthy evidence excerpts from memory, return an empty spans list.

Return only JSON matching the schema.
""".strip()

    r = backend.call_json_schema(
        prompt=prompt,
        schema=schema,
        model=model,
        name="evidence_retrieval",
        instructions="Return ONLY a JSON object matching the schema.",
        temperature=float(temperature),
        max_output_tokens=1200,
        include_logprobs=False,
        reasoning=reasoning,
    )

    raw = list((r.data or {}).get("spans", []) or [])
    return _normalize_spans(raw, max_spans=max_spans)


def answer_with_citations(
    *,
    backend: Any,
    model: str,
    question: str,
    spans: List[EvidenceSpan],
    temperature: float,
    reasoning: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Answer using ONLY provided evidence spans, returning structured answer+citations."""

    schema = _answer_schema()

    ctx_lines: List[str] = []
    for s in spans:
        # Keep SOURCE outside the excerpt body so the verifier doesn't confuse it as factual content.
        src = (s.source or "").strip()
        if src:
            ctx_lines.append(f"[{s.sid}] {s.excerpt}\nSOURCE: {src}")
        else:
            ctx_lines.append(f"[{s.sid}] {s.excerpt}")
    ctx = "\n\n".join(ctx_lines).strip()

    prompt = f"""
You must answer the QUESTION using ONLY the factual assertions in the CONTEXT SPANS.
Do NOT use world knowledge, plausibility, or memory beyond the provided spans.

If the CONTEXT SPANS do not entail a specific answer, you MUST abstain:
- set abstain=true
- set answer="UNSUPPORTED"
- set cites=[]
- set confidence <= 0.5

Otherwise:
- set abstain=false
- answer concisely
- claim must be a SINGLE declarative sentence that would be TRUE iff your answer is correct
- cites must list ONLY span IDs that directly support the claim
- confidence is your probability the claim is entailed by the cited spans (0..1)

CONTEXT SPANS:
{ctx if ctx else "[NO CONTEXT SPANS PROVIDED]"}

QUESTION:
{question.strip()}

Return only JSON matching the schema.
""".strip()

    r = backend.call_json_schema(
        prompt=prompt,
        schema=schema,
        model=model,
        name="answer_with_citations",
        instructions="Return ONLY a JSON object matching the schema.",
        temperature=float(temperature),
        max_output_tokens=900,
        include_logprobs=False,
        reasoning=reasoning,
    )
    data = dict(r.data or {})

    # Normalize cites to strings
    cites = [str(x) for x in (data.get("cites", []) or [])]
    data["cites"] = cites

    # Clamp confidence if absent
    if "confidence" not in data or data["confidence"] is None:
        data["confidence"] = 0.95
    try:
        data["confidence"] = float(data["confidence"])
    except Exception:
        data["confidence"] = 0.95

    return data


def validate_with_trace_budget(
    *,
    backend_cfg: BackendConfig,
    verifier_model: str,
    claim: str,
    cites: List[str],
    spans: List[EvidenceSpan],
    target: float,
    top_logprobs: int,
    placeholder: str,
    reasoning: Optional[Dict[str, Any]],
) -> Optional[BudgetResult]:
    """Run a single-step trace budget check for the answer claim."""

    claim = (claim or "").strip()
    if not claim:
        return None

    trace = SimpleTrace(
        steps=[SimpleStep(idx=0, claim=claim, cites=list(cites), confidence=float(target))],
        spans=[SimpleSpan(sid=s.sid, text=s.excerpt) for s in spans],
    )

    res = score_trace_budget(
        trace=trace,
        verifier_model=verifier_model,
        backend_cfg=backend_cfg,
        default_target=float(target),
        temperature=0.0,
        top_logprobs=int(top_logprobs),
        placeholder=str(placeholder),
        reasoning=reasoning,
    )
    return res[0] if res else None


def _budget_to_dict(b: BudgetResult) -> Dict[str, Any]:
    return {
        "idx": b.idx,
        "claim": b.claim,
        "cites": b.cites,
        "target": b.target,
        "p0_yes": {"lower": b.prior_yes.p_yes_lower, "upper": b.prior_yes.p_yes_upper, "generated": b.prior_yes.generated},
        "p1_yes": {"lower": b.post_yes.p_yes_lower, "upper": b.post_yes.p_yes_upper, "generated": b.post_yes.generated},
        "required_bits": {"min": b.required_bits_min, "max": b.required_bits_max},
        "observed_bits": {"min": b.observed_bits_min, "max": b.observed_bits_max},
        "budget_gap": {"min": b.budget_gap_min, "max": b.budget_gap_max},
        "flagged": b.flagged,
    }


# -------------------------
# CLI
# -------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Closed-system factual recall auditor (answer -> cite -> trace-budget validate).")

    p.add_argument("--question", type=str, default=None, help="The factual recall question.")
    p.add_argument("--question_file", type=str, default=None, help="Path to a UTF-8 text file containing the question.")

    p.add_argument("--backend", type=str, default="openai", choices=["openai", "vllm"], help="Execution backend.")
    p.add_argument("--generator_model", type=str, required=True, help="Model used for evidence retrieval + answering.")
    p.add_argument("--verifier_model", type=str, default=None, help="Model used for entailment/logprob budget checks (default: generator_model).")

    # OpenAI-compatible server knobs
    p.add_argument("--base_url", type=str, default=None, help="Optional OpenAI-compatible base_url (e.g., vLLM server).")
    p.add_argument("--api_key", type=str, default=None, help="Optional API key override (else use OPENAI_API_KEY).")

    # vLLM knobs
    p.add_argument("--vllm_tensor_parallel", type=int, default=1)
    p.add_argument("--vllm_max_model_len", type=int, default=None)
    p.add_argument("--vllm_dtype", type=str, default="bfloat16")
    p.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--vllm_trust_remote_code", action="store_true")

    # Evidence retrieval controls
    p.add_argument("--max_spans", type=int, default=4)
    p.add_argument("--max_words_per_span", type=int, default=45)

    # Budget controls
    p.add_argument("--budget_target", type=float, default=None, help="Target reliability τ for RequiredBits. Default: use model's returned confidence (else 0.95).")
    p.add_argument("--top_logprobs", type=int, default=10, help="Top-K logprobs for YES/NO/UNSURE (0-20).")
    p.add_argument("--placeholder", type=str, default="[REDACTED]")

    # Procedural tell
    p.add_argument("--null_answer_check", action="store_true", help="Re-answer after scrubbing cited spans and compare.")

    # Misc
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_concurrency", type=int, default=16)
    p.add_argument("--timeout_s", type=float, default=None)
    p.add_argument("--out", type=str, default=None, help="Write full JSON report to this path (else print to stdout).")
    p.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")

    # Reasoning param passthrough (useful for OpenAI reasoning models; ignored by vLLM)
    p.add_argument("--reasoning_effort", type=str, default=None, help="Optional OpenAI reasoning.effort (low/medium/high).")

    return p


def _load_question(args: argparse.Namespace) -> str:
    if args.question_file:
        with open(args.question_file, "r", encoding="utf-8") as f:
            q = f.read().strip()
            if q:
                return q
    if args.question:
        return str(args.question).strip()
    # Fallback: stdin
    data = sys.stdin.read().strip()
    if not data:
        raise SystemExit("No question provided. Use --question or --question_file, or pipe via stdin.")
    return data


def main() -> None:
    args = build_argparser().parse_args()

    question = _load_question(args)

    verifier_model = args.verifier_model or args.generator_model

    backend_cfg = BackendConfig(
        kind=str(args.backend),
        max_concurrency=int(args.max_concurrency),
        timeout_s=float(args.timeout_s) if args.timeout_s is not None else None,
        base_url=args.base_url,
        api_key=args.api_key,
        vllm_tensor_parallel=int(args.vllm_tensor_parallel),
        vllm_max_model_len=args.vllm_max_model_len,
        vllm_dtype=str(args.vllm_dtype),
        vllm_gpu_memory_utilization=float(args.vllm_gpu_memory_utilization),
        vllm_trust_remote_code=bool(args.vllm_trust_remote_code),
    )

    reasoning: Optional[Dict[str, Any]] = None
    if args.reasoning_effort:
        reasoning = {"effort": str(args.reasoning_effort)}

    backend = make_backend(backend_cfg, model_hint=args.generator_model if args.backend == "vllm" else None)

    # 1) Closed-system evidence retrieval
    spans = retrieve_evidence(
        backend=backend,
        model=args.generator_model,
        question=question,
        max_spans=int(args.max_spans),
        max_words_per_span=int(args.max_words_per_span),
        temperature=float(args.temperature),
        reasoning=reasoning,
    )

    # 2) Evidence-only answer + citations
    ans = answer_with_citations(
        backend=backend,
        model=args.generator_model,
        question=question,
        spans=spans,
        temperature=float(args.temperature),
        reasoning=reasoning,
    )

    abstain = bool(ans.get("abstain", False))
    claim = str(ans.get("claim", "") or "")
    cites = [str(x) for x in (ans.get("cites", []) or [])]
    answer_text = str(ans.get("answer", "") or "")

    # Budget target τ
    if args.budget_target is not None:
        target = float(args.budget_target)
    else:
        # If model provided confidence, use it; else default 0.95
        try:
            target = float(ans.get("confidence", 0.95))
        except Exception:
            target = 0.95

    budget: Optional[BudgetResult] = None
    if not abstain:
        budget = validate_with_trace_budget(
            backend_cfg=backend_cfg,
            verifier_model=verifier_model,
            claim=claim,
            cites=cites,
            spans=spans,
            target=target,
            top_logprobs=int(args.top_logprobs),
            placeholder=str(args.placeholder),
            reasoning=reasoning,
        )

    # 3) Optional: re-answer under scrubbed evidence (procedural tell)
    null_check: Optional[Dict[str, Any]] = None
    if bool(args.null_answer_check) and spans and (cites or not abstain):
        scrubbed = scrub_spans_by_id([SimpleSpan(sid=s.sid, text=s.excerpt) for s in spans], cites, placeholder=str(args.placeholder))
        # Convert back to EvidenceSpan list
        scrubbed_spans = [EvidenceSpan(sid=getattr(s, "sid"), excerpt=str(getattr(s, "text")), source="") for s in scrubbed]
        ans_null = answer_with_citations(
            backend=backend,
            model=args.generator_model,
            question=question,
            spans=scrubbed_spans,
            temperature=float(args.temperature),
            reasoning=reasoning,
        )

        null_check = {
            "answer": str(ans_null.get("answer", "") or ""),
            "abstain": bool(ans_null.get("abstain", False)),
            "cites": [str(x) for x in (ans_null.get("cites", []) or [])],
            "confidence": float(ans_null.get("confidence", 0.0) or 0.0),
            "claim": str(ans_null.get("claim", "") or ""),
        }

        # Evidence-insensitivity smell: same answer despite evidence removal, without abstaining.
        null_check["evidence_insensitive"] = (
            (not null_check["abstain"])
            and (str(null_check["answer"]).strip() == str(answer_text).strip())
        )

    # Build report
    report: Dict[str, Any] = {
        "question": question,
        "backend": args.backend,
        "generator_model": args.generator_model,
        "verifier_model": verifier_model,
        "evidence_spans": [asdict(s) for s in spans],
        "answer": {
            "answer": answer_text,
            "claim": claim,
            "cites": cites,
            "confidence": float(ans.get("confidence", 0.0) or 0.0),
            "abstain": abstain,
            "notes": str(ans.get("notes", "") or ""),
        },
        "budget_target": float(target),
        "trace_budget": _budget_to_dict(budget) if budget is not None else None,
        "null_answer_check": null_check,
    }

    # Final decision heuristic
    flagged = False
    reasons: List[str] = []
    if abstain:
        reasons.append("abstained")
    else:
        if budget is None:
            flagged = True
            reasons.append("no_budget_result")
        else:
            if bool(budget.flagged):
                flagged = True
                reasons.append("under_budget")

    if null_check and bool(null_check.get("evidence_insensitive")):
        flagged = True
        reasons.append("evidence_insensitive_answer")

    report["flagged"] = bool(flagged)
    report["flag_reasons"] = reasons

    # Emit
    txt = json.dumps(report, ensure_ascii=False, indent=2 if args.pretty else None)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(txt)
        print(args.out)
    else:
        print(txt)


if __name__ == "__main__":
    main()
