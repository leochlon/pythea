
"""strawberry.cot_detector

A *reasoning-trace* ("chain-of-thought") hallucination detector built on top of the toolkit.

Important caveat
----------------
OpenAI hosted reasoning models generate internal reasoning tokens that are not exposed via the API.
Therefore this module audits *user-visible* reasoning traces: a short, structured list of verifiable
claims with citations to provided context spans.

What it detects
---------------
1) **Grounding hallucinations**: a trace step asserts a claim not entailed by the provided context.
2) **Procedural (post-hoc) hallucinations**: the trace does not justify the final answer
   (an independent executor cannot derive the answer from the trace).
3) **Trace-budget insufficiency** (optional): a step is asserted with high confidence while the
   evidence supplies insufficient "bits" relative to a pseudo-prior defined by a causal null.

Parallelism (safe)
------------------
- We keep the core logic barriers:
    answer-only scoring (optional) -> trace generation -> (verification / budget / executor)
- Trace budgeting is *batched across steps* (posterior vs pseudo-prior prompts) for speed.
- Step verification is kept sequential by default because it can incorporate previously verified premises.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import re
import warnings

from .backend import BackendConfig, make_backend
from .stage_ab import StageABReport, compute_stage_ab, TokenTopK, Stage2A, Stage2B
from .trace_budget import BudgetResult, score_trace_budget


TraceKind = Literal["copy", "deduce", "arithmetic", "assumption"]
Verdict = Literal["ENTAILED", "CONTRADICTED", "NOT_IN_CONTEXT", "UNVERIFIABLE"]


@dataclass
class Span:
    sid: str
    text: str


@dataclass
class TraceStep:
    idx: int
    claim: str
    kind: TraceKind
    cites: List[str]
    # Optional target reliability for budgeting (if omitted, use global budget_target)
    confidence: float = 0.95


@dataclass
class Trace:
    answer: str
    steps: List[TraceStep]
    spans: List[Span]


@dataclass
class StepCheck:
    idx: int
    verdict: Verdict
    confidence: float
    notes: str


@dataclass
class TraceReport:
    answer: str
    answer_from_trace: str
    answer_match: bool
    grounded_fraction: float
    hallucinated_steps: int
    not_in_context_steps: int
    contradicted_steps: int
    unverifiable_steps: int
    checks: List[StepCheck]
    # Optional output-level Stage 2A/2B metrics (computed from logprobs) when choices are supplied.
    scored_answer: Optional[str] = None
    stage2a: Optional[Dict[str, Any]] = None
    stage2b: Optional[Dict[str, Any]] = None
    stage_meta: Optional[Dict[str, Any]] = None
    # Stage-2A/2B status string (e.g., computed, skipped)
    stage_ab: Optional[str] = None
    # Optional: pre-answer deliberation sweep (answer drift vs reasoning length)
    deliberation_sweep: Optional[List[Dict[str, Any]]] = None
    # Optional trace-level budget diagnostics
    trace_budget: Optional[List[Dict[str, Any]]] = None
    budget_flagged_steps: int = 0
    max_budget_gap_min: Optional[float] = None
    max_budget_gap_max: Optional[float] = None


def spanize(text: str, *, max_chars: int = 650) -> List[Span]:
    """Chunk text into moderately-sized spans for citation."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    spans: List[Span] = []
    buf: List[str] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if buf:
            sid = f"S{len(spans)}"
            spans.append(Span(sid=sid, text="\n\n".join(buf).strip()))
            buf = []
            buf_len = 0

    for p in paras:
        if buf_len + len(p) + 2 > max_chars and buf:
            flush()
        buf.append(p)
        buf_len += len(p) + 2
    flush()
    if not spans:
        spans = [Span(sid="S0", text=text.strip())]
    return spans


def _trace_schema(max_steps: int) -> Dict[str, Any]:
    # Include optional confidence per step for trace-budget.
    return {
        "type": "object",
        "properties": {
            "final_answer": {"type": "string"},
            "trace": {
                "type": "array",
                "maxItems": int(max_steps),
                "items": {
                    "type": "object",
                    "properties": {
                        "idx": {"type": "integer", "minimum": 0},
                        "claim": {"type": "string"},
                        "kind": {"type": "string", "enum": ["copy", "deduce", "arithmetic", "assumption"]},
                        "cites": {"type": "array", "items": {"type": "string"}},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                    "required": ["idx", "claim", "kind", "cites"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["final_answer", "trace"],
        "additionalProperties": False,
    }


def _verdict_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["ENTAILED", "CONTRADICTED", "NOT_IN_CONTEXT", "UNVERIFIABLE"]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "notes": {"type": "string"},
        },
        "required": ["verdict", "confidence"],
        "additionalProperties": False,
    }


def _answer_from_trace_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "determined": {"type": "boolean"},
            "notes": {"type": "string"},
        },
        "required": ["answer", "determined"],
        "additionalProperties": False,
    }


def generate_reasoning_trace(
    *,
    prompt: str,
    model: str,
    backend_cfg: Optional[BackendConfig] = None,
    max_steps: int = 8,
    temperature: float = 0.0,
    reasoning: Optional[Dict[str, Any]] = None,
    given_answer: Optional[str] = None,
) -> Trace:
    """Ask a model to produce a short, checkable reasoning trace."""
    spans = spanize(prompt)
    span_block = "\n".join([f"[{s.sid}] {s.text}" for s in spans])

    cfg = backend_cfg or BackendConfig(kind="openai")
    backend = make_backend(cfg, model_hint=model if cfg.kind.lower() == "vllm" else None)

    instructions = "You are a careful reasoning-trace generator. Return ONLY a JSON object that matches the schema."

    ans_line = "" if given_answer is None else f"\nFINAL ANSWER (already chosen): {given_answer}\n"

    user = f"""
CONTEXT SPANS (cite these by ID):
{span_block}

TASK:
1) Provide the final answer.{ans_line}
2) Provide a short reasoning trace as a list of *atomic claims*.

TRACE RULES:
- Each claim must be checkable.
- Each claim should be <= 20 words.
- Use kind=copy if directly stated in one span.
- Use kind=deduce if it follows from earlier trace steps + the question.
- Use kind=arithmetic only for numeric manipulation.
- Use kind=assumption ONLY if not supported by context; then cites must be [].
- cites must be a list of span IDs you relied on (or [] for assumptions).
- confidence (optional) is your subjective probability the claim is entailed.
- Do NOT write a long scratchpad.
""".strip()

    if given_answer is not None:
        user += "\n\nCONSTRAINT: The field final_answer MUST EXACTLY equal the provided FINAL ANSWER."

    r = backend.call_json_schema(
        prompt=user,
        schema=_trace_schema(max_steps=max_steps),
        model=model,
        name="reasoning_trace",
        instructions=instructions,
        temperature=temperature,
        max_output_tokens=1200,
        include_logprobs=False,
        reasoning=reasoning,
    )

    data = r.data
    steps_raw = data.get("trace", [])
    steps: List[TraceStep] = []
    for st in steps_raw:
        steps.append(
            TraceStep(
                idx=int(st["idx"]),
                claim=str(st["claim"]).strip(),
                kind=str(st["kind"]),
                cites=[str(x) for x in st.get("cites", [])],
                confidence=float(st.get("confidence", 0.95) or 0.95),
            )
        )
    steps.sort(key=lambda s: s.idx)
    return Trace(answer=str(data.get("final_answer", "")).strip(), steps=steps, spans=spans)



# -------------------------
# Choice / checkpoint helpers (Stage-2A/2B safety + auto-choice modes)
# -------------------------

def _infer_query_key(prompt: str, question_hint: Optional[str] = None) -> Optional[str]:
    """Best-effort inference of a queried key/variable name from the prompt."""
    if question_hint:
        q = str(question_hint).strip()
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,64}", q):
            return q

    # Common patterns
    pats = [
        r"value of\s+([A-Za-z][A-Za-z0-9_]{0,64})",
        r"what is\s+([A-Za-z][A-Za-z0-9_]{0,64})\s*\?",
        r"query\s+([A-Za-z][A-Za-z0-9_]{0,64})",
    ]
    for pat in pats:
        m = re.search(pat, prompt, flags=re.IGNORECASE)
        if m:
            return m.group(1)
    return None




def _generate_deliberation_steps(
    *,
    prompt: str,
    model: str,
    backend_cfg: BackendConfig,
    max_steps: int = 8,
    temperature: float = 0.0,
    reasoning: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate a *pre-answer* deliberation trace consisting of up to `max_steps` short claims
    with citations to context spans.

    This is used to study *answer drift* as a function of reasoning length K: we first
    generate these steps, append them to the prompt, then re-score Stage-2A/2B.
    """
    backend = make_backend(backend_cfg, model_hint=model if backend_cfg.kind.lower() == "vllm" else None)

    schema = {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "maxItems": int(max_steps),
                "items": {
                    "type": "object",
                    "properties": {
                        "idx": {"type": "integer", "minimum": 0},
                        "claim": {"type": "string"},
                        "cites": {"type": "array", "items": {"type": "string"}},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                    "required": ["idx", "claim", "cites"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["steps"],
        "additionalProperties": False,
    }

    system = (
        "You will produce a short deliberation trace BEFORE answering.\n"
        "Rules:\n"
        "- Produce up to the requested number of steps.\n"
        "- Each step must be a short, verifiable claim grounded in the provided spans.\n"
        "- Each step MUST cite at least one span id [S#].\n"
        "- Do NOT output the final answer.\n"
        "- Output JSON matching the schema exactly.\n"
    )

    r = backend.call_json_schema(
        prompt=prompt,
        model=model,
        system=system,
        schema=schema,
        temperature=float(temperature),
        max_output_tokens=900,
        reasoning=reasoning,
    )
    steps = list((r.data or {}).get("steps", []) or [])
    # Sanitize: keep only dict-like steps with required keys
    clean: List[Dict[str, Any]] = []
    for st in steps:
        if not isinstance(st, dict):
            continue
        if "claim" not in st or "cites" not in st:
            continue
        clean.append(
            {
                "idx": int(st.get("idx", len(clean))),
                "claim": str(st.get("claim", "")).strip(),
                "cites": [str(x) for x in (st.get("cites", []) or [])],
                "confidence": float(st.get("confidence", 0.85) or 0.85),
            }
        )
    clean.sort(key=lambda x: int(x.get("idx", 0)))
    return clean


def _format_deliberation_text(steps: Sequence[Dict[str, Any]], *, header: str = "MODEL DELIBERATION") -> str:
    """Format deliberation steps as plain text to append into the prompt."""
    lines: List[str] = []
    for st in steps:
        cites = ",".join([c.strip().lstrip("[").rstrip("]") for c in (st.get("cites", []) or []) if str(c).strip()])
        claim = str(st.get("claim", "")).strip()
        if not claim:
            continue
        if cites:
            lines.append(f"- ({cites}) {claim}")
        else:
            lines.append(f"- {claim}")
    if not lines:
        return f"{header}:\n- (no steps)"
    return f"{header}:\n" + "\n".join(lines)

def _extract_bindings_from_context(prompt: str, *, max_bindings: int = 64) -> List[Dict[str, Any]]:
    """Heuristic extraction of key=value pairs from labeled spans in the prompt."""
    spans = spanize(prompt)
    out: List[Dict[str, Any]] = []
    seen = set()
    # Very permissive: KEY = VALUE up to end-of-line / delimiter
    rx = re.compile(r"\b([A-Za-z][A-Za-z0-9_]{0,64})\s*=\s*([^\s,;:.]+)", flags=re.MULTILINE)
    for sp in spans:
        for m in rx.finditer(sp.text):
            key = m.group(1).strip()
            val = m.group(2).strip()
            if not key or not val:
                continue
            tup = (key, val, sp.sid)
            if tup in seen:
                continue
            seen.add(tup)
            out.append({"key": key, "value": val, "cites": [sp.sid]})
            if len(out) >= int(max_bindings):
                return out
    return out


def _candidates_from_bindings(
    bindings: Sequence[Dict[str, Any]],
    *,
    key: Optional[str] = None,
    max_candidates: int = 32,
) -> List[str]:
    """Extract a de-duplicated candidate value list from bindings (optionally for a specific key)."""
    key_norm = (str(key).strip().lower() if key else None)
    vals: List[str] = []
    seen = set()
    for b in bindings:
        k = str(b.get("key", "")).strip().lower()
        v = str(b.get("value", "")).strip()
        if not v:
            continue
        if key_norm is not None and k != key_norm:
            continue
        if v in seen:
            continue
        seen.add(v)
        vals.append(v)
        if len(vals) >= int(max_candidates):
            break
    return vals


def _recode_candidates_to_codes(
    candidates: Sequence[str],
    *,
    other_code: str = "Z",
    max_codes: int = 25,
) -> Tuple[List[str], Dict[str, str], str]:
    """Assign candidates to single-token codes A..Y, reserving Z for OTHER/ABSTAIN."""
    letters = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c != other_code]
    letters = letters[: int(max_codes)]
    cands = list(candidates)[: len(letters)]
    code_map = {letters[i]: str(cands[i]) for i in range(len(cands))}
    choices_codes = list(code_map.keys()) + [other_code]
    return choices_codes, code_map, other_code


def _format_code_block(code_map: Dict[str, str], other_code: str) -> str:
    lines = [f"{k} = {v}" for k, v in code_map.items()]
    lines.append(f"{other_code} = OTHER / ABSTAIN")
    return "\n".join(lines)


def _build_stage_ab_from_sequence_scores(
    *,
    choices: Sequence[str],
    scores: Sequence[float],
    correct: Optional[str],
    other_token: str,
) -> Tuple[str, StageABReport, Dict[str, Any]]:
    """Construct a StageABReport from exact sequence scores (multi-token aware)."""
    if len(choices) != len(scores):
        raise ValueError("choices/scores length mismatch")

    # Argmax prediction
    best_i = max(range(len(scores)), key=lambda i: float(scores[i]))
    pred = str(choices[best_i])

    # Stage 2A: candidate-vs-other
    if other_token not in choices:
        raise ValueError(f"other_token {other_token!r} missing from choices")
    idx_other = list(choices).index(other_token)
    score_other = float(scores[idx_other])

    # Best non-other candidate
    non_other = [(i, float(s)) for i, s in enumerate(scores) if str(choices[i]) != other_token]
    best_non_other_i, best_non_other_s = max(non_other, key=lambda t: t[1]) if non_other else (idx_other, score_other)

    gate_success = (pred != other_token)
    gate_gap = float(best_non_other_s - score_other)
    stage2a = Stage2A(
        gate_success=bool(gate_success),
        predicted=str(pred),
        gate_gap_upper=float(gate_gap),
        gate_gap_lower=float(gate_gap),
        cert_candidate_over_other=bool(gate_gap > 0.0),
        coverage_candidates_in_topk=1.0,
    )

    # Stage 2B: correct-vs-competitor among non-other
    binding_eval = False
    binding_correct = None
    value_gap = None
    cert_correct_worse = None
    corr = None
    if correct is not None and str(correct) in choices:
        corr = str(correct)
        # Competitor: best among non-other excluding correct
        comp_scores = [(i, float(s)) for i, s in enumerate(scores) if (str(choices[i]) != other_token and str(choices[i]) != corr)]
        if gate_success and comp_scores:
            binding_eval = True
            best_comp_i, best_comp_s = max(comp_scores, key=lambda t: t[1])
            corr_s = float(scores[list(choices).index(corr)])
            value_gap = float(corr_s - best_comp_s)
            binding_correct = (pred == corr)
            cert_correct_worse = bool(corr_s < float(scores[best_i])) if pred != corr else False

    stage2b = Stage2B(
        binding_evaluated=bool(binding_eval),
        binding_correct=binding_correct,
        correct=str(corr) if corr is not None else "",
        predicted=str(pred),
        value_gap_upper=float(value_gap) if value_gap is not None else None,
        value_gap_lower=float(value_gap) if value_gap is not None else None,
        cert_correct_worse_than_pred=cert_correct_worse,
        coverage_candidates_in_topk=1.0,
    )

    # Overload TokenTopK with sequence scores for transparency (not token-level topk)
    topk = TokenTopK(
        pos=0,
        generated_token=str(pred),
        generated_logprob=float(scores[best_i]),
        topk_logprobs={str(c): float(s) for c, s in zip(choices, scores)},
        kth_logprob=None,
    )

    meta = {
        "scoring_mode": "sequence",
        "sequence_scores": {str(c): float(s) for c, s in zip(choices, scores)},
    }
    return pred, StageABReport(topk=topk, stage2a=stage2a, stage2b=stage2b), meta


def _warn_or_raise_openai_multitoken(choices: Sequence[str], *, choices_are_codes: bool) -> None:
    """Heuristic guardrail for OpenAI backend where tokenizer is not available."""
    if choices_are_codes:
        return
    # If any choice has whitespace or punctuation, it's very likely multi-token.
    suspicious = [c for c in choices if re.search(r"\s|[^A-Za-z0-9_]", str(c))]
    if suspicious:
        raise ValueError(
            "OpenAI backend Stage-2A/2B requires single-token choices for rigorous logprob gaps. "
            "One or more choices look multi-token (whitespace/punctuation): "
            + ", ".join(repr(s) for s in suspicious[:8])
            + ". Provide single-token codes (recommended) and pass choices_are_codes=True."
        )
    warnings.warn(
        "Stage-2A/2B assumes choices correspond to a single token at the answer position. "
        "OpenAI backend cannot verify tokenization; metrics may be approximate. "
        "Consider providing single-token codes and setting choices_are_codes=True.",
        RuntimeWarning,
    )


def _validate_vllm_single_token_choices(tokenizer, prefix: str, choices: Sequence[str]) -> List[str]:
    """Return list of choices that are NOT single-token continuations after prefix."""
    prefix = str(prefix)
    if not prefix.endswith((" ", "\n", "\t")):
        # ensure stable boundary
        prefix = prefix + " "
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    bad: List[str] = []
    for c in choices:
        full_ids = tokenizer(prefix + str(c), add_special_tokens=False).input_ids
        suffix_len = len(full_ids) - len(prefix_ids)
        if suffix_len != 1:
            bad.append(str(c))
    return bad


def _generate_binding_checkpoint(
    *,
    prompt: str,
    model: str,
    backend_cfg: BackendConfig,
    max_bindings: int = 32,
    temperature: float = 0.0,
    reasoning: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """LLM-extracted binding checkpoint with citations: bindings[{key,value,cites}]."""
    backend = make_backend(backend_cfg, model_hint=model if backend_cfg.kind.lower() == "vllm" else None)

    schema = {
        "type": "object",
        "properties": {
            "bindings": {
                "type": "array",
                "maxItems": int(max_bindings),
                "items": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                        "value": {"type": "string"},
                        "cites": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["key", "value", "cites"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["bindings"],
        "additionalProperties": False,
    }

    system = (
        "You extract explicit key=value bindings from the provided context spans.\n"
        "Rules:\n"
        "- Only include bindings that are explicitly stated in the context.\n"
        "- Each binding MUST cite at least one span id [S#] where the assignment appears.\n"
        "- Keep values short (no newlines).\n"
        "- Output JSON that matches the schema exactly.\n"
    )

    r = backend.call_json_schema(
        prompt=prompt,
        model=model,
        system=system,
        schema=schema,
        temperature=temperature,
        max_output_tokens=512,
        reasoning=reasoning,
    )
    data = r.data if isinstance(r.data, dict) else {}
    # Normalize cites to bare ids like "S1"
    binds = []
    for b in list(data.get("bindings", []) or []):
        key = str(b.get("key", "")).strip()
        val = str(b.get("value", "")).strip().replace("\n", " ")
        cites = []
        for c in list(b.get("cites", []) or []):
            cc = str(c).strip()
            cc = cc.strip("[]")
            if re.fullmatch(r"S\d+", cc):
                cites.append(cc)
        if key and val and cites:
            binds.append({"key": key, "value": val, "cites": cites})
    return {"bindings": binds}

def score_answer_stage_ab(
    *,
    prompt: str,
    choices: Sequence[str],
    model: str,
    backend_cfg: Optional[BackendConfig] = None,
    correct: Optional[str] = None,
    other_token: str = "OTHER",
    temperature: float = 0.0,
    top_logprobs: int = 20,
    reasoning: Optional[Dict[str, Any]] = None,
    choices_are_codes: bool = False,
    vllm_multitoken_fallback: Optional[str] = None,  # None | "sequence"
) -> Tuple[str, StageABReport, Dict[str, Any]]:
    """Get an answer-only completion with logprobs and compute Stage 2A/2B.

    Safety:
    - vLLM backend: we can verify whether each choice is a *single token* at the answer position.
      If any are multi-token, we raise by default, or fall back to rigorous sequence scoring
      when vllm_multitoken_fallback="sequence".
    - OpenAI backend: we cannot tokenize, so we enforce codes if choices contain whitespace/punctuation,
      and otherwise warn that metrics may be approximate.
    """
    cfg = backend_cfg or BackendConfig(kind="openai")
    backend = make_backend(cfg, model_hint=model if cfg.kind.lower() == "vllm" else None)

    # Basic invariants
    ch = [str(c) for c in choices]
    if other_token not in ch:
        raise ValueError(f"Stage-2A requires other_token {other_token!r} to be included in choices.")

    # Build a stable answer prefix so tokenization checks are meaningful.
    # NOTE: The model's output begins immediately after this prefix.
    options = "\n".join(ch)
    user_prefix = (
        prompt.strip()
        + "\n\nOPTIONS:\n"
        + options
        + "\n\nReturn EXACTLY one option from OPTIONS. Output ONLY the option text.\nAnswer: "
    )

    meta: Dict[str, Any] = {
        "other_token": other_token,
        "choices_count": len(ch),
        "backend_kind": cfg.kind,
    }

    # Tokenization guardrails
    if cfg.kind.lower() == "openai":
        _warn_or_raise_openai_multitoken(ch, choices_are_codes=choices_are_codes)
        meta["tokenization_validated"] = False
    elif cfg.kind.lower() == "vllm":
        tok = getattr(backend, "tokenizer", None)
        if tok is None:
            warnings.warn("vLLM backend missing tokenizer attribute; cannot validate single-token choices.", RuntimeWarning)
            meta["tokenization_validated"] = False
        else:
            bad = _validate_vllm_single_token_choices(tok, user_prefix, ch)
            meta["tokenization_validated"] = True
            meta["multi_token_choices"] = bad
            if bad:
                if (vllm_multitoken_fallback or "").lower() == "sequence":
                    # Rigorous sequence scoring fallback
                    if not hasattr(backend, "score_choice_sequences"):
                        raise ValueError("vLLM backend does not implement score_choice_sequences; cannot sequence-score.")
                    scores = backend.score_choice_sequences(prefix=user_prefix, choices=ch)
                    pred, report, meta2 = _build_stage_ab_from_sequence_scores(
                        choices=ch, scores=scores, correct=correct, other_token=other_token
                    )
                    meta.update(meta2)
                    meta["stage_ab_status"] = "computed (sequence scoring fallback)"
                    return pred, report, meta
                raise ValueError(
                    "Stage-2A/2B token-topK metrics require single-token choices at the answer position. "
                    "The following choices are multi-token under the model tokenizer: "
                    + ", ".join(repr(x) for x in bad[:10])
                    + ". Either (a) provide single-token codes, or (b) enable vLLM sequence fallback "
                      "with vllm_multitoken_fallback='sequence'."
                )
    else:
        meta["tokenization_validated"] = False

    # Token-topK scoring pass (fast path)
    r = backend.call_text(
        prompt=user_prefix,
        model=model,
        instructions="Answer using exactly one of the provided OPTIONS. Output only the option text.",
        temperature=temperature,
        max_output_tokens=8,
        include_logprobs=True,
        top_logprobs=int(top_logprobs),
        reasoning=reasoning,
    )
    ans = (r.text or "").strip()
    stage = compute_stage_ab(
        generated_text=ans,
        logprobs=r.logprobs,
        choices=ch,
        correct=correct,
        other_token=other_token,
    )
    meta["stage_ab_status"] = "computed (token-topK)"
    return ans, stage, meta


def verify_trace_steps(
    *,
    trace: Trace,
    verifier_model: str,
    backend_cfg: Optional[BackendConfig] = None,
    temperature: float = 0.0,
    reasoning: Optional[Dict[str, Any]] = None,
) -> List[StepCheck]:
    """Verify each step using a strict span-grounded verifier.

    We verify steps sequentially so that previously VERIFIED steps can act as premises.
    """
    cfg = backend_cfg or BackendConfig(kind="openai")
    backend = make_backend(cfg, model_hint=verifier_model if cfg.kind.lower() == "vllm" else None)

    span_block = "\n".join([f"[{s.sid}] {s.text}" for s in trace.spans])
    premises: List[str] = []
    checks: List[StepCheck] = []

    instructions = (
        "You are a strict verifier. Use ONLY the provided spans and premises; "
        "do not use external knowledge. Return ONLY a JSON object that matches the schema."
    )

    for st in trace.steps:
        prem_block = "\n".join([f"- {p}" for p in premises]) if premises else "(none)"
        cited = ", ".join(st.cites) if st.cites else "(none)"

        user = f"""
CONTEXT SPANS:
{span_block}

VERIFIED PREMISES:
{prem_block}

CLAIM TO CHECK:
- kind: {st.kind}
- cites: {cited}
- claim: {st.claim}

DECISION INSTRUCTIONS:
- Verdict ENTAILED if the claim is supported by the spans/premises.
- Verdict CONTRADICTED if spans/premises imply the opposite.
- Verdict NOT_IN_CONTEXT if it asserts new factual content absent from spans/premises.
- Verdict UNVERIFIABLE if it is too vague or ill-formed to judge.
- confidence should reflect how clear the entailment/contradiction is.
""".strip()

        r = backend.call_json_schema(
            prompt=user,
            schema=_verdict_schema(),
            model=verifier_model,
            name="trace_step_verdict",
            instructions=instructions,
            temperature=temperature,
            max_output_tokens=400,
            include_logprobs=False,
            reasoning=reasoning,
        )

        verdict: Verdict = r.data["verdict"]
        conf = float(r.data.get("confidence", 0.5))
        notes = str(r.data.get("notes", "")).strip()
        checks.append(StepCheck(idx=st.idx, verdict=verdict, confidence=conf, notes=notes))
        if verdict == "ENTAILED":
            premises.append(st.claim)

    return checks


def answer_from_trace(
    *,
    question_hint: str,
    trace: Trace,
    executor_model: str,
    backend_cfg: Optional[BackendConfig] = None,
    temperature: float = 0.0,
    reasoning: Optional[Dict[str, Any]] = None,
) -> Tuple[str, bool, str]:
    """Ask an independent executor to answer using ONLY the trace."""
    cfg = backend_cfg or BackendConfig(kind="openai")
    backend = make_backend(cfg, model_hint=executor_model if cfg.kind.lower() == "vllm" else None)

    steps_block = "\n".join([f"{s.idx}. ({s.kind}) {s.claim}" for s in trace.steps]) or "(no steps)"

    instructions = (
        "You are an executor. Use ONLY the provided trace statements as premises. "
        "If the answer is not determined by the trace, set determined=false. "
        "Return ONLY a JSON object that matches the schema."
    )

    user = f"""
QUESTION:
{question_hint}

TRACE PREMISES:
{steps_block}

Return the answer you can *derive from the trace premises only*.
""".strip()

    r = backend.call_json_schema(
        prompt=user,
        schema=_answer_from_trace_schema(),
        model=executor_model,
        name="answer_from_trace",
        instructions=instructions,
        temperature=temperature,
        max_output_tokens=200,
        include_logprobs=False,
        reasoning=reasoning,
    )

    ans = str(r.data.get("answer", "")).strip()
    det = bool(r.data.get("determined", False))
    notes = str(r.data.get("notes", "")).strip()
    return ans, det, notes


def detect_cot_hallucinations(
    *,
    prompt: str,
    question_hint: Optional[str] = None,
    generator_model: str,
    verifier_model: str,
    executor_model: Optional[str] = None,
    # Optional: enable Stage-2A/2B analysis.
    # Provide explicit candidates via `choices`, or derive them via `choices_mode`.
    choices: Optional[Sequence[str]] = None,
    choices_mode: Optional[str] = None,  # auto_trace | auto_context
    choices_key: Optional[str] = None,   # target key for auto choices (if inferable)
    choices_are_codes: bool = False,     # OpenAI: assert choices are single-token codes
    auto_choice_mode: str = "codes",   # codes (default) | sequence (vLLM only)
    vllm_multitoken_fallback: Optional[str] = None,  # None | "sequence"
    correct: Optional[str] = None,
    # Optional: pre-answer deliberation sweep (answer drift vs reasoning length)
    deliberation_sweep: Optional[Sequence[int]] = None,
    deliberation_kind: str = "trace",  # trace | checkpoint
    deliberation_model: Optional[str] = None,
    deliberation_temperature: float = 0.0,
    score_model: Optional[str] = None,
    top_logprobs: int = 20,
    # Trace budget knobs (default ON)
    trace_budget: bool = True,
    budget_model: Optional[str] = None,
    budget_target: float = 0.95,
    budget_top_logprobs: int = 10,
    budget_placeholder: str = "[REDACTED]",
    # Backend knobs
    backend_kind: str = "openai",
    backend_cfg: Optional[BackendConfig] = None,
    max_steps: int = 8,
    temperature: float = 0.0,
    reasoning: Optional[Dict[str, Any]] = None,
) -> Tuple[Trace, TraceReport]:
    """End-to-end detector.

    1) Optional answer-only Stage-2A/2B scoring (logprobs).
    2) Trace generation (certificate).
    3) Step verification (grounding).
    4) Optional trace-budget scoring (batched across steps).
    5) Procedural faithfulness check via independent executor.
    """
    cfg = backend_cfg or BackendConfig(kind=str(backend_kind))
    # For vLLM, we assume all roles use the same local model. If you need different models,
    # run separate processes or use the OpenAI backend.
    if cfg.kind.lower() == "vllm":
        role_models = [m for m in [generator_model, verifier_model, executor_model, score_model, budget_model] if m is not None]
        if len(set(role_models)) > 1:
            raise ValueError("vLLM backend currently assumes a single local model for all roles. "
                             "Set generator_model=verifier_model=... or run separate processes.")

    scored_answer = None
    stage2a: Optional[Dict[str, Any]] = None
    stage2b: Optional[Dict[str, Any]] = None
    stage_meta: Optional[Dict[str, Any]] = None

        # Optional Stage-A/B scoring pass (output-side)
    # If no explicit choices are provided, you can request auto choices via choices_mode.
    stage_ab_status: Optional[str] = None
    if choices is None and choices_mode is None:
        stage_ab_status = "skipped (no choices provided)"
    else:
        resolved_choices: Optional[List[str]] = list(choices) if choices is not None else None
        stage_origin: str = "explicit"
        local_vllm_fallback = vllm_multitoken_fallback

        # Auto choices: derive candidates from a checkpoint (auto_trace) or by regex from context (auto_context)
        if resolved_choices is None and choices_mode is not None:
            mode = str(choices_mode).strip().lower()
            stage_origin = mode
            if mode == "auto_context":
                bindings = _extract_bindings_from_context(prompt)
            elif mode == "auto_trace":
                ckpt = _generate_binding_checkpoint(
                    prompt=prompt,
                    model=generator_model,
                    backend_cfg=cfg,
                    max_bindings=32,
                    temperature=temperature,
                    reasoning=reasoning,
                )
                bindings = list(ckpt.get("bindings", []) or [])
            else:
                raise ValueError(f"Unknown choices_mode: {choices_mode!r} (expected auto_trace or auto_context)")

            qkey = choices_key or _infer_query_key(prompt, question_hint)
            candidates = _candidates_from_bindings(bindings, key=qkey)
            stage_meta = (stage_meta or {})
            stage_meta.update(
                {
                    "choices_source": mode,
                    "choices_key": qkey,
                    "candidates": candidates,
                    "bindings": bindings,
                }
            )

            if not candidates:
                stage_ab_status = "skipped (no candidates discovered)"
            else:
                # Either recode to single-token codes (recommended) or do sequence scoring (vLLM only).
                if str(auto_choice_mode).strip().lower() == "sequence":
                    if cfg.kind.lower() != "vllm":
                        raise ValueError("auto_choice_mode='sequence' requires backend=vllm")
                    resolved_choices = list(candidates) + ["OTHER"]
                    local_vllm_fallback = "sequence"
                else:
                    codes, code_map, other_code = _recode_candidates_to_codes(candidates, other_code="Z")
                    code_block = _format_code_block(code_map, other_code)
                    scoring_prompt = (
                        prompt.strip()
                        + "\n\nCANDIDATE CODES:\n"
                        + code_block
                        + "\n\nReturn a CODE (A/B/C/... or Z) corresponding to the correct value.\n"
                    )
                    # Map correct (string) -> correct code if possible
                    correct_code = None
                    if correct is not None:
                        corr = str(correct).strip()
                        for k, v in code_map.items():
                            if str(v).strip() == corr:
                                correct_code = str(k)
                                break
                    ans, stage, meta = score_answer_stage_ab(
                        prompt=scoring_prompt,
                        choices=codes,
                        model=score_model or generator_model,
                        backend_cfg=cfg,
                        correct=correct_code,
                        other_token=other_code,
                        temperature=temperature,
                        top_logprobs=top_logprobs,
                        reasoning=reasoning,
                        choices_are_codes=True,
                        vllm_multitoken_fallback=None,
                    )
                    scored_answer = code_map.get(ans, ans)
                    stage_ab_status = meta.get("stage_ab_status", "computed") + f" (codes; source={mode})"
                    stage2a = {
                        "gate_success": stage.stage2a.gate_success,
                        "predicted": stage.stage2a.predicted,
                        "predicted_value": code_map.get(stage.stage2a.predicted, None),
                        "gate_gap_lower": stage.stage2a.gate_gap_lower,
                        "cert_candidate_over_other": stage.stage2a.cert_candidate_over_other,
                        "coverage_candidates_in_topk": stage.stage2a.coverage_candidates_in_topk,
                    }
                    stage2b = {
                        "binding_evaluated": stage.stage2b.binding_evaluated,
                        "binding_correct": stage.stage2b.binding_correct,
                        "correct": stage.stage2b.correct,
                        "correct_value": code_map.get(stage.stage2b.correct, None),
                        "predicted": stage.stage2b.predicted,
                        "predicted_value": code_map.get(stage.stage2b.predicted, None),
                        "value_gap_upper": stage.stage2b.value_gap_upper,
                        "value_gap_lower": stage.stage2b.value_gap_lower,
                        "cert_correct_worse_than_pred": stage.stage2b.cert_correct_worse_than_pred,
                        "coverage_candidates_in_topk": stage.stage2b.coverage_candidates_in_topk,
                    }
                    stage_meta = (stage_meta or {})
                    stage_meta.update(
                        {
                            **meta,
                            "choice_coding": {"other_code": other_code, "code_map": code_map},
                        }
                    )

        

                    # Internal: record the exact scoring prompt and candidates used (for deliberation sweep reuse)
                    stage_meta.update({"_stage_prompt_used": scoring_prompt, "_stage_choices_used": list(codes), "_stage_correct_used": correct_code, "_stage_other_used": other_code})
# Explicit choices or auto-sequence path
        if resolved_choices is not None and stage_ab_status is None:
            ans, stage, meta = score_answer_stage_ab(
                prompt=prompt,
                choices=list(resolved_choices),
                model=score_model or generator_model,
                backend_cfg=cfg,
                correct=correct,
                other_token="OTHER",
                temperature=temperature,
                top_logprobs=top_logprobs,
                reasoning=reasoning,
                choices_are_codes=choices_are_codes,
                vllm_multitoken_fallback=local_vllm_fallback,
            )
            scored_answer = ans
            stage2a = {
                "gate_success": stage.stage2a.gate_success,
                "predicted": stage.stage2a.predicted,
                "gate_gap_lower": stage.stage2a.gate_gap_lower,
                "cert_candidate_over_other": stage.stage2a.cert_candidate_over_other,
                "coverage_candidates_in_topk": stage.stage2a.coverage_candidates_in_topk,
            }
            stage2b = {
                "binding_evaluated": stage.stage2b.binding_evaluated,
                "binding_correct": stage.stage2b.binding_correct,
                "correct": stage.stage2b.correct,
                "predicted": stage.stage2b.predicted,
                "value_gap_upper": stage.stage2b.value_gap_upper,
                "value_gap_lower": stage.stage2b.value_gap_lower,
                "cert_correct_worse_than_pred": stage.stage2b.cert_correct_worse_than_pred,
                "coverage_candidates_in_topk": stage.stage2b.coverage_candidates_in_topk,
            }
            stage_meta = (stage_meta or {})
            stage_meta.update(meta)
            # Internal: record the exact scoring prompt and candidates used (for deliberation sweep reuse)
            stage_meta.update({"_stage_prompt_used": prompt, "_stage_choices_used": list(resolved_choices), "_stage_correct_used": correct, "_stage_other_used": "OTHER"})
            stage_ab_status = meta.get("stage_ab_status", "computed") + f" (source={stage_origin})"


    # Optional: Deliberation sweep (pre-answer reasoning length K -> answer drift)
    deliberation_sweep_results: Optional[List[Dict[str, Any]]] = None
    if deliberation_sweep is not None and stage2a is not None:
        ks = [int(x) for x in deliberation_sweep if int(x) >= 0]
        ks = sorted(set(ks))
        if ks:
            kind = str(deliberation_kind or "trace").strip().lower()
            if kind not in ("trace", "checkpoint"):
                kind = "trace"
            delib_model = deliberation_model or generator_model
            base_pred = (stage2a or {}).get("predicted") if stage2a else scored_answer

            # Reuse the same candidate set and scoring prompt used for baseline stage scoring when possible.
            stage_prompt_used = prompt
            stage_choices_used: Optional[List[str]] = list(choices) if choices is not None else None
            stage_correct_used = correct
            stage_other_used = "OTHER"

            if stage_meta is not None:
                stage_prompt_used = stage_meta.get("_stage_prompt_used", stage_prompt_used)
                stage_choices_used = stage_meta.get("_stage_choices_used", stage_choices_used)
                stage_correct_used = stage_meta.get("_stage_correct_used", stage_correct_used)
                stage_other_used = stage_meta.get("_stage_other_used", stage_other_used)

            if stage_choices_used is not None:
                deliberation_sweep_results = []
                for k in ks:
                    if k == 0:
                        deliberation_sweep_results.append(
                            {
                                "k": 0,
                                "kind": kind,
                                "predicted": base_pred,
                                "drift_from_base": False,
                                "stage2a": stage2a,
                                "stage2b": stage2b,
                                "deliberation_steps": [],
                            }
                        )
                        continue

                    # 1) Generate deliberation content
                    if kind == "checkpoint":
                        ck = _generate_binding_checkpoint(
                            prompt=stage_prompt_used,
                            model=delib_model,
                            backend_cfg=cfg,
                            max_bindings=int(k),
                            temperature=float(deliberation_temperature),
                            reasoning=reasoning,
                        )
                        binds = list((ck or {}).get("bindings", []) or [])
                        # Format as key=value lines with cites
                        lines: List[str] = []
                        for b in binds[: int(k)]:
                            if not isinstance(b, dict):
                                continue
                            key = str(b.get("key", "")).strip()
                            val = str(b.get("value", "")).strip()
                            cites = ",".join([str(x).strip().lstrip("[").rstrip("]") for x in (b.get("cites", []) or []) if str(x).strip()])
                            if key and val:
                                if cites:
                                    lines.append(f"- ({cites}) {key} = {val}")
                                else:
                                    lines.append(f"- {key} = {val}")
                        delib_text = "MODEL CHECKPOINT:\n" + ("\n".join(lines) if lines else "- (none)")
                        delib_steps = binds[: int(k)]
                    else:
                        steps = _generate_deliberation_steps(
                            prompt=stage_prompt_used,
                            model=delib_model,
                            backend_cfg=cfg,
                            max_steps=int(k),
                            temperature=float(deliberation_temperature),
                            reasoning=reasoning,
                        )
                        delib_text = _format_deliberation_text(steps, header="MODEL DELIBERATION")
                        delib_steps = steps

                    # 2) Append deliberation and re-score Stage-2A/2B
                    aug_prompt = stage_prompt_used.strip() + "\n\n" + delib_text + "\n\n"
                    try:
                        ans_k, stage_k, meta_k = score_answer_stage_ab(
                            prompt=aug_prompt,
                            choices=stage_choices_used,
                            model=score_model or generator_model,
                            backend_cfg=cfg,
                            correct=stage_correct_used,
                            other_token=stage_other_used,
                            temperature=temperature,
                            top_logprobs=top_logprobs,
                            reasoning=reasoning,
                            choices_are_codes=choices_are_codes,
                            vllm_multitoken_fallback=local_vllm_fallback,
                        )
                        pred_k = stage_k.stage2a.predicted
                        stage2a_k = {
                            "gate_success": stage_k.stage2a.gate_success,
                            "predicted": stage_k.stage2a.predicted,
                            "gate_gap_lower": stage_k.stage2a.gate_gap_lower,
                            "cert_candidate_over_other": stage_k.stage2a.cert_candidate_over_other,
                        }
                        stage2b_k = None
                        if stage_k.stage2b is not None:
                            stage2b_k = {
                                "binding_correct": stage_k.stage2b.binding_correct,
                                "correct": stage_k.stage2b.correct,
                                "predicted": stage_k.stage2b.predicted,
                                "value_gap_lower": stage_k.stage2b.value_gap_lower,
                                "cert_correct_worse_than_pred": stage_k.stage2b.cert_correct_worse_than_pred,
                            }
                        err = None
                    except Exception as e:
                        pred_k = ""
                        stage2a_k = None
                        stage2b_k = None
                        err = str(e)

                    deliberation_sweep_results.append(
                        {
                            "k": int(k),
                            "kind": kind,
                            "predicted": pred_k,
                            "drift_from_base": (pred_k != base_pred) if (base_pred is not None and pred_k is not None) else None,
                            "stage2a": stage2a_k,
                            "stage2b": stage2b_k,
                            "deliberation_steps": delib_steps,
                            "error": err,
                        }
                    )



    # Trace generation (must happen after scoring if we condition on the scored answer)
    trace = generate_reasoning_trace(
        prompt=prompt,
        model=generator_model,
        backend_cfg=cfg,
        max_steps=max_steps,
        temperature=temperature,
        reasoning=reasoning,
        given_answer=scored_answer,
    )

    # Grounding checks (sequential, premise-aware)
    checks = verify_trace_steps(trace=trace, verifier_model=verifier_model, backend_cfg=cfg, temperature=temperature, reasoning=reasoning)

    hall = sum(1 for c in checks if c.verdict in ("CONTRADICTED", "NOT_IN_CONTEXT"))
    n_not = sum(1 for c in checks if c.verdict == "NOT_IN_CONTEXT")
    n_con = sum(1 for c in checks if c.verdict == "CONTRADICTED")
    n_unv = sum(1 for c in checks if c.verdict == "UNVERIFIABLE")
    n_ent = sum(1 for c in checks if c.verdict == "ENTAILED")
    grounded_fraction = (n_ent / len(checks)) if checks else 0.0

    # Trace-budget (batched across steps; posterior and pseudo-prior evaluated separately)
    budget_payload: Optional[List[Dict[str, Any]]] = None
    budget_flagged_steps = 0
    max_gap_min: Optional[float] = None
    max_gap_max: Optional[float] = None
    if trace_budget and trace.steps:
        bm = budget_model or verifier_model
        budget_results = score_trace_budget(
            trace=trace,
            verifier_model=bm,
            backend_cfg=cfg,
            default_target=float(budget_target),
            temperature=temperature,
            top_logprobs=int(budget_top_logprobs),
            placeholder=str(budget_placeholder),
            reasoning=reasoning,
        )
        budget_payload = []
        for br in budget_results:
            budget_payload.append(
                {
                    "idx": br.idx,
                    "claim": br.claim,
                    "cites": br.cites,
                    "target": br.target,
                    "prior_yes": {
                        "p_yes_lower": br.prior_yes.p_yes_lower,
                        "p_yes_upper": br.prior_yes.p_yes_upper,
                        "generated": br.prior_yes.generated,
                        "generated_logprob": br.prior_yes.generated_logprob,
                        "kth_logprob": br.prior_yes.kth_logprob,
                    },
                    "post_yes": {
                        "p_yes_lower": br.post_yes.p_yes_lower,
                        "p_yes_upper": br.post_yes.p_yes_upper,
                        "generated": br.post_yes.generated,
                        "generated_logprob": br.post_yes.generated_logprob,
                        "kth_logprob": br.post_yes.kth_logprob,
                    },
                    "required_bits_min": br.required_bits_min,
                    "required_bits_max": br.required_bits_max,
                    "observed_bits_min": br.observed_bits_min,
                    "observed_bits_max": br.observed_bits_max,
                    "budget_gap_min": br.budget_gap_min,
                    "budget_gap_max": br.budget_gap_max,
                    "flagged": br.flagged,
                }
            )
        budget_flagged_steps = int(sum(1 for b in budget_results if b.flagged))
        if budget_results:
            # "max gap" is the most under-budget step (largest positive gap)
            max_gap_min = float(max(br.budget_gap_min for br in budget_results))
            max_gap_max = float(max(br.budget_gap_max for br in budget_results))

    # Procedural check: can an executor derive the answer from the trace?
    exec_model = executor_model or verifier_model
    exec_ans, determined, _notes = answer_from_trace(
        question_hint=question_hint or prompt,
        trace=trace,
        executor_model=exec_model,
        backend_cfg=cfg,
        temperature=temperature,
        reasoning=reasoning,
    )
    answer_match = (exec_ans.strip() == trace.answer.strip()) and determined

    report = TraceReport(
        answer=trace.answer,
        answer_from_trace=exec_ans,
        answer_match=bool(answer_match),
        grounded_fraction=float(grounded_fraction),
        hallucinated_steps=int(hall),
        not_in_context_steps=int(n_not),
        contradicted_steps=int(n_con),
        unverifiable_steps=int(n_unv),
        checks=checks,
        scored_answer=scored_answer,
        stage2a=stage2a,
        stage2b=stage2b,
        stage_meta=stage_meta,
        stage_ab=stage_ab_status,
        trace_budget=budget_payload,
        budget_flagged_steps=budget_flagged_steps,
        max_budget_gap_min=max_gap_min,
        max_budget_gap_max=max_gap_max,
        deliberation_sweep=deliberation_sweep_results,
    )
    return trace, report
