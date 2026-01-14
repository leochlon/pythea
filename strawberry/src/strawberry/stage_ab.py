"""strawberry.stage_ab

Output-level Stage-2A / Stage-2B metrics using Responses API logprobs.

This module implements the *output-side* analog of the Stage 2A / Stage 2B
decomposition from the proximity/binding paper:

  - Stage 2A (gating): does the model enter "answer mode" (choose a candidate)
    vs abstain (choose OTHER)?
  - Stage 2B (binding): conditional on answer mode, does the model choose the
    correct candidate vs the best competing candidate?

We compute these using the per-token logprobs returned by the Responses API
when called with:
  include=["message.output_text.logprobs"] and top_logprobs in [0,20].

Important caveats
-----------------
1) The API only returns the top-K alternatives per output token position. If a
   candidate token is not present in top-K, we do not know its exact logprob.
   However, we *do* get a rigorous upper bound: its logprob is <= the K-th
   logprob at that position. We use this to produce **certificates** of sign
   (e.g., to certify that correct < competitor).

2) These metrics are most reliable when each candidate corresponds to a
   single-token first piece at the answer position. For larger multi-token
   candidates, Stage metrics become approximate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import math


def _as_dict(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    out: Dict[str, Any] = {}
    for k in dir(x):
        if k.startswith("_"):
            continue
        try:
            v = getattr(x, k)
        except Exception:
            continue
        if callable(v):
            continue
        out[k] = v
    return out


def _get_token(obj: Any) -> str:
    d = _as_dict(obj)
    tok = d.get("token")
    if tok is None:
        # some SDKs use "text" for tokens
        tok = d.get("text")
    return "" if tok is None else str(tok)


def _get_logprob(obj: Any) -> Optional[float]:
    d = _as_dict(obj)
    lp = d.get("logprob")
    if lp is None:
        lp = d.get("log_prob")
    try:
        return None if lp is None else float(lp)
    except Exception:
        return None


def _get_top_logprobs(obj: Any) -> List[Any]:
    d = _as_dict(obj)
    top = d.get("top_logprobs")
    if top is None:
        top = d.get("top_log_probs")
    if top is None:
        return []
    return list(top)


@dataclass
class TokenTopK:
    """Top-K distribution at the answer-start token position."""

    pos: int
    generated_token: str
    generated_logprob: float
    # Map from canonical token (lstrip'd) -> logprob
    topk_logprobs: Dict[str, float]
    # The smallest logprob among returned top-k tokens (upper bound for any missing token)
    kth_logprob: Optional[float]


def extract_answer_topk(logprobs: Any) -> TokenTopK:
    """Extract a token-level top-K distribution for the *first non-whitespace* output token.

    The Responses API returns logprobs for each generated output token.
    Models sometimes emit leading whitespace/newlines; we skip those.
    """
    if logprobs is None:
        raise ValueError("logprobs is None; call the API with include=['message.output_text.logprobs']")

    seq = list(logprobs)
    if not seq:
        raise ValueError("empty logprobs list")

    pos = 0
    for i, tokinfo in enumerate(seq):
        tok = _get_token(tokinfo)
        if tok.strip() != "":
            pos = i
            break

    tokinfo = seq[pos]
    gen_tok = _get_token(tokinfo)
    gen_lp = _get_logprob(tokinfo)
    if gen_lp is None:
        raise ValueError("missing logprob for generated token")

    top_list = _get_top_logprobs(tokinfo)
    topk: Dict[str, float] = {}
    for t in top_list:
        tt = _get_token(t)
        lp = _get_logprob(t)
        if lp is None:
            continue
        key = tt.lstrip()  # canonicalize leading whitespace
        if key == "":
            continue
        # Keep max logprob if duplicates appear
        topk[key] = max(topk.get(key, -math.inf), float(lp))

    kth = None
    if top_list:
        lps = [lp for lp in ([_get_logprob(t) for t in top_list]) if lp is not None]
        kth = min(lps) if lps else None

    return TokenTopK(
        pos=int(pos),
        generated_token=str(gen_tok),
        generated_logprob=float(gen_lp),
        topk_logprobs=topk,
        kth_logprob=kth,
    )


def canonical_choice(s: str) -> str:
    """Canonicalize a model string output to compare with choices.

    We strip whitespace and then take the first whitespace-delimited token.
    """
    s = str(s).strip()
    if not s:
        return ""
    return s.split()[0]


def logprob_upper(topk: TokenTopK, choice: str) -> float:
    """Upper bound on logprob(choice) at the answer-start position."""
    c = str(choice)
    if c in topk.topk_logprobs:
        return float(topk.topk_logprobs[c])
    # Missing from top-k: logprob <= kth_logprob (if available)
    if topk.kth_logprob is None:
        return float("inf")
    return float(topk.kth_logprob)


def logprob_lower(topk: TokenTopK, choice: str) -> float:
    """Lower bound on logprob(choice) at the answer-start position."""
    c = str(choice)
    if c in topk.topk_logprobs:
        return float(topk.topk_logprobs[c])
    return float("-inf")


def set_max_upper(topk: TokenTopK, choices: Sequence[str]) -> float:
    """Upper bound on max_{c in choices} logprob(c)."""
    known = [topk.topk_logprobs[c] for c in choices if c in topk.topk_logprobs]
    m_known = max(known) if known else float("-inf")
    if topk.kth_logprob is None:
        return m_known
    return max(m_known, float(topk.kth_logprob))


@dataclass
class Stage2A:
    gate_success: bool  # predicted != OTHER
    predicted: str
    # lower bound on (best_candidate - OTHER)
    gate_gap_lower: Optional[float]
    # certificate: best_candidate definitely beats OTHER?
    cert_candidate_over_other: bool
    coverage_candidates_in_topk: float


@dataclass
class Stage2B:
    binding_evaluated: bool  # only if gate_success and correct provided
    binding_correct: Optional[bool]
    correct: Optional[str]
    predicted: str
    # Upper bound on value gap: logp(correct) - logp(best competitor)
    value_gap_upper: Optional[float]
    # Lower bound on value gap (often -inf unless all tokens observed)
    value_gap_lower: Optional[float]
    # certificate that correct is definitely worse than predicted competitor
    cert_correct_worse_than_pred: bool
    coverage_candidates_in_topk: float


@dataclass
class StageABReport:
    topk: TokenTopK
    stage2a: Stage2A
    stage2b: Stage2B


def compute_stage_ab(
    *,
    generated_text: str,
    logprobs: Any,
    choices: Sequence[str],
    correct: Optional[str] = None,
    other_token: str = "OTHER",
) -> StageABReport:
    """Compute Stage 2A / Stage 2B metrics from a single answer-only generation."""
    if other_token not in choices:
        raise ValueError("choices must include OTHER for Stage 2A")

    topk = extract_answer_topk(logprobs)

    cand = [c for c in choices if c != other_token]
    pred = canonical_choice(generated_text)
    if pred not in choices:
        # Defensive fallback
        pred = other_token

    gate_success = pred != other_token

    # Coverage: how many candidates appear explicitly in top-k list
    cov = 0.0
    if cand:
        cov = sum(1 for c in cand if c in topk.topk_logprobs) / float(len(cand))

    gate_gap_lower = None
    cert_c_over_o = False
    if gate_success:
        # best candidate is the generated token (top-1), assume it is a candidate
        best_c_lp = float(topk.generated_logprob)
        other_up = logprob_upper(topk, other_token)
        if math.isfinite(other_up):
            gate_gap_lower = best_c_lp - other_up
            cert_c_over_o = gate_gap_lower > 0

    stage2a = Stage2A(
        gate_success=bool(gate_success),
        predicted=str(pred),
        gate_gap_lower=gate_gap_lower,
        cert_candidate_over_other=bool(cert_c_over_o),
        coverage_candidates_in_topk=float(cov),
    )

    # Stage 2B
    binding_eval = bool(gate_success and (correct is not None))
    binding_correct = None
    value_gap_upper = None
    value_gap_lower = None
    cert_correct_worse = False
    corr = str(correct) if correct is not None else None

    if binding_eval:
        assert corr is not None
        binding_correct = (pred == corr)

        # Competitor: the predicted token (since it is top-1 overall)
        comp = pred
        comp_low = float(topk.generated_logprob)

        corr_up = logprob_upper(topk, corr)
        if math.isfinite(corr_up):
            value_gap_upper = float(corr_up) - comp_low
            # If even the upper bound is below competitor lower bound => certified misbinding
            cert_correct_worse = corr_up < comp_low

        # Lower bound is only finite if we know corr exactly and have an upper bound on competitor.
        corr_low = logprob_lower(topk, corr)
        comp_up = logprob_upper(topk, comp)
        if math.isfinite(corr_low) and math.isfinite(comp_up):
            value_gap_lower = float(corr_low) - float(comp_up)

    stage2b = Stage2B(
        binding_evaluated=bool(binding_eval),
        binding_correct=binding_correct,
        correct=corr,
        predicted=str(pred),
        value_gap_upper=value_gap_upper,
        value_gap_lower=value_gap_lower,
        cert_correct_worse_than_pred=bool(cert_correct_worse),
        coverage_candidates_in_topk=float(cov),
    )

    return StageABReport(topk=topk, stage2a=stage2a, stage2b=stage2b)
