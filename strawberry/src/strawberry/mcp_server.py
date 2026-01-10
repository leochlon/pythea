#!/usr/bin/env python3
"""
MCP server: hallucination / citation confabulation detector.

Exposes:
  - detect_hallucination(answer, spans, ...)
  - audit_trace_budget(steps, spans, ...)

Supports two backends:
  - OpenAI API (default): Set OPENAI_API_KEY environment variable
  - Azure OpenAI pool: Set AOAI_POOL_JSON to path of pool config

Runs over STDIO for Claude Code.

IMPORTANT (stdio servers):
- Never print to stdout (it will corrupt JSON-RPC).
- Use logging (stderr) instead.
"""

from __future__ import annotations

import logging
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Ensure stderr logging only
logger = logging.getLogger("hallucination-detector-mcp")
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="[%(name)s] %(levelname)s: %(message)s",
)


# ----------------------------
# Minimal object types expected by strawberry.trace_budget
# ----------------------------
@dataclass
class Span:
    sid: str
    text: str


@dataclass
class Step:
    idx: int
    claim: str
    cites: List[str]
    confidence: float


@dataclass
class Trace:
    steps: List[Step]
    spans: List[Span]


# ----------------------------
# Helpers
# ----------------------------
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# Default: citations like [S0], [S12], or [12]
_DEFAULT_CITE_RE = re.compile(r"\[(?P<id>[A-Za-z]\w*|\d+)\]")

_LN2 = math.log(2.0)


def _to_bits(nats: float) -> float:
    return float(nats) / _LN2


def _normalize_spans(spans: List[Dict[str, str]]) -> List[Span]:
    out: List[Span] = []
    for s in spans or []:
        sid = str(s.get("sid", "")).strip()
        text = str(s.get("text", "")).strip()
        if sid and text:
            out.append(Span(sid=sid, text=text))
    return out


def _normalize_steps(steps: List[Dict[str, Any]], default_target: float) -> List[Step]:
    out: List[Step] = []
    for i, st in enumerate(steps or []):
        claim = str(st.get("claim", "")).strip()
        if not claim:
            continue
        idx = int(st.get("idx", i))
        cites = [str(c).strip() for c in (st.get("cites") or []) if str(c).strip()]
        conf = float(st.get("confidence", default_target) or default_target)
        out.append(Step(idx=idx, claim=claim, cites=cites, confidence=conf))
    out.sort(key=lambda x: x.idx)
    return out


def _extract_cites(text: str, cite_re: re.Pattern) -> List[str]:
    return [m.group("id") for m in cite_re.finditer(text or "")]


def _split_claims(answer: str, mode: str, max_claims: int) -> List[str]:
    a = (answer or "").strip()
    if not a:
        return []

    if mode == "lines":
        raw = [ln.strip() for ln in a.splitlines() if ln.strip()]
    else:
        raw = [s.strip() for s in _SENTENCE_SPLIT_RE.split(a) if s.strip()]

    return raw[: max(1, int(max_claims))]


def _map_cites_to_known_ids(cites: List[str], known: set) -> List[str]:
    """
    Best-effort mapping so numeric cites can match either:
      - "12" (span id "12")
      - "S12" (span id "S12")
      - "S11" if caller uses 1-based in answer but spans are 0-based (common)
    """
    mapped: List[str] = []
    for c in cites:
        if c in known:
            mapped.append(c)
            continue

        if c.isdigit():
            n = int(c)
            if f"S{n}" in known:
                mapped.append(f"S{n}")
                continue
            if n > 0 and f"S{n-1}" in known:
                mapped.append(f"S{n-1}")
                continue

        if c.startswith("S") and c[1:].isdigit():
            tail = c[1:]
            if tail in known:
                mapped.append(tail)
                continue

        mapped.append(c)

    # de-dupe preserving order
    seen = set()
    out = []
    for c in mapped:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _format_result(r, units: str) -> Dict[str, Any]:
    """Format a BudgetResult for JSON output."""
    if units == "bits":
        req_min = _to_bits(r.required_bits_min)
        req_max = _to_bits(r.required_bits_max)
        obs_min = _to_bits(r.observed_bits_min)
        obs_max = _to_bits(r.observed_bits_max)
        gap_min = _to_bits(r.budget_gap_min)
        gap_max = _to_bits(r.budget_gap_max)
    else:
        req_min, req_max = r.required_bits_min, r.required_bits_max
        obs_min, obs_max = r.observed_bits_min, r.observed_bits_max
        gap_min, gap_max = r.budget_gap_min, r.budget_gap_max

    return {
        "idx": r.idx,
        "claim": r.claim,
        "cites": r.cites,
        "target": r.target,
        "prior_yes": {
            "p_lower": r.prior_yes.p_yes_lower,
            "p_upper": r.prior_yes.p_yes_upper,
            "generated": r.prior_yes.generated,
            "topk": r.prior_yes.topk,
        },
        "post_yes": {
            "p_lower": r.post_yes.p_yes_lower,
            "p_upper": r.post_yes.p_yes_upper,
            "generated": r.post_yes.generated,
            "topk": r.post_yes.topk,
        },
        "required": {"min": req_min, "max": req_max, "units": units},
        "observed": {"min": obs_min, "max": obs_max, "units": units},
        "budget_gap": {"min": gap_min, "max": gap_max, "units": units},
        "flagged": bool(r.flagged),
        "has_any_citations": bool(r.cites),
    }


# ----------------------------
# Core detection logic
# ----------------------------

def run_detect_hallucination(
    answer: str,
    spans: List[Dict[str, str]],
    pool_json_path: Optional[str] = None,
    verifier_model: str = "gpt-4o-mini",
    default_target: float = 0.95,
    placeholder: str = "[REDACTED]",
    max_claims: int = 25,
    claim_split: str = "sentences",
    citation_regex: Optional[str] = None,
    temperature: float = 0.0,
    top_logprobs: int = 10,
    max_concurrency: int = 8,
    timeout_s: Optional[float] = 30.0,
    units: str = "bits",
) -> Dict[str, Any]:
    """
    Given a fully-written answer containing citations and the cited spans,
    compute an information-budget (scrub -> p0/p1 -> KL) diagnostic per claim.
    """
    from .backend import BackendConfig
    from .trace_budget import score_trace_budget

    span_objs = _normalize_spans(spans)
    if not span_objs:
        return {
            "flagged": True,
            "under_budget": True,
            "error": "No spans provided (cannot verify citations).",
            "details": [],
        }

    cite_re = re.compile(citation_regex) if citation_regex else _DEFAULT_CITE_RE
    known_ids = {s.sid for s in span_objs}
    claims = _split_claims(answer, mode=claim_split, max_claims=max_claims)

    steps: List[Step] = []
    for i, cl in enumerate(claims):
        cites = _extract_cites(cl, cite_re=cite_re)
        cites = _map_cites_to_known_ids(cites, known=known_ids)
        steps.append(Step(idx=i, claim=cl, cites=cites, confidence=float(default_target)))

    trace = Trace(steps=steps, spans=span_objs)

    # Choose backend: aoai_pool if pool config provided, otherwise openai
    if pool_json_path:
        cfg = BackendConfig(
            kind="aoai_pool",
            aoai_pool_json_path=pool_json_path,
            max_concurrency=int(max_concurrency),
            timeout_s=timeout_s,
        )
        backend_name = "aoai_pool"
    else:
        cfg = BackendConfig(
            kind="openai",
            max_concurrency=int(max_concurrency),
            timeout_s=timeout_s,
        )
        backend_name = "openai"

    results = score_trace_budget(
        trace=trace,
        verifier_model=verifier_model,
        backend_cfg=cfg,
        default_target=float(default_target),
        temperature=float(temperature),
        top_logprobs=int(top_logprobs),
        placeholder=str(placeholder),
        reasoning=None,
    )

    details = [_format_result(r, units) for r in results]
    flagged = any(d["flagged"] for d in details)
    flagged_idxs = [d["idx"] for d in details if d["flagged"]]

    return {
        "flagged": flagged,
        "under_budget": flagged,
        "summary": {
            "claims_scored": len(details),
            "flagged_claims": len(flagged_idxs),
            "flagged_idxs": flagged_idxs[:50],
            "units": units,
            "verifier_model": verifier_model,
            "backend": backend_name,
        },
        "details": details,
    }


def run_audit_trace_budget(
    steps: List[Dict[str, Any]],
    spans: List[Dict[str, str]],
    pool_json_path: Optional[str] = None,
    verifier_model: str = "gpt-4o-mini",
    default_target: float = 0.95,
    placeholder: str = "[REDACTED]",
    temperature: float = 0.0,
    top_logprobs: int = 10,
    max_concurrency: int = 8,
    timeout_s: Optional[float] = 30.0,
    units: str = "bits",
) -> Dict[str, Any]:
    """
    Lower-level / more reliable entrypoint:
    provide already-atomic steps with explicit cite IDs, plus spans.
    """
    from .backend import BackendConfig
    from .trace_budget import score_trace_budget

    span_objs = _normalize_spans(spans)
    step_objs = _normalize_steps(steps, default_target=float(default_target))

    trace = Trace(steps=step_objs, spans=span_objs)

    # Choose backend: aoai_pool if pool config provided, otherwise openai
    if pool_json_path:
        cfg = BackendConfig(
            kind="aoai_pool",
            aoai_pool_json_path=pool_json_path,
            max_concurrency=int(max_concurrency),
            timeout_s=timeout_s,
        )
        backend_name = "aoai_pool"
    else:
        cfg = BackendConfig(
            kind="openai",
            max_concurrency=int(max_concurrency),
            timeout_s=timeout_s,
        )
        backend_name = "openai"

    results = score_trace_budget(
        trace=trace,
        verifier_model=verifier_model,
        backend_cfg=cfg,
        default_target=float(default_target),
        temperature=float(temperature),
        top_logprobs=int(top_logprobs),
        placeholder=str(placeholder),
        reasoning=None,
    )

    out = []
    for r in results:
        if units == "bits":
            out.append({
                "idx": r.idx,
                "claim": r.claim,
                "cites": r.cites,
                "flagged": bool(r.flagged),
                "required": {"min": _to_bits(r.required_bits_min), "max": _to_bits(r.required_bits_max), "units": "bits"},
                "observed": {"min": _to_bits(r.observed_bits_min), "max": _to_bits(r.observed_bits_max), "units": "bits"},
                "budget_gap": {"min": _to_bits(r.budget_gap_min), "max": _to_bits(r.budget_gap_max), "units": "bits"},
            })
        else:
            out.append({
                "idx": r.idx,
                "claim": r.claim,
                "cites": r.cites,
                "flagged": bool(r.flagged),
                "required": {"min": r.required_bits_min, "max": r.required_bits_max, "units": "nats"},
                "observed": {"min": r.observed_bits_min, "max": r.observed_bits_max, "units": "nats"},
                "budget_gap": {"min": r.budget_gap_min, "max": r.budget_gap_max, "units": "nats"},
            })

    flagged = any(x["flagged"] for x in out)
    return {
        "flagged": flagged,
        "under_budget": flagged,
        "summary": {
            "steps_scored": len(out),
            "flagged_steps": sum(1 for x in out if x["flagged"]),
            "units": units,
            "verifier_model": verifier_model,
            "backend": backend_name,
        },
        "details": out,
    }


# ----------------------------
# MCP server setup
# ----------------------------

def create_mcp_server(pool_json_path: Optional[str] = None):
    """Create and configure the MCP server.

    Args:
        pool_json_path: Optional path to Azure OpenAI pool config.
                       If None, uses standard OpenAI API with OPENAI_API_KEY.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        raise ImportError(
            "MCP SDK not installed. Run: pip install 'mcp[cli]'"
        )

    mcp = FastMCP("hallucination-detector", json_response=True)

    @mcp.tool()
    def detect_hallucination(
        answer: str,
        spans: List[Dict[str, str]],
        verifier_model: str = "gpt-4o-mini",
        default_target: float = 0.95,
        placeholder: str = "[REDACTED]",
        max_claims: int = 25,
        claim_split: str = "sentences",
        citation_regex: Optional[str] = None,
        temperature: float = 0.0,
        top_logprobs: int = 10,
        max_concurrency: int = 8,
        timeout_s: Optional[float] = 30.0,
        units: str = "bits",
    ) -> Dict[str, Any]:
        """
        Given a fully-written answer containing citations and the cited spans,
        compute an information-budget (scrub -> p0/p1 -> KL) diagnostic per claim.

        Args:
            answer: The final answer text containing citations like [S0], [S1], etc.
            spans: List of {"sid": "S0", "text": "..."} objects for each cited span.
            verifier_model: Model to use for verification (e.g. gpt-4o-mini).
            default_target: Target confidence level (default 0.95).
            placeholder: Text to replace scrubbed citations with.
            max_claims: Maximum number of claims to process.
            claim_split: How to split answer into claims ("sentences" or "lines").
            citation_regex: Custom regex for extracting citation IDs.
            temperature: Sampling temperature for verifier.
            top_logprobs: Number of top logprobs to request.
            max_concurrency: Maximum concurrent requests.
            timeout_s: Timeout per request in seconds.
            units: Output units ("bits" or "nats").

        Returns:
            Dict with flagged status, summary, and per-claim details.
        """
        return run_detect_hallucination(
            answer=answer,
            spans=spans,
            pool_json_path=pool_json_path,
            verifier_model=verifier_model,
            default_target=default_target,
            placeholder=placeholder,
            max_claims=max_claims,
            claim_split=claim_split,
            citation_regex=citation_regex,
            temperature=temperature,
            top_logprobs=top_logprobs,
            max_concurrency=max_concurrency,
            timeout_s=timeout_s,
            units=units,
        )

    @mcp.tool()
    def audit_trace_budget(
        steps: List[Dict[str, Any]],
        spans: List[Dict[str, str]],
        verifier_model: str = "gpt-4o-mini",
        default_target: float = 0.95,
        placeholder: str = "[REDACTED]",
        temperature: float = 0.0,
        top_logprobs: int = 10,
        max_concurrency: int = 8,
        timeout_s: Optional[float] = 30.0,
        units: str = "bits",
    ) -> Dict[str, Any]:
        """
        Lower-level / more reliable entrypoint:
        provide already-atomic steps with explicit cite IDs, plus spans.

        Args:
            steps: List of {"idx": 0, "claim": "...", "cites": ["S0"], "confidence": 0.95}.
            spans: List of {"sid": "S0", "text": "..."} objects.
            verifier_model: Model to use for verification (e.g. gpt-4o-mini).
            default_target: Default confidence if not specified per-step.
            placeholder: Text to replace scrubbed citations with.
            temperature: Sampling temperature for verifier.
            top_logprobs: Number of top logprobs to request.
            max_concurrency: Maximum concurrent requests.
            timeout_s: Timeout per request in seconds.
            units: Output units ("bits" or "nats").

        Returns:
            Dict with flagged status, summary, and per-step details.
        """
        return run_audit_trace_budget(
            steps=steps,
            spans=spans,
            pool_json_path=pool_json_path,
            verifier_model=verifier_model,
            default_target=default_target,
            placeholder=placeholder,
            temperature=temperature,
            top_logprobs=top_logprobs,
            max_concurrency=max_concurrency,
            timeout_s=timeout_s,
            units=units,
        )

    return mcp


def main() -> None:
    """Entry point for the MCP server."""
    # Get pool config path from environment or command line (optional)
    pool_json_path = os.environ.get("AOAI_POOL_JSON")

    if not pool_json_path and len(sys.argv) > 1:
        pool_json_path = sys.argv[1]

    # Validate pool config if provided
    if pool_json_path:
        if not os.path.exists(pool_json_path):
            logger.error(f"Pool config file not found: {pool_json_path}")
            sys.exit(1)
        logger.info(f"Starting hallucination-detector MCP server with Azure OpenAI pool: {pool_json_path}")
    else:
        # Check for OpenAI API key
        if not os.environ.get("OPENAI_API_KEY"):
            logger.error(
                "No backend configured. Either:\n"
                "  1. Set OPENAI_API_KEY environment variable for OpenAI API, or\n"
                "  2. Provide Azure pool config: python -m strawberry.mcp_server /path/to/aoai_pool.json\n"
                "     or set AOAI_POOL_JSON environment variable"
            )
            sys.exit(1)
        logger.info("Starting hallucination-detector MCP server with OpenAI API")

    mcp = create_mcp_server(pool_json_path)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
