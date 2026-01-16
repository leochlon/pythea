#!/usr/bin/env python3
"""strawberry.nala_mcp_server

An MCP server for *autonomous math research + Lean proof work* with Strawberry
hallucination/citation gating as the **enforcement layer**.

Design goal
-----------
Put the hallucination detector at the center of decision making:

* The agent must create an *audited micro-plan* (claim+cites steps) before it can
  invoke any execution / fetching / ingestion tool.
* After *every* execution (Lean build, command, python, download, etc.), the
  server invalidates the current plan, forcing the agent to re-audit its next
  decision using updated evidence.

This replaces "skills" / prompt-only discipline with a server-side state machine.

Tools exposed
------------
State + evidence
  - start_run, load_run, list_spans, get_span
  - add_span, add_file_span
  - search_spans (TF-IDF over span texts)

Planning (gated)
  - set_microplan, get_microplan, clear_microplan

Evidence generation (gated, requires plan step)
  - run_cmd, rg_search
  - lean_build, lean_query
  - python_run, cpp_run
  - arxiv_search, fetch_url, download_url, pdf_extract

Detection
  - detect_hallucination, audit_trace_budget

Backend
-------
Uses the same backends as strawberry.mcp_server:
  - OpenAI API (OPENAI_API_KEY)
  - Azure OpenAI pool (AOAI_POOL_JSON or first CLI arg)

IMPORTANT (stdio servers):
- Never print to stdout (it corrupts JSON-RPC). Use logging (stderr).
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tarfile
import textwrap
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


logger = logging.getLogger("nala-mcp")
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="[%(name)s] %(levelname)s: %(message)s",
)


# ----------------------------
# Strawberry imports
# ----------------------------

try:
    from .mcp_server import run_audit_trace_budget, run_detect_hallucination
except Exception as e:  # pragma: no cover
    raise RuntimeError(f"Failed to import strawberry.mcp_server helpers: {e}")


# ----------------------------
# Data model
# ----------------------------


@dataclass
class SpanMeta:
    sid: str
    kind: str
    source: str
    created_at: float
    path: str
    extra: Dict[str, Any]


def _now() -> float:
    return time.time()


def _safe_slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    return s or "run"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_text(path: Path, *, max_chars: int = 200_000) -> str:
    txt = path.read_text(errors="replace")
    if len(txt) > max_chars:
        return txt[:max_chars] + "\n\n[TRUNCATED]"
    return txt


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", errors="replace")


def _strip_html(html: str) -> str:
    """Very small HTML -> text stripper (no external deps)."""
    # Remove script/style blocks
    html = re.sub(r"<\s*(script|style)[^>]*>.*?<\s*/\s*\1\s*>", " ", html, flags=re.I | re.S)
    # Replace <br> and <p> with newlines
    html = re.sub(r"<\s*br\s*/?\s*>", "\n", html, flags=re.I)
    html = re.sub(r"<\s*/\s*p\s*>", "\n", html, flags=re.I)
    # Drop remaining tags
    txt = re.sub(r"<[^>]+>", " ", html)
    # Unescape a few common entities
    txt = txt.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()


# ----------------------------
# Run state
# ----------------------------


class RunState:
    def __init__(self) -> None:
        self.run_dir: Optional[Path] = None
        self.state_path: Optional[Path] = None
        self.spans: List[SpanMeta] = []
        self.attempts: List[Dict[str, Any]] = []
        self.microplan: Optional[List[Dict[str, Any]]] = None
        self.plan_version: int = 0
        # After every execution (cmd/build/fetch/etc.) we force a
        # "check-in" step before the agent is allowed to propose the next
        # microplan. This prevents the failure mode where a plan is audited,
        # an action is taken, but the model doesn't actually read/validate the
        # tool output and drifts away from the intended microplan.
        #
        # Shape:
        #   {
        #     "attempt_id": int,
        #     "ts": float,
        #   }
        self.pending_checkin: Optional[Dict[str, Any]] = None

    # ---------- persistence ----------
    def _state_obj(self) -> Dict[str, Any]:
        return {
            "run_dir": str(self.run_dir) if self.run_dir else None,
            "spans": [
                {
                    "sid": s.sid,
                    "kind": s.kind,
                    "source": s.source,
                    "created_at": s.created_at,
                    "path": s.path,
                    "extra": s.extra,
                }
                for s in self.spans
            ],
            "attempts": self.attempts,
            "microplan": self.microplan,
            "plan_version": self.plan_version,
            "pending_checkin": self.pending_checkin,
        }

    def save(self) -> None:
        if not self.state_path:
            return
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state_obj(), indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.state_path)

    def load(self, run_dir: Path) -> None:
        run_dir = run_dir.resolve()
        state_path = run_dir / "state.json"
        if not state_path.exists():
            raise FileNotFoundError(f"state.json not found in {run_dir}")
        data = json.loads(state_path.read_text(encoding="utf-8"))
        self.run_dir = run_dir
        self.state_path = state_path
        self.spans = [
            SpanMeta(
                sid=str(d["sid"]),
                kind=str(d.get("kind", "note")),
                source=str(d.get("source", "")),
                created_at=float(d.get("created_at", 0.0)),
                path=str(d.get("path", "")),
                extra=dict(d.get("extra") or {}),
            )
            for d in (data.get("spans") or [])
        ]
        self.attempts = list(data.get("attempts") or [])
        self.microplan = data.get("microplan")
        self.plan_version = int(data.get("plan_version") or 0)
        self.pending_checkin = data.get("pending_checkin")

    # ---------- helpers ----------
    def require_loaded(self) -> Tuple[Path, Path]:
        if not self.run_dir or not self.state_path:
            raise RuntimeError("No active run loaded. Call start_run(...) or load_run(...).")
        return self.run_dir, self.state_path

    def spans_dir(self) -> Path:
        run_dir, _ = self.require_loaded()
        d = run_dir / "spans"
        _ensure_dir(d)
        return d

    def files_dir(self) -> Path:
        run_dir, _ = self.require_loaded()
        d = run_dir / "files"
        _ensure_dir(d)
        return d

    def scripts_dir(self) -> Path:
        run_dir, _ = self.require_loaded()
        d = run_dir / "scripts"
        _ensure_dir(d)
        return d

    def _next_sid(self) -> str:
        return f"S{len(self.spans)}"

    def add_span(self, *, kind: str, source: str, text: str, extra: Optional[Dict[str, Any]] = None) -> str:
        sid = self._next_sid()
        path = self.spans_dir() / f"{sid}.txt"
        _write_text(path, text)
        meta = SpanMeta(
            sid=sid,
            kind=kind,
            source=source,
            created_at=_now(),
            path=str(path.relative_to(self.run_dir)) if self.run_dir else str(path),
            extra=dict(extra or {}),
        )
        self.spans.append(meta)
        self.save()
        return sid

    def get_span_text(self, sid: str) -> str:
        run_dir, _ = self.require_loaded()
        for s in self.spans:
            if s.sid == sid:
                return _read_text(run_dir / s.path)
        raise KeyError(f"Unknown span id: {sid}")

    def all_span_dicts(self, *, max_chars_per_span: int = 200_000) -> List[Dict[str, str]]:
        run_dir, _ = self.require_loaded()
        out: List[Dict[str, str]] = []
        for s in self.spans:
            txt = _read_text(run_dir / s.path, max_chars=max_chars_per_span)
            out.append({"sid": s.sid, "text": txt})
        return out

    # ---------- plan enforcement ----------
    def clear_plan(self) -> None:
        self.microplan = None
        self.save()

    def set_plan(self, steps: List[Dict[str, Any]]) -> None:
        self.require_no_pending_checkin()
        self.microplan = steps
        self.plan_version += 1
        self.save()

    def require_no_pending_checkin(self) -> None:
        if self.pending_checkin is None:
            return
        attempt_id = self.pending_checkin.get("attempt_id")
        raise RuntimeError(
            "A tool was executed, but the run is still pending a post-action check-in. "
            f"Call checkin_last_action(...) first (pending attempt_id={attempt_id})."
        )

    def require_plan_step(self, idx: int) -> Dict[str, Any]:
        if not self.microplan:
            raise RuntimeError("No active microplan. Call set_microplan(...) first.")
        if idx < 0 or idx >= len(self.microplan):
            raise IndexError(f"plan_step_idx {idx} out of range (0..{len(self.microplan)-1})")
        return self.microplan[idx]

    def invalidate_plan_after_execution(
        self,
        *,
        tool: str,
        note: str,
        plan_step_idx: int,
        step_snapshot: Optional[Dict[str, Any]] = None,
        span_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Record an execution attempt, invalidate the current microplan, and require a check-in.

        Returns the attempt_id.
        """
        plan_step_span_id: Optional[str] = None
        if step_snapshot:
            # Persist the *planned* step as an evidence span so the subsequent
            # check-in can cite both:
            #   (a) what we said we would do (microplan step)
            #   (b) what actually happened (tool output)
            # This helps detect plan/action drift.
            try:
                payload = {
                    "plan_version": self.plan_version,
                    "plan_step_idx": int(plan_step_idx),
                    "step": dict(step_snapshot),
                }
                plan_step_span_id = self.add_span(
                    kind="microplan_step",
                    source="set_microplan",
                    text=json.dumps(payload, indent=2, sort_keys=True),
                    extra={"plan_version": self.plan_version, "plan_step_idx": int(plan_step_idx)},
                )
            except Exception as e:  # pragma: no cover
                logger.warning(f"Failed to write microplan_step span: {e}")

        attempt: Dict[str, Any] = {
            "ts": _now(),
            "tool": tool,
            "note": note,
            "span_id": span_id,
            "plan_version": self.plan_version,
            "plan_step_idx": int(plan_step_idx),
            "step": dict(step_snapshot or {}),
            "plan_step_span_id": plan_step_span_id,
        }
        if extra:
            attempt["extra"] = dict(extra)
        self.attempts.append(attempt)
        attempt_id = len(self.attempts) - 1

        # Force a post-action check-in before allowing another microplan.
        self.pending_checkin = {"attempt_id": attempt_id, "ts": _now()}

        # Invalidate plan to enforce evidence-first replanning.
        self.microplan = None
        self.save()
        return attempt_id


STATE = RunState()


# ----------------------------
# Gating
# ----------------------------


class AuditTraceBudgetError(RuntimeError):
    def __init__(self, message: str, report: Dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


def _audit_steps_or_raise(
    *,
    steps: List[Dict[str, Any]],
    verifier_model: str,
    default_target: float,
    pool_json_path: Optional[str],
    units: str,
) -> Dict[str, Any]:
    # Prune span list aggressively: the trace-budget verifier prompt size scales with
    # the number of spans we pass in. Passing the full run trace (potentially 1000+ spans)
    # into *every* claim prompt is both slow and expensive.
    spans_all = STATE.all_span_dicts()

    # Keep only spans that are actually cited by the proposed steps.
    cited: set[str] = set()
    for st in steps or []:
        for c in (st.get("cites") or []):
            if c is not None:
                cited.add(str(c))

    if cited:
        spans = [s for s in spans_all if str(s.get("sid")) in cited]
    else:
        spans = []
    report = run_audit_trace_budget(
        steps=steps,
        spans=spans,
        pool_json_path=pool_json_path,
        verifier_model=verifier_model,
        default_target=default_target,
        units=units,
        # Crucial: verify each claim against its cited evidence only.
        context_mode="cited",
    )
    if bool(report.get("flagged")):
        raise AuditTraceBudgetError(
            "Hallucination detector flagged the plan/step (insufficient evidence). "
            "Revise claims or add evidence spans before proceeding.",
            report,
        )
    return report


def _require_cited_steps(steps: List[Dict[str, Any]]) -> None:
    if not steps:
        raise ValueError("steps must be non-empty")
    missing: List[str] = []
    for i, st in enumerate(steps):
        claim = str(st.get("claim", "")).strip()
        cites = st.get("cites")
        cite_list = cites if isinstance(cites, list) else []
        cite_list = [str(c).strip() for c in cite_list if str(c).strip()]
        parts: List[str] = []
        if not claim:
            parts.append("claim")
        if not cite_list:
            parts.append("cites")
        if parts:
            missing.append(f"step[{i}] missing {', '.join(parts)}")
    if missing:
        raise ValueError(
            "Each microplan step must include a non-empty 'claim' and 'cites' list. "
            + "; ".join(missing)
        )


def _require_allowed_cmd(cmd0: str, *, allow_unsafe: bool) -> None:
    if allow_unsafe or os.environ.get("RESEARCH_MCP_ALLOW_UNSAFE") == "1":
        return
    allow = {
        "rg",
        "grep",
        "sed",
        "awk",
        "cat",
        "ls",
        "find",
        "python",
        "python3",
        "bash",
        "sh",
        "git",
        "lake",
        "lean",
        "g++",
        "clang++",
        "make",
        "cmake",
        "ninja",
        "cargo",
        "wget",
        "curl",
    }
    base = os.path.basename(cmd0)
    if base not in allow:
        raise RuntimeError(
            f"Command '{base}' is not allowlisted. "
            "Set RESEARCH_MCP_ALLOW_UNSAFE=1 or pass allow_unsafe=True if you really want this."
        )


def _require_note_approval(*, approved_by: Optional[str], approval_token: Optional[str]) -> None:
    token = os.environ.get("NALA_MCP_NOTE_APPROVAL_TOKEN") or os.environ.get(
        "RESEARCH_MCP_NOTE_APPROVAL_TOKEN"
    )
    if token:
        if str(approval_token or "") != token:
            raise RuntimeError(
                "add_span kind='note' requires approval_token matching NALA_MCP_NOTE_APPROVAL_TOKEN."
            )
        return
    if not (approved_by or "").strip():
        raise RuntimeError("add_span kind='note' requires approved_by (user approval).")


# ----------------------------
# External fetch helpers
# ----------------------------


def _http_get(url: str, *, timeout_s: float = 30.0, max_bytes: int = 10_000_000) -> Tuple[str, str]:
    """Return (content_type, text_or_bytes_decoded_as_utf8)."""
    r = requests.get(url, timeout=timeout_s, headers={"User-Agent": "nala-mcp/0.1"}, stream=True)
    r.raise_for_status()
    ct = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    buf = io.BytesIO()
    n = 0
    for chunk in r.iter_content(chunk_size=65536):
        if not chunk:
            continue
        n += len(chunk)
        if n > max_bytes:
            break
        buf.write(chunk)
    data = buf.getvalue()
    try:
        txt = data.decode("utf-8", errors="replace")
    except Exception:
        txt = data.decode(errors="replace")
    return ct, txt


def _arxiv_api_search(query: str, *, max_results: int = 10) -> List[Dict[str, Any]]:
    q = (query or "").strip()
    if not q:
        return []
    api = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{q}",
        "start": 0,
        "max_results": int(max_results),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = api + "?" + urllib.parse.urlencode(params)
    ct, txt = _http_get(url, timeout_s=30.0, max_bytes=5_000_000)
    if "xml" not in ct and not txt.lstrip().startswith("<?xml"):
        raise RuntimeError(f"arXiv API returned unexpected content-type: {ct}")

    root = ET.fromstring(txt)
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    out: List[Dict[str, Any]] = []
    for entry in root.findall("atom:entry", ns):
        eid = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
        published = (entry.findtext("atom:published", default="", namespaces=ns) or "").strip()
        updated = (entry.findtext("atom:updated", default="", namespaces=ns) or "").strip()
        authors = [
            (a.findtext("atom:name", default="", namespaces=ns) or "").strip()
            for a in entry.findall("atom:author", ns)
        ]
        links = {
            (ln.get("title") or ln.get("rel") or "link"): (ln.get("href") or "")
            for ln in entry.findall("atom:link", ns)
        }
        pdf_url = ""
        for ln in entry.findall("atom:link", ns):
            if (ln.get("type") or "").lower() == "application/pdf":
                pdf_url = ln.get("href") or ""
                break
        primary_cat = (entry.findtext("arxiv:primary_category", default="", namespaces=ns) or "")
        out.append(
            {
                "id": eid,
                "title": " ".join(title.split()),
                "summary": " ".join(summary.split()),
                "authors": authors,
                "published": published,
                "updated": updated,
                "pdf_url": pdf_url,
                "links": links,
                "primary_category": primary_cat,
            }
        )
    return out


# ----------------------------
# MCP server
# ----------------------------


def create_mcp_server(pool_json_path: Optional[str] = None):
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        raise ImportError("MCP SDK not installed. Run: pip install 'mcp[cli]'")

    mcp = FastMCP("nala-mcp", json_response=True)

    # ----------------------------
    # Detection passthrough
    # ----------------------------

    @mcp.tool()
    def detect_hallucination(
        answer: str,
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
        """Run Strawberry's claim-level hallucination detector over `answer` using current spans."""
        spans = STATE.all_span_dicts()
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
        verifier_model: str = "gpt-4o-mini",
        default_target: float = 0.95,
        placeholder: str = "[REDACTED]",
        context_mode: str = "cited",
        temperature: float = 0.0,
        top_logprobs: int = 10,
        max_concurrency: int = 8,
        timeout_s: Optional[float] = 30.0,
        units: str = "bits",
    ) -> Dict[str, Any]:
        """Direct access to Strawberry's trace-budget audit using current spans."""
        spans = STATE.all_span_dicts()
        return run_audit_trace_budget(
            steps=steps,
            spans=spans,
            pool_json_path=pool_json_path,
            verifier_model=verifier_model,
            default_target=default_target,
            placeholder=placeholder,
            context_mode=context_mode,
            temperature=temperature,
            top_logprobs=top_logprobs,
            max_concurrency=max_concurrency,
            timeout_s=timeout_s,
            units=units,
        )

    # ----------------------------
    # Run lifecycle
    # ----------------------------

    @mcp.tool()
    def start_run(
        problem_statement: str,
        root_dir: str = "./research_runs",
        slug: Optional[str] = None,
        seed_charter: bool = True,
    ) -> Dict[str, Any]:
        """Create a new run directory and seed span S0."""
        root = Path(root_dir).expanduser().resolve()
        _ensure_dir(root)
        slug2 = _safe_slug(slug or problem_statement[:60])
        ts = time.strftime("%Y%m%d-%H%M%S")
        run_dir = root / f"{ts}-{slug2}"
        _ensure_dir(run_dir)
        _ensure_dir(run_dir / "spans")
        _ensure_dir(run_dir / "files")
        _ensure_dir(run_dir / "scripts")

        # initialize
        STATE.run_dir = run_dir
        STATE.state_path = run_dir / "state.json"
        STATE.spans = []
        STATE.attempts = []
        STATE.microplan = None
        STATE.plan_version = 0
        STATE.pending_checkin = None

        # Seed a charter span that makes "evidence collection" steps supportable.
        if seed_charter:
            charter = textwrap.dedent(
                f"""
                PROJECT CHARTER
                ===============
                Problem: {problem_statement.strip()}

                Rules:
                - All actions must be justified by an audited micro-plan (claim+cites).
                - Evidence is collected via tool outputs and saved as spans [S#].
                - After each execution/fetch/build, the plan is invalidated and must be recreated.
                - After each execution/fetch/build, you must *check in* and cite the tool output
                  before creating the next micro-plan (prevents plan/action drift).

                Allowed evidence-generation actions include:
                - Searching and downloading academic sources (e.g., arXiv) and extracting text.
                - Running local computations (Python/C++), and storing code+outputs.
                - Running Lean builds/queries when a Lean project is present.
                """
            ).strip()
            s0 = STATE.add_span(kind="charter", source="start_run", text=charter)
        else:
            s0 = STATE.add_span(kind="problem", source="start_run", text=problem_statement.strip())

        return {"run_dir": str(run_dir), "seed_span": s0}

    @mcp.tool()
    def load_run(run_dir: str) -> Dict[str, Any]:
        """Load an existing run directory (state.json)."""
        STATE.load(Path(run_dir).expanduser().resolve())
        return {"run_dir": str(STATE.run_dir), "num_spans": len(STATE.spans), "plan_version": STATE.plan_version}

    # ----------------------------
    # Evidence spans
    # ----------------------------

    @mcp.tool()
    def list_spans(limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """List span metadata (without full text)."""
        STATE.require_loaded()
        metas = STATE.spans[offset : offset + limit]
        return {
            "run_dir": str(STATE.run_dir),
            "total": len(STATE.spans),
            "spans": [
                {
                    "sid": m.sid,
                    "kind": m.kind,
                    "source": m.source,
                    "created_at": m.created_at,
                    "path": m.path,
                    "extra": m.extra,
                }
                for m in metas
            ],
        }

    @mcp.tool()
    def get_span(sid: str, max_chars: int = 20_000) -> Dict[str, Any]:
        """Fetch a span's full text (truncated) + metadata."""
        run_dir, _ = STATE.require_loaded()
        meta = next((m for m in STATE.spans if m.sid == sid), None)
        if not meta:
            raise KeyError(f"Unknown span id: {sid}")
        txt = _read_text(run_dir / meta.path, max_chars=max_chars)
        return {
            "sid": meta.sid,
            "kind": meta.kind,
            "source": meta.source,
            "created_at": meta.created_at,
            "path": meta.path,
            "extra": meta.extra,
            "text": txt,
        }

    @mcp.tool()
    def add_span(
        text: str,
        kind: str = "note",
        source: str = "manual",
        approved_by: Optional[str] = None,
        approval_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add an evidence span from explicit text."""
        STATE.require_loaded()
        kind_norm = (kind or "").strip().lower()
        extra: Optional[Dict[str, Any]] = None
        if kind_norm == "note":
            _require_note_approval(approved_by=approved_by, approval_token=approval_token)
            extra = {"note_approved": True}
            if (approved_by or "").strip():
                extra["approved_by"] = str(approved_by).strip()
        sid = STATE.add_span(kind=kind, source=source, text=text.strip(), extra=extra)
        return {"sid": sid}

    @mcp.tool()
    def add_file_span(
        path: str,
        start_line: int,
        end_line: int,
        kind: str = "file",
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Capture a slice of a local text file as an evidence span."""
        STATE.require_loaded()
        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(str(p))
        lines = p.read_text(errors="replace").splitlines()
        s = max(1, int(start_line))
        e = max(s, int(end_line))
        s0 = s - 1
        e0 = min(len(lines), e)
        excerpt = "\n".join(lines[s0:e0])
        sid = STATE.add_span(
            kind=kind,
            source=source or f"file:{p}",
            text=excerpt,
            extra={"path": str(p), "start_line": s, "end_line": e0},
        )
        return {"sid": sid, "path": str(p), "start_line": s, "end_line": e0}

    @mcp.tool()
    def search_spans(query: str, top_k: int = 5) -> Dict[str, Any]:
        """Lightweight search over span texts using a tiny TF-IDF implementation.

        Rationale: keep this server dependency-light (no sklearn).
        """
        import math
        from collections import Counter, defaultdict

        STATE.require_loaded()

        def tok(s: str) -> List[str]:
            return [w for w in re.findall(r"[A-Za-z0-9_]+", (s or "").lower()) if len(w) >= 2]

        docs = [tok(STATE.get_span_text(s.sid)) for s in STATE.spans]
        if not docs:
            return {"matches": []}

        N = len(docs)
        df: Dict[str, int] = defaultdict(int)
        for d in docs:
            for w in set(d):
                df[w] += 1

        q = tok(query)
        if not q:
            return {"matches": []}

        q_tf = Counter(q)
        q_vec: Dict[str, float] = {}
        for w, c in q_tf.items():
            idf = math.log((N + 1) / (df.get(w, 0) + 1)) + 1.0
            q_vec[w] = float(c) * idf

        def norm(v: Dict[str, float]) -> float:
            return math.sqrt(sum(x * x for x in v.values())) or 1.0

        qn = norm(q_vec)
        scores: List[Tuple[int, float]] = []
        for i, d in enumerate(docs):
            tf = Counter(d)
            v: Dict[str, float] = {}
            for w in q_vec.keys():
                if w in tf:
                    idf = math.log((N + 1) / (df.get(w, 0) + 1)) + 1.0
                    v[w] = float(tf[w]) * idf
            dn = norm(v)
            dot = sum(q_vec[w] * v.get(w, 0.0) for w in q_vec.keys())
            scores.append((i, float(dot / (qn * dn))))

        scores.sort(key=lambda t: t[1], reverse=True)
        matches = [{"sid": STATE.spans[i].sid, "score": sc} for i, sc in scores[: max(1, int(top_k))] if sc > 0]
        return {"matches": matches}

    # ----------------------------
    # Planning (must be audited)
    # ----------------------------

    @mcp.tool()
    def set_microplan(
        steps: List[Dict[str, Any]],
        verifier_model: str = "gpt-4o-mini",
        default_target: float = 0.95,
        units: str = "bits",
    ) -> Dict[str, Any]:
        """Audit and store a microplan. Required before any execution tools."""
        STATE.require_loaded()
        STATE.require_no_pending_checkin()
        _require_cited_steps(steps)
        try:
            report = _audit_steps_or_raise(
                steps=steps,
                verifier_model=verifier_model,
                default_target=default_target,
                pool_json_path=pool_json_path,
                units=units,
            )
        except AuditTraceBudgetError as exc:
            return {
                "ok": False,
                "plan_version": STATE.plan_version,
                "audit": exc.report,
                "error": str(exc),
            }
        STATE.set_plan(steps)
        return {"ok": True, "plan_version": STATE.plan_version, "audit": report}

    @mcp.tool()
    def get_microplan() -> Dict[str, Any]:
        """Return current microplan (if any)."""
        STATE.require_loaded()
        return {"plan_version": STATE.plan_version, "microplan": STATE.microplan}

    @mcp.tool()
    def clear_microplan() -> Dict[str, Any]:
        """Manually clear the plan (normally cleared automatically after execution)."""
        STATE.require_loaded()
        STATE.clear_plan()
        return {"ok": True}

    # ----------------------------
    # Post-action check-in (required before the next microplan)
    # ----------------------------

    def _get_attempt(attempt_id: int) -> Dict[str, Any]:
        if attempt_id < 0 or attempt_id >= len(STATE.attempts):
            raise IndexError(f"attempt_id {attempt_id} out of range (0..{len(STATE.attempts)-1})")
        a = STATE.attempts[attempt_id]
        if not isinstance(a, dict):
            raise RuntimeError(f"Corrupt attempt record at {attempt_id}")
        return a

    @mcp.tool()
    def get_pending_checkin() -> Dict[str, Any]:
        """Return the most recent action that still requires a check-in (if any)."""
        STATE.require_loaded()
        if STATE.pending_checkin is None:
            return {"pending": False}
        attempt_id = int(STATE.pending_checkin.get("attempt_id"))
        return {
            "pending": True,
            "attempt_id": attempt_id,
            "attempt": _get_attempt(attempt_id),
        }

    @mcp.tool()
    def list_attempts(limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """List the execution attempt ledger (actions + check-ins)."""
        STATE.require_loaded()
        sl = STATE.attempts[offset : offset + limit]
        return {
            "total": len(STATE.attempts),
            "pending_checkin": STATE.pending_checkin,
            "attempts": sl,
        }

    @mcp.tool()
    def checkin_last_action(
        status: str,
        claims: List[Dict[str, Any]],
        summary: str = "",
        verifier_model: str = "gpt-4o-mini",
        default_target: float = 0.95,
        units: str = "bits",
    ) -> Dict[str, Any]:
        """Resolve the post-action check-in gate by citing the last tool output.

        Why this exists:
        - The server requires evidence -> microplan -> action.
        - But models can still *drift* by not reading/validating the tool output.
        - This tool forces an explicit, evidence-backed check-in before allowing
          the next microplan.

        Inputs:
        - status: "succeeded" | "failed" | "inconclusive"
        - claims: a list of *present-tense* claims about what happened, with cites.
          At least one claim must cite the last action's output span.
          If a planned-step span is available, at least one claim must cite it too.
        """
        STATE.require_loaded()

        if STATE.pending_checkin is None:
            raise RuntimeError("No pending check-in. Run an execution tool first.")

        status_norm = (status or "").strip().lower()
        if status_norm not in {"succeeded", "failed", "inconclusive"}:
            raise ValueError("status must be one of: succeeded, failed, inconclusive")

        attempt_id = int(STATE.pending_checkin.get("attempt_id"))
        attempt = _get_attempt(attempt_id)
        last_span = attempt.get("span_id")
        plan_span = attempt.get("plan_step_span_id")

        if not claims:
            raise ValueError("claims must be non-empty")

        # Enforce that the model actually looked at the tool output.
        if last_span:
            cited_last = False
            for st in claims:
                cites = st.get("cites") or []
                if isinstance(cites, list) and last_span in [str(c) for c in cites]:
                    cited_last = True
                    break
            if not cited_last:
                raise RuntimeError(
                    f"At least one check-in claim must cite the last tool output span [{last_span}]."
                )

        # If we recorded the planned microplan step as a span, require it to be cited
        # too, so the check-in explicitly connects plan -> action.
        if plan_span:
            cited_plan = False
            for st in claims:
                cites = st.get("cites") or []
                if isinstance(cites, list) and plan_span in [str(c) for c in cites]:
                    cited_plan = True
                    break
            if not cited_plan:
                raise RuntimeError(
                    f"At least one check-in claim must cite the planned microplan step span [{plan_span}]."
                )

        attempt_payload = {
            "tool": attempt.get("tool"),
            "plan_version": attempt.get("plan_version"),
            "plan_step_idx": attempt.get("plan_step_idx"),
            "step": attempt.get("step"),
            "span_id": attempt.get("span_id"),
            "plan_step_span_id": plan_span,
            "note": attempt.get("note"),
            "extra": attempt.get("extra"),
        }

        try:
            report = _audit_steps_or_raise(
                steps=claims,
                verifier_model=verifier_model,
                default_target=float(default_target),
                pool_json_path=pool_json_path,
                units=units,
            )
        except AuditTraceBudgetError as exc:
            return {
                "ok": False,
                "attempt_id": attempt_id,
                "status": status_norm,
                "summary": (summary or "").strip(),
                "attempt": attempt_payload,
                "claims": claims,
                "audit": exc.report,
                "error": str(exc),
            }

        payload = {
            "attempt_id": attempt_id,
            "status": status_norm,
            "summary": (summary or "").strip(),
            "attempt": attempt_payload,
            "claims": claims,
            "audit": report,
        }
        sid = STATE.add_span(
            kind="checkin",
            source="checkin_last_action",
            text=json.dumps(payload, indent=2, sort_keys=True),
            extra={"attempt_id": attempt_id, "status": status_norm, "last_span": last_span},
        )

        # Attach check-in to the attempt record and clear the gate.
        attempt["checkin"] = {
            "ts": _now(),
            "status": status_norm,
            "summary": (summary or "").strip(),
            "checkin_span_id": sid,
            "audit_summary": report.get("summary"),
        }
        STATE.pending_checkin = None
        STATE.save()

        return {"ok": True, "attempt_id": attempt_id, "checkin_span_id": sid, "audit": report}

    # ----------------------------
    # Evidence generation (gated by plan_step_idx)
    # ----------------------------

    def _note_step(idx: int) -> str:
        st = STATE.require_plan_step(idx)
        return f"plan_step_idx={idx} claim={st.get('claim','')[:120]}"

    @mcp.tool()
    def run_cmd(
        cmd: List[str],
        plan_step_idx: int,
        cwd: Optional[str] = None,
        timeout_s: float = 60.0,
        allow_unsafe: bool = False,
    ) -> Dict[str, Any]:
        """Run a local command, capture stdout/stderr, store as a span, invalidate plan."""
        STATE.require_loaded()
        st = STATE.require_plan_step(int(plan_step_idx))
        if not cmd:
            raise ValueError("cmd must be non-empty")
        _require_allowed_cmd(cmd[0], allow_unsafe=allow_unsafe)

        run_dir, _ = STATE.require_loaded()
        proc = subprocess.run(
            cmd,
            cwd=(Path(cwd).expanduser().resolve() if cwd else run_dir),
            capture_output=True,
            text=True,
            timeout=float(timeout_s),
        )
        out = f"$ {' '.join(cmd)}\n\n[exit={proc.returncode}]\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
        sid = STATE.add_span(kind="cmd", source="run_cmd", text=out, extra={"cmd": cmd, "cwd": str(cwd or run_dir)})
        attempt_id = STATE.invalidate_plan_after_execution(
            tool="run_cmd",
            note=_note_step(int(plan_step_idx)),
            plan_step_idx=int(plan_step_idx),
            step_snapshot=st,
            span_id=sid,
            extra={"returncode": int(proc.returncode)},
        )
        return {"sid": sid, "returncode": proc.returncode, "attempt_id": attempt_id}

    @mcp.tool()
    def rg_search(
        pattern: str,
        path: str,
        plan_step_idx: int,
        timeout_s: float = 30.0,
    ) -> Dict[str, Any]:
        """Ripgrep search with output captured as a span."""
        return run_cmd(cmd=["rg", "-n", pattern, path], plan_step_idx=plan_step_idx, timeout_s=timeout_s)

    @mcp.tool()
    def lean_build(
        project_dir: str,
        plan_step_idx: int,
        timeout_s: float = 600.0,
    ) -> Dict[str, Any]:
        """Run `lake build` in a Lean project and store output as span."""
        return run_cmd(cmd=["lake", "build"], plan_step_idx=plan_step_idx, cwd=project_dir, timeout_s=timeout_s)

    @mcp.tool()
    def lean_query(
        project_dir: str,
        imports: List[str],
        commands: List[str],
        plan_step_idx: int,
        timeout_s: float = 120.0,
    ) -> Dict[str, Any]:
        """Run a temporary Lean file with given imports and commands via `lake env lean`.

        This is useful for `#check`, `#find`, `#print`, or small proof-state experiments.
        """
        STATE.require_loaded()
        _ = STATE.require_plan_step(int(plan_step_idx))
        run_dir, _ = STATE.require_loaded()

        tmp_dir = run_dir / "tmp_lean"
        _ensure_dir(tmp_dir)
        fname = tmp_dir / f"query_{int(time.time()*1000)}.lean"
        body = "\n".join([f"import {im}" for im in (imports or [])])
        body += "\n\n" + "\n".join(commands or []) + "\n"
        _write_text(fname, body)

        sid_res = run_cmd(
            cmd=["lake", "env", "lean", str(fname)],
            plan_step_idx=plan_step_idx,
            cwd=project_dir,
            timeout_s=timeout_s,
        )
        # run_cmd already invalidated the plan.
        return {"sid": sid_res["sid"], "file": str(fname)}

    @mcp.tool()
    def python_run(
        code: str,
        plan_step_idx: int,
        args: Optional[List[str]] = None,
        timeout_s: float = 120.0,
    ) -> Dict[str, Any]:
        """Execute Python code in a subprocess, store code+output as a span, invalidate plan."""
        STATE.require_loaded()
        _ = STATE.require_plan_step(int(plan_step_idx))
        scripts = STATE.scripts_dir()
        fname = scripts / f"script_{int(time.time()*1000)}.py"
        _write_text(fname, code)
        cmd = [sys.executable, str(fname)] + list(args or [])
        res = run_cmd(cmd=cmd, plan_step_idx=plan_step_idx, timeout_s=timeout_s)
        return {"sid": res["sid"], "script": str(fname), "returncode": res["returncode"]}

    @mcp.tool()
    def cpp_run(
        code: str,
        plan_step_idx: int,
        args: Optional[List[str]] = None,
        cxx: str = "g++",
        cxxflags: Optional[List[str]] = None,
        timeout_s: float = 120.0,
    ) -> Dict[str, Any]:
        """Compile *and* run a small C++ program, storing code + build + run logs as one span.

        This is intentionally a single tool call so the plan is invalidated *once* (after the
        combined build+run attempt).
        """
        STATE.require_loaded()
        st = STATE.require_plan_step(int(plan_step_idx))

        run_dir, _ = STATE.require_loaded()
        scripts = STATE.scripts_dir()
        stamp = int(time.time() * 1000)
        src = scripts / f"prog_{stamp}.cpp"
        exe = scripts / f"prog_{stamp}.out"
        _write_text(src, code)
        flags = cxxflags or ["-O2", "-std=c++20"]

        # Build
        build_cmd = [cxx, str(src), "-o", str(exe)] + list(flags)
        _require_allowed_cmd(build_cmd[0], allow_unsafe=False)
        build = subprocess.run(
            build_cmd,
            cwd=run_dir,
            capture_output=True,
            text=True,
            timeout=float(timeout_s),
        )

        run_rc: Optional[int] = None
        run_stdout = ""
        run_stderr = ""
        run_cmdline = [str(exe)] + list(args or [])
        if build.returncode == 0:
            runp = subprocess.run(
                run_cmdline,
                cwd=run_dir,
                capture_output=True,
                text=True,
                timeout=float(timeout_s),
            )
            run_rc = runp.returncode
            run_stdout = runp.stdout
            run_stderr = runp.stderr

        log = textwrap.dedent(
            f"""
            === C++ SOURCE ({src.name}) ===
            {code}

            === BUILD ===
            $ {' '.join(build_cmd)}
            [exit={build.returncode}]
            STDOUT:\n{build.stdout}
            STDERR:\n{build.stderr}

            === RUN ===
            $ {' '.join(run_cmdline)}
            [exit={run_rc if run_rc is not None else 'SKIPPED'}]
            STDOUT:\n{run_stdout}
            STDERR:\n{run_stderr}
            """
        ).strip()

        sid = STATE.add_span(
            kind="cpp",
            source="cpp_run",
            text=log,
            extra={"source": str(src), "exe": str(exe), "build_cmd": build_cmd, "run_cmd": run_cmdline},
        )
        attempt_id = STATE.invalidate_plan_after_execution(
            tool="cpp_run",
            note=_note_step(int(plan_step_idx)),
            plan_step_idx=int(plan_step_idx),
            step_snapshot=st,
            span_id=sid,
            extra={
                "build_returncode": int(build.returncode),
                "run_returncode": (int(run_rc) if run_rc is not None else None),
            },
        )
        return {
            "sid": sid,
            "source": str(src),
            "exe": str(exe),
            "build_returncode": int(build.returncode),
            "run_returncode": (int(run_rc) if run_rc is not None else None),
            "attempt_id": attempt_id,
        }

    # ----------------------------
    # Web + papers
    # ----------------------------

    @mcp.tool()
    def arxiv_search(query: str, plan_step_idx: int, max_results: int = 10) -> Dict[str, Any]:
        """Search arXiv via its Atom API (returns metadata, does not auto-ingest)."""
        STATE.require_loaded()
        st = STATE.require_plan_step(int(plan_step_idx))
        results = _arxiv_api_search(query, max_results=int(max_results))
        # Store results as span for provenance.
        sid = STATE.add_span(kind="arxiv_search", source="arxiv", text=json.dumps(results, indent=2))
        attempt_id = STATE.invalidate_plan_after_execution(
            tool="arxiv_search",
            note=_note_step(int(plan_step_idx)),
            plan_step_idx=int(plan_step_idx),
            step_snapshot=st,
            span_id=sid,
            extra={"num_results": len(results)},
        )
        return {"sid": sid, "results": results, "attempt_id": attempt_id}

    @mcp.tool()
    def fetch_url(
        url: str,
        plan_step_idx: int,
        timeout_s: float = 30.0,
        max_bytes: int = 5_000_000,
        strip_html: bool = True,
    ) -> Dict[str, Any]:
        """Fetch a URL and store the text content as a span."""
        STATE.require_loaded()
        st = STATE.require_plan_step(int(plan_step_idx))
        ct, txt = _http_get(url, timeout_s=float(timeout_s), max_bytes=int(max_bytes))
        if strip_html and ("html" in ct):
            txt = _strip_html(txt)
        sid = STATE.add_span(kind="web", source=url, text=txt, extra={"content_type": ct, "url": url})
        attempt_id = STATE.invalidate_plan_after_execution(
            tool="fetch_url",
            note=_note_step(int(plan_step_idx)),
            plan_step_idx=int(plan_step_idx),
            step_snapshot=st,
            span_id=sid,
            extra={"content_type": ct, "url": url},
        )
        return {"sid": sid, "content_type": ct, "attempt_id": attempt_id}

    @mcp.tool()
    def download_url(
        url: str,
        plan_step_idx: int,
        filename: Optional[str] = None,
        timeout_s: float = 60.0,
        max_bytes: int = 50_000_000,
    ) -> Dict[str, Any]:
        """Download a URL into the run's files/ directory and store a metadata span."""
        STATE.require_loaded()
        st = STATE.require_plan_step(int(plan_step_idx))
        files = STATE.files_dir()
        name = filename or os.path.basename(urllib.parse.urlparse(url).path) or f"file_{int(time.time()*1000)}"
        out = files / name
        r = requests.get(url, timeout=float(timeout_s), stream=True, headers={"User-Agent": "nala-mcp/0.1"})
        r.raise_for_status()
        n = 0
        with out.open("wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                n += len(chunk)
                if n > int(max_bytes):
                    break
                f.write(chunk)
        meta = {"url": url, "saved_to": str(out)}
        sid = STATE.add_span(kind="download", source=url, text=json.dumps(meta, indent=2))
        attempt_id = STATE.invalidate_plan_after_execution(
            tool="download_url",
            note=_note_step(int(plan_step_idx)),
            plan_step_idx=int(plan_step_idx),
            step_snapshot=st,
            span_id=sid,
            extra={"url": url, "path": str(out), "bytes": int(n)},
        )
        return {"sid": sid, "path": str(out), "attempt_id": attempt_id}

    @mcp.tool()
    def pdf_extract(
        path: str,
        plan_step_idx: int,
        page_start: int = 1,
        page_end: int = 1,
        max_chars: int = 200_000,
    ) -> Dict[str, Any]:
        """Extract text from a PDF (1-indexed pages) and store as a span."""
        STATE.require_loaded()
        st = STATE.require_plan_step(int(plan_step_idx))
        from pypdf import PdfReader

        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(str(p))
        reader = PdfReader(str(p))
        ps = max(1, int(page_start))
        pe = max(ps, int(page_end))
        pe = min(pe, len(reader.pages))
        texts = []
        for i in range(ps - 1, pe):
            try:
                texts.append(reader.pages[i].extract_text() or "")
            except Exception as e:
                texts.append(f"[EXTRACT_ERROR page {i+1}: {e}]")
        txt = "\n\n".join(texts)
        if len(txt) > int(max_chars):
            txt = txt[: int(max_chars)] + "\n\n[TRUNCATED]"
        sid = STATE.add_span(
            kind="pdf",
            source=str(p),
            text=txt,
            extra={"path": str(p), "page_start": ps, "page_end": pe},
        )
        attempt_id = STATE.invalidate_plan_after_execution(
            tool="pdf_extract",
            note=_note_step(int(plan_step_idx)),
            plan_step_idx=int(plan_step_idx),
            step_snapshot=st,
            span_id=sid,
            extra={"path": str(p), "page_start": ps, "page_end": pe},
        )
        return {"sid": sid, "pages": [ps, pe], "attempt_id": attempt_id}

    @mcp.tool()
    def arxiv_download_source(
        arxiv_id: str,
        plan_step_idx: int,
    ) -> Dict[str, Any]:
        """Download arXiv source tarball (e-print) and extract into files/; store as span."""
        STATE.require_loaded()
        st = STATE.require_plan_step(int(plan_step_idx))
        aid = arxiv_id.strip()
        url = f"https://arxiv.org/e-print/{aid}"
        ct, txt = _http_get(url, timeout_s=60.0, max_bytes=100_000_000)
        # The e-print endpoint returns tar/tex; treat as bytes in txt (utf-8 replace) isn't safe.
        # Re-fetch as bytes via requests.
        r = requests.get(url, timeout=60.0, stream=True, headers={"User-Agent": "nala-mcp/0.1"})
        r.raise_for_status()
        data = r.content
        files = STATE.files_dir()
        tar_path = files / f"{aid.replace('/', '_')}.tar"
        tar_path.write_bytes(data)
        extract_dir = files / f"{aid.replace('/', '_')}_src"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        _ensure_dir(extract_dir)
        try:
            with tarfile.open(tar_path, "r:*") as tf:
                tf.extractall(extract_dir)
        except Exception as e:
            # Some sources aren't tar; keep raw.
            logger.warning(f"Failed to extract arXiv source tarball: {e}")

        meta = {"arxiv_id": aid, "url": url, "tar_path": str(tar_path), "extract_dir": str(extract_dir)}
        sid = STATE.add_span(kind="arxiv_src", source=url, text=json.dumps(meta, indent=2))
        attempt_id = STATE.invalidate_plan_after_execution(
            tool="arxiv_download_source",
            note=_note_step(int(plan_step_idx)),
            plan_step_idx=int(plan_step_idx),
            step_snapshot=st,
            span_id=sid,
            extra=dict(meta),
        )
        return {"sid": sid, "attempt_id": attempt_id, **meta}

    return mcp


def main() -> None:
    pool_json_path = os.environ.get("AOAI_POOL_JSON")
    if not pool_json_path and len(sys.argv) > 1:
        pool_json_path = sys.argv[1]

    # Prefer OpenAI API when available, even if an AOAI pool is configured.
    force_aoai = os.environ.get("STRAWBERRY_FORCE_AOAI_POOL", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    have_openai = bool(os.environ.get("OPENAI_API_KEY"))
    use_aoai = bool(pool_json_path) and (force_aoai or not have_openai)

    if use_aoai:
        if not os.path.exists(pool_json_path or ""):
            logger.error(f"Pool config file not found: {pool_json_path}")
            sys.exit(1)
        logger.info(f"Starting nala-mcp with Azure OpenAI pool: {pool_json_path}")
        selected_pool_json_path: Optional[str] = pool_json_path
    else:
        if not have_openai:
            logger.error(
                "No backend configured. Either:\n"
                "  1) Set OPENAI_API_KEY for OpenAI API, or\n"
                "  2) Provide AOAI_POOL_JSON or pass pool path as argv[1]."
            )
            sys.exit(1)
        if pool_json_path and not force_aoai:
            logger.info(
                "Starting nala-mcp with OpenAI API "
                f"(ignoring AOAI pool config; set STRAWBERRY_FORCE_AOAI_POOL=1 to force Azure pool): {pool_json_path}"
            )
        else:
            logger.info("Starting nala-mcp with OpenAI API")
        selected_pool_json_path = None

    mcp = create_mcp_server(selected_pool_json_path)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
