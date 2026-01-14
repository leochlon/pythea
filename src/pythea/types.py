from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class BackendCredentials(TypedDict, total=False):
    openai_api_key: str
    azure_api_key: str
    azure_endpoint: str
    azure_api_version: str
    openrouter_api_key: str
    anthropic_api_key: str
    vertex_credentials_json: str
    vertex_project: str
    vertex_location: str


class UnifiedAnswerRequest(TypedDict, total=False):
    # Required
    question: str

    # Optional request fields supported by the server
    evidence: Optional[str]
    backend: str
    interpretability: bool
    prompt_rewrite: bool
    creds: BackendCredentials

    # knobs (all optional)
    m: int
    tau: float
    eta: float
    seed: int
    workers: int

    cbg_samples_per_subq: int
    cbg_subqs_mece: bool
    cbg_subq_boundary: bool
    cbg_tree_depth: int
    cbg_subq_oversample: int
    cbg_subq_regen_rounds: int
    cbg_tree_max_expansions: int
    cbg_tree_expand_per_node: int
    cbg_answer_max_chars: int
    cbg_sample_temp: float
    cbg_presence_penalty: float
    cbg_frequency_penalty: float
    cbg_max_evidence_blocks: int
    cbg_abstain_on_any_unknown: bool

    model: Optional[str]
    hstar: float
    prior_quantile: float
    top_logprobs: int
    temperature: float
    use_logit_bias: bool

    judge: Dict[str, Any]


class UnifiedAnswerResponse(TypedDict, total=False):
    mode: str
    backend: str
    passed: bool
    decision: str
    abstain_reason: Optional[str]
    candidates: Optional[List[str]]
    picked: Optional[str]
    metrics: Dict[str, Any]
    branches: Optional[List[Dict[str, Any]]]
    judge: Optional[Dict[str, Any]]
    rewritten_prompt: Optional[str]
    recommendations: Optional[List[str]]
    eg_metrics: Optional[Dict[str, Any]]


class MinimalUnifiedAnswerResponse(TypedDict, total=False):
    """Minimal public response returned by /api/unified-answer in many deployments.

    Only includes coarse fields; intentionally omits backend/metrics/trace.
    Compatibility fields (passed/mode/eg_metrics) may be present depending on server.
    """
    status: str  # "answered" | "abstained"
    answer: Optional[str]
    reason_code: Optional[str]
    request_id: Optional[str]
    rewritten_prompt: Optional[str]
    judge_explanation: Optional[str]
    # Optional compatibility fields
    passed: Optional[bool]
    mode: Optional[str]
    eg_metrics: Optional[Dict[str, Any]]
