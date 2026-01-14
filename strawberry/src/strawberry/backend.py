
"""strawberry.backend

Backend abstraction + parallel helpers.

This toolkit supports two execution backends:
- "openai": OpenAI Responses API (Structured Outputs + logprobs)
- "vllm": local GPU inference via vLLM

We keep the public surface compatible with the original toolkit functions, but expose
backend objects so evaluation loops can:
- batch/parallelize calls safely
- switch between backends without changing core logic
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import concurrent.futures as cf

from .openai_backend import ModelChoice, StructuredResult, TextResult, call_text_chat
from . import openai_backend as oai


@dataclass
class BackendConfig:
    kind: str = "openai"  # "openai", "vllm", or "aoai_pool"
    max_concurrency: int = 16
    timeout_s: Optional[float] = None
    # OpenAI-compatible server knobs (optional)
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    # vLLM knobs (only used when kind="vllm")
    vllm_tensor_parallel: int = 1
    vllm_max_model_len: Optional[int] = None
    vllm_dtype: str = "bfloat16"
    vllm_gpu_memory_utilization: float = 0.90
    vllm_trust_remote_code: bool = True
    # AOAI pool knobs (only used when kind="aoai_pool")
    aoai_pool_json_path: Optional[str] = None
    aoai_pool_max_attempts: Optional[int] = None


class OpenAIBackend:
    """Thin wrapper around openai_backend.* with batch helpers (thread-pool parallelism)."""

    def __init__(self, cfg: BackendConfig):
        self.cfg = cfg

    def call_text(self, **kwargs: Any) -> TextResult:
        # Use Chat Completions API for reliable logprobs support
        return call_text_chat(
            **kwargs,
            timeout_s=self.cfg.timeout_s,
            base_url=self.cfg.base_url,
            api_key=self.cfg.api_key,
        )

    def call_json_schema(self, **kwargs: Any) -> StructuredResult:
        return oai.call_json_schema(
            **kwargs,
            timeout_s=self.cfg.timeout_s,
            base_url=self.cfg.base_url,
            api_key=self.cfg.api_key,
        )

    def call_choice(self, **kwargs: Any) -> ModelChoice:
        return oai.call_choice(
            **kwargs,
            timeout_s=self.cfg.timeout_s,
            base_url=self.cfg.base_url,
            api_key=self.cfg.api_key,
        )

    def call_text_batch(self, *, prompts: Sequence[str], **kwargs: Any) -> List[TextResult]:
        max_workers = max(1, int(self.cfg.max_concurrency))
        with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(self.call_text, prompt=p, **kwargs) for p in prompts]
            return [f.result() for f in futs]

    def call_choice_batch(self, *, prompts: Sequence[str], choices: Sequence[str], **kwargs: Any) -> List[ModelChoice]:
        max_workers = max(1, int(self.cfg.max_concurrency))
        with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(self.call_choice, prompt=p, choices=choices, **kwargs) for p in prompts]
            return [f.result() for f in futs]

    def call_json_schema_batch(self, *, prompts: Sequence[str], schema: Dict[str, Any], **kwargs: Any) -> List[StructuredResult]:
        max_workers = max(1, int(self.cfg.max_concurrency))
        with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(self.call_json_schema, prompt=p, schema=schema, **kwargs) for p in prompts]
            return [f.result() for f in futs]


def make_backend(cfg: BackendConfig, *, model_hint: Optional[str] = None):
    """Create a backend object.

    model_hint is used only for vLLM (single-model backend).
    """
    kind = (cfg.kind or "openai").lower().strip()
    if kind == "openai":
        return OpenAIBackend(cfg)
    if kind == "vllm":
        if model_hint is None:
            raise ValueError("vLLM backend requires model_hint (the HF model id/path).")
        from .vllm_backend import VLLMBackend
        return VLLMBackend(
            model=model_hint,
            tensor_parallel=cfg.vllm_tensor_parallel,
            max_model_len=cfg.vllm_max_model_len,
            dtype=cfg.vllm_dtype,
            gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
            trust_remote_code=cfg.vllm_trust_remote_code,
        )
    if kind == "aoai_pool":
        if cfg.aoai_pool_json_path is None:
            raise ValueError("aoai_pool backend requires aoai_pool_json_path.")
        from .aoai_pool_backend import AoaiPoolBackend, AoaiPoolConfig
        pool_cfg = AoaiPoolConfig(
            pool_json_path=cfg.aoai_pool_json_path,
            max_concurrency=cfg.max_concurrency,
            timeout_s=cfg.timeout_s,
            max_attempts=cfg.aoai_pool_max_attempts,
        )
        return AoaiPoolBackend(pool_cfg)
    raise ValueError(f"Unknown backend kind: {cfg.kind!r}")
