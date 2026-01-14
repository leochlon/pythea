"""Hallucination detection utilities for pythea."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .evidence_guard import (
    MixtureInputs,
    MixtureConfig,
    LogprobBernoulliProbe,
    ClaimSupportProbe,
    evaluate_evidence_mixture_logprob,
    bernoulli_kl,
)


# ---------------------------------------------------------------------------
# Azure OpenAI Pool Backend
# ---------------------------------------------------------------------------

@dataclass
class GenOutput:
    """Output from a generation call."""
    content: str
    top_logprobs: Optional[List[List[Any]]] = None
    raw: Any = None


class AoaiPoolBackend:
    """Backend using AoaiPool for Azure OpenAI."""

    def __init__(self, pool_or_path: Any):
        """
        Args:
            pool_or_path: Either an AoaiPool instance or a path to a pool JSON config.
        """
        if isinstance(pool_or_path, str):
            from aoai_pool_python import AoaiPool
            self.pool = AoaiPool.from_file(pool_or_path)
        else:
            self.pool = pool_or_path

    def generate_one_token(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_logprobs: int = 20,
        logit_bias: Optional[Dict[int, float]] = None,
    ) -> GenOutput:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        extra: Dict[str, Any] = {
            "logprobs": True,
            "top_logprobs": top_logprobs,
        }
        if logit_bias:
            extra["logit_bias"] = logit_bias

        response = self.pool.chat(
            messages,
            max_tokens=1,
            temperature=temperature,
            top_p=top_p,
            extra=extra,
        )

        # Extract content
        content = ""
        choices = response.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")

        # Extract logprobs
        top_lps = None
        if choices:
            logprobs_data = choices[0].get("logprobs")
            if logprobs_data and "content" in logprobs_data:
                content_lps = logprobs_data["content"]
                if content_lps:
                    first_pos = content_lps[0]
                    top_lps = [first_pos.get("top_logprobs", [])]

        return GenOutput(content=content, top_logprobs=top_lps, raw=response)


def evaluate_evidence_mixture_logprob_with_backend(
    inputs: MixtureInputs,
    cfg: MixtureConfig,
    backend: Any,
    *,
    workers: int = 1,
):
    """Same as evaluate_evidence_mixture_logprob but with explicit backend parameter."""
    return evaluate_evidence_mixture_logprob(inputs, cfg, workers=workers, backend=backend)


__all__ = [
    "MixtureInputs",
    "MixtureConfig",
    "LogprobBernoulliProbe",
    "ClaimSupportProbe",
    "evaluate_evidence_mixture_logprob",
    "evaluate_evidence_mixture_logprob_with_backend",
    "bernoulli_kl",
    "AoaiPoolBackend",
    "GenOutput",
]
