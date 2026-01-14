"""strawberry.aoai_pool_backend

Azure OpenAI pool backend that uses the AoaiPool for load-balanced inference.

This backend adapts the AoaiPool (Chat Completions API) to work with the
strawberry toolkit's interface (which was designed for OpenAI's Responses API).

Key differences handled:
- Azure uses Chat Completions API, not Responses API
- Logprobs format differs slightly but is compatible after extraction
"""
from __future__ import annotations

import json
import os
import sys
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .openai_backend import ModelChoice, StructuredResult, TextResult


@dataclass
class AoaiPoolConfig:
    """Configuration for the AOAI pool backend."""
    pool_json_path: str
    max_concurrency: int = 16
    timeout_s: Optional[float] = 60.0
    max_attempts: Optional[int] = None  # defaults to number of backends


# Thread-local pool instances
_thread_local = threading.local()
_pool_lock = threading.Lock()
_pool_cache: Dict[str, Any] = {}


def _get_pool(pool_json_path: str):
    """Get or create a shared AoaiPool instance."""
    pool_path = os.path.abspath(pool_json_path)

    with _pool_lock:
        if pool_path in _pool_cache:
            return _pool_cache[pool_path]

    # Import AoaiPool - first try the bundled module, then external locations
    AoaiPool = None

    try:
        from .aoai_pool_python import AoaiPool
    except ImportError:
        pass

    if AoaiPool is None:
        try:
            from aoai_pool_python import AoaiPool
        except ImportError:
            pass

    if AoaiPool is None:
        # Try loading from the directory containing the pool JSON file
        pool_dir = os.path.dirname(pool_path)
        aoai_pool_py = os.path.join(pool_dir, "aoai_pool_python.py")
        if os.path.exists(aoai_pool_py):
            import importlib.util
            spec = importlib.util.spec_from_file_location("aoai_pool_python", aoai_pool_py)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                AoaiPool = module.AoaiPool

    if AoaiPool is None:
        raise ImportError(
            "Could not import AoaiPool. Ensure aoai_pool_python.py is bundled "
            "with the toolkit or in your PYTHONPATH."
        )

    pool = AoaiPool.from_file(pool_path)

    with _pool_lock:
        _pool_cache[pool_path] = pool

    return pool


def _extract_logprobs_from_chat_response(response: Dict[str, Any]) -> Optional[List[Any]]:
    """Extract logprobs from Azure Chat Completions response.

    Azure format:
    {
        "choices": [{
            "logprobs": {
                "content": [
                    {"token": "YES", "logprob": -0.5, "top_logprobs": [...]},
                    ...
                ]
            }
        }]
    }

    Returns the content list, which is compatible with extract_answer_topk().
    """
    try:
        choices = response.get("choices", [])
        if not choices:
            return None

        logprobs = choices[0].get("logprobs")
        if logprobs is None:
            return None

        content = logprobs.get("content")
        if content is None:
            return None

        return list(content)
    except Exception:
        return None


def _extract_text_from_chat_response(response: Dict[str, Any]) -> str:
    """Extract the assistant's message text from a chat completion response."""
    try:
        choices = response.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        return str(message.get("content", "") or "")
    except Exception:
        return ""


class AoaiPoolBackend:
    """Backend that uses AoaiPool for load-balanced Azure OpenAI inference."""

    def __init__(self, cfg: AoaiPoolConfig):
        self.cfg = cfg
        self._pool = _get_pool(cfg.pool_json_path)

    def call_text(
        self,
        *,
        prompt: str,
        model: str,  # Ignored - pool uses configured deployments
        instructions: str = "You are a helpful assistant.",
        temperature: float = 0.0,
        max_output_tokens: int = 64,
        include_logprobs: bool = False,
        top_logprobs: int = 0,
        reasoning: Optional[Dict[str, Any]] = None,  # Not used for Azure
        retries: int = 3,
        retry_backoff_s: float = 1.5,
        timeout_s: Optional[float] = None,
        base_url: Optional[str] = None,  # Ignored - pool manages endpoints
        api_key: Optional[str] = None,  # Ignored - pool manages keys
        **kwargs,
    ) -> TextResult:
        """Call the Azure OpenAI pool for plain text output with optional logprobs."""

        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": prompt},
        ]

        options: Dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }

        if include_logprobs:
            options["extra"] = {
                "logprobs": True,
                "top_logprobs": min(20, max(0, int(top_logprobs))),
            }

        if self.cfg.max_attempts:
            options["max_attempts"] = self.cfg.max_attempts

        timeout = timeout_s if timeout_s is not None else self.cfg.timeout_s
        if timeout is not None:
            options["timeout_ms"] = int(timeout * 1000)

        response = self._pool.chat(messages, **options)

        text = _extract_text_from_chat_response(response)
        logprobs = _extract_logprobs_from_chat_response(response) if include_logprobs else None

        return TextResult(
            text=text,
            response_id=response.get("id"),
            logprobs=logprobs,
        )

    def call_json_schema(
        self,
        *,
        prompt: str,
        schema: Dict[str, Any],
        model: str,
        name: str = "structured",
        instructions: str = "Return only a JSON object that matches the schema.",
        temperature: float = 0.0,
        max_output_tokens: int = 800,
        include_logprobs: bool = False,
        reasoning: Optional[Dict[str, Any]] = None,
        retries: int = 3,
        retry_backoff_s: float = 1.5,
        timeout_s: Optional[float] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> StructuredResult:
        """Call with JSON schema structured output."""

        # Build a prompt that asks for JSON
        full_instructions = f"{instructions}\n\nYou must respond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"

        messages = [
            {"role": "system", "content": full_instructions},
            {"role": "user", "content": prompt},
        ]

        options: Dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_output_tokens,
            "response_format": {"type": "json_object"},
        }

        if include_logprobs:
            options["extra"] = {
                "logprobs": True,
                "top_logprobs": 10,
            }

        if self.cfg.max_attempts:
            options["max_attempts"] = self.cfg.max_attempts

        timeout = timeout_s if timeout_s is not None else self.cfg.timeout_s
        if timeout is not None:
            options["timeout_ms"] = int(timeout * 1000)

        response = self._pool.chat(messages, **options)

        text = _extract_text_from_chat_response(response)
        logprobs = _extract_logprobs_from_chat_response(response) if include_logprobs else None

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse JSON response: {e}\nResponse: {text[:500]}")

        return StructuredResult(
            data=data,
            response_id=response.get("id"),
            logprobs=logprobs,
        )

    def call_choice(
        self,
        *,
        prompt: str,
        choices: Sequence[str],
        model: str,
        system: str = "Return only a JSON object that matches the schema.",
        temperature: float = 0.0,
        max_output_tokens: int = 200,
        retries: int = 3,
        retry_backoff_s: float = 1.5,
        timeout_s: Optional[float] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> ModelChoice:
        """Ask the model to pick a choice from an enum."""

        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "enum": list(choices)},
                "rationale": {"type": "string"},
            },
            "required": ["answer"],
            "additionalProperties": False,
        }

        r = self.call_json_schema(
            prompt=prompt,
            schema=schema,
            model=model,
            name="binding_choice",
            instructions=system,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            include_logprobs=False,
            retries=retries,
            retry_backoff_s=retry_backoff_s,
            timeout_s=timeout_s,
        )

        return ModelChoice(
            answer=r.data.get("answer", ""),
            raw_json=r.data,
            response_id=r.response_id,
        )

    def call_text_batch(
        self,
        *,
        prompts: Sequence[str],
        **kwargs,
    ) -> List[TextResult]:
        """Batch call_text with thread-pool parallelism."""
        import concurrent.futures as cf

        max_workers = max(1, int(self.cfg.max_concurrency))
        with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(self.call_text, prompt=p, **kwargs) for p in prompts]
            return [f.result() for f in futs]

    def call_choice_batch(
        self,
        *,
        prompts: Sequence[str],
        choices: Sequence[str],
        **kwargs,
    ) -> List[ModelChoice]:
        """Batch call_choice with thread-pool parallelism."""
        import concurrent.futures as cf

        max_workers = max(1, int(self.cfg.max_concurrency))
        with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(self.call_choice, prompt=p, choices=choices, **kwargs) for p in prompts]
            return [f.result() for f in futs]

    def call_json_schema_batch(
        self,
        *,
        prompts: Sequence[str],
        schema: Dict[str, Any],
        **kwargs,
    ) -> List[StructuredResult]:
        """Batch call_json_schema with thread-pool parallelism."""
        import concurrent.futures as cf

        max_workers = max(1, int(self.cfg.max_concurrency))
        with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(self.call_json_schema, prompt=p, schema=schema, **kwargs) for p in prompts]
            return [f.result() for f in futs]
