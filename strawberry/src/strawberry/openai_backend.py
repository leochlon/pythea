"""
OpenAI backend utilities.

Uses the Responses API with Structured Outputs (json_schema) so that:
  - the model must return a single enum value from the candidate set (plus optional OTHER)
  - parsing is reliable without prompt hacks

Docs:
- Structured outputs guide (json_schema) and SDK examples.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import json
import os
import time

try:
    from openai import OpenAI
except Exception as _e:
    OpenAI = None  # type: ignore

import threading

_thread_local = threading.local()

def _get_client(*, timeout_s: Optional[float] = None, base_url: Optional[str] = None, api_key: Optional[str] = None):
    """Thread-local cached OpenAI client for higher throughput under parallel calls."""
    if OpenAI is None:
        raise ImportError("openai package not installed. Run: pip install openai")
    # Allow env override for base_url (useful for OpenAI-compatible servers).
    if base_url is None:
        base_url = os.environ.get("OPENAI_BASE_URL")
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    key = (timeout_s, base_url, api_key)
    cache = getattr(_thread_local, "clients", None)
    if cache is None:
        cache = {}
        _thread_local.clients = cache
    if key in cache:
        return cache[key]
    kwargs: Dict[str, Any] = {}
    if timeout_s is not None:
        kwargs["timeout"] = timeout_s
    if base_url is not None:
        kwargs["base_url"] = base_url
    if api_key is not None:
        kwargs["api_key"] = api_key
    client = OpenAI(**kwargs) if kwargs else OpenAI()
    cache[key] = client
    return client




@dataclass
class ModelChoice:
    answer: str
    raw_json: Dict[str, Any]
    response_id: Optional[str] = None


@dataclass
class StructuredResult:
    """Generic structured output result."""

    data: Dict[str, Any]
    response_id: Optional[str] = None
    # Optional output_text logprobs if requested via Responses `include`.
    logprobs: Optional[Any] = None


@dataclass
class TextResult:
    """Plain-text output, optionally with per-token logprobs."""

    text: str
    response_id: Optional[str] = None
    logprobs: Optional[Any] = None


def _schema_for_choices(choices: Sequence[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "enum": list(choices)},
            "rationale": {"type": "string"},
        },
        "required": ["answer"],
        "additionalProperties": False,
    }


def call_json_schema(
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
) -> StructuredResult:
    """Call an OpenAI model with Structured Outputs (JSON Schema) and parse the JSON."""

    if OpenAI is None:
        raise ImportError("openai package not installed. Run: pip install openai")
    client = _get_client(timeout_s=timeout_s, base_url=base_url, api_key=api_key)

    include = ["message.output_text.logprobs"] if include_logprobs else None

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            kwargs: Dict[str, Any] = {}
            if include is not None:
                kwargs["include"] = include
            if reasoning is not None:
                kwargs["reasoning"] = reasoning

            resp = client.responses.create(
                model=model,
                instructions=instructions,
                input=[
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": name,
                        "schema": schema,
                        "strict": True,
                    }
                },
                **kwargs,
            )

            # Extract output_text and optional logprobs
            out_text = getattr(resp, "output_text", None)
            out_logprobs = None
            if not out_text:
                # Find first output_text content piece
                for item in resp.output:
                    for c in getattr(item, "content", []):
                        if getattr(c, "type", None) == "output_text":
                            out_text = getattr(c, "text", None)
                            out_logprobs = getattr(c, "logprobs", None)
                            break
                    if out_text:
                        break
            else:
                # Best-effort: SDKs differ in how they attach logprobs
                for item in getattr(resp, "output", []):
                    for c in getattr(item, "content", []):
                        if getattr(c, "type", None) == "output_text":
                            out_logprobs = getattr(c, "logprobs", None)
                            break
                    if out_logprobs is not None:
                        break

            if not out_text:
                raise RuntimeError("No output_text found in response")
            data = json.loads(out_text)
            return StructuredResult(data=data, response_id=getattr(resp, "id", None), logprobs=out_logprobs)

        except Exception as e:
            last_err = e
            if attempt >= retries:
                break
            time.sleep(retry_backoff_s * (attempt + 1))

    raise RuntimeError(f"OpenAI call failed after retries: {last_err}")


def call_choice(
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
) -> ModelChoice:
    """
    Ask an OpenAI model to pick a choice from an enum using Structured Outputs.

    Returns:
        ModelChoice(answer=..., raw_json=..., response_id=...)
    """
    schema = _schema_for_choices(choices)
    r = call_json_schema(
        prompt=prompt,
        schema=schema,
        model=model,
        name="binding_choice",
        instructions=system,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        include_logprobs=False,
        reasoning=None,
        retries=retries,
        retry_backoff_s=retry_backoff_s,
        timeout_s=timeout_s,
        base_url=base_url,
        api_key=api_key,
    )
    return ModelChoice(answer=r.data["answer"], raw_json=r.data, response_id=r.response_id)


def call_text(
    *,
    prompt: str,
    model: str,
    instructions: str = "You are a helpful assistant.",
    temperature: float = 0.0,
    max_output_tokens: int = 64,
    include_logprobs: bool = False,
    top_logprobs: int = 0,
    reasoning: Optional[Dict[str, Any]] = None,
    retries: int = 3,
    retry_backoff_s: float = 1.5,
    timeout_s: Optional[float] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> TextResult:
    """Call an OpenAI model for *plain text* output.

    If include_logprobs is True, the response includes per-token logprobs for the generated
    output tokens, plus up to `top_logprobs` alternative tokens at each position.

    Notes
    -----
    - This uses the Responses API. Logprobs must be enabled via the `include` parameter:
      `message.output_text.logprobs`.
    - `top_logprobs` must be between 0 and 20 (per the API).
    """

    if OpenAI is None:
        raise ImportError("openai package not installed. Run: pip install openai")
    client = _get_client(timeout_s=timeout_s, base_url=base_url, api_key=api_key)

    if top_logprobs < 0 or top_logprobs > 20:
        raise ValueError("top_logprobs must be between 0 and 20")

    include = ["message.output_text.logprobs"] if include_logprobs else None

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            kwargs: Dict[str, Any] = {}
            if include is not None:
                kwargs["include"] = include
                kwargs["top_logprobs"] = int(top_logprobs)
            if reasoning is not None:
                kwargs["reasoning"] = reasoning

            resp = client.responses.create(
                model=model,
                instructions=instructions,
                input=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                text={"format": {"type": "text"}},
                **kwargs,
            )

            out_text = getattr(resp, "output_text", None)
            out_logprobs = None
            if not out_text:
                for item in getattr(resp, "output", []):
                    for c in getattr(item, "content", []):
                        if getattr(c, "type", None) == "output_text":
                            out_text = getattr(c, "text", None)
                            out_logprobs = getattr(c, "logprobs", None)
                            break
                    if out_text:
                        break
            else:
                for item in getattr(resp, "output", []):
                    for c in getattr(item, "content", []):
                        if getattr(c, "type", None) == "output_text":
                            out_logprobs = getattr(c, "logprobs", None)
                            break
                    if out_logprobs is not None:
                        break

            if not out_text:
                raise RuntimeError("No output_text found in response")

            return TextResult(text=str(out_text), response_id=getattr(resp, "id", None), logprobs=out_logprobs)

        except Exception as e:
            last_err = e
            if attempt >= retries:
                break
            time.sleep(retry_backoff_s * (attempt + 1))

    raise RuntimeError(f"OpenAI call failed after retries: {last_err}")


def call_text_chat(
    *,
    prompt: str,
    model: str = "gpt-4o-mini",
    instructions: str = "You are a helpful assistant.",
    temperature: float = 0.0,
    max_output_tokens: int = 64,
    include_logprobs: bool = False,
    top_logprobs: int = 0,
    retries: int = 3,
    retry_backoff_s: float = 1.5,
    timeout_s: Optional[float] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,  # ignore extra args like 'reasoning'
) -> TextResult:
    """Call an OpenAI model using Chat Completions API (supports logprobs).

    This is an alternative to call_text() that uses the Chat Completions API
    instead of the Responses API, providing reliable logprobs support.
    """
    if OpenAI is None:
        raise ImportError("openai package not installed. Run: pip install openai")
    client = _get_client(timeout_s=timeout_s, base_url=base_url, api_key=api_key)

    if top_logprobs < 0 or top_logprobs > 20:
        raise ValueError("top_logprobs must be between 0 and 20")

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            api_kwargs: Dict[str, Any] = {}
            if include_logprobs:
                api_kwargs["logprobs"] = True
                api_kwargs["top_logprobs"] = int(top_logprobs)

            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_output_tokens,
                **api_kwargs,
            )

            choice = resp.choices[0]
            out_text = choice.message.content or ""

            # Convert logprobs to the format expected by the toolkit
            out_logprobs = None
            if include_logprobs and choice.logprobs and choice.logprobs.content:
                out_logprobs = []
                for token_info in choice.logprobs.content:
                    token_data = {
                        "token": token_info.token,
                        "logprob": token_info.logprob,
                    }
                    if token_info.top_logprobs:
                        token_data["top_logprobs"] = [
                            {"token": t.token, "logprob": t.logprob}
                            for t in token_info.top_logprobs
                        ]
                    out_logprobs.append(token_data)

            return TextResult(text=str(out_text), response_id=resp.id, logprobs=out_logprobs)

        except Exception as e:
            last_err = e
            if attempt >= retries:
                break
            time.sleep(retry_backoff_s * (attempt + 1))

    raise RuntimeError(f"OpenAI chat call failed after retries: {last_err}")
