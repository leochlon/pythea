
"""strawberry.vllm_backend

Local GPU backend built on **vLLM**.

This backend is intended for high-throughput sweeps on your own GPU cluster.
It provides the same surface as the OpenAI backend helpers used by this toolkit:

- call_text(..., include_logprobs=True, top_logprobs=K)
- call_json_schema(...): best-effort JSON-only output + jsonschema validation + retries
- call_choice(...): enum selection via a JSON schema wrapper

Important notes
---------------
1) vLLM does not natively enforce OpenAI Structured Outputs (json_schema) in the same way
   OpenAI hosted models do. We implement:
     - strict prompting ("ONLY JSON")
     - robust parsing (extract first JSON object)
     - jsonschema validation
     - retries with escalating instructions
   This is usually reliable for instruction-tuned models.

2) Logprobs format:
   We convert vLLM's per-token logprob objects into a lightweight dict structure compatible
   with strawberry.stage_ab.extract_answer_topk(), so Stage-2A/2B and trace-budget code
   can be backend-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json
import math
import re
import time

try:
    from vllm import LLM, SamplingParams
except Exception:
    LLM = None  # type: ignore
    SamplingParams = None  # type: ignore

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None  # type: ignore

try:
    import jsonschema
except Exception:
    jsonschema = None  # type: ignore

from .openai_backend import ModelChoice, StructuredResult, TextResult

_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}", re.MULTILINE)


def _extract_json_object(text: str) -> Dict[str, Any]:
    s = (text or "").strip()
    # Fast path
    try:
        return json.loads(s)
    except Exception:
        pass
    # Extract first {...} block
    m = _JSON_OBJ_RE.search(s)
    if not m:
        raise ValueError("No JSON object found in text")
    chunk = m.group(0)
    return json.loads(chunk)


def _validate_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> None:
    if jsonschema is None:
        # No validator installed; accept best-effort.
        return
    jsonschema.validate(instance=data, schema=schema)


def _vllm_logprobs_to_openaiish(tokenizer, token_ids: List[int], logprobs: List[Dict[int, Any]]) -> List[Dict[str, Any]]:
    """
    Convert vLLM output logprobs to a list of dicts:
      {"token": <str>, "logprob": <float>, "top_logprobs": [{"token":<str>,"logprob":<float>}, ...]}
    compatible with stage_ab.extract_answer_topk().
    """
    out: List[Dict[str, Any]] = []
    for pos, tid in enumerate(token_ids):
        tok = tokenizer.decode([tid])
        lp_dict = logprobs[pos] if pos < len(logprobs) else {}
        # lp_dict: token_id -> Logprob or float
        top_list: List[Dict[str, Any]] = []
        gen_lp = None
        for alt_id, obj in lp_dict.items():
            lp = float(obj.logprob) if hasattr(obj, "logprob") else float(obj)
            alt_tok = tokenizer.decode([int(alt_id)])
            top_list.append({"token": alt_tok, "logprob": lp})
            if int(alt_id) == int(tid):
                gen_lp = lp
        if gen_lp is None:
            # If vLLM didn't include generated token in the map, approximate as -inf (shouldn't happen)
            gen_lp = float("-inf")
        out.append({"token": tok, "logprob": float(gen_lp), "top_logprobs": top_list})
    return out


class VLLMBackend:
    """A single-model vLLM backend."""

    def __init__(
        self,
        *,
        model: str,
        tensor_parallel: int = 1,
        max_model_len: Optional[int] = None,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.90,
        trust_remote_code: bool = True,
    ):
        if LLM is None or SamplingParams is None or AutoTokenizer is None:
            raise ImportError("vllm/transformers not installed. Install extras: pip install vllm transformers torch jsonschema")
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code, use_fast=True)
        self.llm = LLM(
            model=model,
            tensor_parallel_size=int(tensor_parallel),
            max_model_len=max_model_len,
            dtype=dtype,
            gpu_memory_utilization=float(gpu_memory_utilization),
            trust_remote_code=trust_remote_code,
        )

    def call_text(
        self,
        *,
        prompt: str,
        model: str,
        instructions: str = "You are a helpful assistant.",
        temperature: float = 0.0,
        max_output_tokens: int = 64,
        include_logprobs: bool = False,
        top_logprobs: int = 0,
        reasoning: Optional[Dict[str, Any]] = None,
        retries: int = 1,
        retry_backoff_s: float = 0.2,
        timeout_s: Optional[float] = None,
        **_: Any,
    ) -> TextResult:
        # We ignore 'model' and use the instantiated model; keep signature compatible.
        # Combine instructions and prompt in a simple way.
        full_prompt = prompt if not instructions else (instructions.strip() + "\n\n" + prompt)

        if top_logprobs < 0 or top_logprobs > 20:
            raise ValueError("top_logprobs must be between 0 and 20")

        last_err: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                sp = SamplingParams(
                    temperature=float(temperature),
                    max_tokens=int(max_output_tokens),
                    logprobs=int(top_logprobs) if include_logprobs else None,
                )
                out = self.llm.generate([full_prompt], sp)[0].outputs[0]
                text = out.text
                lp = None
                if include_logprobs:
                    lp = _vllm_logprobs_to_openaiish(self.tokenizer, out.token_ids, out.logprobs)
                return TextResult(text=str(text), response_id=None, logprobs=lp)
            except Exception as e:
                last_err = e
                if attempt >= retries:
                    break
                time.sleep(float(retry_backoff_s) * (attempt + 1))
        raise RuntimeError(f"vLLM call_text failed: {last_err}")

    def call_text_batch(
        self,
        *,
        prompts: Sequence[str],
        model: str,
        instructions: str = "You are a helpful assistant.",
        temperature: float = 0.0,
        max_output_tokens: int = 64,
        include_logprobs: bool = False,
        top_logprobs: int = 0,
        **_: Any,
    ) -> List[TextResult]:
        full_prompts = [p if not instructions else (instructions.strip() + "\n\n" + p) for p in prompts]
        sp = SamplingParams(
            temperature=float(temperature),
            max_tokens=int(max_output_tokens),
            logprobs=int(top_logprobs) if include_logprobs else None,
        )
        outs = self.llm.generate(list(full_prompts), sp)
        results: List[TextResult] = []
        for o in outs:
            gen = o.outputs[0]
            lp = None
            if include_logprobs:
                lp = _vllm_logprobs_to_openaiish(self.tokenizer, gen.token_ids, gen.logprobs)
            results.append(TextResult(text=str(gen.text), response_id=None, logprobs=lp))
        return results

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
        retries: int = 2,
        retry_backoff_s: float = 0.4,
        timeout_s: Optional[float] = None,
        **_: Any,
    ) -> StructuredResult:
        # vLLM: best-effort JSON-only output + schema validation
        last_err: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                strict = (
                    "Return ONLY a JSON object, no markdown, no commentary.\n"
                    "The JSON MUST validate against the provided schema.\n"
                )
                schema_hint = json.dumps(schema)
                user = (
                    strict
                    + f"SCHEMA (JSON):\n{schema_hint}\n\n"
                    + prompt.strip()
                )
                tr = self.call_text(
                    prompt=user,
                    model=model,
                    instructions=instructions,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    include_logprobs=False,
                    top_logprobs=0,
                    retries=0,
                )
                data = _extract_json_object(tr.text)
                _validate_schema(data, schema)
                return StructuredResult(data=data, response_id=None, logprobs=None)
            except Exception as e:
                last_err = e
                if attempt >= retries:
                    break
                time.sleep(float(retry_backoff_s) * (attempt + 1))
        raise RuntimeError(f"vLLM call_json_schema failed: {last_err}")

    def call_choice(
        self,
        *,
        prompt: str,
        choices: Sequence[str],
        model: str,
        system: str = "Return only a JSON object that matches the schema.",
        temperature: float = 0.0,
        max_output_tokens: int = 200,
        retries: int = 2,
        retry_backoff_s: float = 0.4,
        timeout_s: Optional[float] = None,
        **_: Any,
    ) -> ModelChoice:
        # Use the same schema as OpenAI backend
        from .openai_backend import _schema_for_choices  # local import
        schema = _schema_for_choices(choices)
        r = self.call_json_schema(
            prompt=prompt,
            schema=schema,
            model=model,
            name="binding_choice",
            instructions=system,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            retries=retries,
            retry_backoff_s=retry_backoff_s,
        )
        ans = str(r.data.get("answer", "")).strip()
        return ModelChoice(answer=ans, raw_json=r.data, response_id=None)

    def call_choice_batch(
        self,
        *,
        prompts: Sequence[str],
        choices: Sequence[str],
        model: str,
        system: str = "Return only a JSON object that matches the schema.",
        temperature: float = 0.0,
        max_output_tokens: int = 200,
        **_: Any,
    ) -> List[ModelChoice]:
        from .openai_backend import _schema_for_choices
        schema = _schema_for_choices(choices)
        # Use call_json_schema in a loop; for speed, we do a single batched text generation then parse each.
        strict = (
            "Return ONLY a JSON object, no markdown, no commentary.\n"
            "The JSON MUST validate against the provided schema.\n"
        )
        schema_hint = json.dumps(schema)
        full_prompts = [strict + f"SCHEMA (JSON):\n{schema_hint}\n\n" + p.strip() for p in prompts]
        trs = self.call_text_batch(
            prompts=full_prompts,
            model=model,
            instructions=system,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            include_logprobs=False,
            top_logprobs=0,
        )
        out: List[ModelChoice] = []
        for tr in trs:
            try:
                data = _extract_json_object(tr.text)
                _validate_schema(data, schema)
                ans = str(data.get("answer","")).strip()
                out.append(ModelChoice(answer=ans, raw_json=data, response_id=None))
            except Exception:
                # fallback: mark as OTHER if parsing fails
                out.append(ModelChoice(answer="OTHER", raw_json={"answer": "OTHER"}, response_id=None))
        return out


    # -------------------------
    # Optional: sequence-level scoring for multi-token choices
    # -------------------------

    def score_choice_sequences(
        self,
        *,
        prefix: str,
        choices: Sequence[str],
        batch_size: int = 64,
        use_prompt_logprobs: bool = True,
    ) -> List[float]:
        """
        Compute log P(choice | prefix) for each choice as a *sequence* (multi-token aware).

        This is used as a rigorous fallback when Stage-2A/2B token-topK metrics are invalid
        because one or more choices are multi-token at the answer position.

        Implementation strategy:
        1) Prefer vLLM prompt_logprobs (uses the already-loaded model).
        2) If unavailable or unsupported, fall back to a transformers forward-pass scorer.

        Notes
        -----
        - prefix SHOULD end with whitespace to prevent BPE boundary merges between prefix and choice.
        - Returned scores are natural-log probabilities (sum of per-token logprobs).
        """
        prefix = str(prefix)
        if not prefix.endswith((" ", "\n", "\t")):
            prefix = prefix + " "

        # Fast path: vLLM prompt_logprobs
        if use_prompt_logprobs:
            try:
                return self._score_choice_sequences_vllm_prompt_logprobs(prefix=prefix, choices=choices, batch_size=batch_size)
            except Exception:
                # Fall through to transformers scorer
                pass
        return self._score_choice_sequences_transformers(prefix=prefix, choices=choices, batch_size=batch_size)

    def _score_choice_sequences_vllm_prompt_logprobs(
        self,
        *,
        prefix: str,
        choices: Sequence[str],
        batch_size: int,
    ) -> List[float]:
        """Best-effort sequence scoring using vLLM prompt_logprobs."""
        from vllm import SamplingParams  # type: ignore

        prefix_ids = self.tokenizer(prefix, add_special_tokens=False).input_ids
        prefix_len = len(prefix_ids)
        full_prompts = [prefix + str(c) for c in choices]

        # prompt_logprobs: request at least the token's own logprob
        sp = SamplingParams(temperature=0.0, max_tokens=0, prompt_logprobs=1)
        outs = self.llm.generate(full_prompts, sp)

        def _as_float_logprob(obj) -> float:
            if obj is None:
                raise KeyError("missing logprob object")
            if isinstance(obj, (float, int)):
                return float(obj)
            # vLLM sometimes returns Logprob objects with .logprob
            lp = getattr(obj, "logprob", None)
            if lp is not None:
                return float(lp)
            # or dicts
            if isinstance(obj, dict) and "logprob" in obj:
                return float(obj["logprob"])
            raise KeyError(f"unrecognized logprob object type: {type(obj)}")

        scores: List[float] = []
        for full, out in zip(full_prompts, outs):
            full_ids = self.tokenizer(full, add_special_tokens=False).input_ids
            # Locate prompt_logprobs array
            plp = getattr(out, "prompt_logprobs", None)
            if plp is None and getattr(out, "outputs", None):
                plp = getattr(out.outputs[0], "prompt_logprobs", None)
            if plp is None:
                raise RuntimeError("vLLM did not return prompt_logprobs (unsupported version/config).")
            if len(plp) != len(full_ids):
                # Some vLLM versions include special tokens; be conservative and bail to transformers.
                raise RuntimeError("prompt_logprobs length mismatch; falling back to transformers scorer.")
            if prefix_len >= len(full_ids):
                scores.append(0.0)
                continue
            total = 0.0
            for i in range(prefix_len, len(full_ids)):
                tid = int(full_ids[i])
                entry = plp[i]
                if entry is None:
                    # Some implementations omit the first token's logprob; treat as missing.
                    raise RuntimeError("prompt_logprobs missing at position; falling back to transformers.")
                if isinstance(entry, dict):
                    if tid not in entry:
                        raise RuntimeError("token id missing in prompt_logprobs entry; falling back to transformers.")
                    total += _as_float_logprob(entry[tid])
                else:
                    # Unknown structure
                    raise RuntimeError("unrecognized prompt_logprobs entry type; falling back to transformers.")
            scores.append(float(total))
        return scores

    def _score_choice_sequences_transformers(
        self,
        *,
        prefix: str,
        choices: Sequence[str],
        batch_size: int,
    ) -> List[float]:
        """Fallback sequence scoring using transformers forward pass (may load an extra model)."""
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM  # type: ignore
        except Exception as e:
            raise ImportError("Sequence scoring fallback requires torch+transformers.") from e

        # Lazy-load a scoring model (can be memory-heavy; prefer prompt_logprobs path above).
        mdl = getattr(self, "_sequence_scorer_model", None)
        dev = getattr(self, "_sequence_scorer_device", None)
        if mdl is None:
            # Choose device heuristically.
            device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                mdl = AutoModelForCausalLM.from_pretrained(self.model, torch_dtype=torch.bfloat16 if device=="cuda" else None)
                mdl.to(device)
            except Exception:
                # Last resort: CPU float32
                device = "cpu"
                mdl = AutoModelForCausalLM.from_pretrained(self.model)
                mdl.to(device)
            mdl.eval()
            setattr(self, "_sequence_scorer_model", mdl)
            setattr(self, "_sequence_scorer_device", device)
            dev = device

        prefix_ids = self.tokenizer(prefix, add_special_tokens=False).input_ids
        prefix_len = len(prefix_ids)

        scores: List[float] = []
        for i in range(0, len(choices), batch_size):
            batch = [prefix + str(c) for c in choices[i:i+batch_size]]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False)
            input_ids = enc["input_ids"].to(dev)
            attn = enc.get("attention_mask", None)
            if attn is not None:
                attn = attn.to(dev)

            with torch.no_grad():
                out = mdl(input_ids=input_ids, attention_mask=attn)
                logits = out.logits  # [B, T, V]
                logp = torch.log_softmax(logits, dim=-1)

            # Compute per-example logprob of tokens after prefix_len (excluding padding)
            for b in range(input_ids.shape[0]):
                ids = input_ids[b]
                # Determine actual length (exclude padding)
                if attn is not None:
                    L = int(attn[b].sum().item())
                else:
                    # No padding mask; assume full length
                    L = int((ids != self.tokenizer.pad_token_id).sum().item()) if self.tokenizer.pad_token_id is not None else ids.shape[0]
                # Candidate tokens start at prefix_len; score their logprobs
                if prefix_len >= L:
                    scores.append(0.0)
                    continue
                total = 0.0
                # logp at position t predicts token at t+1; standard shift
                for t in range(prefix_len, L):
                    if t == 0:
                        continue
                    tok_id = int(ids[t])
                    total += float(logp[b, t-1, tok_id].item())
                scores.append(float(total))
        return scores
