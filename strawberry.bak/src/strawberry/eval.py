
"""
End-to-end evaluation helpers:
- generate dataset
- query a model
- compute confusion matrix MI, error rates, pseudo-prior mass, and B3

Backends
--------
This module supports:
- OpenAI Responses API (Structured Outputs) via strawberry.openai_backend
- Local vLLM inference via strawberry.vllm_backend

Parallelism
-----------
- Across items: we batch/parallelize choice calls (safe because items are independent).
- Null evaluation is also batched.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import numpy as np
import pandas as pd
from tqdm import tqdm

from .tasks import BindingItem, make_null_item
from .metrics import (
    mutual_information_from_confusion,
    bootstrap_confusion_mi,
    bits_to_trust,
    fano_required_mi,
    invert_fano_symmetric,
)
from .backend import BackendConfig, make_backend


@dataclass
class EvalResult:
    df: pd.DataFrame
    summary: Dict[str, float]


def _index_map(choices: Sequence[str]) -> Dict[str, int]:
    return {c: i for i, c in enumerate(choices)}


def _chunked(xs: List[BindingItem], n: int) -> List[List[BindingItem]]:
    return [xs[i : i + n] for i in range(0, len(xs), n)]


def run_eval(
    *,
    items: List[BindingItem],
    model: str,
    temperature: float = 0.0,
    null_mode: Optional[str] = None,
    show_progress: bool = True,
    # NEW (backward compatible): backend selection + batching
    backend_kind: str = "openai",
    backend_cfg: Optional[BackendConfig] = None,
    batch_size: int = 256,
) -> EvalResult:
    """
    Evaluate a model on items.

    If null_mode is provided, also evaluate a null (pseudo-prior) variant of each item and report:
      - p_tilde: fraction of null trials where model answers correctly
      - B3(p_tilde, eps) at the dataset level

    Notes
    -----
    - This function is *output-only*; it does not require activation access.
    - Calls are parallelized/batched across items (safe).
    """
    if not items:
        raise ValueError("items is empty")
    choices = items[0].choices
    if any(it.choices != choices for it in items):
        raise ValueError("all items must share the same choices list for confusion matrix evaluation")

    cfg = backend_cfg or BackendConfig(kind=str(backend_kind))
    backend = make_backend(cfg, model_hint=model if cfg.kind.lower() == "vllm" else None)

    rows: List[Dict[str, object]] = []
    chunks = _chunked(items, max(1, int(batch_size)))

    it = tqdm(chunks, disable=not show_progress, desc="Querying model (batched)")
    for chunk in it:
        prompts = [c.prompt for c in chunk]
        # Use batch if available; otherwise fall back to per-item calls.
        if hasattr(backend, "call_choice_batch"):
            res = backend.call_choice_batch(prompts=prompts, choices=choices, model=model, temperature=temperature)
        else:
            res = [backend.call_choice(prompt=p, choices=choices, model=model, temperature=temperature) for p in prompts]

        for item, out in zip(chunk, res):
            pred = out.answer
            rows.append(
                {
                    "prompt": item.prompt,
                    "correct": item.correct,
                    "pred": pred,
                    "is_correct": float(pred == item.correct),
                    "is_other": float(pred == "OTHER"),
                    **{f"meta_{k}": v for k, v in item.meta.items()},
                }
            )

    df = pd.DataFrame(rows)

    # Confusion matrix on candidate-only subset (exclude OTHER)
    cand_choices = [c for c in choices if c != "OTHER"]
    M = len(cand_choices)
    idx = _index_map(cand_choices)

    df_cand = df[df["pred"] != "OTHER"].copy()
    y_true = [idx[v] for v in df_cand["correct"].tolist()]
    y_pred = [idx[v] for v in df_cand["pred"].tolist()]
    conf = np.zeros((M, M), dtype=float)
    for t, p in zip(y_true, y_pred):
        conf[t, p] += 1

    mi_res = mutual_information_from_confusion(conf)
    boot = bootstrap_confusion_mi(y_true, y_pred, M=M, n_boot=500, seed=0) if len(y_true) >= 20 else {}

    summary: Dict[str, float] = {
        "n": float(len(df)),
        "acc_total": float(df["is_correct"].mean()),
        "gate_rate": float(1.0 - df["is_other"].mean()),  # proxy for Stage 2A success
        "acc_given_not_other": float(df_cand["is_correct"].mean()) if len(df_cand) else float("nan"),
        "M_candidates": float(M),
        "I_used_nats": float(mi_res.mi_nats),
        "eps_B": float(mi_res.error_rate),
        "Fano_required_I_nats": float(fano_required_mi(M, mi_res.error_rate)) if len(df_cand) else float("nan"),
        "eps_from_I_sym": float(invert_fano_symmetric(M, mi_res.mi_nats)) if len(df_cand) else float("nan"),
    }
    summary.update({f"boot_{k}": float(v) for k, v in boot.items()})

    # Null evaluation (pseudo-prior)
    if null_mode is not None:
        null_rows: List[Dict[str, float]] = []
        null_items = [make_null_item(it, null_mode=null_mode) for it in items]
        null_chunks = _chunked(null_items, max(1, int(batch_size)))
        it0 = tqdm(null_chunks, disable=not show_progress, desc=f"Null ({null_mode}) (batched)")
        for chunk in it0:
            prompts0 = [c.prompt for c in chunk]
            if hasattr(backend, "call_choice_batch"):
                res0 = backend.call_choice_batch(prompts=prompts0, choices=choices, model=model, temperature=temperature)
            else:
                res0 = [backend.call_choice(prompt=p, choices=choices, model=model, temperature=temperature) for p in prompts0]

            for item0, out0 in zip(chunk, res0):
                pred0 = out0.answer
                null_rows.append(
                    {
                        "is_correct0": float(pred0 == item0.correct),
                        "is_other0": float(pred0 == "OTHER"),
                    }
                )
        df0 = pd.DataFrame(null_rows)
        p_tilde = float(df0["is_correct0"].mean())
        eps = float(mi_res.error_rate)
        summary.update(
            {
                "null_mode": 1.0,  # marker
                "p_tilde_dataset": p_tilde,
                "B3_dataset_nats": float(bits_to_trust(p_tilde, eps)),
                "acc0_total": float(df0["is_correct0"].mean()),
                "gate0_rate": float(1.0 - df0["is_other0"].mean()),
            }
        )

    return EvalResult(df=df, summary=summary)
