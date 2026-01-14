
"""
CLI for the binding-routing toolkit.

Examples
--------
OpenAI backend:
  export OPENAI_API_KEY=...
  binding-routing run --backend openai --model gpt-4o-2024-08-06 --n 200 --M 10 --distance 512 --query FIRST --null SCRUB_FIRST

Local vLLM backend (GPU):
  binding-routing run --backend vllm --model meta-llama/Meta-Llama-3.1-8B-Instruct --vllm_tensor_parallel 2 --n 400 --distance 1024
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .backend import BackendConfig
from .tasks import generate_dataset, generate_proximity_item
from .eval import run_eval
from .cot_detector import detect_cot_hallucinations


def _add_backend_args(p: argparse.ArgumentParser):
    p.add_argument("--backend", type=str, default="openai", choices=["openai", "vllm"], help="LLM execution backend")
    p.add_argument("--max_concurrency", type=int, default=16, help="OpenAI: threadpool concurrency for batched calls")
    p.add_argument("--timeout_s", type=float, default=None, help="OpenAI: request timeout (seconds)")
    p.add_argument("--base_url", type=str, default=None, help="OpenAI-compatible server base URL (optional)")
    p.add_argument("--api_key", type=str, default=None, help="API key override (optional)")

    # vLLM knobs
    p.add_argument("--vllm_tensor_parallel", type=int, default=1, help="vLLM: tensor parallel size")
    p.add_argument("--vllm_max_model_len", type=int, default=None, help="vLLM: max model length (tokens)")
    p.add_argument("--vllm_dtype", type=str, default="bfloat16", help="vLLM: dtype (bfloat16/float16/float32)")
    p.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.90, help="vLLM: GPU memory util fraction")
    p.add_argument("--vllm_trust_remote_code", action="store_true", help="vLLM: trust_remote_code for HF models")


def main():
    parser = argparse.ArgumentParser(prog="binding-routing")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run a synthetic proximity binding eval")
    run.add_argument("--model", type=str, required=True)
    run.add_argument("--n", type=int, default=200)
    run.add_argument("--M", type=int, default=10)
    run.add_argument("--distance", type=int, default=256, help="approx filler token count")
    run.add_argument("--query", type=str, default="FIRST", choices=["FIRST", "LAST"])
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--null", type=str, default=None, help="null mode: SCRUB_FIRST, SCRUB_BOTH, REMOVE_FILLER")
    run.add_argument("--batch_size", type=int, default=256, help="Batch size for backend calls")
    run.add_argument("--out", type=str, default="strawberry_results.json")
    _add_backend_args(run)

    cot = sub.add_parser("cot", help="Run a reasoning-trace (CoT) hallucination detector")
    cot.add_argument("--generator_model", type=str, required=True)
    cot.add_argument("--verifier_model", type=str, required=True)
    cot.add_argument("--executor_model", type=str, default=None)
    cot.add_argument("--max_steps", type=int, default=8)
    cot.add_argument("--temperature", type=float, default=0.0)
    cot.add_argument("--prompt", type=str, default=None, help="Prompt string. If omitted, use --prompt_file or synthetic.")
    cot.add_argument("--prompt_file", type=str, default=None, help="Path to a text file containing the prompt")
    # Synthetic convenience
    cot.add_argument("--synthetic", action="store_true", help="Generate a single synthetic binding prompt")
    cot.add_argument("--M", type=int, default=10)
    cot.add_argument("--distance", type=int, default=256)
    cot.add_argument("--query", type=str, default="FIRST", choices=["FIRST", "LAST"])
    cot.add_argument("--seed", type=int, default=0)
    # Optional Stage-2A/2B logprob analysis
    cot.add_argument("--choices_file", type=str, default=None, help="Path to JSON array of choices (must include OTHER)")
    cot.add_argument("--choices", type=str, default=None, choices=["auto_trace", "auto_context"],
                     help="Auto-generate candidate choices: auto_trace (LLM checkpoint) or auto_context (regex from spans)")
    cot.add_argument("--choices_key", type=str, default=None,
                     help="For auto choices: target key/variable name (if omitted, inferred when possible)")
    cot.add_argument("--choices_are_codes", action="store_true",
                     help="Assert that choices are single-token codes (recommended for OpenAI Stage-2A/2B)")
    cot.add_argument("--auto_choice_mode", type=str, default="codes", choices=["codes", "sequence"],
                     help="Auto-choice scoring: codes (recode candidates to single-token IDs) or sequence (vLLM only)")
    cot.add_argument("--vllm_multitoken_fallback", type=str, default=None, choices=["sequence"],
                     help="vLLM only: if explicit choices are multi-token, use sequence scoring instead of raising")
    # Optional: pre-answer deliberation sweep (answer drift vs reasoning length)
    cot.add_argument("--deliberation_sweep", type=int, nargs="+", default=None,
                     help="Optional sweep of pre-answer deliberation step counts K (e.g., 0 2 4 8). "
                          "For each K>0, the model generates K deliberation steps, those steps are appended "
                          "to the prompt, then Stage-2A/2B are recomputed to measure drift.")
    cot.add_argument("--deliberation_kind", type=str, default="trace", choices=["trace", "checkpoint"],
                     help="Deliberation content type: trace (claims+cites) or checkpoint (key=value+cites).")
    cot.add_argument("--deliberation_model", type=str, default=None,
                     help="Model used to generate deliberation (default: generator_model).")
    cot.add_argument("--deliberation_temperature", type=float, default=0.0,
                     help="Temperature for deliberation generation (default 0.0).")
    cot.add_argument("--correct", type=str, default=None, help="Correct choice (enables Stage-2B metrics)")
    cot.add_argument("--score_model", type=str, default=None, help="Model to use for answer-only logprob scoring")
    cot.add_argument("--top_logprobs", type=int, default=20, help="Top-K alternatives to request per token (0-20)")
    # Optional: trace-level information-budget scoring
    cot.add_argument("--no_trace_budget", action="store_true", help="Disable trace-level budget scoring")
    cot.add_argument("--budget_model", type=str, default=None, help="Model to use for budget YES/NO logprobs (default: verifier_model)")
    cot.add_argument("--budget_target", type=float, default=0.95, help="Default target reliability for each claim (used if trace step has no confidence)")
    cot.add_argument("--budget_top_logprobs", type=int, default=10, help="Top-K for YES/NO/UNSURE queries (0-20)")
    cot.add_argument("--budget_placeholder", type=str, default="[REDACTED]", help="Placeholder text used to scrub cited spans")
    cot.add_argument("--out", type=str, default="cot_report.json")
    _add_backend_args(cot)

    args = parser.parse_args()

    cfg = BackendConfig(
        kind=str(args.backend),
        max_concurrency=int(args.max_concurrency),
        timeout_s=args.timeout_s,
        base_url=args.base_url,
        api_key=args.api_key,
        vllm_tensor_parallel=int(args.vllm_tensor_parallel),
        vllm_max_model_len=args.vllm_max_model_len,
        vllm_dtype=str(args.vllm_dtype),
        vllm_gpu_memory_utilization=float(args.vllm_gpu_memory_utilization),
        vllm_trust_remote_code=bool(args.vllm_trust_remote_code),
    )

    if args.cmd == "run":
        items = generate_dataset(
            n=args.n,
            distance_tokens=args.distance,
            M=args.M,
            query_rule=args.query,
            seed=0,
        )
        res = run_eval(
            items=items,
            model=args.model,
            temperature=args.temperature,
            null_mode=args.null,
            backend_kind=cfg.kind,
            backend_cfg=cfg,
            batch_size=args.batch_size,
        )

        out_path = Path(args.out)
        payload = {
            "summary": res.summary,
            "rows_head": res.df.head(10).to_dict(orient="records"),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote: {out_path}")
        print(json.dumps(res.summary, indent=2))

    elif args.cmd == "cot":
        choices = None
        correct = args.correct
        if args.prompt_file:
            prompt = Path(args.prompt_file).read_text(encoding="utf-8")
        elif args.prompt is not None and not args.synthetic:
            prompt = args.prompt
        else:
            item = generate_proximity_item(M=args.M, distance_tokens=args.distance, seed=args.seed, query_rule=args.query)
            prompt = item.prompt
            choices = item.choices
            correct = item.correct

        if args.choices_file:
            choices = json.loads(Path(args.choices_file).read_text(encoding="utf-8"))

        if args.choices is not None:
            # Force auto-choice mode to take precedence over any explicit choices.
            choices = None

        trace, report = detect_cot_hallucinations(
            prompt=prompt,
            generator_model=args.generator_model,
            verifier_model=args.verifier_model,
            executor_model=args.executor_model,
            choices=choices,
            choices_mode=args.choices,
            choices_key=args.choices_key,
            choices_are_codes=bool(args.choices_are_codes),
            auto_choice_mode=str(args.auto_choice_mode),
            vllm_multitoken_fallback=args.vllm_multitoken_fallback,
            correct=correct,
            score_model=args.score_model,
            top_logprobs=args.top_logprobs,
            trace_budget=(not args.no_trace_budget),
            budget_model=args.budget_model,
            deliberation_sweep=args.deliberation_sweep,
            deliberation_kind=str(args.deliberation_kind),
            deliberation_model=args.deliberation_model,
            deliberation_temperature=float(args.deliberation_temperature),
            budget_target=args.budget_target,
            budget_top_logprobs=args.budget_top_logprobs,
            budget_placeholder=args.budget_placeholder,
            max_steps=args.max_steps,
            temperature=args.temperature,
            backend_kind=cfg.kind,
            backend_cfg=cfg,
        )

        out_path = Path(args.out)
        payload = {
            "trace": {
                "answer": trace.answer,
                "steps": [
                    {"idx": s.idx, "claim": s.claim, "kind": s.kind, "cites": s.cites, "confidence": s.confidence}
                    for s in trace.steps
                ],
                "spans": [{"sid": sp.sid, "text": sp.text} for sp in trace.spans],
            },
            "report": {
                "answer": report.answer,
                "scored_answer": report.scored_answer,
                "answer_from_trace": report.answer_from_trace,
                "answer_match": report.answer_match,
                "grounded_fraction": report.grounded_fraction,
                "hallucinated_steps": report.hallucinated_steps,
                "not_in_context_steps": report.not_in_context_steps,
                "contradicted_steps": report.contradicted_steps,
                "unverifiable_steps": report.unverifiable_steps,
                "checks": [
                    {"idx": c.idx, "verdict": c.verdict, "confidence": c.confidence, "notes": c.notes}
                    for c in report.checks
                ],
                "stage2a": report.stage2a,
                "stage2b": report.stage2b,
                "stage_meta": report.stage_meta,
                "stage_ab": report.stage_ab,
                "trace_budget": report.trace_budget,
                "budget_flagged_steps": report.budget_flagged_steps,
                "max_budget_gap_min": report.max_budget_gap_min,
                "max_budget_gap_max": report.max_budget_gap_max,
                "deliberation_sweep": report.deliberation_sweep,
            },
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote: {out_path}")
        print(json.dumps(payload["report"], indent=2))


if __name__ == "__main__":
    main()
