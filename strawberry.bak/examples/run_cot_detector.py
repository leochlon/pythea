"""Example: reasoning-trace (CoT) hallucination detector on a synthetic binding prompt.

Usage:
  export OPENAI_API_KEY=...
  python examples/run_cot_detector.py

Notes:
- This audits the *visible* reasoning trace the model emits (short, citeable claims),
  not any hidden internal reasoning tokens.
"""

from strawberry.tasks import generate_proximity_item
from strawberry.cot_detector import detect_cot_hallucinations


if __name__ == "__main__":
    item = generate_proximity_item(M=10, distance_tokens=512, query_rule="FIRST", seed=0)

    trace, report = detect_cot_hallucinations(
        prompt=item.prompt,
        generator_model="gpt-4o-2024-08-06",
        verifier_model="gpt-4o-2024-08-06",
        executor_model="gpt-4o-2024-08-06",
        # Enable Stage-2A/2B analysis on the answer token via logprobs.
        choices=item.choices,
        correct=item.correct,
        score_model="gpt-4o-2024-08-06",
        top_logprobs=20,
        max_steps=8,
        temperature=0.0,
    )

    print("=== TRACE ANSWER ===")
    print(trace.answer)
    print("\n=== TRACE STEPS ===")
    for s in trace.steps:
        print(f"{s.idx}. [{s.kind}] ({','.join(s.cites) or '-'}) {s.claim}")

    print("\n=== REPORT ===")
    print(report)
    print("\n=== STAGE-2A / STAGE-2B (output-level) ===")
    print("Stage2A:", report.stage2a)
    print("Stage2B:", report.stage2b)
