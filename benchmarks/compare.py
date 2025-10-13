#!/usr/bin/env python3
"""
Benchmark: Original EDFL vs Glass
===================================

Compares ensemble sampling (30-42 API calls) with grammatical symmetry (1 call).

Metrics:
  - Execution time
  - Number of API calls
  - Cost estimate ($)
  - Decision quality (agreement rate)
"""

import time
import sys
from typing import List, Tuple
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hallbayes.hallucination_toolkit import (
    OpenAIBackend,
    OpenAIItem,
    OpenAIPlanner,
)
from glass import GlassPlanner, GlassItem


# Test prompts covering different types
TEST_PROMPTS = [
    # Factual questions with specific entities/dates
    "Who won the 2019 Nobel Prize in Physics?",
    "What is the capital of France?",
    "When did World War II end?",
    "Who wrote Romeo and Juliet?",

    # Questions requiring reasoning
    "Explain the concept of quantum entanglement.",
    "What are the main differences between DNA and RNA?",

    # Potentially ambiguous
    "What is the fastest animal?",
    "How many planets are in the solar system?",
]


def estimate_cost(n_calls: int, model: str = "gpt-4o-mini") -> float:
    """
    Estimate API cost in USD.

    Rates (as of 2024):
      gpt-4o-mini: ~$0.0001 per call (approximate)
      gpt-4o: ~$0.003 per call
    """
    rate_per_call = {
        "gpt-4o-mini": 0.0001,
        "gpt-4o": 0.003,
        "claude-3-5-sonnet-latest": 0.003,
    }
    return n_calls * rate_per_call.get(model, 0.0001)


def run_original(backend: OpenAIBackend, prompts: List[str]) -> Tuple[List, float, int]:
    """
    Run original EDFL planner (ensemble sampling).

    Returns:
        (metrics, elapsed_time, api_calls)
    """
    print("\n🔬 Running ORIGINAL (Ensemble Sampling)...")

    items = [
        OpenAIItem(
            prompt=p,
            n_samples=3,  # Reduced for speed (normally 5-7)
            m=4,          # Reduced for speed (normally 6)
            skeleton_policy="closed_book"
        )
        for p in prompts
    ]

    planner = OpenAIPlanner(backend, temperature=0.3)

    start = time.time()
    metrics = planner.run(items, h_star=0.05)
    elapsed = time.time() - start

    # Calculate API calls: (n_samples + m*n_samples) per item
    api_calls_per_item = items[0].n_samples * (1 + items[0].m)
    total_calls = api_calls_per_item * len(items)

    print(f"✓ Completed in {elapsed:.2f}s")
    print(f"✓ API calls: {total_calls}")
    print(f"✓ Cost estimate: ${estimate_cost(total_calls):.4f}")

    return metrics, elapsed, total_calls


def run_glass(backend: OpenAIBackend, prompts: List[str]) -> Tuple[List, float, int]:
    """
    Run Glass planner (grammatical symmetry).

    Returns:
        (metrics, elapsed_time, api_calls)
    """
    print("\n✨ Running GLASS (Grammatical Symmetry)...")

    items = [GlassItem(prompt=p) for p in prompts]

    planner = GlassPlanner(backend, temperature=0.3, verbose=False)

    start = time.time()
    metrics = planner.run(items, h_star=0.05)
    elapsed = time.time() - start

    # Glass: 1 call per item
    total_calls = len(items)

    print(f"✓ Completed in {elapsed:.2f}s")
    print(f"✓ API calls: {total_calls}")
    print(f"✓ Cost estimate: ${estimate_cost(total_calls):.4f}")

    return metrics, elapsed, total_calls


def compare_decisions(original_metrics, glass_metrics) -> float:
    """
    Compare decision agreement rate.

    Returns:
        Agreement rate [0, 1]
    """
    if len(original_metrics) != len(glass_metrics):
        return 0.0

    agreements = sum(
        1 for orig, glass in zip(original_metrics, glass_metrics)
        if orig.decision_answer == glass.decision_answer
    )

    return agreements / len(original_metrics)


def print_detailed_comparison(original_metrics, glass_metrics, prompts):
    """Print side-by-side comparison"""
    print("\n" + "="*80)
    print("DETAILED COMPARISON")
    print("="*80)

    for i, (orig, glass, prompt) in enumerate(zip(original_metrics, glass_metrics, prompts)):
        print(f"\n[{i+1}] {prompt}")
        print(f"    Original: {'✓ ANSWER' if orig.decision_answer else '✗ REFUSE'} "
              f"(ISR={orig.isr:.2f}, RoH={orig.roh_bound:.3f})")
        print(f"    Glass:    {'✓ ANSWER' if glass.decision_answer else '✗ REFUSE'} "
              f"(ISR={glass.isr:.2f}, RoH={glass.roh_bound:.3f}, Sym={glass.symmetry_score:.2f})")

        match = "✓ MATCH" if orig.decision_answer == glass.decision_answer else "⚠ DIFFER"
        print(f"    {match}")


def print_summary(orig_time, orig_calls, glass_time, glass_calls, agreement_rate):
    """Print performance summary"""
    speedup = orig_time / glass_time if glass_time > 0 else float('inf')
    call_reduction = orig_calls / glass_calls if glass_calls > 0 else float('inf')

    orig_cost = estimate_cost(orig_calls)
    glass_cost = estimate_cost(glass_calls)
    cost_reduction = orig_cost / glass_cost if glass_cost > 0 else float('inf')

    print("\n" + "="*80)
    print("📊 PERFORMANCE SUMMARY")
    print("="*80)
    print(f"\n⏱️  Time:")
    print(f"    Original: {orig_time:.2f}s")
    print(f"    Glass:    {glass_time:.2f}s")
    print(f"    Speedup:  {speedup:.1f}×")

    print(f"\n📞 API Calls:")
    print(f"    Original: {orig_calls}")
    print(f"    Glass:    {glass_calls}")
    print(f"    Reduction: {call_reduction:.1f}×")

    print(f"\n💰 Cost:")
    print(f"    Original: ${orig_cost:.4f}")
    print(f"    Glass:    ${glass_cost:.4f}")
    print(f"    Savings:  {cost_reduction:.1f}×")

    print(f"\n🎯 Decision Agreement: {agreement_rate*100:.1f}%")

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print(f"Glass is {speedup:.1f}× faster and {cost_reduction:.1f}× cheaper")
    print(f"while maintaining {agreement_rate*100:.1f}% decision agreement.")
    print("="*80 + "\n")


def main():
    """Run benchmark"""
    print("="*80)
    print("BENCHMARK: Original EDFL vs Glass")
    print("="*80)
    print(f"\nTesting {len(TEST_PROMPTS)} prompts...")

    # Initialize backend
    try:
        backend = OpenAIBackend(model="gpt-4o-mini")
        print(f"✓ Backend initialized: {backend.model}")
    except Exception as e:
        print(f"✗ Failed to initialize backend: {e}")
        print("\nMake sure OPENAI_API_KEY is set:")
        print("  export OPENAI_API_KEY=sk-...")
        return

    # Run benchmarks
    try:
        orig_metrics, orig_time, orig_calls = run_original(backend, TEST_PROMPTS)
        glass_metrics, glass_time, glass_calls = run_glass(backend, TEST_PROMPTS)
    except Exception as e:
        print(f"\n✗ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Compare results
    agreement_rate = compare_decisions(orig_metrics, glass_metrics)

    # Print results
    print_detailed_comparison(orig_metrics, glass_metrics, TEST_PROMPTS)
    print_summary(orig_time, orig_calls, glass_time, glass_calls, agreement_rate)


if __name__ == "__main__":
    main()
