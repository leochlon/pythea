#!/usr/bin/env python3
"""
Glass + Ollama Quick Benchmark
===============================

Quick benchmark with 5 prompts (~3-4 minutes).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hallbayes.htk_backends import OllamaBackend
from glass import GlassPlanner, GlassItem


# Quick test set
TEST_PROMPTS = [
    "What is the capital of France?",
    "What is 2+2?",
    "Who wrote Romeo and Juliet?",
    "When did World War II end?",
    "What is the speed of light?",
]


def main():
    print("=" * 70)
    print("GLASS + OLLAMA QUICK BENCHMARK")
    print("=" * 70)
    print(f"\nModel: llama3.1:8b (local)")
    print(f"Test prompts: {len(TEST_PROMPTS)}")
    print(f"Expected time: ~3-4 minutes\n")

    # Initialize
    print("Initializing...")
    backend = OllamaBackend(
        model="llama3.1:8b",
        host="http://localhost:11434",
        request_timeout=180.0
    )
    planner = GlassPlanner(backend, temperature=0.3, verbose=False)
    print("✓ Ready\n")

    # Create items
    items = [GlassItem(prompt=p) for p in TEST_PROMPTS]

    # Run
    print("Running benchmark...")
    start_time = time.time()
    metrics = []

    for i, item in enumerate(items, 1):
        item_start = time.time()
        result = planner.evaluate_item(i-1, item, h_star=0.05)
        metrics.append(result)
        item_time = time.time() - item_start

        decision = "✓" if result.decision_answer else "✗"
        sym = result.symmetry_score if hasattr(result, 'symmetry_score') else 0
        print(f"  [{i}/{len(items)}] {decision} {item_time:.1f}s - Sym: {sym:.3f}")

    total_time = time.time() - start_time

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    answered = sum(1 for m in metrics if m.decision_answer)
    symmetry_scores = [m.symmetry_score for m in metrics if hasattr(m, 'symmetry_score')]
    avg_sym = sum(symmetry_scores) / len(symmetry_scores) if symmetry_scores else 0

    print(f"\n📊 Statistics:")
    print(f"   Answered:    {answered}/{len(metrics)} ({answered/len(metrics)*100:.0f}%)")
    print(f"   Avg Symmetry: {avg_sym:.3f}")

    print(f"\n⏱️  Performance:")
    print(f"   Total time:   {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"   Per item:     {total_time/len(metrics):.1f}s")
    print(f"   Throughput:   {len(metrics)/total_time:.2f} items/s")

    print(f"\n💰 Cost:")
    print(f"   API cost:     $0 (local)")
    print(f"   Privacy:      ✅ 100% local")

    # Detailed
    print(f"\n📋 Detailed Results:")
    for i, (prompt, m) in enumerate(zip(TEST_PROMPTS, metrics), 1):
        decision = "ANSWER" if m.decision_answer else "REFUSE"
        sym = m.symmetry_score if hasattr(m, 'symmetry_score') else 0
        print(f"   [{i}] {decision:6s} | Sym:{sym:.3f} | ISR:{m.isr:.1f} | {prompt[:40]}")

    print("\n" + "=" * 70)
    print("✅ Benchmark complete!")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
