#!/usr/bin/env python3
"""
Simple Glass + Ollama Test
===========================

Tests Glass with Ollama llama3.1:8b (no external dependencies)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    print("="*70)
    print("GLASS + OLLAMA TEST")
    print("="*70)
    print("\nModel: llama3.1:8b (local, no API costs!)\n")

    # Import
    try:
        from hallbayes.htk_backends import OllamaBackend
        from glass import GlassPlanner, GlassItem
        from glass.visualizer import print_single_result, print_batch_results
    except ImportError as e:
        print(f"Import error: {e}")
        return 1

    # Initialize with longer timeout for local models
    print("1. Initializing Ollama backend...")
    try:
        backend = OllamaBackend(
            model="llama3.1:8b",
            host="http://localhost:11434",
            request_timeout=180.0  # 3 minutes for local inference
        )
        print("   ✓ Backend initialized (timeout: 180s)")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        print("\n   Make sure Ollama is running:")
        print("   $ ollama serve")
        return 1

    # Create planner
    print("\n2. Creating Glass planner...")
    planner = GlassPlanner(backend, temperature=0.3, verbose=False)
    print("   ✓ Planner created")

    # Test prompts (starting with just 1 for speed)
    prompts = [
        "What is the capital of France?",
    ]

    print(f"\n3. Testing {len(prompts)} prompts...")
    items = [GlassItem(prompt=p) for p in prompts]

    # Run
    import time
    start = time.time()

    try:
        metrics = planner.run(items, h_star=0.05)
        elapsed = time.time() - start
        print(f"   ✓ Completed in {elapsed:.2f}s")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    for i, (prompt, m) in enumerate(zip(prompts, metrics), 1):
        decision = "✓ ANSWER" if m.decision_answer else "✗ REFUSE"
        sym = f"{m.symmetry_score:.3f}"
        isr = f"{m.isr:.1f}"

        print(f"\n[{i}] {prompt}")
        print(f"    Decision: {decision}")
        print(f"    Symmetry: {sym}")
        print(f"    ISR: {isr}")
        print(f"    RoH bound: {m.roh_bound:.3f}")

    # Summary
    answered = sum(1 for m in metrics if m.decision_answer)
    refused = len(metrics) - answered

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total: {len(metrics)}")
    print(f"Answered: {answered} ({answered/len(metrics)*100:.1f}%)")
    print(f"Refused: {refused} ({refused/len(metrics)*100:.1f}%)")
    print(f"Time: {elapsed:.2f}s ({elapsed/len(metrics):.2f}s/item)")
    print(f"\n✅ Glass works with Ollama!")
    print("✅ No API costs - completely local")
    print("✅ Privacy-first - data never leaves your machine\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
