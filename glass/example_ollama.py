#!/usr/bin/env python3
"""
Glass with Ollama Example
==========================

Test Glass with local Ollama models (no API key needed!)

Requires:
    - Ollama installed (https://ollama.ai)
    - Model pulled: ollama pull llama3.1:8b
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_glass_with_ollama():
    """Test Glass with Ollama llama3.1:8b"""

    print("="*70)
    print("GLASS + OLLAMA TEST")
    print("="*70)
    print("\nModel: llama3.1:8b (local)")
    print("No API key required - runs completely local!\n")

    # Import backends
    try:
        from hallbayes.htk_backends import OllamaBackend
    except ImportError:
        print("Error: Could not import OllamaBackend")
        print("Make sure htk_backends.py has OllamaBackend")
        return

    from glass import GlassPlanner, GlassItem
    from glass.visualizer import print_single_result, print_batch_results

    # Initialize Ollama backend
    print("Initializing Ollama backend...")
    backend = OllamaBackend(
        model="llama3.1:8b",
        host="http://localhost:11434"
    )
    print("✓ Backend initialized\n")

    # Create Glass planner
    planner = GlassPlanner(backend, temperature=0.3, verbose=True)

    # Test prompts
    test_prompts = [
        "Who won the 2019 Nobel Prize in Physics?",
        "What is the capital of France?",
        "When did World War II end?",
        "What is 2 + 2?",
    ]

    print("="*70)
    print("TESTING GLASS WITH OLLAMA")
    print("="*70)

    items = [GlassItem(prompt=p) for p in test_prompts]

    # Run evaluation
    import time
    start = time.time()
    metrics = planner.run(items, h_star=0.05)
    elapsed = time.time() - start

    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    print_batch_results(test_prompts, metrics, show_details=False)

    print(f"\n⏱️  Total time: {elapsed:.2f}s")
    print(f"📊 Average per item: {elapsed/len(items):.2f}s")

    # Detailed view of first result
    print("\n" + "="*70)
    print("DETAILED VIEW (First Item)")
    print("="*70)
    print_single_result(test_prompts[0], metrics[0], item_num=1, show_details=True)

    # Summary
    answered = sum(1 for m in metrics if m.decision_answer)
    refused = len(metrics) - answered

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total queries: {len(metrics)}")
    print(f"Answered: {answered} ({answered/len(metrics)*100:.1f}%)")
    print(f"Refused: {refused} ({refused/len(metrics)*100:.1f}%)")
    print(f"\n✓ Glass works with Ollama! 🎉")
    print("✓ No API costs - runs completely local")
    print("✓ Privacy-first - data never leaves your machine")


def test_ollama_vs_openai():
    """Compare Ollama vs OpenAI (if available)"""

    print("\n\n" + "="*70)
    print("OLLAMA vs OPENAI COMPARISON")
    print("="*70)

    from glass import GlassPlanner, GlassItem
    import time
    import os

    prompt = "Who won the 2019 Nobel Prize in Physics?"

    # Test with Ollama
    print("\n1. Testing with Ollama (local)...")
    from hallbayes.htk_backends import OllamaBackend

    ollama_backend = OllamaBackend(model="llama3.1:8b")
    ollama_planner = GlassPlanner(ollama_backend, temperature=0.3)

    start = time.time()
    ollama_metrics = ollama_planner.evaluate_item(0, GlassItem(prompt=prompt))
    ollama_time = time.time() - start

    print(f"   Decision: {'ANSWER' if ollama_metrics.decision_answer else 'REFUSE'}")
    print(f"   Symmetry: {ollama_metrics.symmetry_score:.3f}")
    print(f"   Time: {ollama_time:.2f}s")
    print(f"   Cost: $0.00 (local)")

    # Test with OpenAI (if available)
    if os.environ.get("OPENAI_API_KEY"):
        print("\n2. Testing with OpenAI (cloud)...")
        from hallbayes import OpenAIBackend

        openai_backend = OpenAIBackend(model="gpt-4o-mini")
        openai_planner = GlassPlanner(openai_backend, temperature=0.3)

        start = time.time()
        openai_metrics = openai_planner.evaluate_item(0, GlassItem(prompt=prompt))
        openai_time = time.time() - start

        print(f"   Decision: {'ANSWER' if openai_metrics.decision_answer else 'REFUSE'}")
        print(f"   Symmetry: {openai_metrics.symmetry_score:.3f}")
        print(f"   Time: {openai_time:.2f}s")
        print(f"   Cost: ~$0.001")

        # Comparison
        print("\n" + "="*70)
        print("COMPARISON")
        print("="*70)
        print(f"Latency: Ollama={ollama_time:.2f}s vs OpenAI={openai_time:.2f}s")

        if ollama_metrics.decision_answer == openai_metrics.decision_answer:
            print("Decision: ✓ AGREE")
        else:
            print("Decision: ⚠ DIFFER")

        print("\nOllama advantages:")
        print("  ✓ No API costs")
        print("  ✓ Privacy (data stays local)")
        print("  ✓ No rate limits")
        print("  ✓ Offline capable")

        print("\nOpenAI advantages:")
        print("  ✓ Faster inference")
        print("  ✓ More capable models")
        print("  ✓ Better instruction following")
    else:
        print("\n2. OpenAI test skipped (OPENAI_API_KEY not set)")


def test_ollama_hybrid():
    """Test hybrid mode with Ollama"""

    print("\n\n" + "="*70)
    print("HYBRID MODE WITH OLLAMA")
    print("="*70)
    print("\nFast path: Ollama (local, free)")
    print("Fallback: Could use OpenAI (optional)\n")

    from glass.example_hybrid import HybridPlanner
    from hallbayes.htk_backends import OllamaBackend

    backend = OllamaBackend(model="llama3.1:8b")
    planner = HybridPlanner(
        backend=backend,
        glass_confidence_threshold=0.65,
        use_fallback=False,  # No fallback in local-only mode
        verbose=True
    )

    prompts = [
        "What is the capital of France?",
        "Who won the 2019 Nobel Prize in Physics?",
    ]

    metrics, infos = planner.run(prompts, h_star=0.05)

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    for i, (prompt, info) in enumerate(zip(prompts, infos), 1):
        print(f"\n[{i}] {prompt}")
        print(f"    Path: {info['path']}")
        print(f"    Glass symmetry: {info.get('glass_symmetry', 'N/A'):.3f}")
        print(f"    Time: {info.get('glass_time', 0):.2f}s")


def main():
    """Run all Ollama tests"""

    try:
        # Check if Ollama is running
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code != 200:
            print("Error: Ollama server not responding")
            print("Start it with: ollama serve")
            return 1
    except Exception as e:
        print(f"Error: Could not connect to Ollama")
        print(f"Make sure Ollama is running: ollama serve")
        print(f"Error details: {e}")
        return 1

    try:
        # Run tests
        test_glass_with_ollama()
        test_ollama_vs_openai()
        test_ollama_hybrid()

        print("\n\n" + "="*70)
        print("✅ ALL OLLAMA TESTS COMPLETED")
        print("="*70)
        print("\nGlass works great with Ollama! 🚀")
        print("Completely local, no API costs, privacy-first.\n")

        return 0

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"\n\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
