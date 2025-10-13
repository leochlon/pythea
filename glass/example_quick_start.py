#!/usr/bin/env python3
"""
Glass Quick Start Example
==========================

Shows how to use Glass for fast hallucination detection.
"""

import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def example_basic():
    """Basic Glass usage"""
    print("="*60)
    print("EXAMPLE 1: Basic Glass Usage")
    print("="*60)

    from glass import GlassPlanner, GlassItem

    # Check if API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n⚠️  OPENAI_API_KEY not set. Using mock backend for demo.\n")

        # Mock backend for demo
        class MockBackend:
            def chat_create(self, messages, **kwargs):
                class MockResponse:
                    class Choice:
                        class Message:
                            content = "James Peebles, Michel Mayor, and Didier Queloz won the 2019 Nobel Prize in Physics."
                        message = Message()
                    choices = [Choice()]
                return MockResponse()

        backend = MockBackend()
        print("Using mock backend (demo mode)")
    else:
        from hallbayes import OpenAIBackend
        backend = OpenAIBackend(model="gpt-4o-mini")
        print(f"Using OpenAI backend: {backend.model}")

    # Create planner
    planner = GlassPlanner(backend, temperature=0.3, verbose=True)

    # Evaluate single prompt
    item = GlassItem(prompt="Who won the 2019 Nobel Prize in Physics?")
    metrics = planner.evaluate_item(0, item)

    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Prompt: {item.prompt}")
    print(f"\nDecision: {'✓ ANSWER' if metrics.decision_answer else '✗ REFUSE'}")
    print(f"Symmetry Score: {metrics.symmetry_score:.3f}")
    print(f"ISR: {metrics.isr:.2f}")
    print(f"RoH Bound: {metrics.roh_bound:.3f}")
    print(f"\nRationale: {metrics.rationale}")


def example_batch():
    """Batch evaluation"""
    print("\n\n" + "="*60)
    print("EXAMPLE 2: Batch Evaluation")
    print("="*60)

    from glass import GlassPlanner, GlassItem

    # Mock backend
    class MockBackend:
        def chat_create(self, messages, **kwargs):
            # Simple mock responses
            responses = {
                "Who won": "James Peebles won the 2019 Nobel Prize in Physics.",
                "What is": "Paris is the capital of France.",
                "When did": "World War II ended in 1945.",
            }
            prompt = messages[-1]["content"]
            for key, response in responses.items():
                if key in prompt:
                    break
            else:
                response = "I don't know."

            class MockResponse:
                class Choice:
                    class Message:
                        content = response
                    message = Message()
                choices = [Choice()]
            return MockResponse()

    backend = MockBackend()
    planner = GlassPlanner(backend, temperature=0.3)

    # Multiple items
    items = [
        GlassItem(prompt="Who won the 2019 Nobel Prize in Physics?"),
        GlassItem(prompt="What is the capital of France?"),
        GlassItem(prompt="When did World War II end?"),
    ]

    # Batch evaluation
    metrics = planner.run(items, h_star=0.05)

    # Print results
    print("\n" + "="*60)
    print("BATCH RESULTS")
    print("="*60)

    for i, (item, metric) in enumerate(zip(items, metrics)):
        decision = "✓ ANSWER" if metric.decision_answer else "✗ REFUSE"
        print(f"\n[{i+1}] {item.prompt}")
        print(f"    {decision} | Symmetry={metric.symmetry_score:.2f} | ISR={metric.isr:.1f}")


def example_comparison():
    """Compare Glass vs Original"""
    print("\n\n" + "="*60)
    print("EXAMPLE 3: Glass vs Original Comparison")
    print("="*60)

    from glass import GlassPlanner, GlassItem
    from hallbayes import OpenAIPlanner, OpenAIItem
    import time

    # Mock backend
    class MockBackend:
        def chat_create(self, messages, **kwargs):
            class MockResponse:
                class Choice:
                    class Message:
                        content = "James Peebles won the 2019 Nobel Prize."
                    message = Message()
                choices = [Choice()]
            return MockResponse()

        def multi_choice(self, messages, n=1, **kwargs):
            choices = []
            for _ in range(n):
                class Choice:
                    class Message:
                        content = '{"decision": "answer"}'
                    message = Message()
                choices.append(Choice())
            return choices

    backend = MockBackend()
    prompt = "Who won the 2019 Nobel Prize in Physics?"

    # Original EDFL
    print("\n🔬 Original EDFL (ensemble sampling):")
    orig_item = OpenAIItem(prompt=prompt, n_samples=3, m=4)  # Reduced for demo
    orig_planner = OpenAIPlanner(backend)

    start = time.time()
    orig_metrics = orig_planner.run([orig_item])
    orig_time = time.time() - start
    orig_calls = (1 + orig_item.m) * orig_item.n_samples

    print(f"   Time: {orig_time:.3f}s")
    print(f"   API calls: {orig_calls}")
    print(f"   Decision: {'ANSWER' if orig_metrics[0].decision_answer else 'REFUSE'}")

    # Glass
    print("\n✨ Glass (grammatical symmetry):")
    glass_item = GlassItem(prompt=prompt)
    glass_planner = GlassPlanner(backend)

    start = time.time()
    glass_metrics = glass_planner.run([glass_item])
    glass_time = time.time() - start
    glass_calls = 1

    print(f"   Time: {glass_time:.3f}s")
    print(f"   API calls: {glass_calls}")
    print(f"   Decision: {'ANSWER' if glass_metrics[0].decision_answer else 'REFUSE'}")

    # Comparison
    speedup = orig_time / glass_time if glass_time > 0 else float('inf')
    call_reduction = orig_calls / glass_calls

    print(f"\n📊 Improvement:")
    print(f"   Speedup: {speedup:.1f}×")
    print(f"   Call reduction: {call_reduction:.1f}×")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("GLASS QUICK START EXAMPLES")
    print("="*60)
    print("\nThese examples demonstrate Glass's key features:")
    print("  1. Basic usage")
    print("  2. Batch evaluation")
    print("  3. Comparison with Original EDFL")
    print("\nNote: Using mock backend for demo (no API key needed)")
    print("="*60)

    try:
        example_basic()
        example_batch()
        example_comparison()

        print("\n\n" + "="*60)
        print("🎉 ALL EXAMPLES COMPLETED")
        print("="*60)
        print("\nNext steps:")
        print("  1. Set OPENAI_API_KEY to use real backend")
        print("  2. Run benchmarks: python benchmarks/compare.py")
        print("  3. See glass/README.md for full documentation")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
