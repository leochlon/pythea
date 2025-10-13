#!/usr/bin/env python3
"""
Glass Quick Check - One-liner hallucination detection
======================================================

Usage:
    python glass_check.py "Who won the 2019 Nobel Prize?"

    # With custom backend
    python glass_check.py "Your prompt" --model gpt-4o

    # Batch mode
    python glass_check.py "Prompt 1" "Prompt 2" "Prompt 3"

    # JSON output
    python glass_check.py "Prompt" --json

    # Compare with Original
    python glass_check.py "Prompt" --compare
"""

import sys
import argparse
import json
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))


def check_single(prompt: str, backend, verbose: bool = True) -> dict:
    """Check single prompt with Glass"""
    from glass import GlassPlanner, GlassItem
    from glass.visualizer import print_single_result, quick_print
    import time

    planner = GlassPlanner(backend, temperature=0.3, verbose=False)
    item = GlassItem(prompt=prompt)

    start = time.time()
    metrics = planner.evaluate_item(0, item)
    elapsed = time.time() - start

    if verbose:
        print_single_result(prompt, metrics, show_details=True)
        print(f"\n⏱️  Time: {elapsed:.3f}s")
    else:
        quick_print(metrics, prompt=prompt)

    return {
        "prompt": prompt,
        "decision": "answer" if metrics.decision_answer else "refuse",
        "symmetry": metrics.symmetry_score,
        "isr": metrics.isr,
        "roh_bound": metrics.roh_bound,
        "time": elapsed,
    }


def check_batch(prompts: list, backend, verbose: bool = True) -> list:
    """Check multiple prompts"""
    from glass import GlassPlanner, GlassItem
    from glass.visualizer import print_batch_results

    planner = GlassPlanner(backend, temperature=0.3, verbose=False)
    items = [GlassItem(prompt=p) for p in prompts]

    metrics = planner.run(items)

    if verbose:
        print_batch_results(prompts, metrics)

    return [
        {
            "prompt": p,
            "decision": "answer" if m.decision_answer else "refuse",
            "symmetry": m.symmetry_score,
            "isr": m.isr,
            "roh_bound": m.roh_bound,
        }
        for p, m in zip(prompts, metrics)
    ]


def compare_with_original(prompt: str, backend) -> dict:
    """Compare Glass with Original EDFL"""
    from glass import GlassPlanner, GlassItem
    from hallbayes import OpenAIPlanner, OpenAIItem
    from glass.visualizer import print_comparison
    import time

    # Glass
    glass_planner = GlassPlanner(backend, temperature=0.3)
    glass_item = GlassItem(prompt=prompt)

    glass_start = time.time()
    glass_metrics = glass_planner.evaluate_item(0, glass_item)
    glass_time = time.time() - glass_start

    # Original
    orig_planner = OpenAIPlanner(backend, temperature=0.3)
    orig_item = OpenAIItem(prompt=prompt, n_samples=3, m=4)  # Reduced for speed

    orig_start = time.time()
    orig_metrics = orig_planner.evaluate_item(0, orig_item)
    orig_time = time.time() - orig_start

    # Print comparison
    print_comparison(prompt, glass_metrics, orig_metrics)

    print(f"\n⏱️  Time:")
    print(f"   Glass:    {glass_time:.3f}s (1 call)")
    print(f"   Original: {orig_time:.3f}s ({(1+orig_item.m)*orig_item.n_samples} calls)")
    print(f"   Speedup:  {orig_time/glass_time:.1f}×")

    return {
        "glass": {
            "decision": "answer" if glass_metrics.decision_answer else "refuse",
            "symmetry": glass_metrics.symmetry_score,
            "time": glass_time,
        },
        "original": {
            "decision": "answer" if orig_metrics.decision_answer else "refuse",
            "time": orig_time,
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Glass Quick Check - Fast hallucination detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python glass_check.py "Who won the 2019 Nobel Prize?"
  python glass_check.py "Prompt 1" "Prompt 2" "Prompt 3"
  python glass_check.py "Prompt" --model gpt-4o
  python glass_check.py "Prompt" --json
  python glass_check.py "Prompt" --compare
        """
    )

    parser.add_argument('prompts', nargs='+', help='Prompt(s) to evaluate')
    parser.add_argument('--model', default='gpt-4o-mini', help='Model to use (default: gpt-4o-mini)')
    parser.add_argument('--json', action='store_true', help='Output JSON instead of formatted text')
    parser.add_argument('--compare', action='store_true', help='Compare with Original EDFL')
    parser.add_argument('--quiet', '-q', action='store_true', help='Minimal output')

    args = parser.parse_args()

    # Initialize backend
    try:
        from hallbayes import OpenAIBackend
        backend = OpenAIBackend(model=args.model)
    except Exception as e:
        print(f"Error initializing backend: {e}")
        print("Make sure OPENAI_API_KEY is set.")
        return 1

    # Process
    try:
        if args.compare:
            if len(args.prompts) > 1:
                print("Error: --compare only works with single prompt")
                return 1

            result = compare_with_original(args.prompts[0], backend)

            if args.json:
                print(json.dumps(result, indent=2))

        elif len(args.prompts) == 1:
            result = check_single(args.prompts[0], backend, verbose=not args.quiet)

            if args.json:
                print(json.dumps(result, indent=2))

        else:
            results = check_batch(args.prompts, backend, verbose=not args.quiet)

            if args.json:
                print(json.dumps(results, indent=2))

    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
