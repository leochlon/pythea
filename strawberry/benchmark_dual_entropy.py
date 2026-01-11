#!/usr/bin/env python3
"""
Benchmark: Dual Entropy (Max vs Factual First-Mention)

Compares two confidence scoring methods:
1. Max entropy across all tokens (original)
2. Factual first-mention p95 (dual entropy)

Measures false positive rate and confidence calibration.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Any

# Add src to path
sys.path.insert(0, "/Users/nellwatson/Documents/GitHub/pythea/strawberry/src")

from openai import AsyncOpenAI

from strawberry.entropy_confidence import (
    extract_dual_entropy_from_openai_response,
    get_confidence_level,
    ConfidenceLevel,
)

# Test queries with expected correctness
TEST_QUERIES = [
    # Easy - should NOT be flagged
    {"query": "What is 2 + 2?", "difficulty": "easy", "expected_correct": True},
    {"query": "What is the capital of France?", "difficulty": "easy", "expected_correct": True},
    {"query": "What color is the sky on a clear day?", "difficulty": "easy", "expected_correct": True},
    {"query": "How many legs does a dog have?", "difficulty": "easy", "expected_correct": True},

    # Medium - should NOT be flagged
    {"query": "What year did World War II end?", "difficulty": "medium", "expected_correct": True},
    {"query": "What is the chemical formula for water?", "difficulty": "medium", "expected_correct": True},
    {"query": "Who wrote Romeo and Juliet?", "difficulty": "medium", "expected_correct": True},
    {"query": "What is the boiling point of water in Celsius?", "difficulty": "medium", "expected_correct": True},

    # Hard - may have lower confidence but often correct
    {"query": "What is the 15th prime number?", "difficulty": "hard", "expected_correct": True},
    {"query": "What was the population of Tokyo in 2019?", "difficulty": "hard", "expected_correct": True},
    {"query": "In what year was the Treaty of Tordesillas signed?", "difficulty": "hard", "expected_correct": True},

    # Very hard - likely incorrect, should be flagged
    {"query": "What is the 47th digit of pi?", "difficulty": "very_hard", "expected_correct": False},
    {"query": "What was the exact GDP of Monaco in 1987?", "difficulty": "very_hard", "expected_correct": False},
    {"query": "What was the population of Tokyo in 1950 to the nearest million?", "difficulty": "very_hard", "expected_correct": False},
]


async def run_query(client: AsyncOpenAI, query: str, model: str = "gpt-4o-mini") -> dict[str, Any]:
    """Run a query and extract both max entropy and dual entropy metrics."""
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": query}],
        temperature=0.7,
        max_tokens=150,
        logprobs=True,
        top_logprobs=5,
    )

    answer = response.choices[0].message.content or ""

    # Extract dual entropy metrics
    metrics = extract_dual_entropy_from_openai_response(response, model)

    # Compute confidence scores using both methods
    # Method 1: Max entropy (original)
    max_entropy_confidence = 1.0 / (1.0 + metrics.max_entropy)

    # Method 2: Factual first-mention p95 (dual entropy)
    if metrics.n_factual_tokens > 0 and metrics.factual_first_mention_p95 is not None:
        factual_confidence = 1.0 / (1.0 + metrics.factual_first_mention_p95)
    else:
        # Fall back to p95 if no factual tokens
        factual_confidence = 1.0 / (1.0 + metrics.p95_entropy)

    # Flagging thresholds (confidence < 0.5 = flagged)
    max_entropy_flagged = max_entropy_confidence < 0.5
    factual_flagged = factual_confidence < 0.5

    return {
        "query": query,
        "answer": answer[:100] + "..." if len(answer) > 100 else answer,
        "n_tokens": metrics.n_tokens,
        "n_factual_tokens": metrics.n_factual_tokens,
        "n_expressive_tokens": metrics.n_expressive_tokens,
        # Max entropy method
        "max_entropy": metrics.max_entropy,
        "max_entropy_confidence": max_entropy_confidence,
        "max_entropy_flagged": max_entropy_flagged,
        # Dual entropy method
        "factual_first_mention_p95": metrics.factual_first_mention_p95,
        "factual_p95": metrics.factual_p95,
        "expressive_p95": metrics.expressive_p95,
        "factual_confidence": factual_confidence,
        "factual_flagged": factual_flagged,
    }


async def run_benchmark():
    """Run full benchmark comparing max entropy vs dual entropy."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return

    client = AsyncOpenAI(api_key=api_key, timeout=60.0)

    print("=" * 80)
    print("DUAL ENTROPY BENCHMARK: Max Entropy vs Factual First-Mention")
    print("=" * 80)
    print(f"Model: gpt-4o-mini")
    print(f"Queries: {len(TEST_QUERIES)}")
    print()

    results = []
    confidence_by_difficulty = {
        "easy": {"max": [], "factual": []},
        "medium": {"max": [], "factual": []},
        "hard": {"max": [], "factual": []},
        "very_hard": {"max": [], "factual": []},
    }

    false_positives = {"max": 0, "factual": 0}
    correct_flags = {"max": 0, "factual": 0}

    for item in TEST_QUERIES:
        query = item["query"]
        difficulty = item["difficulty"]
        expected_correct = item["expected_correct"]

        print(f"\n{'─' * 60}")
        print(f"[{difficulty.upper()}] {query}")

        try:
            result = await run_query(client, query)
            result["difficulty"] = difficulty
            result["expected_correct"] = expected_correct
            results.append(result)

            # Track confidence by difficulty
            confidence_by_difficulty[difficulty]["max"].append(result["max_entropy_confidence"])
            confidence_by_difficulty[difficulty]["factual"].append(result["factual_confidence"])

            # Track false positives (flagged when should be correct)
            if expected_correct:
                if result["max_entropy_flagged"]:
                    false_positives["max"] += 1
                if result["factual_flagged"]:
                    false_positives["factual"] += 1
            else:
                # Track correct flags (flagged when should be incorrect)
                if result["max_entropy_flagged"]:
                    correct_flags["max"] += 1
                if result["factual_flagged"]:
                    correct_flags["factual"] += 1

            # Display results
            print(f"  Answer: {result['answer'][:60]}...")
            print(f"  Tokens: {result['n_tokens']} total ({result['n_factual_tokens']} factual, {result['n_expressive_tokens']} expressive)")
            print()
            print(f"  MAX ENTROPY method:")
            print(f"    max_entropy: {result['max_entropy']:.3f} bits")
            print(f"    confidence:  {result['max_entropy_confidence']:.3f}")
            print(f"    flagged:     {result['max_entropy_flagged']}")
            print()
            print(f"  DUAL ENTROPY method:")
            print(f"    factual_first_mention_p95: {result['factual_first_mention_p95']:.3f} bits" if result['factual_first_mention_p95'] else "    factual_first_mention_p95: N/A")
            print(f"    confidence:  {result['factual_confidence']:.3f}")
            print(f"    flagged:     {result['factual_flagged']}")

            # Highlight differences
            if result["max_entropy_flagged"] != result["factual_flagged"]:
                if result["max_entropy_flagged"] and not result["factual_flagged"]:
                    print(f"  >>> DUAL ENTROPY FIXED FALSE POSITIVE <<<")
                else:
                    print(f"  >>> METHODS DISAGREE <<<")

        except Exception as e:
            print(f"  ERROR: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\nConfidence by difficulty (mean):")
    print(f"{'Difficulty':<12} {'Max Entropy':<15} {'Factual FM':<15} {'Delta':<10}")
    print("-" * 52)

    for diff in ["easy", "medium", "hard", "very_hard"]:
        max_scores = confidence_by_difficulty[diff]["max"]
        fac_scores = confidence_by_difficulty[diff]["factual"]
        if max_scores and fac_scores:
            max_mean = sum(max_scores) / len(max_scores)
            fac_mean = sum(fac_scores) / len(fac_scores)
            delta = fac_mean - max_mean
            print(f"{diff:<12} {max_mean:<15.3f} {fac_mean:<15.3f} {delta:+.3f}")

    n_correct_expected = sum(1 for q in TEST_QUERIES if q["expected_correct"])
    n_incorrect_expected = len(TEST_QUERIES) - n_correct_expected

    print(f"\nFalse positives (flagged when should be correct):")
    print(f"  Max entropy:  {false_positives['max']}/{n_correct_expected}")
    print(f"  Dual entropy: {false_positives['factual']}/{n_correct_expected}")

    print(f"\nCorrect flags (flagged when likely incorrect):")
    print(f"  Max entropy:  {correct_flags['max']}/{n_incorrect_expected}")
    print(f"  Dual entropy: {correct_flags['factual']}/{n_incorrect_expected}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"/Users/nellwatson/Documents/GitHub/Entropy/The Universal Algorithm/demos/path3_architecture/activation_steering/results/dual_entropy_benchmark_{timestamp}.json"

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "dual_entropy_benchmark",
        "model": "gpt-4o-mini",
        "n_queries": len(TEST_QUERIES),
        "results": results,
        "summary": {
            "confidence_by_difficulty": {
                diff: {
                    "max_mean": sum(confidence_by_difficulty[diff]["max"]) / len(confidence_by_difficulty[diff]["max"]) if confidence_by_difficulty[diff]["max"] else 0,
                    "factual_mean": sum(confidence_by_difficulty[diff]["factual"]) / len(confidence_by_difficulty[diff]["factual"]) if confidence_by_difficulty[diff]["factual"] else 0,
                }
                for diff in ["easy", "medium", "hard", "very_hard"]
            },
            "false_positives": false_positives,
            "correct_flags": correct_flags,
            "n_correct_expected": n_correct_expected,
            "n_incorrect_expected": n_incorrect_expected,
        }
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
