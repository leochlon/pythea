#!/usr/bin/env python3
"""
Benchmark: Entropy Confidence Detector Overhead

Compares:
1. Baseline: OpenAI API call without logprobs
2. With logprobs: Same call + logprobs=True (entropy extraction)
3. Full detector: check_entropy_confidence (includes flagging logic)

Measures latency overhead and shows confidence scores.
"""

import asyncio
import os
import statistics
import time
from typing import Any

# Test queries of varying difficulty
TEST_QUERIES = [
    # Easy (should be high confidence)
    ("What is 2 + 2?", "easy"),
    ("What is the capital of France?", "easy"),
    ("What color is the sky on a clear day?", "easy"),

    # Medium (should be medium-high confidence)
    ("What year did World War II end?", "medium"),
    ("What is the chemical formula for water?", "medium"),
    ("Who wrote Romeo and Juliet?", "medium"),

    # Hard (may show lower confidence)
    ("What is the 15th prime number?", "hard"),
    ("What was the population of Tokyo in 2019?", "hard"),
    ("What is the derivative of x^3 * sin(x)?", "hard"),

    # Very hard (should show high entropy)
    ("What is the 47th digit of pi?", "very_hard"),
    ("What was the exact GDP of Monaco in 1987?", "very_hard"),
]


async def benchmark_baseline(client, query: str, model: str = "gpt-4o-mini") -> dict[str, Any]:
    """Baseline: No logprobs, no entropy."""
    start = time.perf_counter()
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": query}],
        temperature=0.7,
        max_tokens=100,
    )
    elapsed = time.perf_counter() - start

    return {
        "answer": response.choices[0].message.content or "",
        "latency_ms": elapsed * 1000,
        "has_logprobs": False,
    }


async def benchmark_with_logprobs(client, query: str, model: str = "gpt-4o-mini") -> dict[str, Any]:
    """With logprobs enabled (same API call, just +logprobs)."""
    start = time.perf_counter()
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": query}],
        temperature=0.7,
        max_tokens=100,
        logprobs=True,
        top_logprobs=5,
    )
    elapsed = time.perf_counter() - start

    return {
        "answer": response.choices[0].message.content or "",
        "latency_ms": elapsed * 1000,
        "has_logprobs": True,
        "logprobs_count": len(response.choices[0].logprobs.content) if response.choices[0].logprobs else 0,
    }


async def benchmark_full_detector(query: str, model: str = "gpt-4o-mini") -> dict[str, Any]:
    """Full entropy detector with confidence scoring."""
    # Import here to measure full path
    import sys
    sys.path.insert(0, "/Users/nellwatson/Documents/GitHub/pythea/strawberry/src")
    from strawberry.entropy_confidence import check_entropy_confidence_async

    start = time.perf_counter()
    result = await check_entropy_confidence_async(
        query=query,
        model=model,
        task_type="factual",
        temperature=0.7,
        max_tokens=100,
    )
    elapsed = time.perf_counter() - start

    return {
        "answer": result["answer"],
        "latency_ms": elapsed * 1000,
        "confidence_score": result["confidence"]["score"],
        "confidence_level": result["confidence"]["level"],
        "max_entropy": result["confidence"]["max_entropy"],
        "flagged": result["flagged"],
        "should_verify": result["should_verify"],
    }


async def run_benchmark(n_runs: int = 3):
    """Run full benchmark suite."""
    from openai import AsyncOpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return

    client = AsyncOpenAI(api_key=api_key, timeout=30.0)

    print("=" * 80)
    print("ENTROPY CONFIDENCE DETECTOR BENCHMARK")
    print("=" * 80)
    print(f"Model: gpt-4o-mini | Runs per query: {n_runs}")
    print(f"Queries: {len(TEST_QUERIES)}")
    print()

    # Aggregate results
    baseline_latencies = []
    logprobs_latencies = []
    detector_latencies = []
    confidence_by_difficulty = {"easy": [], "medium": [], "hard": [], "very_hard": []}

    for query, difficulty in TEST_QUERIES:
        print(f"\n{'─' * 60}")
        print(f"Query [{difficulty}]: {query[:50]}...")

        baseline_times = []
        logprobs_times = []
        detector_times = []
        detector_result = None

        for run in range(n_runs):
            # Run all three in sequence
            b = await benchmark_baseline(client, query)
            baseline_times.append(b["latency_ms"])

            lp = await benchmark_with_logprobs(client, query)
            logprobs_times.append(lp["latency_ms"])

            d = await benchmark_full_detector(query)
            detector_times.append(d["latency_ms"])
            detector_result = d

        # Calculate stats
        baseline_avg = statistics.mean(baseline_times)
        logprobs_avg = statistics.mean(logprobs_times)
        detector_avg = statistics.mean(detector_times)

        baseline_latencies.extend(baseline_times)
        logprobs_latencies.extend(logprobs_times)
        detector_latencies.extend(detector_times)

        logprobs_overhead = logprobs_avg - baseline_avg
        detector_overhead = detector_avg - baseline_avg

        confidence_by_difficulty[difficulty].append(detector_result["confidence_score"])

        print(f"  Baseline:     {baseline_avg:7.1f} ms")
        print(f"  +Logprobs:    {logprobs_avg:7.1f} ms  (+{logprobs_overhead:+.1f} ms)")
        print(f"  Full detector:{detector_avg:7.1f} ms  (+{detector_overhead:+.1f} ms)")
        print(f"  Confidence:   {detector_result['confidence_score']:.3f} ({detector_result['confidence_level']})")
        print(f"  Max entropy:  {detector_result['max_entropy']:.3f} bits")
        print(f"  Flagged:      {detector_result['flagged']} | Verify: {detector_result['should_verify']}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    baseline_mean = statistics.mean(baseline_latencies)
    logprobs_mean = statistics.mean(logprobs_latencies)
    detector_mean = statistics.mean(detector_latencies)

    print(f"\nLatency (mean of {len(baseline_latencies)} samples):")
    print(f"  Baseline:      {baseline_mean:7.1f} ms")
    print(f"  +Logprobs:     {logprobs_mean:7.1f} ms  (overhead: +{logprobs_mean - baseline_mean:+.1f} ms, {((logprobs_mean/baseline_mean)-1)*100:+.1f}%)")
    print(f"  Full detector: {detector_mean:7.1f} ms  (overhead: +{detector_mean - baseline_mean:+.1f} ms, {((detector_mean/baseline_mean)-1)*100:+.1f}%)")

    print(f"\nConfidence by difficulty:")
    for diff, scores in confidence_by_difficulty.items():
        if scores:
            print(f"  {diff:10s}: {statistics.mean(scores):.3f} (n={len(scores)})")

    print(f"\nEntropy computation overhead (detector - logprobs): {detector_mean - logprobs_mean:+.1f} ms")
    print("\nConclusion:")
    if detector_mean - baseline_mean < 50:
        print("  Minimal overhead (<50ms) - suitable for real-time use")
    elif detector_mean - baseline_mean < 200:
        print("  Moderate overhead - acceptable for most use cases")
    else:
        print("  Significant overhead - consider async/background processing")


if __name__ == "__main__":
    asyncio.run(run_benchmark(n_runs=3))
