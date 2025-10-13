#!/usr/bin/env python3
"""
Glass Hybrid Mode Example
==========================

Demonstrates the recommended hybrid approach:
1. Fast path with Glass (1 call)
2. Fallback to Original EDFL if Glass refuses (30-42 calls)

This gives you 30× average speedup with original quality on uncertain cases.
"""

import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from glass import GlassPlanner, GlassItem
from hallbayes import OpenAIPlanner, OpenAIItem, ItemMetrics


class HybridPlanner:
    """
    Hybrid planner that combines Glass (fast) with Original EDFL (accurate).

    Strategy:
    1. Try Glass first (1 API call)
    2. If Glass answers with high confidence → return immediately
    3. If Glass refuses or low confidence → fallback to Original EDFL

    This achieves:
    - Average 20-30× speedup (most queries answered by Glass)
    - Original quality on edge cases
    - Automatic fallback for safety
    """

    def __init__(
        self,
        backend,
        glass_confidence_threshold: float = 0.7,
        use_fallback: bool = True,
        verbose: bool = True,
    ):
        """
        Args:
            backend: LLM backend (OpenAI, Anthropic, etc.)
            glass_confidence_threshold: Min symmetry for Glass to answer
            use_fallback: If True, use Original EDFL as fallback
            verbose: Print decision path
        """
        self.glass_planner = GlassPlanner(
            backend,
            temperature=0.3,
            symmetry_threshold=glass_confidence_threshold,
            verbose=False,
        )
        self.original_planner = OpenAIPlanner(backend, temperature=0.3)
        self.glass_confidence_threshold = glass_confidence_threshold
        self.use_fallback = use_fallback
        self.verbose = verbose

        # Statistics
        self.stats = {
            "total": 0,
            "glass_only": 0,
            "fallback_used": 0,
            "glass_time": 0.0,
            "original_time": 0.0,
        }

    def evaluate_item(
        self,
        idx: int,
        prompt: str,
        h_star: float = 0.05,
        **kwargs
    ) -> tuple[ItemMetrics, dict]:
        """
        Evaluate single item with hybrid approach.

        Returns:
            (metrics, decision_info)
        """
        self.stats["total"] += 1
        decision_info = {"path": "unknown", "glass_tried": False, "fallback_used": False}

        # Step 1: Try Glass first
        if self.verbose:
            print(f"\n[Hybrid] Evaluating: {prompt[:60]}...")
            print(f"[Hybrid] 🚀 Trying Glass first (fast path)...")

        glass_item = GlassItem(prompt=prompt, symmetry_threshold=self.glass_confidence_threshold)

        glass_start = time.time()
        glass_metrics = self.glass_planner.evaluate_item(idx, glass_item, h_star=h_star, **kwargs)
        glass_elapsed = time.time() - glass_start

        self.stats["glass_time"] += glass_elapsed
        decision_info["glass_tried"] = True
        decision_info["glass_symmetry"] = glass_metrics.symmetry_score
        decision_info["glass_time"] = glass_elapsed

        # Step 2: Decide if Glass is confident enough
        high_confidence = (
            glass_metrics.decision_answer and
            glass_metrics.symmetry_score >= self.glass_confidence_threshold
        )

        if high_confidence:
            # Glass is confident → use it!
            if self.verbose:
                print(f"[Hybrid] ✅ Glass confident (sym={glass_metrics.symmetry_score:.3f}) → ANSWER")
                print(f"[Hybrid] ⚡ Fast path completed in {glass_elapsed:.3f}s")

            self.stats["glass_only"] += 1
            decision_info["path"] = "glass_only"

            # Convert GlassMetrics to ItemMetrics
            return self._glass_to_item_metrics(glass_metrics), decision_info

        # Step 3: Glass uncertain → fallback to Original?
        if not self.use_fallback:
            if self.verbose:
                print(f"[Hybrid] ⚠️  Glass uncertain (sym={glass_metrics.symmetry_score:.3f}) but fallback disabled")
                print(f"[Hybrid] → Using Glass result anyway")

            decision_info["path"] = "glass_uncertain"
            return self._glass_to_item_metrics(glass_metrics), decision_info

        # Step 4: Fallback to Original EDFL
        if self.verbose:
            print(f"[Hybrid] ⚠️  Glass uncertain (sym={glass_metrics.symmetry_score:.3f})")
            print(f"[Hybrid] 🔬 Falling back to Original EDFL (accurate path)...")

        original_item = OpenAIItem(
            prompt=prompt,
            n_samples=5,
            m=6,
            skeleton_policy="closed_book"
        )

        original_start = time.time()
        original_metrics = self.original_planner.evaluate_item(
            idx, original_item, h_star=h_star, **kwargs
        )
        original_elapsed = time.time() - original_start

        self.stats["original_time"] += original_elapsed
        self.stats["fallback_used"] += 1
        decision_info["fallback_used"] = True
        decision_info["original_time"] = original_elapsed
        decision_info["path"] = "fallback"

        if self.verbose:
            decision = "ANSWER" if original_metrics.decision_answer else "REFUSE"
            print(f"[Hybrid] ✅ Original EDFL decided: {decision}")
            print(f"[Hybrid] 🐢 Accurate path completed in {original_elapsed:.3f}s")
            total_time = glass_elapsed + original_elapsed
            print(f"[Hybrid] ⏱️  Total time: {total_time:.3f}s")

        return original_metrics, decision_info

    def run(
        self,
        prompts: List[str],
        h_star: float = 0.05,
        **kwargs
    ) -> tuple[List[ItemMetrics], List[dict]]:
        """
        Evaluate multiple prompts with hybrid approach.

        Returns:
            (metrics_list, decision_info_list)
        """
        metrics = []
        infos = []

        for i, prompt in enumerate(prompts):
            m, info = self.evaluate_item(i, prompt, h_star=h_star, **kwargs)
            metrics.append(m)
            infos.append(info)

        return metrics, infos

    def print_stats(self):
        """Print performance statistics"""
        total = self.stats["total"]
        if total == 0:
            print("No items evaluated yet.")
            return

        glass_only = self.stats["glass_only"]
        fallback = self.stats["fallback_used"]
        glass_rate = (glass_only / total) * 100
        fallback_rate = (fallback / total) * 100

        avg_glass_time = self.stats["glass_time"] / total
        avg_original_time = (
            self.stats["original_time"] / fallback if fallback > 0 else 0
        )

        print("\n" + "="*60)
        print("HYBRID PLANNER STATISTICS")
        print("="*60)
        print(f"Total items: {total}")
        print(f"Glass only: {glass_only} ({glass_rate:.1f}%)")
        print(f"Fallback used: {fallback} ({fallback_rate:.1f}%)")
        print(f"\nAverage time per item:")
        print(f"  Glass path: {avg_glass_time:.3f}s")
        if fallback > 0:
            print(f"  Original path: {avg_original_time:.3f}s")
        print("="*60)

    def _glass_to_item_metrics(self, glass_metrics) -> ItemMetrics:
        """Convert GlassMetrics to ItemMetrics for compatibility"""
        return ItemMetrics(
            item_id=glass_metrics.item_id,
            delta_bar=glass_metrics.delta_bar,
            q_avg=glass_metrics.q_avg,
            q_conservative=glass_metrics.q_conservative,
            b2t=glass_metrics.b2t,
            isr=glass_metrics.isr,
            roh_bound=glass_metrics.roh_bound,
            decision_answer=glass_metrics.decision_answer,
            rationale=glass_metrics.rationale,
            attempted=glass_metrics.attempted,
            answered_correctly=glass_metrics.answered_correctly,
            meta=glass_metrics.meta,
        )


def main():
    """Demo of hybrid approach"""
    print("="*60)
    print("HYBRID MODE DEMO")
    print("="*60)
    print("\nCombines Glass (fast) with Original EDFL (accurate)")
    print("Strategy: Try Glass first, fallback if uncertain\n")

    # Mock backend for demo
    class MockBackend:
        def __init__(self):
            self.call_count = 0

        def chat_create(self, messages, **kwargs):
            self.call_count += 1
            prompt = messages[-1]["content"]

            # Simulate different response qualities
            if "Nobel Prize" in prompt:
                content = "James Peebles, Michel Mayor, and Didier Queloz won the 2019 Nobel Prize in Physics."
            elif "capital" in prompt:
                content = "Paris is the capital of France."
            else:
                content = "I'm not entirely sure about that."

            class MockResponse:
                class Choice:
                    class Message:
                        pass
                    message = Message()
                choices = [Choice()]

            resp = MockResponse()
            resp.choices[0].message.content = content
            return resp

        def multi_choice(self, messages, n=1, **kwargs):
            choices = []
            for _ in range(n):
                self.call_count += 1
                class Choice:
                    class Message:
                        content = '{"decision": "answer"}'
                    message = Message()
                choices.append(Choice())
            return choices

    backend = MockBackend()
    planner = HybridPlanner(backend, glass_confidence_threshold=0.65, verbose=True)

    # Test prompts with varying difficulty
    prompts = [
        "Who won the 2019 Nobel Prize in Physics?",  # High confidence
        "What is the capital of France?",            # High confidence
        "What is the meaning of life?",              # Low confidence → fallback
    ]

    # Run hybrid evaluation
    print("="*60)
    print("EVALUATING PROMPTS")
    print("="*60)

    metrics, infos = planner.run(prompts, h_star=0.05)

    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    for i, (prompt, metric, info) in enumerate(zip(prompts, metrics, infos)):
        decision = "✓ ANSWER" if metric.decision_answer else "✗ REFUSE"
        path = info["path"]
        print(f"\n[{i+1}] {prompt}")
        print(f"    Decision: {decision}")
        print(f"    Path: {path}")
        if "glass_symmetry" in info:
            print(f"    Glass symmetry: {info['glass_symmetry']:.3f}")
        if info["fallback_used"]:
            print(f"    ⚠️  Fallback was used (high-accuracy mode)")

    # Statistics
    planner.print_stats()

    print(f"\nTotal API calls: {backend.call_count}")
    print(f"Average calls per item: {backend.call_count / len(prompts):.1f}")
    print("\n💡 Tip: Glass handles confident cases fast, Original ensures accuracy on edge cases\n")


if __name__ == "__main__":
    main()
