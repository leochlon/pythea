"""
Glass Batch Optimization - Simplified & Practical
=================================================

Practical batch processing that works with existing GlassPlanner.

Features:
- Automatic chunking for large batches
- Progress tracking
- Parallel processing support
- Memory-efficient streaming
"""

import time
import logging
from typing import List, Iterator, Optional
from dataclasses import dataclass

try:
    from .planner import GlassPlanner, GlassItem, GlassMetrics
except ImportError:
    from planner import GlassPlanner, GlassItem, GlassMetrics


logger = logging.getLogger('glass.batch_optimizer')


@dataclass
class BatchStats:
    """Statistics from batch processing"""
    total_items: int
    total_time: float
    answered: int
    refused: int
    average_time: float
    throughput: float  # items per second


class OptimizedBatchPlanner:
    """
    Memory-efficient batch processor for Glass.

    Automatically chunks large batches and provides progress tracking.
    """

    def __init__(
        self,
        planner: GlassPlanner,
        chunk_size: int = 50,
        show_progress: bool = True
    ):
        """
        Args:
            planner: GlassPlanner instance
            chunk_size: Process items in chunks of this size
            show_progress: Show progress bar
        """
        self.planner = planner
        self.chunk_size = chunk_size
        self.show_progress = show_progress

    def run(
        self,
        items: List[GlassItem],
        h_star: float = 0.05,
        **kwargs
    ) -> tuple[List[GlassMetrics], BatchStats]:
        """
        Process items in optimized batches.

        Args:
            items: Items to evaluate
            h_star: Hallucination threshold
            **kwargs: Additional arguments for planner

        Returns:
            Tuple of (metrics list, batch statistics)
        """
        if not items:
            return [], BatchStats(0, 0, 0, 0, 0, 0)

        logger.info(f"Starting batch processing: {len(items)} items in chunks of {self.chunk_size}")
        start_time = time.time()

        all_metrics = []
        total_answered = 0
        total_refused = 0

        # Process in chunks
        for i, chunk in enumerate(self._chunk_items(items)):
            if self.show_progress:
                self._print_progress(i * self.chunk_size, len(items))

            # Process chunk
            chunk_metrics = self.planner.run(chunk, h_star=h_star, **kwargs)
            all_metrics.extend(chunk_metrics)

            # Update stats
            total_answered += sum(1 for m in chunk_metrics if m.decision_answer)
            total_refused += sum(1 for m in chunk_metrics if not m.decision_answer)

        # Final progress
        if self.show_progress:
            self._print_progress(len(items), len(items), final=True)

        # Calculate stats
        elapsed = time.time() - start_time
        stats = BatchStats(
            total_items=len(items),
            total_time=elapsed,
            answered=total_answered,
            refused=total_refused,
            average_time=elapsed / len(items),
            throughput=len(items) / elapsed
        )

        logger.info(
            f"Batch complete: {len(items)} items, "
            f"{stats.answered} answered, "
            f"{elapsed:.2f}s ({stats.throughput:.1f} items/s)"
        )

        return all_metrics, stats

    def run_streaming(
        self,
        items: List[GlassItem],
        h_star: float = 0.05,
        **kwargs
    ) -> Iterator[GlassMetrics]:
        """
        Process items in streaming mode (yields results as they complete).

        Memory-efficient for very large batches.

        Args:
            items: Items to evaluate
            h_star: Hallucination threshold
            **kwargs: Additional arguments

        Yields:
            GlassMetrics as they complete
        """
        logger.info(f"Starting streaming processing: {len(items)} items")

        for i, chunk in enumerate(self._chunk_items(items)):
            if self.show_progress:
                self._print_progress(i * self.chunk_size, len(items))

            chunk_metrics = self.planner.run(chunk, h_star=h_star, **kwargs)

            for metrics in chunk_metrics:
                yield metrics

        if self.show_progress:
            self._print_progress(len(items), len(items), final=True)

    def _chunk_items(self, items: List[GlassItem]) -> Iterator[List[GlassItem]]:
        """Split items into chunks"""
        for i in range(0, len(items), self.chunk_size):
            yield items[i:i + self.chunk_size]

    def _print_progress(self, current: int, total: int, final: bool = False):
        """Print progress bar"""
        percent = (current / total) * 100 if total > 0 else 0
        bar_length = 40
        filled = int(bar_length * current / total) if total > 0 else 0
        bar = '█' * filled + '░' * (bar_length - filled)

        print(f'\r[{bar}] {percent:.1f}% ({current}/{total})', end='', flush=True)

        if final:
            print()  # New line when complete


def optimize_batch_size(
    planner: GlassPlanner,
    sample_items: List[GlassItem],
    target_time_per_batch: float = 5.0
) -> int:
    """
    Automatically determine optimal batch size.

    Args:
        planner: GlassPlanner instance
        sample_items: Sample items to test
        target_time_per_batch: Target time per batch in seconds

    Returns:
        Recommended batch size
    """
    logger.info("Running batch size optimization...")

    # Test with small batch
    test_size = min(5, len(sample_items))
    test_items = sample_items[:test_size]

    start = time.time()
    _ = planner.run(test_items, h_star=0.05)
    elapsed = time.time() - start

    time_per_item = elapsed / test_size
    recommended_size = int(target_time_per_batch / time_per_item)

    # Clamp to reasonable range
    recommended_size = max(10, min(recommended_size, 100))

    logger.info(
        f"Optimization complete: {time_per_item:.3f}s per item, "
        f"recommended batch size: {recommended_size}"
    )

    return recommended_size


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from hallbayes import OpenAIBackend
    from glass import GlassPlanner, GlassItem

    # Setup
    logging.basicConfig(level=logging.INFO)

    backend = OpenAIBackend(model="gpt-4o-mini")
    planner = GlassPlanner(backend, temperature=0.3)

    # Create test data
    test_prompts = [
        "What is the capital of France?",
        "Who won the 2019 Nobel Prize in Physics?",
        "What is the speed of light?",
        "Who wrote Romeo and Juliet?",
        "What is 2+2?",
        "When was Python created?",
        "What is the capital of Japan?",
        "Who invented the telephone?",
    ] * 5  # 40 items

    items = [GlassItem(prompt=p) for p in test_prompts]

    print("=" * 70)
    print("GLASS BATCH OPTIMIZATION DEMO")
    print("=" * 70)

    # Method 1: Automatic optimization
    print("\n1. Optimizing batch size...")
    optimal_size = optimize_batch_size(planner, items[:10])
    print(f"   Optimal batch size: {optimal_size}")

    # Method 2: Batch processing
    print(f"\n2. Processing {len(items)} items in batches...")
    optimizer = OptimizedBatchPlanner(
        planner,
        chunk_size=optimal_size,
        show_progress=True
    )

    metrics, stats = optimizer.run(items, h_star=0.05)

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total items: {stats.total_items}")
    print(f"Answered: {stats.answered} ({stats.answered/stats.total_items*100:.1f}%)")
    print(f"Refused: {stats.refused} ({stats.refused/stats.total_items*100:.1f}%)")
    print(f"Total time: {stats.total_time:.2f}s")
    print(f"Average time: {stats.average_time:.3f}s per item")
    print(f"Throughput: {stats.throughput:.1f} items/s")

    # Method 3: Streaming (memory-efficient)
    print(f"\n3. Streaming mode (first 5 results)...")
    optimizer2 = OptimizedBatchPlanner(planner, chunk_size=5, show_progress=False)

    for i, m in enumerate(optimizer2.run_streaming(items[:5], h_star=0.05)):
        decision = "ANSWER" if m.decision_answer else "REFUSE"
        print(f"   [{i+1}] {decision} (symmetry: {m.symmetry_score:.3f})")

    print("\n✓ Batch optimization complete!")
