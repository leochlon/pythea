"""
Glass Monitoring & Logging Utilities
=====================================

Production-ready monitoring, logging, and metrics collection.
"""

import logging
import time
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(
    log_file: Optional[str] = None,
    level: str = "INFO",
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup structured logging for Glass.

    Args:
        log_file: Path to log file (None = console only)
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        format_string: Custom format string

    Returns:
        Configured logger

    Example:
        logger = setup_logging("glass.log", level="INFO")
        logger.info("Glass started")
    """
    if format_string is None:
        format_string = (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '%(funcName)s:%(lineno)d - %(message)s'
        )

    handlers = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=handlers
    )

    logger = logging.getLogger('glass')
    return logger


# =============================================================================
# Metrics Collection
# =============================================================================

@dataclass
class EvaluationMetrics:
    """Metrics for a single evaluation run"""
    timestamp: str
    num_items: int
    num_answered: int
    num_refused: int
    total_time: float
    average_time: float
    backend: str
    symmetry_scores: List[float]
    isr_scores: List[float]
    cache_hit_rate: Optional[float] = None


class MetricsCollector:
    """
    Collects and exports Glass metrics.

    Supports JSON export, Prometheus format, and real-time monitoring.
    """

    def __init__(self, export_dir: Optional[Path] = None):
        """
        Args:
            export_dir: Directory to export metrics (None = memory only)
        """
        self.export_dir = export_dir
        if export_dir:
            export_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_history: List[EvaluationMetrics] = []
        self.logger = logging.getLogger('glass.metrics')

    def record_evaluation(
        self,
        items: List,
        metrics: List,
        elapsed_time: float,
        backend_name: str,
        cache_stats: Optional[Dict] = None
    ):
        """Record metrics from an evaluation run"""

        num_answered = sum(1 for m in metrics if m.decision_answer)
        num_refused = len(metrics) - num_answered

        symmetry_scores = [
            m.symmetry_score for m in metrics
            if hasattr(m, 'symmetry_score')
        ]
        isr_scores = [m.isr for m in metrics]

        cache_hit_rate = None
        if cache_stats and cache_stats.get('enabled'):
            cache_hit_rate = cache_stats.get('hit_rate')

        eval_metrics = EvaluationMetrics(
            timestamp=datetime.now().isoformat(),
            num_items=len(items),
            num_answered=num_answered,
            num_refused=num_refused,
            total_time=elapsed_time,
            average_time=elapsed_time / len(items) if items else 0,
            backend=backend_name,
            symmetry_scores=symmetry_scores,
            isr_scores=isr_scores,
            cache_hit_rate=cache_hit_rate,
        )

        self.metrics_history.append(eval_metrics)

        # Log summary
        self.logger.info(
            f"Evaluation completed: {num_answered}/{len(items)} answered, "
            f"{elapsed_time:.2f}s total, {eval_metrics.average_time:.2f}s avg"
        )

        # Export if configured
        if self.export_dir:
            self._export_metrics(eval_metrics)

    def _export_metrics(self, metrics: EvaluationMetrics):
        """Export metrics to JSON file"""
        filename = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.export_dir / filename

        with open(filepath, 'w') as f:
            json.dump(asdict(metrics), f, indent=2)

        self.logger.debug(f"Metrics exported to {filepath}")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all collected metrics"""
        if not self.metrics_history:
            return {"total_evaluations": 0}

        total_items = sum(m.num_items for m in self.metrics_history)
        total_answered = sum(m.num_answered for m in self.metrics_history)
        total_time = sum(m.total_time for m in self.metrics_history)

        all_symmetry = [
            score
            for m in self.metrics_history
            for score in m.symmetry_scores
        ]

        return {
            "total_evaluations": len(self.metrics_history),
            "total_items": total_items,
            "total_answered": total_answered,
            "total_refused": total_items - total_answered,
            "answer_rate": total_answered / total_items if total_items > 0 else 0,
            "total_time": total_time,
            "average_time_per_item": total_time / total_items if total_items > 0 else 0,
            "average_symmetry": sum(all_symmetry) / len(all_symmetry) if all_symmetry else 0,
        }

    def export_summary(self, filepath: str):
        """Export summary to JSON file"""
        summary = self.get_summary()
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Summary exported to {filepath}")

    def to_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        summary = self.get_summary()

        lines = [
            "# HELP glass_evaluations_total Total number of evaluations",
            "# TYPE glass_evaluations_total counter",
            f"glass_evaluations_total {summary['total_evaluations']}",
            "",
            "# HELP glass_items_total Total number of items evaluated",
            "# TYPE glass_items_total counter",
            f"glass_items_total {summary['total_items']}",
            "",
            "# HELP glass_decisions_total Total decisions by type",
            "# TYPE glass_decisions_total counter",
            f'glass_decisions_total{{decision="answer"}} {summary["total_answered"]}',
            f'glass_decisions_total{{decision="refuse"}} {summary["total_refused"]}',
            "",
            "# HELP glass_answer_rate Proportion of answered vs refused",
            "# TYPE glass_answer_rate gauge",
            f"glass_answer_rate {summary['answer_rate']:.4f}",
            "",
            "# HELP glass_evaluation_time_seconds Time spent evaluating",
            "# TYPE glass_evaluation_time_seconds summary",
            f"glass_evaluation_time_seconds_sum {summary['total_time']:.4f}",
            f"glass_evaluation_time_seconds_count {summary['total_evaluations']}",
        ]

        return "\n".join(lines)


# =============================================================================
# Monitored Planner Wrapper
# =============================================================================

class MonitoredPlanner:
    """
    Wrapper around GlassPlanner with built-in monitoring.

    Automatically logs execution, collects metrics, and exports data.
    """

    def __init__(
        self,
        planner,
        backend_name: str = "unknown",
        enable_logging: bool = True,
        enable_metrics: bool = True,
        metrics_dir: Optional[str] = None,
    ):
        """
        Args:
            planner: GlassPlanner instance
            backend_name: Name of backend for logging
            enable_logging: Enable structured logging
            enable_metrics: Enable metrics collection
            metrics_dir: Directory to export metrics
        """
        self.planner = planner
        self.backend_name = backend_name

        # Setup logging
        if enable_logging:
            self.logger = logging.getLogger('glass.monitored')
        else:
            self.logger = None

        # Setup metrics
        if enable_metrics:
            export_path = Path(metrics_dir) if metrics_dir else None
            self.metrics = MetricsCollector(export_dir=export_path)
        else:
            self.metrics = None

    def run(self, items, **kwargs):
        """Run evaluation with monitoring"""

        if self.logger:
            self.logger.info(f"Starting evaluation of {len(items)} items")

        start_time = time.time()

        try:
            # Run evaluation
            metrics = self.planner.run(items, **kwargs)
            elapsed = time.time() - start_time

            # Collect metrics
            if self.metrics:
                cache_stats = None
                if hasattr(self.planner, 'mapper'):
                    if hasattr(self.planner.mapper, 'get_cache_stats'):
                        cache_stats = self.planner.mapper.get_cache_stats()

                self.metrics.record_evaluation(
                    items=items,
                    metrics=metrics,
                    elapsed_time=elapsed,
                    backend_name=self.backend_name,
                    cache_stats=cache_stats,
                )

            # Log results
            if self.logger:
                answered = sum(1 for m in metrics if m.decision_answer)
                self.logger.info(
                    f"Evaluation completed: {answered}/{len(items)} answered, "
                    f"{elapsed:.2f}s ({elapsed/len(items):.2f}s per item)"
                )

            return metrics

        except Exception as e:
            if self.logger:
                self.logger.error(f"Evaluation failed: {e}", exc_info=True)
            raise

    def evaluate_item(self, idx, item, **kwargs):
        """Evaluate single item with monitoring"""
        if self.logger:
            self.logger.debug(f"Evaluating item {idx}")

        try:
            return self.planner.evaluate_item(idx, item, **kwargs)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Item {idx} failed: {e}")
            raise

    def get_metrics_summary(self) -> Dict:
        """Get metrics summary"""
        if self.metrics:
            return self.metrics.get_summary()
        return {}

    def export_metrics(self, filepath: str):
        """Export metrics to file"""
        if self.metrics:
            self.metrics.export_summary(filepath)

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        if self.metrics:
            return self.metrics.to_prometheus()
        return ""


# =============================================================================
# Usage Examples
# =============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    # Example: Setup logging
    logger = setup_logging("glass.log", level="INFO")
    logger.info("Glass monitoring started")

    # Example: Metrics collector
    metrics_collector = MetricsCollector(export_dir=Path("metrics"))

    # Simulate some data
    from dataclasses import dataclass

    @dataclass
    class MockMetric:
        decision_answer: bool
        symmetry_score: float
        isr: float

    mock_items = [1, 2, 3]
    mock_metrics = [
        MockMetric(True, 0.85, 15.2),
        MockMetric(True, 0.72, 12.3),
        MockMetric(False, 0.45, 5.1),
    ]

    metrics_collector.record_evaluation(
        items=mock_items,
        metrics=mock_metrics,
        elapsed_time=2.5,
        backend_name="test-backend",
    )

    # Print summary
    summary = metrics_collector.get_summary()
    print("\nMetrics Summary:")
    print(json.dumps(summary, indent=2))

    # Prometheus format
    print("\nPrometheus Format:")
    print(metrics_collector.to_prometheus())

    print("\n✓ Monitoring utilities working!")
