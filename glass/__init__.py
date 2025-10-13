"""
Glass: Grammatical LLM Analysis & Symmetry System
===================================================

Fast hallucination detection using grammatical symmetry checking
instead of ensemble sampling. Inspired by Chomsky's Universal Grammar.

Reduces API calls from O(n×m) to O(1) while maintaining detection quality.

Features:
- Single API call detection (30× speedup)
- Cloud & Local model support (OpenAI, Ollama, etc.)
- Production monitoring & metrics
- Batch processing optimization
- Docker deployment ready
"""

# Core components
from .grammatical_mapper import GrammaticalMapper, StructurePattern
from .planner import GlassPlanner, GlassItem, GlassMetrics

# Batch processing
try:
    from .batch_optimizer import OptimizedBatchPlanner, BatchStats, optimize_batch_size
    _batch_available = True
except ImportError:
    _batch_available = False

# Monitoring
try:
    from .monitoring import MonitoredPlanner, MetricsCollector, setup_logging
    _monitoring_available = True
except ImportError:
    _monitoring_available = False

# Caching
try:
    from .cache import CachedGrammaticalMapper, StructureCache
    _cache_available = True
except ImportError:
    _cache_available = False

# Visualization
try:
    from .visualizer import print_single_result, print_batch_results
    _visualizer_available = True
except ImportError:
    _visualizer_available = False


__version__ = "1.0.0"
__author__ = "HallBayes Team"

__all__ = [
    # Core
    "GrammaticalMapper",
    "StructurePattern",
    "GlassPlanner",
    "GlassItem",
    "GlassMetrics",
]

# Add optional exports if available
if _batch_available:
    __all__.extend(["OptimizedBatchPlanner", "BatchStats", "optimize_batch_size"])

if _monitoring_available:
    __all__.extend(["MonitoredPlanner", "MetricsCollector", "setup_logging"])

if _cache_available:
    __all__.extend(["CachedGrammaticalMapper", "StructureCache"])

if _visualizer_available:
    __all__.extend(["print_single_result", "print_batch_results"])
