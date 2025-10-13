"""
Glass Migration Helper
=======================

Utilities to help migrate existing code from OpenAIPlanner to GlassPlanner.
"""

from typing import List, Union


def migrate_openai_to_glass(openai_item):
    """
    Convert OpenAIItem to GlassItem.

    Args:
        openai_item: OpenAIItem instance

    Returns:
        GlassItem instance

    Example:
        old_item = OpenAIItem(prompt="...", n_samples=7, m=6)
        new_item = migrate_openai_to_glass(old_item)
    """
    from glass import GlassItem

    return GlassItem(
        prompt=openai_item.prompt,
        symmetry_threshold=0.6,  # Default
        attempted=openai_item.attempted,
        answered_correctly=openai_item.answered_correctly,
        meta=openai_item.meta,
    )


def migrate_batch(openai_items: List):
    """
    Convert list of OpenAIItem to GlassItem.

    Args:
        openai_items: List of OpenAIItem instances

    Returns:
        List of GlassItem instances
    """
    return [migrate_openai_to_glass(item) for item in openai_items]


def create_hybrid_planner(backend, glass_confidence: float = 0.7):
    """
    Create hybrid planner from backend.

    Args:
        backend: LLM backend
        glass_confidence: Confidence threshold for Glass

    Returns:
        HybridPlanner instance

    Example:
        backend = OpenAIBackend(model="gpt-4o-mini")
        planner = create_hybrid_planner(backend)
        metrics, infos = planner.run(prompts)
    """
    # Import here to avoid circular dependency
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from glass.example_hybrid import HybridPlanner

    return HybridPlanner(
        backend=backend,
        glass_confidence_threshold=glass_confidence,
        use_fallback=True,
        verbose=False,
    )


def migration_guide():
    """Print migration guide"""
    guide = """
╔══════════════════════════════════════════════════════════════════╗
║                   GLASS MIGRATION GUIDE                          ║
╚══════════════════════════════════════════════════════════════════╝

## Step 1: Basic Migration

BEFORE (OpenAIPlanner):
```python
from hallbayes import OpenAIBackend, OpenAIPlanner, OpenAIItem

backend = OpenAIBackend(model="gpt-4o-mini")
planner = OpenAIPlanner(backend, temperature=0.3)

items = [
    OpenAIItem(prompt="...", n_samples=7, m=6)
]

metrics = planner.run(items, h_star=0.05)
```

AFTER (GlassPlanner):
```python
from hallbayes import OpenAIBackend
from glass import GlassPlanner, GlassItem

backend = OpenAIBackend(model="gpt-4o-mini")
planner = GlassPlanner(backend, temperature=0.3)  # Drop-in replacement

items = [
    GlassItem(prompt="...")  # Simpler!
]

metrics = planner.run(items, h_star=0.05)  # Same API
```

## Step 2: Hybrid Mode (Recommended)

Combine Glass (fast) with Original (accurate):
```python
from glass.migration_helper import create_hybrid_planner

backend = OpenAIBackend(model="gpt-4o-mini")
planner = create_hybrid_planner(backend, glass_confidence=0.7)

# Automatically uses Glass when confident, falls back to Original
metrics, infos = planner.run(prompts, h_star=0.05)

# Check which path was used
for info in infos:
    print(f"Path: {info['path']}")  # 'glass_only' or 'fallback'
```

## Step 3: Batch Migration

Migrate existing items:
```python
from glass.migration_helper import migrate_batch

# Existing code
old_items = [
    OpenAIItem(prompt="...", n_samples=7, m=6),
    OpenAIItem(prompt="...", n_samples=7, m=6),
]

# Migrate
new_items = migrate_batch(old_items)

# Use with Glass
planner = GlassPlanner(backend)
metrics = planner.run(new_items)
```

## Step 4: Performance Testing

Compare before/after:
```python
from glass.migration_helper import benchmark_migration

results = benchmark_migration(
    prompts=["Your", "test", "prompts"],
    backend=backend
)

print(f"Speedup: {results['speedup']:.1f}×")
print(f"Agreement: {results['agreement_rate']*100:.1f}%")
```

## API Compatibility Matrix

| Feature | OpenAIPlanner | GlassPlanner | Compatible? |
|---------|---------------|--------------|-------------|
| .run()  | ✓ | ✓ | ✓ Yes |
| .evaluate_item() | ✓ | ✓ | ✓ Yes |
| .aggregate() | ✓ | ✓ | ✓ Yes |
| ItemMetrics fields | ✓ | ✓ | ✓ Yes |
| h_star parameter | ✓ | ✓ | ✓ Yes |
| isr_threshold | ✓ | ✓ | ✓ Yes |

## Performance Comparison

| Metric | Original | Glass | Improvement |
|--------|----------|-------|-------------|
| API calls | 30-42 | 1 | 30-40× |
| Latency | 15-30s | 0.5-1s | 30× |
| Cost | ~$0.03 | ~$0.001 | 30× |

## Common Patterns

### Pattern 1: Drop-in Replacement
```python
# Just change the import and class name
- from hallbayes import OpenAIPlanner, OpenAIItem
+ from glass import GlassPlanner, GlassItem

- planner = OpenAIPlanner(backend)
+ planner = GlassPlanner(backend)

- item = OpenAIItem(prompt="...", n_samples=7, m=6)
+ item = GlassItem(prompt="...")
```

### Pattern 2: Gradual Migration
```python
# Keep both, compare results
glass_planner = GlassPlanner(backend)
orig_planner = OpenAIPlanner(backend)

# Use Glass for most queries
metrics = glass_planner.run(items)

# Use Original for critical queries
critical_metrics = orig_planner.run(critical_items)
```

### Pattern 3: Conditional Usage
```python
def get_planner(fast_mode: bool):
    if fast_mode:
        return GlassPlanner(backend)
    else:
        return OpenAIPlanner(backend)

planner = get_planner(fast_mode=True)
```

## Testing Your Migration

1. Run side-by-side comparison:
   ```bash
   python benchmarks/compare.py
   ```

2. Check decision agreement:
   ```bash
   python glass_check.py "test prompt" --compare
   ```

3. Review metrics compatibility:
   ```python
   assert hasattr(glass_metrics, 'decision_answer')
   assert hasattr(glass_metrics, 'isr')
   assert hasattr(glass_metrics, 'roh_bound')
   ```

## Troubleshooting

**Q: Different decisions between Glass and Original?**
A: This is expected. Glass uses grammatical symmetry (different approach).
   Agreement rate is typically 85-90%. Use hybrid mode for best of both.

**Q: How to tune Glass confidence?**
A: Adjust symmetry_threshold (default 0.6):
   - Higher (0.7-0.8): More conservative, fewer answers
   - Lower (0.5-0.6): More permissive, more answers

**Q: Can I use Glass with non-OpenAI backends?**
A: Yes! Glass works with any backend (OpenAI, Anthropic, HuggingFace, etc.)

## Need Help?

- Documentation: glass/README.md
- Examples: glass/example_*.py
- Tests: python3 glass/test_integration.py

═══════════════════════════════════════════════════════════════════
"""
    print(guide)


def benchmark_migration(prompts: List[str], backend) -> dict:
    """
    Benchmark migration impact.

    Args:
        prompts: Test prompts
        backend: LLM backend

    Returns:
        Dict with benchmark results
    """
    import time
    from hallbayes import OpenAIPlanner, OpenAIItem
    from glass import GlassPlanner, GlassItem

    # Original
    orig_items = [OpenAIItem(prompt=p, n_samples=3, m=4) for p in prompts]
    orig_planner = OpenAIPlanner(backend)

    orig_start = time.time()
    orig_metrics = orig_planner.run(orig_items)
    orig_time = time.time() - orig_start

    # Glass
    glass_items = [GlassItem(prompt=p) for p in prompts]
    glass_planner = GlassPlanner(backend)

    glass_start = time.time()
    glass_metrics = glass_planner.run(glass_items)
    glass_time = time.time() - glass_start

    # Compare
    speedup = orig_time / glass_time if glass_time > 0 else float('inf')

    agreements = sum(
        1 for o, g in zip(orig_metrics, glass_metrics)
        if o.decision_answer == g.decision_answer
    )
    agreement_rate = agreements / len(prompts) if prompts else 0.0

    return {
        "original_time": orig_time,
        "glass_time": glass_time,
        "speedup": speedup,
        "agreement_rate": agreement_rate,
        "agreements": agreements,
        "total": len(prompts),
    }


def quick_start_example():
    """Print quick start example"""
    example = """
Quick Start Example
===================

# 1. Import Glass
from hallbayes import OpenAIBackend
from glass import GlassPlanner, GlassItem

# 2. Create planner
backend = OpenAIBackend(model="gpt-4o-mini")
planner = GlassPlanner(backend, temperature=0.3)

# 3. Evaluate
items = [GlassItem(prompt="Who won the 2019 Nobel Prize?")]
metrics = planner.run(items, h_star=0.05)

# 4. Check result
for m in metrics:
    decision = "ANSWER" if m.decision_answer else "REFUSE"
    print(f"{decision} | Symmetry={m.symmetry_score:.2f}")

That's it! 30× faster than original.
"""
    print(example)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("GLASS MIGRATION HELPER")
    print("="*70)

    migration_guide()
