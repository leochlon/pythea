#!/usr/bin/env python3
"""
Integration tests for Glass module
===================================

Tests:
1. Import compatibility
2. Basic functionality
3. API compatibility with OpenAIPlanner
4. Structure extraction
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all modules import correctly"""
    print("🔍 Testing imports...")

    try:
        from glass import GlassPlanner, GlassItem, GrammaticalMapper, StructurePattern
        print("  ✓ glass imports OK")
    except ImportError as e:
        print(f"  ✗ glass import failed: {e}")
        return False

    try:
        from hallbayes import OpenAIBackend, OpenAIItem, OpenAIPlanner
        print("  ✓ hallbayes imports OK")
    except ImportError as e:
        print(f"  ✗ hallbayes import failed: {e}")
        return False

    return True


def test_grammatical_mapper():
    """Test grammatical structure extraction"""
    print("\n🧪 Testing GrammaticalMapper...")

    from glass import GrammaticalMapper

    mapper = GrammaticalMapper()

    # Test case 1: Basic entity extraction
    text1 = "Who won the 2019 Nobel Prize in Physics?"
    struct1 = mapper.extract_structure(text1)

    print(f"  Text: {text1}")
    print(f"  Entities: {struct1.entities}")
    print(f"  Temporal: {struct1.temporal_markers}")

    if "2019" in struct1.temporal_markers:
        print("  ✓ Temporal extraction OK")
    else:
        print("  ✗ Temporal extraction failed")
        return False

    # Test case 2: Symmetry detection
    text2a = "James Peebles won the 2019 Nobel Prize"
    text2b = "James Peebles won the 2020 Nobel Prize"

    struct2a = mapper.extract_structure(text2a)
    struct2b = mapper.extract_structure(text2b)

    score_match = struct2a.symmetry_score(struct2a)  # Should be 1.0
    score_diff = struct2a.symmetry_score(struct2b)   # Should be < 1.0

    print(f"\n  Symmetry (same): {score_match:.3f}")
    print(f"  Symmetry (diff year): {score_diff:.3f}")

    if score_match > 0.9 and score_diff < score_match:
        print("  ✓ Symmetry detection OK")
    else:
        print("  ✗ Symmetry detection failed")
        return False

    return True


def test_glass_planner_basic():
    """Test GlassPlanner basic functionality (without API)"""
    print("\n🧪 Testing GlassPlanner (basic)...")

    from glass import GlassPlanner, GlassItem

    # Mock backend
    class MockBackend:
        def chat_create(self, messages, **kwargs):
            class MockResponse:
                class Choice:
                    class Message:
                        content = "James Peebles won the 2019 Nobel Prize in Physics."
                    message = Message()
                choices = [Choice()]
            return MockResponse()

    backend = MockBackend()
    planner = GlassPlanner(backend, temperature=0.3)

    print("  ✓ GlassPlanner initialized")

    # Test single item
    item = GlassItem(prompt="Who won the 2019 Nobel Prize in Physics?")
    metrics = planner.evaluate_item(0, item)

    print(f"  Symmetry: {metrics.symmetry_score:.3f}")
    print(f"  Decision: {'ANSWER' if metrics.decision_answer else 'REFUSE'}")
    print(f"  ISR: {metrics.isr:.2f}")

    if hasattr(metrics, 'symmetry_score') and hasattr(metrics, 'decision_answer'):
        print("  ✓ GlassPlanner evaluation OK")
    else:
        print("  ✗ GlassPlanner evaluation failed")
        return False

    return True


def test_api_compatibility():
    """Test API compatibility with OpenAIPlanner"""
    print("\n🧪 Testing API compatibility...")

    from glass import GlassPlanner, GlassItem
    from hallbayes import OpenAIPlanner, OpenAIItem

    # Mock backend
    class MockBackend:
        def chat_create(self, messages, **kwargs):
            class MockResponse:
                class Choice:
                    class Message:
                        content = "Test response"
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

    # Both should have same interface
    glass_planner = GlassPlanner(backend)
    openai_planner = OpenAIPlanner(backend)

    # Check interface compatibility
    glass_methods = {'run', 'evaluate_item', 'aggregate'}
    openai_methods = {'run', 'evaluate_item', 'aggregate'}

    glass_has = {m for m in dir(glass_planner) if not m.startswith('_')}
    openai_has = {m for m in dir(openai_planner) if not m.startswith('_')}

    common_methods = glass_has & openai_has
    print(f"  Common methods: {common_methods}")

    if glass_methods.issubset(common_methods):
        print("  ✓ API compatibility OK")
    else:
        missing = glass_methods - common_methods
        print(f"  ✗ Missing methods: {missing}")
        return False

    return True


def test_metrics_compatibility():
    """Test that GlassMetrics has same fields as ItemMetrics"""
    print("\n🧪 Testing metrics compatibility...")

    from glass import GlassPlanner, GlassItem
    from hallbayes import ItemMetrics

    # Mock backend
    class MockBackend:
        def chat_create(self, messages, **kwargs):
            class MockResponse:
                class Choice:
                    class Message:
                        content = "Test"
                    message = Message()
                choices = [Choice()]
            return MockResponse()

    backend = MockBackend()
    planner = GlassPlanner(backend)
    item = GlassItem(prompt="Test")
    glass_metrics = planner.evaluate_item(0, item)

    # Check required fields
    required_fields = {
        'item_id', 'delta_bar', 'q_avg', 'q_conservative',
        'b2t', 'isr', 'roh_bound', 'decision_answer', 'rationale'
    }

    glass_fields = set(glass_metrics.__dataclass_fields__.keys())
    has_all = required_fields.issubset(glass_fields)

    print(f"  Required fields: {len(required_fields)}")
    print(f"  Glass has: {len(glass_fields & required_fields)}")

    if has_all:
        print("  ✓ Metrics compatibility OK")
    else:
        missing = required_fields - glass_fields
        print(f"  ✗ Missing fields: {missing}")
        return False

    return True


def main():
    """Run all tests"""
    print("="*60)
    print("GLASS INTEGRATION TESTS")
    print("="*60)

    tests = [
        ("Imports", test_imports),
        ("GrammaticalMapper", test_grammatical_mapper),
        ("GlassPlanner Basic", test_glass_planner_basic),
        ("API Compatibility", test_api_compatibility),
        ("Metrics Compatibility", test_metrics_compatibility),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  ✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\n  Total: {passed}/{total} tests passed")
    print("="*60)

    if passed == total:
        print("\n🎉 All tests passed! Glass is ready to use.\n")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review.\n")
        return 1


if __name__ == "__main__":
    exit(main())
