"""
Glass Visualizer - Beautiful Result Display
============================================

Utilities for pretty-printing Glass evaluation results.
"""

from typing import List, Optional, Union
from dataclasses import dataclass


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @classmethod
    def disable(cls):
        """Disable colors (for non-TTY output)"""
        cls.HEADER = ''
        cls.OKBLUE = ''
        cls.OKCYAN = ''
        cls.OKGREEN = ''
        cls.WARNING = ''
        cls.FAIL = ''
        cls.ENDC = ''
        cls.BOLD = ''
        cls.UNDERLINE = ''


def format_decision(decision: bool, colored: bool = True) -> str:
    """Format decision (ANSWER/REFUSE) with color"""
    if decision:
        symbol = "✓"
        text = "ANSWER"
        color = Colors.OKGREEN if colored else ""
    else:
        symbol = "✗"
        text = "REFUSE"
        color = Colors.FAIL if colored else ""

    end = Colors.ENDC if colored else ""
    return f"{color}{symbol} {text}{end}"


def format_score(score: float, threshold: float = 0.6, colored: bool = True) -> str:
    """Format symmetry score with color based on threshold"""
    if score >= threshold:
        color = Colors.OKGREEN if colored else ""
    elif score >= threshold * 0.8:
        color = Colors.WARNING if colored else ""
    else:
        color = Colors.FAIL if colored else ""

    end = Colors.ENDC if colored else ""
    return f"{color}{score:.3f}{end}"


def format_bar(value: float, width: int = 20, char: str = "█") -> str:
    """Create a text-based bar chart"""
    filled = int(value * width)
    empty = width - filled
    return f"{char * filled}{'·' * empty}"


def print_header(title: str, width: int = 60):
    """Print a formatted header"""
    print("\n" + "="*width)
    print(f"{Colors.BOLD}{title}{Colors.ENDC}".center(width + len(Colors.BOLD) + len(Colors.ENDC)))
    print("="*width)


def print_single_result(
    prompt: str,
    metrics,
    item_num: Optional[int] = None,
    show_details: bool = True,
    colored: bool = True,
):
    """
    Print result for a single item in a beautiful format.

    Args:
        prompt: The input prompt
        metrics: GlassMetrics or ItemMetrics object
        item_num: Optional item number
        show_details: Show detailed metrics
        colored: Use ANSI colors
    """
    prefix = f"[{item_num}] " if item_num is not None else ""

    print(f"\n{Colors.BOLD}{prefix}Query:{Colors.ENDC} {prompt}")

    # Decision
    decision_str = format_decision(metrics.decision_answer, colored=colored)
    print(f"{Colors.BOLD}Decision:{Colors.ENDC} {decision_str}")

    if show_details:
        # Symmetry (if available)
        if hasattr(metrics, 'symmetry_score'):
            sym_str = format_score(metrics.symmetry_score, colored=colored)
            bar = format_bar(metrics.symmetry_score)
            print(f"{Colors.BOLD}Symmetry:{Colors.ENDC} {sym_str} {Colors.OKCYAN}{bar}{Colors.ENDC}")

        # EDFL metrics
        print(f"{Colors.BOLD}Metrics:{Colors.ENDC}")
        print(f"  ISR:       {metrics.isr:>7.2f}")
        print(f"  RoH bound: {metrics.roh_bound:>7.3f}")
        print(f"  Δ̄:         {metrics.delta_bar:>7.3f} nats")
        print(f"  B2T:       {metrics.b2t:>7.3f} nats")


def print_batch_results(
    prompts: List[str],
    metrics_list: List,
    show_details: bool = False,
    colored: bool = True,
):
    """Print results for multiple items in a table format"""

    print_header("BATCH RESULTS")

    # Header row
    print(f"\n{'#':<4} {'Decision':<10} {'Symmetry':<10} {'ISR':<8} {'Prompt'}")
    print("-" * 80)

    # Data rows
    for i, (prompt, metrics) in enumerate(zip(prompts, metrics_list), 1):
        decision = "✓ ANSWER" if metrics.decision_answer else "✗ REFUSE"
        decision_colored = format_decision(metrics.decision_answer, colored=colored)

        if hasattr(metrics, 'symmetry_score'):
            sym = format_score(metrics.symmetry_score, colored=colored)
        else:
            sym = "N/A"

        isr = f"{metrics.isr:.1f}"
        prompt_short = prompt[:50] + "..." if len(prompt) > 50 else prompt

        # Uncolored for alignment
        print(f"{i:<4} {decision:<10} {metrics.symmetry_score if hasattr(metrics, 'symmetry_score') else 'N/A':<10.2f} {isr:<8} {prompt_short}")

    # Summary
    total = len(metrics_list)
    answered = sum(1 for m in metrics_list if m.decision_answer)
    refused = total - answered

    print(f"\n{Colors.BOLD}Summary:{Colors.ENDC}")
    print(f"  Total:    {total}")
    print(f"  Answered: {Colors.OKGREEN}{answered}{Colors.ENDC} ({answered/total*100:.1f}%)")
    print(f"  Refused:  {Colors.FAIL}{refused}{Colors.ENDC} ({refused/total*100:.1f}%)")


def print_comparison(
    prompt: str,
    glass_metrics,
    original_metrics,
    colored: bool = True,
):
    """Compare Glass vs Original results side-by-side"""

    print_header("GLASS vs ORIGINAL COMPARISON")

    print(f"\n{Colors.BOLD}Prompt:{Colors.ENDC} {prompt}\n")

    # Decisions
    glass_decision = format_decision(glass_metrics.decision_answer, colored=colored)
    orig_decision = format_decision(original_metrics.decision_answer, colored=colored)

    print(f"{Colors.BOLD}Glass:{Colors.ENDC}    {glass_decision}")
    print(f"{Colors.BOLD}Original:{Colors.ENDC} {orig_decision}")

    # Agreement
    agree = glass_metrics.decision_answer == original_metrics.decision_answer
    agree_str = f"{Colors.OKGREEN}✓ AGREE{Colors.ENDC}" if agree else f"{Colors.WARNING}⚠ DIFFER{Colors.ENDC}"
    print(f"{Colors.BOLD}Agreement:{Colors.ENDC} {agree_str}")

    # Metrics comparison
    print(f"\n{Colors.BOLD}Metrics Comparison:{Colors.ENDC}")
    print(f"{'Metric':<15} {'Glass':<12} {'Original':<12}")
    print("-" * 40)

    if hasattr(glass_metrics, 'symmetry_score'):
        print(f"{'Symmetry':<15} {glass_metrics.symmetry_score:<12.3f} {'N/A':<12}")
    print(f"{'ISR':<15} {glass_metrics.isr:<12.2f} {original_metrics.isr:<12.2f}")
    print(f"{'RoH bound':<15} {glass_metrics.roh_bound:<12.3f} {original_metrics.roh_bound:<12.3f}")
    print(f"{'Δ̄ (nats)':<15} {glass_metrics.delta_bar:<12.3f} {original_metrics.delta_bar:<12.3f}")


def print_performance_summary(
    glass_time: float,
    original_time: float,
    glass_calls: int,
    original_calls: int,
    cost_per_call: float = 0.0001,
):
    """Print performance comparison summary"""

    print_header("PERFORMANCE SUMMARY")

    speedup = original_time / glass_time if glass_time > 0 else float('inf')
    call_reduction = original_calls / glass_calls if glass_calls > 0 else float('inf')

    glass_cost = glass_calls * cost_per_call
    orig_cost = original_calls * cost_per_call
    cost_savings = orig_cost / glass_cost if glass_cost > 0 else float('inf')

    print(f"\n{Colors.BOLD}⏱️  Time:{Colors.ENDC}")
    print(f"  Original: {original_time:>8.2f}s")
    print(f"  Glass:    {glass_time:>8.2f}s")
    print(f"  {Colors.OKGREEN}Speedup:  {speedup:>8.1f}×{Colors.ENDC}")

    print(f"\n{Colors.BOLD}📞 API Calls:{Colors.ENDC}")
    print(f"  Original: {original_calls:>8}")
    print(f"  Glass:    {glass_calls:>8}")
    print(f"  {Colors.OKGREEN}Reduction: {call_reduction:>7.1f}×{Colors.ENDC}")

    print(f"\n{Colors.BOLD}💰 Cost (estimated):{Colors.ENDC}")
    print(f"  Original: ${orig_cost:>8.4f}")
    print(f"  Glass:    ${glass_cost:>8.4f}")
    print(f"  {Colors.OKGREEN}Savings:  {cost_savings:>7.1f}×{Colors.ENDC}")


def create_markdown_report(
    prompts: List[str],
    metrics_list: List,
    title: str = "Glass Evaluation Report",
) -> str:
    """Create a markdown report of results"""

    lines = [
        f"# {title}",
        "",
        f"**Evaluated:** {len(prompts)} items",
        "",
        "## Results",
        "",
        "| # | Decision | Symmetry | ISR | RoH | Prompt |",
        "|---|----------|----------|-----|-----|--------|",
    ]

    for i, (prompt, m) in enumerate(zip(prompts, metrics_list), 1):
        decision = "✓ ANSWER" if m.decision_answer else "✗ REFUSE"
        sym = f"{m.symmetry_score:.3f}" if hasattr(m, 'symmetry_score') else "N/A"
        isr = f"{m.isr:.2f}"
        roh = f"{m.roh_bound:.3f}"
        prompt_short = prompt[:50] + "..." if len(prompt) > 50 else prompt

        lines.append(f"| {i} | {decision} | {sym} | {isr} | {roh} | {prompt_short} |")

    # Summary
    total = len(metrics_list)
    answered = sum(1 for m in metrics_list if m.decision_answer)

    lines.extend([
        "",
        "## Summary",
        "",
        f"- **Total items:** {total}",
        f"- **Answered:** {answered} ({answered/total*100:.1f}%)",
        f"- **Refused:** {total - answered} ({(total-answered)/total*100:.1f}%)",
        "",
    ])

    return "\n".join(lines)


# Quick utility functions

def quick_print(metrics, prompt: Optional[str] = None):
    """Quick one-liner to print result"""
    if prompt:
        print(f"Query: {prompt}")
    decision = "✓ ANSWER" if metrics.decision_answer else "✗ REFUSE"
    sym = f"Sym={metrics.symmetry_score:.2f}" if hasattr(metrics, 'symmetry_score') else ""
    print(f"{decision} | ISR={metrics.isr:.1f} | RoH={metrics.roh_bound:.3f} | {sym}")


def export_json(metrics_list: List, filepath: str):
    """Export results to JSON"""
    import json
    from dataclasses import asdict

    data = [asdict(m) for m in metrics_list]

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    print(f"Results exported to {filepath}")


# Example usage
if __name__ == "__main__":
    # Demo
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from glass import GlassPlanner, GlassItem

    class MockBackend:
        def chat_create(self, messages, **kwargs):
            class MockResponse:
                class Choice:
                    class Message:
                        content = "Paris is the capital of France."
                    message = Message()
                choices = [Choice()]
            return MockResponse()

    backend = MockBackend()
    planner = GlassPlanner(backend)

    prompt = "What is the capital of France?"
    item = GlassItem(prompt=prompt)
    metrics = planner.evaluate_item(0, item)

    print_single_result(prompt, metrics, item_num=1, show_details=True, colored=True)
