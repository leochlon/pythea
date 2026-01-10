"""
Synthetic prompt generators for proximity-limited key-value binding tasks.

These tasks are *not* meant to replicate any specific paper's dataset exactly.
They are meant to let others exercise the theory objects (error, MI, pseudo-prior) on any API model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import random


@dataclass
class BindingItem:
    """
    A single binding instance.

    - prompt: the text shown to the model
    - choices: list of candidate strings (values)
    - correct: the correct choice string
    - meta: auxiliary metadata (distance, etc.)
    """
    prompt: str
    choices: List[str]
    correct: str
    meta: dict


DEFAULT_VALUES = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
    "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey", "xray",
    "yankee", "zulu"
]


def make_filler(n_tokens: int, seed: int = 0) -> str:
    """
    Make a filler paragraph whose length scales with n_tokens (approx).
    We use short repeated tokens to approximate a controlled distance.
    """
    rng = random.Random(seed)
    words = []
    vocab = ["lorem", "ipsum", "dolor", "sit", "amet", "filler", "token", "noise"]
    for _ in range(n_tokens):
        words.append(rng.choice(vocab))
    return " ".join(words)


def generate_proximity_item(
    *,
    key: str = "KEY",
    values: Sequence[str] = DEFAULT_VALUES,
    M: int = 10,
    distance_tokens: int = 256,
    seed: int = 0,
    query_rule: str = "FIRST",  # or "LAST"
    include_other: bool = True
) -> BindingItem:
    """
    Generate a single instance where the key is assigned twice:
      - an early assignment (true or false depending on query_rule)
      - a late assignment (opposite)

    The task queries either the FIRST assignment or the LAST assignment.
    Long-distance binding corresponds to query_rule="FIRST" with large distance_tokens
    (the correct value is far away, and recency encourages the last value).

    include_other adds an "OTHER" option for stage-2A style abstention.
    """
    rng = random.Random(seed)
    values = list(values)
    if len(values) < M + 2:
        raise ValueError("need more candidate values; increase values list or reduce M")

    # Candidate set and two distinct values for the key.
    candidates = rng.sample(values, M)
    v_first, v_last = rng.sample(candidates, 2)

    if query_rule.upper() == "FIRST":
        correct = v_first
    elif query_rule.upper() == "LAST":
        correct = v_last
    else:
        raise ValueError("query_rule must be 'FIRST' or 'LAST'")

    filler = make_filler(distance_tokens, seed=seed + 1337)

    prompt = f"""
You will see assignments of the form {key} = value.

(1) {key} = {v_first}

[FILLER BEGIN]
{filler}
[FILLER END]

(2) {key} = {v_last}

Question: What is the value of {key} according to the {query_rule.upper()} assignment?
Answer with exactly one value from the provided options (or OTHER if you cannot determine).
""".strip()

    choices = list(candidates)
    if include_other and "OTHER" not in choices:
        choices.append("OTHER")

    return BindingItem(
        prompt=prompt,
        choices=choices,
        correct=correct,
        meta={
            "key": key,
            "query_rule": query_rule.upper(),
            "distance_tokens": int(distance_tokens),
            "v_first": v_first,
            "v_last": v_last,
            "M": int(M),
        },
    )


def generate_dataset(
    *,
    n: int = 200,
    distance_tokens: int = 256,
    M: int = 10,
    query_rule: str = "FIRST",
    seed: int = 0
) -> List[BindingItem]:
    """Generate a list of BindingItems."""
    items: List[BindingItem] = []
    for i in range(n):
        items.append(
            generate_proximity_item(
                M=M,
                distance_tokens=distance_tokens,
                seed=seed + i,
                query_rule=query_rule,
            )
        )
    return items


def make_null_item(item: BindingItem, *, null_mode: str = "SCRUB_FIRST") -> BindingItem:
    """
    Create a pseudo-prior / null variant of a binding item.

    null_mode:
      - SCRUB_FIRST: replace the first assignment value with an unknown token (removes the far evidence)
      - SCRUB_BOTH: replace both assignments (removes all evidence; measures default biases)
      - REMOVE_FILLER: remove filler (a crude checkpointing-style intervention on the prompt)
    """
    prompt = item.prompt
    meta = dict(item.meta)
    if null_mode.upper() == "SCRUB_FIRST":
        prompt = prompt.replace(f"(1) {meta['key']} = {meta['v_first']}", f"(1) {meta['key']} = [UNKNOWN]")
        meta["null_mode"] = "SCRUB_FIRST"
    elif null_mode.upper() == "SCRUB_BOTH":
        prompt = prompt.replace(f"(1) {meta['key']} = {meta['v_first']}", f"(1) {meta['key']} = [UNKNOWN]")
        prompt = prompt.replace(f"(2) {meta['key']} = {meta['v_last']}", f"(2) {meta['key']} = [UNKNOWN]")
        meta["null_mode"] = "SCRUB_BOTH"
    elif null_mode.upper() == "REMOVE_FILLER":
        # Delete filler block as a checkpointing-style edit
        start = prompt.find("[FILLER BEGIN]")
        end = prompt.find("[FILLER END]")
        if start != -1 and end != -1 and end > start:
            prompt = prompt[:start] + "[FILLER REMOVED]\n\n" + prompt[end + len("[FILLER END]") :]
        meta["null_mode"] = "REMOVE_FILLER"
    else:
        raise ValueError("unknown null_mode")

    return BindingItem(prompt=prompt, choices=item.choices, correct=item.correct, meta=meta)
