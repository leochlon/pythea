"""
Output Entropy Confidence Scoring for Strawberry.

Extracts calibrated confidence from model logprobs using DUAL ENTROPY:
- Factual entropy: Uncertainty about facts (names, numbers, technical terms)
- Expressive entropy: Uncertainty about phrasing (stylistic choices)

Key insight: High expressive entropy with low factual entropy indicates
an expert explanation, not hallucination. We flag on FACTUAL entropy.

Based on cross-architecture validation (GPT-4, Llama, Qwen, Mistral):
- Lower entropy → correct responses
- Higher entropy → incorrect responses
- Effect replicates with d > 2.0 in frontier models

This complements Strawberry's budget-gap analysis:
- Budget gap: "Does cited evidence support this claim?" (post-hoc)
- Entropy: "How confident was the model?" (real-time)

Reference: Watson & Claude (2026), "Output Entropy Predicts Errors Across LLM Architectures"
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("entropy_confidence")

# =============================================================================
# Factual Token Classification
# =============================================================================

# Technical/scientific terms that carry factual weight
FACTUAL_TERMS: set[str] = {
    # Scientific concepts
    "rayleigh", "scattering", "wavelength", "wavelengths", "atmosphere",
    "phenomenon", "molecules", "photon", "photons", "spectrum", "frequency",
    "quantum", "electron", "proton", "neutron", "atom", "atomic",
    "entropy", "thermodynamic", "thermodynamics", "energy", "momentum",
    # Colors (factual in "what color" contexts)
    "blue", "red", "green", "yellow", "violet", "orange", "purple",
    "white", "black", "brown", "pink", "cyan", "magenta",
    # Common answer words
    "yes", "no", "true", "false", "correct", "incorrect",
}

# Named entity types that indicate factual content
# Excludes DATE/TIME/CARDINAL which cause false positives
FACTUAL_ENTITY_TYPES: set[str] = {
    "PERSON", "ORG", "GPE", "LOC", "FAC", "PRODUCT",
    "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "NORP",
}

# Stopwords for expressive filtering
STOPWORDS: set[str] = {
    # Articles
    "a", "an", "the",
    # Conjunctions
    "and", "or", "but", "so", "yet", "for", "nor",
    # Prepositions
    "in", "on", "at", "to", "of", "by", "with", "from", "as", "into",
    # Pronouns
    "i", "you", "he", "she", "it", "we", "they", "this", "that",
    # Common verbs
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    # Punctuation tokens
    ".", ",", "!", "?", ":", ";", "-", "'", '"', "(", ")", "[", "]", "{", "}",
}

PUNCTUATION_PATTERN = re.compile(r'^[^\w\s]+$')


# =============================================================================
# Data Classes
# =============================================================================

class ConfidenceLevel(str, Enum):
    """Discrete confidence levels based on entropy thresholds."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


@dataclass
class DualEntropyMetrics:
    """
    Dual entropy metrics separating factual from expressive uncertainty.

    Factual entropy: Uncertainty about facts (dangerous if high)
    Expressive entropy: Uncertainty about phrasing (often benign)
    """
    # Factual metrics (primary for hallucination detection)
    factual_p95: float = 0.0
    factual_max: float = 0.0
    factual_first_mention_p95: float = 0.0  # First occurrence of each term
    n_factual_tokens: int = 0

    # Expressive metrics (informational)
    expressive_p95: float = 0.0
    expressive_max: float = 0.0
    n_expressive_tokens: int = 0

    # Overall metrics (for backwards compatibility)
    max_entropy: float = 0.0
    mean_entropy: float = 0.0
    p95_entropy: float = 0.0
    n_tokens: int = 0

    # Confidence scoring (based on factual entropy)
    confidence_score: float = 1.0
    confidence_level: ConfidenceLevel = ConfidenceLevel.HIGH

    def to_dict(self) -> Dict[str, Any]:
        return {
            # Dual entropy (new)
            "factual_p95": round(self.factual_p95, 4),
            "factual_first_mention_p95": round(self.factual_first_mention_p95, 4),
            "factual_max": round(self.factual_max, 4),
            "n_factual_tokens": self.n_factual_tokens,
            "expressive_p95": round(self.expressive_p95, 4),
            "expressive_max": round(self.expressive_max, 4),
            "n_expressive_tokens": self.n_expressive_tokens,
            # Overall (backwards compatible)
            "max_entropy": round(self.max_entropy, 4),
            "mean_entropy": round(self.mean_entropy, 4),
            "p95_entropy": round(self.p95_entropy, 4),
            "n_tokens": self.n_tokens,
            # Confidence
            "confidence_score": round(self.confidence_score, 4),
            "confidence_level": self.confidence_level.value,
        }


# Backwards compatibility alias
EntropyMetrics = DualEntropyMetrics


# =============================================================================
# Model Thresholds
# =============================================================================

ENTROPY_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "gpt-4": {"high": 0.30, "medium": 0.80, "low": 1.50},
    "gpt-4-turbo": {"high": 0.30, "medium": 0.80, "low": 1.50},
    "gpt-4o": {"high": 0.35, "medium": 0.90, "low": 1.60},
    "gpt-4o-mini": {"high": 0.40, "medium": 1.00, "low": 1.80},
    "gpt-3.5-turbo": {"high": 0.50, "medium": 1.20, "low": 2.00},
    "default": {"high": 0.40, "medium": 1.00, "low": 2.00},
}


# =============================================================================
# spaCy Integration (Lazy Loading)
# =============================================================================

_nlp = None

def _get_nlp():
    """Lazy load spaCy model."""
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
            logger.debug("Loaded spaCy model for dual entropy analysis")
        except ImportError:
            logger.warning("spaCy not installed - falling back to term-based classification")
            _nlp = False
        except OSError:
            logger.warning("spaCy model not found - run: python -m spacy download en_core_web_sm")
            _nlp = False
    return _nlp if _nlp else None


def is_factual_token(token_text: str, doc: Any = None) -> Tuple[bool, Optional[str]]:
    """
    Determine if a token carries factual content.

    Returns (is_factual, reason) tuple.
    """
    token_clean = token_text.strip().lower()

    # Skip very short tokens
    if len(token_clean) <= 1:
        return False, None

    # Check technical/answer terms
    if token_clean in FACTUAL_TERMS:
        return True, "TERM"

    # Check if it's a number
    if token_clean.replace(".", "").replace(",", "").replace("-", "").isdigit():
        return True, "NUMBER"

    # Check named entities via spaCy (if available)
    if doc is not None:
        for ent in doc.ents:
            if ent.label_ in FACTUAL_ENTITY_TYPES:
                if token_clean in ent.text.lower():
                    return True, f"ENTITY:{ent.label_}"

    return False, None


def is_stopword_token(token: str) -> bool:
    """Check if a token should be filtered from entropy calculation."""
    token_lower = token.lower().strip()

    if token_lower in STOPWORDS:
        return True

    if PUNCTUATION_PATTERN.match(token_lower):
        return True

    if len(token_lower) <= 1 and not token_lower.isalnum():
        return True

    return False


# =============================================================================
# Core Functions
# =============================================================================

def percentile(values: List[float], p: float) -> float:
    """Compute the p-th percentile of values (0-100)."""
    if not values:
        return 0.0

    sorted_vals = sorted(values)
    n = len(sorted_vals)

    if n == 1:
        return sorted_vals[0]

    k = (p / 100.0) * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        return sorted_vals[int(k)]

    return sorted_vals[int(f)] * (c - k) + sorted_vals[int(c)] * (k - f)


def length_adjusted_threshold(base_threshold: float, n_tokens: int) -> float:
    """Adjust threshold based on response length."""
    if n_tokens <= 0:
        return base_threshold
    return base_threshold * (1.0 + 0.05 * math.log(n_tokens + 1))


def compute_token_entropy(logprobs: List[float]) -> float:
    """Compute Shannon entropy in bits from log probabilities."""
    if not logprobs:
        return 0.0

    probs = [math.exp(lp) for lp in logprobs]
    total = sum(probs)
    if total <= 0:
        return 0.0
    probs = [p / total for p in probs]

    return -sum(p * math.log2(p + 1e-10) for p in probs if p > 0)


def extract_dual_entropy_from_openai_response(
    response: Any,
    answer_text: str = "",
    model_name: str = "default",
) -> DualEntropyMetrics:
    """
    Extract dual entropy metrics (factual vs expressive) from OpenAI response.

    Uses spaCy NER when available to classify tokens as factual or expressive.
    Factual entropy is used for confidence scoring (hallucination detection).
    Expressive entropy is informational (high values often indicate expertise).

    Args:
        response: OpenAI ChatCompletion response with logprobs
        answer_text: The generated answer text (for NER parsing)
        model_name: Model name for threshold calibration

    Returns:
        DualEntropyMetrics with separated factual/expressive entropy
    """
    # Parse answer with spaCy if available
    nlp = _get_nlp()
    doc = nlp(answer_text) if nlp and answer_text else None

    # Collect token-level data
    all_entropies: List[float] = []
    factual_entropies: List[float] = []
    expressive_entropies: List[float] = []

    # Track first mentions for factual terms
    seen_factual_terms: set[str] = set()
    first_mention_entropies: List[float] = []

    try:
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            if hasattr(choice, 'logprobs') and choice.logprobs:
                content = choice.logprobs.content
                if content:
                    for token_info in content:
                        if hasattr(token_info, 'top_logprobs') and token_info.top_logprobs:
                            token_text = getattr(token_info, 'token', '')
                            lps = [lp.logprob for lp in token_info.top_logprobs]
                            entropy = compute_token_entropy(lps)

                            all_entropies.append(entropy)

                            # Classify token
                            is_fact, reason = is_factual_token(token_text, doc)

                            if is_fact:
                                factual_entropies.append(entropy)

                                # Track first mentions
                                term_key = token_text.strip().lower()
                                if term_key not in seen_factual_terms:
                                    seen_factual_terms.add(term_key)
                                    first_mention_entropies.append(entropy)
                            else:
                                # Skip stopwords for expressive
                                if not is_stopword_token(token_text):
                                    expressive_entropies.append(entropy)

    except (AttributeError, TypeError, IndexError) as e:
        logger.debug(f"Failed to extract logprobs: {e}")

    # Compute metrics
    n_total = len(all_entropies)

    if not all_entropies:
        return DualEntropyMetrics(confidence_level=ConfidenceLevel.HIGH)

    # Factual entropy (primary signal)
    fact_p95 = percentile(factual_entropies, 95) if factual_entropies else 0.0
    fact_max = max(factual_entropies) if factual_entropies else 0.0
    fact_first_p95 = percentile(first_mention_entropies, 95) if first_mention_entropies else 0.0

    # Expressive entropy (informational)
    expr_p95 = percentile(expressive_entropies, 95) if expressive_entropies else 0.0
    expr_max = max(expressive_entropies) if expressive_entropies else 0.0

    # Overall metrics
    overall_max = max(all_entropies)
    overall_mean = sum(all_entropies) / n_total
    overall_p95 = percentile(all_entropies, 95)

    # Confidence based on FACTUAL first-mention entropy (most reliable signal)
    # Falls back to overall if no factual tokens detected
    primary_entropy = fact_first_p95 if first_mention_entropies else overall_p95
    confidence_score = 1.0 / (1.0 + primary_entropy)
    confidence_level = get_confidence_level(primary_entropy, model_name, n_tokens=n_total)

    return DualEntropyMetrics(
        # Factual
        factual_p95=fact_p95,
        factual_max=fact_max,
        factual_first_mention_p95=fact_first_p95,
        n_factual_tokens=len(factual_entropies),
        # Expressive
        expressive_p95=expr_p95,
        expressive_max=expr_max,
        n_expressive_tokens=len(expressive_entropies),
        # Overall
        max_entropy=overall_max,
        mean_entropy=overall_mean,
        p95_entropy=overall_p95,
        n_tokens=n_total,
        # Confidence
        confidence_score=confidence_score,
        confidence_level=confidence_level,
    )


# Backwards compatibility alias
def extract_entropy_from_openai_response(
    response: Any,
    model_name: str = "default",
    filter_stopwords: bool = True,
) -> DualEntropyMetrics:
    """Extract entropy metrics (backwards compatible wrapper)."""
    answer = ""
    try:
        if hasattr(response, 'choices') and response.choices:
            message = response.choices[0].message
            if message and message.content:
                answer = message.content
    except Exception:
        pass

    return extract_dual_entropy_from_openai_response(response, answer, model_name)


def extract_entropy_from_logprobs_list(logprobs_list: List[List[float]]) -> DualEntropyMetrics:
    """Extract entropy from raw list of logprobs per token."""
    entropies = [compute_token_entropy(lps) for lps in logprobs_list if lps]

    if not entropies:
        return DualEntropyMetrics()

    return DualEntropyMetrics(
        max_entropy=max(entropies),
        mean_entropy=sum(entropies) / len(entropies),
        p95_entropy=percentile(entropies, 95),
        n_tokens=len(entropies),
        confidence_score=1.0 / (1.0 + percentile(entropies, 95)),
        confidence_level=get_confidence_level(percentile(entropies, 95)),
    )


def get_confidence_level(
    entropy: float,
    model_name: str = "default",
    n_tokens: int = 0,
) -> ConfidenceLevel:
    """Classify confidence level based on entropy threshold."""
    model_key = _normalize_model_name(model_name)
    base_thresholds = ENTROPY_THRESHOLDS.get(model_key, ENTROPY_THRESHOLDS["default"])

    if n_tokens > 0:
        thresholds = {
            k: length_adjusted_threshold(v, n_tokens)
            for k, v in base_thresholds.items()
        }
    else:
        thresholds = base_thresholds

    if entropy < thresholds["high"]:
        return ConfidenceLevel.HIGH
    elif entropy < thresholds["medium"]:
        return ConfidenceLevel.MEDIUM
    elif entropy < thresholds["low"]:
        return ConfidenceLevel.LOW
    else:
        return ConfidenceLevel.VERY_LOW


def _normalize_model_name(model_name: str) -> str:
    """Normalize model name for threshold lookup."""
    name = model_name.lower()
    if "gpt-4o-mini" in name:
        return "gpt-4o-mini"
    if "gpt-4o" in name:
        return "gpt-4o"
    if "gpt-4-turbo" in name or "gpt-4-turbo-preview" in name:
        return "gpt-4-turbo"
    if "gpt-4" in name:
        return "gpt-4"
    if "gpt-3.5" in name:
        return "gpt-3.5-turbo"
    return "default"


def should_flag(
    metrics: DualEntropyMetrics,
    task_type: str = "factual"
) -> Tuple[bool, str]:
    """
    Determine if response should be flagged based on FACTUAL entropy.

    Uses first-mention factual entropy as primary signal.
    High expressive entropy alone does NOT trigger flagging.
    """
    level = metrics.confidence_level

    if task_type == "factual":
        flag_levels = {ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW}
    elif task_type == "math":
        flag_levels = {ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW}
    elif task_type == "code":
        flag_levels = {ConfidenceLevel.VERY_LOW}
    elif task_type == "creative":
        flag_levels = set()
    else:
        flag_levels = {ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW}

    if level in flag_levels:
        reason = (
            f"Low factual confidence: {level.value} "
            f"(factual_first_p95={metrics.factual_first_mention_p95:.3f}, "
            f"factual_tokens={metrics.n_factual_tokens}, "
            f"expressive_p95={metrics.expressive_p95:.3f})"
        )
        return True, reason

    return False, ""


# =============================================================================
# Main API
# =============================================================================

async def check_entropy_confidence_async(
    query: str,
    model: str = "gpt-4o-mini",
    task_type: str = "factual",
    temperature: float = 0.7,
    max_tokens: int = 100,
    timeout_s: float = 30.0,
) -> Dict[str, Any]:
    """
    Generate an answer and compute dual entropy-based confidence.

    Uses FACTUAL entropy (first-mention) for confidence scoring.
    Reports both factual and expressive entropy for transparency.

    Args:
        query: The question or prompt to answer.
        model: OpenAI model to use (must support logprobs).
        task_type: One of "factual", "math", "code", "creative", "general".
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        timeout_s: Request timeout in seconds.

    Returns:
        Dict with answer, dual confidence metrics, and flagging status.
    """
    import os

    if not query or not query.strip():
        raise ValueError("query cannot be empty")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Entropy confidence requires OpenAI API for logprobs access."
        )

    try:
        from openai import AsyncOpenAI, APIError, APIConnectionError, RateLimitError
    except ImportError:
        raise ImportError("openai package required: pip install openai")

    client = AsyncOpenAI(api_key=api_key, timeout=timeout_s)

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": query}],
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=5,
        )
    except (APIError, APIConnectionError, RateLimitError) as e:
        return {
            "answer": "",
            "confidence": {
                "score": 0.0,
                "level": "error",
                "factual_p95": 0.0,
                "expressive_p95": 0.0,
            },
            "flagged": True,
            "flag_reason": f"API error: {type(e).__name__}: {e}",
            "should_verify": True,
            "model": model,
            "task_type": task_type,
            "error": str(e),
        }

    # Extract answer
    answer = ""
    if response.choices:
        message = response.choices[0].message
        if message and message.content:
            answer = message.content

    # Extract dual entropy
    metrics = extract_dual_entropy_from_openai_response(response, answer, model)
    flagged, reason = should_flag(metrics, task_type)

    return {
        "answer": answer,
        "confidence": {
            "score": metrics.confidence_score,
            "level": metrics.confidence_level.value,
            # Dual entropy (new)
            "factual_p95": metrics.factual_p95,
            "factual_first_mention_p95": metrics.factual_first_mention_p95,
            "n_factual_tokens": metrics.n_factual_tokens,
            "expressive_p95": metrics.expressive_p95,
            "n_expressive_tokens": metrics.n_expressive_tokens,
            # Overall (backwards compatible)
            "max_entropy": metrics.max_entropy,
            "mean_entropy": metrics.mean_entropy,
            "p95_entropy": metrics.p95_entropy,
            "n_tokens": metrics.n_tokens,
        },
        "flagged": flagged,
        "flag_reason": reason if flagged else None,
        "should_verify": flagged or metrics.confidence_level in {
            ConfidenceLevel.MEDIUM, ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW
        },
        "model": model,
        "task_type": task_type,
    }


def check_entropy_confidence_sync(
    query: str,
    model: str = "gpt-4o-mini",
    task_type: str = "factual",
    temperature: float = 0.7,
    max_tokens: int = 100,
    timeout_s: float = 30.0,
) -> Dict[str, Any]:
    """Synchronous version of check_entropy_confidence_async."""
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                asyncio.run,
                check_entropy_confidence_async(
                    query=query,
                    model=model,
                    task_type=task_type,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout_s=timeout_s,
                )
            )
            return future.result(timeout=timeout_s + 5)
    else:
        return asyncio.run(check_entropy_confidence_async(
            query=query,
            model=model,
            task_type=task_type,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        ))
