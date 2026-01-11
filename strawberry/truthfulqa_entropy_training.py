#!/usr/bin/env python3
"""
TruthfulQA Entropy Training (Phase 2a)

Downloads TruthfulQA, generates responses with logprobs,
extracts entropy metrics, and trains position-aware weights.

Usage:
    python truthfulqa_entropy_training.py --samples 100  # Quick test
    python truthfulqa_entropy_training.py --full         # All 817 samples

Requirements:
    pip install datasets openai scikit-learn numpy
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from openai import OpenAI
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np


@dataclass
class EntropyFeatures:
    """Entropy features extracted from a response."""
    p10_entropy: float  # 10th percentile
    p50_entropy: float  # median
    p90_entropy: float  # 90th percentile
    p95_entropy: float  # 95th percentile
    max_entropy: float
    mean_entropy: float
    std_entropy: float
    n_tokens: int
    n_tokens_filtered: int

    # Position-binned entropy (early, middle, late thirds)
    early_mean: float
    middle_mean: float
    late_mean: float

    # Confidence metrics
    confidence_score: float  # 1/(1+p95)

    def to_feature_vector(self) -> list[float]:
        """Convert to feature vector for ML."""
        return [
            self.p10_entropy,
            self.p50_entropy,
            self.p90_entropy,
            self.p95_entropy,
            self.max_entropy,
            self.mean_entropy,
            self.std_entropy,
            self.early_mean,
            self.middle_mean,
            self.late_mean,
            self.confidence_score,
            math.log(self.n_tokens + 1),  # Log-scaled token count
        ]

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "p10_entropy", "p50_entropy", "p90_entropy", "p95_entropy",
            "max_entropy", "mean_entropy", "std_entropy",
            "early_mean", "middle_mean", "late_mean",
            "confidence_score", "log_n_tokens"
        ]


# Stopwords for filtering
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "so", "yet", "for", "nor",
    "in", "on", "at", "to", "of", "by", "with", "from", "as", "into",
    "i", "you", "he", "she", "it", "we", "they", "this", "that",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    ".", ",", "!", "?", ":", ";", "-", "'", '"', "(", ")", "[", "]", "{", "}",
}


def is_stopword(token: str) -> bool:
    """Check if token should be filtered."""
    t = token.lower().strip()
    if t in STOPWORDS:
        return True
    if len(t) <= 1 and not t.isalnum():
        return True
    return False


def percentile(values: list[float], p: float) -> float:
    """Compute percentile (0-100)."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    k = (p / 100.0) * (n - 1)
    f, c = int(math.floor(k)), int(math.ceil(k))
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def compute_token_entropy(logprobs: list[float]) -> float:
    """Compute Shannon entropy from logprobs."""
    if not logprobs:
        return 0.0
    probs = [math.exp(lp) for lp in logprobs]
    total = sum(probs)
    if total <= 0:
        return 0.0
    probs = [p / total for p in probs]
    return -sum(p * math.log2(p + 1e-10) for p in probs if p > 0)


def extract_entropy_features(logprobs_content: list) -> EntropyFeatures:
    """Extract entropy features from OpenAI logprobs content."""
    all_entropies = []
    filtered_entropies = []

    for item in logprobs_content:
        if hasattr(item, 'top_logprobs') and item.top_logprobs:
            lps = [lp.logprob for lp in item.top_logprobs]
            entropy = compute_token_entropy(lps)
            all_entropies.append(entropy)

            token = getattr(item, 'token', '')
            if not is_stopword(token):
                filtered_entropies.append(entropy)

    if not filtered_entropies:
        filtered_entropies = all_entropies if all_entropies else [0.0]

    n = len(filtered_entropies)

    # Position-binned means
    third = max(1, n // 3)
    early = filtered_entropies[:third] if n > 0 else [0.0]
    middle = filtered_entropies[third:2*third] if n > third else [0.0]
    late = filtered_entropies[2*third:] if n > 2*third else [0.0]

    mean_ent = sum(filtered_entropies) / n if n > 0 else 0.0
    if n > 1:
        std_ent = math.sqrt(sum((e - mean_ent)**2 for e in filtered_entropies) / (n - 1))
    else:
        std_ent = 0.0

    p95 = percentile(filtered_entropies, 95)

    return EntropyFeatures(
        p10_entropy=percentile(filtered_entropies, 10),
        p50_entropy=percentile(filtered_entropies, 50),
        p90_entropy=percentile(filtered_entropies, 90),
        p95_entropy=p95,
        max_entropy=max(filtered_entropies) if filtered_entropies else 0.0,
        mean_entropy=mean_ent,
        std_entropy=std_ent,
        n_tokens=len(all_entropies),
        n_tokens_filtered=len(filtered_entropies),
        early_mean=sum(early) / len(early) if early else 0.0,
        middle_mean=sum(middle) / len(middle) if middle else 0.0,
        late_mean=sum(late) / len(late) if late else 0.0,
        confidence_score=1.0 / (1.0 + p95),
    )


def check_answer_correctness(response: str, correct_answers: list[str], incorrect_answers: list[str]) -> tuple[bool, str]:
    """
    Check if response is correct based on TruthfulQA ground truth.

    Returns (is_correct, match_type)
    """
    response_lower = response.lower().strip()

    # Check correct answers
    for correct in correct_answers:
        if correct.lower().strip() in response_lower:
            return True, "correct_match"

    # Check incorrect answers (hallucination indicators)
    for incorrect in incorrect_answers:
        if incorrect.lower().strip() in response_lower:
            return False, "incorrect_match"

    # Ambiguous - no clear match
    return None, "ambiguous"


def generate_with_entropy(
    client: OpenAI,
    question: str,
    model: str = "gpt-4o-mini",
    max_tokens: int = 150,
) -> tuple[str, EntropyFeatures]:
    """Generate response with entropy extraction."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Answer the question directly and concisely."},
            {"role": "user", "content": question}
        ],
        max_tokens=max_tokens,
        temperature=0.7,
        logprobs=True,
        top_logprobs=5,
    )

    answer = response.choices[0].message.content or ""
    logprobs_content = response.choices[0].logprobs.content if response.choices[0].logprobs else []

    features = extract_entropy_features(logprobs_content)
    return answer, features


def run_truthfulqa_experiment(
    n_samples: Optional[int] = None,
    model: str = "gpt-4o-mini",
    output_dir: str = "results",
):
    """Run the TruthfulQA entropy experiment."""

    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable required")

    client = OpenAI(api_key=api_key)

    # Load TruthfulQA
    print("Loading TruthfulQA dataset...")
    dataset = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")

    if n_samples:
        # Random sample
        indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
        samples = [dataset[int(i)] for i in indices]
    else:
        samples = list(dataset)

    print(f"Processing {len(samples)} samples...")

    # Collect data
    results = []
    X = []  # Feature vectors
    y = []  # Labels (1=correct, 0=hallucination)

    for i, sample in enumerate(samples):
        question = sample["question"]
        correct_answers = sample["correct_answers"]
        incorrect_answers = sample["incorrect_answers"]

        try:
            answer, features = generate_with_entropy(client, question, model)
            is_correct, match_type = check_answer_correctness(answer, correct_answers, incorrect_answers)

            result = {
                "question": question,
                "answer": answer,
                "correct_answers": correct_answers,
                "incorrect_answers": incorrect_answers,
                "is_correct": is_correct,
                "match_type": match_type,
                "entropy_features": asdict(features),
            }
            results.append(result)

            # Only use clear correct/incorrect for training
            if is_correct is not None:
                X.append(features.to_feature_vector())
                y.append(1 if is_correct else 0)

            if (i + 1) % 10 == 0:
                correct_count = sum(1 for r in results if r["is_correct"] is True)
                incorrect_count = sum(1 for r in results if r["is_correct"] is False)
                ambiguous_count = sum(1 for r in results if r["is_correct"] is None)
                print(f"  [{i+1}/{len(samples)}] Correct: {correct_count}, Incorrect: {incorrect_count}, Ambiguous: {ambiguous_count}")

            # Rate limiting
            time.sleep(0.1)

        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            continue

    # Save raw results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"truthfulqa_entropy_{timestamp}.json")

    with open(results_file, "w") as f:
        json.dump({
            "model": model,
            "n_samples": len(samples),
            "n_labeled": len(y),
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Train classifier if we have enough data
    if len(y) < 20:
        print(f"\nInsufficient labeled data ({len(y)} samples). Need at least 20.")
        return results, None

    X = np.array(X)
    y = np.array(y)

    print(f"\n{'='*60}")
    print("TRAINING ENTROPY-BASED HALLUCINATION DETECTOR")
    print(f"{'='*60}")
    print(f"Total labeled samples: {len(y)}")
    print(f"Correct: {sum(y)} ({100*sum(y)/len(y):.1f}%)")
    print(f"Hallucinations: {len(y)-sum(y)} ({100*(len(y)-sum(y))/len(y):.1f}%)")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train logistic regression
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = None

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    if auc:
        print(f"AUC-ROC:   {auc:.3f}")

    # Feature importance
    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCE (Logistic Regression Coefficients)")
    print(f"{'='*60}")
    feature_names = EntropyFeatures.feature_names()
    coefs = list(zip(feature_names, clf.coef_[0]))
    coefs.sort(key=lambda x: abs(x[1]), reverse=True)
    for name, coef in coefs:
        direction = "↑ correct" if coef > 0 else "↓ hallucination"
        print(f"  {name:20s}: {coef:+.4f} ({direction})")

    # Save model coefficients
    model_file = os.path.join(output_dir, f"entropy_classifier_{timestamp}.json")
    with open(model_file, "w") as f:
        json.dump({
            "feature_names": feature_names,
            "coefficients": clf.coef_[0].tolist(),
            "intercept": clf.intercept_[0],
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "auc": auc,
            },
            "training_samples": len(y_train),
            "test_samples": len(y_test),
        }, f, indent=2)
    print(f"\nModel saved to {model_file}")

    return results, clf


def main():
    parser = argparse.ArgumentParser(description="TruthfulQA Entropy Training")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples (default: 100)")
    parser.add_argument("--full", action="store_true", help="Use all 817 samples")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model to use")
    parser.add_argument("--output", default="results", help="Output directory")
    args = parser.parse_args()

    n_samples = None if args.full else args.samples

    run_truthfulqa_experiment(
        n_samples=n_samples,
        model=args.model,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
