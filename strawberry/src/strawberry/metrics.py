"""
Information-theoretic metrics used by the binding/routing framework.

All information quantities are in nats (natural logarithms) unless explicitly stated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple
import math
import numpy as np


def binary_entropy(p: float) -> float:
    """Binary entropy h(p) in nats."""
    p = float(p)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log(p) - (1.0 - p) * math.log(1.0 - p)


def fano_required_mi(M: int, eps: float) -> float:
    """
    Fano lower bound (tight for M-ary symmetric channel, uniform prior):
        I(V;Y) >= log M - h(eps) - eps log(M-1)
    Returns the RHS in nats.
    """
    if M <= 1:
        return 0.0
    eps = min(max(float(eps), 0.0), 1.0)
    if eps == 1.0:
        # If always wrong, bound becomes log M - 0 - log(M-1) = log(M/(M-1)).
        # But this isn't meaningful at eps=1 for classification; clamp slightly.
        eps = 1.0 - 1e-12
    return math.log(M) - binary_entropy(eps) - eps * math.log(M - 1)


def invert_fano_symmetric(M: int, I: float, tol: float = 1e-10) -> float:
    """
    Invert I = log M - h(eps) - eps log(M-1) for eps, assuming the M-ary symmetric channel model.
    This gives the *exact* eps if the confusion is symmetric; otherwise it's a heuristic.

    Uses bisection on eps in [0, 1-1/M].
    """
    if M <= 1:
        return 0.0
    I = float(I)
    # In symmetric channel, eps ∈ [0, 1-1/M]. At eps=1-1/M, I=0.
    lo, hi = 0.0, 1.0 - 1.0 / M
    # Clamp I to [0, log M]
    I = max(0.0, min(I, math.log(M)))
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        val = fano_required_mi(M, mid)
        # val decreases as eps increases
        if val > I:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def kl_bernoulli(p: float, q: float, clip: float = 1e-12) -> float:
    """
    KL(Ber(p) || Ber(q)) in nats, with clipping to avoid log(0).
    """
    p = min(max(float(p), clip), 1.0 - clip)
    q = min(max(float(q), clip), 1.0 - clip)
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


def bits_to_trust(p_tilde: float, eps: float) -> float:
    """
    B3(p_tilde, eps) = KL(Ber(1-eps) || Ber(p_tilde)) in nats.
    Interpretable as a minimal information budget needed to raise success from p_tilde to 1-eps.
    """
    return kl_bernoulli(1.0 - float(eps), float(p_tilde))


@dataclass
class ConfusionMI:
    """Results computed from a confusion matrix."""
    mi_nats: float
    error_rate: float
    M: int
    n: int


def mutual_information_from_confusion(conf: np.ndarray, eps_smooth: float = 1e-12) -> ConfusionMI:
    """
    Compute I(V;Y) from a confusion matrix conf[v, y] (counts).
    Uses additive smoothing to avoid log(0) in the MI sum.
    """
    conf = np.asarray(conf, dtype=float)
    if conf.ndim != 2 or conf.shape[0] != conf.shape[1]:
        raise ValueError("confusion matrix must be square (M x M)")
    M = conf.shape[0]
    n = int(conf.sum())
    if n <= 0:
        return ConfusionMI(mi_nats=0.0, error_rate=float("nan"), M=M, n=0)
    # Smooth
    pxy = (conf + eps_smooth) / (conf.sum() + eps_smooth * M * M)
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = pxy / (px @ py)
        mi = float(np.sum(pxy * np.log(ratio)))
    acc = float(np.trace(conf) / conf.sum())
    err = 1.0 - acc
    return ConfusionMI(mi_nats=mi, error_rate=err, M=M, n=n)


def bootstrap_confusion_mi(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    M: int,
    n_boot: int = 1000,
    seed: int = 0
) -> Dict[str, float]:
    """
    Nonparametric bootstrap CIs for MI and error based on resampling items.
    Returns dict with keys: mi_mean, mi_lo, mi_hi, err_mean, err_lo, err_hi.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    n = len(y_true)
    mi_vals = []
    err_vals = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        conf = np.zeros((M, M), dtype=float)
        for t, p in zip(y_true[idx], y_pred[idx]):
            conf[t, p] += 1
        r = mutual_information_from_confusion(conf)
        mi_vals.append(r.mi_nats)
        err_vals.append(r.error_rate)
    mi_vals = np.sort(np.asarray(mi_vals))
    err_vals = np.sort(np.asarray(err_vals))
    def q(a, qq): return float(np.quantile(a, qq))
    return {
        "mi_mean": float(np.mean(mi_vals)),
        "mi_lo": q(mi_vals, 0.025),
        "mi_hi": q(mi_vals, 0.975),
        "err_mean": float(np.mean(err_vals)),
        "err_lo": q(err_vals, 0.025),
        "err_hi": q(err_vals, 0.975),
        "n": int(n),
        "M": int(M),
    }
