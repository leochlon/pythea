"""pythea.offline.qmv

Packaged version of the offline QMV / permutation-mixture feature builders.

This module is intentionally model-agnostic. To use it with a real LLM, pass a backend
object that implements `generate_one_token(...)` and returns first-token logprobs.

For deterministic local tests, use `DummyBackend`.
"""
from __future__ import annotations

import dataclasses
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor


# ---------------------------------------------------------------------------
# Optional backend import (same interface used by evidence_guard.py)
# ---------------------------------------------------------------------------

try:
    # If your repo contains backend.py (as evidence_guard.py expects), we use it.
    from backend import make_backend, BaseBackend, GenOutput  # type: ignore
except Exception:  # pragma: no cover
    BaseBackend = Any  # type: ignore
    GenOutput = Any  # type: ignore

    def make_backend(name: Optional[str] = None) -> Any:  # type: ignore
        raise RuntimeError(
            "backend.py not found. Pass an explicit backend=... to BernoulliProbe "
            "or install/provide a compatible backend module."
        )


# Optional tokenizer for OpenAI-style logit_bias
try:
    import tiktoken  # type: ignore

    _HAVE_TIKTOKEN = True
except Exception:  # pragma: no cover
    tiktoken = None
    _HAVE_TIKTOKEN = False


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _clip01(x: float, eps: float = 1e-9) -> float:
    return min(max(float(x), eps), 1.0 - eps)


def logit(p: float, eps: float = 1e-9) -> float:
    p = _clip01(p, eps)
    return math.log(p) - math.log(1.0 - p)


def sigmoid(t: float) -> float:
    # stable sigmoid
    if t >= 0:
        z = math.exp(-t)
        return 1.0 / (1.0 + z)
    z = math.exp(t)
    return z / (1.0 + z)


def bernoulli_kl(p: float, q: float, eps: float = 1e-9) -> float:
    """KL(Bern(p) || Bern(q)) in nats."""
    p = _clip01(p, eps)
    q = _clip01(q, eps)
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))



def mean_bernoulli_kl(
    ps: Sequence[float],
    q: float,
    *,
    weights: Optional[Sequence[float]] = None,
) -> float:
    """Mean KL(Ber(p_i) || Ber(q))."""
    vals = [bernoulli_kl(float(p), float(q)) for p in ps]
    return float(_mean(vals, weights))

def mean_symmetric_bernoulli_kl(
    ps: Sequence[float],
    q: float,
    *,
    weights: Optional[Sequence[float]] = None,
) -> float:
    """Mean 0.5*(KL(p_i||q) + KL(q||p_i)) for Bernoulli probabilities."""
    vals = [0.5 * (bernoulli_kl(float(p), float(q)) + bernoulli_kl(float(q), float(p))) for p in ps]
    return float(_mean(vals, weights))

def bernoulli_entropy(p: float, eps: float = 1e-12) -> float:
    p = _clip01(p, eps)
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


def _mean(xs: Sequence[float], ws: Optional[Sequence[float]] = None) -> float:
    if not xs:
        return float("nan")
    if ws is None:
        return sum(xs) / len(xs)
    s = 0.0
    tw = 0.0
    for x, w in zip(xs, ws):
        s += float(x) * float(w)
        tw += float(w)
    return s / max(tw, 1e-12)


def _weighted_quantile(xs: Sequence[float], q: float, ws: Optional[Sequence[float]] = None) -> float:
    """Weighted quantile (q in [0,1])."""
    assert 0.0 <= q <= 1.0
    if not xs:
        return float("nan")
    if ws is None:
        xs_sorted = sorted(float(x) for x in xs)
        idx = int((len(xs_sorted) - 1) * q + 0.5)
        return xs_sorted[idx]
    pairs = sorted(zip(xs, ws), key=lambda x: float(x[0]))
    total_w = sum(float(w) for _, w in pairs)
    if total_w <= 0:
        return float(pairs[-1][0])
    target = q * total_w
    cum = 0.0
    for val, w in pairs:
        cum += float(w)
        if cum >= target:
            return float(val)
    return float(pairs[-1][0])


def jsd_bernoulli(q_list: Sequence[float], ws: Optional[Sequence[float]] = None, eps: float = 1e-9) -> float:
    """Generalized Jensen–Shannon divergence for Bernoulli parameters.

    For distributions S_k = Ber(q_k) and S̄ = Σ w_k S_k, we compute:
        JSD = Σ w_k KL(S_k || S̄).

    In the Bernoulli case, S̄ is Ber(q̄) where q̄ = Σ w_k q_k.

    This matches the quantity used in Proposition 2 (up to weighting), which yields:
        E|qπ - q̄| ≤ sqrt(1/2 * JSD)

    (Pinsker + data processing through g). See 2509.11208 Proposition 2.
    """
    if not q_list:
        return float("nan")
    q_bar = _mean(list(q_list), ws)
    # JSD = E KL(Ber(q_k) || Ber(q̄))
    kls = [bernoulli_kl(float(q), q_bar, eps=eps) for q in q_list]
    return _mean(kls, ws)


def js_dispersion_bound(q_list: Sequence[float], ws: Optional[Sequence[float]] = None) -> float:
    """Proposition-2-style bound on E|qπ - q̄|.

    For Bernoulli predicates: E|qπ - q̄| ≤ sqrt(1/2 * E KL(Ber(qπ) || Ber(q̄))).
    """
    jsd = jsd_bernoulli(q_list, ws)
    if not math.isfinite(jsd):
        return float("nan")
    return math.sqrt(0.5 * max(jsd, 0.0))


def mean_abs_deviation(q_list: Sequence[float], ws: Optional[Sequence[float]] = None) -> float:
    if not q_list:
        return float("nan")
    q_bar = _mean(list(q_list), ws)
    devs = [abs(float(q) - q_bar) for q in q_list]
    return _mean(devs, ws)


def jensen_gap_bernoulli(q_list: Sequence[float], y: int, ws: Optional[Sequence[float]] = None, eps: float = 1e-12) -> float:
    """Jensen gap for a fixed label y ∈ {0,1} under permutation mixtures.

    Given pk = Pr(Y=y | Γπk(x)), Jensen says:
        E[-log pk] - (-log E[pk]) >= 0.

    For Bernoulli with parameter qk = Pr(Y=1):
        pk = qk      if y=1
           = 1-qk    if y=0
    """
    if y not in (0, 1):
        raise ValueError("y must be 0 or 1")
    if not q_list:
        return float("nan")
    pk_list = [(_clip01(float(q), eps) if y == 1 else _clip01(1.0 - float(q), eps)) for q in q_list]
    mean_ce = _mean([-math.log(pk) for pk in pk_list], ws)
    mix_p = _mean(pk_list, ws)
    mix_ce = -math.log(_clip01(mix_p, eps))
    return float(mean_ce - mix_ce)


# ---------------------------------------------------------------------------
# Permutations / exchangeable prompt rendering
# ---------------------------------------------------------------------------


def banded_permutation_indices(n: int, *, num_bands: int = 6, rng: random.Random) -> List[int]:
    """Return a single banded permutation of indices 0..n-1.

    We split indices into `num_bands` contiguous bands (as equal as possible),
    then shuffle *within* each band. This matches the paper's "6 bands, shuffle within".
    """
    if n <= 1:
        return list(range(n))
    num_bands = max(1, int(num_bands))
    # Compute band boundaries
    # Example: n=10, bands=6 -> sizes [2,2,2,2,1,1]
    base = n // num_bands
    rem = n % num_bands
    sizes = [base + (1 if i < rem else 0) for i in range(num_bands)]
    # Remove empty bands if num_bands>n
    sizes = [s for s in sizes if s > 0]
    perm: List[int] = []
    start = 0
    for s in sizes:
        band = list(range(start, start + s))
        rng.shuffle(band)
        perm.extend(band)
        start += s
    assert len(perm) == n
    return perm


def generate_banded_permutations(
    n: int,
    *,
    m: int,
    num_bands: int = 6,
    seed: int = 0,
    include_identity: bool = True,
) -> List[List[int]]:
    """Generate m banded permutations (deterministic given seed).

    If include_identity=True, the first permutation is the identity ordering.
    """
    m = int(m)
    if m <= 0:
        return []
    rng = random.Random(int(seed))

    perms: List[List[int]] = []
    if include_identity:
        perms.append(list(range(n)))
    while len(perms) < m:
        perms.append(banded_permutation_indices(n, num_bands=num_bands, rng=rng))
    return perms[:m]


@dataclass
class ExchangeablePromptParts:
    """A prompt built from a fixed prefix/suffix and a set of exchangeable blocks."""

    prefix: str
    blocks: List[str]
    suffix: str
    block_sep: str = "\n\n"

    def render(self, perm: Sequence[int]) -> str:
        ordered = [self.blocks[i] for i in perm]
        body = self.block_sep.join(ordered)
        if self.prefix and not self.prefix.endswith("\n"):
            pre = self.prefix + "\n"
        else:
            pre = self.prefix
        if self.suffix and not self.suffix.startswith("\n"):
            suf = "\n" + self.suffix
        else:
            suf = self.suffix
        return f"{pre}{body}{suf}".strip() + "\n"


# ---------------------------------------------------------------------------
# Bernoulli probe (0/1 via first-token logprobs)
# ---------------------------------------------------------------------------


def _extract_usage_tokens(raw: Any) -> Tuple[int, int]:
    """Extract (prompt_tokens, completion_tokens) from a provider response."""
    if raw is None:
        return (0, 0)
    try:
        usage = getattr(raw, "usage", None)
        if usage is not None:
            if isinstance(usage, dict):
                pt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
                ct = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
                return (pt, ct)
            pt = int(getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0)
            ct = int(getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0) or 0)
            if pt or ct:
                return (pt, ct)
    except Exception:
        pass
    if isinstance(raw, dict):
        u = raw.get("usage") or {}
        if isinstance(u, dict):
            pt = int(u.get("prompt_tokens") or u.get("input_tokens") or 0)
            ct = int(u.get("completion_tokens") or u.get("output_tokens") or 0)
            return (pt, ct)
    for meth in ("model_dump", "to_dict", "dict"):
        try:
            if hasattr(raw, meth):
                d = getattr(raw, meth)()
                if isinstance(d, dict):
                    u = d.get("usage") or {}
                    if isinstance(u, dict):
                        pt = int(u.get("prompt_tokens") or u.get("input_tokens") or 0)
                        ct = int(u.get("completion_tokens") or u.get("output_tokens") or 0)
                        return (pt, ct)
        except Exception:
            pass
    return (0, 0)


@dataclass
class ProbeDetails:
    q: float
    emitted: str
    p1_mass: Optional[float] = None
    p0_mass: Optional[float] = None
    tokens_in: int = 0
    tokens_out: int = 0
    raw: Any = None


class BernoulliProbe:
    """Two-label logprob probe over {0,1} using first-token top-logprobs."""

    def __init__(
        self,
        backend: Optional[BaseBackend] = None,
        *,
        temperature: float = 0.0,
        top_logprobs: int = 20,
        use_logit_bias: bool = True,
        tokenizer_name: Optional[str] = None,
        # Accept variants so that we renormalize even if backend adds spaces/newlines
        tokens_one: Optional[Iterable[str]] = None,
        tokens_zero: Optional[Iterable[str]] = None,
        system_prompt: Optional[str] = None,
    ):
        self.backend: BaseBackend = backend or make_backend(os.environ.get("BACKEND"))
        self.temperature = float(temperature)
        self.top_logprobs = int(top_logprobs)

        self.tokens_one = set(tokens_one or {"1", " 1", "\n1", "\t1"})
        self.tokens_zero = set(tokens_zero or {"0", " 0", "\n0", "\t0"})
        self.system_prompt = system_prompt or (
            "Output exactly ONE token: 1 or 0.\n"
            "1 = YES / predicate holds.\n"
            "0 = NO / predicate does not hold.\n"
            "No extra text."
        )

        self.logit_bias: Optional[Dict[int, float]] = None
        if use_logit_bias and _HAVE_TIKTOKEN:
            try:
                enc = (
                    tiktoken.get_encoding(tokenizer_name)
                    if tokenizer_name
                    else tiktoken.get_encoding("cl100k_base")
                )
                ids: List[int] = []
                for s in sorted(self.tokens_one | self.tokens_zero):
                    try:
                        toks = enc.encode(s)
                        if len(toks) == 1:
                            ids.append(toks[0])
                    except Exception:
                        pass
                self.logit_bias = {tid: 80.0 for tid in ids} if ids else None
            except Exception:
                self.logit_bias = None

    def _renorm_01(self, top: List[Any]) -> Tuple[float, float]:
        p0 = 0.0
        p1 = 0.0
        for item in top or []:
            tok = getattr(item, "token", None)
            lp = getattr(item, "logprob", None)
            if tok is None or lp is None:
                if isinstance(item, dict):
                    tok = item.get("token")
                    lp = item.get("logprob")
                elif isinstance(item, (tuple, list)) and len(item) >= 2:
                    tok, lp = item[0], item[1]
            if tok is None or lp is None:
                continue
            try:
                prob = math.exp(float(lp))
            except Exception:
                continue
            s = str(tok)
            if tok in self.tokens_one or s.lstrip() == "1":
                p1 += prob
            elif tok in self.tokens_zero or s.lstrip() == "0":
                p0 += prob
        return p0, p1

    def probe(self, prompt: str) -> ProbeDetails:
        """Return q=P('1') renormalized over {0,1}."""
        user_prompt = (prompt or "").strip() + "\n\nReturn exactly one token: 1 or 0."

        out: GenOutput = self.backend.generate_one_token(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            temperature=self.temperature,
            top_p=1.0,
            top_logprobs=self.top_logprobs,
            logit_bias=self.logit_bias,
        )

        top = (out.top_logprobs[0] if getattr(out, "top_logprobs", None) else None)
        q: Optional[float] = None
        p0 = p1 = None
        if top:
            p0, p1 = self._renorm_01(top)
            denom = p0 + p1
            if denom > 0:
                q = p1 / denom

        emitted = (getattr(out, "content", "") or "").strip()
        if q is None:
            if emitted.startswith("1"):
                q = 1.0
            elif emitted.startswith("0"):
                q = 0.0
            else:
                q = 0.0

        # Token usage if available
        raw = getattr(out, "raw", None)
        tokens_in, tokens_out = _extract_usage_tokens(raw)
        if tokens_in == 0 and tokens_out == 0:
            # Some backends store usage on the GenOutput directly
            tokens_in, tokens_out = _extract_usage_tokens(out)

        return ProbeDetails(
            q=float(q),
            emitted=emitted,
            p1_mass=p1,
            p0_mass=p0,
            tokens_in=int(tokens_in),
            tokens_out=int(tokens_out),
            raw=raw,
        )


# ---------------------------------------------------------------------------
# Mixture evaluation and feature extraction
# ---------------------------------------------------------------------------


@dataclass
class MixtureGateConfig:
    """Configuration for EDFL-style gates."""

    hstar: float = 0.05
    prior_quantile: float = 0.05


@dataclass
class MixtureRunResult:
    """All metrics we compute for a posterior vs a family of skeletons."""

    decision_answer: bool
    p_ref: float
    q_bar: float
    q_lo: float
    delta_avgKL: float
    delta_refKL: float
    B2T: float
    ISR_ref: Optional[float]
    # "Worth keeping" add-ons
    jsd: float
    js_bound: float
    mean_abs_dev: float
    # Jensen gap is optional (needs a label)
    jensen_gap: Optional[float] = None
    # Raw per-skeleton probabilities
    q_list: Optional[List[float]] = None
    # Debug
    ref_debug: Optional[Dict[str, Any]] = None
    priors_debug: Optional[List[Dict[str, Any]]] = None
    tokens_in: int = 0
    tokens_out: int = 0


def evaluate_gate_with_features(
    *,
    probe: BernoulliProbe,
    posterior_prompt: str,
    skeleton_prompts: Sequence[str],
    skeleton_weights: Optional[Sequence[float]] = None,
    gate: Optional[MixtureGateConfig] = None,
    label_y: Optional[int] = None,
    workers: int = 1,
) -> MixtureRunResult:
    """Extended version of evidence_guard.evaluate_evidence_mixture_logprob.

    In addition to EDFL planners (B2T/ISR), we compute:
      - JS certificate bound (Prop 2)
      - mean absolute deviation across permutations
      - Jensen gap (if label_y is provided)
    """

    gate = gate or MixtureGateConfig()
    ws = list(skeleton_weights) if skeleton_weights is not None else None
    if ws is not None and len(ws) != len(skeleton_prompts):
        raise ValueError("skeleton_weights length must match skeleton_prompts")

    ref = probe.probe(posterior_prompt)
    p_ref = float(ref.q)

    # Priors
    if workers <= 1:
        priors = [probe.probe(p) for p in skeleton_prompts]
    else:
        with ThreadPoolExecutor(max_workers=int(workers)) as ex:
            priors = list(ex.map(probe.probe, skeleton_prompts))

    q_list = [float(r.q) for r in priors]
    q_bar = _mean(q_list, ws)
    q_lo = _weighted_quantile(q_list, gate.prior_quantile, ws)

    # EDFL metrics (same as evidence_guard)
    delta_avgKL = _mean([bernoulli_kl(qi, p_ref) for qi in q_list], ws)
    delta_refKL = _mean([bernoulli_kl(p_ref, qi) for qi in q_list], ws)
    target = 1.0 - float(gate.hstar)
    B2T = bernoulli_kl(target, q_lo)
    ISR_ref = None if B2T <= 0 else float("inf") if B2T == 0 else (delta_refKL / B2T)

    super_pass = (p_ref > target) and (q_lo > target)
    decision_answer = bool((ISR_ref is not None and ISR_ref > 1.0) or super_pass)

    # Worth-keeping extras
    jsd = jsd_bernoulli(q_list, ws)
    js_bound = js_dispersion_bound(q_list, ws)
    mad = mean_abs_deviation(q_list, ws)

    jgap = None
    if label_y is not None:
        jgap = jensen_gap_bernoulli(q_list, int(label_y), ws)

    # Tokens
    tokens_in = int(ref.tokens_in)
    tokens_out = int(ref.tokens_out)
    for r in priors:
        tokens_in += int(r.tokens_in)
        tokens_out += int(r.tokens_out)

    return MixtureRunResult(
        decision_answer=decision_answer,
        p_ref=p_ref,
        q_bar=float(q_bar),
        q_lo=float(q_lo),
        delta_avgKL=float(delta_avgKL),
        delta_refKL=float(delta_refKL),
        B2T=float(B2T),
        ISR_ref=None if ISR_ref is None else float(ISR_ref),
        jsd=float(jsd),
        js_bound=float(js_bound),
        mean_abs_dev=float(mad),
        jensen_gap=None if jgap is None else float(jgap),
        q_list=list(q_list),
        ref_debug=dataclasses.asdict(ref),
        priors_debug=[dataclasses.asdict(r) for r in priors],
        tokens_in=int(tokens_in),
        tokens_out=int(tokens_out),
    )


@dataclass
class PermutationEvalConfig:
    m: int = 6
    num_bands: int = 6
    seed: int = 0
    include_identity: bool = True


@dataclass
class PermutationEvalResult:
    perms: List[List[int]]
    prompts: List[str]
    q_list: List[float]
    q_bar: float
    q_lo: float
    jsd: float
    js_bound: float
    mean_abs_dev: float
    jensen_gap: Optional[float] = None
    # canonical residuals
    canonical_q: Optional[float] = None
    canonical_resid_prob: Optional[float] = None
    canonical_resid_logit: Optional[float] = None


def evaluate_permutation_family(
    *,
    probe: BernoulliProbe,
    parts: ExchangeablePromptParts,
    cfg: Optional[PermutationEvalConfig] = None,
    prior_quantile: float = 0.05,
    weights: Optional[Sequence[float]] = None,
    label_y: Optional[int] = None,
    workers: int = 1,
) -> PermutationEvalResult:
    """Evaluate a Bernoulli predicate across a banded permutation family."""

    cfg = cfg or PermutationEvalConfig()
    n = len(parts.blocks)
    perms = generate_banded_permutations(
        n,
        m=cfg.m,
        num_bands=cfg.num_bands,
        seed=cfg.seed,
        include_identity=cfg.include_identity,
    )
    prompts = [parts.render(p) for p in perms]
    ws = list(weights) if weights is not None else None
    if ws is not None and len(ws) != len(prompts):
        raise ValueError("weights length must match number of permutations/prompts")

    if workers <= 1:
        res = [probe.probe(p) for p in prompts]
    else:
        with ThreadPoolExecutor(max_workers=int(workers)) as ex:
            res = list(ex.map(probe.probe, prompts))
    q_list = [float(r.q) for r in res]

    q_bar = _mean(q_list, ws)
    q_lo = _weighted_quantile(q_list, prior_quantile, ws)
    jsd = jsd_bernoulli(q_list, ws)
    jsb = js_dispersion_bound(q_list, ws)
    mad = mean_abs_deviation(q_list, ws)

    jgap = None
    if label_y is not None:
        jgap = jensen_gap_bernoulli(q_list, int(label_y), ws)

    # Canonical residuals compare the identity perm (index 0 if include_identity) to mixture
    canonical_q = None
    resid_prob = None
    resid_logit = None
    if cfg.include_identity and q_list:
        canonical_q = float(q_list[0])
        resid_prob = abs(canonical_q - float(q_bar))
        resid_logit = abs(logit(canonical_q) - logit(float(q_bar)))

    return PermutationEvalResult(
        perms=perms,
        prompts=prompts,
        q_list=q_list,
        q_bar=float(q_bar),
        q_lo=float(q_lo),
        jsd=float(jsd),
        js_bound=float(jsb),
        mean_abs_dev=float(mad),
        jensen_gap=None if jgap is None else float(jgap),
        canonical_q=canonical_q,
        canonical_resid_prob=resid_prob,
        canonical_resid_logit=resid_logit,
    )


# ---------------------------------------------------------------------------
# Leakage curves (Bits-to-Leak)
# ---------------------------------------------------------------------------


@dataclass
class LeakageCurvePoint:
    budget: int
    num_hints: int
    # Mixture prior stats
    q_bar: float
    q_lo: float
    # EDFL/B2T planner term
    B2T: float
    # Information-budget proxies in nats (Bernoulli specialization)
    delta_pi_to_mix: float
    delta_mix_to_pi: float
    delta_sym: float
    # Dispersion diagnostics
    jsd: float
    js_bound: float
    mean_abs_dev: float


def build_leakage_curve(
    *,
    probe: BernoulliProbe,
    base_prefix: str,
    hints: Sequence[str],
    base_suffix: str,
    budgets: Sequence[int],
    hstar: float = 0.05,
    prior_quantile: float = 0.05,
    perm_cfg: Optional[PermutationEvalConfig] = None,
    workers: int = 1,
) -> List[LeakageCurvePoint]:
    """Build a leakage/extraction curve over an external 'budget' schedule.

    This helper is intentionally flexible about what a "budget" is. The default
    interpretation is "number of hints included" (first b hints), but *the*
    theoretically grounded x-axis for EDFL-style reporting is an information
    budget (in nats). For Bernoulli probes, a natural proxy is:

        Δ̄_pi→mix(b) = E_pi KL(Ber(q_pi) || Ber(q̄))

    which is exactly the quantity that controls permutation dispersion via
    Pinsker/data-processing (Prop 2 style). So a principled report is:
      - q̄(b), q_lo(b)
      - B2T(b) = KL(target || q_lo(b))
      - Δ̄_pi→mix(b) in nats

    so you can plot (Δ̄_pi→mix, q̄) or (Δ̄_pi→mix, q_lo), not only (num_hints, q̄).

    Parameters
    ----------
    probe:
        BernoulliProbe instance (real backend or DummyBackend).
    base_prefix/base_suffix:
        Text around the hint blocks.
    hints:
        Hint strings; treated as exchangeable blocks.
    budgets:
        Iterable of integers; each b uses the first b hints. If you want a
        different selection strategy (e.g., most-informative hints), do it upstream.
    hstar:
        Paper-style tolerance (target reliability = 1-hstar).
    prior_quantile:
        Conservative prior quantile over permutations.
    perm_cfg:
        Permutation family config (banded shuffles).
    workers:
        Parallelism for probing permutations.

    Returns
    -------
    List[LeakageCurvePoint]
    """
    points: List[LeakageCurvePoint] = []
    gate_target = 1.0 - float(hstar)
    perm_cfg = perm_cfg or PermutationEvalConfig()

    for b in budgets:
        b_int = int(b)
        b_int = max(0, min(b_int, len(hints)))
        blocks = list(hints[:b_int])

        parts = ExchangeablePromptParts(prefix=base_prefix, blocks=blocks or [""], suffix=base_suffix)

        res = evaluate_permutation_family(
            probe=probe,
            parts=parts,
            cfg=perm_cfg,
            prior_quantile=prior_quantile,
            workers=workers,
        )

        # EDFL/B2T planner term (paper)
        B2T = bernoulli_kl(gate_target, res.q_lo)

        # Info-budget proxies (nats)
        delta_pi_to_mix = mean_bernoulli_kl(res.q_list, res.q_bar)
        delta_mix_to_pi = float(_mean([bernoulli_kl(res.q_bar, qi) for qi in res.q_list]))
        delta_sym = mean_symmetric_bernoulli_kl(res.q_list, res.q_bar)

        points.append(
            LeakageCurvePoint(
                budget=b_int,
                num_hints=b_int,
                q_bar=res.q_bar,
                q_lo=res.q_lo,
                B2T=float(B2T),
                delta_pi_to_mix=float(delta_pi_to_mix),
                delta_mix_to_pi=float(delta_mix_to_pi),
                delta_sym=float(delta_sym),
                jsd=res.jsd,
                js_bound=res.js_bound,
                mean_abs_dev=res.mean_abs_dev,
            )
        )

    return points


# ---------------------------------------------------------------------------
# Contamination audit helpers (QMV-aware null calibration)
# ---------------------------------------------------------------------------


@dataclass
class NullCalibration:
    """Empirical null distribution for a scalar statistic."""

    values: List[float]

    def __post_init__(self) -> None:
        self.values = [float(v) for v in (self.values or [])]
        self.values.sort()

    def quantile(self, q: float) -> float:
        return _weighted_quantile(self.values, q, None)

    def cdf(self, x: float) -> float:
        """Empirical CDF at x."""
        if not self.values:
            return float("nan")
        # bisect right
        lo, hi = 0, len(self.values)
        xv = float(x)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.values[mid] <= xv:
                lo = mid + 1
            else:
                hi = mid
        return lo / len(self.values)

    def p_value_right_tail(self, x: float) -> float:
        """Right-tail p-value P(V >= x) under the empirical null."""
        c = self.cdf(x)
        if not math.isfinite(c):
            return float("nan")
        return max(0.0, 1.0 - c)


@dataclass
class ContaminationScore:
    resid: float
    p_value: float
    threshold: float
    flagged: bool


def score_canonical_outlier(
    *,
    canonical_resid: float,
    null: NullCalibration,
    alpha: float = 0.01,
) -> ContaminationScore:
    """Score whether canonical-ordering residual is an outlier vs calibrated null.

    - canonical_resid: e.g., |logit(q_id) - logit(q̄)| or |q_id - q̄|
    - null: fitted on clean exchangeable controls
    - alpha: flag threshold
    """
    thr = null.quantile(1.0 - float(alpha))
    p = null.p_value_right_tail(float(canonical_resid))
    return ContaminationScore(
        resid=float(canonical_resid),
        p_value=float(p),
        threshold=float(thr),
        flagged=bool(float(canonical_resid) >= thr),
    )


# ---------------------------------------------------------------------------
# Optimal chain-of-thought length helper
# ---------------------------------------------------------------------------


def heuristic_cot_tokens(
    *,
    n_context_tokens: int,
    epsilon: float,
    c: float = 1.0,
    log_base: str = "e",
) -> int:
    """Heuristic implementation of k* ≈ c * sqrt(n) * log(1/ε).

    Paper 2507.11768 derives k* = Θ(√n log(1/ε)) with model-specific constants.
    Here we provide a rule-of-thumb implementation unless you estimate constants from data.

    Parameters
    ----------
    n_context_tokens:
        Rough size of context/problem; you can use token count or chunk count.
    epsilon:
        Target error probability (e.g., 0.05).
    c:
        Constant factor (environment/model dependent).
    log_base:
        'e' for natural log, '2' for log2.
    """
    n = max(0, int(n_context_tokens))
    eps = float(epsilon)
    if eps <= 0.0:
        raise ValueError("epsilon must be > 0")
    if eps >= 1.0:
        return 0
    if n == 0:
        return 0
    if log_base == "2":
        L = math.log(1.0 / eps, 2)
    else:
        L = math.log(1.0 / eps)
    k = float(c) * math.sqrt(float(n)) * L
    return int(math.ceil(k))


def estimate_cot_c_from_observations(
    observations: Sequence[Dict[str, float]],
    *,
    log_base: str = "e",
) -> float:
    """Estimate the constant c in k ≈ c * sqrt(n) * log(1/ε) from observations.

    This is a lightweight alternative to implementing Algorithm 1 from the paper.
    Provide observations with keys:
      - n_context_tokens
      - epsilon
      - k  (observed/selected reasoning token budget)

    We fit c by least squares:  k_i ≈ c * sqrt(n_i) * log(1/ε_i).
    """
    xs: List[float] = []
    ys: List[float] = []
    for obs in observations:
        n = float(obs.get("n_context_tokens", 0.0))
        eps = float(obs.get("epsilon", 0.0))
        k = float(obs.get("k", 0.0))
        if n <= 0 or not (0.0 < eps < 1.0) or k <= 0:
            continue
        if log_base == "2":
            L = math.log(1.0 / eps, 2)
        elif log_base == "10":
            L = math.log10(1.0 / eps)
        else:
            L = math.log(1.0 / eps)
        x = math.sqrt(n) * L
        xs.append(x)
        ys.append(k)
    if not xs:
        return 1.0
    num = sum(x * y for x, y in zip(xs, ys))
    den = sum(x * x for x in xs)
    return float(num / den) if den > 0 else 1.0


def optimal_cot_tokens(
    *,
    n_context_tokens: int,
    epsilon: float,
    c: float = 1.0,
    log_base: str = "e",
) -> int:
    """Backward-compatible alias for `heuristic_cot_tokens` (deprecated)."""
    import warnings
    warnings.warn(
        "optimal_cot_tokens is a heuristic in this module. Use heuristic_cot_tokens "
        "and estimate c from data (estimate_cot_c_from_observations) if you need calibration.",
        DeprecationWarning,
        stacklevel=2,
    )
    return heuristic_cot_tokens(n_context_tokens=n_context_tokens, epsilon=epsilon, c=c, log_base=log_base)


# ---------------------------------------------------------------------------
# Minimal dummy backend support (optional)
# ---------------------------------------------------------------------------


@dataclass
class _DummyTopLogprob:
    token: str
    logprob: float


@dataclass
class DummyGenOutput:
    content: str
    top_logprobs: List[List[_DummyTopLogprob]]
    raw: Any = None


class DummyBackend:
    """A tiny backend for offline testing.

    The probability of '1' is determined by a user-supplied function that maps the
    concatenated prompts -> q in [0,1].
    """

    def __init__(self, prob_fn):
        self.prob_fn = prob_fn

    def generate_one_token(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        top_p: float,
        top_logprobs: int,
        logit_bias: Optional[Dict[int, float]] = None,
    ) -> DummyGenOutput:
        text = (system_prompt or "") + "\n" + (user_prompt or "")
        q = float(self.prob_fn(text))
        q = _clip01(q, 1e-9)
        # deterministic emission
        content = "1" if q >= 0.5 else "0"
        top = [
            _DummyTopLogprob(token="1", logprob=math.log(q)),
            _DummyTopLogprob(token="0", logprob=math.log(1.0 - q)),
        ]
        return DummyGenOutput(content=content, top_logprobs=[top], raw=None)
