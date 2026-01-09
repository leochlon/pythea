# Offline QMV Probing

Model-agnostic evaluation utilities for permutation-mixture / QMV-style probing via a **Bernoulli (0/1) first-token logprob probe**.

---

## What problem this solves

Many evaluation tasks reduce to a binary predicate:

> "Does the model say **YES** (1) or **NO** (0) for this prompt?"

`pythea.offline.qmv` provides:

- A **Bernoulli probe** estimating `q = P(token == "1")` from first-token logprobs
- Helpers to evaluate across **permutation families** of exchangeable blocks
- Derived metrics (mean, conservative quantile, JS dispersion, residuals) for robustness/leakage/contamination analysis

---

## Installation

```bash
pip install -e .
pip install -e ".[offline]"  # Optional: tiktoken for logit_bias
```

---

## Backend Contract

Your backend must implement:

```python
backend.generate_one_token(
    *,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    top_p: float,
    top_logprobs: int,
    logit_bias: dict[int, float] | None,
)
```

Must return an object with:
- `content: str` — the emitted token
- `top_logprobs: list[list[item]]` — first-token logprobs where each `item` has `token` and `logprob`

Optional (for token accounting):
- `raw` / `usage` fields with `prompt_tokens` / `completion_tokens`

Use `DummyBackend` for deterministic tests. See `examples/run_audit_openai.py` for a real adapter.

---

## BernoulliProbe

```python
from pythea.offline import qmv

probe = qmv.BernoulliProbe(
    backend=...,
    temperature=0.0,
    top_logprobs=20,
    use_logit_bias=True,
)
```

How it works:
1. Appends instruction: "Return exactly one token: 1 or 0."
2. Requests one token with `top_logprobs`
3. Renormalizes probability mass over 0/1-like tokens

Returns `ProbeDetails` with `q`, emitted string, token masses, usage.

---

## ExchangeablePromptParts

Represents a prompt where blocks can be reordered:

```python
parts = qmv.ExchangeablePromptParts(
    prefix="TASK: Decide if the predicate holds.",
    blocks=["HINT A: ...", "HINT B: ...", "HINT C: ..."],
    suffix="Return 1 if yes else 0.",
)

# Render specific permutation
prompt = parts.render(perm=[2, 0, 1])
```

---

## Permutation Families

```python
cfg = qmv.PermutationEvalConfig(
    m=6,               # number of permutations
    num_bands=6,       # shuffle-within-bands granularity
    seed=0,
    include_identity=True,
)
```

`generate_banded_permutations(n, m, num_bands, seed, include_identity)` builds deterministic families with indices shuffled within contiguous bands.

---

## evaluate_permutation_family(...)

Main entry point for QMV-style probing:

```python
res = qmv.evaluate_permutation_family(
    probe=probe,
    parts=parts,
    cfg=cfg,
    prior_quantile=0.05,
    workers=4,
)

print(res.q_bar, res.q_lo, res.js_bound)
```

**Outputs:**

| Field | Description |
|-------|-------------|
| `q_list` | Per-permutation P("1") |
| `q_bar` | Mean across permutations |
| `q_lo` | Conservative lower quantile |
| `jsd` | Generalized Jensen-Shannon divergence |
| `js_bound` | Certificate-style dispersion bound |
| `mean_abs_dev` | Empirical MAD around `q_bar` |
| `canonical_q` | Identity permutation probability |
| `canonical_resid_*` | Residuals vs mixture |

---

## Mixture Gating: evaluate_gate_with_features(...)

For posterior + skeleton prompt families:

```python
gate = qmv.MixtureGateConfig(hstar=0.05, prior_quantile=0.05)

result = qmv.evaluate_gate_with_features(
    probe=probe,
    posterior_prompt="...",
    skeleton_prompts=["...", "...", "..."],
    gate=gate,
    workers=4,
)

print(result.decision_answer, result.q_lo, result.B2T, result.ISR_ref)
```

**Outputs:**

| Field | Description |
|-------|-------------|
| `p_ref` | Posterior prompt probability |
| `q_bar`, `q_lo` | Skeleton mixture stats |
| `delta_avgKL`, `delta_refKL` | KL discrepancy summaries |
| `B2T`, `ISR_ref` | Planner-style gating terms |
| `jsd`, `js_bound` | Robustness diagnostics |

---

## Leakage Curves: build_leakage_curve(...)

Sweep over budgets (e.g., number of hints):

```python
curve = qmv.build_leakage_curve(
    probe=probe,
    base_prefix="LEAKAGE AUDIT",
    hints=["HINT 1", "HINT 2", "HINT 3"],
    base_suffix="Return 1 if you can output the secret exactly.",
    budgets=[0, 1, 2, 3],
    perm_cfg=qmv.PermutationEvalConfig(m=6, num_bands=2, seed=0),
)
```

Returns `LeakageCurvePoint` objects with `q_bar`, `q_lo`, `B2T`, `delta_pi_to_mix`, `js_bound`.

---

## Contamination Scoring

Fit null distribution over clean controls, score candidates:

```python
null = qmv.NullCalibration(values=[0.10, 0.12, 0.09, 0.11])

score = qmv.score_canonical_outlier(
    canonical_resid=0.50,
    null=null,
    alpha=0.01,
)

print(score.flagged, score.p_value, score.threshold)
```

---

## Chain-of-Thought Length Heuristics

Rule-of-thumb: `k ≈ c * sqrt(n) * log(1/ε)`

```python
k = qmv.heuristic_cot_tokens(n_context_tokens=100, epsilon=0.05, c=1.0)
```

Estimate constant from observations:

```python
obs = [
    {"n_context_tokens": 100.0, "epsilon": 0.05, "k": 30.0},
]
c_hat = qmv.estimate_cot_c_from_observations(obs)
```

---

## Package Layout

```
src/pythea/
  offline/qmv.py   # All probing + metrics
```

Examples in `examples/` (not installed).
