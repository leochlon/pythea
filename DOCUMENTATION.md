# pythea documentation

This document describes the public API and intended usage of the `pythea` package.

`pythea` has two main components:

1. `pythea.client` — a small HTTP client for the **Thea Mini Reasoning API**.
2. `pythea.offline.qmv` — offline, model-agnostic tooling for permutation-mixture / QMV-style
   evaluation using a Bernoulli (0/1) first-token logprob probe.

---

## Installation

From source (recommended for this repo):

```bash
pip install -e .
```

Optional extras:

```bash
# Tests / local dev
pip install -e ".[dev]"

# Optional tokenizer helper used by the Bernoulli probe for OpenAI-style logit_bias
pip install -e ".[offline]"
```

Runtime dependencies are intentionally minimal: `httpx>=0.24.0`.

---

## Package layout

```
src/pythea/
  __init__.py          # exports: TheaClient, AsyncTheaClient, __version__
  client.py            # sync + async API clients
  types.py             # TypedDict request/response shapes
  errors.py            # small exception hierarchy
  offline/qmv.py       # offline probing + metrics
```

The `examples/` directory contains standalone scripts and is **not installed** as part of the
package.

---

## Thea API client

### TheaClient (synchronous)

```python
from pythea import TheaClient

with TheaClient(base_url="https://apim-reasoning-core.azure-api.net/reasoning") as client:
    print(client.healthz())
    resp = client.unified_answer(question="What is 2+2?")
    print(resp.get("decision"), resp.get("picked"))
```

#### Constructor

```python
TheaClient(
    *,
    base_url: str,
    apim_subscription_key: str | None = None,
    apim_subscription_key_header: str = "Ocp-Apim-Subscription-Key",
    extra_headers: Mapping[str, str] | None = None,
    timeout_s: float = 60.0,
    verify: bool | str = True,
    http: httpx.Client | None = None,
    transport: httpx.BaseTransport | None = None,
)
```

Key parameters:

- `base_url` (required): the root of the service, e.g. `https://thea.example.com`.
  Trailing slashes are stripped.
- `apim_subscription_key`: if your service is behind Azure API Management.
- `extra_headers`: default headers to include on every request.
- `timeout_s`: overall request timeout (seconds).
- `verify`: TLS verification (`True`/`False` or a CA bundle path). Useful for dev clusters.
- `http`: provide your own `httpx.Client` if you want full control (proxies, connection pooling,
  auth, etc.). When you pass `http=...`, **you own its lifecycle**.
- `transport`: dependency-injection hook for testing (e.g. `httpx.MockTransport`).

#### Methods

##### `healthz()`

Calls `GET /healthz` and returns the decoded JSON object.

```python
status = client.healthz()
```

##### `unified_answer(...)`

Calls `POST /api/unified-answer`.

```python
resp = client.unified_answer(
    question="What is 2+2?",
    evidence=None,
    backend="aoai-pool",
    interpretability=True,
    prompt_rewrite=False,
    creds=None,
    # Any other JSON fields your deployment supports can be passed as kwargs:
    m=6,
    temperature=0.0,
)
```

Notes:

- `question` is the only required field.
- All keyword arguments with value `None` are ignored.
- Extra keyword arguments are passed through into the JSON body unchanged.

The request/response are typed as `TypedDict` in `pythea.types`:

- `UnifiedAnswerRequest`
- `UnifiedAnswerResponse`
- `BackendCredentials` (per-request credentials for some deployments)

### AsyncTheaClient (asynchronous)

```python
import asyncio
from pythea import AsyncTheaClient


async def main() -> None:
    async with AsyncTheaClient(base_url="https://apim-reasoning-core.azure-api.net/reasoning") as client:
        print(await client.healthz())
        resp = await client.unified_answer(question="What is 2+2?")
        print(resp.get("decision"), resp.get("picked"))


asyncio.run(main())
```

The async client mirrors the synchronous client. The only difference is lifecycle management:

- Call `await client.aclose()` to close the underlying `httpx.AsyncClient`
- Or use `async with ...` which closes automatically

---

## Errors and exception handling

All custom exceptions derive from `pythea.errors.TheaError`.

### `TheaHTTPError`

Raised when the server returns `>= 400`. Fields:

- `status_code: int`
- `message: str` (best-effort extracted from the JSON error response)
- `response_text: str | None`
- `request_id: str | None` (from the `X-Request-ID` response header, if present)

### `TheaTimeoutError`

Raised when `httpx` times out.

### `TheaResponseError`

Raised when the response is not JSON or not the shape expected.

Example:

```python
from pythea import TheaClient
from pythea.errors import TheaHTTPError, TheaTimeoutError

try:
    with TheaClient(base_url="http://localhost:8000") as client:
        client.unified_answer(question="hello")
except TheaTimeoutError:
    ...
except TheaHTTPError as e:
    print(e.status_code, e.request_id, e.message)
```

---

## Offline probing utilities (`pythea.offline.qmv`)

### What problem this solves

Many evaluation tasks can be reduced to a binary predicate:

> “Does the model say **YES** (1) or **NO** (0) for this prompt?”

`pythea.offline.qmv` provides:

- A **Bernoulli probe** that estimates `q = P(token == "1")` from **first-token logprobs**.
- Helpers to evaluate that predicate across a **family of permutations** of “exchangeable” blocks
  (hints, evidence chunks, tool outputs, etc.).
- Derived metrics (mean, conservative quantile, JS dispersion certificates, residuals) that are
  useful for robustness / leakage / contamination analyses.

The module is intentionally **model-agnostic**: you supply a backend; the package does not tie you
to a specific LLM provider.

---

### Backend contract

To use `BernoulliProbe` with a real model, your backend object must implement:

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

It must return an object with (at minimum):

- `content: str` — the emitted token/content
- `top_logprobs: list[list[item]]` — first-token top logprob items where each `item` has
  `token` and `logprob` attributes (or dict keys)

Optional (used for token accounting):

- `raw` and/or `usage` fields with `prompt_tokens`/`completion_tokens` (or provider equivalents)

Tip: if you already have an internal backend abstraction, the easiest approach is to write a thin
adapter that conforms to this interface.

This repo includes `DummyBackend` for deterministic tests, and an example OpenAI backend adapter in
[`examples/run_audit_openai.py`](./examples/run_audit_openai.py).

---

### BernoulliProbe

```python
from pythea.offline import qmv

probe = qmv.BernoulliProbe(
    backend=...,              # your backend
    temperature=0.0,
    top_logprobs=20,
    use_logit_bias=True,
)
```

How it works:

- The probe appends a short instruction to the user prompt: “Return exactly one token: 1 or 0.”
- It requests **one token** with `top_logprobs` enabled.
- It then renormalizes probability mass over tokens that look like `0` or `1`.
  (The defaults include common whitespace/newline variants.)

`use_logit_bias=True` attempts to restrict generation to the relevant tokens. This requires
`tiktoken` (`pip install -e ".[offline]"`). If `tiktoken` is not available, the probe falls back
to not using logit bias.

Return type: `ProbeDetails` which contains `q`, the emitted string, optional mass for 0/1 tokens,
and token usage fields when available.

---

### ExchangeablePromptParts

`ExchangeablePromptParts` represents a prompt where some blocks can be re-ordered.

```python
parts = qmv.ExchangeablePromptParts(
    prefix="TASK: Decide if the predicate holds.",
    blocks=[
        "HINT A: ...",
        "HINT B: ...",
        "HINT C: ...",
    ],
    suffix="Return 1 if yes else 0.",
)
```

Use `.render(perm)` to render a specific permutation ordering.

---

### Permutation families

`generate_banded_permutations(n, m, num_bands, seed, include_identity)` builds a deterministic
family of permutations. “Banded” means indices are split into contiguous bands and shuffled within
each band (matching the intended “shuffle within bands” setting).

Configuration:

```python
cfg = qmv.PermutationEvalConfig(
    m=6,              # number of permutations to evaluate
    num_bands=6,      # shuffle-within-bands granularity
    seed=0,
    include_identity=True,
)
```

---

### `evaluate_permutation_family(...)`

This is the most common entry point for QMV-style probing.

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

Important outputs:

- `q_list`: per-permutation probabilities `P("1")`
- `q_bar`: mean probability across permutations
- `q_lo`: conservative lower quantile of `q_list` (controlled by `prior_quantile`)
- `jsd`: generalized Jensen–Shannon divergence across permutations (Bernoulli specialization)
- `js_bound`: a certificate-style bound on dispersion (see `js_dispersion_bound`)
- `mean_abs_dev`: empirical mean absolute deviation around `q_bar`
- `canonical_q` and `canonical_resid_*`: residuals comparing the identity permutation vs the
  mixture (only available when `include_identity=True`)

The `workers` parameter parallelizes probing via a `ThreadPoolExecutor`.

---

### Mixture gating: `evaluate_gate_with_features(...)`

If you have:

- A **posterior** prompt (your “best” prompt / canonical ordering), and
- A family of **skeleton** prompts (permutations, ablations, or other variants)

…then `evaluate_gate_with_features` computes a richer set of diagnostics.

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

Outputs include:

- `p_ref`: probe probability on the posterior prompt
- `q_bar`, `q_lo`: mixture stats across skeletons
- `delta_avgKL`, `delta_refKL`: KL-based discrepancy summaries
- `B2T`, `ISR_ref`: planner-style terms used for gating / decision rules
- `jsd`, `js_bound`, `mean_abs_dev`: robustness/dispersion diagnostics

---

### Leakage curves: `build_leakage_curve(...)`

`build_leakage_curve` sweeps over a schedule of budgets (often “number of hints included”) and
returns `LeakageCurvePoint` objects with:

- `q_bar`, `q_lo`
- `B2T`
- information-budget proxies (`delta_pi_to_mix`, etc.)
- dispersion diagnostics (`js_bound`, etc.)

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

---

### Contamination scoring

If you have “clean” controls where blocks are truly exchangeable, you can fit a null distribution
over a scalar statistic (often a canonical residual), and then score a candidate prompt:

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

### Chain-of-thought length heuristics

The module includes a lightweight helper implementing the rule-of-thumb:

`k ≈ c * sqrt(n) * log(1/ε)`

```python
k = qmv.heuristic_cot_tokens(n_context_tokens=100, epsilon=0.05, c=1.0)
```

If you have empirical observations, you can estimate the constant `c`:

```python
obs = [
    {"n_context_tokens": 100.0, "epsilon": 0.05, "k": 30.0},
]
c_hat = qmv.estimate_cot_c_from_observations(obs)
```

---

## Repository examples

This repo contains a few standalone scripts under `examples/` (not installed as part of the
package), including a prompt-injection audit harness and a real OpenAI backend adapter for
`BernoulliProbe`.

These scripts are intended for **defensive testing** of systems you own or have explicit
permission to test.

---

## Test suite

Unit tests:

```bash
pytest -q tests/unit
```

End-to-end tests (require a reachable Thea service):

```bash
export THEA_E2E=1
export THEA_BASE_URL="https://apim-reasoning-core.azure-api.net/reasoning"

pytest -q -m e2e
```

Optional environment variables used by e2e tests:

- `THEA_BACKEND`
- `THEA_APIM_SUBSCRIPTION_KEY`
- `THEA_OPENAI_API_KEY`
- `THEA_AZURE_OPENAI_API_KEY`, `THEA_AZURE_OPENAI_ENDPOINT`, `THEA_AZURE_OPENAI_API_VERSION`

---

## License

MIT — see [`LICENSE.md`](./LICENSE.md).
