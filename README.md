# pythea

`pythea` is a small, typed Python package with two goals:

1. **A lightweight client** for the **Thea Mini Reasoning API** (currently `GET /healthz` and `POST /api/unified-answer`).
2. **Offline, model-agnostic evaluation utilities** for permutation-mixture / QMV-style probing via a **Bernoulli (0/1) first-token logprob probe**.

This repo also includes a few optional security-testing scripts under [`examples/`](./examples) (not installed as part of the package).

- **Docs:** see [`DOCUMENTATION.md`](./DOCUMENTATION.md)
- **License:** see [`LICENSE.md`](./LICENSE.md)

## Requirements

- Python **3.9+**
- `httpx>=0.24.0`

## Install

```bash
git clone https://github.com/leochlon/hallbayes.git
cd hallbayes
pip install -e .
```

Dev/test dependencies:

```bash
pip install -e ".[dev]"
```

Optional extra used by the offline probe for OpenAI-style `logit_bias` tokenization:

```bash
pip install -e ".[offline]"
```

## API client quickstart

### Synchronous

```python
from pythea import TheaClient

with TheaClient(base_url="https://apim-reasoning-core.azure-api.net/reasoning") as client:
    print(client.healthz())

    resp = client.unified_answer(
        question="What is 2+2?",
        backend="aoai-pool",          # depends on your deployment
        interpretability=True,
    )

    print(resp.get("decision"), resp.get("picked"))
```

### Asynchronous

```python
import asyncio

from pythea import AsyncTheaClient


async def main() -> None:
    async with AsyncTheaClient(base_url="https://apim-reasoning-core.azure-api.net/reasoning") as client:
        print(await client.healthz())
        # Explicitly pass a backend; e.g. "aoai-pool", "openai", or "azure"
        resp = await client.unified_answer(question="What is 2+2?", backend="aoai-pool")
        print(resp.get("decision"), resp.get("picked"))


asyncio.run(main())
```

### Azure API Management (APIM)

If your Thea service is fronted by Azure APIM, pass the subscription key:

```python
from pythea import TheaClient

client = TheaClient(
    base_url="https://apim-reasoning-core.azure-api.net/reasoning",
    apim_subscription_key="...",
    # apim_subscription_key_header defaults to "Ocp-Apim-Subscription-Key"
)
```

### Error handling

The client raises a small set of exceptions:

- `TheaHTTPError`: non-2xx response (includes `status_code` and optional `request_id`)
- `TheaTimeoutError`: request timed out
- `TheaResponseError`: response parsing/shape problems

```python
from pythea import TheaClient
from pythea.errors import TheaHTTPError

try:
    with TheaClient(base_url="https://apim-reasoning-core.azure-api.net/reasoning") as client:
        client.unified_answer(question="hello")
except TheaHTTPError as e:
    print(e.status_code, e.request_id, e.message)
```

## Offline utilities quickstart (`pythea.offline.qmv`)

The offline tools are **model-agnostic**. You provide a backend that can:

- Generate exactly **one token**
- Provide **first-token logprobs** (so we can implement a `0/1` Bernoulli probe)

For deterministic local tests, use the included `DummyBackend`:

```python
from pythea.offline import qmv

# Deterministic "model": always returns q=P("1") = 0.8
backend = qmv.DummyBackend(lambda _prompt: 0.8)
probe = qmv.BernoulliProbe(backend=backend, use_logit_bias=False)

parts = qmv.ExchangeablePromptParts(
    prefix="TASK",
    blocks=["HINT A", "HINT B", "HINT C"],
    suffix="Return 1 if predicate holds else 0.",
)

res = qmv.evaluate_permutation_family(
    probe=probe,
    parts=parts,
    cfg=qmv.PermutationEvalConfig(m=6, num_bands=2, seed=0, include_identity=True),
    prior_quantile=0.2,
)

print(res.q_bar, res.q_lo, res.js_bound)
```

More details (metrics, backend contract, leakage curves, contamination scoring, etc.) are in
[`DOCUMENTATION.md`](./DOCUMENTATION.md).

## Running tests

Unit tests (offline + client):

```bash
pytest -q tests/unit
```

End-to-end tests require a reachable Thea service and are gated behind `THEA_E2E=1`:

```bash
export THEA_E2E=1
export THEA_BASE_URL="https://apim-reasoning-core.azure-api.net/reasoning"
export THEA_BACKEND="aoai-pool"  # optional

pytest -q -m e2e
```

## License

MIT — see [`LICENSE.md`](./LICENSE.md).
