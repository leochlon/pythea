# Thea API Client

Lightweight HTTP client for the **Thea Mini Reasoning API**.

---

## Installation

```bash
pip install -e .
```

Runtime dependencies: `httpx>=0.24.0`

---

## TheaClient (synchronous)

```python
from pythea import TheaClient

with TheaClient(base_url="https://apim-reasoning-core.azure-api.net/reasoning") as client:
    print(client.healthz())
    
    resp = client.unified_answer(
        question="What is 2+2?",
        backend="aoai-pool",
        interpretability=True,
        m=6,
    )
    print(resp.get("status"), resp.get("answer"), resp.get("reason_code"))
```

### Constructor

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

| Parameter | Description |
|-----------|-------------|
| `base_url` | Root of the service (required) |
| `apim_subscription_key` | Azure API Management key |
| `extra_headers` | Default headers for every request |
| `timeout_s` | Request timeout in seconds |
| `verify` | TLS verification (`True`/`False` or CA bundle path) |
| `http` | Custom `httpx.Client` (you own its lifecycle) |
| `transport` | DI hook for testing (e.g., `httpx.MockTransport`) |

### Methods

#### `healthz()`

```python
status = client.healthz()  # GET /healthz
```

#### `unified_answer(...)`

```python
resp = client.unified_answer(
    question="What is 2+2?",
    backend="aoai-pool",
    interpretability=True,
    m=6,
)
```

Response shape: `status` ("answered" or "abstained"), optional `answer`, and `reason_code`.

---

## AsyncTheaClient (asynchronous)

```python
import asyncio
from pythea import AsyncTheaClient

async def main() -> None:
    async with AsyncTheaClient(base_url="https://...") as client:
        print(await client.healthz())
        resp = await client.unified_answer(
            question="What is 2+2?",
            backend="aoai-pool",
            m=6,
        )
        print(resp.get("answer"))

asyncio.run(main())
```

Mirrors the sync client. Use `await client.aclose()` or `async with` for lifecycle management.

---

## Closed-Book Guard Controls

```python
resp = client.unified_answer(
    question="Where did Ada Lovelace work and when?",
    backend="aoai-pool",
    interpretability=True,
    m=8,
    cbg_tree_depth=2,            # 1 = flat; 2 = expand into leaves
    cbg_tree_expand_per_node=3,  # max children per branch
    cbg_tree_max_expansions=6,   # overall cap
)
```

---

## EvidenceGuard Knobs

```python
resp = client.unified_answer(
    question="Is the claim supported by the evidence?",
    evidence="1) ...\n2) ...",
    backend="aoai-pool",
    hstar=0.05,          # h* strictness (target = 1 - h*)
    prior_quantile=0.05, # quantile for conservative prior q_lo
    top_logprobs=10,     # top-k for 1-token probe
    temperature=0.0,
)
```

---

## Azure API Management (APIM)

```python
client = TheaClient(
    base_url="https://apim-reasoning-core.azure-api.net/reasoning",
    apim_subscription_key="...",
)
```

---

## Error Handling

All exceptions derive from `pythea.errors.TheaError`.

| Exception | When |
|-----------|------|
| `TheaHTTPError` | Non-2xx response |
| `TheaTimeoutError` | Request timed out |
| `TheaResponseError` | Response not JSON or wrong shape |

```python
from pythea import TheaClient
from pythea.errors import TheaHTTPError

try:
    with TheaClient(base_url="...") as client:
        client.unified_answer(question="hello")
except TheaHTTPError as e:
    print(e.status_code, e.request_id, e.message)
```

`TheaHTTPError` fields:
- `status_code: int`
- `message: str`
- `response_text: str | None`
- `request_id: str | None` (from `X-Request-ID` header)

---

## Types

Request/response typed in `pythea.types`:
- `UnifiedAnswerRequest`
- `UnifiedAnswerResponse`
- `BackendCredentials`
