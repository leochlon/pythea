# Hallucination Risk Calculator & Prompt Re-engineering Toolkit (Multi-Provider Support)

**Post-hoc calibration without retraining** for large language models. This toolkit turns a raw prompt into:  
1) a **bounded hallucination risk** using the Expectation-level Decompression Law (EDFL), and  
2) a **decision** to **ANSWER** or **REFUSE** under a target SLA, with transparent math (nats).

## 🎯 Key Features

- **Multi-Provider Support**: Works with OpenAI, Anthropic (Claude), Hugging Face, and Ollama models
- **No Retraining Required**: Pure inference-time calibration
- **Two Deployment Modes**:
  - **Evidence-based:** prompts include *evidence/context*; rolling priors are built by erasing that evidence
  - **Closed-book:** prompts have *no evidence*; rolling priors are built by semantic masking of entities/numbers/titles
- **Mathematically Grounded**: Based on EDFL/B2T/ISR framework from NeurIPS 2024 preprint
- **⚡ NEW: Glass Mode** - 30× faster detection using grammatical symmetry instead of ensemble sampling (1 API call vs 30-42)

---

## Table of Contents
- [Install & Setup](#install--setup)
- [Supported Model Providers](#supported-model-providers)
- [Quick Start Examples](#quick-start-examples)
- [⚡ Glass: Fast Mode (NEW)](#-glass-fast-mode-new)
- [Core Mathematical Framework](#core-mathematical-framework)
- [Understanding System Behavior](#understanding-system-behavior)
- [Two Ways to Build Rolling Priors](#two-ways-to-build-rolling-priors)
- [API Surface](#api-surface)
- [Calibration & Validation](#calibration--validation)
- [Practical Considerations](#practical-considerations)
- [Project Layout](#project-layout)
- [Deployment Options](#deployment-options)

---

## Install & Setup

### Basic Installation

```bash
# Core requirement
pip install --upgrade openai

# For additional providers (optional)
pip install anthropic              # For Claude models
pip install transformers torch     # For local Hugging Face models
pip install ollama                 # For Ollama models
pip install requests               # For HTTP-based backends
```

### API Keys Setup

```bash
# For OpenAI
export OPENAI_API_KEY=sk-...

# For Anthropic (Claude)
export ANTHROPIC_API_KEY=sk-ant-...

# For Hugging Face Inference API
export HF_TOKEN=hf_...
```

---

## Supported Model Providers

The toolkit now supports multiple LLM providers through universal backend adapters:

### 1. OpenAI (Original)
- GPT-4o, GPT-4o-mini, and other Chat Completions models
- Requires `OPENAI_API_KEY`

### 2. Anthropic (Claude)
- Claude 3.5 Sonnet, Claude 3 Opus, and other Claude models
- Requires `anthropic` package and `ANTHROPIC_API_KEY`

### 3. Hugging Face (Three Modes)
- **Local Transformers**: Run models locally with `transformers`
- **TGI Server**: Connect to Text Generation Inference servers
- **Inference API**: Use hosted models via Hugging Face API

### 4. Ollama
- Run any Ollama-supported model locally
- Supports both Python SDK and HTTP API

### 5. OpenRouter (Recommended for Production)
- Single API for 100+ models from OpenAI, Anthropic, Google, Meta, and more
- Automatic fallbacks and load balancing
- Often cheaper than direct API access due to volume aggregation
- Built-in rate limiting and retry logic

---

## Quick Start Examples

### Using OpenAI (Original)

```python
from hallucination_toolkit import OpenAIBackend, OpenAIItem, OpenAIPlanner

backend = OpenAIBackend(model="gpt-4o-mini")
planner = OpenAIPlanner(backend, temperature=0.3)

item = OpenAIItem(
    prompt="Who won the 2019 Nobel Prize in Physics?",
    n_samples=7,
    m=6,
    skeleton_policy="closed_book"
)

metrics = planner.run(
    [item], 
    h_star=0.05,           # Target 5% hallucination max
    isr_threshold=1.0,     # Standard ISR gate
    margin_extra_bits=0.2, # Safety margin
    B_clip=12.0,          # Clipping bound
    clip_mode="one-sided" # Conservative mode
)

for m in metrics:
    print(f"Decision: {'ANSWER' if m.decision_answer else 'REFUSE'}")
    print(f"Risk bound: {m.roh_bound:.3f}")
```

### Using Anthropic (Claude)

```python
from hallucination_toolkit import OpenAIPlanner, OpenAIItem
from htk_backends import AnthropicBackend

# Use Claude instead of GPT
backend = AnthropicBackend(model="claude-3-5-sonnet-latest")
planner = OpenAIPlanner(backend, temperature=0.3)

# Rest of the code remains identical
items = [OpenAIItem(prompt="What is quantum entanglement?", n_samples=7, m=6)]
metrics = planner.run(items, h_star=0.05)
```

### Using Hugging Face (Local)

```python
from hallucination_toolkit import OpenAIPlanner, OpenAIItem
from htk_backends import HuggingFaceBackend

# Run Llama locally
backend = HuggingFaceBackend(
    mode="transformers",
    model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
    device_map="auto"  # or "cuda" or "cpu"
)
planner = OpenAIPlanner(backend, temperature=0.3)

# Same evaluation flow
metrics = planner.run([...], h_star=0.05)
```

### Using Hugging Face (TGI Server)

```python
from htk_backends import HuggingFaceBackend

# Connect to a Text Generation Inference server
backend = HuggingFaceBackend(
    mode="tgi",
    tgi_url="http://localhost:8080"
)
planner = OpenAIPlanner(backend, temperature=0.3)
```

### Using Hugging Face (Inference API)

```python
from htk_backends import HuggingFaceBackend
import os

# Use Hugging Face's hosted models
backend = HuggingFaceBackend(
    mode="inference_api",
    model_id="mistralai/Mistral-7B-Instruct-v0.3",
    hf_token=os.environ["HF_TOKEN"]
)
planner = OpenAIPlanner(backend, temperature=0.3)
```

### Using OpenRouter (Recommended for Multi-Model Testing)

[OpenRouter](https://openrouter.ai/) provides access to 100+ models through a single API, making it ideal for comparing hallucination bounds across providers:

```python
from hallucination_toolkit import OpenAIPlanner, OpenAIItem
from htk_backends import OpenRouterBackend

# Access any model through OpenRouter's unified API
backend = OpenRouterBackend(
    model="openrouter/auto",                  # Auto-selects best available model
    # model="anthropic/claude-3.5-sonnet",    # Or specify exact model
    # api_key="...",                          # Uses OPENROUTER_API_KEY env var if not provided
    http_referer="https://your.app",          # Optional but recommended
    x_title="EDFL Decision Head (prod)",      # Optional app identifier
    providers={"allow": ["anthropic", "google", "openai"]},  # Optional: limit providers
)

planner = OpenAIPlanner(
    backend=backend,
    temperature=0.5,
    max_tokens_decision=8,     # Tiny JSON decision head
    q_floor=None,              # Or set your prior floor
)

items = [OpenAIItem(
    prompt="What is quantum entanglement?", 
    n_samples=3, 
    m=6, 
    skeleton_policy="auto"
)]

metrics = planner.run(
    items, 
    h_star=0.05, 
    isr_threshold=1.0, 
    B_clip=12.0, 
    clip_mode="one-sided"
)

for m in metrics:
    print(f"Decision: {'ANSWER' if m.decision_answer else 'REFUSE'}")
    print(f"ISR: {m.isr:.3f}, RoH bound: {m.roh_bound:.3f}")
```

**Why OpenRouter for this toolkit?**
- Test calibration across many models without managing multiple API keys
- Automatic fallbacks ensure high availability for production deployments
- Cost optimization through intelligent routing
- Perfect for A/B testing different models' hallucination characteristics

### Using Ollama

```python
from htk_backends import OllamaBackend

# Use any Ollama model
backend = OllamaBackend(
    model="llama3.1:8b-instruct",
    host="http://localhost:11434"  # Default Ollama port
)
planner = OpenAIPlanner(backend, temperature=0.3)
```

---

## ⚡ Glass: Fast Mode (NEW)

**30× faster hallucination detection using grammatical symmetry**

Instead of ensemble sampling (30-42 API calls), Glass uses **Chomsky's Universal Grammar** to detect hallucinations in O(1) time with just **1 API call**.

### Quick Start with Glass

```python
from hallbayes import OpenAIBackend
from glass import GlassPlanner, GlassItem

# Initialize backend (any provider)
backend = OpenAIBackend(model="gpt-4o-mini")

# Create Glass planner (drop-in replacement for OpenAIPlanner)
planner = GlassPlanner(backend, temperature=0.3)

# Evaluate items
items = [
    GlassItem(prompt="Who won the 2019 Nobel Prize in Physics?"),
    GlassItem(prompt="What is the capital of France?"),
]

metrics = planner.run(items, h_star=0.05)

for m in metrics:
    print(f"Decision: {'ANSWER' if m.decision_answer else 'REFUSE'}")
    print(f"Symmetry: {m.symmetry_score:.3f} | ISR: {m.isr:.2f} | RoH: {m.roh_bound:.3f}")
```

### Performance Comparison

| Metric | Original EDFL | Glass | Improvement |
|--------|---------------|-------|-------------|
| **API Calls** | 30-42 per query | 1 per query | **30-40×** |
| **Latency** | 15-30 seconds | 0.5-1 second | **30×** |
| **Cost** | ~$0.03 per query | ~$0.001 per query | **30×** |
| **Decision Agreement** | Baseline | 85-90% | Good |

### How It Works

Glass uses **grammatical symmetry** inspired by Chomsky's Universal Grammar:

1. **Extract deep structure** from prompt (entities, relations, predicates)
2. **Get response** from LLM (1 call)
3. **Extract deep structure** from response
4. **Compute symmetry score** - if structures are consistent, response is likely truthful
5. **Map to EDFL metrics** (delta_bar, ISR, RoH) for compatibility

**Example:**
```
Prompt:  "Who won the 2019 Nobel Prize in Physics?"
Response: "James Peebles won the 2019 Nobel Prize"

Deep Structures:
  - Entities: {james peebles, nobel prize, physics}
  - Temporal: {2019}
  - Relations: (peebles, AGENT_ACTION, won)

Symmetry Score: 0.85 → High consistency → ANSWER
```

### When to Use Glass vs Original

**Use Glass when:**
- Speed is critical (production APIs)
- Cost matters (high volume)
- Factual queries (names, dates, places)
- 85-90% agreement is acceptable

**Use Original when:**
- Maximum accuracy required
- Complex reasoning queries
- Research/validation
- Cost/latency not a constraint

### Hybrid Approach (Recommended)

```python
# Fast path with Glass
glass_metrics = glass_planner.run([item])

# Fallback to Original if Glass refuses
if not glass_metrics[0].decision_answer:
    orig_metrics = orig_planner.run([item])
    return orig_metrics[0]

return glass_metrics[0]
```

This gives **30× average speedup** with original quality on uncertain cases.

### Running Benchmarks

```bash
python benchmarks/compare.py
```

See `glass/README.md` for detailed documentation.

---

## Backend Configuration Details

### AnthropicBackend

```python
AnthropicBackend(
    model="claude-3-5-sonnet-latest",  # or any Claude model
    api_key=None,                       # Uses ANTHROPIC_API_KEY env var if None
    request_timeout=60.0
)
```

**Requirements**: `pip install anthropic`

### OpenRouterBackend

```python
OpenRouterBackend(
    model="openrouter/auto",           # Auto-routing or specific model
    api_key=None,                      # Uses OPENROUTER_API_KEY env var
    http_referer="https://your.app",   # Recommended for tracking
    x_title="Your App Name",           # Optional identifier
    providers={"allow": ["anthropic", "google"]},  # Optional filtering
)
```

**Requirements**: `pip install openai` (OpenRouter uses OpenAI-compatible API)

**Available models include**:
- `anthropic/claude-3.5-sonnet`, `openai/gpt-4-turbo`, `google/gemini-pro`
- `meta-llama/llama-3-70b-instruct`, `mistralai/mixtral-8x7b`
- See [OpenRouter models](https://openrouter.ai/models) for full list

### HuggingFaceBackend

The Hugging Face backend supports three operational modes:

#### Mode 1: Local Transformers

```python
HuggingFaceBackend(
    mode="transformers",
    model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
    device_map="auto",           # GPU allocation strategy
    torch_dtype="float16",       # Optional: precision setting
    trust_remote_code=True,      # For custom model code
    model_kwargs={}              # Additional model parameters
)
```

**Requirements**: `pip install transformers torch`

#### Mode 2: TGI Server

```python
HuggingFaceBackend(
    mode="tgi",
    tgi_url="http://localhost:8080",  # Your TGI server URL
    model_id=None                      # Not needed for TGI
)
```

**Requirements**: `pip install requests` and a running TGI server

#### Mode 3: Inference API

```python
HuggingFaceBackend(
    mode="inference_api",
    model_id="mistralai/Mistral-7B-Instruct-v0.3",
    hf_token="hf_..."  # Your Hugging Face token
)
```

**Requirements**: `pip install requests` and a Hugging Face account

### OllamaBackend

```python
OllamaBackend(
    model="llama3.1:8b-instruct",  # Any Ollama model
    host="http://localhost:11434",  # Ollama server URL
    request_timeout=60.0
)
```

**Requirements**: `pip install ollama` (optional) or `pip install requests`, and Ollama installed locally

---

## Comparing Providers

Here's a complete example comparing different providers on the same prompt:

```python
from hallucination_toolkit import OpenAIPlanner, OpenAIItem
from htk_backends import AnthropicBackend, HuggingFaceBackend, OllamaBackend

# Define test prompt
prompt = "What are the main differences between quantum and classical computing?"
item = OpenAIItem(prompt=prompt, n_samples=5, m=6, skeleton_policy="closed_book")

# Test configuration
config = dict(
    h_star=0.05,
    isr_threshold=1.0,
    margin_extra_bits=0.2,
    B_clip=12.0,
    clip_mode="one-sided"
)

# Compare providers
providers = {
    "GPT-4o-mini": OpenAIBackend(model="gpt-4o-mini"),
    "Claude-3.5": AnthropicBackend(model="claude-3-5-sonnet-latest"),
    "Llama-3.1": HuggingFaceBackend(mode="transformers", model_id="meta-llama/Meta-Llama-3.1-8B-Instruct"),
    "Ollama": OllamaBackend(model="llama3.1:8b-instruct")
}

results = {}
for name, backend in providers.items():
    try:
        planner = OpenAIPlanner(backend, temperature=0.3)
        metrics = planner.run([item], **config)
        results[name] = metrics[0]
        print(f"{name}: {'ANSWER' if metrics[0].decision_answer else 'REFUSE'} (RoH={metrics[0].roh_bound:.3f})")
    except Exception as e:
        print(f"{name}: Error - {e}")
```

---

## Core Mathematical Framework

### The EDFL Principle

Let the binary event $\mathcal{A}$ be the thing you want to guarantee (e.g., **Answer** in decision mode, or **Correct** for factual accuracy).  
Build an ensemble of **content-weakened prompts** (the *rolling priors*) $\{S_k\}_{k=1}^m$. For the realized label $y$, estimate:

- **Information budget:**  
  $$\bar{\Delta} = \tfrac{1}{m}\sum_k \mathrm{clip}_+(\log P(y) - \log S_k(y), B)$$
  (one-sided clipping; default $B=12$ nats to prevent outliers while maintaining conservative bounds).

- **Prior masses:** $q_k = S_k(\mathcal{A})$, with:
  - $\bar{q}=\tfrac{1}{m}\sum_k q_k$ (average prior for EDFL bound)
  - $q_{\text{lo}}=\min_k q_k$ (worst-case prior for SLA gating)

By EDFL, the achievable reliability is bounded by:  
$$\bar{\Delta} \ge \mathrm{KL}(\mathrm{Ber}(p) \| \mathrm{Ber}(\bar{q})) \Rightarrow p\le p_{\max}(\bar{\Delta},\bar{q})$$

Thus the **hallucination risk** (error) is bounded by $\overline{\mathrm{RoH}} \le 1 - p_{\max}$.

### Decision Rule (SLA Gating)

For target hallucination rate $h^*$:
- **Bits-to-Trust:** $\mathrm{B2T} = \mathrm{KL}(\mathrm{Ber}(1-h^*) \| \mathrm{Ber}(q_{\text{lo}}))$
- **Information Sufficiency Ratio:** $\mathrm{ISR} = \bar{\Delta}/\mathrm{B2T}$
- **ANSWER** iff $\mathrm{ISR}\ge 1$ *and* $\bar{\Delta} \ge \mathrm{B2T} + \text{margin}$ (default `margin≈0.2` nats)

> **Why two priors?** The gate uses **worst-case** $q_{\text{lo}}$ for strict SLA compliance. The RoH bound uses **average** $\bar{q}$ per EDFL theory. This dual approach ensures conservative safety while providing realistic risk bounds.

---

## Understanding System Behavior

### Expected Behavioral Patterns

The toolkit exhibits different behaviors across query types, which is **mathematically consistent** with the framework:

#### Simple Arithmetic Queries
**Observation:** May abstain despite apparent simplicity  
**Explanation:**
- Models often attempt answers even with masked numbers (pattern recognition)
- This yields **low information lift** $\bar{\Delta} \approx 0$ between full prompt and skeletons
- Despite potentially low EDFL risk bound, worst-case prior gate triggers **abstention** (ISR < 1)

#### Named-Entity Factoids
**Observation:** Generally answered with confidence  
**Explanation:**
- Masking entities/dates substantially reduces answer propensity in skeletons
- Restoring these yields **large** $\bar{\Delta}$ that clears B2T threshold
- System **answers** with tight EDFL risk bound

**This is not a bug but a feature**: The framework prioritizes safety through worst-case guarantees while providing realistic average-case bounds.

### Provider-Specific Considerations

Different model providers may exhibit varying behaviors:

- **OpenAI/Anthropic**: Generally produce clean JSON decisions with high compliance
- **Hugging Face (Local)**: May require instruction-tuned variants for best results
- **Ollama**: Performance depends on the specific model; instruction-tuned models recommended
- **Base Models**: May need adjusted prompting or higher sampling for stable priors

---

## Two Ways to Build Rolling Priors

### 1) Evidence-based (when you have context)

- **Prompt** contains a field like `Evidence:` (or JSON keys)
- **Skeletons** erase the evidence content but preserve structure and roles; then permute blocks deterministically (seeded)
- **Decision head**: "Answer only if the provided evidence is sufficient; otherwise refuse."

**Example with Multiple Providers**
```python
from hallucination_toolkit import OpenAIItem, OpenAIPlanner
from htk_backends import AnthropicBackend

backend = AnthropicBackend(model="claude-3-5-sonnet-latest")
prompt = """Task: Answer strictly based on the evidence below.
Question: Who won the Nobel Prize in Physics in 2019?
Evidence:
- Nobel Prize press release (2019): James Peebles (1/2); Michel Mayor & Didier Queloz (1/2).
Constraints: If evidence is insufficient or conflicting, refuse.
"""

item = OpenAIItem(
    prompt=prompt, 
    n_samples=5, 
    m=6, 
    fields_to_erase=["Evidence"], 
    skeleton_policy="auto"
)

planner = OpenAIPlanner(backend, temperature=0.3)
metrics = planner.run([item], h_star=0.05, isr_threshold=1.0)
```

### 2) Closed-book (no evidence)

- **Prompt** has no evidence
- **Skeletons** apply **semantic masking** of:
  - Multi-word proper nouns (e.g., "James Peebles" → "[…]")
  - Years (e.g., "2019" → "[…]")
  - Numbers (e.g., "3.14" → "[…]")
  - Quoted spans (e.g., '"Nobel Prize"' → "[…]")
- **Masking strengths**: Progressive levels (0.25, 0.35, 0.5, 0.65, 0.8, 0.9) across skeleton ensemble

**Example with Multiple Providers**
```python
from hallucination_toolkit import OpenAIItem, OpenAIPlanner
from htk_backends import OllamaBackend

backend = OllamaBackend(model="mixtral:8x7b-instruct")
item = OpenAIItem(
    prompt="Who won the 2019 Nobel Prize in Physics?",
    n_samples=7,
    m=6,
    skeleton_policy="closed_book"
)

planner = OpenAIPlanner(backend, temperature=0.3)
metrics = planner.run([item], h_star=0.05)
```

---

## API Surface

### Core Classes

- `OpenAIBackend(model, api_key=None)` – Original OpenAI wrapper
- `AnthropicBackend(model, api_key=None)` – Anthropic Claude adapter
- `HuggingFaceBackend(mode, model_id, ...)` – Hugging Face adapter (3 modes)
- `OllamaBackend(model, host)` – Ollama local model adapter
- `OpenAIItem(prompt, n_samples=5, m=6, fields_to_erase=None, skeleton_policy="auto")` – One evaluation item
- `OpenAIPlanner(backend, temperature=0.5, q_floor=None)` – Runs evaluation (works with any backend):
  - `run(items, h_star, isr_threshold, margin_extra_bits, B_clip=12.0, clip_mode="one-sided") -> List[ItemMetrics]`
  - `aggregate(items, metrics, alpha=0.05, h_star, ...) -> AggregateReport`

### Helper Functions

- `make_sla_certificate(report, model_name)` – Creates formal SLA certificate
- `save_sla_certificate_json(cert, path)` – Exports certificate for audit
- `generate_answer_if_allowed(backend, item, metric)` – Only emits answer if decision was ANSWER

### ItemMetrics Fields

Every `ItemMetrics` includes:
- `delta_bar`: Information budget (nats)
- `q_conservative`: Worst-case prior $q_{\text{lo}}$
- `q_avg`: Average prior $\bar{q}$
- `b2t`: Bits-to-Trust requirement
- `isr`: Information Sufficiency Ratio
- `roh_bound`: EDFL hallucination risk bound
- `decision_answer`: Boolean decision
- `rationale`: Human-readable explanation
- `meta`: Dict with `q_list`, `S_list_y`, `P_y`, `closed_book`, etc.

---

## Calibration & Validation

### Validation Set Calibration

On a labeled validation set:

1. **Sweep the margin** parameter from 0 to 1 nats
2. For each margin, compute:
   - Empirical hallucination rate among answered items
   - Wilson upper bound at 95% confidence
3. **Select smallest margin** where Wilson upper bound ≤ target $h^*$ (e.g., 5%)
4. **Freeze policy**: $(h^*, \tau, \text{margin}, B, \text{clip\_mode}, m, r, \text{skeleton\_policy})$

### Portfolio Reporting

The toolkit provides comprehensive metrics:
- Answer/abstention rates
- Empirical hallucination rate + Wilson bound
- Distribution of per-item EDFL RoH bounds
- Worst-case and median risk bounds
- Complete audit trail

---

## Practical Considerations

### Choosing the Right Provider

| Provider | Best For | Considerations |
|----------|----------|----------------|
| **OpenAI** | Production deployment, consistent JSON | Requires API key, costs per token |
| **Anthropic** | High-quality reasoning, safety-critical | Requires API key, may have rate limits |
| **OpenRouter** | Multi-model testing, cost optimization | Single API for 100+ models, automatic fallbacks |
| **HuggingFace (Local)** | Full control, no API costs | Requires GPU, setup complexity |
| **HuggingFace (TGI)** | Team deployments, caching | Requires server setup |
| **HuggingFace (API)** | Quick prototyping | Rate limits, requires HF token |
| **Ollama** | Local experimentation | Easy setup, model quality varies |

### Performance Characteristics by Provider

| Provider | Latency per Item | Cost | Setup Complexity |
|----------|-----------------|------|------------------|
| OpenAI | 2-5 seconds | ~$0.01-0.03 | Low |
| Anthropic | 3-6 seconds | ~$0.02-0.05 | Low |
| HF Local | 1-10 seconds | Free (GPU cost) | Medium-High |
| HF TGI | 1-3 seconds | Server costs | High |
| HF API | 3-8 seconds | Free tier/paid | Low |
| Ollama | 2-15 seconds | Free (local) | Low |

### Common Issues & Solutions

#### Issue: Non-JSON responses from local models
**Solution**: Use instruction-tuned model variants (e.g., `-Instruct` suffixes)

#### Issue: Different risk bounds across providers
**Expected**: Models have different knowledge/calibration; the framework adapts accordingly

#### Issue: Timeouts with local models
**Solution**: Increase `request_timeout` parameter or reduce batch size

---

## Project Layout

```
.
├── app/                    # Application entry points
│   ├── web/web_app.py     # Streamlit UI
│   ├── cli/frontend.py    # Interactive CLI
│   ├── examples/          # Example scripts
│   └── launcher/entry.py  # Unified launcher
├── hallbayes/             # Core modules
│   ├── hallucination_toolkit.py  # Main toolkit (EDFL)
│   ├── htk_backends.py          # Universal backend adapters
│   └── build_offline_backend.sh
├── glass/                 # ⚡ Fast mode (NEW)
│   ├── grammatical_mapper.py    # Deep structure extraction
│   ├── planner.py              # GlassPlanner (O(1) detection)
│   └── README.md               # Glass documentation
├── benchmarks/            # Performance comparisons
│   └── compare.py         # Original vs Glass benchmark
├── electron/              # Desktop wrapper
├── launch/                # Platform launchers
├── release/              # Packaged artifacts
├── bin/                  # Offline backend binary
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Deployment Options

### 1. Direct Python Usage (Any Provider)

```python
from hallbayes import OpenAIPlanner, OpenAIItem, make_sla_certificate
from hallbayes.htk_backends import AnthropicBackend  # or any other backend

# Choose your provider
backend = AnthropicBackend(model="claude-3-5-sonnet-latest")

# Configure and run
items = [OpenAIItem(prompt="...", n_samples=7, m=6)]
planner = OpenAIPlanner(backend, temperature=0.3)
metrics = planner.run(items, h_star=0.05)

# Generate SLA certificate
report = planner.aggregate(items, metrics)
cert = make_sla_certificate(report, model_name="Claude-3.5-Sonnet")
save_sla_certificate_json(cert, "sla.json")
```

### 2. Web Interface (Streamlit)

```bash
streamlit run app/web/web_app.py
```

### 3. Batch Processing with Multiple Providers

```python
from hallucination_toolkit import OpenAIPlanner, OpenAIItem
from htk_backends import AnthropicBackend, OllamaBackend
import json

# Load prompts
with open("prompts.json") as f:
    prompts = json.load(f)

# Setup providers
providers = {
    "claude": AnthropicBackend(model="claude-3-5-sonnet-latest"),
    "llama": OllamaBackend(model="llama3.1:8b-instruct")
}

# Process with each provider
results = {}
for name, backend in providers.items():
    planner = OpenAIPlanner(backend, temperature=0.3)
    items = [OpenAIItem(prompt=p, n_samples=5, m=6) for p in prompts]
    metrics = planner.run(items, h_star=0.05)
    results[name] = planner.aggregate(items, metrics)
```

---

## Quick Migration Guide

If you're already using the toolkit with OpenAI, here's how to try other providers:

```python
# Original (OpenAI only)
from hallucination_toolkit import OpenAIBackend
backend = OpenAIBackend(model="gpt-4o-mini")

# New (Any provider) - just change these two lines:
from htk_backends import AnthropicBackend  # or HuggingFaceBackend, OllamaBackend
backend = AnthropicBackend(model="claude-3-5-sonnet-latest")

# Everything else stays exactly the same!
planner = OpenAIPlanner(backend, temperature=0.3)
# ... rest of your code unchanged
```

---

Based on the Paper: Predictable Compression Failures: Why Language Models Actually Hallucinate - https://arxiv.org/abs/2509.11208
## License

This project is licensed under the MIT License – see the LICENSE file for details.

## Attribution

Developed by Hassana Labs (https://hassana.io).
