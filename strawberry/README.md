# Binding / Routing Toolkit

**Detect procedural hallucinations before they ship.**

LLMs don't just hallucinate because they don't know things. They hallucinate because they *know but don't use*—the answer is in the context, the model just doesn't route to it.

This toolkit detects those failures mathematically, using only API outputs and logprobs. No retraining. No activation access required.

---

## What it catches

- **RAG that retrieves but doesn't read** — documents in context, answer from vibes
- **Chain-of-thought that cites steps it ignored** — shows work ≠ uses work  
- **Self-verification that validates without checking** — citations as decoration
- **Binding failures** — model writes "3", outputs "2" (strawberry problem)

The core insight: scrub the cited evidence, measure confidence change. No change? The model was confabulating.

---

## Install

```bash
cd strawberry_toolkit
pip install -e .
```

**With vLLM support:**
```bash
pip install -e ".[vllm]"
```

---

## Quick start

### 1. Factual recall audit (catch citation confabulation)

The flagship use case: detect when a model's citations don't actually support its answer.

```bash
export OPENAI_API_KEY=...

python -m strawberry.factual_recall \
  --backend openai \
  --generator_model gpt-4o-mini \
  --question "Which US senators from Minnesota graduated from Princeton" \
  --out report.json \
  --pretty
```

**What it does:**

1. Model generates evidence spans from memory
2. Model answers using *only* those spans (with citations)
3. Toolkit scrubs cited spans → measures entailment probability drop
4. Flags if **ObservedBits < RequiredBits** (citations don't justify confidence)

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--generator_model` | required | Model for evidence + answering |
| `--verifier_model` | generator | Model for entailment checks |
| `--max_spans` | 4 | Max evidence spans to generate |
| `--budget_target` | model's confidence | Target reliability τ |
| `--null_answer_check` | off | Re-answer after scrubbing, compare |
| `--placeholder` | `[REDACTED]` | Scrub replacement text |

**Output interpretation:**

- `flagged: true` + `under_budget` → citations don't support the claim
- `flagged: true` + `evidence_insensitive_answer` → same answer with/without evidence (confabulation)

---

### 2. Synthetic binding evaluation

Test how well models bind values to variables across distance.

```bash
strawberry run \
  --backend openai \
  --model gpt-4o-2024-08-06 \
  --n 200 \
  --M 10 \
  --distance 512 \
  --query FIRST \
  --null SCRUB_FIRST \
  --out results.json
```

**What the knobs mean:**

| Flag | Description |
|------|-------------|
| `--M` | Number of candidates (e.g., 10-way choice) |
| `--distance` | Filler tokens between binding and query |
| `--query` | `FIRST` or `LAST` — where to retrieve from |
| `--null` | Intervention for pseudo-prior (`SCRUB_FIRST`, etc.) |

---

### 3. Chain-of-thought / trace auditing

Detect when reasoning traces don't support their conclusions.

```bash
strawberry cot \
  --backend openai \
  --generator_model gpt-4o-mini \
  --verifier_model gpt-4o-mini \
  --synthetic \
  --M 10 \
  --distance 256 \
  --query FIRST
```

**With deliberation sweep** (measure answer drift vs reasoning length):

```bash
strawberry cot \
  --backend openai \
  --generator_model gpt-4o-mini \
  --verifier_model gpt-4o-mini \
  --choices auto_trace \
  --auto_choice_mode codes \
  --deliberation_sweep 0 2 4 8 12 \
  --deliberation_kind trace
```

---

## Python API

```python
from strawberry.tasks import generate_dataset
from strawberry.eval import run_eval

# Generate synthetic binding task
items = generate_dataset(n=50, distance_tokens=512, M=10, query_rule="FIRST", seed=0)

# Run evaluation with null intervention
results = run_eval(items=items, model="gpt-4o-2024-08-06", null_mode="SCRUB_FIRST")
print(results.summary)
```

---

## The theory (TL;DR)

**Stage 2A failure:** Model doesn't recognise it should answer from candidates.  
**Stage 2B failure:** Model tries to answer but routes to wrong value.

Stage 2B dominates. The model knows what to do—it grabs the wrong thing.

**Detection method:**

1. Compute `p1` = P(correct | full context)
2. Scrub evidence → compute `p0` = P(correct | scrubbed)  
3. **ObservedBits** = KL(Ber(p1) || Ber(p0))
4. **RequiredBits** = KL(Ber(τ) || Ber(p0)) where τ = target reliability

If ObservedBits < RequiredBits → evidence doesn't justify confidence → flag.

See the paper for proofs and empirical validation across Qwen, Llama, and Gemma.

---

## Repo layout

```
src/strawberry/
├── cli.py              # strawberry CLI
├── tasks.py            # Synthetic prompt generation
├── eval.py             # Evaluation runner + null interventions
├── stage_ab.py         # Stage 2A/2B logprob analysis
├── metrics.py          # KL, bits-to-trust, Fano bounds
├── trace_budget.py     # Scrub + p0/p1 + budget calculations
├── cot_detector.py     # Trace hallucination detection
├── factual_recall.py   # Closed-system factual recall auditor
├── backend.py          # OpenAI/vLLM backend abstraction
└── openai_backend.py   # OpenAI Responses API implementation
```

---

## vLLM usage

```bash
python -m strawberry.factual_recall \
  --backend vllm \
  --generator_model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --question "..." \
  --vllm_tensor_parallel 2 \
  --out report.json
```

**vLLM flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--vllm_tensor_parallel` | 1 | GPU parallelism |
| `--vllm_max_model_len` | auto | Context length cap |
| `--vllm_gpu_memory_utilization` | 0.90 | VRAM fraction |
| `--vllm_dtype` | bfloat16 | Model dtype |

---

## Troubleshooting

- **OpenAI:** Set `OPENAI_API_KEY` or pass `--api_key`
- **OpenAI-compatible servers:** Pass `--base_url`
- **vLLM OOM:** Reduce `--vllm_gpu_memory_utilization` or increase `--vllm_tensor_parallel`

---

## Citation

```bibtex
@article{chlon2026procedural,
  title={An Information-Theoretic and Causal Theory of Procedural Hallucinations},
  author={Chlon, Leon},
  institution={Hassana Labs},
  journal={arXiv preprint},
  year={2026}
}
```

---

## License

MIT
