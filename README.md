# pythea

**LLM reliability research from [Hassana Labs](https://hassana.io).**

Three tools, one goal: catch models that know but don't use.

---

## 🍓 Strawberry: Procedural Hallucination Toolkit

Ask Claude to count the r's in "strawberry." It writes "s-t-r-a-w-b-e-r-r-y," identifies each r, gets to 3. Then outputs "2."

The model didn't lack information. The answer was right there—in text it generated moments earlier. The computation worked. The routing failed.

This toolkit detects those failures mathematically.

```bash
pip install pythea
python -m strawberry.factual_recall \
  --question "Which US senators from Minnesota graduated from Princeton" \
  --out report.json
```

**What it catches:**
- RAG that retrieves but doesn't read
- Chain-of-thought that cites steps it ignored
- Self-verification that validates without checking
- Citation confabulation (decorative sources)

**How:** Scrub the cited evidence, measure confidence change. No change? The model was confabulating.

[→ Full docs](./strawberry/README.md)

---

## 🔌 Thea API Client

Lightweight client for the Thea Mini Reasoning API.

```python
from pythea import TheaClient

with TheaClient(base_url="https://...") as client:
    resp = client.unified_answer(
        question="What is 2+2?",
        backend="aoai-pool",
        m=6,
    )
    print(resp.get("answer"))
```

[→ Full docs](./docs/CLIENT.md)

---

## 📊 Offline QMV Probing

Model-agnostic permutation-mixture evaluation via Bernoulli first-token logprob probes.

```python
from pythea.offline import qmv

res = qmv.evaluate_permutation_family(
    probe=probe,
    parts=parts,
    cfg=qmv.PermutationEvalConfig(m=6, num_bands=2, seed=0),
)
print(res.q_bar, res.q_lo, res.js_bound)
```

[→ Full docs](./docs/QMV.md)

---

## Install

```bash
git clone https://github.com/leochlon/pythea.git
cd pythea
pip install -e .
```

**Extras:**
```bash
pip install -e ".[dev]"      # tests
pip install -e ".[offline]"  # tiktoken for logit bias
pip install -e ".[vllm]"     # local inference
```

---

## Repo layout

```
pythea/
├── strawberry/          # Procedural hallucination toolkit
│   ├── README.md
│   └── src/
├── src/pythea/          # Thea client + QMV probing
├── docs/                # Detailed documentation
├── examples/
├── tests/
└── benchmarks/
```

---

## Citation

```bibtex
@article{hassanalabs2026procedural,
  title={An Information-Theoretic and Causal Theory of Procedural Hallucinations},
  author={{Hassana Labs}},
  journal={arXiv preprint},
  year={2026}
}
```

---

## License

MIT — see [LICENSE.md](./LICENSE.md)