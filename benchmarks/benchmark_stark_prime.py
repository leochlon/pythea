#!/usr/bin/env python3
"""STaRK-Prime benchmark with evidence guard + OpenAI API.

Evaluates graph retrieval quality on STaRK-Prime with optional:
  - LLM-based reranking of retrieved candidates
  - Evidence sufficiency guard (pythea hallucination detector)

Performance optimizations:
  - Parallel graph retrieval across CPU processes (PCST/graph ops are CPU-bound)
  - Async LLM calls with configurable concurrency
  - Overlapped guard evaluation during retrieval

Usage:
  python benchmark_stark_prime.py --data-dir /path/to/stark_data \\
    --retrieval-workers 24 --guard-workers 12 --llm-rerank

Set OPENAI_API_KEY environment variable or use --openai-api-key flag.
"""

import sys
from pathlib import Path
from typing import Optional

# Add Downloads/PyG for local imports
sys.path.insert(0, str(Path.home() / "Downloads" / "PyG"))
# Add benchmarks dir itself for local imports
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import asyncio
import json
import os
import queue
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import aiohttp
import torch
from tqdm import tqdm

from data import ensure_stark_data


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_skb(data_dir: Path):
    import pickle

    d = data_dir / "processed" if (data_dir / "processed").exists() else data_dir
    with open(d / "node_info.pkl", "rb") as f:
        node_info = pickle.load(f)
    edge_index = torch.load(d / "edge_index.pt", weights_only=False)
    doc_info = {}
    if (d / "doc_info.pkl").exists():
        with open(d / "doc_info.pkl", "rb") as f:
            doc_info = pickle.load(f)
    edge_type = torch.load(d / "edge_type.pt", weights_only=False) if (d / "edge_type.pt").exists() else None
    # Also check edge_types.pt (alternative naming)
    if edge_type is None and (d / "edge_types.pt").exists():
        edge_type = torch.load(d / "edge_types.pt", weights_only=False)
    return node_info, edge_index, doc_info, edge_type


def load_qa(data_dir: Path):
    import csv

    splits = torch.load(data_dir / "split.pt", weights_only=False)
    split_map = {int(i): s for s, ids in splits.items() for i in (ids.tolist() if hasattr(ids, "tolist") else ids)}

    qa_path = data_dir / "stark_qa.csv"
    records = []
    with open(qa_path) as f:
        for idx, row in enumerate(csv.DictReader(f)):
            ans = row.get("answer_ids", "").strip("[]")
            records.append(
                {
                    "id": int(row.get("id", idx)),
                    "question": row.get("query", ""),
                    "answer_ids": [int(x) for x in ans.split(",") if x.strip()],
                    "split": split_map.get(idx, "unknown"),
                }
            )
    return records


def load_emb_dict(path: Path, n: int) -> torch.Tensor:
    d = torch.load(path, weights_only=False)
    dim = next(iter(d.values())).shape[-1]
    out = torch.zeros(n, dim)
    for k, v in d.items():
        if k < n:
            out[k] = v.squeeze()
    return out / out.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def load_query_emb(path: Path, ids: list) -> torch.Tensor:
    d = torch.load(path, weights_only=False)
    dim = next(iter(d.values())).shape[-1]
    out = torch.stack([d[i].squeeze() if i in d else torch.zeros(dim) for i in ids])
    return out / out.norm(dim=-1, keepdim=True).clamp(min=1e-12)


# ---------------------------------------------------------------------------
# Graph Pack
# ---------------------------------------------------------------------------


def build_pack(node_info, edge_index, doc_info, edge_type, node_emb):
    from utils import GraphPack

    n = len(node_info)
    node_text = []
    for i in range(n):
        info = node_info.get(i, {})
        name = info.get("name", f"Node_{i}")
        node_type = info.get("type", "unknown")

        # Get description from details (like data.py does)
        desc = ""
        details = info.get("details", {})
        if isinstance(details, dict):
            for key in ["summary", "description"]:
                val = details.get(key)
                if val and isinstance(val, str):
                    desc = val[:400]
                    break

        # Fallback to doc_info if no details
        if not desc:
            doc = doc_info.get(i, {})
            if isinstance(doc, dict):
                desc = doc.get("doc", "")[:400]
            elif isinstance(doc, str):
                desc = doc[:400]

        if desc:
            node_text.append(f"name: {name}, type: {node_type}, description: {desc}"[:512])
        else:
            node_text.append(f"name: {name}, type: {node_type}")

    edge_text = None
    if edge_type is not None:
        rels = [
            "ppi",
            "carrier",
            "enzyme",
            "target",
            "transporter",
            "contraindication",
            "indication",
            "off-label use",
            "synergistic interaction",
            "associated with",
            "parent-child",
            "phenotype absent",
            "phenotype present",
            "side effect",
            "interacts with",
            "linked to",
            "expression present",
            "expression absent",
        ]
        edge_text = [
            rels[e].upper().replace(" ", "_") if 0 <= e < len(rels) else "RELATED_TO" for e in edge_type.tolist()
        ]

    return GraphPack(edge_index=edge_index, node_emb=node_emb, node_text=node_text, edge_text=edge_text)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(preds, answers):
    n = len(preds)

    def hits(k):
        return 100 * sum(any(a in p[:k] for a in ans) for p, ans in zip(preds, answers)) / n

    def recall(k):
        return (
            100
            * sum(sum(a in p[:k] for a in ans) / len(ans) if ans else 0 for p, ans in zip(preds, answers))
            / n
        )

    def mrr():
        return 100 * sum(next((1 / (i + 1) for i, x in enumerate(p) if x in set(ans)), 0) for p, ans in zip(preds, answers)) / n

    return {
        "hits@1": hits(1),
        "hits@5": hits(5),
        "hits@10": hits(10),
        "hits@20": hits(20),
        "recall@1": recall(1),
        "recall@5": recall(5),
        "recall@10": recall(10),
        "recall@20": recall(20),
        "mrr": mrr(),
    }


# ---------------------------------------------------------------------------
# LLM Reranking (async, using standard OpenAI API)
# ---------------------------------------------------------------------------


async def async_score_candidate(
    session,
    query: str,
    candidate_text: str,
    api_key: str,
    semaphore: asyncio.Semaphore,
    errors: list,
    error_log: list,
    model: str = "gpt-4o-mini",
) -> float:
    """Score a single candidate asynchronously using OpenAI API."""
    prompt = f"""Score how relevant this candidate is to answering the query.

Query: {query}

Candidate:
{candidate_text[:1500]}

Score 0.0 to 1.0 (1.0=directly answers, 0.0=irrelevant).
Respond with ONLY a number."""

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10,
        "temperature": 0,
    }

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(3):
        try:
            # Only hold semaphore during actual request, not during retry sleep
            async with semaphore:
                async with session.post(url, headers=headers, json=payload, timeout=30) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        score_text = result["choices"][0]["message"]["content"].strip()
                        try:
                            return float(score_text)
                        except Exception:
                            return 0.5
                    elif resp.status == 429:
                        # Rate limited - backoff outside semaphore
                        backoff = min(2 ** attempt, 8)  # 1, 2, 4 (capped at 8)
                        error_log.append(f"429 rate limit, backoff {backoff}s")
                    else:
                        errors[0] += 1
                        error_log.append(f"HTTP {resp.status}")
                        return 0.5
        except asyncio.TimeoutError:
            error_log.append(f"Timeout (attempt {attempt+1})")
            backoff = min(2 ** attempt, 8)
        except Exception as e:
            error_log.append(f"Exception: {type(e).__name__}: {e}")
            backoff = min(2 ** attempt, 8)

        # Sleep OUTSIDE semaphore (backoff is set in all retry paths above)
        if attempt < 2:
            await asyncio.sleep(backoff)

    errors[0] += 1
    return 0.5


async def async_rerank_question(
    session,
    question: str,
    node_ids: list,
    node_texts: list,
    api_key: str,
    semaphore: asyncio.Semaphore,
    errors: list,
    error_log: list,
    model: str = "gpt-4o-mini",
) -> tuple:
    """Rerank all candidates for a question in parallel."""
    if len(node_ids) <= 1:
        return node_ids, [1.0] * len(node_ids)

    tasks = []
    for _nid, txt in zip(node_ids, node_texts):
        tasks.append(async_score_candidate(session, question, txt, api_key, semaphore, errors, error_log, model))

    scores = await asyncio.gather(*tasks)

    ranked = sorted(zip(node_ids, scores), key=lambda x: -x[1])
    reranked_ids = [nid for nid, _ in ranked]
    reranked_scores = [s for _, s in ranked]

    return reranked_ids, reranked_scores


async def run_async_reranking(
    retrieval_data: list,
    pack,
    api_key: str,
    pbar,
    top_k: int = 20,
    model: str = "gpt-4o-mini",
) -> tuple:
    """Run all reranking in parallel using async.

    Only scores top_k candidates per question.
    Returns (results_dict, error_log_list).
    """
    semaphore = asyncio.Semaphore(64)
    connector = aiohttp.TCPConnector(limit=128)
    errors = [0]
    error_log = []
    results = {}

    async with aiohttp.ClientSession(connector=connector) as session:

        async def process_one(data):
            idx = data["idx"]
            question = data["record"]["question"]
            nids = data["nids"][:top_k]
            node_texts = [pack.node_text[n] for n in nids]

            reranked_ids, scores = await async_rerank_question(
                session, question, nids, node_texts, api_key, semaphore, errors, error_log, model
            )
            pbar.update(1)
            return idx, reranked_ids, scores

        tasks = [process_one(d) for d in retrieval_data]

        for coro in asyncio.as_completed(tasks):
            idx, reranked_ids, scores = await coro
            results[idx] = (reranked_ids, scores)
            pbar.set_postfix(errors=errors[0])

    return results, error_log


# ---------------------------------------------------------------------------
# OpenAI Backend for pythea guard
# ---------------------------------------------------------------------------


class OpenAIBackend:
    """OpenAI API backend compatible with pythea's hallucination detector."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        self._session = None
    
    def _get_session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session
    
    def chat(self, messages: list, **options) -> dict:
        """Non-streaming chat completions against OpenAI API."""
        import requests
        
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": options.get("temperature", 0.2),
            "stream": False,
        }
        
        if options.get("max_tokens"):
            body["max_tokens"] = options["max_tokens"]
        
        # Add optional parameters
        for key in ["tools", "tool_choice", "n", "response_format",
                    "top_p", "frequency_penalty", "presence_penalty",
                    "stop", "seed", "logprobs", "top_logprobs"]:
            if key in options:
                body[key] = options[key]
        
        session = self._get_session()
        timeout_s = float(options.get("timeout_ms", 60000)) / 1000.0
        
        max_attempts = int(options.get("max_attempts", 3))
        last_err = None
        
        for attempt in range(max_attempts):
            try:
                res = session.post(url, headers=headers, json=body, timeout=timeout_s)
                
                if res.status_code == 429:
                    # Rate limited - backoff and retry
                    backoff = min(2 ** attempt, 8)
                    time.sleep(backoff)
                    continue
                
                if res.status_code >= 500:
                    # Server error - backoff and retry
                    backoff = min(2 ** attempt, 8)
                    time.sleep(backoff)
                    last_err = RuntimeError(f"HTTP {res.status_code}: {res.text[:200]}")
                    continue
                
                res.raise_for_status()
                return res.json()
                
            except requests.RequestException as e:
                last_err = e
                backoff = min(2 ** attempt, 8)
                time.sleep(backoff)
        
        if last_err is not None:
            raise last_err
        raise RuntimeError("All attempts failed")


# ---------------------------------------------------------------------------
# Multiprocess retrieval helpers
# ---------------------------------------------------------------------------


_G_PACK = None
_G_RETRIEVER = None
_G_Q_EMB = None
_G_NO_GRAPH_EXPANSION = False
_G_MAX_PRUNED_NODES = 128
_G_COSINE_TOPK_CHUNKED = None


def _worker_ping() -> int:
    return 1


def _retrieve_one(idx: int):
    """Run retrieval for a single query index (executed in worker process)."""
    # NOTE: everything heavy lives in globals (shared via fork/COW)
    q = _G_Q_EMB[idx]

    with torch.inference_mode():
        if _G_NO_GRAPH_EXPANSION:
            topk_idx, _topk_scores = _G_COSINE_TOPK_CHUNKED(_G_PACK.node_emb, q, k=_G_MAX_PRUNED_NODES)
            nids = topk_idx.tolist()
            graph_text = "node_id,node_attr\n" + "\n".join(
                f'{nid},"{_G_PACK.node_text[nid][:500]}"' for nid in nids
            )
            return idx, nids, graph_text

        ret = _G_RETRIEVER.retrieve_subgraph(q)
        nids = ret["pruned"]["global_nid"].tolist()

        # Fast embedding-based sort before optional LLM rerank
        if len(nids) > 1:
            sims = (_G_PACK.node_emb[nids] @ q.unsqueeze(1)).squeeze()
            order = sims.argsort(descending=True).tolist()
            nids = [nids[j] for j in order]

        graph_text = _G_RETRIEVER.textualize(ret["pruned"])  # evidence for guard
        return idx, nids, graph_text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(
    data_dir: str,
    split: str = "test",
    max_samples: Optional[int] = None,
    k_seeds: int = 4,
    hop: int = 1,
    top_m: int = 100,
    max_pruned_nodes: int = 64,
    output: Optional[str] = None,
    no_guard: bool = False,
    openai_api_key: Optional[str] = None,
    llm_rerank_flag: bool = False,
    no_graph_expansion: bool = False,
    rerank_top_k: int = 20,
    retrieval_workers: int = 0,
    guard_workers: int = 0,
    mp_start: str = "fork",
    torch_threads: int = 1,
    model: str = "gpt-4o-mini",
):

    # Import here so the script still starts even if deps missing
    from utils import (
        NativePyGGraphRAG,
        RetrievalConfig,
        cosine_topk_chunked,
        GraphPack,
    )

    # ---- Thread caps (big deal when you add process-level parallelism) ----
    torch_threads = max(1, int(torch_threads))
    torch.set_num_threads(torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    data = Path(data_dir)
    ensure_stark_data(data)

    # Load data
    print("Loading data...")
    graph_pack_path = data / "stark_prime_graph_pack.pt"
    if graph_pack_path.exists():
        print(f"  Loading pre-built graph pack from {graph_pack_path}")
        pack_data = torch.load(graph_pack_path, weights_only=False)
        pack = GraphPack(
            edge_index=pack_data["edge_index"],
            node_emb=pack_data["node_emb"],
            node_text=pack_data["node_text"],
            edge_text=pack_data.get("edge_text"),
        )
        print(f"  {pack.num_nodes} nodes, {pack.num_edges} edges, emb_dim={pack.node_emb.shape[1]}")
    else:
        node_info, edge_index, doc_info, edge_type = load_skb(data)
        node_emb = load_emb_dict(data / "candidate_emb_dict.pt", len(node_info))
        pack = build_pack(node_info, edge_index, doc_info, edge_type, node_emb)
        print(f"  {pack.num_nodes} nodes, {pack.num_edges} edges, emb_dim={node_emb.shape[1]}")

    qa = [r for r in load_qa(data) if r["split"] == split] if split != "all" else load_qa(data)
    if max_samples:
        qa = qa[:max_samples]
    print(f"  {len(qa)} {split} samples")

    q_emb = load_query_emb(data / "query_emb_dict.pt", [r["id"] for r in qa])
    print(f"  query_emb_dim={q_emb.shape[1]}")

    # Setup retriever
    retriever = None
    if not no_graph_expansion:
        cfg = RetrievalConfig(k_seeds=k_seeds, hop=hop, top_m_prize=top_m, max_pruned_nodes=max_pruned_nodes)
        retriever = NativePyGGraphRAG(pack, cfg)
    else:
        print(f"  Using simple top-{max_pruned_nodes} retrieval (no graph expansion)")

    # Get OpenAI API key
    api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
    
    # LLM rerank setup
    if llm_rerank_flag:
        if not api_key:
            print("Warning: OPENAI_API_KEY required for LLM reranking, falling back to embedding rerank")
            llm_rerank_flag = False
        else:
            print(f"LLM reranking enabled using OpenAI API (model: {model})")

    # Decide worker counts
    if retrieval_workers <= 0:
        # Default: enough to saturate CPU but not blow up RAM
        retrieval_workers = min(32, max(1, (os.cpu_count() or 8) // 2))
    if guard_workers <= 0:
        # Default guard concurrency: small-ish to avoid 429 storms
        guard_workers = min(16, max(4, 8))

    # ---- Install globals for workers (fork-friendly) ----
    global _G_PACK, _G_RETRIEVER, _G_Q_EMB, _G_NO_GRAPH_EXPANSION, _G_MAX_PRUNED_NODES, _G_COSINE_TOPK_CHUNKED
    _G_PACK = pack
    _G_RETRIEVER = retriever
    _G_Q_EMB = q_emb
    _G_NO_GRAPH_EXPANSION = bool(no_graph_expansion)
    _G_MAX_PRUNED_NODES = int(max_pruned_nodes)
    _G_COSINE_TOPK_CHUNKED = cosine_topk_chunked

    # ---- Start retrieval pool (spawn/fork) BEFORE any threads ----
    mp_start = (mp_start or "fork").lower()
    if mp_start not in ("fork", "spawn", "forkserver"):
        mp_start = "fork"

    import multiprocessing as mp

    try:
        mp_ctx = mp.get_context(mp_start)
    except ValueError:
        mp_ctx = mp.get_context("spawn")

    print(f"Evaluating... (retrieval_workers={retrieval_workers}, mp_start={mp_ctx.get_start_method()}, guard_workers={guard_workers})")

    t0 = time.time()

    # Evidence guard setup (threads) — start AFTER pool is created to avoid fork-with-threads issues
    guard_queue, guard_results, guard_threads = None, {}, []
    if not no_guard:
        try:
            from pythea.hallucination_detector import (
                MixtureInputs,
                MixtureConfig,
                evaluate_evidence_mixture_logprob_with_backend,
            )

            if not api_key:
                raise ValueError("OPENAI_API_KEY is required when running with guard")

            print(f"Using OpenAI API for guard (model: {model})")
            backend = OpenAIBackend(api_key, model=model)
            guard_queue = queue.Queue(maxsize=guard_workers * 4)

            guard_pbar = tqdm(total=len(qa), desc="Guard", position=2)

            def worker():
                while True:
                    item = guard_queue.get()
                    if item is None:
                        break
                    idx, post, skel = item
                    try:
                        decision, m = evaluate_evidence_mixture_logprob_with_backend(
                            MixtureInputs(posterior_prompt=post, skeleton_prompts=[skel]),
                            MixtureConfig(model="gpt-4o-mini", hstar=0.05),
                            backend=backend,
                        )
                        guard_results[idx] = (decision, m)
                    except Exception as e:
                        guard_results[idx] = (None, {"error": str(e)})
                    guard_pbar.update(1)

            for _ in range(guard_workers):
                t = threading.Thread(target=worker, daemon=True)
                t.start()
                guard_threads.append(t)
        except ImportError as e:
            print(f"pythea not found, skipping guard: {e}")
            no_guard = True
        except ValueError as e:
            print(f"Error: {e}")
            no_guard = True

    # Phase 1: Retrieval (parallel)
    retrieval_data = []
    retrieval_pbar = tqdm(total=len(qa), desc="Retrieval", position=0)

    # Submit all retrieval jobs up front, then stream results as they complete.
    with ProcessPoolExecutor(max_workers=retrieval_workers, mp_context=mp_ctx) as ex:
        # Force worker start *now* (so we don't fork after threads start).
        # This is especially important if guard workers are enabled.
        _ = ex.submit(_worker_ping).result()

        futs = {ex.submit(_retrieve_one, i): i for i in range(len(qa))}

        for fut in as_completed(futs):
            i, nids, graph_text = fut.result()
            r = qa[i]

            retrieval_data.append({"idx": i, "record": r, "nids": nids, "graph_text": graph_text})

            # Queue guard job ASAP (overlap network with remaining retrieval)
            if guard_queue is not None:
                guard_queue.put(
                    (
                        i,
                        f"EVIDENCE:\n{graph_text}\n\nQUESTION: {r['question']}\n\nCan you answer this question using ONLY the evidence above?",
                        f"QUESTION: {r['question']}\n\nCan you answer this question without any context?",
                    )
                )

            retrieval_pbar.update(1)

    retrieval_pbar.close()

    # Keep retrieval_data in original order
    retrieval_data.sort(key=lambda d: d["idx"])

    # Phase 2: Async LLM reranking (unchanged semantics)
    rerank_results = {}
    rerank_error_log = []
    if llm_rerank_flag and api_key:
        print(
            f"Running async LLM reranking on {len(retrieval_data)} questions (top {rerank_top_k} candidates each)..."
        )
        rerank_pbar = tqdm(total=len(retrieval_data), desc="LLM Rerank", position=1)
        rerank_results, rerank_error_log = asyncio.run(
            run_async_reranking(retrieval_data, pack, api_key, rerank_pbar, top_k=rerank_top_k, model=model)
        )
        rerank_pbar.close()

        # Print error summary
        if rerank_error_log:
            from collections import Counter
            error_counts = Counter(rerank_error_log)
            print(f"\nLLM Rerank errors ({len(rerank_error_log)} total):")
            for err, count in error_counts.most_common(10):
                print(f"  {count:4d}x {err[:80]}")

    # Phase 3: Assemble results
    all_preds, all_ans, results = [], [], []
    for data in retrieval_data:
        i = data["idx"]
        r = data["record"]
        nids = data["nids"]
        scores = None

        if i in rerank_results:
            nids, scores = rerank_results[i]

        all_preds.append(nids)
        all_ans.append(r["answer_ids"])

        entry = {
            "id": r["id"],
            "question": r["question"],
            "answer_ids": r["answer_ids"],
            "original_ids": data["nids"],
            "reranked_ids": nids,
            "pred_ids": nids,
        }
        if scores is not None:
            entry["scores"] = scores
        results.append(entry)

    # Wait for guard
    if guard_queue is not None:
        for _ in range(len(guard_threads)):
            guard_queue.put(None)
        for t in guard_threads:
            t.join()
        # guard_pbar is local in try block; look it up in locals
        try:
            guard_pbar.close()
        except Exception:
            pass

        for i, (dec, m) in guard_results.items():
            results[i]["guard_decision"] = dec
            results[i]["ISR_ref"] = (m or {}).get("ISR_ref")

    # Metrics
    metrics = compute_metrics(all_preds, all_ans)
    baselines = {"hits@1": 15.57, "hits@5": 33.42, "recall@20": 39.09, "mrr": 24.11}

    print("\n" + "=" * 50)
    for k, v in metrics.items():
        bl = baselines.get(k, 0)
        print(f"{k:<12} {v:>7.2f}  (baseline: {bl:.2f}, Δ: {v-bl:+.2f})")

    if output:
        with open(output, "w") as f:
            json.dump(
                {
                    "metrics": metrics,
                    "results": results,
                    "config": {
                        "k_seeds": k_seeds,
                        "hop": hop,
                        "top_m": top_m,
                        "max_pruned_nodes": max_pruned_nodes,
                        "llm_rerank": llm_rerank_flag,
                        "no_graph_expansion": no_graph_expansion,
                        "retrieval_workers": retrieval_workers,
                        "guard_workers": guard_workers,
                        "mp_start": mp_ctx.get_start_method(),
                        "torch_threads": torch_threads,
                        "model": model,
                    },
                },
                f,
            )
        print(f"Saved to {output}")

    print(f"Total wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--max-samples", type=int)
    p.add_argument("--k-seeds", type=int, default=16)
    p.add_argument("--hop", type=int, default=1)
    p.add_argument("--top-m", type=int, default=100)
    p.add_argument("--max-pruned-nodes", type=int, default=128)
    p.add_argument("--output")
    p.add_argument("--no-guard", action="store_true")
    p.add_argument("--llm-rerank", action="store_true", help="Use LLM-based reranking (STaRK-style)")
    p.add_argument("--rerank-top-k", type=int, default=20, help="Number of candidates to LLM rerank")
    p.add_argument("--no-graph-expansion", action="store_true", help="Skip graph expansion")
    p.add_argument("--openai-api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    p.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use (default: gpt-4o-mini)")

    p.add_argument("--retrieval-workers", type=int, default=0, help="CPU processes for retrieval (0=auto)")
    p.add_argument("--guard-workers", type=int, default=0, help="Threads for guard calls (0=auto)")
    p.add_argument(
        "--mp-start",
        default="fork",
        choices=["fork", "spawn", "forkserver"],
        help="Multiprocessing start method (fork recommended on Linux for large tensors)",
    )
    p.add_argument("--torch-threads", type=int, default=1, help="torch.set_num_threads() per process")

    a = p.parse_args()
    run(
        a.data_dir,
        a.split,
        a.max_samples,
        a.k_seeds,
        a.hop,
        a.top_m,
        a.max_pruned_nodes,
        a.output,
        a.no_guard,
        a.openai_api_key,
        a.llm_rerank,
        a.no_graph_expansion,
        a.rerank_top_k,
        a.retrieval_workers,
        a.guard_workers,
        a.mp_start,
        a.torch_threads,
        a.model,
    )
