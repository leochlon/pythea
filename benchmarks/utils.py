"""
Graph retrieval utilities for STaRK-Prime benchmarks.

Provides:
  - GraphPack: container for graph data (edge_index, node_emb, node_text)
  - NativePyGGraphRAG: retriever with seed selection, k-hop expansion, and PCST-style pruning
  - RetrievalConfig: configuration for retrieval parameters
  - cosine_topk_chunked: memory-efficient top-k similarity search
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _l2_normalize(x: Tensor, eps: float = 1e-12) -> Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def cosine_topk_chunked(
    node_emb: Tensor,
    q_emb: Tensor,
    k: int,
    chunk_size: int = 200_000,
) -> Tuple[Tensor, Tensor]:
    """Return (topk_idx, topk_scores) for a single query embedding."""
    assert q_emb.dim() == 1, "single-query only"
    k = int(k)
    if k <= 0:
        return (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.float))

    q = _l2_normalize(q_emb).to(node_emb.device)
    best_scores: Optional[Tensor] = None
    best_idx: Optional[Tensor] = None

    n = node_emb.size(0)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = node_emb[start:end]
        sims = _l2_normalize(chunk) @ q  # [chunk]
        kk = min(k, sims.numel())
        scores, idx = torch.topk(sims, k=kk, largest=True)
        idx = idx + start

        if best_scores is None:
            best_scores, best_idx = scores, idx
        else:
            comb_scores = torch.cat([best_scores, scores], dim=0)
            comb_idx = torch.cat([best_idx, idx], dim=0)
            kk2 = min(k, comb_scores.numel())
            new_scores, pos = torch.topk(comb_scores, k=kk2, largest=True)
            best_scores = new_scores
            best_idx = comb_idx[pos]

    assert best_scores is not None and best_idx is not None
    return best_idx, best_scores


def assign_prizes(
    base_node_emb: Tensor,
    q_emb: Tensor,
    top_m: int = 100,
    max_prize: float = 4.0,
) -> Tensor:
    """Assign prizes to top-M nodes (higher prize = more relevant)."""
    m = min(int(top_m), base_node_emb.size(0))
    prizes = torch.zeros(base_node_emb.size(0), dtype=torch.float, device=base_node_emb.device)
    if m <= 0:
        return prizes

    idx, _scores = cosine_topk_chunked(base_node_emb, q_emb, k=m, chunk_size=base_node_emb.size(0))
    # Linear schedule from max_prize -> 0
    if m == 1:
        prizes[idx[0]] = float(max_prize)
        return prizes

    step = float(max_prize) / float(m - 1)
    for rank, n in enumerate(idx.tolist()):
        prizes[n] = float(max_prize) - step * float(rank)
    return prizes


def _build_undirected_adj(edge_index: Tensor, num_nodes: int) -> List[List[int]]:
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    for u, v in zip(src, dst):
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            adj[u].append(v)
            adj[v].append(u)
    return adj


def _bfs_parents(adj: List[List[int]], root: int) -> List[int]:
    n = len(adj)
    parent = [-1] * n
    parent[root] = root
    q = [root]
    qi = 0
    while qi < len(q):
        u = q[qi]
        qi += 1
        for v in adj[u]:
            if parent[v] == -1:
                parent[v] = u
                q.append(v)
    return parent


def _path_to_root(parent: List[int], node: int, root: int) -> List[int]:
    if node < 0 or node >= len(parent) or parent[node] == -1:
        return []
    path = [node]
    while path[-1] != root:
        p = parent[path[-1]]
        if p == -1 or p == path[-1]:
            break
        path.append(p)
        if len(path) > len(parent) + 5:
            break
    if path[-1] != root:
        return []
    return path


def steiner_prune_via_shortest_paths(
    edge_index: Tensor,
    terminals: Sequence[int],
    num_nodes: int,
    root: Optional[int] = None,
    max_nodes: Optional[int] = None,
) -> Tensor:
    """Lightweight PCST-like prune: connect terminals to a root via BFS paths.

    Returns:
        keep_mask: BoolTensor [num_nodes] indicating nodes to keep.
    """
    if len(terminals) == 0 or num_nodes == 0:
        return torch.zeros(num_nodes, dtype=torch.bool)

    terminals = list(dict.fromkeys(int(t) for t in terminals))  # dedup preserve
    root = int(root) if root is not None else terminals[0]

    adj = _build_undirected_adj(edge_index, num_nodes)
    parent = _bfs_parents(adj, root)

    keep = set([root])
    for t in terminals:
        path = _path_to_root(parent, t, root)
        keep.update(path)

    if max_nodes is not None and len(keep) > int(max_nodes):
        ordered = [root] + [n for n in range(num_nodes) if n in keep and n != root]
        keep = set(ordered[: int(max_nodes)])

    keep_mask = torch.zeros(num_nodes, dtype=torch.bool)
    keep_mask[list(keep)] = True
    return keep_mask


def induce_subgraph(
    edge_index: Tensor,
    keep_mask: Tensor,
    edge_mask_global: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Induce a subgraph and relabel nodes.

    Args:
        edge_index: [2, E] (local edges in base subgraph)
        keep_mask: [N] bool (which local nodes to keep)
        edge_mask_global: [E_global] bool mask from k_hop_subgraph (optional)

    Returns:
        new_edge_index: [2, E']
        old_to_new: [N] long with -1 for dropped nodes
        new_global_edge_indices: [E'] indices into original global edge list (or None)
    """
    num_nodes = keep_mask.numel()
    old_to_new = torch.full((num_nodes,), -1, dtype=torch.long)
    kept = keep_mask.nonzero(as_tuple=False).view(-1)
    old_to_new[kept] = torch.arange(kept.numel(), dtype=torch.long)

    src, dst = edge_index[0], edge_index[1]
    edge_keep = keep_mask[src] & keep_mask[dst]
    new_edge = edge_index[:, edge_keep]
    new_edge = old_to_new[new_edge]

    # Track which global edges survive pruning
    new_global_edge_indices: Optional[Tensor] = None
    if edge_mask_global is not None:
        # edge_mask_global is a bool mask over ALL global edges
        # We need to find indices of edges that were in the base subgraph
        global_edge_indices = edge_mask_global.nonzero(as_tuple=False).view(-1)
        # Now filter to only those kept after pruning
        new_global_edge_indices = global_edge_indices[edge_keep]

    return new_edge, old_to_new, new_global_edge_indices


def escape_csv_field(s: str) -> str:
    return s.replace('"', '""')


# ---------------------------------------------------------------------------
# Embedding model (for questions; nodes are assumed pre-embedded in graph pack)
# ---------------------------------------------------------------------------

class HFTextEncoder(torch.nn.Module):
    """Minimal transformer sentence encoder (mean pooling)."""

    def __init__(self, model_name: str, device: str = "cuda"):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.device = torch.device(device)

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 32, max_length: int = 256) -> Tensor:
        outs: List[Tensor] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            tok = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)
            enc = self.model(**tok)
            last = enc.last_hidden_state  # [B, T, H]
            mask = tok["attention_mask"].unsqueeze(-1)  # [B, T, 1]
            summed = (last * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp_min(1)
            sent = summed / denom
            outs.append(sent.detach().cpu())
        return torch.cat(outs, dim=0)


# ---------------------------------------------------------------------------
# Graph pack + retriever
# ---------------------------------------------------------------------------

@dataclass
class GraphPack:
    edge_index: Tensor            # [2, E] global
    node_emb: Tensor              # [N, D] (for retrieval)
    node_text: List[str]          # len N
    edge_text: Optional[List[str]] = None  # len E (relation names)

    @property
    def num_nodes(self) -> int:
        return int(self.node_emb.size(0))

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.size(1))


@dataclass
class RetrievalConfig:
    k_seeds: int = 4
    hop: int = 1
    top_m_prize: int = 100
    max_prize: float = 4.0
    max_pruned_nodes: int = 64
    similarity_chunk_size: int = 200_000


class NativePyGGraphRAG:
    def __init__(self, pack: GraphPack, cfg: RetrievalConfig):
        self.pack = pack
        self.cfg = cfg

    def retrieve_subgraph(self, q_emb: Tensor) -> Dict[str, Any]:
        """Retrieve base/pruned subgraphs."""
        # Keep retrieval on CPU by default (node_emb likely huge).
        node_emb = self.pack.node_emb
        q = q_emb.detach().cpu()

        seed_idx, _ = cosine_topk_chunked(
            node_emb,
            q,
            k=self.cfg.k_seeds,
            chunk_size=self.cfg.similarity_chunk_size,
        )
        seed_idx = seed_idx.to(torch.long)

        base = self._k_hop_base(seed_idx, k_hops=self.cfg.hop)

        prizes = assign_prizes(
            base["x"],
            q,
            top_m=self.cfg.top_m_prize,
            max_prize=self.cfg.max_prize,
        )

        pruned = self._pcst_prune(base, prizes, seed_nodes_local=base["seed_local"])

        return {
            "seed_nodes": seed_idx.cpu(),
            "base": base,
            "pruned": pruned,
            "prizes": prizes.detach().cpu(),
        }

    def _k_hop_base(self, seed_nodes_global: Tensor, k_hops: int = 1) -> Dict[str, Any]:
        """Return base subgraph around global seed nodes."""
        from torch_geometric.utils import k_hop_subgraph

        subset, edge_index_local, mapping, edge_mask = k_hop_subgraph(
            seed_nodes_global,
            k_hops,
            self.pack.edge_index,
            relabel_nodes=True,
            num_nodes=self.pack.num_nodes,
        )

        x = self.pack.node_emb[subset]  # stays on CPU
        base = {
            "edge_index": edge_index_local,
            "global_nid": subset,
            "x": x,
            "edge_mask": edge_mask,  # bool mask over global edges
            "seed_local": mapping,
        }
        return base

    def _pcst_prune(
        self,
        base: Dict[str, Any],
        prizes: Tensor,
        seed_nodes_local: Tensor,
    ) -> Dict[str, Any]:
        """Prune base graph down to a compact evidence subgraph (PCST-ish)."""
        edge_index = base["edge_index"]
        n = int(base["x"].size(0))

        terminals = prizes.nonzero(as_tuple=False).view(-1).tolist()
        terminals += seed_nodes_local.view(-1).tolist()
        terminals = list(dict.fromkeys(int(t) for t in terminals))
        if len(terminals) == 0:
            terminals = seed_nodes_local.view(-1).tolist()

        root = terminals[0]
        if prizes.numel() == n and len(terminals) > 0:
            root = int(prizes.argmax().item())

        keep_mask = steiner_prune_via_shortest_paths(
            edge_index=edge_index,
            terminals=terminals,
            num_nodes=n,
            root=root,
            max_nodes=self.cfg.max_pruned_nodes,
        )

        pruned_edge_index, old_to_new, global_edge_indices = induce_subgraph(
            edge_index,
            keep_mask,
            edge_mask_global=base.get("edge_mask"),
        )
        kept_local = keep_mask.nonzero(as_tuple=False).view(-1)

        pruned = {
            "edge_index": pruned_edge_index,
            "global_nid": base["global_nid"][kept_local],
            "x": base["x"][kept_local],
            "keep_local": kept_local,
            "old_to_new": old_to_new,
            "global_edge_indices": global_edge_indices,  # for edge text lookup
        }
        return pruned

    def textualize(
        self,
        pruned: Dict[str, Any],
        edge_attr_fallback: str = "RELATED_TO",
    ) -> str:
        """Textualize pruned subgraph into the blog-style CSV-ish text.

        Format matches the NVIDIA blog example:
            node_id,node_attr
            14609,"name: Zopiclone, description: 'Zopiclone is...'"
            ...
            src,edge_attr,dst
            15570,SYNERGISTIC_INTERACTION,15441
            ...
        """
        global_nid = pruned["global_nid"].tolist()
        lines: List[str] = ["node_id,node_attr"]

        for gid in global_nid:
            txt = self.pack.node_text[gid]
            safe = escape_csv_field(txt)
            lines.append(f'{gid},"{safe}"')

        lines.append("src,edge_attr,dst")
        ei = pruned["edge_index"]
        src = ei[0].tolist()
        dst = ei[1].tolist()

        # Get global edge indices for looking up edge text
        global_edge_indices = pruned.get("global_edge_indices", None)

        for idx, (u, v) in enumerate(zip(src, dst)):
            src_g = int(global_nid[u])
            dst_g = int(global_nid[v])

            # Use actual edge text if available
            edge_attr = edge_attr_fallback
            if (
                self.pack.edge_text is not None
                and global_edge_indices is not None
                and idx < len(global_edge_indices)
            ):
                global_eid = int(global_edge_indices[idx].item())
                if 0 <= global_eid < len(self.pack.edge_text):
                    edge_attr = self.pack.edge_text[global_eid]

            lines.append(f"{src_g},{edge_attr},{dst_g}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Answer parsing + uncertainty
# ---------------------------------------------------------------------------

def extract_answer(decoded: str) -> str:
    if "Answer:" in decoded:
        return decoded.rsplit("Answer:", 1)[-1].strip()
    return decoded.strip()


def uncertainty_from_logprobs(avg_nll: float, min_logprob: float) -> float:
    """A simple uncertainty scalar (higher = more uncertain)."""
    return float(avg_nll) + float(max(0.0, -min_logprob))


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_graph_pack(path: str) -> GraphPack:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise ValueError(f"graph pack must be a dict, got {type(obj)}")

    edge_index = obj.get("edge_index", None)
    if edge_index is None:
        raise KeyError("graph pack missing 'edge_index'")

    node_emb = obj.get("x", None)
    if node_emb is None:
        node_emb = obj.get("node_emb", None)
    if node_emb is None:
        raise KeyError("graph pack missing 'x' or 'node_emb' (node embeddings)")

    node_text = obj.get("node_text", None)
    if node_text is None:
        names = obj.get("node_name", None)
        desc = obj.get("node_desc", None)
        if names is None or desc is None:
            raise KeyError("graph pack missing 'node_text' (or 'node_name'+'node_desc')")
        node_text = [f"name: {n}, description: {d}" for n, d in zip(names, desc)]

    edge_text = obj.get("edge_text", None)

    if isinstance(node_text, Tensor):
        node_text = node_text.tolist()

    return GraphPack(
        edge_index=edge_index,
        node_emb=node_emb,
        node_text=list(node_text),
        edge_text=edge_text,
    )


def load_questions(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
        return obj["data"]
    raise ValueError(f"Unsupported questions format in {path}")


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Native PyG GraphRAG for STaRK-Prime with uncertainty estimation"
    )
    p.add_argument(
        "--graph-pack",
        type=str,
        required=True,
        help="torch.load-able graph pack .pt",
    )
    p.add_argument(
        "--questions",
        type=str,
        default="",
        help="JSON or JSONL of questions (field: question)",
    )
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Write results to JSONL",
    )
    p.add_argument(
        "--embed-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HF model for question embeddings (must match node_emb space)",
    )
    p.add_argument(
        "--llm-model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
    )
    p.add_argument(
        "--sys-prompt",
        type=str,
        default=(
            "Use the provided knowledge graph to answer the question. "
            "If the answer is not in the graph, say you don't know."
        ),
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--k-seeds", type=int, default=4)
    p.add_argument("--hop", type=int, default=1)
    p.add_argument("--top-m", type=int, default=100)
    p.add_argument("--max-pruned-nodes", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--max-txt-len", type=int, default=4096)
    args = p.parse_args()

    pack = load_graph_pack(args.graph_pack)
    print(f"Loaded graph: {pack.num_nodes} nodes, {pack.num_edges} edges")
    if pack.edge_text:
        print(f"  edge_text available ({len(pack.edge_text)} entries)")

    cfg = RetrievalConfig(
        k_seeds=args.k_seeds,
        hop=args.hop,
        top_m_prize=args.top_m,
        max_pruned_nodes=args.max_pruned_nodes,
    )
    retriever = NativePyGGraphRAG(pack, cfg)

    encoder = HFTextEncoder(args.embed_model, device=args.device)

    # Import LLM from the same directory (examples/llm/)
    # Adjust this import path based on your repo structure:
    #   - If llm.py is in the same folder: from llm import LLM
    #   - If installed as package: from torch_geometric.nn.models import LLM
    from llm import LLM

    llm = LLM(
        args.llm_model,
        sys_prompt=args.sys_prompt,
        max_txt_len=args.max_txt_len,
        max_new_tokens=args.max_new_tokens,
    )

    if args.questions:
        rows = load_questions(args.questions)
        questions = [r["question"] for r in rows]
    else:
        rows = [{"question": "What is the function of CYP3A4 in drug metabolism?"}]
        questions = [rows[0]["question"]]

    q_emb = encoder.encode(questions)  # [B, D] on CPU

    results = []
    for i, row in enumerate(rows):
        q = row["question"]
        emb_i = q_emb[i].detach()  # CPU embedding

        retrieved = retriever.retrieve_subgraph(emb_i)
        pruned = retrieved["pruned"]
        graph_text = retriever.textualize(pruned)

        q_prompt = f"Question: {q}\nAnswer:"

        out = llm.inference(
            [q_prompt],
            context=[graph_text],
            max_tokens=args.max_new_tokens,
            return_logprobs=True,
            return_token_logprobs=False,
        )[0]

        decoded = out["text"]
        answer = extract_answer(decoded)

        u = uncertainty_from_logprobs(out["avg_nll"], out["min_logprob"])

        result = dict(row)
        result.update({
            "answer_pred": answer,
            "avg_nll": out["avg_nll"],
            "min_logprob": out["min_logprob"],
            "uncertainty": u,
            "seed_nodes": retrieved["seed_nodes"].tolist(),
            "num_base_nodes": int(retrieved["base"]["x"].size(0)),
            "num_pruned_nodes": int(pruned["x"].size(0)),
            "num_pruned_edges": int(pruned["edge_index"].size(1)),
        })
        results.append(result)

        print(f"[{i+1}/{len(rows)}] uncertainty={u:.4f} answer={answer[:120]!r}")

    if args.out:
        write_jsonl(args.out, results)
        print(f"Wrote {len(results)} results to {args.out}")


if __name__ == "__main__":
    main()
