#!/usr/bin/env python3
"""Convert STaRK-Prime data to graph pack format.

Uses official text-embedding-ada-002 embeddings from HuggingFace,
but builds richer node_text from node_info details/summary/description.

Can be run standalone or imported:
    from data import ensure_stark_data
    ensure_stark_data(Path("stark_data"))
"""

import urllib.request
import zipfile
from pathlib import Path

import torch
import pickle
import json
import csv
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Data Download
# ---------------------------------------------------------------------------

STARK_HF = "https://huggingface.co/datasets/snap-stanford/stark/resolve/main"

REQUIRED_FILES = {
    "processed.zip": f"{STARK_HF}/skb/prime/processed.zip",
    "stark_qa.csv": f"{STARK_HF}/qa/prime/stark_qa/stark_qa.csv",
    "split.pt": f"{STARK_HF}/qa/prime/split/split.pt",
    "candidate_emb_dict.pt": f"{STARK_HF}/emb/prime/text-embedding-ada-002/doc/candidate_emb_dict.pt",
    "query_emb_dict.pt": f"{STARK_HF}/emb/prime/text-embedding-ada-002/query/query_emb_dict.pt",
}


def download_file(url: str, dest: Path, desc: str = None):
    """Download a file if it doesn't exist."""
    if dest.exists():
        print(f"  [skip] {desc or dest.name} already exists")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {desc or dest.name}...")

    def progress_hook(count, block_size, total_size):
        if total_size > 0:
            pct = count * block_size * 100 // total_size
            print(f"\r    {pct}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, progress_hook)
    print()


def download_stark_data(data_dir: Path):
    """Download all required STaRK-Prime data."""
    print("=" * 60)
    print("Checking/downloading STaRK-Prime data...")
    print("=" * 60)

    data_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = data_dir / "processed"

    # Download and extract processed.zip if needed
    if not processed_dir.exists() or not (processed_dir / "node_info.pkl").exists():
        zip_path = data_dir / "processed.zip"
        download_file(REQUIRED_FILES["processed.zip"], zip_path, "Knowledge base (processed.zip)")

        if zip_path.exists():
            print("  Extracting processed.zip...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(data_dir)
            zip_path.unlink()
            print(f"  Extracted to {processed_dir}/")
    else:
        print("  [skip] processed/ already exists")

    # Download QA and splits
    download_file(REQUIRED_FILES["stark_qa.csv"], data_dir / "stark_qa.csv", "QA data (stark_qa.csv)")
    download_file(REQUIRED_FILES["split.pt"], data_dir / "split.pt", "Official splits (split.pt)")

    # Download official embeddings
    download_file(REQUIRED_FILES["candidate_emb_dict.pt"], data_dir / "candidate_emb_dict.pt",
                  "Official node embeddings (candidate_emb_dict.pt ~800MB)")
    download_file(REQUIRED_FILES["query_emb_dict.pt"], data_dir / "query_emb_dict.pt",
                  "Official query embeddings (query_emb_dict.pt ~70MB)")

    print("Download complete!\n")


def build_graph_pack(data_dir: Path):
    """Build stark_prime_graph_pack.pt from downloaded data."""
    print('Loading data...')
    edge_index = torch.load(data_dir / 'processed/edge_index.pt', weights_only=True)
    edge_types = torch.load(data_dir / 'processed/edge_types.pt', weights_only=True)

    with open(data_dir / 'processed/node_info.pkl', 'rb') as f:
        node_info = pickle.load(f)
    with open(data_dir / 'processed/edge_type_dict.pkl', 'rb') as f:
        edge_type_dict = pickle.load(f)

    num_nodes = len(node_info)
    print(f'Nodes: {num_nodes:,}, Edges: {edge_index.shape[1]:,}')

    # Build RICH node texts
    print('Building rich node texts...')
    node_texts = []
    for i in range(num_nodes):
        info = node_info[i]
        name = info.get('name', f'Node_{i}')
        node_type = info.get('type', 'unknown')
        details = info.get('details', {})
        desc = ''
        if isinstance(details, dict):
            for key in ['summary', 'description']:
                val = details.get(key)
                if val and isinstance(val, str):
                    desc = val[:400]
                    break
        if desc:
            text = f'name: {name}, type: {node_type}, description: {desc}'
        else:
            text = f'name: {name}, type: {node_type}'
        node_texts.append(text[:512])

    # Edge texts
    print('Building edge texts...')
    edge_texts = [edge_type_dict[int(t)].upper().replace(' ', '_').replace('-', '_') for t in edge_types.tolist()]

    # Load OFFICIAL embeddings
    print('Loading official embeddings from candidate_emb_dict.pt...')
    emb_dict = torch.load(data_dir / 'candidate_emb_dict.pt', weights_only=False)
    emb_dim = next(iter(emb_dict.values())).shape[-1]
    print(f'  Embedding dim: {emb_dim}, entries: {len(emb_dict)}')

    node_emb = torch.zeros(num_nodes, emb_dim)
    for nid, emb in tqdm(emb_dict.items(), desc='  Converting'):
        if nid < num_nodes:
            node_emb[nid] = emb.squeeze()

    # Normalize
    node_emb = node_emb / node_emb.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    print(f'Final embedding shape: {node_emb.shape}')

    # Save graph pack
    print('Saving graph pack...')
    graph_pack = {
        'edge_index': edge_index,
        'x': node_emb,
        'node_emb': node_emb,
        'node_text': node_texts,
        'edge_text': edge_texts,
        'num_nodes': num_nodes,
        'num_edges': edge_index.shape[1],
    }
    torch.save(graph_pack, data_dir / 'stark_prime_graph_pack.pt')
    print(f'Saved {data_dir}/stark_prime_graph_pack.pt')


def ensure_stark_data(data_dir: Path):
    """Ensure all STaRK-Prime data is downloaded and graph pack is built.

    This is the main entry point for the benchmark script.
    """
    data_dir = Path(data_dir)

    # Check if graph pack already exists
    graph_pack_path = data_dir / "stark_prime_graph_pack.pt"
    query_emb_path = data_dir / "query_emb_dict.pt"
    split_path = data_dir / "split.pt"
    qa_path = data_dir / "stark_qa.csv"

    # If all required files exist, skip
    if all(p.exists() for p in [graph_pack_path, query_emb_path, split_path, qa_path]):
        print(f"STaRK-Prime data ready at {data_dir}")
        return

    # Download missing files
    download_stark_data(data_dir)

    # Build graph pack if needed
    if not graph_pack_path.exists():
        build_graph_pack(data_dir)

    print(f"\nSTaRK-Prime data ready at {data_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download and prepare STaRK-Prime data")
    parser.add_argument("--data-dir", type=str, default="stark_data", help="Output directory")
    args = parser.parse_args()

    ensure_stark_data(Path(args.data_dir))
