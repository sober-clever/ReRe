"""Average pairwise semantic similarity over an item set.

Every unordered pair {i, j} with i != j is counted exactly once, so the score is
    mean_cosine = sum_{i<j} cos(e_i, e_j) / (N * (N - 1) / 2)

Usage:
    python calc_item_sim.py --item_path data/Amazon/info/Industrial_and_Scientific_5_1996-10-2018-11.txt
"""
import json
import os

import fire
import numpy as np
import torch
from tqdm import tqdm

from calc_sim import TextEncoder


def load_item_names(item_path, dedup=False):
    """Read an info txt the same way calc.py / evaluate.py do: the item name is
    everything before the last tab-separated field."""
    if not item_path.endswith(".txt"):
        item_path = item_path + ".txt"
    with open(item_path, "r") as f:
        lines = f.readlines()
    names = [_[:-len(_.split("\t")[-1])].strip() for _ in lines]
    names = [_ for _ in names if _]
    if dedup:
        seen = set()
        unique = []
        for name in names:
            if name not in seen:
                seen.add(name)
                unique.append(name)
        print(f"dedup: {len(names)} -> {len(unique)} item names")
        names = unique
    return names


def pairwise_stats(embs, block_size=1024):
    """Mean / std of cosine similarity over all unordered pairs i < j.

    Blocked over rows so the N x N matrix is never materialized: for each row
    block only its similarities against the columns to its right are summed.
    """
    N = len(embs)
    if N < 2:
        raise ValueError(f"need at least 2 items, got {N}")
    n_pairs = N * (N - 1) // 2

    total = 0.0
    total_sq = 0.0
    smallest, largest = np.inf, -np.inf
    for start in tqdm(range(0, N, block_size), desc="pairs"):
        end = min(start + block_size, N)
        # columns start..N only, then drop the lower triangle of the diagonal
        # square so each pair is visited exactly once and i == j is excluded
        sims = embs[start:end] @ embs[start:].T
        keep = np.ones(sims.shape, dtype=bool)
        keep[:, : end - start] = np.triu(
            np.ones((end - start, end - start), dtype=bool), k=1
        )
        vals = sims[keep].astype(np.float64)
        total += vals.sum()
        total_sq += np.square(vals).sum()
        if vals.size:
            smallest = min(smallest, float(vals.min()))
            largest = max(largest, float(vals.max()))

    mean = total / n_pairs
    var = max(total_sq / n_pairs - mean * mean, 0.0)
    return {
        "num_items": N,
        "num_pairs": n_pairs,
        "mean_cosine": mean,
        "std_cosine": float(np.sqrt(var)),
        "min_cosine": smallest,
        "max_cosine": largest,
    }


def gao(
    item_path="data/Amazon/info/Toys_and_Games_5_2016-10-2018-11.txt",
    model_name: str = "./bge-m3",
    batch_size: int = 256,
    max_len: int = 128,
    block_size: int = 1024,
    device: str = None,
    dedup: bool = False,
    save_path: str = "",
):
    """
    item_path:   info txt of the item set, or a list of them
    model_name:  any sentence-transformers / HF encoder checkpoint
    block_size:  rows per block when summing similarities
    dedup:       drop duplicate item names before pairing
    save_path:   optional json dump of the stats
    """
    if not isinstance(item_path, (list, tuple)):
        item_path = [item_path]

    all_names = {}
    for p in item_path:
        names = load_item_names(p, dedup=dedup)
        all_names[p] = names
        print(f"{p}: {len(names)} items")

    encoder = TextEncoder(model_name, device=device, batch_size=batch_size, max_len=max_len)

    summary = {}
    for p, names in all_names.items():
        embs = encoder.encode(names)
        stats = pairwise_stats(np.ascontiguousarray(embs), block_size=block_size)
        stats["model"] = model_name
        stats["dedup"] = dedup
        summary[p] = stats
        print(f"\n{p}")
        print(f"num_items:\t{stats['num_items']}")
        print(f"num_pairs:\t{stats['num_pairs']}")
        print(f"mean_cosine:\t{stats['mean_cosine']:.4f}")
        print(f"std_cosine:\t{stats['std_cosine']:.4f}")
        print(f"min / max:\t{stats['min_cosine']:.4f} / {stats['max_cosine']:.4f}")

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"\nsaved to {save_path}")
    return summary


if __name__ == "__main__":
    fire.Fire(gao)
