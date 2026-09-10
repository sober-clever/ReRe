"""Compute the average cosine similarity between predicted items and the target item.

Usage:
    python calc_sim.py --path results/rere_industrial_seed0_rule-final_result.json \
        --model_name sentence-transformers/all-MiniLM-L6-v2 --topk 10

For every sample the similarity is averaged over its predictions first (weighted by
rank), then the per-sample scores are averaged over the dataset.
"""
import json
import os

import fire
import numpy as np
import torch
from tqdm import tqdm


def clean(text):
    """Strip the quotes / whitespace that the generation pipeline leaves around item names."""
    return text.strip().strip("\"").strip()


def get_target(sample):
    output = sample["output"]
    if isinstance(output, list):
        output = output[0]
    return clean(output)


def get_weights(ranks, scheme="reciprocal"):
    """Rank weights for a list of 1-based ranks.

    reciprocal: 1 / (n + 1)        -> 1/2, 1/3, 1/4, ...
    log:        1 / log2(n + 1)    -> the NDCG discount
    none:       1                  -> plain average
    """
    ranks = np.asarray(ranks, dtype=np.float64)
    if scheme == "reciprocal":
        return 1.0 / (ranks + 1.0)
    if scheme == "log":
        return 1.0 / np.log2(ranks + 1.0)
    if scheme == "none":
        return np.ones_like(ranks)
    raise ValueError(f"unknown weight_scheme: {scheme}")


class TextEncoder:
    """Sentence embedding model. Uses sentence-transformers when available,
    otherwise falls back to a plain transformers model with mean pooling."""

    def __init__(self, model_name, device=None, batch_size=256, max_len=128):
        self.batch_size = batch_size
        self.max_len = max_len
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        print(f"Loading embedding model {model_name} on {device}")

        self.st_model = None
        try:
            from sentence_transformers import SentenceTransformer
            self.st_model = SentenceTransformer(model_name, device=device)
        except ImportError:
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(device)
            self.model.eval()
        print("Embedding model loaded")

    @torch.no_grad()
    def encode(self, texts):
        """Return L2-normalized embeddings, shape (len(texts), dim)."""
        if self.st_model is not None:
            return self.st_model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            )

        embs = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="encoding"):
            batch = texts[i: i + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
            embs.append(pooled.float().cpu().numpy())
        return np.concatenate(embs, axis=0)


def gao(
    path="./ckpt/rere_toys_seed0_rule/checkpoint-7048/final_result.json",
    model_name: str = "./bge-m3",
    topk: int = 10,
    batch_size: int = 256,
    max_len: int = 128,
    device: str = None,
    exclude_hit: bool = False,
    weight_scheme: str = "reciprocal",
    save_path: str = "",
):
    """
    path:           one result json, or a list of them
    model_name:     any sentence-transformers / HF encoder checkpoint
    topk:           only use the first topk predictions of each sample (0 = all)
    exclude_hit:    skip predictions that exactly match the target item
    weight_scheme:  rank weighting, "reciprocal" 1/(n+1) | "log" 1/log2(n+1) | "none"
    save_path:      optional json dump of the per-file / per-sample scores
    """
    if not isinstance(path, (list, tuple)):
        path = [path]

    # Collect every distinct string once so the encoder never sees a duplicate.
    all_data = {}
    unique_texts = {}

    def register(text):
        if text not in unique_texts:
            unique_texts[text] = len(unique_texts)
        return unique_texts[text]

    for p in path:
        with open(p, "r") as f:
            test_data = json.load(f)
        samples = []
        for sample in test_data:
            target = get_target(sample)
            preds = [clean(_) for _ in sample["predict"]]
            if topk > 0:
                preds = preds[:topk]
            # rank is 1-based and always the original position, so dropping an
            # exact hit does not promote the predictions that follow it
            ranked = [(register(_), n) for n, _ in enumerate(preds, start=1)
                      if not (exclude_hit and _ == target)]
            samples.append((register(target), ranked))
        all_data[p] = samples
        print(f"{p}: {len(samples)} samples")

    texts = [None] * len(unique_texts)
    for text, idx in unique_texts.items():
        texts[idx] = text
    print(f"{len(texts)} unique item names to encode")

    encoder = TextEncoder(model_name, device=device, batch_size=batch_size, max_len=max_len)
    embs = encoder.encode(texts)

    summary = {}
    for p, samples in all_data.items():
        per_sample = []
        for target_idx, ranked in samples:
            if len(ranked) == 0:
                continue
            pred_idxs = [_[0] for _ in ranked]
            weights = get_weights([_[1] for _ in ranked], weight_scheme)
            # embeddings are normalized, so a dot product is the cosine similarity
            sims = embs[pred_idxs] @ embs[target_idx]
            # normalize by the weight sum so the score stays on the cosine scale
            per_sample.append(float(np.dot(sims, weights) / weights.sum()))
        mean_sim = float(np.mean(per_sample)) if per_sample else float("nan")
        summary[p] = {
            "model": model_name,
            "topk": topk if topk > 0 else max(len(_[1]) for _ in samples),
            "weight_scheme": weight_scheme,
            "num_samples": len(per_sample),
            "mean_cosine": mean_sim,
            "std_over_samples": float(np.std(per_sample)) if per_sample else float("nan"),
        }
        print(f"\n{p}")
        print(f"weight_scheme:\t{weight_scheme}")
        print(f"num_samples:\t{len(per_sample)}")
        print(f"mean_cosine:\t{mean_sim:.4f}")
        print(f"std:\t\t{summary[p]['std_over_samples']:.4f}")
        if save_path:
            summary[p]["per_sample"] = per_sample

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"\nsaved to {save_path}")
    return summary


if __name__ == "__main__":
    fire.Fire(gao)
