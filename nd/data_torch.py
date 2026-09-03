"""Pack jsonl proof records into padded tensors, masking the prompt.

Loss is computed on the proof body only: the prompt is the condition, not
something we want the model to spend capacity predicting.
"""
import json

import torch

from .tokenizer import PAD, STOI, encode_example, n_premises


def load(path):
    return [json.loads(l) for l in open(path)]


def encode_records(recs, block):
    """-> (x, y) int64 tensors, y = -100 wherever loss is masked out."""
    seqs = []
    for r in recs:
        toks = encode_example(r['prompt'], r['proof'])
        if toks is None or len(toks) > block:
            continue
        ids = [STOI[t] for t in toks]
        n_prompt = 1 + len(r['prompt'].split())
        seqs.append((ids, n_prompt))
    if not seqs:
        raise ValueError('no encodable records')
    T = max(len(s) for s, _ in seqs)
    x = torch.full((len(seqs), T - 1), PAD, dtype=torch.long)
    y = torch.full((len(seqs), T - 1), -100, dtype=torch.long)
    for i, (ids, npr) in enumerate(seqs):
        n = len(ids)
        x[i, :n - 1] = torch.tensor(ids[:-1])
        tgt = torch.tensor(ids[1:])
        y[i, :n - 1] = tgt
        y[i, :npr - 1] = -100          # do not train on predicting the prompt
    return x, y


def batches(x, y, bs, shuffle=True, device='cuda'):
    n = x.size(0)
    order = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        j = order[i:i + bs]
        yield x[j].to(device, non_blocking=True), y[j].to(device, non_blocking=True)
