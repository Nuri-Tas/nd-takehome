#!/usr/bin/env python3
"""Frontier probe for the trained absolute-reference model (H1)."""
import collections, json

import torch

from ablation_abs_refs import ITOS, PAD, V, sample
from nd.evaluate import load_ckpt
from nd_verify import verify_text

K = 32
m = load_ckpt('ckpt/sft_absrefs.pt')
tg = [json.loads(l) for l in open('data/rl_targets_hard.jsonl')][:2000]
ps = [r['prompt'] for r in tg]
by_len, solved = collections.defaultdict(set), set()
for i in range(K):
    for p, b in zip(ps, sample(m, ps, 1.0, 2000, 380)):
        ok, _, n = verify_text(p + ' ' + b)
        if ok:
            solved.add(p)
            by_len[n].add(p + '|||' + ' '.join(b.split()))
counts = {int(L): len(s) for L, s in sorted(by_len.items())}
robust = max([L for L, s in by_len.items() if len(s) >= 5], default=0)
print('ABSOLUTE-REF model, 2000 hard targets, k=%d, T=1.0' % K)
print('  solved          %d/2000' % len(solved))
print('  written lengths', counts)
print('  robust frontier', robust)
print('  longest single ', max(by_len) if by_len else 0)
json.dump({'solved': len(solved), 'written_lengths': counts,
           'robust_frontier': robust, 'heldout_greedy': 0.966, 'vocab': V},
          open('numbers_ablation_absrefs.json', 'w'), indent=1)
print('wrote numbers_ablation_absrefs.json')
