#!/usr/bin/env python3
"""Test the prediction that the frontier grows logarithmically with the budget.

Reachability of a proof y needs k*p(y) >~ 1, i.e. E(y) <= log k, where
E = -log p is total surprisal. If the length-dependent part of E grows with
slope c nats per line past the cap, the frontier solves E(L*) = log k, giving

    L*(k) ~ L_0 + log(k)/c

so each extra line of frontier costs a factor e^c more samples. Sampling once
with a large k and then evaluating prefixes gives L*(k) for every smaller k at
no extra cost.
"""
import collections, json, sys

import torch

from nd.data_torch import load
from nd.evaluate import judge, load_ckpt, sample_proofs

K = int(sys.argv[1]) if len(sys.argv) > 1 else 64
ROBUST = 5
tg = load('data/rl_targets_hard.jsonl')[:2000]
ps = [r['prompt'] for r in tg]
out = {}

for name, path in (('stage1', 'ckpt/sft_rope.pt'),
                   ('after_RL', 'runs/ei_seed0/final.pt')):
    m = load_ckpt(path)
    per_draw = []                      # per_draw[i] = {length: set(proof keys)}
    for i in range(K):
        d = collections.defaultdict(set)
        bodies = sample_proofs(m, ps, temperature=1.0, bs=2000, max_new=380)
        for p, b, (ok, _, n) in zip(ps, bodies, judge(ps, bodies)):
            if ok:
                d[n].add(p + '|||' + ' '.join(b.split()))
        per_draw.append(d)
        print(f'  {name} draw {i+1}/{K}', flush=True)
    curve = {}
    cum = collections.defaultdict(set)
    for i, d in enumerate(per_draw, start=1):
        for L, s in d.items():
            cum[L] |= s
        robust = max([L for L, s in cum.items() if len(s) >= ROBUST], default=0)
        longest = max([L for L, s in cum.items() if s], default=0)
        curve[i] = {'robust': robust, 'longest': longest,
                    'counts': {int(L): len(s) for L, s in sorted(cum.items())}}
    out[name] = curve
    print(f'{name}: frontier at k=1,2,4,8,16,32,{K}: '
          + ', '.join(str(curve[j]['robust']) for j in (1, 2, 4, 8, 16, 32, K) if j in curve),
          flush=True)

json.dump(out, open('numbers_frontier_vs_k.json', 'w'), indent=1)
print('wrote numbers_frontier_vs_k.json')
