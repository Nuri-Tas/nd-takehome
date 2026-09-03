#!/usr/bin/env python3
"""Energy-entropy decomposition of proof search.

A theorem x is proved if ANY of its valid proofs is sampled, so the relevant
quantity is not the energy of one proof but the free energy of the whole
proof set Y(x):

    p(prove x) = sum over y in Y(x) of exp(-E(y))  =  exp(-F(x))
    F(x) = -log sum_y exp(-E(y))

If the reachable proofs have comparable energy E, then F = E - log g, where
g = |Y(x)| is the DEGENERACY (number of distinct proofs). So

    log p(prove x)  =  log g  -  E

i.e. at fixed energy, a theorem with more distinct proofs is exponentially
easier, with slope 1 in log g. This is the same entropy term that makes a
macrostate with many microstates dominate a partition function.

This script measures both sides: for each theorem, the empirical solve
probability, the degeneracy actually observed, and the minimum energy.
"""
import collections, json, math, sys

import torch
import torch.nn.functional as Fn

from nd.data_torch import load
from nd.evaluate import judge, load_ckpt, sample_proofs
from nd.tokenizer import STOI, encode_example

K = int(sys.argv[1]) if len(sys.argv) > 1 else 64
N = int(sys.argv[2]) if len(sys.argv) > 2 else 800
CK = sys.argv[3] if len(sys.argv) > 3 else 'runs/clean_seed0/final.pt'

tg = load('data/rl_targets_hard.jsonl')[:N]
ps = [r['prompt'] for r in tg]
m = load_ckpt(CK)

succ = collections.Counter()          # theorem -> number of accepted samples
proofs = collections.defaultdict(set)  # theorem -> set of distinct proof texts
for i in range(K):
    bodies = sample_proofs(m, ps, temperature=1.0, bs=800, max_new=380)
    for p, b, (ok, _, n) in zip(ps, bodies, judge(ps, bodies)):
        if ok:
            succ[p] += 1
            proofs[p].add(' '.join(b.split()))
    print(f'  draw {i+1}/{K}', flush=True)


@torch.no_grad()
def energy(prompt, body):
    toks = encode_example(prompt, body)
    if toks is None:
        return None
    ids = [STOI[t] for t in toks]
    x = torch.tensor([ids], device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        lg, _ = m(x)
    lp = Fn.log_softmax(lg.float(), -1)[0]
    npr = 1 + len(prompt.split())
    return float(-sum(lp[t, ids[t + 1]] for t in range(npr - 1, len(ids) - 1)))


rows = []
for p in ps:
    if not proofs[p]:
        continue
    g = len(proofs[p])
    e = min(x for x in (energy(p, b) for b in list(proofs[p])[:12]) if x is not None)
    rows.append({'p_hat': succ[p] / K, 'g': g, 'E_min': e, 'n_succ': succ[p]})

print(f'\n{len(rows)} theorems solved at least once out of {N}, k={K}\n')
print('  Grouped by observed degeneracy g = number of DISTINCT proofs found:')
print('   g range     n   mean p_hat   mean E_min   log g   log p_hat')
buckets = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 64)]
out = []
for lo, hi in buckets:
    sel = [r for r in rows if lo <= r['g'] <= hi]
    if len(sel) < 5:
        continue
    mp = sum(r['p_hat'] for r in sel) / len(sel)
    me = sum(r['E_min'] for r in sel) / len(sel)
    mg = sum(r['g'] for r in sel) / len(sel)
    print('  %3d-%-3d %5d   %9.3f   %10.3f  %6.2f   %8.2f'
          % (lo, hi, len(sel), mp, me, math.log(mg), math.log(max(mp, 1e-9))))
    out.append({'g_lo': lo, 'g_hi': hi, 'n': len(sel), 'p_hat': mp,
                'E_min': me, 'log_g': math.log(mg), 'log_p': math.log(max(mp, 1e-9))})

# least-squares fit of  log p_hat = a*log g + b*E_min + c
import statistics
if len(rows) > 50:
    xs = [(math.log(r['g']), r['E_min'], math.log(max(r['p_hat'], 1e-9))) for r in rows]
    n = len(xs)
    mg = statistics.mean(a for a, _, _ in xs)
    me = statistics.mean(b for _, b, _ in xs)
    mp = statistics.mean(c for _, _, c in xs)
    sgg = sum((a - mg) ** 2 for a, _, _ in xs)
    sgp = sum((a - mg) * (c - mp) for a, _, c in xs)
    see = sum((b - me) ** 2 for _, b, _ in xs)
    sep = sum((b - me) * (c - mp) for _, b, c in xs)
    print(f'\n  univariate slope  d log p / d log g   = {sgp/sgg:+.3f}   (theory: +1)')
    print(f'  univariate slope  d log p / d E_min   = {sep/see:+.3f}   (theory: -1)')
json.dump(out, open('numbers_degeneracy.json', 'w'), indent=1)
print('\nwrote numbers_degeneracy.json')
