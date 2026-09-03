#!/usr/bin/env python3
"""Compare Stage-1 checkpoints by the length of the proofs they can WRITE.

This is the measurement of P, and the test of whether the positional scheme is
a length barrier. It reports, for each checkpoint, over a large sampling budget
on beyond-cap targets:

  * the histogram of written lengths among verified proofs;
  * the robust frontier -- the longest written length with >= 5 DISTINCT
    verified proofs (a single lucky long proof is an anecdote, not a
    capability);
  * the single longest verified proof, reported separately and labelled as the
    anecdote it is.

Nothing here trains anything; it only samples and verifies.
"""
import argparse, collections, json

import torch

from nd.data_torch import load
from nd.evaluate import judge, load_ckpt, sample_proofs

ROBUST_K = 5


def probe(model, prompts, k, temperature, bs, max_new):
    by_len = collections.defaultdict(set)      # written length -> distinct proofs
    solved = set()
    for _ in range(k):
        bodies = sample_proofs(model, prompts, temperature=temperature, bs=bs,
                               max_new=max_new)
        for p, b, (ok, _, n) in zip(prompts, bodies, judge(prompts, bodies)):
            if ok:
                solved.add(p)
                by_len[n].add(p + ' ||| ' + ' '.join(b.split()))
    return solved, by_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', nargs='+', required=True, help='name=path')
    ap.add_argument('--targets', default='data/rl_targets.jsonl')
    ap.add_argument('--n', type=int, default=1500)
    ap.add_argument('--k', type=int, default=32)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--bs', type=int, default=2048)
    ap.add_argument('--max_new', type=int, default=380)
    ap.add_argument('--out', default='numbers_length_probe.json')
    a = ap.parse_args()

    recs = load(a.targets)[:a.n]
    prompts = [r['prompt'] for r in recs]
    print(f'{len(prompts)} targets, k={a.k} at T={a.temperature}\n')

    out = {}
    for spec in a.ckpts:
        name, path = spec.split('=', 1)
        m = load_ckpt(path)
        mode = getattr(m.c, 'pos_mode', 'learned')
        solved, by_len = probe(m, prompts, a.k, a.temperature, a.bs, a.max_new)
        counts = {int(L): len(s) for L, s in sorted(by_len.items())}
        robust = max([L for L, s in by_len.items() if len(s) >= ROBUST_K],
                     default=0)
        longest = max(by_len) if by_len else 0
        beyond = sum(c for L, c in counts.items() if L > 6)
        print(f'{name:14s} (pos_mode={mode})')
        print(f'   solved            {len(solved)}/{len(prompts)}')
        print(f'   written lengths   {counts}')
        print(f'   proofs > 6 lines  {beyond}')
        print(f'   robust frontier   {robust}   (>= {ROBUST_K} distinct proofs)')
        print(f'   longest single    {longest}   (anecdote, not a rate)\n')
        out[name] = {'pos_mode': mode, 'solved': len(solved),
                     'n': len(prompts), 'written_lengths': counts,
                     'beyond_6': beyond, 'robust_frontier': robust,
                     'longest_single': longest, 'k': a.k,
                     'temperature': a.temperature}

    with open(a.out, 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
