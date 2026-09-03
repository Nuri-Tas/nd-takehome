#!/usr/bin/env python3
"""Greedy held-out solve rate, broken down by proof length, with Wilson CIs."""
import argparse, collections, json

from nd.data_torch import load
from nd.evaluate import judge, load_ckpt, sample_proofs, wilson


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='ckpt/sft.pt')
    ap.add_argument('--data', default='data/heldout.jsonl')
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--reasons', action='store_true')
    a = ap.parse_args()

    recs = load(a.data)[:a.n]
    m = load_ckpt(a.ckpt)
    prompts = [r['prompt'] for r in recs]
    bodies = sample_proofs(m, prompts, temperature=a.temperature)
    res = judge(prompts, bodies)

    tot = sum(ok for ok, _, _ in res)
    p, lo, hi = wilson(tot, len(res))
    print(f'overall  {100*p:5.1f}%  [{100*lo:.1f}-{100*hi:.1f}]  n={len(res)}')

    by = collections.defaultdict(lambda: [0, 0])
    nontriv = [0, 0]
    for r, (ok, _, _) in zip(recs, res):
        b = by[r['n_lines']]
        b[0] += ok; b[1] += 1
        if not r['trivial']:
            nontriv[0] += ok; nontriv[1] += 1
    for L in sorted(by):
        k, n = by[L]
        p, lo, hi = wilson(k, n)
        print(f'  len {L}   {100*p:5.1f}%  [{100*lo:.1f}-{100*hi:.1f}]  n={n}')
    p, lo, hi = wilson(*nontriv)
    print(f'non-trivial only  {100*p:5.1f}%  [{100*lo:.1f}-{100*hi:.1f}]  n={nontriv[1]}')

    if a.reasons:
        c = collections.Counter(reason for ok, reason, _ in res if not ok)
        print('failure reasons:', c.most_common(8))


if __name__ == '__main__':
    main()
