#!/usr/bin/env python3
"""Score a checkpoint on targets/validation_36.jsonl, split by bin.

Also reports any proof we wrote that is SHORTER than the set's `min_lines_ub`
upper bound, which the README asks about explicitly.
"""
import argparse, collections, json

from nd.data_torch import load
from nd.evaluate import judge, load_ckpt, sample_proofs, wilson


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='ckpt/sft.pt')
    ap.add_argument('--data', default='targets/validation_36.jsonl')
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--show', action='store_true', help='print each verdict')
    ap.add_argument('--dump', default=None, help='write proofs jsonl here')
    a = ap.parse_args()

    recs = load(a.data)
    m = load_ckpt(a.ckpt)
    prompts = [r['prompt'] for r in recs]
    bodies = sample_proofs(m, prompts, temperature=a.temperature)
    res = judge(prompts, bodies)

    by_bin = collections.defaultdict(lambda: [0, 0])
    shorter = []
    for r, b, (ok, reason, n) in zip(recs, bodies, res):
        s = by_bin[r['bin']]
        s[0] += ok; s[1] += 1
        if ok and 'min_lines_ub' in r and n < r['min_lines_ub']:
            shorter.append((r['name'], n, r['min_lines_ub']))
        if a.show:
            print(f"  {'OK ' if ok else '   '} {r['name']:28s} bin={r['bin']:>3s} "
                  f"ub={r.get('min_lines_ub','?'):>3} wrote={n if ok else '-'}"
                  f"{'' if ok else '  ' + reason}")

    tot = sum(ok for ok, _, _ in res)
    print(f'\nvalidation_36 total: {tot}/{len(recs)}')
    for b in sorted(by_bin):
        k, n = by_bin[b]
        p, lo, hi = wilson(k, n)
        print(f'  bin {b:>3s}: {k}/{n}  {100*p:.1f}% [{100*lo:.1f}-{100*hi:.1f}]')
    if shorter:
        print('proofs SHORTER than the published upper bound:')
        for name, n, ub in shorter:
            print(f'  {name}: wrote {n}, ub was {ub}')

    if a.dump:
        with open(a.dump, 'w') as f:
            for r, b in zip(recs, bodies):
                f.write(json.dumps({'name': r['name'], 'prompt': r['prompt'],
                                    'proof': b}) + '\n')


if __name__ == '__main__':
    main()
