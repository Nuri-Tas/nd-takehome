#!/usr/bin/env python3
"""Stage 3: one table, rows = checkpoints, columns = every evaluation set.

  python eval_all.py --ckpts sft=ckpt/sft.pt final=runs/ei/final.pt

Reports, per checkpoint:
  * Stage-1 held-out (<=6 lines), greedy, overall and by proof length;
  * the transfer pool (never sampled during RL), greedy and pass@k;
  * validation_36 split by its `bin` field;
  * the written-length histogram of everything it proved.

Every rate is greedy unless a pass@k column says otherwise, and every count is
of proofs the verifier accepted for the prompted sequent.
"""
import argparse, collections, json

import torch

from nd.data_torch import load
from nd.evaluate import judge, load_ckpt, sample_proofs, wilson


def rate(model, recs, temperature=0.0, bs=2048, k=1, max_new=300):
    """-> (n_solved, n, {written_length: count}) over k attempts per theorem."""
    prompts = [r['prompt'] for r in recs]
    solved = set()
    lens = collections.Counter()
    for _ in range(k):
        bodies = sample_proofs(model, prompts, temperature=temperature, bs=bs,
                               max_new=max_new)
        for p, (ok, _, n) in zip(prompts, judge(prompts, bodies)):
            if ok and p not in solved:
                solved.add(p)
                lens[n] += 1
    return len(solved), len(prompts), lens


def fmt(k, n):
    p, lo, hi = wilson(k, n)
    return f'{100*p:5.1f}% [{100*lo:4.1f}-{100*hi:4.1f}]  {k}/{n}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', nargs='+', required=True,
                    help='name=path pairs')
    ap.add_argument('--heldout', default='data/heldout.jsonl')
    ap.add_argument('--transfer', default='data/transfer_final.jsonl')
    ap.add_argument('--validation', default='targets/validation_36.jsonl')
    ap.add_argument('--n_heldout', type=int, default=1500)
    ap.add_argument('--passk', type=int, default=32)
    ap.add_argument('--bs', type=int, default=2048)
    ap.add_argument('--out', default='numbers_stage3.json')
    a = ap.parse_args()

    ho = load(a.heldout)[:a.n_heldout]
    tf = load(a.transfer)
    va = load(a.validation)
    results = {}

    for spec in a.ckpts:
        name, path = spec.split('=', 1)
        m = load_ckpt(path)
        print(f'\n=== {name}  ({path}) ===')
        r = {}

        k, n, lens = rate(m, ho, k=1)
        r['heldout_greedy'] = [k, n]
        print(f'  held-out (<=6)  greedy   {fmt(k, n)}')
        by = collections.defaultdict(lambda: [0, 0])
        prompts = [x['prompt'] for x in ho]
        bodies = sample_proofs(m, prompts, temperature=0.0, bs=a.bs)
        for x, (ok, _, _) in zip(ho, judge(prompts, bodies)):
            s = by[x['n_lines']]
            s[0] += ok; s[1] += 1
        r['heldout_by_len'] = {int(L): v for L, v in sorted(by.items())}
        for L in sorted(by):
            print(f'      len {L}: {fmt(*by[L])}')

        k, n, lens = rate(m, tf, k=1)
        r['transfer_greedy'] = [k, n]
        r['transfer_greedy_lens'] = dict(sorted(lens.items()))
        print(f'  transfer (>6)   greedy   {fmt(k, n)}   written {dict(sorted(lens.items()))}')
        k, n, lens = rate(m, tf, temperature=1.0, k=a.passk)
        r['transfer_passk'] = [k, n, a.passk]
        r['transfer_passk_lens'] = dict(sorted(lens.items()))
        print(f'  transfer (>6)   pass@{a.passk:<3d} {fmt(k, n)}   written {dict(sorted(lens.items()))}')

        prompts = [x['prompt'] for x in va]
        bodies = sample_proofs(m, prompts, temperature=0.0, bs=a.bs)
        vb = collections.defaultdict(lambda: [0, 0])
        vlens = collections.Counter()
        shorter = []
        for x, (ok, _, nn) in zip(va, judge(prompts, bodies)):
            s = vb[x['bin']]
            s[0] += ok; s[1] += 1
            if ok:
                vlens[nn] += 1
                if nn < x.get('min_lines_ub', 99):
                    shorter.append((x['name'], nn, x['min_lines_ub']))
        r['validation'] = {b: v for b, v in vb.items()}
        r['validation_lens'] = dict(sorted(vlens.items()))
        tot = sum(v[0] for v in vb.values())
        print(f'  validation_36   greedy   {tot}/{len(va)}')
        for b in sorted(vb):
            print(f'      bin {b:>3s}: {fmt(*vb[b])}')
        if shorter:
            r['shorter_than_ub'] = shorter
            print('      shorter than published upper bound:', shorter)

        results[name] = r

    with open(a.out, 'w') as f:
        json.dump(results, f, indent=1)
    print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()
