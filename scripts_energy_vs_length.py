#!/usr/bin/env python3
"""Measure the ACTUAL total surprisal E(y) = -log p(y|x) of verified proofs,
as a function of proof length, for Stage-1 and the RL model.

The reachability criterion is E(y) <= log k. E_stop (the stopping-prior term)
is only a lower bound on E, so this measures the whole thing: every token of a
verified proof, teacher-forced, summed.

min_y E(y) over proofs of length L is what actually determines whether an
L-line proof is sampleable at budget k.
"""
import collections, json, math

import torch
import torch.nn.functional as Fn

from nd.evaluate import load_ckpt
from nd.tokenizer import STOI, encode_example


@torch.no_grad()
def energies(model, recs, bs=128, device='cuda'):
    """-> list of (n_lines, total surprisal in nats) for each record."""
    out = []
    for i in range(0, len(recs), bs):
        chunk = recs[i:i + bs]
        seqs, meta = [], []
        for r in chunk:
            toks = encode_example(r['prompt'], r['proof'])
            if toks is None:
                continue
            seqs.append([STOI[t] for t in toks])
            meta.append((r['n_lines'], 1 + len(r['prompt'].split())))
        if not seqs:
            continue
        T = max(len(s) for s in seqs)
        x = torch.full((len(seqs), T), STOI['<pad>'], dtype=torch.long)
        for j, s in enumerate(seqs):
            x[j, :len(s)] = torch.tensor(s)
        x = x.to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lg, _ = model(x)
        logp = Fn.log_softmax(lg.float(), -1)
        for j, (nl, npr) in enumerate(meta):
            n = len(seqs[j])
            # sum -log p over the proof body only (prompt is the condition)
            tot = 0.0
            for t in range(npr - 1, n - 1):
                tot -= float(logp[j, t, seqs[j][t + 1]])
            out.append((nl, tot))
    return out


def main():
    pool = [json.loads(l) for l in open('runs/ei_seed0/pool.jsonl')]
    # cap per length so the comparison is not dominated by the common lengths
    byl = collections.defaultdict(list)
    for r in pool:
        if len(byl[r['n_lines']]) < 400:
            byl[r['n_lines']].append(r)
    recs = [r for v in byl.values() for r in v]
    print(f'{len(recs)} verified proofs, lengths {sorted(byl)}\n')

    res = {}
    for name, path in (('stage1', 'ckpt/sft_rope.pt'),
                       ('after_RL', 'runs/ei_seed0/final.pt')):
        m = load_ckpt(path)
        e = energies(m, recs)
        g = collections.defaultdict(list)
        for nl, tot in e:
            g[nl].append(tot)
        res[name] = {int(L): {'min': min(v), 'median': sorted(v)[len(v)//2],
                              'n': len(v)} for L, v in sorted(g.items())}
        print(f'--- {name} ---')
        print('  L    n     min E(y)   median E(y)   log k needed to reach min')
        for L in sorted(g):
            v = g[L]
            mn = min(v)
            print('  %2d  %4d   %8.2f   %10.2f   k >= %.3g' % (
                L, len(v), mn, sorted(v)[len(v)//2], math.exp(mn)))
        print()
    json.dump(res, open('numbers_energy_vs_length.json', 'w'), indent=1)
    print('wrote numbers_energy_vs_length.json')


if __name__ == '__main__':
    main()
