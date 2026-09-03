#!/usr/bin/env python3
"""Track P(QED | n lines written) at every RL round.

The claim in the report is that expert iteration works by lowering the model's
probability of stopping, and that this is what lets proofs get longer. That is a
mechanistic claim about one specific quantity, so it should be visible as a
trajectory across rounds -- not just as a before/after pair.

Method: teacher-force each round's checkpoint along KNOWN-VALID 9-16 line proofs
and read off, at each line boundary, the probability the model assigns to
ending the proof. Nothing is sampled; one forward pass per proof.
"""
import collections, json, os

import torch

from nd.data_torch import load
from nd.evaluate import load_ckpt
from scripts_qed_prior import qed_prior

src = load('data/rl_targets_hard.jsonl')[:1200]
recs = [{'prompt': r['prompt'], 'proof': r['gen_proof']} for r in src if 'gen_proof' in r]
print(f'{len(recs)} known-valid long proofs\n')

ckpts = [('SFT (round 0)', 'ckpt/sft_rope_clean.pt')]
for i in (1, 2, 3, 4):
    p = f'runs/clean_seed0/policy_r{i}.pt'
    if os.path.exists(p):
        ckpts.append((f'after round {i}', p))
ckpts.append(('final (round 5)', 'runs/clean_seed0/final.pt'))

out = {}
for name, path in ckpts:
    m = load_ckpt(path)
    pri = qed_prior(m, recs)
    q = {int(L): sum(v) / len(v) for L, v in pri.items() if L <= 14}
    out[name] = q
    row = '  '.join(f'{q.get(L, float("nan")):.3f}' for L in range(5, 13))
    print(f'{name:16s}  L=5..12: {row}')

json.dump(out, open('numbers_qed_rounds.json', 'w'), indent=1)
print('\nwrote numbers_qed_rounds.json')
