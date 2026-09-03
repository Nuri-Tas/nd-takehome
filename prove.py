#!/usr/bin/env python3
"""Sampling interface (see submission_template/prove.py).

    python prove.py --ckpt <path> --in targets.jsonl --out proofs.jsonl [--greedy | --temperature T --seed S]

One proof per theorem, raw model output. The verifier is deliberately not
called here: no filtering, no retrying, no re-sampling.
"""
import argparse, json

import torch

from nd.evaluate import load_ckpt, sample_proofs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--greedy', action='store_true')
    ap.add_argument('--temperature', type=float, default=None)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    recs = [json.loads(l) for l in open(a.inp) if l.strip()]
    temp = 0.0 if (a.greedy or a.temperature is None) else a.temperature

    model = load_ckpt(a.ckpt, device=dev)
    bodies = sample_proofs(model, [r['prompt'] for r in recs], device=dev,
                           temperature=temp)
    with open(a.out, 'w') as f:
        for r, b in zip(recs, bodies):
            f.write(json.dumps({'name': r.get('name'), 'prompt': r['prompt'],
                                'proof': b}) + '\n')


if __name__ == '__main__':
    main()
