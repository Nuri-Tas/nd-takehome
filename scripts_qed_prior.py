#!/usr/bin/env python3
"""Why does the frontier stop at 7? Measure the model's stopping prior.

Every Stage-1 model writes proofs of at most 7 lines, across ~64k samples each
on genuinely hard targets, regardless of positional scheme. A hard stop rather
than a decay points at the length distribution of the training data rather than
at anything about reasoning or reference tokens.

This tests that directly. We teacher-force each model along a KNOWN verified
proof longer than the cap, and at every line boundary read off the probability
it assigns to ending the proof (`QED`) instead of starting another line. If
cap-6 data has taught a stopping prior, P(QED) climbs steeply around line 6 --
and the model then cannot write a long proof however good its local rule
application is.

Nothing is trained or sampled here; one forward pass per proof.
"""
import argparse, collections, json

import torch
import torch.nn.functional as Fn

from nd.data_torch import load
from nd.evaluate import load_ckpt
from nd.tokenizer import EOS, STOI, encode_example


@torch.no_grad()
def qed_prior(model, recs, device='cuda', bs=256):
    """-> {lines_written: [P(QED) at that boundary, ...]} over the given proofs."""
    out = collections.defaultdict(list)
    for i in range(0, len(recs), bs):
        chunk = recs[i:i + bs]
        seqs, marks = [], []
        for r in chunk:
            toks = encode_example(r['prompt'], r['proof'])
            if toks is None:
                continue
            # positions where a line has just ended: the model is about to
            # choose between another line and QED
            ends = [j for j, t in enumerate(toks) if t == ';']
            seqs.append([STOI[t] for t in toks])
            marks.append(ends)
        if not seqs:
            continue
        T = max(len(s) for s in seqs)
        x = torch.full((len(seqs), T), STOI['<pad>'], dtype=torch.long)
        for j, s in enumerate(seqs):
            x[j, :len(s)] = torch.tensor(s)
        x = x.to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
            logits, _ = model(x)
        probs = Fn.softmax(logits.float(), -1)
        for j, ends in enumerate(marks):
            for line_no, pos in enumerate(ends, start=1):
                if pos < probs.size(1):
                    out[line_no].append(float(probs[j, pos, EOS]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', nargs='+', required=True, help='name=path')
    ap.add_argument('--proofs', default='data/rl_targets_hard.jsonl',
                    help='records carrying a verified long proof in `gen_proof`')
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--out', default='numbers_qed_prior.json')
    a = ap.parse_args()

    src = load(a.proofs)[:a.n]
    recs = [{'prompt': r['prompt'], 'proof': r['gen_proof']}
            for r in src if 'gen_proof' in r]
    print(f'{len(recs)} known-good proofs longer than the cap\n')

    res = {}
    for spec in a.ckpts:
        name, path = spec.split('=', 1)
        m = load_ckpt(path)
        pri = qed_prior(m, recs)
        row = {}
        print(f'{name}  (pos_mode={getattr(m.c, "pos_mode", "learned")})')
        print('   after line :  mean P(QED)      n')
        for L in sorted(pri):
            if L > 16:
                continue
            v = pri[L]
            mean = sum(v) / len(v)
            row[L] = [mean, len(v)]
            print(f'   {L:>10} :  {mean:.3f}  {len(v):>6}  {"#" * int(60 * mean)}')
        print()
        res[name] = row

    with open(a.out, 'w') as f:
        json.dump(res, f, indent=1)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
