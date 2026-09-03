#!/usr/bin/env python3
"""ORACLE CONTROL: what if we simply had long proofs to train on?

The objection this answers: "if you did supervised learning on long proofs you
would also get long proofs, so what does RL add?"

The answer turns on where the data comes from. The exam forbids supervised
training on proofs longer than 6 lines, and no external source of them exists --
that is the whole premise. RL's contribution is that it MANUFACTURES that data:
the model samples, the verifier labels, and the accepted samples become training
examples that did not exist before.

This control breaks that rule deliberately, as analysis rather than as a
submission model. It hands a model the generator's own 9-16 line proofs for the
RL targets -- gold data RL never had -- and measures the frontier on the
transfer set, which neither arm ever trains on. Comparing:

    Stage 1        : cap-6 data only                      -> frontier P
    RL (5 rounds)  : cap-6 + ~13k SELF-FOUND proofs       -> frontier L
    oracle SFT     : cap-6 + 1994 GOLD long proofs        -> frontier L_oracle

If oracle SFT >> RL, the bottleneck is data quality and RL is an inefficient way
to get it. If they are comparable, RL recovered most of what gold data would
have given. Either way the comparison is only meaningful because the transfer
set is held out from both.
"""
import argparse, collections, json, math, os, random, time

import torch

from nd.data_torch import batches, encode_records, load
from nd.evaluate import judge, load_ckpt, sample_proofs
from nd.model import GPT, Config
from nd_verify import verify_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=1200)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--block', type=int, default=512)
    ap.add_argument('--sft_mix', type=int, default=20000)
    ap.add_argument('--k', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    dev = 'cuda'

    # gold long proofs for the RL targets -- exactly the theorems RL practised on
    gold = [{'prompt': r['prompt'], 'proof': r['gen_proof'], 'n_lines': r['gen_lines']}
            for r in load('data/rl_targets_hard.jsonl') if 'gen_proof' in r]
    bad = sum(1 for r in gold if not verify_text(r['prompt'] + ' ' + r['proof'])[0])
    print(f'{len(gold)} gold long proofs, {bad} invalid')
    c = collections.Counter(r['n_lines'] for r in gold)
    print('gold length distribution:', dict(sorted(c.items())))

    mix = rng.sample(load('data/train.jsonl'), a.sft_mix)
    recs = gold + mix
    rng.shuffle(recs)
    print(f'training on {len(gold)} gold long + {len(mix)} cap-6 = {len(recs)}\n')

    ck = torch.load('ckpt/sft_rope_clean.pt', map_location=dev, weights_only=False)
    m = GPT(Config(**ck['cfg'])).to(dev)
    m.load_state_dict(ck['model'])
    m.train()
    x, y = encode_records(recs, a.block)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.1, betas=(0.9, 0.95))
    step, t0 = 0, time.time()
    while step < a.steps:
        for xb, yb in batches(x, y, a.bs, device=dev):
            if step >= a.steps:
                break
            for g in opt.param_groups:
                g['lr'] = a.lr * min(1.0, (step + 1) / 100) * \
                    0.5 * (1 + math.cos(math.pi * step / a.steps))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, loss = m(xb, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            step += 1
    m.eval()
    torch.save({'cfg': ck['cfg'], 'model': m.state_dict()}, 'ckpt/oracle_sft.pt')
    print(f'trained {a.steps} steps in {time.time()-t0:.0f}s')

    # frontier on the TRANSFER set -- held out from every arm
    tf = load('data/transfer_hard.jsonl')
    ps = [r['prompt'] for r in tf]
    by_len, solved = collections.defaultdict(set), set()
    for _ in range(a.k):
        bodies = sample_proofs(m, ps, temperature=1.0, bs=800, max_new=380)
        for p, b, (ok, _, n) in zip(ps, bodies, judge(ps, bodies)):
            if ok:
                solved.add(p)
                by_len[n].add(p + '|||' + ' '.join(b.split()))
    counts = {int(L): len(s) for L, s in sorted(by_len.items())}
    robust = max([L for L, s in by_len.items() if len(s) >= 5], default=0)

    # greedy too, for comparability with the reported table
    gb = sample_proofs(m, ps, temperature=0.0, bs=800, max_new=380)
    gok = sum(o for o, _, _ in judge(ps, gb))

    print(f'\nORACLE SFT on the transfer set (n={len(ps)}, k={a.k}, T=1.0)')
    print(f'  greedy solve      {gok}/{len(ps)} = {100*gok/len(ps):.1f}%')
    print(f'  pass@{a.k} solved    {len(solved)}/{len(ps)} = {100*len(solved)/len(ps):.1f}%')
    print(f'  written lengths   {counts}')
    print(f'  robust frontier   {robust}')
    json.dump({'greedy': [gok, len(ps)], 'passk': [len(solved), len(ps), a.k],
               'written_lengths': counts, 'robust_frontier': robust,
               'n_gold': len(gold)}, open('numbers_oracle_sft.json', 'w'), indent=1)
    print('wrote numbers_oracle_sft.json')


if __name__ == '__main__':
    main()
