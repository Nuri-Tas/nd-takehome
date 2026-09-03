#!/usr/bin/env python3
"""H1 ablation: does absolute line numbering actually pin the frontier?

The main model never emits `N<i>` line indices. Lines are implicit and
references are relative (`B<k>` = k lines back, `P<i>` = i-th premise). I chose
that at the start on the theory that absolute indices above `N6` are untrained
under a 6-line cap -- but I never tested it, so it stayed an assumption.

This trains an otherwise identical Stage-1 model on the SPEC format instead:
every line begins with its absolute index and every reference is `N<j>`. Same
architecture, same data, same steps, same seed. Then it runs the same
length probe. If the relative scheme matters, the frontiers differ.

Self-contained on purpose: it defines its own vocabulary and codec so nothing
in nd/ has to change and the main results cannot be perturbed.
"""
import argparse, collections, json, math, os, random, time

import torch

from nd.model import GPT, Config
from nd.tokenizer import RULES, SYMS, SPECIAL, split_lines
from nd_verify import verify_text

MAX_LINE = 32
VOCAB = SPECIAL + SYMS + RULES + ['N%d' % i for i in range(1, MAX_LINE + 1)]
STOI = {t: i for i, t in enumerate(VOCAB)}
# Build the reverse map by index, not by inverting STOI: 'R' is both the atom
# and the reiteration rule, so inverting STOI drops the atom's index and
# sampling it raises KeyError. (Same bug as in nd/tokenizer.py, reintroduced
# here because this script defines its own vocabulary.)
ITOS = {i: t for i, t in enumerate(VOCAB)}
PAD, BOS, EOS = STOI['<pad>'], STOI['<bos>'], STOI['QED']
V = len(VOCAB)


def encode_example(prompt, body):
    """Spec format verbatim, renumbered from N1. None if it will not fit."""
    lines = split_lines(body.split())
    if not lines:
        return None
    base = int(lines[0][0][1:])
    out = []
    for i, ln in enumerate(lines, start=1):
        if i > MAX_LINE:
            return None
        out.append('N%d' % i)
        for t in ln[1:]:
            if t.startswith('N') and t[1:].isdigit():
                r = int(t[1:]) - base + 1
                if not (1 <= r <= MAX_LINE):
                    return None
                out.append('N%d' % r)
            else:
                out.append(t)
    out.append('QED')
    return ['<bos>'] + prompt.split() + out


def decode_body(toks):
    """Model output is already spec format; just stop at QED."""
    out = []
    for t in toks:
        if t == 'QED':
            break
        if t in ('<pad>', '<bos>'):
            continue
        out.append(t)
    return ' '.join(out + ['QED'])


def encode_records(recs, block):
    seqs = []
    for r in recs:
        toks = encode_example(r['prompt'], r['proof'])
        if toks is None or len(toks) > block:
            continue
        seqs.append(([STOI[t] for t in toks], 1 + len(r['prompt'].split())))
    T = max(len(s) for s, _ in seqs)
    x = torch.full((len(seqs), T - 1), PAD, dtype=torch.long)
    y = torch.full((len(seqs), T - 1), -100, dtype=torch.long)
    for i, (ids, npr) in enumerate(seqs):
        n = len(ids)
        x[i, :n - 1] = torch.tensor(ids[:-1])
        y[i, :n - 1] = torch.tensor(ids[1:])
        y[i, :npr - 1] = -100
    return x, y


@torch.no_grad()
def sample(model, prompts, temperature, bs, max_new, device='cuda'):
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i].split()))
    out = [None] * len(prompts)
    for i in range(0, len(order), bs):
        grp = order[i:i + bs]
        seqs = [[BOS] + [STOI[t] for t in prompts[g].split()] for g in grp]
        T = max(len(s) for s in seqs)
        x = torch.full((len(seqs), T), PAD, dtype=torch.long)
        for r, s in enumerate(seqs):
            x[r, T - len(s):] = torch.tensor(s)
        x = x.to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            y = model.generate(x, max_new, EOS, temperature=temperature,
                               pad=PAD, left_padded=True)
        for r, g in enumerate(grp):
            out[g] = decode_body([ITOS[t] for t in y[r, T:].tolist()])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--block', type=int, default=512)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--k', type=int, default=32)
    ap.add_argument('--out', default='ckpt/sft_absrefs.pt')
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = 'cuda'
    tr = [json.loads(l) for l in open('data/train.jsonl')]
    va = [json.loads(l) for l in open('data/heldout.jsonl')]
    xt, yt = encode_records(tr, a.block)
    xv, yv = encode_records(va, a.block)
    print(f'vocab {V}  train {tuple(xt.shape)}  val {tuple(xv.shape)}', flush=True)

    cfg = Config(V, 4, 4, 256, a.block, pos_mode='rope')
    m = GPT(cfg).to(dev)
    print(f'params {m.n_params()/1e6:.2f}M  pos_mode=rope  refs=ABSOLUTE', flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.1,
                            betas=(0.9, 0.95))

    def lr_at(s):
        if s < 200:
            return a.lr * s / 200
        p = (s - 200) / max(1, a.steps - 200)
        return a.lr * 0.5 * (1 + math.cos(math.pi * p))

    step, t0 = 0, time.time()
    n = xt.size(0)
    while step < a.steps:
        perm = torch.randperm(n)
        for i in range(0, n, a.bs):
            if step >= a.steps:
                break
            j = perm[i:i + a.bs]
            xb, yb = xt[j].to(dev), yt[j].to(dev)
            for g in opt.param_groups:
                g['lr'] = lr_at(step)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, loss = m(xb, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 1000 == 0:
                m.eval()
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    vl = sum(m(xv[i:i+512].to(dev), yv[i:i+512].to(dev))[1].item()
                             for i in range(0, xv.size(0), 512))
                    vl /= math.ceil(xv.size(0) / 512)
                m.train()
                print(f'step {step:5d} train {loss.item():.4f} val {vl:.4f} '
                      f'{time.time()-t0:.0f}s', flush=True)
    m.eval()
    os.makedirs('ckpt', exist_ok=True)
    torch.save({'cfg': vars(cfg), 'model': m.state_dict()}, a.out)

    # in-distribution check
    ho = va[:1500]
    ps = [r['prompt'] for r in ho]
    b = sample(m, ps, 0.0, 1500, 220)
    ok = sum(verify_text(p + ' ' + x)[0] for p, x in zip(ps, b))
    print(f'\nheld-out greedy: {ok}/{len(ps)} = {100*ok/len(ps):.1f}%', flush=True)

    # the frontier probe, identical protocol to scripts_probe_length.py
    tg = [json.loads(l) for l in open('data/rl_targets_hard.jsonl')][:2000]
    ps = [r['prompt'] for r in tg]
    by_len = collections.defaultdict(set)
    solved = set()
    for _ in range(a.k):
        bodies = sample(m, ps, 1.0, 2000, 380)
        for p, x in zip(ps, bodies):
            okk, _, nl = verify_text(p + ' ' + x)
            if okk:
                solved.add(p)
                by_len[nl].add(p + '|||' + ' '.join(x.split()))
    counts = {int(L): len(s) for L, s in sorted(by_len.items())}
    robust = max([L for L, s in by_len.items() if len(s) >= 5], default=0)
    print(f'\nABSOLUTE-REF MODEL on 2000 hard targets, k={a.k}, T=1.0')
    print(f'  solved          {len(solved)}/2000')
    print(f'  written lengths {counts}')
    print(f'  robust frontier {robust}')
    print(f'  longest single  {max(by_len) if by_len else 0}')
    json.dump({'solved': len(solved), 'written_lengths': counts,
               'robust_frontier': robust, 'heldout_greedy': ok / 1500,
               'vocab': V}, open('numbers_ablation_absrefs.json', 'w'), indent=1)
    print('wrote numbers_ablation_absrefs.json')


if __name__ == '__main__':
    main()
