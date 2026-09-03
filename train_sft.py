#!/usr/bin/env python3
"""Stage 1: supervised training on generated cap-6 proofs."""
import argparse, math, time

import torch

from nd.data_torch import batches, encode_records, load
from nd.model import GPT, Config
from nd.tokenizer import V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='data/train.jsonl')
    ap.add_argument('--val', default='data/heldout.jsonl')
    ap.add_argument('--out', default='ckpt/sft.pt')
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--n_layer', type=int, default=4)
    ap.add_argument('--d_model', type=int, default=256)
    ap.add_argument('--n_head', type=int, default=4)
    ap.add_argument('--block', type=int, default=320)
    ap.add_argument('--pos_mode', default='learned',
                    choices=['learned', 'rope', 'nope'])
    ap.add_argument('--warmup', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    tr, va = load(a.train), load(a.val)
    xt, yt = encode_records(tr, a.block)
    xv, yv = encode_records(va, a.block)
    print(f'train {xt.shape} val {xv.shape}')

    cfg = Config(V, a.n_layer, a.n_head, a.d_model, a.block,
                 pos_mode=a.pos_mode)
    m = GPT(cfg).to(dev)
    print(f'params {m.n_params()/1e6:.2f}M  pos_mode={a.pos_mode}')
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.1,
                            betas=(0.9, 0.95))

    def lr_at(s):
        if s < a.warmup:
            return a.lr * s / max(1, a.warmup)
        p = (s - a.warmup) / max(1, a.steps - a.warmup)
        return a.lr * 0.5 * (1 + math.cos(math.pi * p))

    step, t0 = 0, time.time()
    while step < a.steps:
        for x, y in batches(xt, yt, a.bs, device=dev):
            if step >= a.steps:
                break
            for g in opt.param_groups:
                g['lr'] = lr_at(step)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')):
                _, loss = m(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 500 == 0 or step == a.steps:
                m.eval()
                with torch.no_grad():
                    vl, nb = 0.0, 0
                    for xb, yb in batches(xv, yv, a.bs, shuffle=False, device=dev):
                        with torch.autocast('cuda', dtype=torch.bfloat16,
                                            enabled=(dev == 'cuda')):
                            _, l = m(xb, yb)
                        vl += l.item(); nb += 1
                m.train()
                print(f'step {step:5d}  train {loss.item():.4f}  val {vl/nb:.4f}'
                      f'  {time.time()-t0:.0f}s', flush=True)
    import os
    os.makedirs('ckpt', exist_ok=True)
    torch.save({'cfg': vars(cfg), 'model': m.state_dict()}, a.out)
    print('saved', a.out)


if __name__ == '__main__':
    main()
