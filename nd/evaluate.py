"""Sampling from a checkpoint and scoring against the verifier."""
import math

import torch

from nd_verify import verify_text
from .model import GPT, Config
from .tokenizer import EOS, ITOS, PAD, STOI, decode_body, n_premises


def load_ckpt(path, device='cuda'):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = Config(**ck['cfg'])
    m = GPT(cfg).to(device)
    m.load_state_dict(ck['model'])
    m.eval()
    return m


def wilson(k, n, z=1.96):
    """Wilson score interval -- honest at the small n and extreme rates we hit."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


@torch.no_grad()
def sample_proofs(model, prompts, device='cuda', temperature=0.0, bs=512,
                  max_new=260):
    """One decoded spec-format proof body per prompt, in order.

    Prompts are left-padded to a common width; the model masks the padding out
    of both attention and its position count, so a short prompt decodes exactly
    as it would alone. Sorting by length first keeps the padding cheap.
    """
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i].split()))
    out = [None] * len(prompts)
    for i in range(0, len(order), bs):
        grp = order[i:i + bs]
        seqs = [[STOI['<bos>']] + [STOI[t] for t in prompts[g].split()] for g in grp]
        T = max(len(s) for s in seqs)
        x = torch.full((len(seqs), T), PAD, dtype=torch.long)
        for r, s in enumerate(seqs):
            x[r, T - len(s):] = torch.tensor(s)
        x = x.to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
            y = model.generate(x, max_new, EOS, temperature=temperature,
                               pad=PAD, left_padded=True)
        for r, g in enumerate(grp):
            toks = []
            for t in y[r, T:].tolist():
                tok = ITOS[t]
                if tok == 'QED':
                    break
                if tok in ('<pad>', '<bos>'):
                    continue
                toks.append(tok)
            out[g] = decode_body(toks + ['QED'], n_premises(prompts[g]))
    return out


def judge(prompts, bodies):
    """-> list of (ok, reason, n_lines)."""
    return [verify_text(p + ' ' + b) for p, b in zip(prompts, bodies)]
