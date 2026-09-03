"""A small from-scratch decoder-only transformer (GPT-2 style).

Attention is hand-rolled rather than nn.MultiheadAttention so that generation
can carry a KV cache: without one, every sampled token re-runs the whole
prefix, which is the difference between minutes and hours for the RL sampling
budget. Parameter names deliberately mirror nn.MultiheadAttention
(`in_proj_weight`, `in_proj_bias`, `out_proj`) so checkpoints trained with the
stock module load unchanged.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn

# The cuDNN fused-attention backend fails on this shape family at large batch
# ("mha_graph.execute ... got false"), which showed up only as a crash deep in
# a long sampling run. The flash / mem-efficient / math backends are correct
# here, so take cuDNN out of the running.
try:
    torch.backends.cuda.enable_cudnn_sdp(False)
except Exception:
    pass


class Config:
    def __init__(self, vocab, n_layer=4, n_head=4, d_model=256, block=320,
                 dropout=0.0, pos_mode='learned', rope_base=10000.0):
        self.vocab, self.n_layer, self.n_head = vocab, n_layer, n_head
        self.d_model, self.block, self.dropout = d_model, block, dropout
        # 'learned' -- GPT-2 style absolute position embeddings. Every position
        #   past the longest training sequence (194 tokens) is untrained, so a
        #   14-line proof is decoded from embeddings that never saw a gradient.
        #   That is the same length barrier the reference tokenisation removes,
        #   reappearing in the positional scheme.
        # 'rope'    -- rotary embeddings: attention sees relative offsets, which
        #   are in-distribution at any absolute length.
        # 'nope'    -- no positional signal at all; a causal decoder can infer
        #   order from the mask alone, and it cannot go out of range.
        self.pos_mode, self.rope_base = pos_mode, rope_base


def rope_tables(pos, hd, base, dtype):
    """cos/sin lookup for rotary embeddings at the given absolute positions.

    `pos` is [B, T]; returns [B, 1, T, hd] so it broadcasts over heads. Using
    the position tensor (rather than arange) keeps left-padded rows correct.
    """
    half = hd // 2
    inv = 1.0 / (base ** (torch.arange(0, half, device=pos.device,
                                       dtype=torch.float32) / half))
    ang = pos.float()[..., None] * inv                      # [B, T, half]
    ang = torch.cat([ang, ang], dim=-1)                     # [B, T, hd]
    return ang.cos().to(dtype)[:, None], ang.sin().to(dtype)[:, None]


def apply_rope(t, cos, sin):
    half = t.size(-1) // 2
    rot = torch.cat([-t[..., half:], t[..., :half]], dim=-1)
    return t * cos + rot * sin


class SelfAttn(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.h = c.n_head
        self.d = c.d_model
        self.hd = c.d_model // c.n_head
        self.in_proj_weight = nn.Parameter(torch.empty(3 * c.d_model, c.d_model))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * c.d_model))
        self.out_proj = nn.Linear(c.d_model, c.d_model)
        nn.init.normal_(self.in_proj_weight, std=0.02)

    def _split(self, t, B, T):
        return t.view(B, T, self.h, self.hd).transpose(1, 2)

    def forward(self, x, causal=True, kpm=None, cache=None, rope=None):
        """kpm: [B, T_total] True where the key is padding.
        cache: dict with 'k','v' -- appended to in place when decoding."""
        B, T, _ = x.shape
        q, k, v = Fn.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, -1)
        q, k, v = (self._split(t, B, T) for t in (q, k, v))
        if rope is not None:
            cos, sin = rope
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            # The cache buffer is preallocated by generate() and written in
            # place. Growing it with torch.cat instead would recopy the whole
            # cache on every token -- quadratic traffic, and tens of GB of
            # churn at the batch sizes the RL sampling budget needs.
            n = cache['n']
            cache['k'][:, :, n:n + T] = k
            cache['v'][:, :, n:n + T] = v
            cache['n'] = n + T
            k = cache['k'][:, :, :n + T]
            v = cache['v'][:, :, :n + T]
        S = k.size(2)
        mask = None
        if kpm is not None:
            # A large finite penalty, not -inf. A left-padded query row is
            # itself padding and would otherwise be masked at every key, so
            # softmax would see all -inf and emit NaN -- which then leaks into
            # real positions through the next layer's values (masked weights
            # are exactly 0, but 0 * NaN is NaN). exp(-1e4) underflows to zero
            # anyway, so real rows are unaffected.
            neg = torch.finfo(q.dtype).min / 4
            mask = torch.zeros(B, 1, 1, S, device=x.device, dtype=q.dtype)
            mask = mask.masked_fill(kpm[:, None, None, :], neg)
            mask = mask.expand(B, 1, T, S)
        if causal and T > 1:
            # sdpa will not combine is_causal with an explicit mask, so fold the
            # causal triangle into the additive mask instead.
            cm = torch.triu(torch.full((T, S), float('-inf'), device=x.device,
                                       dtype=q.dtype), 1 + S - T)
            mask = cm if mask is None else mask + cm
        y = Fn.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).reshape(B, T, self.d)
        return self.out_proj(y)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.d_model)
        self.attn = SelfAttn(c)
        self.ln2 = nn.LayerNorm(c.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(c.d_model, 4 * c.d_model), nn.GELU(),
            nn.Linear(4 * c.d_model, c.d_model), nn.Dropout(c.dropout))

    def forward(self, x, causal=True, kpm=None, cache=None, rope=None):
        x = x + self.attn(self.ln1(x), causal=causal, kpm=kpm, cache=cache,
                          rope=rope)
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab, c.d_model)
        self.pos = nn.Embedding(c.block, c.d_model) if c.pos_mode == 'learned' else None
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        self.lnf = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab, bias=False)
        self.head.weight = self.tok.weight          # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, pad=None, pos=None, kpm=None,
                caches=None):
        """`pad`: positions equal to it are treated as left padding -- masked
        out of attention and excluded from the position count, so a padded
        prompt occupies the same positions as an unpadded one."""
        B, T = idx.shape
        if pos is None:
            if pad is None:
                pos = torch.arange(T, device=idx.device).expand(B, T)
            else:
                real = (idx != pad)
                pos = (real.cumsum(1) - 1).clamp(min=0)
                kpm = ~real
        h = self.tok(idx)
        rope = None
        if self.c.pos_mode == 'learned':
            h = h + self.pos(pos.clamp(max=self.c.block - 1))
        elif self.c.pos_mode == 'rope':
            rope = rope_tables(pos, self.c.d_model // self.c.n_head,
                               self.c.rope_base, h.dtype)
        x = self.drop(h)
        for i, b in enumerate(self.blocks):
            x = b(x, causal=True, kpm=kpm,
                  cache=(caches[i] if caches is not None else None), rope=rope)
        logits = self.head(self.lnf(x))
        loss = None
        if targets is not None:
            loss = Fn.cross_entropy(logits.view(-1, logits.size(-1)),
                                    targets.reshape(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new, eos, temperature=0.0, pad=0,
                 left_padded=False):
        """Batched generation with a KV cache. temperature=0 means greedy.

        With `left_padded`, prompts of different lengths may be right-aligned in
        one batch; the padding is masked out of attention and of the position
        count, so each row decodes exactly as it would on its own.
        """
        B, Tp = idx.shape
        dev = idx.device
        if left_padded:
            real = (idx != pad)
            pos = (real.cumsum(1) - 1).clamp(min=0)
            kpm = ~real
        else:
            pos = torch.arange(Tp, device=dev).expand(B, Tp)
            kpm = torch.zeros(B, Tp, dtype=torch.bool, device=dev)

        total = Tp + max_new
        hd = self.c.d_model // self.c.n_head
        caches = [{'k': torch.empty(B, self.c.n_head, total, hd, device=dev,
                                    dtype=self.tok.weight.dtype),
                   'v': torch.empty(B, self.c.n_head, total, hd, device=dev,
                                    dtype=self.tok.weight.dtype),
                   'n': 0} for _ in self.blocks]
        has_pad = bool(kpm.any())
        logits, _ = self(idx, pos=pos, kpm=(kpm if has_pad else None),
                         caches=caches)
        nxt_pos = pos[:, -1] + 1
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        pad_col = torch.zeros(B, 1, dtype=torch.bool, device=dev)
        toks = []
        # Testing `done.all()` in Python forces a GPU sync on every token, which
        # dominates the runtime for a model this small. Check it periodically
        # instead: a few extra tokens are far cheaper than 300 syncs.
        CHECK_EVERY = 16
        for i in range(max_new):
            lg = logits[:, -1, :]
            if temperature and temperature > 0:
                nxt = torch.multinomial(Fn.softmax(lg / temperature, -1), 1)
            else:
                nxt = lg.argmax(-1, keepdim=True)
            nxt = torch.where(done[:, None], torch.full_like(nxt, pad), nxt)
            toks.append(nxt)
            done |= (nxt[:, 0] == eos)
            if i % CHECK_EVERY == CHECK_EVERY - 1 and bool(done.all()):
                break
            if has_pad:
                kpm = torch.cat([kpm, pad_col], 1)
            logits, _ = self(nxt, pos=nxt_pos[:, None],
                             kpm=(kpm if has_pad else None), caches=caches)
            nxt_pos = nxt_pos + 1
        return torch.cat([idx] + toks, 1)
