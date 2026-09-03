#!/usr/bin/env python3
"""Deliverables the original brief asks for that were not yet produced.

1. Stage 1: "The generator ... what its length / rule / premise distributions
   look like (a histogram or two)."
2. Stage 3 item 2: "Your transfer set (7-16 lines), BY LENGTH."
3. Required figure: "the per-length curve for the transfer set with the
   frozen-model control".
4. Required figure: "the found-length histogram across RL rounds".
"""
import collections, json, random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nd.data_torch import load
from nd.evaluate import judge, load_ckpt, sample_proofs, wilson
from nd.tokenizer import split_lines

# ---------- 1. generator distributions -------------------------------------
tr = load('data/train.jsonl')
lens = collections.Counter(r['n_lines'] for r in tr)
rules = collections.Counter()
prem = collections.Counter()
for r in tr:
    for ln in split_lines(r['proof'].split()):
        # the rule is the token immediately after ':' -- searching the whole
        # line would count the ATOM 'R' as the reiteration rule 'R'
        if ':' in ln:
            rules[ln[ln.index(':') + 1]] += 1
    prem[sum(1 for ln in split_lines(r['proof'].split()) if 'PR' in ln)] += 1

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
ks = sorted(lens)
axes[0].bar(ks, [lens[k] for k in ks], color='#4477aa')
axes[0].set_xlabel('proof length (lines)'); axes[0].set_ylabel('theorems')
axes[0].set_title('Training set: proof length\n(cap = 6, enforced)')
rk = [k for k, _ in rules.most_common()]
axes[1].bar(range(len(rk)), [rules[k] for k in rk], color='#ee6677')
axes[1].set_xticks(range(len(rk))); axes[1].set_xticklabels(rk, rotation=60, fontsize=8)
axes[1].set_ylabel('rule applications'); axes[1].set_yscale('log')
axes[1].set_title('Training set: rule usage\n(log scale; ORE is rare by nature)')
pk = sorted(prem)
axes[2].bar(pk, [prem[k] for k in pk], color='#228833')
axes[2].set_xlabel('number of premises'); axes[2].set_ylabel('theorems')
axes[2].set_title('Training set: premise count')
for a in axes:
    a.grid(alpha=0.25, axis='y')
fig.tight_layout(); fig.savefig('figures/fig10_generator_dists.png', dpi=160)
print('figures/fig10_generator_dists.png')
print('  lengths :', dict(sorted(lens.items())))
print('  rules   :', dict(rules.most_common()))
print('  premises:', dict(sorted(prem.items())))

# ---------- 2 & 3. transfer set BY LENGTH, with frozen control -------------
tf = load('data/transfer_hard.jsonl')
by_gen = collections.defaultdict(list)
for r in tf:
    by_gen[r['gen_lines']].append(r)
models = [('Stage 1 (frozen control)', 'ckpt/sft_rope_clean.pt'),
          ('after RL', 'runs/clean_seed0/final.pt')]
res = {}
for name, path in models:
    m = load_ckpt(path)
    ps = [r['prompt'] for r in tf]
    bodies = sample_proofs(m, ps, temperature=0.0, bs=800, max_new=380)
    verdict = dict(zip(ps, judge(ps, bodies)))
    row = {}
    for L in sorted(by_gen):
        sub = by_gen[L]
        k = sum(verdict[r['prompt']][0] for r in sub)
        row[L] = (k, len(sub))
    res[name] = row
    print(f'\n{name} -- transfer set by GENERATING length (greedy):')
    for L in sorted(row):
        k, n = row[L]
        p, lo, hi = wilson(k, n)
        print(f'   gen_lines {L:2d}:  {k:3d}/{n:3d} = {100*p:5.1f}%  [{100*lo:4.1f}-{100*hi:4.1f}]')

fig, ax = plt.subplots(figsize=(7.5, 4.4))
for (name, _), col, mk in zip(models, ['#888888', '#ee6677'], ['s--', 'o-']):
    row = res[name]
    ks = sorted(row)
    ps_ = [100 * row[L][0] / row[L][1] for L in ks]
    los = [100 * wilson(*row[L])[1] for L in ks]
    his = [100 * wilson(*row[L])[2] for L in ks]
    ax.plot(ks, ps_, mk, color=col, lw=2, ms=5, label=name)
    ax.fill_between(ks, los, his, color=col, alpha=0.15)
ax.set_xlabel('generating length of the transfer theorem (an UPPER BOUND on its shortest proof)')
ax.set_ylabel('greedy solve rate (%)')
ax.set_title('Transfer set by length, with the frozen-model control\n'
             '(transfer theorems are never sampled during RL)')
ax.legend(frameon=False); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig('figures/fig11_transfer_by_length.png', dpi=160)
print('\nfigures/fig11_transfer_by_length.png')
json.dump({k: {str(a): list(b) for a, b in v.items()} for k, v in res.items()},
          open('numbers_transfer_by_length.json', 'w'), indent=1)

# ---------- 4. found-length histogram across rounds ------------------------
d = json.load(open('runs/clean_seed0/log.json'))
fig, ax = plt.subplots(figsize=(8, 4.4))
w = 0.15
cols = plt.cm.viridis([0.1, 0.3, 0.5, 0.7, 0.9])
for i, r in enumerate(d):
    fl = {int(k): v for k, v in r['found_lengths'].items()}
    ks = sorted(fl)
    ax.bar([k + (i - 2) * w for k in ks], [fl[k] for k in ks], width=w,
           color=cols[i], label=f"round {r['round']}")
ax.axvline(6.5, color='k', ls='--', lw=1)
ax.text(6.6, ax.get_ylim()[1] * 0.85, 'training cap', fontsize=9)
ax.set_xlabel('length of the proof actually WRITTEN')
ax.set_ylabel('distinct verified proofs (cumulative)')
ax.set_title('Found-proof-length histogram across RL rounds')
ax.set_yscale('log'); ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.25, axis='y')
fig.tight_layout(); fig.savefig('figures/fig12_found_lengths_rounds.png', dpi=160)
print('figures/fig12_found_lengths_rounds.png')
