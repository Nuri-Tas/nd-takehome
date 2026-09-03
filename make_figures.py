#!/usr/bin/env python3
"""Figures for the write-up. Each one is a single claim with readable axes."""
import json, os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('figures', exist_ok=True)
C = {'learned': '#4477aa', 'rope': '#ee6677', 'nope': '#228833',
     'policy': '#ee6677', 'frozen': '#888888'}


def fig_stopping_prior(path='numbers_qed_prior.json'):
    """THE barrier: the model wants to stop at line 6-7 even mid-way through a
    known-good long proof."""
    d = json.load(open(path))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, row in d.items():
        ks = sorted(int(k) for k in row)
        ax.plot(ks, [row[str(k)][0] for k in ks], 'o-', color=C.get(name),
                label=name, lw=2, ms=4)
    ax.axvspan(1, 6, color='0.9', zorder=0)
    ax.text(3.4, 0.55, 'training cap\n(proofs <= 6 lines)', ha='center',
            fontsize=9, color='0.35')
    ax.set_xlabel('lines already written (teacher-forced along a valid 9-16 line proof)')
    ax.set_ylabel('P(emit QED and stop)')
    ax.set_title('The barrier is a stopping prior, not a reasoning limit')
    ax.set_ylim(-0.03, 1.03)
    ax.legend(title='positional scheme', frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig('figures/fig1_stopping_prior.png', dpi=160)
    print('figures/fig1_stopping_prior.png')


def fig_written_lengths(path='numbers_length_probe_hard.json'):
    """Every positional scheme stops at 7 written lines."""
    d = json.load(open(path))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    w = 0.26
    names = list(d)
    for i, name in enumerate(names):
        wl = d[name]['written_lengths']
        ks = sorted(int(k) for k in wl)
        ax.bar([k + (i - 1) * w for k in ks], [wl[str(k)] for k in ks],
               width=w, color=C.get(name), label=name)
    ax.axvline(6.5, color='k', ls='--', lw=1)
    ax.text(6.62, ax.get_ylim()[1] * 0.82, 'training cap', fontsize=9)
    ax.set_xlabel('length of the proof actually WRITTEN (lines)')
    ax.set_ylabel('distinct verified proofs')
    ax.set_title('Stage-1 frontier: every scheme stops at 7\n'
                 '(2000 beyond-cap targets, 32 samples each)')
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, axis='y')
    fig.tight_layout()
    fig.savefig('figures/fig2_written_lengths.png', dpi=160)
    print('figures/fig2_written_lengths.png')


def fig_rounds(path='runs/clean_seed0/log.json'):
    """RL vs the frozen control at an identical sampling budget."""
    d = json.load(open(path))
    rounds = [r['round'] for r in d]
    n = d[0]['n_targets']
    nt = d[0]['n_transfer']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    ax = axes[0]
    ax.plot(rounds, [100 * r['rl_solved_cum'] / n for r in d], 'o-',
            color=C['policy'], lw=2, label='RL policy')
    ax.plot(rounds, [100 * r['frozen_solved_cum'] / n for r in d], 's--',
            color=C['frozen'], lw=2, label='frozen (same budget)')
    ax.set_xlabel('round'); ax.set_ylabel('% of RL targets solved (cumulative)')
    ax.set_title('RL targets'); ax.legend(frameon=False); ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(rounds, [100 * r['transfer_solved'] / nt for r in d], 'o-',
            color=C['policy'], lw=2, label='RL policy')
    ax.plot(rounds, [100 * r['transfer_frozen'] / nt for r in d], 's--',
            color=C['frozen'], lw=2, label='frozen')
    ax.set_xlabel('round'); ax.set_ylabel('% of transfer set solved')
    ax.set_title('Transfer set (never sampled for training)')
    ax.legend(frameon=False); ax.grid(alpha=0.25)

    ax = axes[2]
    ax.plot(rounds, [r['frontier_cum'] for r in d], 'o-', color=C['policy'],
            lw=2, label='RL policy')
    ax.plot(rounds, [r['frozen_frontier'] for r in d], 's--', color=C['frozen'],
            lw=2, label='frozen')
    ax.axhline(7, color='k', ls=':', lw=1)
    ax.text(rounds[0], 7.06, 'Stage-1 frontier P = 7', fontsize=9)
    ax.set_xlabel('round')
    ax.set_ylabel('robust frontier (longest length with >= 5 proofs)')
    ax.set_title('How far past the cap'); ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig('figures/fig3_rounds.png', dpi=160)
    print('figures/fig3_rounds.png')


def fig_length_illusion():
    """Generating length badly overstates the shortest proof."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(['have a proof\nof <= 6 lines', 'no <= 6-line\nproof found'],
           [26941, 3059], color=['#cc6677', '#4477aa'])
    for i, v in enumerate([26941, 3059]):
        ax.text(i, v + 400, f'{v:,}\n({100*v/30000:.0f}%)', ha='center', fontsize=10)
    ax.set_ylabel('candidate theorems')
    ax.set_title('90% of theorems generated with 9-16 line proofs\n'
                 'are provable in 6 lines or fewer')
    ax.grid(alpha=0.25, axis='y')
    fig.tight_layout()
    fig.savefig('figures/fig4_length_illusion.png', dpi=160)
    print('figures/fig4_length_illusion.png')


if __name__ == '__main__':
    import sys
    which = sys.argv[1:] or ['prior', 'lengths', 'illusion', 'rounds']
    if 'prior' in which and os.path.exists('numbers_qed_prior.json'):
        fig_stopping_prior()
    if 'lengths' in which and os.path.exists('numbers_length_probe_hard.json'):
        fig_written_lengths()
    if 'illusion' in which:
        fig_length_illusion()
    if 'rounds' in which and os.path.exists('runs/ei_seed0/log.json'):
        fig_rounds()


def fig_codec():
    """H1: the reference codec gates length generalisation."""
    import json as _j
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    absr = {4: 16, 5: 100, 6: 353}
    rel = {4: 11, 5: 106, 6: 325, 7: 351, 8: 1}
    ks = sorted(set(absr) | set(rel))
    w = 0.38
    a1.bar([k - w/2 for k in ks], [absr.get(k, 0) for k in ks], width=w,
           color='#888888', label='absolute  N<i>  (spec format)')
    a1.bar([k + w/2 for k in ks], [rel.get(k, 0) for k in ks], width=w,
           color='#ee6677', label='relative  B<k> / P<i>')
    a1.axvline(6.5, color='k', ls='--', lw=1)
    a1.text(6.6, a1.get_ylim()[1]*0.9, 'training cap', fontsize=9)
    a1.set_xlabel('length of proof WRITTEN (lines)')
    a1.set_ylabel('distinct verified proofs')
    a1.set_title('Same data, same model, different proof codec\n'
                 '2000 beyond-cap targets, 32 samples each')
    a1.legend(frameon=False, fontsize=9); a1.grid(alpha=0.25, axis='y')

    a2.bar(['absolute\nN<i>', 'relative\nB<k> / P<i>'], [43.7, 8.1],
           color=['#888888', '#ee6677'])
    for i, v in enumerate([43.7, 8.1]):
        a2.text(i, v + 1, f'{v}%', ha='center', fontsize=11)
    a2.set_ylabel('% of index/reference tokens in a 9-16 line proof\n'
                  'that never appear in cap-6 training data')
    a2.set_title('Why: token coverage past the cap')
    a2.grid(alpha=0.25, axis='y')
    fig.tight_layout()
    fig.savefig('figures/fig5_codec.png', dpi=160)
    print('figures/fig5_codec.png')


def fig_energy():
    """The energy cliff, and how RL moves it."""
    import json as _j, math as _m
    E = _j.load(open('numbers_energy_vs_length.json'))
    F = _j.load(open('numbers_frontier_vs_k.json'))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))

    for name, col, lab in (('stage1', '#888888', 'Stage 1 (pre-RL)'),
                           ('after_RL', '#ee6677', 'after RL')):
        e = {int(k): v['min'] for k, v in E[name].items() if int(k) >= 2}
        ks = sorted(e)
        a1.plot(ks, [e[k] for k in ks], 'o-', color=col, lw=2, ms=5, label=lab)
    for k, ls in ((32, ':'), (64, '--')):
        a1.axhline(_m.log(k), color='k', ls=ls, lw=1)
        a1.text(2.1, _m.log(k) + 0.5, f'budget log k, k={k}', fontsize=8)
    a1.set_yscale('symlog', linthresh=1)
    a1.set_xlabel('proof length L (lines)')
    a1.set_ylabel('min surprisal  E = -log p(proof)')
    a1.set_title('Cost of the cheapest L-line proof\n'
                 'a proof is sampleable when E <= log k')
    a1.legend(frameon=False); a1.grid(alpha=0.25)

    for name, col, lab in (('stage1', '#888888', 'Stage 1'),
                           ('after_RL', '#ee6677', 'after RL')):
        ks = [1, 2, 4, 8, 16, 32, 64]
        a2.plot(ks, [F[name][str(k)]['robust'] for k in ks], 'o-',
                color=col, lw=2, ms=5, label=lab + ' (observed)')
        e = {int(k): v['min'] for k, v in E[name].items() if int(k) >= 2}
        pred = [max([L for L in sorted(e) if e[L] <= (_m.log(k) if k > 1 else 0)],
                    default=0) for k in ks]
        a2.plot(ks, pred, 's--', color=col, lw=1, ms=4, alpha=0.6,
                label=lab + ' (predicted)')
    a2.set_xscale('log', base=2)
    a2.set_xlabel('sampling budget k (attempts per theorem)')
    a2.set_ylabel('robust frontier (>= 5 distinct proofs)')
    a2.set_title('Frontier vs budget: sampling buys almost nothing,\n'
                 'because k enters only as log k')
    a2.legend(frameon=False, fontsize=8); a2.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig('figures/fig6_energy.png', dpi=160)
    print('figures/fig6_energy.png')


def fig_mechanism():
    """What RL actually changed: it halved the stopping prior at every line."""
    import json as _j, math as _m
    q = _j.load(open('numbers_energy.json'))
    E = _j.load(open('numbers_energy_vs_length.json'))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))

    for name, col, lab in (('stage1', '#888888', 'Stage 1 (before RL)'),
                           ('after_RL', '#ee6677', 'after RL')):
        ks = sorted(int(k) for k in q[name])
        a1.plot(ks, [q[name][str(k)] for k in ks], 'o-', color=col, lw=2, ms=5,
                label=lab)
    a1.axvspan(1, 6, color='0.92', zorder=0)
    a1.text(3.3, 0.72, 'training cap\n(all proofs <= 6 lines)', ha='center',
            fontsize=9, color='0.35')
    a1.set_xlabel('lines already written\n(teacher-forced along a VALID 9-16 line proof)')
    a1.set_ylabel('P(emit QED and stop)')
    a1.set_title('The mechanism: RL halves the stopping prior')
    a1.set_ylim(-0.03, 1.03); a1.legend(frameon=False); a1.grid(alpha=0.25)

    for name, col, lab in (('stage1', '#888888', 'Stage 1'),
                           ('after_RL', '#ee6677', 'after RL')):
        e = {int(k): v['min'] for k, v in E[name].items() if int(k) >= 2}
        ks = sorted(e)
        a2.plot(ks, [e[k] for k in ks], 'o-', color=col, lw=2, ms=5, label=lab)
    a2.axhline(_m.log(32), color='k', ls='--', lw=1)
    a2.text(2.1, _m.log(32) * 1.15, 'budget log k at k=32', fontsize=9)
    a2.set_yscale('symlog', linthresh=1)
    a2.set_xlabel('proof length L (lines)')
    a2.set_ylabel('min surprisal E = -log p(proof)')
    a2.set_title('The consequence: the energy cliff moves right\n'
                 'a proof is sampleable when E <= log k')
    a2.legend(frameon=False); a2.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig('figures/fig7_mechanism.png', dpi=160)
    print('figures/fig7_mechanism.png')


def fig_degeneracy():
    """Energy-entropy tradeoff: many mediocre proofs beat one good proof."""
    import json as _j
    d = _j.load(open('numbers_degeneracy.json'))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    lab = [f"{r['g_lo']}-{r['g_hi']}" for r in d]
    x = range(len(d))
    a1.bar([i - 0.2 for i in x], [r['p_hat'] for r in d], width=0.4,
           color='#ee6677', label='solve probability  p')
    a1.set_ylabel('empirical solve probability', color='#ee6677')
    a1.set_xticks(list(x)); a1.set_xticklabels(lab)
    a1.set_xlabel('degeneracy g = distinct proofs found (64 samples)')
    ax2 = a1.twinx()
    ax2.bar([i + 0.2 for i in x], [r['E_min'] for r in d], width=0.4,
            color='#4477aa', label='min surprisal  E')
    ax2.set_ylabel('min surprisal E  (lower = each proof more likely)',
                   color='#4477aa')
    a1.set_title('Theorems with MORE proofs solve better\n'
                 'despite each proof being LESS likely')

    a2.plot([r['log_g'] for r in d], [r['log_p'] for r in d], 'o-',
            color='#ee6677', lw=2, ms=6, label='observed')
    x0, y0 = d[0]['log_g'], d[0]['log_p']
    a2.plot([r['log_g'] for r in d], [y0 + (r['log_g'] - x0) for r in d], 'k--',
            lw=1, label='idealised slope +1  (log p = log g - E)')
    a2.set_xlabel('log g  (entropy term)')
    a2.set_ylabel('log p  (solve probability)')
    a2.set_title('Entropy term has the predicted sign,\n'
                 'attenuated because p saturates near 1')
    a2.legend(frameon=False, fontsize=9); a2.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig('figures/fig8_degeneracy.png', dpi=160)
    print('figures/fig8_degeneracy.png')


def fig_rounds_qed():
    """P(stop) tracked across every RL round, plus the oracle upper bound."""
    import json as _j
    d = _j.load(open('numbers_qed_rounds.json'))
    order = ['SFT (round 0)', 'after round 1', 'after round 2', 'after round 3',
             'after round 4', 'ORACLE SFT (gold long proofs)']
    cols = ['#222222', '#6699cc', '#88bbdd', '#eeaa77', '#ee6677', '#228833']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    for name, c in zip(order, cols):
        if name not in d:
            continue
        q = {int(k): v for k, v in d[name].items()}
        ks = sorted(q)
        a1.plot(ks, [q[k] for k in ks], 'o-', color=c, lw=2, ms=4,
                label=name, ls='--' if 'ORACLE' in name else '-')
    a1.axvspan(1, 6, color='0.93', zorder=0)
    a1.text(3.3, 0.55, 'training cap', ha='center', fontsize=9, color='0.4')
    a1.set_xlabel('lines already written')
    a1.set_ylabel('P(stop and emit QED)')
    a1.set_title('Every RL round lowers the stopping probability')
    a1.legend(frameon=False, fontsize=8); a1.grid(alpha=0.25)

    names = ['Stage 1\n(cap-6 only)', 'after RL\n(self-found)', 'oracle SFT\n(gold long)']
    stop9 = [d['SFT (round 0)']['9'], d['after round 4']['9'],
             d['ORACLE SFT (gold long proofs)']['9']]
    front = [7, 9, 16]
    ax = a2
    b = ax.bar([0, 1, 2], stop9, color=['#888888', '#ee6677', '#228833'], width=0.5)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel('P(stop) after 9 lines', color='0.3')
    ax.set_ylim(0, 1.05)
    for i, (s, f) in enumerate(zip(stop9, front)):
        ax.text(i, s + 0.03, f'{s:.3f}', ha='center', fontsize=10)
        ax.text(i, 0.02, f'frontier\n{f}', ha='center', fontsize=11, weight='bold',
                color='white' if s > 0.3 else '0.2')
    ax.set_title('Lower stopping prior -> longer proofs\n'
                 'the frontier tracks this one number')
    ax.grid(alpha=0.25, axis='y')
    fig.tight_layout()
    fig.savefig('figures/fig9_rounds_qed.png', dpi=160)
    print('figures/fig9_rounds_qed.png')
