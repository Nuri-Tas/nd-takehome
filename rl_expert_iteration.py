#!/usr/bin/env python3
"""Stage 2: expert iteration (rejection-sampling fine-tuning) past the cap.

Each round:
  1. sample K attempts per RL target from the current policy at temperature T;
  2. keep the attempts the verifier accepts *for the prompted sequent*;
  3. fine-tune from the Stage-1 checkpoint on all successes accumulated so far,
     mixed with Stage-1 data to hold the in-distribution skill in place;
  4. measure everything that the claim "RL got further" depends on.

The control that makes the headline meaningful: a FROZEN copy of the Stage-1
model gets exactly the same number of attempts per round and is never
retrained. Anything RL solves that frozen resampling also finds is not a result.

Reported per round:
  * RL-target and transfer solve rates, separately (transfer is never sampled
    for training);
  * the found-proof-length histogram -- the length of the proof actually
    WRITTEN, never the length the theorem was generated with;
  * the robust frontier: the longest written length with >= 5 distinct
    verified proofs;
  * Stage-1 held-out greedy solve rate, to catch in-distribution regression.
"""
import argparse, collections, json, math, os, random, time

import torch

from nd.data_torch import batches, encode_records, load
from nd.evaluate import judge, load_ckpt, sample_proofs, wilson
from nd.model import GPT, Config
from nd.relabel import RelabelFilter, relabel
from nd.tokenizer import V

ROBUST_K = 5          # distinct verified proofs needed to own a length


def dedup_key(prompt, body):
    return prompt + ' ||| ' + ' '.join(body.split())


def attempt_round(model, prompts, k, temperature, device, bs, max_new):
    """k sampled attempts per prompt. -> list of (prompt, body, ok, n_lines)."""
    out = []
    for _ in range(k):
        bodies = sample_proofs(model, prompts, device=device,
                               temperature=temperature, bs=bs, max_new=max_new)
        for p, b, (ok, _, n) in zip(prompts, bodies, judge(prompts, bodies)):
            out.append((p, b, ok, n))
    return out


def solved_stats(attempts):
    """-> (set of solved prompts, {written_length: set of distinct proofs})."""
    solved = set()
    by_len = collections.defaultdict(set)
    for p, b, ok, n in attempts:
        if ok:
            solved.add(p)
            by_len[n].add(dedup_key(p, b))
    return solved, by_len


def robust_frontier(by_len, k=ROBUST_K):
    """Longest written length with >= k distinct verified proofs."""
    good = [L for L, s in by_len.items() if len(s) >= k]
    return max(good) if good else 0


def finetune(base_ckpt, recs, steps, bs, lr, device, block, seed=0):
    """Fresh fine-tune from the Stage-1 weights on the given records."""
    torch.manual_seed(seed)
    ck = torch.load(base_ckpt, map_location=device, weights_only=False)
    m = GPT(Config(**ck['cfg'])).to(device)
    m.load_state_dict(ck['model'])
    m.train()
    x, y = encode_records(recs, block)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.1,
                            betas=(0.9, 0.95))
    step = 0
    while step < steps:
        for xb, yb in batches(x, y, bs, device=device):
            if step >= steps:
                break
            for g in opt.param_groups:
                g['lr'] = lr * min(1.0, (step + 1) / 100) * \
                    0.5 * (1 + math.cos(math.pi * step / max(1, steps)))
            with torch.autocast('cuda', dtype=torch.bfloat16,
                                enabled=(device == 'cuda')):
                _, loss = m(xb, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            step += 1
    m.eval()
    return m, ck['cfg']


def greedy_rate(model, recs, device, bs):
    prompts = [r['prompt'] for r in recs]
    bodies = sample_proofs(model, prompts, device=device, temperature=0.0, bs=bs)
    res = judge(prompts, bodies)
    return sum(ok for ok, _, _ in res), len(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='ckpt/sft.pt')
    ap.add_argument('--rounds', type=int, default=4)
    ap.add_argument('--k', type=int, default=32, help='attempts per target')
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--rl_data', default='data/rl_targets_final.jsonl')
    ap.add_argument('--transfer_data', default='data/transfer_final.jsonl')
    ap.add_argument('--n_targets', type=int, default=1500)
    ap.add_argument('--n_transfer', type=int, default=500)
    ap.add_argument('--n_heldout', type=int, default=500)
    ap.add_argument('--ft_steps', type=int, default=1200)
    ap.add_argument('--ft_bs', type=int, default=256)
    ap.add_argument('--ft_lr', type=float, default=2e-4)
    ap.add_argument('--sft_mix', type=int, default=20000,
                    help='Stage-1 records mixed into each fine-tune')
    ap.add_argument('--bs', type=int, default=2048)
    ap.add_argument('--max_new', type=int, default=300)
    ap.add_argument('--block', type=int, default=320)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='runs/ei')
    ap.add_argument('--relabel', action='store_true',
                    help='reuse off-target but valid proofs as training data')
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    rl = load(a.rl_data)[:a.n_targets]
    tf = load(a.transfer_data)[:a.n_transfer]
    ho = load('data/heldout.jsonl')[:a.n_heldout]
    sft = load('data/train.jsonl')
    rl_prompts = [r['prompt'] for r in rl]
    tf_prompts = [r['prompt'] for r in tf]
    print(f'RL targets {len(rl)}  transfer {len(tf)}  heldout {len(ho)}')

    policy = load_ckpt(a.ckpt, device=dev)
    frozen = load_ckpt(a.ckpt, device=dev)
    filt = RelabelFilter() if a.relabel else None
    if filt is not None:
        print(f'relabelling on; contamination filter holds {filt.n_sources} theorems')

    train_pool = {}                       # dedup_key -> record
    cum_policy, cum_frozen = set(), set()
    cum_len_policy = collections.defaultdict(set)
    cum_len_frozen = collections.defaultdict(set)
    log = []

    for rd in range(1, a.rounds + 1):
        t0 = time.time()

        # --- policy attempts on RL targets -----------------------------
        att = attempt_round(policy, rl_prompts, a.k, a.temperature, dev,
                            a.bs, a.max_new)
        solved, by_len = solved_stats(att)
        cum_policy |= solved
        for L, s in by_len.items():
            cum_len_policy[L] |= s
        n_relab = 0
        relab_len = collections.Counter()
        for p, b, ok, n in att:
            if ok:
                train_pool.setdefault(dedup_key(p, b),
                                      {'prompt': p, 'proof': b, 'n_lines': n,
                                       'source': 'on_target'})
            elif filt is not None:
                # Hindsight: the sample failed this sequent but may be a valid
                # proof of the theorem it actually derived.
                r = relabel(b, filt)
                if r is not None:
                    rp, rn = r
                    key = dedup_key(rp, b)
                    if key not in train_pool:
                        train_pool[key] = {'prompt': rp, 'proof': b,
                                         'n_lines': rn, 'source': 'relabel'}
                        n_relab += 1
                        relab_len[rn] += 1

        # --- frozen control: identical budget, no retraining ------------
        fatt = attempt_round(frozen, rl_prompts, a.k, a.temperature, dev,
                             a.bs, a.max_new)
        fsolved, fby_len = solved_stats(fatt)
        cum_frozen |= fsolved
        for L, sset in fby_len.items():
            cum_len_frozen[L] |= sset

        # --- transfer, never trained on --------------------------------
        tatt = attempt_round(policy, tf_prompts, a.k, a.temperature, dev,
                             a.bs, a.max_new)
        tsolved, tby_len = solved_stats(tatt)
        ftatt = attempt_round(frozen, tf_prompts, a.k, a.temperature, dev,
                              a.bs, a.max_new)
        ftsolved, _ = solved_stats(ftatt)

        # --- in-distribution check --------------------------------------
        hk, hn = greedy_rate(policy, ho, dev, a.bs)

        rec = {
            'round': rd,
            'rl_solved_round': len(solved), 'rl_solved_cum': len(cum_policy),
            'frozen_solved_round': len(fsolved), 'frozen_solved_cum': len(cum_frozen),
            'n_targets': len(rl),
            'transfer_solved': len(tsolved), 'transfer_frozen': len(ftsolved),
            'n_transfer': len(tf),
            'heldout_greedy': hk / hn, 'heldout_n': hn,
            'found_lengths': {int(L): len(s) for L, s in sorted(by_len.items())},
            'found_lengths_cum': {int(L): len(s) for L, s in sorted(cum_len_policy.items())},
            'frontier_round': robust_frontier(by_len),
            'frontier_cum': robust_frontier(cum_len_policy),
            'frozen_frontier': robust_frontier(cum_len_frozen),
            'frozen_lengths_cum': {int(L): len(sx) for L, sx in sorted(cum_len_frozen.items())},
            'transfer_lengths': {int(L): len(s) for L, s in sorted(tby_len.items())},
            'pool_size': len(train_pool),
            'relabelled_round': n_relab,
            'relabelled_lengths': {int(L): c for L, c in sorted(relab_len.items())},
            'secs': round(time.time() - t0, 1),
        }
        log.append(rec)
        p1, l1, h1 = wilson(len(solved), len(rl))
        p2, l2, h2 = wilson(len(fsolved), len(rl))
        p3, l3, h3 = wilson(len(tsolved), len(tf))
        print(f"round {rd}: RL {100*p1:.1f}% [{100*l1:.1f}-{100*h1:.1f}] "
              f"frozen {100*p2:.1f}% [{100*l2:.1f}-{100*h2:.1f}] | "
              f"transfer {100*p3:.1f}% (frozen {100*len(ftsolved)/len(tf):.1f}%) | "
              f"heldout greedy {100*hk/hn:.1f}% | "
              f"frontier {rec['frontier_cum']} (frozen {rec['frozen_frontier']}) | "
              f"pool {len(train_pool)} | {rec['secs']}s", flush=True)
        print('   found lengths (written):', rec['found_lengths'], flush=True)
        if filt is not None:
            print(f'   relabelled +{n_relab} '
                  f'(lengths {dict(sorted(relab_len.items()))})', flush=True)
        with open(f'{a.out}/log.json', 'w') as f:
            json.dump(log, f, indent=1)

        if rd == a.rounds:
            break

        # --- retrain from Stage-1 weights -------------------------------
        mix = rng.sample(sft, min(a.sft_mix, len(sft)))
        recs = list(train_pool.values()) + mix
        rng.shuffle(recs)
        policy, cfg = finetune(a.ckpt, recs, a.ft_steps, a.ft_bs, a.ft_lr,
                               dev, a.block, seed=a.seed + rd)
        torch.save({'cfg': cfg, 'model': policy.state_dict()},
                   f'{a.out}/policy_r{rd}.pt')

    torch.save({'cfg': load_ckpt(a.ckpt, device=dev).c.__dict__,
                'model': policy.state_dict()}, f'{a.out}/final.pt')
    with open(f'{a.out}/pool.jsonl', 'w') as f:
        for r in train_pool.values():
            f.write(json.dumps(r) + '\n')
    print('saved', a.out)


if __name__ == '__main__':
    main()
