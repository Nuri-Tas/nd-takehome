#!/usr/bin/env python3
"""Build RL/transfer pools of theorems that actually need more than 6 lines.

The first attempt at these pools generated theorems with 7-16 line proofs and
assumed that made them hard. It did not: the Stage-1 model solved 2706/4000 of
them from a single sample, writing proofs of 3-6 lines. A forward random walk
inserts redundant steps (ORI against a fresh formula, ANDI/ANDE round trips,
reiterations), so the generating length badly overestimates the shortest proof.
Running RL against that pool would have measured almost nothing.

So we filter by search instead of trusting the generating length. For every
candidate we spend a large sampling budget with the Stage-1 model itself, and
drop the theorem if any sample is a verified proof of 6 lines or fewer.

What survives is labelled honestly: `no <=6-line proof found in N samples`.
That is an upper-bound statement, not a proof of hardness -- the same status
the repo's own `min_lines_ub` field carries. Using a prover for analysis and
filtering is allowed; none of this generates training data.
"""
import argparse, collections, json, random, time

import torch

from nd.dataset import canon_rename, is_trivial
from nd.evaluate import judge, load_ckpt, sample_proofs
from nd.gen import generate
from nd_verify import verify_text

EASY_MAX = 6          # a proof this short means the theorem is inside the cap


def candidates(rng, n_want, seen, seen_canon, lo, hi, max_attempts):
    """Generate distinct, non-trivial theorems with generating length lo..hi."""
    pool = {}
    att = 0
    while len(pool) < n_want and att < max_attempts:
        att += 1
        L = rng.randint(lo, hi)
        st = generate(rng, L)
        if st is None:
            continue
        ok, _, n = verify_text(st.text())
        if not ok or not (lo <= n <= hi):
            continue
        p = st.prompt()
        if p in seen or canon_rename(p) in seen_canon or is_trivial(p, n):
            continue
        if p not in pool or n < pool[p][1]:
            pool[p] = (st.text().split(' PRF ', 1)[1], n)
    return pool, att


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='ckpt/sft.pt')
    ap.add_argument('--candidates', type=int, default=30000)
    ap.add_argument('--probe_k', type=int, default=32,
                    help='samples per candidate used to look for a short proof')
    ap.add_argument('--lo', type=int, default=9)
    ap.add_argument('--hi', type=int, default=16)
    ap.add_argument('--bs', type=int, default=2048)
    ap.add_argument('--n_rl', type=int, default=1500)
    ap.add_argument('--n_transfer', type=int, default=600)
    ap.add_argument('--seed', type=int, default=11)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)

    seen, seen_canon = set(), set()
    for fn in ('data/train.jsonl', 'data/heldout.jsonl',
               'targets/validation_36.jsonl',
               'targets/test_short_prompts.jsonl',
               'targets/test_long_prompts.jsonl'):
        try:
            for l in open(fn):
                if l.strip():
                    r = json.loads(l)
                    seen.add(r['prompt'])
                    seen_canon.add(canon_rename(r['prompt']))
        except FileNotFoundError:
            pass
    print(f'excluding {len(seen)} known theorems (train, heldout, validation, test)')

    t0 = time.time()
    pool, att = candidates(rng, a.candidates, seen, seen_canon, a.lo, a.hi,
                           a.candidates * 60)
    print(f'{len(pool)} candidates from {att} attempts ({time.time()-t0:.0f}s)')

    # --- probe: does a short proof exist? --------------------------------
    m = load_ckpt(a.ckpt)
    prompts = sorted(pool)
    best = {p: None for p in prompts}          # shortest verified proof found
    t0 = time.time()
    for i in range(a.probe_k):
        temp = 0.0 if i == 0 else 1.0          # one greedy pass, then sample
        bodies = sample_proofs(m, prompts, temperature=temp, bs=a.bs,
                               max_new=300)
        for p, b, (ok, _, n) in zip(prompts, bodies, judge(prompts, bodies)):
            if ok and (best[p] is None or n < best[p]):
                best[p] = n
        found = sum(1 for p in prompts if best[p] is not None and best[p] <= EASY_MAX)
        print(f'  probe {i+1}/{a.probe_k}: {found}/{len(prompts)} shown easy '
              f'({time.time()-t0:.0f}s)', flush=True)

    hard = [p for p in prompts if best[p] is None or best[p] > EASY_MAX]
    easy = len(prompts) - len(hard)
    print(f'\n{easy} candidates had a <=6-line proof and were dropped')
    print(f'{len(hard)} survive: no <=6-line proof found in {a.probe_k} samples')
    gl = collections.Counter(pool[p][1] for p in hard)
    print('  generating-length distribution of survivors:', sorted(gl.items()))

    rng.shuffle(hard)
    rl, tf = hard[:a.n_rl], hard[a.n_rl:a.n_rl + a.n_transfer]
    for name, ks in (('rl_targets_hard', rl), ('transfer_hard', tf)):
        with open(f'data/{name}.jsonl', 'w') as f:
            for k in ks:
                f.write(json.dumps({
                    'thm': k, 'prompt': k,
                    'gen_lines': pool[k][1],
                    'gen_proof': pool[k][0],
                    'shortest_found': best[k],
                    'probe_k': a.probe_k,
                }) + '\n')
        print(f'{name}: n={len(ks)}')
    assert not (set(rl) & set(tf)), 'RL and transfer overlap'
    print('disjoint: ok')


if __name__ == '__main__':
    main()
