#!/usr/bin/env python3
"""Bucket probed candidates by the shortest proof anyone has found for them.

Expert iteration needs a ladder, not a wall. If every RL target is beyond the
Stage-1 model, round 1 collects no successes and nothing is learned; if every
target is inside the cap, the frontier never moves. So we grade the pool by
`shortest_found` from the probe:

    <= 6        inside the cap -- dropped, these prove nothing about "beyond 6"
    7, 8, 9...  known to be provable at that length and NOT provable in <= 6
                (as far as the probe searched) -- the curriculum rungs
    None        no proof found at all -- the stretch targets and transfer set

`shortest_found` is an upper bound: it is the shortest proof the probe
happened to find, so a theorem in the 8 bucket may still have a 7-line proof.
It is never treated as the theorem's true difficulty; every reported number
uses the length of the proof the model actually WROTE.
"""
import argparse, collections, json, random


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inp', nargs='+',
                    default=['data/rl_targets_hard.jsonl',
                             'data/transfer_hard.jsonl'])
    ap.add_argument('--n_rl', type=int, default=1200)
    ap.add_argument('--n_transfer', type=int, default=600)
    ap.add_argument('--seed', type=int, default=13)
    a = ap.parse_args()

    recs = []
    for p in a.inp:
        recs.extend(load(p))
    seen = set()
    uniq = []
    for r in recs:
        if r['prompt'] not in seen:
            seen.add(r['prompt'])
            uniq.append(r)
    print(f'{len(uniq)} probed candidates')

    buckets = collections.defaultdict(list)
    for r in uniq:
        s = r.get('shortest_found')
        buckets['unsolved' if s is None else int(s)].append(r)
    for k in sorted(buckets, key=lambda x: (x == 'unsolved', x)):
        print(f'  shortest_found={k}: {len(buckets[k])}')

    rng = random.Random(a.seed)
    # Rungs: everything the probe proved needs more than 6 lines.
    rungs = [r for k, v in buckets.items() if k != 'unsolved' and k > 6 for r in v]
    unsolved = list(buckets.get('unsolved', []))
    rng.shuffle(rungs)
    rng.shuffle(unsolved)
    print(f'\n{len(rungs)} graded rungs (7+), {len(unsolved)} unsolved')

    # Transfer is drawn from BOTH so it spans the same difficulty range as the
    # RL targets; it is never sampled during training.
    n_tr_rung = min(len(rungs) // 4, a.n_transfer // 2)
    n_tr_uns = min(len(unsolved) // 4, a.n_transfer - n_tr_rung)
    transfer = rungs[:n_tr_rung] + unsolved[:n_tr_uns]
    rl = rungs[n_tr_rung:] + unsolved[n_tr_uns:]
    rng.shuffle(transfer)
    rng.shuffle(rl)
    rl = rl[:a.n_rl]

    assert not ({r['prompt'] for r in rl} & {r['prompt'] for r in transfer})
    for name, rs in (('rl_targets_final', rl), ('transfer_final', transfer)):
        with open(f'data/{name}.jsonl', 'w') as f:
            for r in rs:
                f.write(json.dumps(r) + '\n')
        c = collections.Counter(
            'unsolved' if r.get('shortest_found') is None else r['shortest_found']
            for r in rs)
        print(f'{name}: n={len(rs)}  shortest_found={dict(sorted(c.items(), key=str))}')
    print('disjoint: ok')


if __name__ == '__main__':
    main()
