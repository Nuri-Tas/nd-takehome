#!/usr/bin/env python3
"""Build the Stage-2 pools: RL targets and a held-apart transfer set.

Both are generated at 7-16 lines with the same generator as Stage 1, then made
disjoint by theorem from Stage-1 train/held-out AND from each other. The
transfer set is never sampled during RL, so it measures generalisation rather
than memorisation of the targets.

Note the generating length is an UPPER BOUND on the shortest proof: a theorem
generated with a 12-line proof may well have a 5-line one. We record it as
`gen_lines` and never call it the theorem's difficulty.
"""
import collections, json, random, sys

from nd.dataset import canon_rename, is_trivial, write_jsonl
from nd.gen import generate
from nd_verify import verify_text

N_RL, N_TRANSFER = 4000, 1500


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    rng = random.Random(seed)
    seen = set()
    seen_canon = set()
    for fn in ('data/train.jsonl', 'data/heldout.jsonl'):
        for l in open(fn):
            r = json.loads(l)
            seen.add(r['prompt'])
            seen_canon.add(canon_rename(r['prompt']))
    # Never generate anything that is the validation set or a renaming of it.
    for fn in ('targets/validation_36.jsonl',):
        for l in open(fn):
            r = json.loads(l)
            seen.add(r['prompt'])
            seen_canon.add(canon_rename(r['prompt']))
    print('excluding', len(seen), 'known theorems')

    pool = {}
    attempts = 0
    while len(pool) < N_RL + N_TRANSFER and attempts < 4_000_000:
        attempts += 1
        L = rng.randint(7, 16)
        st = generate(rng, L)
        if st is None:
            continue
        ok, reason, n = verify_text(st.text())
        if not ok or not (7 <= n <= 16):
            continue
        p = st.prompt()
        if p in seen or canon_rename(p) in seen_canon:
            continue
        if is_trivial(p, n):
            continue
        if p not in pool or n < pool[p][1]:
            pool[p] = (st.text(), n)
    print(f'{len(pool)} theorems from {attempts} attempts')

    keys = sorted(pool)
    rng.shuffle(keys)
    rl, tr = keys[:N_RL], keys[N_RL:N_RL + N_TRANSFER]
    for name, ks in (('rl_targets', rl), ('transfer', tr)):
        recs = [{'thm': k, 'prompt': k, 'gen_lines': pool[k][1],
                 'gen_proof': pool[k][0].split(' PRF ', 1)[1]} for k in ks]
        write_jsonl(f'data/{name}.jsonl', recs)
        c = collections.Counter(r['gen_lines'] for r in recs)
        print(f'{name}: n={len(recs)} gen_lines={sorted(c.items())}')
    assert not (set(rl) & set(tr)), 'RL and transfer overlap'
    print('disjoint: ok')


if __name__ == '__main__':
    main()
