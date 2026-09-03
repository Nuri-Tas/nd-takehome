import json, time, collections, sys
from nd.dataset import build, split, renaming_overlap, records, write_jsonl, canon_rename

def contamination_filter():
    """Theorems that must never enter Stage-1 training: the validation set, the
    test sets, and atom-renamings of either. With only four atoms a random
    generator hits textbook sequents by chance, so this has to be explicit."""
    exact, canon = set(), set()
    for fn in ('targets/validation_36.jsonl',
               'targets/test_short_prompts.jsonl',
               'targets/test_long_prompts.jsonl'):
        for l in open(fn):
            if l.strip():
                p = json.loads(l)['prompt']
                exact.add(p); canon.add(canon_rename(p))
    return exact, canon

t0=time.time()
N=int(sys.argv[1]) if len(sys.argv)>1 else 120000
by_thm, attempts = build(N, seed=1)
print(f'{len(by_thm)} distinct theorems from {attempts} attempts  ({time.time()-t0:.0f}s)')
ex, cn = contamination_filter()
before = len(by_thm)
by_thm = {k: v for k, v in by_thm.items()
          if k not in ex and canon_rename(k) not in cn}
print(f'decontamination: removed {before-len(by_thm)} theorems that are '
      f'validation/test theorems or renamings of them -> {len(by_thm)} kept')
train, held = split(by_thm, n_heldout=3000, seed=1)
print('train', len(train), 'held', len(held))
print('renaming overlap held->train: %.1f%%' % (100*renaming_overlap(train, held)))
tr, hd = records(by_thm, train), records(by_thm, held)
for name, rs in (('train', tr), ('heldout', hd)):
    write_jsonl(f'data/{name}.jsonl', rs)
    c=collections.Counter(r['n_lines'] for r in rs)
    triv=sum(r['trivial'] for r in rs)/len(rs)
    print(f'{name}: n={len(rs)} lens={sorted(c.items())} trivial={100*triv:.1f}%')
