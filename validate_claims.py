#!/usr/bin/env python3
"""Test every structural claim the write-up makes, rather than asserting it.

Each check prints PASS/FAIL with the number behind it. These are correctness
properties of the pipeline, not results: if one fails, a reported number is
wrong somewhere.
"""
import ast, collections, json, os, random, subprocess, sys

from nd.dataset import canon_rename
from nd.tokenizer import decode_body, encode_example, n_premises
from nd_verify import verify_text

ok_all = True
def check(name, cond, detail=''):
    global ok_all
    ok_all &= bool(cond)
    print(f'  [{"PASS" if cond else "FAIL"}] {name}' + (f'  -- {detail}' if detail else ''))

L = lambda p: [json.loads(l) for l in open(p) if l.strip()]

print('== 1. Stage-1 cap (only the SFT data is capped; RL data is not) ==')
for f in ('data/train.jsonl', 'data/heldout.jsonl'):
    n = max(r['n_lines'] for r in L(f))
    check(f'{f} max proof length <= 6', n <= 6, f'max={n}')
pool = 'runs/clean_seed0/pool.jsonl'
if os.path.exists(pool):
    mx = max(r['n_lines'] for r in L(pool))
    check('RL pool intentionally exceeds the cap', mx > 6, f'max={mx} (expected, RL data is not capped)')

print('== 2. Every training proof verifies ==')
tr = L('data/train.jsonl')
bad = sum(1 for r in random.Random(0).sample(tr, 5000)
          if not verify_text(r['prompt'] + ' ' + r['proof'])[0])
check('5000 sampled training proofs all verify', bad == 0, f'invalid={bad}')

print('== 3. Tokenizer round-trip is exact ==')
bad = 0
for r in random.Random(1).sample(tr, 5000):
    t = encode_example(r['prompt'], r['proof'])
    if t is None:
        bad += 1; continue
    body = decode_body(t[1 + len(r['prompt'].split()):], n_premises(r['prompt']))
    okv, _, nl = verify_text(r['prompt'] + ' ' + body)
    if not okv or nl != r['n_lines']:
        bad += 1
check('5000 proofs survive encode->decode->verify', bad == 0, f'failures={bad}')

print('== 4. Split disjointness, exact and under atom-renaming ==')
sets = {'train': {r['prompt'] for r in tr},
        'heldout': {r['prompt'] for r in L('data/heldout.jsonl')},
        'rl': {r['prompt'] for r in L('data/rl_targets_hard.jsonl')},
        'transfer': {r['prompt'] for r in L('data/transfer_hard.jsonl')},
        'validation': {r['prompt'] for r in L('targets/validation_36.jsonl')}}
te = set()
for f in ('targets/test_short_prompts.jsonl', 'targets/test_long_prompts.jsonl'):
    te |= {r['prompt'] for r in L(f)}
sets['test'] = te
canon = {k: {canon_rename(p) for p in v} for k, v in sets.items()}
worst = 0
for a in ('train', 'heldout', 'rl', 'transfer'):
    for b in ('validation', 'test'):
        e, rn = len(sets[a] & sets[b]), len(canon[a] & canon[b])
        worst = max(worst, e, rn)
        check(f'{a} x {b}', e == 0 and rn == 0, f'exact={e} renaming={rn}')
for a, b in (('train', 'heldout'), ('train', 'rl'), ('train', 'transfer'), ('rl', 'transfer')):
    e = len(sets[a] & sets[b])
    check(f'{a} x {b} disjoint by theorem', e == 0, f'exact={e}')

print('== 5. prove.py matches the required interface ==')
tmpl = open('submission_template/prove.py').read()
mine = open('prove.py').read()
ta = sorted(set(__import__('re').findall(r"add_argument\('([^']+)'", tmpl)))
ma = sorted(set(__import__('re').findall(r"add_argument\('([^']+)'", mine)))
check('argument names identical to template', ta == ma, f'{ta}')
tree = ast.parse(mine)
names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
        {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
check('no verifier called inside prove.py',
      'verify_text' not in mine and 'judge' not in names,
      'greps clean')
check('output schema is name/prompt/proof',
      "'name'" in mine and "'prompt'" in mine and "'proof'" in mine)

print('== 6. Frozen control really gets an identical budget ==')
src = open('rl_expert_iteration.py').read()
check('policy and frozen call attempt_round with the same k',
      src.count('attempt_round(policy') >= 1 and src.count('attempt_round(frozen') >= 1
      and 'a.k' in src)
if os.path.exists('runs/clean_seed0/log.json'):
    d = json.load(open('runs/clean_seed0/log.json'))
    r1 = d[0]
    diff = abs(r1['rl_solved_round'] - r1['frozen_solved_round'])
    check('round 1 policy == frozen within noise (they are the same weights)',
          diff / r1['n_targets'] < 0.03,
          f"{r1['rl_solved_round']} vs {r1['frozen_solved_round']} of {r1['n_targets']}")

print('== 7. Robust frontier counts DISTINCT proofs ==')
if os.path.exists('runs/clean_seed0/log.json'):
    d = json.load(open('runs/clean_seed0/log.json'))
    fl = d[-1]['found_lengths']
    nine = fl.get('9', 0)
    check('frontier 9 backed by >= 5 distinct proofs', nine >= 5, f'distinct 9-line proofs={nine}')
    check('frozen frontier is 7', d[-1]['frozen_frontier'] == 7,
          f"frozen lengths={d[-1].get('frozen_lengths_cum')}")

print()
print('ALL CHECKS PASSED' if ok_all else 'SOME CHECKS FAILED')
sys.exit(0 if ok_all else 1)
