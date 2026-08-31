#!/usr/bin/env python3
"""Score a prove.py output on the TEST set. Prints only the aggregate pass rate.

  python prove.py --ckpt <ckpt> --in targets/test_short_prompts.jsonl --out test_short_out.jsonl --greedy
  python score_test.py test_short_out.jsonl
  python prove.py --ckpt <ckpt> --in targets/test_long_prompts.jsonl --out test_long_out.jsonl --greedy
  python score_test.py test_long_out.jsonl

The test set is scored as a whole. Please do not compute or inspect per-theorem
verdicts on it and do not tune anything against it — treat the number the way
you would treat a leaderboard submission. We hold a further set you never see
and will compare. (The verifier is in your hands; this is a rule, not a lock.)
"""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from nd_verify import verify_text

TESTS = {}
for fn in ('test_short_prompts.jsonl', 'test_long_prompts.jsonl'):
    for l in open(os.path.join(HERE, 'targets', fn)):
        r = json.loads(l); TESTS[r['name']] = (fn, r['prompt'])


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    ok = n = 0; files = set(); missing = set(TESTS)
    for l in open(sys.argv[1]):
        if not l.strip():
            continue
        r = json.loads(l)
        if r['name'] not in TESTS:
            sys.exit(f'{r["name"]} is not a test theorem')
        fn, prompt = TESTS[r['name']]
        files.add(fn); missing.discard(r['name'])
        if 'proof' not in r or not isinstance(r['proof'], str):
            sys.exit('expected one "proof" string per record')
        n += 1; ok += verify_text(prompt + ' ' + r['proof'].strip())[0]
    if len(files) != 1:
        sys.exit('score one test file at a time')
    fn = files.pop(); total = sum(1 for v in TESTS.values() if v[0] == fn)
    miss = sum(1 for m in missing if TESTS[m][0] == fn)
    p = ok / total
    z = 1.96; d = 1 + z * z / total
    c = (p + z * z / (2 * total)) / d; h = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    print(f'{fn}: {100*p:.1f}% passed  ({ok}/{total}; 95% CI {100*(c-h):.1f}–{100*(c+h):.1f}%)'
          + (f'  [{miss} theorems missing from your file, counted as failed]' if miss else ''))


if __name__ == '__main__':
    main()
