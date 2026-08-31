#!/usr/bin/env python3
"""Judge a jsonl of proof attempts with the exam verifier.

Each input record is one of
  {"prompt": "THM ... PRF", "proof": "N1 ... QED", ...}     # one proof for one theorem (prove.py output)
  {"text": "THM ... PRF N1 ... QED", ...}                   # full string (e.g. a training example)

Prints the fraction of records whose proof verifies. With --reasons, tallies
failure reasons. With --out, writes each record back with "ok", "reason",
"n_lines" added.
"""
import argparse, json, sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nd_verify import verify_text


def attempt(rec):
    if 'text' in rec:
        return rec['text']
    return rec['prompt'].strip() + ' ' + rec['proof'].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inp')
    ap.add_argument('--out', default=None)
    ap.add_argument('--reasons', action='store_true')
    a = ap.parse_args()
    n = valid = 0
    reasons = collections.Counter()
    fo = open(a.out, 'w') if a.out else None
    for line in open(a.inp):
        if not line.strip():
            continue
        rec = json.loads(line)
        ok, reason, nl = verify_text(attempt(rec))
        n += 1; valid += ok
        if not ok:
            reasons[reason.split(' (line')[0]] += 1
        if fo:
            rec['ok'] = ok; rec['reason'] = reason; rec['n_lines'] = nl
            fo.write(json.dumps(rec) + '\n')
    print(f'records: {n}  valid: {valid}  rate: {valid/max(n,1):.4f}')
    if a.reasons:
        for r, c in reasons.most_common(25):
            print(f'{c:7d}  {r}')


if __name__ == '__main__':
    main()
