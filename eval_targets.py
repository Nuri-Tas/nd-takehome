#!/usr/bin/env python3
"""Score a prove.py output file against a target set.

  python eval_targets.py --proofs out.jsonl                       # defaults to targets/validation_36_reference_proofs.jsonl
  python eval_targets.py --proofs out.jsonl --targets my_transfer.jsonl --by n_lines
  python eval_targets.py --proofs out.jsonl --judged judged.jsonl  # also dump per-theorem verdicts

--proofs : jsonl from prove.py, records {"name", "prompt", "proof": "<string>"} (one proof per theorem)
--targets: jsonl with {"name", "prompt"} and optionally "bin", "reference_lines" /
           "min_lines_ub" / "n_lines" (used for the per-length table) and
           "reference_proof".
A theorem is solved if its proof verifies as a proof of its prompt.
Reports: overall, by "bin", by length field, with 95% Wilson intervals; the
length of the proof written vs the reference length; and a failure-reason tally.
"""
import argparse, json, math, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nd_verify import verify_text


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def table(title, groups):
    print(f'\n{title}')
    print(f'  {"group":<12}{"solved":>8}{"n":>6}{"rate":>8}   95% CI')
    for g, (k, n) in groups:
        lo, hi = wilson(k, n)
        print(f'  {str(g):<12}{k:>8}{n:>6}{k/n:>8.3f}   [{lo:.3f}, {hi:.3f}]')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--proofs', required=True)
    ap.add_argument('--targets', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      'targets', 'validation_36_reference_proofs.jsonl'))
    ap.add_argument('--by', default=None, help='length field for the per-length table '
                    '(default: reference_lines, else min_lines_ub, else n_lines)')
    ap.add_argument('--judged', default=None, help='write per-theorem verdicts here')
    a = ap.parse_args()

    targets = {}
    for line in open(a.targets):
        if line.strip():
            r = json.loads(line); targets[r['name']] = r
    lenf = a.by or next((f for f in ('reference_lines', 'min_lines_ub', 'n_lines')
                         if any(f in t for t in targets.values())), None)

    seen, solved_set = set(), set()
    by_bin = collections.defaultdict(lambda: [0, 0])
    by_len = collections.defaultdict(lambda: [0, 0])
    reasons = collections.Counter()
    n_samples = n_valid = 0
    rows = []
    for line in open(a.proofs):
        if not line.strip():
            continue
        r = json.loads(line)
        name = r['name']
        if name not in targets:
            print(f'warning: {name} not in targets, skipped', file=sys.stderr); continue
        t = targets[name]
        if r.get('prompt', t['prompt']).split() != t['prompt'].split():
            print(f'warning: prompt mismatch for {name}', file=sys.stderr)
        if 'proof' not in r or not isinstance(r['proof'], str):
            sys.exit(f'record {name}: expected a single "proof" string per theorem (see submission_template/prove.py)')
        samples = [r['proof']]
        res = [verify_text(t['prompt'] + ' ' + s.strip()) for s in samples]
        oks = [x[0] for x in res]
        lens = [x[2] for x in res if x[0]]
        for ok, reason, _ in res:
            if not ok:
                reasons[reason.split(' (line')[0]] += 1
        n_samples += len(oks); n_valid += sum(oks)
        seen.add(name); solved = any(oks)
        if solved:
            solved_set.add(name)
        by_bin[t.get('bin', 'all')][0] += solved; by_bin[t.get('bin', 'all')][1] += 1
        L = t.get(lenf) if lenf else None
        by_len[L][0] += solved; by_len[L][1] += 1
        rows.append({'name': name, 'solved': solved,
                     'found_len': min(lens) if lens else None, 'ref_len': L,
                     'reasons': [x[1] for x in res if not x[0]]})

    missing = set(targets) - seen
    n = len(seen)
    print(f'targets: {len(targets)}  scored: {n}  missing from proofs file: {len(missing)}'
          + (f'  ({sorted(missing)[:5]}...)' if missing else ''))
    lo, hi = wilson(len(solved_set), n)
    print(f'\nSOLVED: {len(solved_set)}/{n} = {len(solved_set)/max(n,1):.3f}  [{lo:.3f}, {hi:.3f}]')
    if len(by_bin) > 1:
        table('by bin', sorted(by_bin.items(), key=lambda x: str(x[0])))
    if lenf:
        table(f'by {lenf}', sorted(by_len.items(), key=lambda x: (x[0] is None, x[0])))
        shorter = [(r['name'], r['found_len'], r['ref_len']) for r in rows
                   if r['found_len'] and r['ref_len'] and r['found_len'] < r['ref_len']]
        longer = [r for r in rows if r['found_len'] and r['ref_len'] and r['found_len'] > r['ref_len']]
        print(f'\nwritten-proof length vs reference: shorter {len(shorter)}, equal '
              f'{sum(1 for r in rows if r["found_len"] and r["found_len"] == r["ref_len"])}, longer {len(longer)}')
        for nm, b, rl in shorter:
            print(f'  SHORTER than reference: {nm}  found {b} < ref {rl}  <-- check it and tell us')
    print('\nsolved:   ' + ', '.join(sorted(solved_set)))
    print('unsolved: ' + ', '.join(sorted(seen - solved_set)))
    if reasons:
        print('\nfailure reasons:')
        for rs, c in reasons.most_common(12):
            print(f'  {c:6d}  {rs}')
    if a.judged:
        with open(a.judged, 'w') as f:
            for r in rows:
                f.write(json.dumps(r) + '\n')


if __name__ == '__main__':
    main()
