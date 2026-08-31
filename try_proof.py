#!/usr/bin/env python3
"""Write a proof by hand for a validation theorem and have the verifier judge it.

  python try_proof.py                      # list the 36 validation theorems with reference lengths
  python try_proof.py modus_tollens        # prints the prompt, then reads your proof from stdin
  python try_proof.py modus_tollens --show # also print the reference proof (spoiler)
  python try_proof.py --thm "( P > Q ) , P |- Q"   # any sequent, not only the 36

Type the proof body one line per proof line (or all on one line), finish with
QED, then Ctrl-D. Example:

  N1 ( P > Q ) : PR ;
  N2 ( ~ Q ) : PR ;
  N3 | P : AS ;
  N4 | Q : IMPE N1 N3 ;
  N5 | F : NEGE N4 N2 ;
  N6 ( ~ P ) : NEGI N3 N5 ;
  QED
"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from nd_verify import verify_text


def pretty(prompt, body):
    print('  ' + prompt)
    for ln in body.replace(' QED', '').split(' ; '):
        ln = ln.strip()
        if not ln or ln == 'QED':
            continue
        left, rule = ln.split(' : ')
        rule = rule.rstrip(' ;')
        idx, rest = left.split(' ', 1); d = 0
        while rest.startswith('| '):
            d += 1; rest = rest[2:]
        print(f'   {idx:>4} {"|   " * d}{rest:<34} {rule}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name', nargs='?')
    ap.add_argument('--thm', default=None, help='sequent "<p1> , <p2> |- <c>" (fully parenthesised)')
    ap.add_argument('--show', action='store_true')
    a = ap.parse_args()
    targets = [json.loads(l) for l in open(os.path.join(HERE, 'targets', 'validation_36_reference_proofs.jsonl'))]
    if a.thm:
        prem, concl = a.thm.split('|-')
        prompt = 'THM ' + (prem.strip() + ' ' if prem.strip() else '') + 'SEQ ' + concl.strip() + ' PRF'
        prompt = ' '.join(prompt.split()); ref = None
    elif a.name:
        t = next((t for t in targets if t['name'] == a.name), None)
        if t is None:
            sys.exit(f'unknown theorem {a.name!r}; run without arguments to list them')
        prompt, ref = t['prompt'], t
    else:
        for t in targets:
            print(f'{t["name"]:28s} {t["reference_lines"]:>3} lines  {"classical" if t["classical_only"] else "":9s} {t["thm"]}')
        return
    print(prompt)
    if a.show and ref:
        print('\nreference proof:'); pretty(prompt, ref['reference_proof'])
    print('\n-- type your proof, end with QED then Ctrl-D --')
    body = ' '.join(sys.stdin.read().split())
    if not body.endswith('QED'):
        body += ' QED'
    ok, reason, n = verify_text(prompt + ' ' + body)
    print(f'\nverifier: {"VALID" if ok else "INVALID"} — {reason} ({n} lines)')
    if ok:
        pretty(prompt, body)
        if ref:
            print(f'reference length {ref["reference_lines"]}; yours {n}'
                  + ('  <-- shorter than our reference, well done' if n < ref['reference_lines'] else ''))


if __name__ == '__main__':
    main()
