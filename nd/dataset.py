"""Build the Stage-1 corpus: sample proofs, dedupe by theorem, split.

Two properties the exam asks about explicitly and that we measure here:
  * the split is disjoint *by theorem* (a sequent never appears on both sides);
  * how much of the held-out set is a mere atom-renaming of a training theorem,
    which would make the held-out number look better than it is.
"""
import json
import random
import re
from collections import Counter

from nd_verify import verify_text
from .formula import ATOMS
from .gen import generate

TRIVIAL_MAXLEN = 2


def canon_rename(prompt):
    """Canonical form under atom renaming: rename atoms P,Q,R,S in order of
    first appearance. Two theorems with the same key differ only by which
    letters were used."""
    seen = {}
    out = []
    for t in prompt.split():
        if t in ATOMS:
            if t not in seen:
                seen[t] = ATOMS[len(seen)]
            out.append(seen[t])
        else:
            out.append(t)
    return ' '.join(out)


def parse_prompt(prompt):
    """'THM a , b SEQ c PRF' -> (['a','b'], 'c'). Zero premises is 'THM SEQ c PRF'."""
    body = prompt[len('THM '):-len(' PRF')]
    head, concl = body.rsplit(' SEQ ', 1) if ' SEQ ' in body else ('', body[len('SEQ '):])
    prems = [x.strip() for x in head.split(' , ')] if head.strip() else []
    return prems, concl.strip()


def is_trivial(prompt, n_lines):
    """Theorems that inflate a solve rate: conclusion is literally a premise,
    or a premise is F (everything follows), or the proof is a single step."""
    prems, concl = parse_prompt(prompt)
    if concl in prems:
        return True
    if 'F' in prems:
        return True
    return n_lines <= TRIVIAL_MAXLEN


def build(n_target, seed=0, lo=2, hi=6, max_attempts=None):
    """Sample until n_target distinct theorems of length lo..hi are collected."""
    rng = random.Random(seed)
    by_thm = {}
    attempts = 0
    max_attempts = max_attempts or n_target * 40
    while len(by_thm) < n_target and attempts < max_attempts:
        attempts += 1
        L = rng.randint(lo, hi)
        st = generate(rng, L)
        if st is None:
            continue
        text = st.text()
        ok, reason, n = verify_text(text)
        if not ok or not (lo <= n <= hi):
            continue
        p = st.prompt()
        # Keep the shortest proof found for a theorem: it is the one we want
        # the model to imitate.
        if p not in by_thm or n < by_thm[p][1]:
            by_thm[p] = (text, n)
    return by_thm, attempts


def split(by_thm, n_heldout, seed=0):
    """Theorem-disjoint train / held-out split."""
    rng = random.Random(seed)
    keys = sorted(by_thm)
    rng.shuffle(keys)
    held = keys[:n_heldout]
    train = keys[n_heldout:]
    return train, held


def renaming_overlap(train_keys, held_keys):
    """Fraction of held-out theorems that are an atom-renaming of a train one."""
    tr = {canon_rename(k) for k in train_keys}
    hit = sum(1 for k in held_keys if canon_rename(k) in tr)
    return hit / max(1, len(held_keys))


def records(by_thm, keys):
    out = []
    for k in keys:
        text, n = by_thm[k]
        body = text.split(' PRF ', 1)[1]
        out.append({'thm': k, 'prompt': k, 'text': text, 'proof': body,
                    'n_lines': n, 'trivial': is_trivial(k, n)})
    return out


def write_jsonl(path, recs):
    with open(path, 'w') as f:
        for r in recs:
            f.write(json.dumps(r) + '\n')
