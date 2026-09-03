"""Hindsight relabelling of off-target proofs.

At temperature 1 most samples fail the *prompted* sequent, but a good fraction
of them are still perfectly valid proofs -- of some other theorem. Reward for
the target is zero, yet the tokens are a sound derivation, and the README
allows reusing them as training data as long as they are declared and kept
disjoint from every evaluation set.

So we read the proof back: its PR lines are its premises, its last line is its
conclusion, and that pair is the theorem it actually proved. If the verifier
accepts it under that reading, it becomes a training record.

This matters most at the cold start. The Stage-1 model almost never nails a
7+ line target head-on, so on-target rewards alone would give expert iteration
nothing to learn from in round 1; relabelling turns the same sampling budget
into a supply of long, self-generated, verified proofs.

Contamination control: a relabelled theorem is dropped if it (or an atom
renaming of it) appears in the Stage-1 held-out set, the transfer pool, the
validation 36, or either test file. Filtering against the test prompts is
hygiene, not tuning -- we never look at test verdicts.
"""
import json
import os

from nd_verify import verify_text
from nd_verify.verify import ParseError, parse_proof_tokens

from .dataset import canon_rename
from .formula import fmt


def implied_prompt(body):
    """The sequent a proof body actually proves, or None if it is malformed."""
    try:
        lines = parse_proof_tokens(body.split())
    except ParseError:
        return None
    if not lines:
        return None
    last = lines[-1]
    if last['depth'] != 0 or last['rule'] == 'AS':
        return None
    prems = [ln['formula'] for ln in lines if ln['rule'] == 'PR']
    concl = last['formula']
    if last['rule'] == 'PR' and concl not in prems:
        return None
    head = ' , '.join(fmt(p) for p in prems)
    return 'THM ' + (head + ' ' if head else '') + 'SEQ ' + fmt(concl) + ' PRF'


class RelabelFilter:
    """Holds every theorem a relabelled proof must avoid."""

    def __init__(self, paths=('data/heldout.jsonl', 'data/transfer.jsonl',
                              'targets/validation_36.jsonl',
                              'targets/test_short_prompts.jsonl',
                              'targets/test_long_prompts.jsonl')):
        self.exact, self.canon = set(), set()
        for p in paths:
            if not os.path.exists(p):
                continue
            for l in open(p):
                if not l.strip():
                    continue
                r = json.loads(l)
                pr = r.get('prompt')
                if pr:
                    self.exact.add(pr)
                    self.canon.add(canon_rename(pr))
        self.n_sources = len(self.exact)

    def blocked(self, prompt):
        return prompt in self.exact or canon_rename(prompt) in self.canon


def relabel(body, filt, min_lines=1):
    """-> (prompt, n_lines) for a verified, non-contaminating relabel, else None."""
    p = implied_prompt(body)
    if p is None or filt.blocked(p):
        return None
    ok, _, n = verify_text(p + ' ' + body)
    if not ok or n < min_lines:
        return None
    return p, n
