"""Tokenisation with position-free line references.

The spec's surface format numbers lines absolutely (`N5 : IMPE N1 N4`). Trained
under a 6-line cap, a model never sees `N7` and up, so at test time on a 12-line
proof it must emit tokens it has essentially no training signal for. That is a
tokenisation artefact, not a reasoning limit, and it is the thing most likely to
pin the frontier at the cap.

So we drop absolute indices from what the model sees:

  * the line index is implicit -- lines are separated by `;`, and the decoder
    re-numbers them N1, N2, ... when writing spec format;
  * a reference to an earlier *derived* line becomes a back-distance `B<k>`
    ("k lines above this one"), which stays small no matter how long the proof;
  * a reference to a *premise* becomes `P<i>` (the i-th premise), because
    premises get cited from arbitrarily far away and would otherwise be the one
    source of large, length-dependent distances.

Under this scheme a 14-line proof is built from the same reference tokens as a
4-line one. Everything is decoded back to spec format before the verifier
sees it, so `nd_verify` remains the only judge.
"""
from .dataset import parse_prompt

MAX_BACK = 24
MAX_PREM = 6

RULES = ['ANDI', 'ANDE1', 'ANDE2', 'IMPE', 'IMPI', 'ORI1', 'ORI2', 'ORE',
         'NEGE', 'NEGI', 'BOTE', 'DN', 'PR', 'AS', 'R']
SYMS = ['(', ')', '~', '&', 'v', '>', 'P', 'Q', 'R', 'S', 'F',
        'THM', ',', 'SEQ', 'PRF', 'QED', '|', ':', ';']
SPECIAL = ['<pad>', '<bos>']

def _dedup(seq):
    """Order-preserving unique. The atom `R` and the reiteration rule `R` are
    the same surface symbol, so they must share one id: leaving a duplicate in
    the list would strand an id that no token decodes to, and temperature
    sampling can emit exactly that id."""
    seen, out = set(), []
    for t in seq:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


VOCAB = _dedup(SPECIAL + SYMS + RULES
               + ['B%d' % i for i in range(1, MAX_BACK + 1)]
               + ['P%d' % i for i in range(1, MAX_PREM + 1)])
STOI = {t: i for i, t in enumerate(VOCAB)}
ITOS = {i: t for t, i in STOI.items()}
PAD, BOS = STOI['<pad>'], STOI['<bos>']
EOS = STOI['QED']
V = len(VOCAB)


def split_lines(body_toks):
    """Split a spec-format proof body into per-line token lists (drops QED)."""
    out, cur = [], []
    for t in body_toks:
        if t == 'QED':
            break
        cur.append(t)
        if t == ';':
            out.append(cur)
            cur = []
    return out


def encode_body(body, n_premises):
    """Spec-format proof body -> relative-reference tokens. None if malformed."""
    lines = split_lines(body.split())
    out = []
    for i, ln in enumerate(lines, start=1):
        if not ln[0].startswith('N'):
            return None
        idx = int(ln[0][1:])
        if i == 1:
            base = idx
        pos = idx - base + 1                      # 1-based position of this line
        j = 1
        toks = []
        while j < len(ln) and ln[j] == '|':
            toks.append('|')
            j += 1
        while j < len(ln) and ln[j] != ':':
            toks.append(ln[j])
            j += 1
        toks.append(':')
        j += 1
        toks.append(ln[j])
        j += 1
        while j < len(ln) and ln[j] != ';':
            r = int(ln[j][1:]) - base + 1          # 1-based position of target
            if r <= n_premises:
                toks.append('P%d' % r)
            else:
                d = pos - r
                if not (1 <= d <= MAX_BACK):
                    return None
                toks.append('B%d' % d)
            j += 1
        toks.append(';')
        out.extend(toks)
    out.append('QED')
    return out


def decode_body(toks, n_premises):
    """Relative-reference tokens -> spec-format body string with N indices.

    Deliberately does not repair anything: a reference the model got wrong
    decodes to an out-of-range index and the verifier rejects it, which is what
    we want to measure.
    """
    out = []
    pos = 0
    cur = []
    for t in toks:
        if t == 'QED':
            break
        if t in ('<pad>', '<bos>'):
            continue
        cur.append(t)
        if t == ';':
            pos += 1
            line = ['N%d' % pos]
            for u in cur:
                if u.startswith('B') and u[1:].isdigit():
                    line.append('N%d' % (pos - int(u[1:])))
                elif u.startswith('P') and u[1:].isdigit():
                    line.append('N%d' % int(u[1:]))
                else:
                    line.append(u)
            out.extend(line)
            cur = []
    out.append('QED')
    return ' '.join(out)


def n_premises(prompt):
    return len(parse_prompt(prompt)[0])


def encode_example(prompt, body):
    """Full training sequence: <bos> prompt-tokens relative-body-tokens."""
    enc = encode_body(body, n_premises(prompt))
    if enc is None:
        return None
    return ['<bos>'] + prompt.split() + enc


def to_ids(toks):
    return [STOI[t] for t in toks]
