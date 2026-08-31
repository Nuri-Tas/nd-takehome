"""Independent Fitch-style proof verifier.

Operates purely on token sequences (the model's output format). Shares only the
token conventions with core.py — all parsing and rule checking is implemented
from scratch here, so generator bugs and verifier bugs are independent.

Token format of one line:   N<i> ['|']*depth <formula-tokens> : <RULE> [N<r>...] ;
A proof is a sequence of lines terminated by QED.

Box (subproof) semantics:
- PR lines: only at the start, depth 0, must exactly match the declared premises
  in order.
- An AS line at depth d (1 <= d <= cur_depth+1) closes any boxes deeper than d-1
  and opens a new box whose hypothesis is the AS formula.
- Any other line at depth d (d <= cur_depth) closes boxes deeper than d.
- A line j is citable from line i iff j < i and ctx(j) is a prefix of ctx(i),
  where ctx is the stack of open box ids after the line is placed.
- A box (s,e) is citable from i iff s is an AS line opening box b, e is the last
  line of b, depth(e) == depth(s), b is closed at i, and the parent context of b
  is a prefix of ctx(i).
"""

BOT = ('bot',)
ATOM_NAMES = {'P', 'Q', 'R', 'S'}
RULE_NAMES = {'ANDI', 'ANDE1', 'ANDE2', 'IMPE', 'IMPI', 'ORI1', 'ORI2', 'ORE',
              'NEGE', 'NEGI', 'BOTE', 'DN', 'PR', 'AS', 'R'}


class ParseError(Exception):
    pass


def parse_formula(toks, i):
    """Recursive-descent parse starting at toks[i]. Returns (formula, next_i)."""
    if i >= len(toks):
        raise ParseError('eof in formula')
    t = toks[i]
    if t in ATOM_NAMES:
        return ('atom', t), i + 1
    if t == 'F':
        return BOT, i + 1
    if t == '(':
        if i + 1 < len(toks) and toks[i + 1] == '~':
            sub, j = parse_formula(toks, i + 2)
            if j >= len(toks) or toks[j] != ')':
                raise ParseError('missing ) after ~')
            return ('not', sub), j + 1
        left, j = parse_formula(toks, i + 1)
        if j >= len(toks) or toks[j] not in ('&', 'v', '>'):
            raise ParseError('missing binop')
        op = {'&': 'and', 'v': 'or', '>': 'imp'}[toks[j]]
        right, k = parse_formula(toks, j + 1)
        if k >= len(toks) or toks[k] != ')':
            raise ParseError('missing )')
        return (op, left, right), k + 1
    raise ParseError(f'bad formula token {t!r}')


def parse_proof_tokens(toks):
    """Parse the proof body (after PRF, before/including QED) into line dicts."""
    lines = []
    i = 0
    while i < len(toks):
        if toks[i] == 'QED':
            return lines
        # N<i>
        t = toks[i]
        if not (t.startswith('N') and t[1:].isdigit()):
            raise ParseError(f'expected line index, got {t!r}')
        idx = int(t[1:])
        i += 1
        depth = 0
        while i < len(toks) and toks[i] == '|':
            depth += 1
            i += 1
        formula, i = parse_formula(toks, i)
        if i >= len(toks) or toks[i] != ':':
            raise ParseError('missing :')
        i += 1
        if i >= len(toks) or toks[i] not in RULE_NAMES:
            raise ParseError('bad rule name')
        rule = toks[i]
        i += 1
        refs = []
        while i < len(toks) and toks[i].startswith('N') and toks[i][1:].isdigit():
            refs.append(int(toks[i][1:]))
            i += 1
        if i >= len(toks) or toks[i] != ';':
            raise ParseError('missing ;')
        i += 1
        lines.append({'idx': idx, 'depth': depth, 'formula': formula,
                      'rule': rule, 'refs': refs})
    raise ParseError('missing QED')


def verify(premises, conclusion, proof_toks):
    """Check that proof_toks is a valid proof of premises |- conclusion.

    premises: list of formulas (tuples), in the order given in the prompt.
    conclusion: formula.
    proof_toks: list of tokens after 'PRF' (QED terminated).
    Returns (ok: bool, reason: str).
    """
    try:
        lines = parse_proof_tokens(proof_toks)
    except ParseError as e:
        return False, f'parse: {e}'
    if not lines:
        return False, 'empty proof'

    n = len(lines)
    # Line indices must be consecutive ascending. The starting index may be any
    # value >= 1 (a proof is judged by its structure, not by where its
    # numbering starts).
    idx0 = lines[0]['idx']
    if idx0 < 1:
        return False, 'line index must be >= 1'
    for k, ln in enumerate(lines):
        if ln['idx'] != idx0 + k:
            return False, f'line index mismatch at position {k+1}'

    # context tracking
    stack = []            # list of box ids
    next_box = [0]
    ctx = {}              # idx -> tuple of open box ids after the line
    box_start = {}        # box id -> start line idx
    box_lines = {}        # box id -> last line idx seen inside the box
    box_depth = {}        # box id -> depth of its AS line
    as_lines = {}         # idx -> box id for AS lines
    in_premise_block = True

    for ln in lines:
        idx, d, rule = ln['idx'], ln['depth'], ln['rule']
        if rule == 'PR':
            if not in_premise_block or d != 0:
                return False, f'PR misplaced at line {idx}'
        else:
            in_premise_block = False
        if rule == 'AS':
            if d < 1 or d > len(stack) + 1:
                return False, f'bad AS depth at line {idx}'
            del stack[d - 1:]
            next_box[0] += 1
            b = next_box[0]
            stack.append(b)
            box_start[b] = idx
            box_depth[b] = d
            as_lines[idx] = b
        else:
            if d > len(stack):
                return False, f'depth jump at line {idx}'
            del stack[d:]
        for b in stack:
            box_lines[b] = idx
        ctx[idx] = tuple(stack)

    # check premises exactly match
    pr = [ln for ln in lines if ln['rule'] == 'PR']
    if [p['formula'] for p in pr] != list(premises):
        return False, 'premise block does not match declared premises'

    # final line
    last = lines[-1]
    if last['depth'] != 0:
        return False, 'proof ends inside a subproof'
    if last['formula'] != conclusion:
        return False, 'final formula is not the conclusion'
    if last['rule'] == 'AS':
        return False, 'conclusion cannot be an assumption'
    if last['rule'] == 'PR' and conclusion not in list(premises):
        return False, 'conclusion PR not among premises'

    fml = {ln['idx']: ln['formula'] for ln in lines}

    def line_citable(j, i):
        if not (1 <= j < i):
            return False
        cj, ci = ctx[j], ctx[i]
        return cj == ci[:len(cj)]

    def box_citable(s, e, i):
        if s not in as_lines:
            return False
        b = as_lines[s]
        if e != box_lines.get(b):
            return False
        # conclusion of box must sit at box level
        eline = lines[e - idx0]
        if eline['depth'] != box_depth[b]:
            return False
        if e < s:
            return False
        ci = ctx[i]
        if b in ci:
            return False  # cannot discharge a box you are inside
        parent = ctx[s][:-1]
        return parent == ci[:len(parent)]

    for ln in lines:
        idx, d, rule, refs, G = (ln['idx'], ln['depth'], ln['rule'],
                                 ln['refs'], ln['formula'])
        if rule in ('PR', 'AS'):
            if refs:
                return False, f'{rule} takes no refs (line {idx})'
            continue
        # arity table
        arity = {'ANDI': 2, 'ANDE1': 1, 'ANDE2': 1, 'IMPE': 2, 'IMPI': 2,
                 'ORI1': 1, 'ORI2': 1, 'ORE': 5, 'NEGE': 2, 'NEGI': 2,
                 'BOTE': 1, 'DN': 1, 'R': 1}[rule]
        if len(refs) != arity:
            return False, f'wrong ref count for {rule} (line {idx})'
        if any(not (idx0 <= r <= idx0 + n - 1) for r in refs):
            return False, f'ref out of range (line {idx})'

        if rule in ('IMPI', 'NEGI'):
            s, e = refs
            if not box_citable(s, e, idx):
                return False, f'bad box cite (line {idx})'
        elif rule == 'ORE':
            j, s1, e1, s2, e2 = refs
            if not line_citable(j, idx):
                return False, f'bad line cite (line {idx})'
            if not box_citable(s1, e1, idx) or not box_citable(s2, e2, idx):
                return False, f'bad box cite (line {idx})'
        else:
            for r in refs:
                if not line_citable(r, idx):
                    return False, f'bad line cite (line {idx})'

        F = lambda r: fml[r]
        ok = True
        if rule == 'R':
            ok = G == F(refs[0])
        elif rule == 'ANDI':
            ok = G == ('and', F(refs[0]), F(refs[1]))
        elif rule == 'ANDE1':
            ok = F(refs[0])[0] == 'and' and F(refs[0])[1] == G
        elif rule == 'ANDE2':
            ok = F(refs[0])[0] == 'and' and F(refs[0])[2] == G
        elif rule == 'IMPE':
            ok = F(refs[0]) == ('imp', F(refs[1]), G)
        elif rule == 'IMPI':
            s, e = refs
            ok = G == ('imp', F(s), F(e))
        elif rule == 'ORI1':
            ok = G[0] == 'or' and G[1] == F(refs[0])
        elif rule == 'ORI2':
            ok = G[0] == 'or' and G[2] == F(refs[0])
        elif rule == 'ORE':
            j, s1, e1, s2, e2 = refs
            ok = (F(j) == ('or', F(s1), F(s2)) and F(e1) == G and F(e2) == G)
        elif rule == 'NEGE':
            ok = F(refs[1]) == ('not', F(refs[0])) and G == BOT
        elif rule == 'NEGI':
            s, e = refs
            ok = F(e) == BOT and G == ('not', F(s))
        elif rule == 'BOTE':
            ok = F(refs[0]) == BOT
        elif rule == 'DN':
            ok = F(refs[0]) == ('not', ('not', G))
        if not ok:
            return False, f'rule check failed: {rule} (line {idx})'

    return True, 'ok'


def verify_text(text):
    """Verify a full example string 'THM ... SEQ ... PRF ... QED'.

    Returns (ok, reason, n_lines)."""
    toks = text.split() if isinstance(text, str) else list(text)
    try:
        if not toks or toks[0] != 'THM':
            return False, 'missing THM', 0
        i = 1
        premises = []
        if toks[i] != 'SEQ':
            while True:
                f, i = parse_formula(toks, i)
                premises.append(f)
                if toks[i] == ',':
                    i += 1
                    continue
                break
        if toks[i] != 'SEQ':
            return False, 'missing SEQ', 0
        concl, i = parse_formula(toks, i + 1)
        if i >= len(toks) or toks[i] != 'PRF':
            return False, 'missing PRF', 0
        body = toks[i + 1:]
        ok, reason = verify(premises, concl, body)
        nl = sum(1 for t in body if t == ';')
        return ok, reason, nl
    except ParseError as e:
        return False, f'parse: {e}', 0
