"""Forward random-walk generator of verifier-valid natural-deduction proofs.

Sampling a proof, not a theorem: we start from random premises, repeatedly
apply a randomly chosen *applicable* rule, and whatever the final line says
becomes the conclusion. Soundness is therefore free -- every proof is valid by
construction, and is re-checked with nd_verify before it is kept.

A plain random walk almost never produces the three box-discharging rules
(IMPI / NEGI / ORE), because they need a box to have been opened earlier with
exactly the right hypothesis. So alongside single-rule steps we sample
*macros* that plan a whole box: assume, derive, discharge.
"""
import random

from .formula import BOT, fmt, random_formula, size, subformulas
from .proofstate import ProofState

MAX_FORMULA_SIZE = 12

LOCAL_RULES = ['ANDI', 'ANDE1', 'ANDE2', 'IMPE', 'ORI1', 'ORI2',
               'NEGE', 'BOTE', 'DN', 'R']
LOCAL_W = [0.7, 1.0, 1.0, 3.0, 0.6, 0.6, 2.0, 1.5, 2.5, 0.10]


def _assumption_pool(st, d, rng):
    """Candidate AS formulas, biased toward things already in play: an
    assumption unrelated to the context leads to a box that proves nothing."""
    cit = st.citable_lines(d)
    pool = []
    for i in cit:
        f = st.f(i)
        for s in subformulas(f):
            if size(s) <= MAX_FORMULA_SIZE:
                pool.append(s)
                if s[0] == 'not':
                    pool.append(s[1])
                elif size(s) < MAX_FORMULA_SIZE:
                    pool.append(('not', s))
    for _ in range(3):
        pool.append(random_formula(rng, max_depth=1))
    return pool


def _moves(st, d, rng):
    """All applicable non-discharging moves, grouped by rule."""
    cit = st.citable_lines(d)
    out = {}
    if not cit:
        return out
    byf = {}
    for i in cit:
        byf.setdefault(st.f(i), i)

    c = [i for i in cit if st.f(i)[0] == 'and']
    if c:
        out['ANDE1'] = [(st.f(i)[1], 'ANDE1', [i]) for i in c]
        out['ANDE2'] = [(st.f(i)[2], 'ANDE2', [i]) for i in c]
    c = [(i, byf[st.f(i)[1]]) for i in cit
         if st.f(i)[0] == 'imp' and st.f(i)[1] in byf]
    if c:
        out['IMPE'] = [(st.f(a)[2], 'IMPE', [a, b]) for a, b in c]
    c = [(byf[st.f(i)[1]], i) for i in cit
         if st.f(i)[0] == 'not' and st.f(i)[1] in byf]
    if c:
        out['NEGE'] = [(BOT, 'NEGE', [a, b]) for a, b in c]
    c = [i for i in cit if st.f(i) == BOT]
    if c:
        out['BOTE'] = [(random_formula(rng, max_depth=1), 'BOTE', [i])
                       for i in c]
    c = [i for i in cit if st.f(i)[0] == 'not' and st.f(i)[1][0] == 'not']
    if c:
        out['DN'] = [(st.f(i)[1][1], 'DN', [i]) for i in c]
    # ANDI / ORI / R apply to anything, so sample a handful rather than
    # enumerating the full quadratic set.
    cand = []
    for _ in range(4):
        a, b = rng.choice(cit), rng.choice(cit)
        g = ('and', st.f(a), st.f(b))
        if size(g) <= MAX_FORMULA_SIZE:
            cand.append((g, 'ANDI', [a, b]))
    if cand:
        out['ANDI'] = cand
    for rule in ('ORI1', 'ORI2'):
        cand = []
        for _ in range(3):
            a = rng.choice(cit)
            x = random_formula(rng, max_depth=1)
            g = ('or', st.f(a), x) if rule == 'ORI1' else ('or', x, st.f(a))
            if size(g) <= MAX_FORMULA_SIZE:
                cand.append((g, rule, [a]))
        if cand:
            out[rule] = cand
    out['R'] = [(st.f(i), 'R', [i]) for i in cit]
    return out


def local_step(st, d, rng, tries=12):
    """Sample one applicable non-discharging rule at depth d."""
    mv = _moves(st, d, rng)
    if not mv:
        return None
    rules = [r for r in LOCAL_RULES if r in mv]
    w = [LOCAL_W[LOCAL_RULES.index(r)] for r in rules]
    return rng.choice(mv[rng.choices(rules, weights=w)[0]])


def walk(st, d, n, rng):
    """Write up to n local lines at depth d. Returns how many were written."""
    k = 0
    for _ in range(n):
        m = local_step(st, d, rng)
        if m is None:
            break
        st.add(d, m[0], m[1], m[2])
        k += 1
    return k


def macro_box(st, rng, budget):
    """assume A -> derive -> discharge with IMPI or NEGI. Costs >= 2 lines."""
    if budget < 2:
        return False
    d = st.depth + 1
    pool = _assumption_pool(st, d - 1, rng)
    if not pool:
        return False
    cit_out = st.citable_lines(d - 1)
    outer = {st.f(i) for i in cit_out}
    # Prefer an assumption whose negation is already available: that box closes
    # into a contradiction and gives us NEGI (reductio), the rule a forward
    # walk otherwise never reaches.
    negatable = [a for a in pool if ('not', a) in outer or a[0] == 'not' and a[1] in outer]
    want_neg = negatable and rng.random() < 0.5
    a = rng.choice(negatable if want_neg else pool)
    st.add(d, a, 'AS', [])
    inner = max(0, budget - 2)
    walk(st, d, rng.randint(0, min(inner, 3)), rng)
    boxes = st.citable_boxes(d - 1)
    if not boxes:
        return False
    s, e = max(boxes)
    fe = st.f(e)
    if fe == BOT and rng.random() < 0.9:
        st.add(d - 1, ('not', st.f(s)), 'NEGI', [s, e])
    else:
        g = ('imp', st.f(s), fe)
        if size(g) > MAX_FORMULA_SIZE:
            g = None
        if g is None:
            return False
        st.add(d - 1, g, 'IMPI', [s, e])
    return True


def macro_ore(st, rng, budget):
    """Case split on a citable disjunction.

    Both branches must end on the *same* formula G. We generate each branch by
    random walk, collect the formulas each reached at box level, intersect, and
    truncate both branches to end at a shared G. This is what makes ORE come
    out of a forward walk at all.
    """
    d0 = st.depth
    cit = st.citable_lines(d0)
    disj = [i for i in cit if st.f(i)[0] == 'or']
    if not disj or budget < 3:
        return False
    j = rng.choice(disj)
    A, B = st.f(j)[1], st.f(j)[2]
    d = d0 + 1
    per = max(1, (budget - 3) // 2)

    def branch(hyp):
        probe = st.clone()
        probe.add(d, hyp, 'AS', [])
        start = len(probe.lines)
        walk(probe, d, rng.randint(0, per), rng)
        reach = {}
        for ln in probe.lines[start - 1:]:
            if ln['depth'] == d:
                reach.setdefault(ln['formula'], ln['idx'])
        return probe, reach

    # Both branches must land on the same formula. One sample of each rarely
    # collides, so retry a few times and prefer a G that is not already
    # available outside the split (i.e. a case analysis that does real work).
    outer = {st.f(i) for i in cit}
    best = None
    for _ in range(5):
        p1, r1 = branch(A)
        p2, r2 = branch(B)
        shared = set(r1) & set(r2)
        if not shared:
            continue
        useful = sorted((shared - outer), key=fmt)
        pick = useful if useful else sorted(shared, key=fmt)
        best = (p1, r1, p2, r2, rng.choice(pick))
        if useful:
            break
    if best is None:
        return False
    p1, r1, p2, r2, g = best
    # Replay both branches for real, truncated to end exactly at g.
    s1 = len(st.lines) + 1
    for ln in p1.lines[s1 - 1:r1[g]]:
        st.add(ln['depth'], ln['formula'], ln['rule'], ln['refs'])
    e1 = len(st.lines)
    off = len(st.lines) - (s1 - 1)
    s2 = len(st.lines) + 1
    for ln in p2.lines[s1 - 1:r2[g]]:
        refs = [r + off if r >= s1 else r for r in ln['refs']]
        st.add(ln['depth'], ln['formula'], ln['rule'], refs)
    e2 = len(st.lines)
    st.add(d0, g, 'ORE', [j, s1, e1, s2, e2])
    return True


def sample_premises(rng, max_premises):
    """Premises with structure.

    Uniformly random premises give a walk almost nothing to work with: no
    top-level disjunction means ORE never fires, and an implication without its
    antecedent means IMPE never fires. Half the time we therefore lay down a
    classic pattern and let the walk take it from there.
    """
    F = lambda d=1: random_formula(rng, max_depth=d)
    if rng.random() < 0.55:
        A, B, C = F(), F(), F(1)
        pat = rng.choices(
            ['imp_mp', 'imp_mt', 'disj', 'disj_syl', 'conj', 'chain', 'neg'],
            weights=[2.0, 1.5, 2.5, 1.5, 1.0, 1.2, 1.0])[0]
        if pat == 'imp_mp':
            p = [('imp', A, B), A]
        elif pat == 'imp_mt':
            p = [('imp', A, B), ('not', B)]
        elif pat == 'disj':
            p = [('or', A, B)]
        elif pat == 'disj_syl':
            p = [('or', A, B), ('not', A)]
        elif pat == 'conj':
            p = [('and', A, B)]
        elif pat == 'chain':
            p = [('imp', A, B), ('imp', B, C), A]
        else:
            p = [A, ('not', A)]
        rng.shuffle(p)
        p = [q for q in p if size(q) <= MAX_FORMULA_SIZE]
        return p[:max_premises]
    npr = rng.choices([0, 1, 2, 3], weights=[0.12, 0.34, 0.36, 0.18])[0]
    return [random_formula(rng, max_depth=rng.choice([1, 2, 2]))
            for _ in range(min(npr, max_premises))]


def generate(rng, target_len, max_premises=3):
    """Sample one proof of exactly `target_len` lines, or None on a dud."""
    prem = sample_premises(rng, max_premises)
    if len(prem) > target_len:
        return None
    st = ProofState(prem)
    guard = 0
    fails = 0
    while len(st.lines) < target_len and guard < 80:
        guard += 1
        remaining = target_len - len(st.lines)
        # Every open box costs one line to discharge; never over-commit.
        if remaining <= st.depth:
            d = st.depth - 1
            boxes = st.citable_boxes(d)
            if not boxes:
                return None
            s, e = max(boxes)
            if st.f(e) == BOT:
                st.add(d, ('not', st.f(s)), 'NEGI', [s, e])
            else:
                g = ('imp', st.f(s), st.f(e))
                if size(g) > MAX_FORMULA_SIZE:
                    return None
                st.add(d, g, 'IMPI', [s, e])
            continue
        room = remaining - st.depth
        move = rng.choices(['local', 'box', 'ore', 'close'],
                           weights=[3.0,
                                    2.0 if room >= 2 else 0.0,
                                    3.0 if room >= 3 else 0.0,
                                    1.0 if st.depth > 0 else 0.0])[0]
        if move == 'local':
            m = local_step(st, st.depth, rng)
            if m is None:
                return None
            st.add(st.depth, m[0], m[1], m[2])
        elif move in ('box', 'ore'):
            # A macro that cannot complete rolls back rather than killing the
            # whole proof; ORE in particular fails often and cheaply.
            snap = st.clone()
            fn = macro_box if move == 'box' else macro_ore
            if not fn(st, rng, room):
                st = snap
                fails += 1
                if fails > 8:
                    return None
        else:
            d = st.depth - 1
            boxes = st.citable_boxes(d)
            if not boxes:
                return None
            s, e = max(boxes)
            if st.f(e) == BOT:
                st.add(d, ('not', st.f(s)), 'NEGI', [s, e])
            else:
                g = ('imp', st.f(s), st.f(e))
                if size(g) > MAX_FORMULA_SIZE:
                    return None
                st.add(d, g, 'IMPI', [s, e])
    if len(st.lines) != target_len or st.depth != 0:
        return None
    if st.lines[-1]['rule'] == 'AS':
        return None
    return st
