"""Formula representation, matching nd_verify's tuple encoding exactly.

('atom','P') | ('bot',) | ('not',X) | ('and',X,Y) | ('or',X,Y) | ('imp',X,Y)
"""
import random

ATOMS = ['P', 'Q', 'R', 'S']
BOT = ('bot',)
OPTOK = {'and': '&', 'or': 'v', 'imp': '>'}


def fmt(f):
    """Formula -> fully-parenthesised token string, as spec.md requires."""
    k = f[0]
    if k == 'atom':
        return f[1]
    if k == 'bot':
        return 'F'
    if k == 'not':
        return '( ~ ' + fmt(f[1]) + ' )'
    return '( ' + fmt(f[1]) + ' ' + OPTOK[k] + ' ' + fmt(f[2]) + ' )'


def size(f):
    """Number of connectives + atoms; used to keep formulas from exploding."""
    k = f[0]
    if k in ('atom', 'bot'):
        return 1
    if k == 'not':
        return 1 + size(f[1])
    return 1 + size(f[1]) + size(f[2])


def random_formula(rng, max_depth=2, p_bot=0.02):
    """Sample a random formula. Shallow by default: deep formulas make for
    long token strings without making the *proof* any more interesting."""
    if max_depth <= 0 or rng.random() < 0.45:
        if rng.random() < p_bot:
            return BOT
        return ('atom', rng.choice(ATOMS))
    k = rng.choice(['not', 'and', 'or', 'imp'])
    if k == 'not':
        return ('not', random_formula(rng, max_depth - 1, p_bot))
    return (k, random_formula(rng, max_depth - 1, p_bot),
            random_formula(rng, max_depth - 1, p_bot))


def subformulas(f, out=None):
    """All subformulas, including f itself."""
    if out is None:
        out = set()
    out.add(f)
    k = f[0]
    if k == 'not':
        subformulas(f[1], out)
    elif k in ('and', 'or', 'imp'):
        subformulas(f[1], out)
        subformulas(f[2], out)
    return out
