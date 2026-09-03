"""Incremental Fitch proof state.

Mirrors nd_verify's box bookkeeping so the generator can ask "what may I write
next?" instead of guessing and being rejected. Every proof this produces is
still run through the real verifier before it is kept -- this class is a
convenience, never the authority.
"""
import copy

from .formula import BOT


class ProofState:
    def __init__(self, premises):
        self.lines = []        # dicts: idx, depth, formula, rule, refs
        self.stack = []        # ids of currently open boxes
        self.nbox = 0
        self.ctx = {}          # line idx -> tuple of open box ids after it
        self.b_start = {}      # box id -> idx of its AS line
        self.b_depth = {}      # box id -> depth of its AS line
        self.b_last = {}       # box id -> idx of last line seen inside it
        self.b_parent = {}     # box id -> ctx enclosing the box
        self.premises = list(premises)
        for p in premises:
            self._push(0, p, 'PR', [])

    # ---------- construction ----------

    def _push(self, depth, formula, rule, refs):
        idx = len(self.lines) + 1
        if rule == 'AS':
            del self.stack[depth - 1:]
            self.nbox += 1
            b = self.nbox
            self.b_parent[b] = tuple(self.stack)
            self.stack.append(b)
            self.b_start[b] = idx
            self.b_depth[b] = depth
        else:
            del self.stack[depth:]
        for b in self.stack:
            self.b_last[b] = idx
        self.ctx[idx] = tuple(self.stack)
        self.lines.append({'idx': idx, 'depth': depth, 'formula': formula,
                           'rule': rule, 'refs': list(refs)})
        return idx

    def add(self, depth, formula, rule, refs=()):
        return self._push(depth, formula, rule, refs)

    def clone(self):
        return copy.deepcopy(self)

    # ---------- queries ----------

    @property
    def depth(self):
        return len(self.stack)

    def f(self, idx):
        return self.lines[idx - 1]['formula']

    def citable_lines(self, at_depth):
        """Line indices citable from a new non-AS line written at `at_depth`."""
        newctx = tuple(self.stack[:at_depth])
        return [ln['idx'] for ln in self.lines
                if self.ctx[ln['idx']] == newctx[:len(self.ctx[ln['idx']])]]

    def citable_boxes(self, at_depth):
        """(start, end) pairs of boxes dischargeable from a line at `at_depth`."""
        newctx = tuple(self.stack[:at_depth])
        out = []
        for b in range(1, self.nbox + 1):
            if b in newctx:
                continue                       # cannot discharge a box you are in
            last = self.b_last.get(b)
            if last is None:
                continue
            if self.lines[last - 1]['depth'] != self.b_depth[b]:
                continue                       # box must end at its own level
            par = self.b_parent[b]
            if par != newctx[:len(par)]:
                continue
            out.append((self.b_start[b], last))
        return out

    # ---------- rendering ----------

    def body_tokens(self):
        toks = []
        for ln in self.lines:
            toks.append('N%d' % ln['idx'])
            toks.extend(['|'] * ln['depth'])
            from .formula import fmt
            toks.extend(fmt(ln['formula']).split())
            toks.append(':')
            toks.append(ln['rule'])
            toks.extend('N%d' % r for r in ln['refs'])
            toks.append(';')
        toks.append('QED')
        return toks

    def prompt(self):
        from .formula import fmt
        prem = ' , '.join(fmt(p) for p in self.premises)
        concl = fmt(self.lines[-1]['formula'])
        mid = (prem + ' ') if prem else ''
        return 'THM ' + mid + 'SEQ ' + concl + ' PRF'

    def text(self):
        return self.prompt() + ' ' + ' '.join(self.body_tokens())
