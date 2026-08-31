# The logic, the proof format, and the verifier

Everything in the exam is judged by `nd_verify/verify.py`. This page is the
human-readable version of what that file checks. Read both.

## 1. Formulas

- Atoms: `P`, `Q`, `R`, `S`. Falsum: `F`.
- Connectives: `~` (not), `&` (and), `v` (or), `>` (implies).
- **Every compound formula is fully parenthesised**, including negation:
  `( ~ P )`, `( P & Q )`, `( ( P > Q ) > P )`. Atoms and `F` are bare.
- Tokens are whitespace-separated. There is no `<->`; there are no other atoms.

## 2. Sequents and prompts

A theorem (sequent) is `premises |- conclusion`. The model is prompted with

```
THM <premise-1> , <premise-2> , ... SEQ <conclusion> PRF
```

and must emit the proof body followed by `QED`. Zero premises is written
`THM SEQ <conclusion> PRF`.

## 3. Proof lines

One line is

```
N<i> ['|' × depth] <formula> : <RULE> [N<ref> ...] ;
```

- `N<i>`: line index. Indices must be consecutive and ascending (the verifier
  accepts any starting value ≥ 1; indices up to `N64` are plenty for this exam).
- `|` repeated `depth` times: how many subproof boxes the line sits inside.
- `<RULE>` and its cited lines/boxes, then `;`.

Fitch box semantics (as implemented in `verify.py`):

- `PR` (premise) lines come first, at depth 0, and must reproduce the declared
  premises **exactly and in order**.
- `AS` (assumption) at depth `d` (1 ≤ d ≤ current depth + 1) closes any boxes
  deeper than `d−1` and opens a new box with that formula as hypothesis.
- Any other line at depth `d` closes boxes deeper than `d`.
- A line `j` can be cited from line `i` iff `j < i` and every box open at `j`
  is still open at `i`.
- A box is cited as `N<start> N<end>`: `start` is its `AS` line, `end` is the
  last line inside the box (at the box's own depth); the box must be closed
  at the citing line and its parent context must still be open.
- The last line must be at depth 0, must be the conclusion, and must not be
  an `AS` (or a `PR` unless the conclusion is literally a premise).

## 4. Rules (name — refs — what is checked)

| rule | refs | derives `G` from |
|---|---|---|
| `PR` | – | declared premise |
| `AS` | – | opens a box with hypothesis `G` |
| `R` | `Nj` | `G` = formula at `j` (reiteration) |
| `ANDI` | `Na Nb` | `G = ( A & B )` |
| `ANDE1` | `Na` | `a: ( G & B )` |
| `ANDE2` | `Na` | `a: ( A & G )` |
| `IMPE` | `Na Nb` | `a: ( B > G )`, `b: B` (modus ponens; **implication first**) |
| `IMPI` | `Ns Ne` | box `s..e` with hypothesis `A` ending in `B`; `G = ( A > B )` |
| `ORI1` | `Na` | `G = ( A v X )` with `a: A` |
| `ORI2` | `Na` | `G = ( X v A )` with `a: A` |
| `ORE` | `Nj Ns1 Ne1 Ns2 Ne2` | `j: ( A v B )`; box `s1..e1` assumes `A`, box `s2..e2` assumes `B`, both end in `G` |
| `NEGE` | `Na Nb` | `a: A`, `b: ( ~ A )`; `G = F` (**positive first**) |
| `NEGI` | `Ns Ne` | box `s..e` assumes `A`, ends in `F`; `G = ( ~ A )` |
| `BOTE` | `Na` | `a: F`; `G` anything (ex falso) |
| `DN` | `Na` | `a: ( ~ ( ~ G ) )` (classical double-negation elimination) |

Discharging rules (`IMPI`, `NEGI`, `ORE`) are the ones that need subproof
bookkeeping; the rest are local. A one-line box is legal: `N3 | P : AS ;`
then `N4 ( P > P ) : IMPI N3 N3 ;`.

## 5. Worked example

Disjunctive syllogism, `( P v Q ) , ( ~ P ) |- Q`, as a single token string:

```
THM ( P v Q ) , ( ~ P ) SEQ Q PRF
N1 ( P v Q ) : PR ;
N2 ( ~ P ) : PR ;
N3 | P : AS ;
N4 | F : NEGE N3 N2 ;
N5 | Q : BOTE N4 ;
N6 | Q : AS ;
N7 Q : ORE N1 N3 N5 N6 N6 ;
QED
```

(Line breaks are for reading only — the verifier splits on whitespace.)
More examples of every length 2–8 are in `examples/proofs_2_to_8.txt`.

## 6. Length

The **length of a proof** is its number of lines (the number of `;` in the
body), premise lines included. The verifier returns it as the third element
of `verify_text(...)`. The exam's cap `L = 6` is a cap on this number.

A *theorem's* length is not well defined: the proof you generated it with is
an upper bound on its shortest proof. Keep that in mind whenever you say a
theorem is "longer than 6".

## 7. Using the verifier

```python
from nd_verify import verify_text
ok, reason, n_lines = verify_text(full_string)   # 'THM ... PRF ... QED'
```

`reason` is `'ok'` or a short diagnostic (`parse: ...`, `bad box cite (line 5)`,
`rule check failed: IMPE (line 4)`, `final formula is not the conclusion`, …).
Command line, over a jsonl of attempts:

```
python verify_cli.py attempts.jsonl            # records: {"prompt":..., "proof":...} or {"text":...}
python verify_cli.py attempts.jsonl --out judged.jsonl --reasons
```

**Tokenisation is yours to design.** The surface alphabet is small — `P Q R S F
( ) ~ & v >`, the structural symbols `THM , SEQ PRF QED | : ;`, the 15 rule
names, and line indices `N1, N2, …` — but how you map it to model tokens
(one symbol per token, split digits, merged symbols, relative references, …)
is a modelling decision; say what you chose and why. The verifier only sees
the whitespace-separated text your model's output decodes to.
