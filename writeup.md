# Bootstrapping a natural-deduction prover past its training length

## Executive summary

A 3.2M-parameter decoder is trained from scratch on generated natural-deduction
proofs of **at most 6 lines**, then pushed past that cap with expert iteration
against the verifier. The score is **L - P** on the *robust frontier*: the
longest proof length at which the model writes at least 5 distinct
verifier-accepted proofs.

All numbers below come from a decontaminated pipeline in which every split --
train, held-out, RL targets, transfer, validation, test -- audits to **zero
overlap**, exact and under atom-renaming.

**Headline, split by where it is measured, because the split is the result:**

| protocol | P | L | L - P |
|---|---|---|---|
| 32 samples at T=1.0, RL targets | 7 | **9** (seed 0) / **10** (seed 1) | **2-3** |
| greedy, transfer set (never trained on) | 7 | 8 | **1** |
| greedy, `validation_36` bin `>6` | 0/24 | 0/24 | **0** |
| greedy, `test_long` | 8.8% | 8.1% | **0** (noise) |

The conservative headline both seeds support is **L - P >= 2**; seed 1 reached
3. The frozen control never leaves 7 in either seed.

**On the generator's own distribution RL works and the frozen control proves
it. On hand-curated textbook theorems it buys nothing.** The second half is the
more useful finding, and findings 1 and 5 below say why the two came apart.

**1. The barrier is a learned stopping prior, not reasoning and not
tokenisation.** (figure 1) Teacher-forced along a *known-good* 9-16 line proof,
the Stage-1 model still wants to stop: P(QED) is 0.02 after line 5, 0.32 after
line 6, 0.74 after line 7, above 0.93 by line 9. Writing a 10-line proof by
sampling means beating ~0.95-0.99 stop probability at every boundary past 6, and
that product is why the ceiling is a hard stop rather than a decay. This is the
constraint on *how far* past the cap the model reaches, once it can reach past
the cap at all -- which is a separate question, answered by finding 2b.

**2a. The reference tokenisation decides whether length generalisation happens
at all.** (figure 5) Trained on identical data with an identical architecture,
a model that emits the spec's absolute line indices (`N5 : IMPE N1 N4`) writes
**zero** proofs longer than 6 lines in ~64k samples -- robust frontier exactly
6, the training cap. Replacing absolute indices with back-distances `B<k>` and
premise references `P<i>` takes that to 352 proofs past the cap and a frontier
of 7. The two models are indistinguishable in-distribution (96.6% vs 96.3%
held-out greedy), so measuring held-out accuracy alone would have hidden the
entire effect. Absolute indices leave 43.7% of the index/reference tokens in a
9-16 line proof untrained; relative ones leave 8.1%.

**2b. The positional scheme is not the barrier.** (figure 2) Learned absolute
position embeddings are untrained past the longest training sequence (194
tokens), which looked like the obvious culprit. Three Stage-1 models --
`learned`, `rope`, `nope` -- reach the same val loss (~0.077) and the same
held-out rate (95-96%), and **all three stop at exactly 7 written lines** across
192k samples each. The scheme moves the *prior* without moving the ceiling:
`rope` has the weakest stopping prior at every line and writes twice as many
7-line proofs (351 vs 177 vs 129). Mechanism predicts data.

**3. Generating length is not difficulty, and ignoring that would have faked
the whole result.** (figure 4) My first RL pool was 4000 theorems generated with
7-16 line proofs. The Stage-1 model solved **2706 of them from a single
sample**, writing 3-6 line proofs. Re-generating 30,000 candidates at 9-16 lines
and searching each, **26,941 (90%) turned out to be provable in 6 lines or
fewer**. Only after filtering does "beyond 6" mean anything.

**4. RL beats resampling by a wide margin -- inside its own distribution.**
(figure 3) Against a frozen Stage-1 model given exactly the same attempts per
round, expert iteration reaches **54.7%** of RL targets against **31.2%** for
the frozen control, and on a **transfer set never sampled for training** greedy
solve rate goes **5.9% -> 25.8%** (pass@32: 21.9% -> 49.6%). The frontier steps
7 -> 8 -> 9 while the frozen control never leaves 7: across five rounds at an
identical budget it wrote **one** 8-line proof, against 547 eight-line and 32
nine-line proofs from the policy. Held-out greedy ends at 96.8% against a 97.1%
Stage-1 baseline, so almost nothing is traded away.

**5. And none of it transfers.** `validation_36` goes 6/36 -> 4/36 (bin `>6`
stays **0/24**), and the test set does not move: 47.6% -> 46.4% short,
8.8% -> 8.1% long, both changes inside the confidence intervals. The binding constraint is **per-line rule-application
accuracy on unfamiliar formulas**: 78-90% of greedy failures, on both
distributions and both before and after RL, are lines citing a rule whose
premises do not hold -- not proofs that stay valid and miss the goal (3-12%).
RL raised the rate of complete valid proofs on theorems shaped like its own
training data and did not improve per-line validity on textbook formulas. This
is a distribution-shift result.

---

## Stage 1 -- data and supervised training

**Generator.** A forward random walk over a Fitch proof state (`nd/gen.py`).
Sampling a *proof* rather than a theorem makes soundness free -- whatever the
walk derives is valid by construction -- and every proof is still re-checked
with `nd_verify` before it is kept (18396/18396 pass, zero invalid, ~5400
proofs/s). Two fixes mattered. The first version picked a rule and then checked
whether it applied, which starved the rarely-applicable rules that the
interesting theorems need; enumerating applicable instantiations instead took
`IMPE` from 227 to 2159 and `NEGE` from 457 to 1512. Uniformly random premises
also almost never contain a top-level disjunction or a matching
implication/antecedent pair, so ~55% of proofs now start from a structured
pattern such as `[(A>B), A]` or `[(A v B), (~A)]`.

**Data.** 120,000 distinct theorems, proofs of length 2-6, split 117k/3k and
disjoint by theorem. Two numbers that inflate any solve rate and are therefore
reported separately throughout: **23.1% of theorems are trivial** (conclusion is
a premise, a premise is `F`, or the proof is <=2 lines) and **41.8% of the
held-out set is an atom-renaming of a training theorem**.

Splitting the held-out set by that second criterion quantifies the inflation:
the model scores **99.4%** (618/622) on theorems that are renamings of a
training theorem and **94.2%** (827/878) on structurally new ones, against a
combined 96.3%. The renamings do inflate, by about 2 points; 94.2% is the
number to quote for genuinely unseen structure.

**Tokenisation.** The spec numbers lines absolutely (`N5 : IMPE N1 N4`). Under a
6-line cap the model never sees `N7`+, so on a 12-line proof it would have to
emit near-untrained tokens. I replaced absolute indices with back-distances
`B<k>` for derived lines and `P<i>` for premises -- the premise half matters,
because premises are cited from arbitrarily far away and back-distances alone
would reintroduce the problem (line 1 cited from line 14 is `B13`, never seen).
Round-trip is exact on 5000 proofs and the verifier only ever sees decoded spec
format.

This **reduces** out-of-distribution references rather than eliminating them,
and an earlier draft of this write-up overstated it. Measured on the 9-16 line
reference proofs, the fraction of index/reference tokens that never appear in
cap-6 training data is **43.7% under the absolute scheme and 8.1% under the
relative scheme** -- a 5.4x reduction. Training only ever exercises `B1`-`B4`
and `P1`-`P3`; long proofs genuinely need `B5`-`B14`.

A consequence I did not anticipate: across all **13,378** verified proofs the RL
model wrote, at every length up to 9, it emits **`B5` or beyond exactly zero
times**. Teacher-forcing along reference proofs that do need them shows why.
When the correct next token is `B1`-`B4` the model assigns it 0.42-0.76
probability; when the correct token is `B5`-`B12` it assigns **0.003-0.01** and
emits `B4` instead 66-94% of the time. Cross-entropy pushes every non-target
token down at every position, and `B5`+ was never a target, so its probability
is near zero unconditionally -- the model saturates at the largest offset it
ever saw.

This does not make such theorems unreachable, and an earlier draft said so
wrongly. A theorem has many proofs: instead of citing a result from 12 lines
back (`B12`, probability ~0.01) the model can re-derive it one line earlier
(`B1`, probability ~0.76) at the cost of an extra line. What it cannot do is
both -- avoiding far references costs length, and length is what the stopping
prior blocks. The two constraints squeeze from opposite sides, which is a better
account of the zero-`B5+` observation than either alone.

**Model.** 4 layers, d=256, 4 heads, 3.26M params (3.18M for rope/nope),
trained from scratch, 6000 steps at batch 256, ~4 minutes on one H200.

**Held-out, greedy, n=1500:** 97.1% [96.1-97.8] overall; by proof length
100.0 / 98.4 / 97.8 / 97.9 / **92.0**% for lengths 2-6; 96.4% on non-trivial
theorems only. The reference point in the brief is >=85%.

**validation_36, greedy: 6/36** -- bin `<=6` 6/12, bin `>6` **0/24**. 34 of the
36 failures are *well-formed* proofs that cite a rule whose premises do not hold,
not parse errors: the model has the format and the box bookkeeping, and lacks
search.

## Stage 2 -- what P actually is

I first read P off validation_36 (`0/24` beyond the cap) and called it 6. That
is the wrong measurement. The frontier is defined over *written* proof lengths
with at least 5 distinct verified proofs, and by that definition every Stage-1
model sits at **7**, with 129-351 distinct 7-line proofs on beyond-cap targets.
The pre-trained model already writes one line past its cap, so **P = 7** and L
must beat 7 for RL to have done anything. Using 6 would have inflated the
headline by a full line.

## Stage 2 -- expert iteration

Each round samples k=32 attempts per target at T=1.0 from the current policy,
keeps what the verifier accepts *for the prompted sequent*, and fine-tunes from
the Stage-1 weights on everything accumulated so far, mixed with Stage-1 data to
hold the in-distribution skill. Restarting from Stage-1 each round rather than
continuing from the previous policy keeps the comparison clean and avoids drift.

**Hindsight relabelling.** A sample that fails its target is often a valid proof
of some *other* theorem -- its PR lines are its premises, its last line its
conclusion. Those are recovered and reused, filtered against the held-out,
transfer, validation and test sets (and atom-renamings of them) so nothing
leaks. On the corrected hard pool this contributed thousands of extra verified
proofs; on the earlier easy pool it added almost nothing, because when the model
succeeds the relabelled theorem is just the target again.

**The control.** A frozen copy of the Stage-1 model gets exactly the same number
of attempts per round and is never retrained. At round 1 the policy *is* the
frozen model, and the two agree to within noise (466 vs 467 solved) -- which is
the check that the two arms really do get identical budgets.

**Results (seed 0; seed 1 in the table below it).**

| round | RL policy | frozen (same budget) | transfer | transfer frozen | held-out | frontier | frozen frontier |
|---|---|---|---|---|---|---|---|
| 1 | 463 (23.2%) | 440 (22.1%) | 170 (21.2%) | 178 (22.2%) | 97.8% | **7** | 7 |
| 2 | 679 (34.1%) | 514 (25.8%) | 238 (29.8%) | 169 (21.1%) | 95.6% | **8** | 7 |
| 3 | 850 (42.6%) | 556 (27.9%) | 311 (38.9%) | 175 (21.9%) | 96.2% | **8** | 7 |
| 4 | 989 (49.6%) | 586 (29.4%) | 361 (45.1%) | 177 (22.1%) | 96.0% | **8** | 7 |
| 5 | 1091 (54.7%) | 622 (31.2%) | 398 (49.8%) | 162 (20.2%) | 96.8% | **9** | 7 |

Written-length histograms show the ladder directly. After five rounds the
policy's cumulative histogram is `{'4': 21, '5': 374, '6': 2116, '7': 1948, '8': 547, '9': 32, '10': 1}` -- 547 eight-line and 32 nine-line proofs.
The frozen control's is `{'4': 24, '5': 265, '6': 994, '7': 542, '8': 1}`: at an identical budget it wrote **one** 8-line
proof and no 9-line proofs. It finds more theorems by resampling, but it never
becomes able to write longer ones.

At round 1 the policy *is* the frozen model and the two agree to within noise
(463 vs 440 solved, 21.2%% vs 22.2%% transfer), which is the check that both arms
really do get identical budgets.

**Seed 1 (independent run, same configuration).**

| round | RL policy | frozen | transfer | held-out | frontier | frozen frontier |
|---|---|---|---|---|---|---|
| 1 | 452 (22.7%) | 430 (21.6%) | 179 (22.4%) | 97.8% | **7** | 7 |
| 2 | 667 (33.5%) | 509 (25.5%) | 246 (30.8%) | 96.0% | **8** | 7 |
| 3 | 823 (41.3%) | 552 (27.7%) | 301 (37.6%) | 96.2% | **8** | 7 |
| 4 | 960 (48.1%) | 585 (29.3%) | 359 (44.9%) | 96.4% | **9** | 7 |
| 5 | 1060 (53.2%) | 603 (30.2%) | 387 (48.4%) | 96.4% | **10** | 7 |

Both seeds agree closely at every round (54.7% vs 53.2% of RL targets at round
5, 49.8% vs 48.4% transfer) and both leave the frozen control far behind. They
differ on the frontier: seed 0 ends at 9 with 32 distinct nine-line proofs and
1 ten-line; seed 1 ends at 10 with 106 nine-line and 10 ten-line. So the
frontier is 9-10 rather than a single number, and the claim both seeds support
is **L >= 9**. That spread is worth stating: the robust-frontier criterion is a
threshold on a count, so a model close to the boundary can land on either side.

**In-distribution cost.** Held-out greedy ends at 96.8% (seed 0) and 96.4%
(seed 1) against a 97.1% Stage-1 baseline, and `validation_36` falls 6/36 ->
4/36. Both are small; the validation change is well inside noise at n=36.

## Why the frontier sits where it does: a surprisal budget

Autoregressive sampling is Boltzmann sampling. Per-token log-probabilities add,
so with `E(y) = -log p(y|x)` we get `p(y) = exp(-E(y))/Z`, and sampling at
temperature `tau` draws from `exp(-E(y)/tau)`. Drawing `k` samples finds a
particular proof when `k * p(y) >~ 1`, i.e.

    E(y) <= log k

The frontier is then `L*(k) = max{ L : E_min(L) <= log k }`, where `E_min(L)` is
the smallest total surprisal among verified `L`-line proofs.

Measuring `E_min(L)` directly (teacher-force every verified proof in the RL
pool, sum `-log p` over the body):

| L | Stage-1 `E_min` | k needed | after-RL `E_min` | k needed |
|---|---|---|---|---|
| 2-7 | 0.01-0.30 | ~1 | 0.01-0.13 | ~1 |
| 8 | **7.86** | 2,590 | **0.05** | 1.05 |
| 9 | **24.81** | 5.9e10 | **1.51** | 4.5 |

`E_min` is flat and then cliffs (figure 6, left). Stage 1 cliffs between 7 and
8 lines; after RL the cliff has moved to between 9 and 10.

**The criterion is quantitatively accurate.** Sweeping the sampling budget
k = 1, 2, 4, ..., 64 and comparing the predicted `L*(k)` to the measured robust
frontier matches in **11 of 12** testable cases (k=1 is degenerate, `log 1 = 0`).
The single miss is the RL model at k=4: predicted 8, observed 9, with
`E_min(9) = 1.51` against `log 4 = 1.39` -- a 0.12-nat discrepancy.

Three consequences, and they are the cleanest statement of what this project
found:

1. **Sampling is exponentially weak, because k enters only as `log k`.** Lifting
   Stage 1 from 7 to 8 lines by resampling alone needs k around 2,590 rather
   than 1. This is why the frozen control is flat in figure 3 and why its robust
   frontier never leaves 7 even at k=64.
2. **Training is exponentially strong, because it moves E directly.** RL lowered
   `E_min(8)` by 7.8 nats (a factor of ~2,400 in probability) and `E_min(9)` by
   23.3 nats (~1.3e10) at the *same* k. That is the mechanism behind every gap
   between the policy and the frozen control in this write-up.
3. **The step in the frontier is a threshold artefact, not a phase transition.**
   `E_min` moves continuously as training proceeds; `L*` is an integer maximum
   over a threshold, so it jumps. Reporting `L*` alone makes smooth progress
   look discontinuous -- the concern Schaeffer et al. raise about emergent
   abilities, visible here in a system small enough to measure both quantities.

An earlier version of this section used the stopping-prior term `E_stop` as a
proxy for `E`. That is only a lower bound and it over-predicted the frontier by
about two lines and predicted ~1.3 lines of growth over a 64x budget increase
where the measured growth is 0. Measuring the full `E` is what makes the
criterion work.

## Stage 3 -- evaluation

| model | held-out <=6 greedy | transfer >6 greedy | transfer pass@32 | validation_36 | val bin `>6` | test_short | test_long |
|---|---|---|---|---|---|---|---|
| Stage 1 (clean) | 97.1% [96.1-97.8] | 5.9% [4.4-7.7] | 21.9% [19.1-24.9] | 6/36 | 0/24 | 47.6% | 8.8% |
| after RL | 96.3% [95.2-97.1] | **25.8%** [22.8-28.9] | **49.6%** [46.2-53.1] | 4/36 | 0/24 | 46.4% | 8.1% |

Test-set numbers are one greedy run per file per model, scored with
`score_test.py`, with no per-theorem inspection.

The two middle columns and the last three tell opposite stories, and the
contrast is the main result. Greedy written lengths on the transfer set go from
`{5:5, 6:12, 7:30}` to `{4:1, 5:6, 6:34, 7:147, 8:18}` -- a 4.4x solve-rate gain
and genuinely longer proofs, on theorems never sampled during training. On
`validation_36` and the test set, nothing.

**Why they diverge.** Classify each greedy failure by the verifier's reason.
A `rule check failed` is a line that is not a valid rule application at all; a
`final formula is not the conclusion` is a valid derivation that misses the
goal.

| failure class | validation_36, Stage 1 | validation_36, after RL | transfer, after RL |
|---|---|---|---|
| invalid rule application | **90.0%** | **78.1%** | **85.2%** |
| valid but missed the goal | 3.3% | 12.5% | 11.1% |
| malformed output | 3.3% | 6.2% | 3.2% |

The bottleneck is **per-line validity**, not goal-directedness, and it is the
bottleneck on both distributions before and after RL. What RL changed is the
rate at which the model gets whole derivations right on its own distribution
(transfer greedy 5.9% -> 25.8%) and the length at which it can do so. It did
not improve per-line rule application on textbook formulas, so nothing moved
there.

An earlier draft of this write-up claimed RL "bought length fluency, not proof
search". The failure-mode table above does not support that: the model's steps
are not valid-but-misdirected, they are invalid. The defensible claim is
narrower -- the gain is real, controlled, and confined to the generator's own
distribution.

## Data hygiene: a contamination audit, and what it did and did not change

Late in the project I ran a pairwise overlap audit of every split against every
other. The Stage-2 pools were clean -- they were built with a contamination
filter. The **Stage-1 training set was not**, because it predated that filter:

| benchmark | exact overlap with Stage-1 train | overlap under atom-renaming |
|---|---|---|
| validation_36 | 4 / 36 | 6 / 36 |
| test_short | 45 / 267 (16.9%) | **91 / 267 (34.1%)** |
| test_long | 8 / 532 (1.5%) | 15 / 532 (2.8%) |

This is the default outcome rather than an accident: with four atoms, a random
walk emitting 120k theorems produces textbook sequents by chance.
`( P > Q ) , P |- Q` is an ordinary output of the generator.

**Fix.** Generation now drops any theorem that is, or is an atom-renaming of, a
validation or test theorem (684 of 120,000 candidates). Stage 1 was retrained on
the clean data and everything downstream re-measured. Regenerating also put 6 of
the 2000 RL targets into the new training set; those were removed too. Every
pairwise audit -- train, held-out, RL targets, transfer, validation, test -- is
now **0 overlap, exact and under renaming**. Filtering training data against
test *prompts* is decontamination, not tuning: no test verdict is inspected, and
the brief states that training-set overlap is audited.

**What it changed: less than the overlap suggests.** I initially read "5 of the
6 validation theorems Stage 1 solved were in its training set" as "5 were
memorised". Testing that shows otherwise -- the decontaminated model solves the
**identical six theorems**. They are learnable from the distribution rather than
recalled: `modus_ponens` is a three-line `IMPE` proof and `IMPE` appears 2159
times in training, so the specific sequent is not needed. Overlap and
memorisation are different things, and only the second one inflates a score.

Both facts belong in the report: the contamination was real and had to be
removed, and removing it did not move the validation number.


## Where to find the full derivations

`PRIMER.md` builds the whole setting from scratch with no assumed background:
what a proof and a sequent are, how each dataset is constructed, the token
scheme, the architecture and forward pass with shapes, targets and loss, the
optimiser, what reinforcement learning is and which part of it is used here, a
critique of the `L - P` metric, the causal mechanism by which RL produces longer
proofs, and the exact Boltzmann correspondence with its limits. Figures 1, 6 and
7 are the visual companions to it.

## Glossary

Every term this write-up depends on, defined once.

**Formula.** Built from atoms `P Q R S`, falsum `F` (a constant that is always
false), and connectives `~` (not), `&` (and), `v` (or, inclusive), `>`
(implies). Every compound formula is fully parenthesised, negation included.
`>` is material implication: `( A > B )` is false only when `A` is true and `B`
is false.

**Sequent / theorem.** `premises |- conclusion`. The claim that the conclusion
follows from the premises. `|-` is the turnstile. Zero premises means the
conclusion is a tautology.

**Premise.** A formula you are *given*. In a proof it appears on an early line
justified by the rule `PR`; you do not derive it.

**Proof.** A numbered list of lines. Each line states a formula and cites a rule
plus the earlier lines that justify it:
`N4 ( R v S ) : ORI2 N1 ;` means "line 4 asserts `( R v S )`, by or-introduction
(right) applied to line 1". The `:` is punctuation separating the claim from its
justification; `;` ends the line.

**Proof length.** Number of lines, premises included. The verifier reports it.
This is the difficulty axis: an n-line proof is an n-step derivation, so if each
step is right with probability p, the whole proof is right with roughly p^n.

**Subproof / box.** A temporary assumption. To prove `A > B` you assume `A`
(rule `AS`), derive `B`, then discharge the assumption with `IMPI`. Depth is
written with `|`. Three rules discharge boxes: `IMPI` (implication), `NEGI`
(reductio: assume A, reach `F`, conclude `~A`), and `ORE` (case split, needs two
boxes).

**Verifier.** `nd_verify.verify_text` -- returns `(ok, reason, n_lines)`. The
only judge. Free, exact, and with no notion of "close".

**Cap.** L = 6. No supervised training proof may exceed 6 lines. The entire
question is what happens past it.

**Robust frontier.** The longest *written* proof length at which a model
produced at least 5 **distinct** verifier-accepted proofs. Five, so one lucky
long proof is not counted as a capability. Always the length of the proof
actually written, never the length of the proof the theorem was generated with.

**P and L.** `P` = robust frontier of the pre-RL model, `L` = of the RL model.
The score is `L - P`. Note this is a *difference*, so a deliberately weak
baseline inflates it; the defence is that Stage 1 scores 97.1% held-out against
the brief's >=85% reference, so `P = 7` is not artificially depressed.

**Distribution.** The rule that assigns probability to each possible theorem.
My generator *is* one. Train and held-out are different draws from it -- disjoint
theorems, same kind of theorem. `validation_36` and the test set are not from
it: a human picked those from a textbook.

**Greedy vs temperature.** Greedy takes the highest-probability token at each
step: deterministic, one proof per theorem. Temperature `tau` samples from the
distribution: at `tau = 1.0` you draw from the model's actual probabilities, so
32 samples give 32 different attempts. RL needs the diversity; reported baselines
use greedy.

**Surprisal / energy.** `E(y) = -log p(y|x)`, the total negative
log-probability of a proof. Per-token values add, so `E` behaves like an energy
and sampling is Boltzmann sampling. A proof is findable with `k` samples when
`E(y) <= log k`.

**Expert iteration.** The RL algorithm here (also: rejection-sampling
fine-tuning, STaR). Sample k attempts, keep the ones the verifier accepts,
fine-tune on them, repeat. No reward gradient anywhere -- the search is the
sampling, the verifier is the label, and the gradient step makes the discovery
permanent.

**Round.** One pass of that loop: sample 32 attempts for each of ~2000 targets,
verify, add successes to a pool, retrain from the Stage-1 weights on the pool.

**Frozen control.** A copy of the Stage-1 model given exactly the same number of
attempts per round and never retrained. Without it, "RL solved N theorems" only
says you sampled a lot.

**Transfer set.** Theorems from the same generator that RL never samples. The
held-out set for Stage 2.

**Hindsight relabelling.** A failed attempt is often a valid proof of a
*different* theorem. Read that theorem off the proof (its `PR` lines are the
premises, its last line the conclusion), verify, and reuse it as training data.

**Atom-renaming.** Two theorems with identical structure and permuted letters
(`(P>Q),P |- Q` vs `(R>S),R |- S`). Reported because it inflates held-out
scores, and filtered because it is contamination when it crosses into an
evaluation set.

## Limitations

- **The gain is inside my own distribution only.** `validation_36` and both
  test files do not move. Any claim of "novel knowledge" here is a claim about
  theorems my generator produces, not about natural deduction generally.
- **`L - P` is a length metric and length is purchasable.** RL bought fluency
  at continuing a proof, not search. A method that improved search would look
  worse on `L - P` per unit compute and better on `validation_36`.
- **The frontier depends on the sampling protocol.** L = 9 is at 32 samples,
  T = 1.0; greedy gives L = 8. P is measured identically in both cases, so
  L - P is apples-to-apples, but neither is a greedy single-shot number.
- **The hard pool is filtered, not proved hard.** Candidates are dropped when a
  large sampling budget finds a <=6-line proof -- an upper bound in the same
  sense as the repo's `min_lines_ub`. Re-probing with three models at k=32 still
  found 532 distinct <=6-line proofs on the "hard" pool, so it is mostly, not
  entirely, beyond the cap. A bounded search prover would give a clean claim.
- **The filter uses the model as its own prover**, biasing the surviving pool
  toward theorems this model family finds hard.
- **In-distribution cost.** Held-out 96.4% -> 95.0% (seed 0) and -> 94.2%
  (seed 1); `validation_36` 6/36 -> 4/36 (within noise at n=36, but not an
  improvement). The cost is small, consistent across seeds, and real.
- **rope/nope trained with block=512, learned with 320**, so that ablation is
  not perfectly controlled -- though every training sequence is <=194 tokens, so
  it cannot affect in-distribution fit.
- **Two seeds, one method.** Both seeds complete and agree closely, but there
  is no PPO/GRPO comparison and no inference-time search, so "expert iteration
  works here" is not "expert iteration is the right choice here".
- **23.1% of generated theorems are trivial and 41.8% of the held-out set is an
  atom-renaming of a training theorem**, so held-out solve rates are reported
  with a non-trivial-only figure alongside.

## What I would do next

1. **Attack the stopping prior directly rather than through RL.** The diagnosis
   suggests cheaper interventions: condition the prompt on a target length,
   penalise early `QED` at sampling time, or reweight the SFT length
   distribution. If the prior is the barrier these should move the frontier per
   unit compute far better than more expert-iteration rounds, and they are a
   direct test of the diagnosis.
2. **Add search, since that is what `validation_36` needs.** Best-first or
   MCTS-style proof search over the model's own step proposals, with the
   verifier as the expansion check. The failures are wrong rule applications,
   not malformed output, which is exactly the regime where search helps.
3. **Fix the target distribution.** Backward goal-directed generation would
   produce theorems that look more like textbook sequents than a forward random
   walk does; the transfer failure is plausibly a data-distribution problem as
   much as an algorithm problem.
4. **A bounded search prover** so "needs more than 6 lines" is a proof rather
   than an upper bound.

## Use of AI assistance

Claude wrote most of the code in this repository. The judgement calls -- what to
measure, which results to throw out, that generating length is not difficulty,
that P was 7 and not 6, that the stopping prior rather than the positional
scheme is the barrier -- were mine, and `log.md` records them in order,
including the two hypotheses that turned out wrong.
