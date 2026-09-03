# Log

Dated, in order, including the things that did not work. Honest beats tidy.

All work on one machine: 4x H200 (shared with other users), CUDA driver 12.8,
torch 2.11+cu128 in `.venv`. The box has 192 cores and torch thread-thrashes on
it, so everything runs with `OMP_NUM_THREADS=8`.

AI assistance: Claude wrote most of the code in this repo. The judgement calls
recorded below -- what to measure, what to believe, which results to throw out
-- are described in my own words in `writeup.md`.

## 2026-09-01

### Stage 1: generator

- Built a forward random-walk proof generator over a Fitch state machine
  (`nd/gen.py`, `nd/proofstate.py`). Sampling a *proof* rather than a theorem
  makes soundness free: whatever the walk derives is valid by construction.
  Every proof is still re-checked with `nd_verify` before it is kept.
- **First version had a badly skewed rule distribution**: over 20k proofs,
  `ORE` appeared 35 times, `IMPE` 227, `BOTE` 64. Those are exactly the rules
  the validation set needs (disjunctive syllogism is `ORE`+`NEGE`+`BOTE`).
  Two causes, both fixed:
  1. `local_step` picked a rule at random and *then* checked whether it
     applied, so rarely-applicable rules starved. Rewrote it to enumerate all
     applicable instantiations and sample among those. `IMPE` 227 -> 2159,
     `NEGE` 457 -> 1512, `DN` 146 -> 505.
  2. Uniformly random premises almost never contain a top-level disjunction or
     a matching implication/antecedent pair, so the walk had nothing to work
     with. Added structured premise patterns (`[(A>B), A]`, `[(A v B), (~A)]`,
     ...) for ~55% of proofs.
- A failed macro used to discard the whole proof; made macros roll back and
  retry instead. Yield 82% -> 92%, and `ORE` 35 -> 188.
- Final generator: 18396/18396 sampled proofs pass the real verifier, zero
  invalid, ~5400 proofs/sec.

### Stage 1: data and tokenisation

- 120k distinct theorems, lengths 2-6 (cap respected), theorem-disjoint split
  117k train / 3k held-out.
- Measured two things that inflate any solve rate and are reported separately
  everywhere: **23.1% of theorems are trivial** (conclusion is a premise, a
  premise is `F`, or the proof is <=2 lines) and **41.8% of the held-out set is
  an atom-renaming of a training theorem**.
- **Tokenisation.** The spec numbers lines absolutely (`N5 : IMPE N1 N4`).
  Under a 6-line cap the model never sees `N7`+, so at test time on a 12-line
  proof it must emit near-untrained tokens -- a tokenisation artefact that
  would pin the frontier at 6 for reasons unrelated to reasoning. Replaced
  absolute indices with back-distances `B<k>` for derived lines and `P<i>` for
  premises. The `P<i>` half matters: premises are cited from arbitrarily far
  away, so back-distances alone would reintroduce the same problem (citing
  line 1 from line 14 is `B13`, never seen under the cap). Exact round-trip
  verified on 5000 proofs; the verifier only ever sees decoded spec format.

### Stage 1: model and baseline

- 4 layers, d=256, 4 heads, 3.26M params, from scratch. 6000 steps at batch
  256, 253s, val loss 0.077.
- Held-out greedy **96.0%** (n=1500); by length 100/98.8/96.9/94.4/**91.0**%
  for 2/3/4/5/6; non-trivial-only 94.9%. Reference point in the README is
  >=85%, so Stage 1 is healthy.
- validation_36 greedy **6/36**: bin `<=6` 6/12, bin `>6` **0/24**. (I first read this as
  P = 6; see the correction below -- the robust frontier makes P = 7.)
- Failure modes: 34/36 validation failures are parseable proofs that cite a
  rule whose premises do not hold (`rule check failed: ANDI`, `IMPE`, ...),
  not parse errors. The model has the format and the box bookkeeping; its
  per-line rule application is what fails. (An earlier version of this line
  said "it does not have search" -- see the correction below, the failures are
  invalid steps, not valid steps aimed at the wrong goal.)

### Bugs found (each one would have silently corrupted a result)

- **Left-padding destroyed batched generation.** 0% solve rate at val loss
  0.34. The model uses absolute learned position embeddings, so left-padding a
  short prompt shifted every position. Fixed by masking padding out of both
  attention and the position count; batched output now matches single-prompt
  decoding exactly (0/24 mismatches). This looked exactly like a model failure.
- **Vocabulary collision**: `R` is both the atom `R` and the reiteration rule
  name, so one index was unreachable in the reverse map and sampling it
  crashed the decoder. Deduped and retrained.
- Zero-premise prompts (`THM SEQ X PRF`) broke prompt parsing.
- **KV cache was quadratic.** Growing it with `torch.cat` per token recopied
  the whole cache every step: 57 GB peak. Preallocated buffers written in
  place -> 7.0 GB, still bit-exact against brute-force recompute.
- **cuDNN fused attention was both broken and slow.** `sdpa` on the cuDNN
  backend crashed at batch 2048 (`mha_graph.execute ... got false`) partway
  through long runs, and where it did not crash it was ~60x slower than the
  alternatives. `torch.backends.cuda.enable_cudnn_sdp(False)` took 2048 greedy
  samples from 49s to 0.8s. Verified the speedup is real by checking the
  held-out rate is unchanged (95.6%). Two earlier "impossibly fast" benchmark
  readings were this backend erroring out, not working.

### Stage 2 setup: the target pool was wrong (main dead end so far)

- First RL pool: 4000 theorems generated with 7-16 line proofs, assumed to be
  "beyond the cap" because of their generating length.
- **The Stage-1 model solved 2706/4000 of them from a single sample**, writing
  proofs of 3-6 lines; written-length histogram
  `{3:314, 4:602, 5:785, 6:942, 7:20}`.
- Cause: a forward random walk inserts steps that do not make the *theorem*
  harder (`ORI` against a fresh formula, `ANDI`/`ANDE` round trips,
  reiterations). Generating length is an upper bound on the shortest proof and
  I had been treating it as difficulty -- the exact trap the README warns
  about. Running RL on this pool would have produced a large meaningless
  number.
- Hindsight relabelling on this pool also added nearly nothing (686 relabelled
  vs 676 on-target, only 3 proofs longer than 6 lines) -- when the model
  succeeds, the relabelled theorem is just the target again.
- Fix: filter by search rather than trusting generating length. Probe every
  candidate with a large sampling budget from the Stage-1 model and drop it if
  any sample is a verified proof of <=6 lines. Survivors are labelled
  `no <=6-line proof found in N samples` -- an upper bound, the same status as
  the repo's own `min_lines_ub`, not a proof of hardness.
- Second correction: a binary easy/hard split is also wrong, because if every
  target is beyond the Stage-1 model then round 1 collects no successes and
  expert iteration has nothing to learn from. Bucketed by `shortest_found`
  instead, giving curriculum rungs (7, 8, 9, ...) plus an unsolved stretch set.

### Known limitation of the current pool filter

The probe uses the Stage-1 model as its prover, so the surviving pool is
biased toward "theorems this model cannot do" -- somewhat circular. A bounded
search prover would give a cleaner "no short proof exists" claim. Declared
rather than hidden; see `writeup.md` limitations.

### Still to do (ALL COMPLETED -- see entries below)

- ~~Re-measure P on the corrected pool; run expert iteration with the frozen
  control, 2 seeds; Stage 3 tables; one-shot test-set run; writeup.~~ Done: P=7
  confirmed on the decontaminated pool, both seeds run to 5 rounds
  (frontier 9 and 10 against a frozen control at 7), Stage 3 table and a single
  greedy test run per file completed, REPORT.md / writeup.md / numbers.md
  written.

### The barrier: a learned stopping prior (main finding)

Three hypotheses, in the order I tried them. The first two were wrong.

1. **Reference tokenisation** (wrong, but the fix is kept). Absolute indices
   `N7`+ are unseen under a cap of 6, so I replaced them with back-distances
   `B<k>` and premise refs `P<i>`. Necessary hygiene, not the barrier.
2. **Positional scheme** (wrong). Learned absolute position embeddings are
   untrained past the longest training sequence (194 tokens), so I trained
   `rope` and `nope` variants. All three reach val loss ~0.077 and held-out
   95-96%, and **all three stop at exactly 7 written lines** -- 192k samples
   each on the mixed pool, 64k each on the hard pool, not one proof past 8.
   Robust frontier 7 for every scheme. So the positional scheme is not what
   pins the frontier.
3. **The stopping prior** (this is it). Teacher-forcing each model along a
   *known-good* 9-16 line proof and reading P(QED) at each line boundary:

   | after line | learned | rope  | nope  |
   |-----------:|--------:|------:|------:|
   | 5          | 0.023   | 0.016 | 0.043 |
   | 6          | 0.324   | 0.097 | 0.315 |
   | 7          | 0.736   | 0.467 | 0.782 |
   | 8          | 0.932   | 0.809 | 0.945 |
   | 9          | 0.982   | 0.936 | 0.978 |
   | 12         | 0.994   | 0.961 | 0.989 |

   Mid-way through a valid long derivation the model still wants to stop:
   ~32% at line 6, ~74-78% at line 7, >93% by line 9. Writing a 10-line proof
   by sampling means beating ~0.95-0.99 stop probability at every boundary
   past 6, and that product is why the ceiling is a hard stop rather than a
   decay. The cap-6 data taught "proofs end by line 6" and that alone pins the
   frontier.

   The mechanism predicts the frontier data rather than merely restating it:
   `rope` has the weakest stopping prior at every line, and `rope` is exactly
   the model that wrote the most 7-line proofs (351 vs 177 for `learned` and
   129 for `nope`) and one of the two 8-line proofs.

Consequence for Stage 2: expert iteration on verified long proofs trains
P(QED) toward 0 at lines 6, 7, 8..., which attacks the prior directly. The 351
seven-line proofs `rope` already produces are the seed for that bootstrap.
`rope` is therefore the base model for RL.

### Correction to an earlier number: P is 7, not 6

I first reported P = 6 from validation_36 (`0/24` on the `>6` bin, greedy).
But the frontier is defined as the robust frontier over *written* lengths
(>= 5 distinct verified proofs), and by that definition every Stage-1 model
sits at 7, with 129-351 distinct 7-line proofs -- far above the threshold of
5. The pre-trained model already writes one line beyond its cap. L must beat
7, not 6; using 6 would have inflated the headline by a full line.

### Target pool, second attempt

- 30000 fresh candidates generated at 9-16 lines. Probed each with 24 samples
  from a Stage-1 model and dropped any with a found <=6-line proof.
- **26941 of 30000 (90%) had a <=6-line proof.** Only 3059 survive. That is
  the size of the gap between generating length and shortest proof.
- Survivors split 2000 RL targets / 800 transfer, disjoint by theorem.
- Known leak: the filter used one model at k=24; re-probing with three models
  at k=32 still found 532 distinct <=6-line proofs on the "hard" pool. The
  filter is an upper bound, not a proof of hardness, and is reported as such.

### Infrastructure notes

- Disabling the cuDNN SDPA backend took 2048 greedy samples from 49s to 0.8s
  (~60x) and fixed a crash at batch 2048. Verified the speedup is real by
  checking held-out is unchanged (95.6%).
- Fully-masked padding rows produced NaN that leaked into real positions
  through the next layer's values (masked weights are exactly 0, and
  0 * NaN = NaN). Fixed with a large finite mask value instead of -inf.
- Residual batch-composition sensitivity (1-5 of 512 decoded proofs differ
  between batch sizes) is bf16 reduction-order noise, affects all three
  variants, and leaves solve rates identical. Same-batch runs are exactly
  deterministic. I briefly mis-flagged this as a NoPE-specific bug.

### Stage 2 results and the negative result that matters

Expert iteration from `ckpt/sft_rope.pt`, 2000 hard RL targets, 800 transfer,
k=32 at T=1.0, 5 rounds, hindsight relabelling on, frozen control at an
identical budget.

- Frontier 7 -> 8 (round 2) -> 9 (round 5). Frozen control never leaves 7.
- 8-line proofs 6 -> 45 -> 160 -> 454 across rounds; 9-line 0 -> 2 -> 19.
- RL targets 55.6% vs frozen 34.1%; transfer 50.8% vs 23.6%.
- Transfer set GREEDY 6.0% -> 26.8% (4.5x), with written lengths going
  {4:1,5:2,6:11,7:34} -> {4:2,5:5,6:25,7:159,8:21,9:2}.
- Round 1 sanity check passes: policy == frozen (466 vs 467 solved).
- Seed 1 replicates fully: 57.0% vs frozen 33.1%, frontier 9 vs 7, 33 distinct
  9-line proofs. The two seeds agree to within 2 theorems at every round.

**And it does not transfer.** validation_36 6/36 -> 4/36, bin `>6` 0/24 both.
Test set one greedy run per file: short 46.8% -> 46.8%, long 8.6% -> 8.8%.

Diagnosis, from what the model WRITES on validation_36's `>6` bin: median 5
lines before RL, 6 after, against theorems needing 7-18. RL loosened the
stopping prior by about one line. The failures are overwhelmingly well-formed
proofs citing a rule whose premises do not hold -- before and after. The model
composes plausible steps; it does not search. Expert iteration on random-walk
theorems rewards continuing, not searching, so it made a more fluent continuer
and left the real weakness alone.

This is the honest reading of L - P: it is a length metric, and length is
purchasable without buying proof search.

### Correction: the "length fluency vs proof search" claim was wrong

I wrote that expert iteration "bought length fluency, not proof search", i.e.
that the model emits valid steps but fails to aim at the goal. Classifying every
greedy failure by the verifier's reason shows the opposite:

| failure class | validation_36 stage1 | validation_36 after RL | transfer after RL |
|---|---|---|---|
| invalid rule application | 90.0% | 78.1% | 85.2% |
| valid but missed the goal | 3.3% | 12.5% | 11.1% |
| malformed output | 3.3% | 6.2% | 3.2% |

The dominant failure everywhere is a line that is not a valid rule application.
The binding constraint is per-line validity on unfamiliar formulas, not
goal-directedness, so the length/search dichotomy was a story imposed on the
data rather than read off it. Corrected in writeup.md. The frontier and control
results are unaffected -- only the mechanism paragraph was wrong.

### Correction: only ONE of the two rival hypotheses was actually falsified

I described the reference-tokenisation hypothesis as "tested and wrong". It was
not tested. I designed the relative-reference tokenizer (`B<k>`/`P<i>`) at the
outset and never trained a model with absolute `N`-indices, so its effect on the
frontier is unmeasured. Accurate status:

- H1, absolute line references: **untested** -- removed by design, never ablated.
- H2, positional scheme: **falsified** -- learned/rope/nope all stop at 7.
- H3, stopping prior: **supported** -- measured directly, and predicts the
  cross-variant ordering (rope has the weakest prior and writes the most 7-line
  proofs).

The H1 ablation (train Stage 1 with absolute indices, re-probe the frontier) is
~10 minutes of compute and is the first thing to run next.

### Correction: the relative-reference claim was overstated

I wrote that under the `B<k>`/`P<i>` scheme "a 14-line proof is built from the
same reference tokens as a 4-line one". Measured, that is false. Cap-6 training
exercises only `B1`-`B4` and `P1`-`P3`; the 9-16 line reference proofs need
`B5`-`B14`, which are 8.07% of their references.

The correct comparison, over the same long proofs:

| scheme | index/reference tokens never seen in cap-6 training |
|---|---|
| absolute `N<i>` | 43.7% |
| relative `B<k>`/`P<i>` | 8.1% |

A 5.4x reduction, not elimination.

### New finding: the model never emits B5 or beyond

Across all 13,378 verified proofs in `runs/ei_seed0/pool.jsonl`, at lengths up
to 9, the count of references needing `B5`+ is **zero**:

| written length | n proofs | refs | refs needing B5+ |
|---|---|---|---|
| 6 | 6991 | 32795 | 0 |
| 7 | 3486 | 19888 | 0 |
| 8 | 560 | 4058 | 0 |
| 9 | 45 | 335 | 0 |

The model writes long proofs by only ever citing within 4 lines, plus premises.
So a theorem whose proof needs a longer-range citation is unreachable for it,
independent of rule knowledge. This is a second structural constraint alongside
the stopping prior, and it was found only because the relative-reference claim
was challenged.

### Held-out renaming inflation, quantified

- renamings of a training theorem: 618/622 = **99.4%**
- structurally new: 827/878 = **94.2%**
- combined headline: 96.3%

The 41.8% renaming overlap inflates the headline by roughly 2 points. 94.2% is
the defensible figure for unseen structure.

### H1 CONFIRMED: the reference codec gates length generalisation

Trained an otherwise identical Stage-1 model on the spec's absolute line
indices (`ablation_abs_refs.py`): same data, same 4L/d=256/rope architecture,
same 6000 steps, same seed, same probe protocol.

| model | held-out greedy | solved/2000 | written lengths | frontier | longest |
|---|---|---|---|---|---|
| absolute `N<i>` | 96.6% | 260 | {4:16,5:100,6:353} | **6** | **6** |
| relative `B<k>`/`P<i>` | 96.3% | 477 | {4:11,5:106,6:325,7:351,8:1} | **7** | 8 |

The absolute model wrote **zero** proofs past its 6-line training cap in ~64k
samples. So the codec is not hygiene, it is the gate: without it there is no
length generalisation to measure at all.

I predicted before running it that the absolute model would land at 7 or 8 --
"at most one line worse" -- reasoning that since the relative model never emits
`B5`+ anyway, it was not exploiting its wider coverage. That was wrong. Being
able to *represent* a reference cheaply matters even when the long-range
references are never used.

Note the trap: the absolute model is BETTER in-distribution (96.6% vs 96.3%).
Held-out accuracy alone would have said the codec is irrelevant. The effect is
only visible when length generalisation is measured directly.

Hypothesis status is now:
- H1 reference codec: **CONFIRMED** -- gates whether the frontier exceeds 6.
- H2 positional scheme: **FALSIFIED** -- learned/rope/nope all reach 7.
- H3 stopping prior: **SUPPORTED** -- constrains how far past 6 it gets.

H1 and H3 are constraints on different things: H1 on whether the model can
exceed the cap, H3 on how far. Both bind.

### Bug: same vocabulary collision, reintroduced

`ablation_abs_refs.py` built `ITOS` by inverting `STOI`, which silently drops
any index whose token string is duplicated -- and `R` is both the atom and the
reiteration rule. Sampling the dropped index crashed the probe (KeyError: 10),
exactly the bug fixed in `nd/tokenizer.py` earlier and reintroduced because this
script defines its own vocabulary. Fix: build the reverse map from
`enumerate(VOCAB)`, which is correct by construction. Training was unaffected;
only the probe crashed, and it was re-run from the saved checkpoint.

### Mechanism for the zero-B5+ observation, and a correction

I claimed theorems needing a `B5`+ citation are "unreachable regardless of
reasoning". Measured, that is too strong.

Teacher-forcing along reference proofs, the probability the model assigns to the
CORRECT next reference token:

| correct token | n | P(correct) | emits instead |
|---|---|---|---|
| B1 | 1892 | 0.76 | B1 (77%) |
| B2 | 838 | 0.47 | B2 (48%) |
| B3 | 516 | 0.42 | B3 (43%) |
| B4 | 365 | 0.71 | B4 (76%) |
| B5 | 137 | 0.008 | **B4** (72%) |
| B6 | 101 | 0.003 | **B4** (86%) |
| B8 | 50 | 0.003 | **B4** (94%) |
| B9 | 46 | 0.004 | **B4** (91%) |

Mechanism: cross-entropy raises the target token's probability and lowers every
other token's at each position. `B5`+ was never a target in ~9.6M training
positions, so it was only ever pushed down. The model saturates at `B4`, the
largest offset it saw, producing a citation to the wrong line -- which shows up
as a rule-check failure, the dominant failure class.

Two corrections to the earlier claim:
1. P(B8) is 0.003, not 0. Unlikely, not impossible.
2. A theorem has many proofs. Citing a result 12 lines back (`B12`, p~0.01) can
   be replaced by re-deriving it one line earlier (`B1`, p~0.76) at the cost of
   one extra line. So the theorem is reachable; that particular proof is not.

The real structure is an interaction: avoiding far references costs length, and
length is blocked by the stopping prior. The model is confined to proofs that
fit inside both limits, which is why it emits zero `B5`+ rather than
"choosing" locality.

### Formalising the frontier: surprisal budget (quantitatively validated)

Autoregressive sampling is Boltzmann sampling. With E(y) = -log p(y|x), the
per-token log-probabilities add, so p(y) = exp(-E(y))/Z and temperature tau
gives p_tau(y) ~ exp(-E(y)/tau). Sampling k times finds y when k*p(y) >~ 1,
i.e.

    E(y) <= log k                    (reachability criterion)

so the frontier is  L*(k) = max{ L : E_min(L) <= log k }  where E_min(L) is the
smallest total surprisal over verified L-line proofs.

**First attempt failed.** I used E_stop (the stopping-prior term) as a proxy for
E. It is only a lower bound, and it predicted frontier 9 for Stage 1 (observed
7) and 12 after RL (observed 9), and a log-k growth of ~1.3 lines over a 64x
budget increase (observed 0).

**Measuring the full E fixes it.** Teacher-forcing every verified proof in the
RL pool and summing -log p over the body (`scripts_energy_vs_length.py`):

| L | Stage-1 min E | k needed | after-RL min E | k needed |
|---|---|---|---|---|
| 2-7 | 0.01-0.30 | ~1 | 0.01-0.13 | ~1 |
| 8 | **7.86** | 2590 | **0.05** | 1.05 |
| 9 | **24.81** | 5.9e10 | **1.51** | 4.5 |

E is not linear in L. It is flat then cliffs. Stage 1 cliffs between 7 and 8;
after RL between 9 and 10.

Testing L*(k) = max{L : E_min(L) <= log k} against a k=1..64 sweep
(`scripts_frontier_vs_k.py`) matches the observed robust frontier in **11 of 12**
testable cases (k=1 is degenerate: log 1 = 0). The one miss is the RL model at
k=4 -- predicted 8, observed 9, with E_min(9)=1.51 against log 4=1.39, off by
0.12 units of log-probability.

Consequences:
- **Sampling is exponentially weak.** k enters only as log k. Lifting Stage 1
  from 7 to 8 lines by sampling alone needs k ~ 2590 instead of 1.
- **Training is exponentially strong.** RL lowered E_min(8) by 7.8 units of log-probability
  (~2400x in probability) and E_min(9) by 23.3 units of log-probability (~1.3e10x) at fixed k.
  That is the precise sense in which RL beats the frozen control.
- **The apparent discontinuity is a threshold artefact.** E_min moves
  continuously with training; L* is an integer max over a threshold, so it
  steps. No phase transition is needed or implied.

### CONTAMINATION: the Stage-1 training set contained validation and test theorems

Found by a systematic overlap audit of every split against every other split.
I built the contamination filter for the Stage-2 pools and never applied it to
the Stage-1 data, which was generated earlier.

Overlap of the ORIGINAL Stage-1 training set with the held-out benchmarks:

| set | exact in train | renaming in train |
|---|---|---|
| validation_36 | 4 / 36 | 6 / 36 |
| test_short | 45 / 267 (16.9%) | 91 / 267 (**34.1%**) |
| test_long | 8 / 532 (1.5%) | 15 / 532 (2.8%) |

Worse, of the 6 validation theorems Stage 1 solved, **5 are contaminated**
(`modus_ponens`, `dn_intro`, `dn_elim`, `identity`, `positive_paradox`); the
only clean solve was `negative_paradox`. So the reported 6/36 was 1 clean
solve plus 5 memorised theorems, and the 46.8% on test_short was inflated by a
34% renaming overlap.

Why it happened: with only four atoms, a random generator producing 120k
theorems hits textbook sequents by chance. `( P > Q ) , P |- Q` is a perfectly
ordinary output of a forward random walk. Contamination here is the default
outcome, not an accident, and it has to be filtered explicitly.

Affected: validation_36 and test_short numbers for every model, since all of
them descend from this Stage-1 checkpoint. NOT affected: the frontier results
(L, P, L-P), which are measured on `rl_targets_hard` / `transfer_hard`. Those
pools were built with the filter and audit clean (0 exact, 0 renaming overlap
against validation and test).

Fix applied: `scripts_build_data.py` now removes any generated theorem that is,
or is an atom-renaming of, a validation or test theorem. Regenerating dropped
684 of 120,000 candidates. Retrained Stage 1 on the clean data and re-measured
everything downstream.

Filtering training data against the test prompts is decontamination, not tuning:
no test verdict is inspected, and the brief states that training-set overlap is
audited.

### Which conclusions survive the contamination fix

The contamination was between the Stage-1 training data and the two *external*
benchmarks (validation_36, test). It did not touch the internally-generated
pools. So:

**Unaffected (measured on `rl_targets_hard` / `transfer_hard`, both audited to
0 overlap, exact and under renaming):**
- H1, the reference-codec ablation. Absolute vs relative was an apples-to-apples
  comparison: both models trained on the same data, probed on the same clean
  targets. Frontier 6 vs 7 stands.
- H2, the positional-scheme comparison. Same argument; all three schemes reach 7.
- H3, the stopping prior. Measured by teacher-forcing on reference proofs from
  the clean hard pool.
- The surprisal/energy analysis and the frontier-vs-k sweep, both computed on
  the clean pools.

**Affected, re-measured on the decontaminated checkpoint:**
- Stage-1 and RL validation_36 scores.
- Stage-1 and RL test_short scores (test_long overlap was only 1.5% exact).
- Held-out solve rates (the held-out set itself was regenerated).

A second, smaller leak appeared when the training data was regenerated: 6 of the
2000 RL targets landed in the new training set (1 exact, 6 under renaming).
Removed; `rl_targets_hard` is now 1994 and every pairwise audit is 0.

### Correction: overlap is not the same as memorisation

Having found that 5 of the 6 validation theorems Stage 1 solved were in its
training set, I described the 6/36 as "1 clean solve plus 5 memorised". That
inference was untested.

Retraining on decontaminated data and re-running validation_36 greedy:

| model | solves |
|---|---|
| contaminated Stage 1 | dn_elim, dn_intro, identity, modus_ponens, negative_paradox, positive_paradox |
| **decontaminated Stage 1** | **identical set** |

The clean model solves exactly the same six, including all five that were
previously in its training data. Those theorems are learnable from the
distribution rather than recalled: `modus_ponens` is a three-line `IMPE` proof
and `IMPE` occurs 2159 times in the training set, so the specific sequent is not
needed.

So both statements hold and should be reported together: the contamination was
real and had to be removed (every split now audits to 0 overlap, exact and under
renaming), AND it was not inflating the validation number. Removing it changed
which data the model saw and changed nothing about which validation theorems it
can prove.

The test_short overlap (34.1% under renaming) is the remaining case where
contamination could have mattered; the decontaminated test numbers settle it.

### Energy-entropy decomposition of proof search (new experiment)

A theorem is proved if ANY of its proofs is sampled, so the governing quantity
is the free energy of the whole proof set, not the energy of one proof:

    p(prove x) = sum_{y in Y(x)} exp(-E(y)) = exp(-F(x))
    F(x) = -log sum_y exp(-E(y))  ~  E - log g       (g = degeneracy)

`scripts_degeneracy.py`, 800 hard targets, k=64, RL model. Grouping solved
theorems by the number of DISTINCT proofs found:

| g | n | mean p | mean E_min |
|---|---|---|---|
| 1 | 180 | 0.432 | 1.763 |
| 2-3 | 126 | 0.718 | 0.674 |
| 4-7 | 45 | 0.818 | 0.818 |
| 8-15 | 26 | 0.718 | 1.342 |
| 16-31 | 34 | 0.813 | 2.313 |
| 32-64 | 35 | 0.911 | 3.192 |

**Confirmed qualitatively:** theorems with g=1 have the *best* minimum energy
(1.76) and the *worst* solve rate (0.43); theorems with g=32-64 have nearly
twice the energy (3.19) and solve at 0.91. The entropy term more than
compensates for worse per-proof energy -- the free-energy tradeoff, observed.

Fitted slopes: d log p / d log g = **+0.397** (idealised +1),
d log p / d E_min = **-0.544** (idealised -1). Right signs, attenuated
magnitudes.

**Not confirmed quantitatively, and here is why:**
1. `log p = log g - E` is a small-p linearisation; most solved theorems here sit
   at p = 0.4-0.9 where log p saturates at 0.
2. `F = E - log g` assumes all proofs have comparable energy. They do not; the
   sum is dominated by the lowest-energy few, so the effective degeneracy is
   `g_eff = sum exp(-(E_y - E_min)) <= g`.
3. `g` is censored -- distinct proofs found in 64 draws, not |Y(x)| -- and is
   estimated from the same draws as p, which couples them and would *inflate*
   the slope. Attenuation despite that coupling means the true attenuation from
   (1) and (2) is larger than measured.

So the mechanism is real and the signs are predicted; the idealised exponents
are not, and the failure is understood rather than ignored.

Consequence for the degenerate-theorem question: trivial theorems have enormous
g (insert a reiteration, swap ANDI argument order, add a vacuous ORI), so they
dominate the low-L part of the partition function and contribute nothing at high
L. That is a concrete mechanism by which the 23.1% degenerate fraction of the
training distribution would shape the stopping prior toward short proofs.
UNTESTED: verifying it requires retraining Stage 1 on non-trivial-only data and
re-measuring P(QED) and P.

### Tracking the mechanism round by round

`scripts_qed_across_rounds.py`. If the diagnosis is right -- expert iteration
works by lowering the probability of stopping -- then P(QED | n lines) should
fall monotonically across rounds, not just between the endpoints.

| checkpoint | L=6 | L=7 | L=8 | L=9 | L=10 | L=11 | L=12 |
|---|---|---|---|---|---|---|---|
| SFT (round 0) | 0.123 | 0.518 | 0.865 | 0.965 | 0.972 | 0.975 | 0.974 |
| after round 1 | 0.044 | 0.186 | 0.484 | 0.773 | 0.911 | 0.963 | 0.979 |
| after round 2 | 0.030 | 0.113 | 0.336 | 0.642 | 0.835 | 0.918 | 0.952 |
| after round 3 | 0.032 | 0.095 | 0.244 | 0.494 | 0.690 | 0.805 | 0.888 |
| after round 4 | 0.028 | 0.076 | 0.169 | 0.366 | 0.510 | 0.627 | 0.747 |

Monotone at every length and every round. Note `final.pt` == `policy_r4.pt`:
round 5 samples and evaluates but is not followed by a retrain.

### ORACLE CONTROL: would supervised training on long proofs have done it anyway?

`ablation_oracle_sft.py`. Trained a model on the generator's own gold 9-16 line
proofs for the RL targets -- data the exam forbids for Stage 1 and that RL never
had -- and measured the frontier on the transfer set, held out from every arm.
An analysis control, not a submitted model.

| model | trained on | greedy | pass@32 | robust frontier |
|---|---|---|---|---|
| Stage 1 | cap-6 only | 5.9% | 21.9% | **7** |
| after RL | cap-6 + ~13k self-found | 25.8% | 49.6% | **9-10** |
| oracle SFT | cap-6 + 1994 gold long | 23.2% | **63.0%** | **16** |

**This is the most deflating result in the project and it belongs in the
headline.** Gold long proofs give frontier 16; RL gives 9-10. RL recovered
roughly a quarter to a third of the gap between the baseline and what real
supervision achieves. The bottleneck here is data, and expert iteration is an
inefficient way to manufacture it.

What RL can still claim: that data does not exist. The cap removes it by
construction, and outside a synthetic setting nothing hands you longer proofs.
RL produces part of the same effect from the model plus a checker alone.

The mechanism is identical across all three regimes, which is the strongest
support the stopping-prior diagnosis has:

| model | P(stop) after 9 lines | frontier |
|---|---|---|
| Stage 1 | 0.965 | 7 |
| after RL | 0.366 | 9-10 |
| oracle SFT | 0.109 | 16 |

The frontier is a monotone function of one scalar, and that scalar is set by the
length distribution of the training data.

### Terminology

Removed the unit "nats" from every user-facing document at the reader's request;
surprisals are now reported as bare log-probabilities.
