# numbers.md

Every number in the write-up, with the file it came from. Rates are greedy
unless stated; intervals are Wilson 95%.

**Provenance note.** An overlap audit late in the project found that the
original Stage-1 training set contained validation and test theorems (see
"Contamination audit" below). Stage 1 was retrained on decontaminated data and
everything downstream re-measured. Numbers are tagged **[clean]** when they come
from the decontaminated pipeline and **[pre-fix]** when they come from the
original one. Every split now audits to 0 overlap, exact and under atom-renaming.

## Contamination audit

Original Stage-1 training set vs the external benchmarks:

| benchmark | exact overlap | overlap under atom-renaming |
|---|---|---|
| validation_36 | 4 / 36 | 6 / 36 |
| test_short | 45 / 267 (16.9%) | 91 / 267 (34.1%) |
| test_long | 8 / 532 (1.5%) | 15 / 532 (2.8%) |

After adding the filter to generation (684 of 120,000 candidates dropped) and
removing 6 RL targets that the regenerated training set had absorbed:

| pair | exact | renaming |
|---|---|---|
| train x heldout / RL / transfer / validation / test | **0** | **0** |
| RL x transfer / validation / test | **0** | **0** |
| transfer x validation / test | **0** | **0** |

**Effect on results: none detectable on validation_36.** The decontaminated
Stage-1 model solves the identical six theorems as the contaminated one
(`dn_elim, dn_intro, identity, modus_ponens, negative_paradox,
positive_paradox`), including all five that had been in the training set. They
are learnable from the distribution, not recalled. Overlap is not the same as
memorisation.

## Stage 1 -- clean baseline [clean]

Model: 4 layers, d=256, 4 heads, rope, 3.18M params, 6000 steps at batch 256,
278s on one H200, val loss 0.0811. Data: 116,316 train / 3,000 held-out,
decontaminated, theorem-disjoint, all proofs <= 6 lines.

| metric | value |
|---|---|
| held-out greedy, n=1500 | **97.1%** [96.1-97.8] |
| by proof length 2/3/4/5/6 | 100.0 / 98.4 / 97.8 / 97.9 / **92.0**% |
| non-trivial only | 96.4% [95.1-97.3], n=1161 |
| validation_36 greedy | **6/36**; bin `<=6` 6/12, bin `>6` **0/24** |
| P (robust frontier, 1994 hard targets, k=32, T=1.0) | **7** |
| written lengths at that probe | {4:14, 5:108, 6:311, 7:300, 8:1} |
| theorems solved in that probe | 449 / 1994 |

## CLEAN RESULTS (decontaminated pipeline) [clean]

### Stage 2 -- expert iteration, seed 0 (`runs/clean_seed0/log.json`)

Base `ckpt/sft_rope_clean.pt`; 1994 RL targets, 800 transfer, k=32 per round at
T=1.0, hindsight relabelling on, frozen control at an identical budget.
Cumulative distinct theorems solved.

| round | RL policy | frozen (same budget) | transfer | transfer frozen | held-out | frontier | frozen frontier |
|---|---|---|---|---|---|---|---|
| 1 | 463 (23.2%) | 440 (22.1%) | 21.2% | 22.2% | 97.8% | **7** | 7 |
| 2 | 679 (34.1%) | 514 (25.8%) | 29.8% | 21.1% | 95.6% | **8** | 7 |
| 3 | 850 (42.6%) | 556 (27.9%) | 38.9% | 21.9% | 96.2% | **8** | 7 |
| 4 | 989 (49.6%) | 586 (29.4%) | 45.1% | 22.1% | 96.0% | **8** | 7 |
| 5 | 1091 (54.7%) | 622 (31.2%) | 49.8% | 20.2% | 96.8% | **9** | 7 |

Final written lengths, RL policy: `{'4': 21, '5': 374, '6': 2116, '7': 1948, '8': 547, '9': 32, '10': 1}`
Frozen cumulative:                `{'4': 24, '5': 265, '6': 994, '7': 542, '8': 1}`

The frozen control found **one** 8-line proof across five rounds at the same
budget; the policy wrote 547 eight-line and 32 nine-line proofs.

**L = 9, P = 7, L - P = 2.**

Note: the per-round console line and the cumulative column are different
quantities. Per round the frozen model steadily solves ~22-23%; cumulatively it
creeps to 31.2% as resampling turns up new theorems.

### Stage 2 -- expert iteration, seed 1 (`runs/clean_seed1/log.json`) [clean]

| round | RL policy | frozen | transfer | held-out | frontier | frozen frontier |
|---|---|---|---|---|---|---|
| 1 | 452 (22.7%) | 430 (21.6%) | 179 (22.4%) | 97.8% | **7** | 7 |
| 2 | 667 (33.5%) | 509 (25.5%) | 246 (30.8%) | 96.0% | **8** | 7 |
| 3 | 823 (41.3%) | 552 (27.7%) | 301 (37.6%) | 96.2% | **8** | 7 |
| 4 | 960 (48.1%) | 585 (29.3%) | 359 (44.9%) | 96.4% | **9** | 7 |
| 5 | 1060 (53.2%) | 603 (30.2%) | 387 (48.4%) | 96.4% | **10** | 7 |

Final written lengths, seed 1 policy: `{'4': 16, '5': 360, '6': 2207, '7': 1889, '8': 611, '9': 106, '10': 10}`
Frozen cumulative, seed 1:            `{'4': 22, '5': 280, '6': 995, '7': 512}`

### Two-seed summary [clean]

| | seed 0 | seed 1 |
|---|---|---|
| RL targets solved | 1091 (54.7%) | 1060 (53.2%) |
| frozen (same budget) | 622 (31.2%) | 603 (30.2%) |
| transfer | 49.8% | 48.4% |
| transfer frozen | 20.2% | 21.6% |
| held-out greedy | 96.8% | 96.4% |
| **robust frontier** | **9** | **10** |
| frozen frontier | 7 | 7 |
| distinct 9-line proofs | 32 | 106 |
| distinct 10-line proofs | 1 | 10 |
| frontier by round | 7,8,8,8,9 | 7,8,8,9,10 |

**L = 9-10, P = 7, so L - P = 2-3.** The value both seeds support is
**L - P >= 2**. The frontier differs between seeds because the robust criterion
thresholds a count (>= 5 distinct proofs): seed 0 wrote 1 ten-line proof and
seed 1 wrote 10.


### Stage 3 -- the table (`numbers_stage3_clean.json`)

| model | held-out <=6 greedy | transfer >6 greedy | transfer pass@32 | validation_36 | val bin `>6` |
|---|---|---|---|---|---|
| Stage 1 (clean) | 97.1% [96.1-97.8] | **5.9%** [4.4-7.7] | 21.9% [19.1-24.9] | 6/36 | 0/24 |
| after RL | 96.3% [95.2-97.1] | **25.8%** [22.8-28.9] | 49.6% [46.2-53.1] | 4/36 | 0/24 |

Greedy written lengths on the transfer set:
- Stage 1 `{5:5, 6:12, 7:30}`
- after RL `{4:1, 5:6, 6:34, 7:147, 8:18}`

### Stage 3 -- test set, ONE greedy run per file per model

| model | test_short (<=6) | test_long (>6) |
|---|---|---|
| Stage 1 (clean) | **47.6%** (127/267) [41.7-53.5] | **8.8%** (47/532) [6.7-11.6] |
| after RL | **46.4%** (124/267) [40.6-52.4] | **8.1%** (43/532) [6.1-10.7] |

RL is slightly negative on both, inside the confidence intervals.

### Contamination had no measurable effect on either external benchmark

| | pre-fix | clean |
|---|---|---|
| validation_36, Stage 1 | 6/36 (same six theorems) | 6/36 (same six) |
| test_short, Stage 1 | 46.8% | **47.6%** |
| test_long, Stage 1 | 8.6% | **8.8%** |

Despite a 34.1% renaming overlap on test_short, removing the contamination moved
the score by +0.8pp, i.e. not at all. In a four-atom formula space the model
learns the *patterns* whether or not the specific sequent was in training.

### The headline [clean]

| protocol | P | L | L - P |
|---|---|---|---|
| 32 samples at T=1.0, RL targets | 7 | 9 (seed 0) / 10 (seed 1) | **2-3** |
| greedy, transfer set (robust frontier) | 7 | 8 | **1** |
| greedy, validation_36 bin `>6` | 0/24 | 0/24 | **0** |
| greedy, test_long | 8.8% | 8.1% | **0** (noise) |


## Stage 1 -- generator (`nd/gen.py`)

| quantity | value | source |
|---|---|---|
| sampled proofs passing `nd_verify` | 18396 / 18396 (zero invalid) | generator self-test |
| generator throughput | ~5400 proofs/s | generator self-test |
| yield (non-dud attempts) | 92% | generator self-test |
| rule counts before/after enumerating applicable rules | `IMPE` 227->2159, `NEGE` 457->1512, `DN` 146->505, `ORE` 35->188 | log.md |

## Stage 1 -- data (`data/`, `scripts_build_data.py`)

| quantity | value |
|---|---|
| distinct theorems | 120,000 (117,000 train / 3,000 held-out, disjoint by theorem) |
| proof lengths | 2-6, cap respected |
| trivial fraction (train / held-out) | 23.1% / 24.5% |
| held-out that is an atom-renaming of a train theorem | 41.8% |
| max training sequence length | 194 tokens |
| vocabulary | 65 tokens |

## Stage 1 -- model and baseline [pre-fix]

Model: 4 layers, d=256, 4 heads, from scratch. `learned` 3.26M params,
`rope`/`nope` 3.18M. 6000 steps, batch 256.

| model | val loss | held-out greedy (n=1500) |
|---|---|---|
| learned | 0.0774 | 96.0% [94.9-96.9] |
| rope | 0.0776 | 96.3% |
| nope | 0.0761 | 96.4% |

Held-out greedy by proof length (`learned`, n=1500), `eval_heldout.py`:

| length | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| solve | 100.0% | 98.8% | 96.9% | 94.4% | 91.0% |
| n | 202 | 340 | 322 | 337 | 299 |

Non-trivial only: 94.9% [93.5-96.0], n=1140.

validation_36 greedy (`eval_validation.py`), `learned` and `rope` both 6/36:
bin `<=6` 6/12 [25.4-74.6], bin `>6` **0/24** [0.0-13.8].
Solved: modus_ponens, dn_intro, dn_elim, identity, negative_paradox,
positive_paradox. 34/36 failures are well-formed proofs failing a rule check,
not parse errors.

## Stage 2 -- the target pool [unaffected: clean pools]

| quantity | value | source |
|---|---|---|
| first pool solved by Stage-1 in ONE sample | 2706 / 4000 | log.md (dead end) |
| written lengths of those solutions | {3:314, 4:602, 5:785, 6:942, 7:20} | |
| candidates generated at 9-16 lines | 30,000 | `scripts_build_hard.py` |
| **had a <=6-line proof, dropped** | **26,941 (90%)** | |
| survive ("no <=6-line proof found in 24 samples") | 3,059 (10%) | |
| final pools | 2000 RL targets / 800 transfer, disjoint | |
| leak: <=6-line proofs still found on the "hard" pool by 3 models at k=32 | 532 distinct proofs | `numbers_length_probe_hard.json` |

## Stage 2 -- P, the pre-training frontier [pre-fix]

Robust frontier = longest WRITTEN length with >= 5 distinct verified proofs.
2000 hard targets, k=32, T=1.0 (`scripts_probe_length.py`,
`numbers_length_probe_hard.json`):

| model | theorems solved | written lengths | robust frontier | longest single |
|---|---|---|---|---|
| learned | 371 / 2000 | {4:13, 5:108, 6:411, 7:177} | **7** | 7 |
| rope | 477 / 2000 | {4:11, 5:106, 6:325, 7:351, 8:1} | **7** | 8 |
| nope | 275 / 2000 | {3:1, 4:10, 5:64, 6:184, 7:129, 8:1} | **7** | 8 |

**P = 7.** (On the mixed pool, same ceiling: all three stop at 7 across 192k
samples each.)

## Stage 2 -- the barrier: P(QED) mid-proof [unaffected: clean pools]

Teacher-forced along known-good 9-16 line proofs, n=1500
(`scripts_qed_prior.py`, `numbers_qed_prior.json`, figure 1):

| after line | 5 | 6 | 7 | 8 | 9 | 12 |
|---|---|---|---|---|---|---|
| learned | 0.023 | 0.324 | 0.736 | 0.932 | 0.982 | 0.994 |
| rope | 0.016 | 0.097 | 0.467 | 0.809 | 0.936 | 0.961 |
| nope | 0.043 | 0.315 | 0.782 | 0.945 | 0.978 | 0.989 |

## H1 ablation -- reference tokenisation (`numbers_ablation_absrefs.json`) [unaffected: clean pools]

Identical data, architecture (4L, d=256, rope), steps (6000), batch (256) and
seed; the only change is the proof codec. `ablation_abs_refs.py`, then
`probe_absrefs.py`. Probe protocol identical to the main one: 2000 hard
targets, k=32, T=1.0.

| model | vocab | val loss | held-out greedy | solved/2000 | written lengths | robust frontier | longest single |
|---|---|---|---|---|---|---|---|
| absolute `N<i>` | 68 | 0.0710 | **96.6%** (1449/1500) | 260 | {4:16, 5:100, 6:353} | **6** | **6** |
| relative `B<k>`/`P<i>` | 65 | 0.0776 | 96.3% | 477 | {4:11, 5:106, 6:325, 7:351, 8:1} | **7** | 8 |

The absolute-reference model produced **no proof longer than its 6-line
training cap in ~64k samples**. H1 confirmed: the codec gates length
generalisation. Note the absolute model is *better* in-distribution, so
held-out accuracy alone hides the effect entirely.

### Reference-token coverage (why)

Over the 9-16 line reference proofs, fraction of index/reference tokens that
never occur in cap-6 training data:

| scheme | unseen |
|---|---|
| absolute `N<i>` | **43.7%** |
| relative `B<k>`/`P<i>` | **8.1%** |

Training exercises only `B1`-`B4` and `P1`-`P3`; long proofs need `B5`-`B14`.
The relative scheme reduces the out-of-distribution fraction 5.4x; it does not
eliminate it.

### What the model actually emits (`runs/ei_seed0/pool.jsonl`, 13,378 proofs)

| written length | n proofs | references | needing `B5`+ |
|---|---|---|---|
| 6 | 6991 | 32795 | **0** |
| 7 | 3486 | 19888 | **0** |
| 8 | 560 | 4058 | **0** |
| 9 | 45 | 335 | **0** |

The RL model never emits `B5` or beyond at any length: it writes long proofs
using only locally-scoped citations plus premises.

## Held-out renaming split (`nd/dataset.py::canon_rename`) [pre-fix]

| held-out subset | n | greedy solve |
|---|---|---|
| atom-renaming of a training theorem | 622 | **99.4%** (618) |
| structurally new | 878 | **94.2%** (827) |
| combined headline | 1500 | 96.3% |

## Failure classes, greedy (verifier `reason` field) [pre-fix]

| class | validation_36 stage1 | validation_36 after RL | transfer after RL |
|---|---|---|---|
| invalid rule application | 90.0% | 78.1% | 85.2% |
| valid but missed the goal | 3.3% | 12.5% | 11.1% |
| malformed output | 3.3% | 6.2% | 3.2% |

## Stage 2 -- expert iteration (`runs/ei_seed0/log.json`) [pre-fix]

Base `ckpt/sft_rope.pt`; 2000 RL targets, 800 transfer, k=32 per round at
T=1.0, hindsight relabelling on, frozen control at an identical budget.
Cumulative distinct theorems solved.

| round | RL policy | frozen (same budget) | transfer | transfer frozen | held-out greedy | frontier | frozen frontier |
|---|---|---|---|---|---|---|---|
| 1 | 466 (23.3%) | 467 (23.4%) | 190 (23.8%) | 197 (24.6%) | 96.4% | **7** | 7 |
| 2 | 728 (36.4%) | 563 (28.1%) | 253 (31.6%) | 184 (23.0%) | 96.4% | **8** | 7 |
| 3 | 895 (44.8%) | 619 (30.9%) | 318 (39.8%) | 195 (24.4%) | 95.2% | **8** | 7 |
| 4 | 1029 (51.5%) | 657 (32.9%) | 371 (46.4%) | 178 (22.2%) | 95.2% | **8** | 7 |
| 5 | 1112 (55.6%) | 683 (34.1%) | 406 (50.8%) | 189 (23.6%) | 95.0% | **9** | 7 |

Written-length histogram per round (distinct verified proofs, RL targets):

| round | written lengths |
|---|---|
| 1 | {'4': 13, '5': 104, '6': 328, '7': 334} |
| 2 | {'4': 16, '5': 194, '6': 730, '7': 723, '8': 6} |
| 3 | {'4': 19, '5': 305, '6': 1260, '7': 1134, '8': 45} |
| 4 | {'4': 16, '5': 351, '6': 1774, '7': 1574, '8': 160, '9': 2} |
| 5 | {'4': 19, '5': 385, '6': 2247, '7': 2070, '8': 454, '9': 19} |

Frozen control cumulative written lengths after round 5:
`{4:19, 5:253, 6:964, 7:701, 8:3}` -- robust frontier stays **7**.

### Seed 1 (`runs/ei_seed1/log.json`), independent run, same configuration

| round | RL policy | frozen | transfer | held-out | frontier | frozen frontier |
|---|---|---|---|---|---|---|
| 1 | 475 (23.8%) | 473 (23.6%) | 198 (24.8%) | 96.4% | 7 | 7 |
| 2 | 729 (36.5%) | 557 (27.9%) | 256 (32.0%) | 95.4% | **8** | 7 |
| 3 | 893 (44.6%) | 617 (30.9%) | 336 (42.0%) | 95.8% | 8 | 7 |
| 4 | 1029 (51.5%) | 642 (32.1%) | 386 (48.2%) | 95.4% | 8 | 7 |
| 5 | **1140 (57.0%)** | 663 (33.1%) | **420 (52.5%)** | 94.2% | **9** | 7 |

Seed 1 final written lengths: `{4:15, 5:345, 6:2230, 7:2174, 8:537, 9:33}`.
Both seeds reach robust frontier 9 (19 and 33 distinct 9-line proofs).

## Stage 3 -- the table (`numbers_stage3.json`, `eval_all.py`) [pre-fix]

| model | held-out <=6 greedy | transfer >6 greedy | transfer pass@32 | validation_36 | val bin >6 |
|---|---|---|---|---|---|
| sft_rope (Stage 1) | 96.3% [95.3-97.2] | **6.0%** [4.6-7.9] | 24.5% [21.6-27.6] | 6/36 | 0/24 |
| final (after RL) | 95.7% [94.5-96.6] | **26.8%** [23.8-29.9] | 50.7% [47.3-54.2] | 4/36 | 0/24 |

Greedy written lengths on the transfer set:
- sft_rope `{4:1, 5:2, 6:11, 7:34}` -> robust frontier **7**
- final `{4:2, 5:5, 6:25, 7:159, 8:21, 9:2}` -> robust frontier **8**

Held-out by length, final: 100.0 / 98.5 / 96.0 / 94.1 / 91.0% for lengths 2-6.

## Stage 3 -- test set (ONE greedy run per file per model, `score_test.py`) [pre-fix]

| model | test_short (<=6) | test_long (>6) |
|---|---|---|
| sft_rope (Stage 1) | **46.8%** (125/267) [40.9-52.8] | **8.6%** (46/532) [6.5-11.3] |
| final (after RL) | **46.8%** (125/267) [40.9-52.8] | **8.8%** (47/532) [6.7-11.6] |

No improvement: identical on short, +1 theorem on long.

## The headline [pre-fix]

| protocol | P (Stage 1) | L (after RL) | L - P |
|---|---|---|---|
| greedy, transfer set | 7 | 8 | **1** |
| k=32 at T=1.0, RL targets | 7 | 9 | **2** |
| greedy, validation_36 `>6` bin | 0/24 | 0/24 | **0** |
| greedy, test_long | 8.6% | 8.8% | **0** (noise) |

On the generator's own distribution the frontier moves, with a frozen control
that does not. On hand-curated external theorems nothing moves at all.

## What the model writes on validation_36's `>6` bin (greedy) [pre-fix]

| model | solved | median lines written | range |
|---|---|---|---|
| sft_rope | 6/36 | 5 | 2-6 |
| final | 4/36 | 6 | 2-7 |

These theorems need 7-18 lines. RL shifted the greedy length up by one and no
further, so the model still writes a short derivation and stops.


## Surprisal / energy analysis [unaffected: clean pools]

`scripts_energy_vs_length.py`, `scripts_frontier_vs_k.py`. Minimum total
surprisal `E = -log p(proof)` over verified proofs of each length:

| L | Stage-1 `E_min` (units of log-probability) | k needed | after-RL `E_min` | k needed |
|---|---|---|---|---|
| 2-7 | 0.01-0.30 | ~1 | 0.01-0.13 | ~1 |
| 8 | **7.86** | 2,590 | **0.05** | 1.05 |
| 9 | **24.81** | 5.9e10 | **1.51** | 4.5 |

Reachability criterion `L*(k) = max{L : E_min(L) <= log k}` vs the measured
robust frontier over a k = 1..64 sweep: **matches in 11 of 12** testable cases
(k=1 is degenerate since log 1 = 0). The single miss is the RL model at k=4,
predicted 8 and observed 9, with `E_min(9) = 1.51` against `log 4 = 1.39`.

Frontier vs budget (robust frontier, >= 5 distinct proofs):

| k | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| Stage 1 | 7 | 7 | 7 | 7 | 7 | 7 | **7** |
| after RL | 8 | 8 | 9 | 9 | 9 | 9 | **9** |

Sampling 64x more buys Stage 1 nothing, because `k` enters only as `log k`.
RL lowered `E_min(8)` by 7.8 units of log-probability (~2,400x in probability) and `E_min(9)` by
23.3 units of log-probability (~1.3e10) at fixed `k`.

## Reference-token coverage [unaffected: clean pools]

Fraction of index/reference tokens in a 9-16 line proof never seen in cap-6
training data: **absolute `N<i>` 43.7%**, **relative `B<k>`/`P<i>` 8.1%**.
Training exercises only `B1`-`B4` and `P1`-`P3`.

Probability the model assigns to the CORRECT next reference token
(teacher-forced on reference proofs):

| correct token | B1 | B2 | B3 | B4 | B5 | B6 | B8 | B9 |
|---|---|---|---|---|---|---|---|---|
| P(correct) | 0.76 | 0.47 | 0.42 | 0.71 | **0.008** | **0.003** | **0.003** | **0.004** |
| emits instead | B1 | B2 | B3 | B4 | **B4** | **B4** | **B4** | **B4** |

Across all 13,378 verified proofs the RL model wrote, `B5`+ appears **0** times.
