# Technical primer: what this project is, built from the ground up

No prior background assumed. Every term is defined where it first appears, and
every quantitative claim points at the experiment that produced it.

---

## 1. The object: what a proof is

### 1.1 Formulas

A **formula** is a syntactic object built from:

- **atoms** `P`, `Q`, `R`, `S` -- placeholders for statements that are either
  true or false;
- **falsum** `F` -- a constant that is always false (a contradiction);
- **connectives** `~` (not), `&` (and), `v` (or, inclusive), `>` (implies).

Formally the set of formulas is the smallest set such that every atom and `F`
is a formula, and if `A` and `B` are formulas then so are `( ~ A )`,
`( A & B )`, `( A v B )`, `( A > B )`. Everything compound is fully
parenthesised, so parsing is unambiguous and needs no precedence rules.

`>` is *material* implication, defined entirely by its truth table:
`( A > B )` is false exactly when `A` is true and `B` is false. It carries no
notion of causation or relevance.

### 1.2 Sequents

A **sequent** is a pair `(premises, conclusion)`, written

    A1 , A2 , ... , An  |-  C

`|-` is the turnstile. The sequent asserts: in every assignment of truth values
to atoms that makes all `Ai` true, `C` is also true. That semantic property is
called **entailment**. A sequent with zero premises asserts that `C` is a
tautology.

### 1.3 Proofs

Entailment is a semantic fact about truth tables. A **proof** is a *syntactic*
certificate of it: a finite object a machine can check without enumerating
truth assignments.

Concretely a proof is a finite sequence of **lines**. Each line is a 4-tuple

    (formula, rule, refs, depth)

rendered in this repo's format as

    N4  ( R v S )  :  ORI2  N1  ;
    ^      ^          ^      ^
    index  formula    rule   the earlier lines it cites

`:` and `;` are punctuation -- a separator and a terminator. They carry no
logical content.

**Premises are not the proof.** A premise is a formula you are *given*; it
appears as a line whose rule is `PR`, and it is free. The proof is the whole
derivation from those premises to the conclusion. A one-line "proof" consisting
only of a premise is valid only in the degenerate case where the conclusion *is*
that premise.

### 1.4 Rules

A **rule** is a schema with a precondition on the cited lines and a conclusion.
Fifteen of them. Examples:

| rule | precondition on cited lines | derives |
|---|---|---|
| `ANDI a b` | line `a` is `A`, line `b` is `B` | `( A & B )` |
| `ANDE1 a` | line `a` is `( A & B )` | `A` |
| `IMPE a b` | line `a` is `( A > B )`, line `b` is `A` | `B` |
| `NEGE a b` | line `a` is `A`, line `b` is `( ~ A )` | `F` |
| `BOTE a` | line `a` is `F` | anything |
| `DN a` | line `a` is `( ~ ( ~ A ) )` | `A` |

Rules come in introduction/elimination pairs, one pair per connective. That
symmetry is the design principle of natural deduction.

### 1.5 Subproofs

Three rules need a **temporary assumption**. To prove `( A > B )` you suppose
`A`, derive `B`, then *discharge* the supposition. The supposed region is a
**box** (Fitch notation), and depth is written with `|`:

    N1 | ( ~ ( ~ Q ) )  : AS ;                    <- open box, assume this
    N2 | Q              : DN N1 ;                 <- inside the box
    N3 ( ( ~ ( ~ Q ) ) > Q ) : IMPI N1 N2 ;       <- close box, discharge

Line 3 sits at depth 0 and no longer depends on the assumption. The three
discharging rules are `IMPI` (implication), `NEGI` (assume `A`, reach `F`,
conclude `~A` -- reductio ad absurdum), and `ORE` (case split on `A v B`,
requiring two boxes that both reach the same conclusion).

**Scope rule.** From inside a box you may cite lines outside it. Once a box
closes, its interior is sealed: you may cite the box as a whole, never a line
within it. Tracking this is the only non-local bookkeeping in the format.

### 1.6 Proof length, and why it is the difficulty axis

The **length** of a proof is its number of lines, premises included.

This is the quantity the whole project is organised around, for a specific
reason. A proof is a chain of steps. If a model emits each step correctly with
probability p, an n-step proof is correct with probability roughly p^n --
errors compound multiplicatively. Length is therefore a difficulty measure that
is (a) an integer, (b) reported by the verifier, (c) impossible to argue about.

**A theorem does not have a length; a proof does.** The same theorem may admit a
12-line proof and a 4-line one. This distinction is not pedantic: I generated
30,000 theorems using proofs of 9-16 lines and found that **26,941 of them
(90%) are provable in 6 lines or fewer** (`scripts_build_hard.py`). Treating
generating length as difficulty would have produced a large meaningless result.

### 1.7 The verifier

`nd_verify.verify_text(text)` returns `(ok, reason, n_lines)`. It parses the
token string, replays the box bookkeeping, and checks every line's rule against
its cited lines. It is a decision procedure: total, exact, fast, and with no
notion of "nearly correct".

This single property -- **a free, perfect, automatic judge** -- is what makes
the whole experiment possible, and Section 7 explains why.

---

## 2. The datasets: how each one is constructed

Six datasets, three constructions.

### 2.1 The generator (`nd/gen.py`)

The central design decision. There are two ways to make training data:

1. sample a *theorem*, then search for a proof of it -- needs a prover, and can
   produce unprovable theorems;
2. sample a *proof*, and read off which theorem it proves.

I do (2). Start from random premises; repeatedly apply a randomly chosen
*applicable* rule; whatever the final line says is a theorem you have just
proved. **Soundness is free by construction**: there is no search and no
possibility of an invalid proof.

The state is `(lines, box stack)`, mirroring the verifier's bookkeeping
(`nd/proofstate.py`). At each step the generator enumerates every rule
instantiation whose precondition is satisfied by the currently-citable lines,
and samples among them.

Two properties had to be engineered:

- **Rule coverage.** A first version chose a rule and *then* tested whether it
  applied, so rarely-applicable rules starved: `ORE` appeared 35 times in 20,000
  proofs. Enumerating applicable instantiations first took `IMPE` from 227 to
  2159 and `NEGE` from 457 to 1512.
- **Non-trivial premises.** Uniformly random premises rarely contain a
  top-level disjunction or a matching implication/antecedent pair, so the walk
  had nothing to work with. About 55% of proofs now start from a structured
  pattern such as `[(A>B), A]` or `[(A v B), (~A)]`.

Subproof rules cannot arise from a step-at-a-time walk (they require a box to
have been opened earlier with exactly the right hypothesis), so they are emitted
as **macros** that plan the whole box: assume, derive, discharge. `ORE` is
hardest -- both branches must land on the *same* formula -- so the generator
samples each branch, intersects the sets of formulas they reach, and truncates
both to a shared conclusion.

Measured: **18,396 of 18,396** generated proofs pass the real verifier, zero
invalid, about 5,400 proofs/second.

### 2.2 The six sets

| set | n | construction | role |
|---|---|---|---|
| `data/train.jsonl` | 116,316 | generator, proofs of 2-6 lines | supervised training |
| `data/heldout.jsonl` | 3,000 | same generator, **disjoint by theorem** | in-distribution generalisation |
| `data/rl_targets_hard.jsonl` | 1,994 | generated at 9-16 lines, then filtered | RL samples against these |
| `data/transfer_hard.jsonl` | 800 | same construction, **never sampled by RL** | out-of-sample generalisation |
| `targets/validation_36.jsonl` | 36 | **written by the task authors** | external benchmark |
| `targets/test_*.jsonl` | 267 + 532 | **written by the task authors** | external benchmark, scored once |

"Disjoint by theorem" means a sequent never appears on both sides of a split.

**The hard-pool filter.** Generating length overstates difficulty (Section 1.6),
so every candidate is probed with 24 samples from a Stage-1 model and dropped if
any sample is a verified proof of <= 6 lines. Survivors are labelled *"no
<=6-line proof found in 24 samples"* -- an upper bound, exactly the epistemic
status of the repo's own `min_lines_ub` field, **not** a proof of hardness.

**Distribution.** Train and held-out are different draws from the same
generator: different theorems, same *kind* of theorem. `validation_36` and the
test set are not from that generator at all -- a human selected them from a
logic textbook. That difference is the single most important fact in the
results (Section 9).

---

## 3. Tokens: turning proofs into integers

A neural network consumes vectors of numbers, so the proof text must be mapped
to a sequence of integers. The map is a design choice with measurable
consequences.

### 3.1 The vocabulary

65 symbols:

    special      <pad> <bos>                                     2
    formula      ( ) ~ & v > P Q R S F                          11
    structural   THM , SEQ PRF QED | : ;                         8
    rules        ANDI ANDE1 ANDE2 IMPE IMPI ORI1 ORI2 ORE       15
                 NEGE NEGI BOTE DN PR AS R
    references   B1..B24                                        24
                 P1..P6                                          6

Each token is an index into this list. A proof becomes a list of integers.

### 3.2 Absolute vs relative references

The spec numbers lines absolutely: `N5 ( R v S ) : ORI2 N1 ;`. Under a 6-line
training cap the tokens `N7`, `N8`, ... **never appear in training**, so at test
time on a 12-line proof the model must emit symbols it has no signal for.

So I delete the line index entirely (lines are `;`-delimited, so position is
recoverable) and rewrite each reference:

- a reference to an earlier **derived** line becomes `B<k>`, where k is the
  number of lines back;
- a reference to a **premise** becomes `P<i>`, the i-th premise.

Worked example (a real training record):

    spec   : N1 ( ( S & P ) > ( P & R ) ) : PR ;
             N2 Q : PR ;
             N3 ( ( Q > Q ) v R ) : PR ;
             N4 ( R v ( ( S & P ) > ( P & R ) ) ) : ORI2 N1 ;

    model  : ( ( S & P ) > ( P & R ) ) : PR ;
             Q : PR ;
             ( ( Q > Q ) v R ) : PR ;
             ( R v ( ( S & P ) > ( P & R ) ) ) : ORI2 P1 ;

Line 4 cites line 1, which is a premise, so `N1` becomes `P1`.

Why premises get their own scheme: they sit at the start and are cited from
anywhere, so their back-distance grows without bound (citing line 1 from line 14
would be `B13`, as unseen as `N14`). Derived lines are usually cited soon after
they are produced, so back-distances stay small.

**Measured effect.** Over 9-16 line reference proofs, the fraction of
index/reference tokens that never occur in cap-6 training data:

| scheme | unseen fraction |
|---|---|
| absolute `N<i>` | **43.7%** |
| relative `B<k>` / `P<i>` | **8.1%** |

A 5.4x reduction, **not** elimination: training only ever exercises `B1`-`B4`
and `P1`-`P3`, while long proofs genuinely need `B5`-`B14`.

Everything is decoded back to spec format before the verifier sees it, and the
round trip is exact on 5,000 proofs (`validate_claims.py`).

---

## 4. The model: architecture and the forward pass

A **decoder-only transformer**. "Decoder-only" means it consumes a token
sequence and, at every position, outputs a probability distribution over what
the *next* token is, with each position able to see only earlier positions.

Configuration: 4 layers, `d_model = 256`, 4 attention heads (head dimension 64),
vocabulary 65, **3.18M parameters**.

### 4.1 Shapes, end to end

For a batch of `B` sequences of length `T`:

    ids                    [B, T]              integers in 0..64
      |  embedding lookup: a [65, 256] table
    h                      [B, T, 256]         one 256-vector per token
      |  positional information (Section 4.3)
      |
      |  --- repeat 4 times ---------------------------------
      |    h_norm = LayerNorm(h)                [B, T, 256]
      |    Q, K, V = h_norm @ W_qkv             [B, T, 768] -> split into 3
      |      reshaped per head                  [B, 4, T, 64]
      |    scores = Q @ K^T / sqrt(64)          [B, 4, T, T]
      |    scores += causal mask                (-inf above the diagonal)
      |    A = softmax(scores)                  [B, 4, T, T]
      |    attn = A @ V                         [B, 4, T, 64]
      |    h = h + merge_heads(attn) @ W_o      residual connection
      |    h = h + MLP(LayerNorm(h))            MLP: 256 -> 1024 -> GELU -> 256
      |  ------------------------------------------------------
      |
    h = LayerNorm(h)                            [B, T, 256]
    logits = h @ E^T                            [B, T, 65]

`E` is the same embedding table used at the input -- **tied embeddings**. One
matrix serves both directions, saving 65x256 parameters and tying a token's
input representation to the direction that predicts it.

### 4.2 What the pieces do

- **Attention.** `softmax(QK^T/sqrt(d))V` is a weighted average of the value
  vectors, where the weight from position i to position j is how much `Q_i`
  aligns with `K_j`. It is the mechanism by which line 9 can depend on line 2.
- **Causal mask.** Adding `-inf` above the diagonal makes `softmax` assign zero
  weight to future positions. This is what lets one forward pass produce a
  prediction at *every* position simultaneously while remaining a valid
  left-to-right model.
- **Residual connections** (`h = h + ...`) keep gradients flowing through depth.
- **LayerNorm** normalises each position's vector to zero mean and unit
  variance, stabilising optimisation.
- **MLP** is a per-position nonlinearity; attention moves information between
  positions, the MLP transforms it in place.

### 4.3 Positional information

Attention as written is permutation-invariant: `QK^T` does not know which token
came first. Position must be injected. Three options, all of which I trained:

- **`learned`** (GPT-2 style): a second `[block, 256]` table, one trainable
  vector per absolute position, added to the token embedding. Rows beyond the
  longest training sequence (194 tokens) never receive a gradient.
- **`rope`** (rotary): no parameters. Before attention, `Q` and `K` are rotated
  by an angle proportional to their position; the algebra makes `Q_m . K_n`
  depend only on the offset `m - n`. A gap of 3 looks the same at position 10 or
  300.
- **`nope`**: inject nothing. A causal decoder can still infer order because
  position i attends over a strictly larger prefix than position i-1.

### 4.4 Generation

At inference: feed the prompt, take the logits at the last position, choose a
token, append, repeat until `QED`. A **KV cache** stores each layer's `K` and
`V` so that step t costs one new row of attention instead of recomputing the
whole prefix.

---

## 5. Targets, loss, and optimiser

### 5.1 Targets

A **target** is simply the correct answer at a training position. Because this
is next-token prediction, the target at position i is the token at position
i+1. Construct inputs and targets by shifting the sequence by one:

    tokens : <bos> THM ( P > Q ) , P SEQ Q PRF ... QED
    x      : <bos> THM ( P > Q ) , P SEQ Q PRF ...
    y      :       THM ( P > Q ) , P SEQ Q PRF ... QED

Then **mask the prompt**: set `y = -100` at every position inside the prompt, a
sentinel meaning "contribute nothing to the loss". The prompt is the
*condition*, not something we want the model to spend capacity generating. In a
121-token example with a 46-token prompt, 75 positions are supervised.

### 5.2 Loss

Standard **cross-entropy**:

    loss = -(1/N) * sum over unmasked positions i of  log p(y_i | x_1..i)

where `p` is the softmax over the 65 logits and `N` is the number of unmasked
positions. Minimising this is exactly maximum-likelihood: make the observed
proofs as probable as possible under the model.

A concrete reading: val loss 0.077 means the model assigns on average
`exp(-0.077) = 0.926`, about 93%, to the correct next token on held-out proofs.

**Loss is not the metric that matters.** A proof is correct only if *every*
token is right, so a model can have low average loss and still fail. Every
solve rate in this project is measured by the verifier, never by loss.

### 5.3 Optimiser

**AdamW**, learning rate 3e-4, betas (0.9, 0.95), weight decay 0.1, gradient
clipping at norm 1.0, batch 256, 6000 steps, bf16 autocast.

Adam maintains per-parameter running averages of the gradient (first moment)
and its square (second moment), and steps in the direction of the first divided
by the square root of the second -- so parameters with consistently small
gradients still move. The "W" is decoupled weight decay: the shrink-toward-zero
term is applied directly to the weights rather than folded into the gradient.

Learning-rate schedule: linear warmup for 200 steps, then cosine decay to zero.
Warmup avoids large early steps when the moment estimates are still noisy.

Wall-clock: **278 seconds** on one H200.

---

## 6. Reinforcement learning: what it is and which part is used here

### 6.1 The general frame

Reinforcement learning studies an **agent** that takes **actions** in a
**state**, receives a scalar **reward**, and adjusts its **policy** -- the map
from states to action probabilities -- to earn more reward. It is distinguished
from supervised learning by two things: nobody supplies the correct action, and
the data the agent learns from is generated by its own behaviour.

Mapping onto this project:

| RL concept | here |
|---|---|
| policy | the language model `p_theta(proof \| theorem)` |
| action | emitting one token (or, coarsely, one whole proof) |
| state | the theorem plus the tokens emitted so far |
| reward | 1 if `nd_verify` accepts the proof of the prompted sequent, else 0 |
| episode | one sampled proof attempt |

The reward is **verifiable**: computed by a program, exact, free, requiring no
human and no learned reward model. This family is called RLVR, RL with
Verifiable Rewards.

### 6.2 Expert iteration, precisely

I use **expert iteration** (also called rejection-sampling fine-tuning, or
STaR). One **round** is:

    for each of ~2000 target theorems:
        sample k = 32 attempts at temperature 1.0        <- exploration
        run each through the verifier                     <- reward
        keep the attempts with reward 1                   <- rejection sampling
    pool <- pool + kept proofs
    policy <- finetune(Stage-1 weights, pool + 20k Stage-1 examples)

Five rounds, about 135 seconds each.

Three deliberate choices:

- **Restart from the Stage-1 weights every round** rather than continuing from
  the previous policy. Prevents drift and keeps each round's comparison clean.
- **Mix in Stage-1 data** so the model does not forget short proofs.
- **Hindsight relabelling**: a failed attempt is often a valid proof of a
  *different* theorem. Read that theorem off the proof (its `PR` lines are the
  premises, its last line the conclusion), verify, and add it. Filtered against
  every evaluation set including atom-renamings.

**Note what is absent: there is no reward gradient.** No policy-gradient
estimator, no PPO ratio, no advantage. The reward is used only as a *filter* on
which self-generated data is kept. The gradient step is ordinary supervised
cross-entropy on the filtered data.

It is still reinforcement learning in the sense that matters: the training
distribution is produced by the current policy, and reward determines what
survives. The search is done by sampling; the verifier supplies the label; the
gradient makes the discovery permanent.

### 6.3 The frozen control

A copy of the Stage-1 model receives **exactly the same number of attempts per
round** and is never retrained. This is not optional bookkeeping. Because
sampling k times is itself a search, a large fraction of any "RL result" can be
pure sampling. The frozen control isolates that.

Verification that the arms really are matched: at round 1 the policy *is* the
frozen model, and they solve 463 vs 440 of 1994 -- equal within noise. If they
had diverged there, every later comparison would be meaningless.

---

## 7. The metric: L, P, and an honest critique of L - P

### 7.1 Definitions

Fix an evaluation protocol (a sampling budget and temperature). The **robust
frontier** of a model is

    L* = the largest n such that the model produced at least 5 DISTINCT
         verifier-accepted proofs whose written length is exactly n

Three things are load-bearing:

- **written** length -- the length of the proof the model emitted, never the
  length of the proof the theorem was generated with;
- **distinct** -- different (theorem, proof-text) pairs, so one proof resampled
  five times counts once;
- **at least 5** -- so a single lucky long proof is an anecdote, not a
  capability. Stage 1 produced exactly one 8-line proof in ~64,000 samples; its
  frontier is therefore 7, not 8.

Then `P` is the frontier of the pre-RL model, `L` of the post-RL model, and the
score is `L - P`.

### 7.2 Your objection is correct

**`L - P` is a difference, so it can be inflated by making `P` small.** A
deliberately undertrained Stage-1 model would have a lower frontier, leaving RL
more headroom, and would score *better* on this metric. That is a real perverse
incentive, and no amount of framing removes it.

Three things constrain it rather than solve it:

1. **The cap is fixed by the rules.** Supervised data may not exceed 6 lines, so
   `P` cannot be inflated in the other direction either.
2. **Baseline quality must be reported.** My Stage-1 scores 97.1% held-out
   greedy against the brief's stated >= 85% reference, so `P = 7` is not a
   depressed baseline. A weak Stage 1 would be visible immediately.
3. **The frozen control makes the comparison within-model.** `L` is compared
   against the *same* model given the *same* budget, not against an arbitrary
   external baseline.

There is also a countervailing force: RL bootstraps *from* the Stage-1 model.
A model too weak to write valid proofs at all gives expert iteration nothing to
filter, so `L` collapses along with `P`. The metric is gameable at the margin,
not arbitrarily.

Worth being equally explicit about a second limitation: **`L - P` is a length
metric**. It measures how many proof steps a model can chain, not whether it can
find proofs of theorems it has not seen the shape of. Section 9 shows these come
apart sharply.

### 7.3 Why longer proofs are a meaningful signal at all

Every supervised example has <= 6 lines. A verified 9-line proof is therefore an
object with **no instance in the training set**. It is not interpolation between
seen examples: the model must compose nine correct rule applications, with
correct box bookkeeping throughout, having only ever been shown six.

And it is checkable. There is no argument about whether a 9-line proof is
"really" novel -- the verifier accepted it, and the training data provably
contains nothing of that length (`validate_claims.py` checks the cap).

---

## 8. How RL actually produces longer proofs

This is the mechanism, in the order the causation runs.

**Step 1: the capability is latent, not absent.** After supervised training,
`p(correct 9-line proof | theorem)` is small but strictly positive. It is
positive because the model has learned *local* structure -- which rules apply to
which formulas, how boxes open and close -- and a long proof is a composition of
local steps. Measured: for Stage 1, the cheapest verified 8-line proof has
`-log p = 7.86` nats, i.e. probability about 4e-4.

**Step 2: sampling is search.** Drawing k = 32 samples at temperature 1.0
explores 32 different continuations rather than the single greedy one. Rare
events surface.

**Step 3: the verifier separates signal from noise, for free.** Of 32 attempts,
31 may be wrong. Normally that is fatal -- you cannot train on outputs you
cannot grade. Here `nd_verify` labels them exactly, in microseconds, with no
human. This is the **generator-verifier gap**: checking a proof is easy,
finding one is hard, and RLVR is the family of methods that exploits the
asymmetry.

**Step 4: the gradient step moves probability mass.** Fine-tuning on an accepted
7-line proof increases its log-probability. Crucially, that training example
contains a *decision to continue* at line 6 -- so the gradient directly reduces
`P(QED | 6 lines written)`.

**Step 5: the bootstrap.** A model fluent at 7 lines has non-trivial probability
at 8. Sample, verify, retrain. Round n's output is round n+1's training data.

### 8.1 The measured mechanism

The barrier is a **learned stopping prior**. Teacher-forcing each model along a
*known-good* 9-16 line proof and reading off `P(QED)` at each line boundary --
that is, the model is midway through a valid long derivation and being asked
"stop or continue?":

| after line | Stage 1 | after RL |
|---|---|---|
| 5 | 0.016 | 0.009 |
| 6 | 0.097 | 0.029 |
| 7 | 0.467 | 0.098 |
| 8 | 0.809 | 0.246 |
| 9 | 0.936 | 0.493 |
| 12 | 0.961 | 0.800 |

Stage 1, mid-way through a proof it is being *shown* is valid, still wants to
stop with probability 0.94 at line 9. To write a 10-line proof by sampling it
must decline to stop at lines 6, 7, 8 and 9 -- a product of small numbers, which
is why the ceiling is a hard stop rather than a gradual decay.

RL halved that prior at every line. That is the causal chain from "train on
verified long proofs" to "writes longer proofs", and it is measured, not
asserted (`scripts_qed_prior.py`, figure 1).

### 8.2 A second, independent constraint

Training only ever exercises reference tokens `B1`-`B4`. When the correct next
token is `B5` or beyond, the model assigns it 0.003-0.01 probability and emits
`B4` instead 66-94% of the time -- because cross-entropy pushes every non-target
token down at every position, and `B5`+ was never a target in ~9.6M training
positions.

Consequence: across all **13,378** verified proofs the RL model wrote, it emits
`B5` or beyond **zero times**.

This is not an absolute barrier -- 0.003 is not 0, and a theorem has many proofs,
so a result needed 12 lines later can be *re-derived* one line earlier (`B1`,
probability 0.76) instead of cited from afar (`B12`, probability 0.01). But
re-deriving costs a line, and lines are what the stopping prior blocks. **The
two constraints squeeze from opposite sides**, which is why the observed count
of `B5`+ is exactly zero rather than merely small.

---

## 9. Statistical physics: an exact correspondence, and its limits

### 9.1 Sampling is Boltzmann sampling

Per-token log-probabilities add:

    -log p(y | x)  =  sum over t of  -log p(y_t | x, y_<t)   ==   E(y)

Call that sum the **energy** `E(y)`. Then

    p(y) = exp(-E(y)) / Z

and sampling at temperature tau draws from `p^(1/tau)`, i.e.

    p_tau(y)  proportional to  exp(-E(y) / tau)

This is the Boltzmann distribution. The correspondence is exact, not
analogical: temperature is the sampling temperature, energy is the negative
log-likelihood, `Z` is the normaliser.

### 9.2 The reachability criterion

Drawing k samples finds a particular proof with probability `1 - (1-p)^k`, which
is about `k*p` for small p. You expect to find it when `k*p >~ 1`:

    E(y)  <=  log k

`log k` is a **surprisal budget**. The frontier is then

    L*(k)  =  max { L : E_min(L) <= log k }

where `E_min(L)` is the smallest energy among verified L-line proofs.

### 9.3 It predicts the data

Measuring `E_min(L)` directly (`scripts_energy_vs_length.py`):

| L | Stage-1 `E_min` | k needed | after-RL `E_min` | k needed |
|---|---|---|---|---|
| 2-7 | 0.01 - 0.30 | ~1 | 0.01 - 0.13 | ~1 |
| 8 | **7.86** | 2,590 | **0.05** | 1.05 |
| 9 | **24.81** | 5.9e10 | **1.51** | 4.5 |

`E_min` is flat and then **cliffs**. Stage 1 cliffs between 7 and 8; after RL
the cliff has moved past 9.

Sweeping the budget k = 1, 2, 4, ..., 64 and comparing predicted `L*(k)` to the
measured robust frontier: **11 of 12 testable cases match**
(`scripts_frontier_vs_k.py`; k=1 is degenerate since log 1 = 0). The single miss
is the RL model at k=4 -- predicted 8, observed 9, with `E_min(9) = 1.51`
against `log 4 = 1.39`, a 0.12-nat discrepancy.

### 9.4 Three consequences

1. **Sampling is exponentially weak.** k enters only as `log k`. Lifting Stage 1
   from 7 to 8 lines by resampling alone needs k around 2,590 instead of 1. This
   is *why* the frozen control's frontier never moves: over a 64x budget
   increase it stays at 7.
2. **Training is exponentially strong.** It moves `E` directly, and `E` sits in
   an exponent. RL lowered `E_min(8)` by 7.8 nats -- a factor of ~2,400 in
   probability -- at the *same* k. That is the precise sense in which RL beats
   resampling.
3. **The frontier step is a threshold artefact.** `E_min` moves continuously as
   training proceeds; `L*` is an integer maximum over a threshold, so it jumps.
   Reporting `L*` alone makes smooth progress look discontinuous. This is
   Schaeffer et al.'s "mirage" argument about emergent abilities, here in a
   system small enough that both quantities are directly measurable.


### 9.6 Degeneracy: the entropy term, and how the data distribution enters

A theorem is proved if **any** of its proofs is sampled, so the governing
quantity is not the energy of one proof but the free energy of the whole proof
set `Y(x)`:

    p(prove x) = sum over y in Y(x) of exp(-E(y))  =  exp(-F(x))
    F(x) = -log sum_y exp(-E(y))

If the reachable proofs have comparable energy this factorises into an
**energy-entropy decomposition**, with the log-count of proofs playing the role
of entropy:

    F(x)  ~  E  -  log g(x),        g(x) = |Y(x)| = degeneracy

Measured (`scripts_degeneracy.py`, 800 hard targets, k=64, figure 8):

| g | n | mean p | mean E_min |
|---|---|---|---|
| 1 | 180 | 0.432 | 1.763 |
| 2-3 | 126 | 0.718 | 0.674 |
| 4-7 | 45 | 0.818 | 0.818 |
| 8-15 | 26 | 0.718 | 1.342 |
| 16-31 | 34 | 0.813 | 2.313 |
| 32-64 | 35 | **0.911** | **3.192** |

**The entropy term is real and can dominate.** Theorems with a single findable
proof have the *lowest* energy (1.76, i.e. that one proof is individually the
most likely) and the *worst* solve rate (0.43). Theorems with 32-64 distinct
proofs have nearly twice the energy -- each proof individually less likely --
and solve at 0.91. Many mediocre proofs beat one good proof, for the same reason
a macrostate with many microstates dominates a partition function.

Fitted slopes: `d log p / d log g = +0.397` and `d log p / d E_min = -0.544`,
against idealised `+1` and `-1`. **Right signs, attenuated magnitudes**, and the
attenuation is understood rather than ignored:

1. `log p = log g - E` is a small-p linearisation. Most solved theorems sit at
   p = 0.4-0.9 where `log p` saturates at 0.
2. The equal-energy assumption is crude. The sum is dominated by the few
   lowest-energy proofs, so the effective degeneracy is
   `g_eff = sum exp(-(E_y - E_min)) <= g`.
3. `g` is censored (distinct proofs found in 64 draws, not `|Y(x)|`) and is
   estimated from the same draws as `p`, which couples them and would *inflate*
   the slope. Attenuation despite that coupling means the true attenuation from
   (1) and (2) is larger than measured.

**This is how the training distribution enters the frontier.** Write the
frontier as a functional of the data distribution `D` and the budget:

    L*(D, k) = max { L : F_min(L; A(D)) <= log k }

for learning algorithm `A`. Emergence is a property of the *pair* (distribution,
budget), not of the model alone. Two structural features of `D` matter through
the entropy term:

- **Degenerate theorems have enormous g.** Insert a reiteration, swap the
  argument order of `ANDI`, add a vacuous `ORI` -- each gives a distinct proof
  text of the same theorem. These are symmetry factors in the entropy count,
  directly analogous to indistinguishable-particle counting. They concentrate
  partition-function mass at *low* L and contribute nothing at high L.
- **Long proofs have low degeneracy.** Fewer ways to do a hard thing. So `F`
  stays high at large L even where `E_min` is moderate, and the frontier stalls.

Concretely: 23.1% of my training theorems are degenerate (16.5% have the
conclusion as a premise, 13.2% have proofs of <= 2 lines, 1.0% have `F` as a
premise). The prediction is that this fraction shapes `p_theta` toward short
proofs -- i.e. it sets where the stopping-prior cliff sits.

**This specific prediction is UNTESTED.** Confirming it requires retraining
Stage 1 on non-trivial-only data and re-measuring `P(QED)` and `P`. It is
stated here as a mechanism with a clear falsification, not as a result.

### 9.5 What does not carry over

The Boltzmann form and the free-energy view (`F(x) = -log sum over proofs of x
of exp(-E)`, which trades energy against entropy -- a theorem with many
mediocre proofs can be easier than one with a single excellent proof) are exact.

**Phase transitions, critical exponents and random-matrix spectra are not
applicable**, and I will not stretch to make them fit. Those need a
thermodynamic limit, an order parameter and a diverging correlation length. Here
the "system size" is proof length, of order 10; there is no lattice and no
criticality. The step in `L*` is a threshold on a smooth curve, **not** a phase
transition.

Similarly, RMT accounts of feature emergence concern the *spectrum of weight
matrices during gradient training*. Everything above concerns the *induced
distribution over outputs* at fixed weights, plus how training moves one scalar.
Different objects; no connection is claimed.

---

## 10. What the experiments found

| claim | status | experiment |
|---|---|---|
| reference codec gates length generalisation | **CONFIRMED** | absolute-index model: frontier 6, **zero** proofs past the cap, vs 352 for relative |
| positional scheme is the barrier | **FALSIFIED** | learned / rope / nope all reach frontier 7 |
| a learned stopping prior throttles the frontier | **SUPPORTED** | P(QED) measured; predicted rope would write the most 7-line proofs, and it did (351 vs 177 vs 129) |
| frontier set by the surprisal budget `E <= log k` | **CONFIRMED** | 11/12 predictions over a k = 1..64 sweep |
| generating length is not difficulty | **MEASURED** | 26,941 / 30,000 "9-16 line" theorems provable in <= 6 |
| gains do not transfer off-distribution | **MEASURED** | transfer 5.9% -> 25.8%; validation_36 and test flat |
| overlap is not memorisation | **MEASURED** | decontaminated model solves the identical 6 validation theorems; test_short 46.8% -> 47.6% |

Headline, two seeds, decontaminated:

    P = 7        (Stage 1, robust frontier at k=32, T=1.0)
    L = 9, 10    (after RL, seeds 0 and 1)
    L - P = 2-3, with the frozen control never leaving 7

and, on external benchmarks, no movement at all: `validation_36` bin `>6` stays
0/24 before and after, `test_long` 8.8% -> 8.1%.

The two halves are the result. RL lowered the surprisal of long proofs on the
distribution it trained against. The binding constraint on textbook theorems is
different -- **78-90% of all greedy failures, before and after RL, are lines
citing a rule whose premises do not hold**, i.e. per-line rule-application
accuracy on unfamiliar formulas, which nothing in this loop improved.
