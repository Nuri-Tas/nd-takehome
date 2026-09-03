
## Setup

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu128 torch
uv pip install --python .venv/bin/python numpy tqdm
```

`nd/model.py` disables the cuDNN fused-attention backend at import
(`torch.backends.cuda.enable_cudnn_sdp(False)`). On this driver that backend
both crashes at batch 2048 and runs ~60x slower than the alternatives; see
`log.md`. Leave it disabled.


## Our Hardware
Hardware: 1x NVIDIA H200 
Driver CUDA 12.8, torch 2.11+cu128, Python 3.11.

The machine has 192 cores and torch thread-thrashes on it, so **every command
below sets `OMP_NUM_THREADS=8`**. Without it, dataset encoding and sampling run
an order of magnitude slower.


## Stage 1 -- data and supervised training

```bash
# 120k distinct theorems, proofs of length 2-6, theorem-disjoint split
OMP_NUM_THREADS=8 .venv/bin/python scripts_build_data.py 120000
#   -> data/train.jsonl (117k), data/heldout.jsonl (3k)
#   prints: trivial fraction, held-out->train atom-renaming overlap

# 3.26M-param decoder, 6000 steps at batch 256  (~4 min on one H200)
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 .venv/bin/python train_sft.py \
    --steps 6000 --bs 256 --out ckpt/sft.pt --seed 0

# positional-scheme variants (see writeup: the second length barrier)
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 .venv/bin/python train_sft.py \
    --steps 6000 --bs 256 --block 512 --pos_mode rope --out ckpt/sft_rope.pt --seed 0
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 .venv/bin/python train_sft.py \
    --steps 6000 --bs 256 --block 512 --pos_mode nope --out ckpt/sft_nope.pt --seed 0
```

## Stage 1 -- baseline numbers

```bash
# held-out greedy solve rate, by proof length, with Wilson CIs + failure reasons
OMP_NUM_THREADS=8 .venv/bin/python eval_heldout.py --ckpt ckpt/sft.pt --n 1500 --reasons

# validation_36, split by bin, with per-theorem verdicts
OMP_NUM_THREADS=8 .venv/bin/python eval_validation.py --ckpt ckpt/sft.pt --show
```

## Stage 2 -- target pools

```bash
# generate beyond-cap candidates, then DROP any with a found <=6-line proof.
# Generating length is only an upper bound on the shortest proof, so it cannot
# be used as difficulty -- see log.md for the 2706/4000 result that forced this.
OMP_NUM_THREADS=8 .venv/bin/python scripts_build_hard.py \
    --candidates 30000 --probe_k 32 --bs 2048 --n_rl 2000 --n_transfer 800

# bucket by shortest_found into curriculum rungs (7, 8, ...) + unsolved stretch
OMP_NUM_THREADS=8 .venv/bin/python scripts_bucket_pool.py
#   -> data/rl_targets_final.jsonl, data/transfer_final.jsonl (disjoint)
```

## Stage 2 -- how long a proof can each Stage-1 model write? (measures P)

```bash
OMP_NUM_THREADS=8 .venv/bin/python scripts_probe_length.py \
    --ckpts learned=ckpt/sft.pt rope=ckpt/sft_rope.pt nope=ckpt/sft_nope.pt \
    --k 32 --out numbers_length_probe.json
```

## Stage 2 -- expert iteration with the frozen control

```bash
OMP_NUM_THREADS=8 .venv/bin/python rl_expert_iteration.py \
    --ckpt <best stage-1 ckpt> --rounds 4 --k 32 --relabel \
    --out runs/ei_seed0 --seed 0
OMP_NUM_THREADS=8 .venv/bin/python rl_expert_iteration.py \
    --ckpt <best stage-1 ckpt> --rounds 4 --k 32 --relabel \
    --out runs/ei_seed1 --seed 1
```

Each round logs, to `runs/*/log.json`: RL-target and transfer solve rates, the
frozen control at an identical sampling budget, the written-length histogram,
the robust frontier (longest written length with >= 5 distinct verified
proofs), and Stage-1 held-out greedy to catch in-distribution regression.

## Stage 3 -- the table

```bash
OMP_NUM_THREADS=8 .venv/bin/python eval_all.py \
    --ckpts sft=ckpt/sft.pt final=runs/ei_seed0/final.pt \
    --out numbers_stage3.json
```

## Stage 3 -- test set (run ONCE)

```bash
.venv/bin/python prove.py --ckpt <ckpt> --in targets/test_short_prompts.jsonl \
    --out test_short_out.jsonl --greedy
.venv/bin/python score_test.py test_short_out.jsonl

.venv/bin/python prove.py --ckpt <ckpt> --in targets/test_long_prompts.jsonl \
    --out test_long_out.jsonl --greedy
.venv/bin/python score_test.py test_long_out.jsonl
```

No per-theorem inspection of the test set, no tuning against it.

---

## Repository map

| path | what |
|---|---|
| `PRIMER.md` | **start here if the domain is unfamiliar** -- proofs, datasets, tokens, architecture, loss, optimiser, RL, the metric, and the statistical-physics formulation, built from scratch with no assumed background |
| `writeup.md` | the report: executive summary, per-stage method, limitations |
| `numbers.md` | every number in the write-up, with its source file |
| `log.md` | dated log, in order, including dead ends and corrections |
| `nd/gen.py`, `nd/proofstate.py` | proof generator (forward random walk over a Fitch state machine) |
| `nd/tokenizer.py` | relative-reference codec (`B<k>` / `P<i>`) + exact round-trip |
| `nd/model.py` | 3.2M decoder, KV cache, `learned` / `rope` / `nope` positional modes |
| `nd/relabel.py` | hindsight relabelling with contamination filtering |
| `train_sft.py` | Stage 1 supervised training |
| `rl_expert_iteration.py` | Stage 2 expert iteration + frozen control |
| `ablation_abs_refs.py`, `probe_absrefs.py` | H1 ablation: absolute vs relative references |
| `scripts_qed_prior.py` | H3: P(QED) mid-proof, the stopping prior |
| `scripts_energy_vs_length.py` | surprisal E(y) vs proof length |
| `scripts_frontier_vs_k.py` | frontier vs sampling budget |
| `eval_all.py`, `eval_heldout.py`, `eval_validation.py` | evaluation harnesses |
| `prove.py` | required submission interface |
| `README_ORIGINAL_TASK.md` | the original task README, kept verbatim |

## Additional experiments (beyond the required stages)

```bash
# H1: does the reference codec gate length generalisation?  (~6 min + probe)
OMP_NUM_THREADS=8 .venv/bin/python ablation_abs_refs.py
OMP_NUM_THREADS=8 .venv/bin/python probe_absrefs.py

# H3: the stopping prior, P(QED) at each line boundary
OMP_NUM_THREADS=8 .venv/bin/python scripts_qed_prior.py \
    --ckpts learned=ckpt/sft.pt rope=ckpt/sft_rope.pt nope=ckpt/sft_nope.pt

# surprisal of verified proofs vs their length
OMP_NUM_THREADS=8 .venv/bin/python scripts_energy_vs_length.py

# frontier as a function of sampling budget k
OMP_NUM_THREADS=8 .venv/bin/python scripts_frontier_vs_k.py 64

# all figures
.venv/bin/python make_figures.py
```
