#!/bin/bash
# Full re-measurement on the decontaminated Stage-1 checkpoint.
set -e
export OMP_NUM_THREADS=8
CK=ckpt/sft_rope_clean.pt
GPU=${1:-1}
P=".venv/bin/python"

echo "=== 1. Stage-1 held-out, by length ==="
CUDA_VISIBLE_DEVICES=$GPU $P eval_heldout.py --ckpt $CK --n 1500 --reasons 2>/dev/null

echo; echo "=== 2. Stage-1 validation_36 (now clean) ==="
CUDA_VISIBLE_DEVICES=$GPU $P eval_validation.py --ckpt $CK 2>/dev/null

echo; echo "=== 3. P: frontier probe on hard targets ==="
CUDA_VISIBLE_DEVICES=$GPU $P scripts_probe_length.py --ckpts clean=$CK \
    --targets data/rl_targets_hard.jsonl --n 2000 --k 32 --bs 2000 \
    --max_new 380 --out numbers_P_clean.json 2>/dev/null

echo; echo "=== 4. expert iteration, seed 0 ==="
CUDA_VISIBLE_DEVICES=$GPU $P rl_expert_iteration.py --ckpt $CK \
    --rl_data data/rl_targets_hard.jsonl --transfer_data data/transfer_hard.jsonl \
    --n_targets 2000 --n_transfer 800 --rounds 5 --k 32 --block 512 \
    --relabel --out runs/clean_seed0 --seed 0 2>/dev/null

echo; echo "=== 5. Stage 3 table ==="
CUDA_VISIBLE_DEVICES=$GPU $P eval_all.py \
    --ckpts stage1=$CK final=runs/clean_seed0/final.pt \
    --transfer data/transfer_hard.jsonl --passk 32 --n_heldout 1500 \
    --out numbers_stage3_clean.json 2>/dev/null

echo; echo "=== 6. TEST SET, one greedy run per file per model ==="
for spec in "stage1:$CK" "final:runs/clean_seed0/final.pt"; do
  nm=${spec%%:*}; ck=${spec##*:}
  for f in short long; do
    CUDA_VISIBLE_DEVICES=$GPU $P prove.py --ckpt $ck \
       --in targets/test_${f}_prompts.jsonl --out test_${f}_${nm}_clean.jsonl --greedy 2>/dev/null
    printf "%-7s %-5s  " "$nm" "$f"; $P score_test.py test_${f}_${nm}_clean.jsonl
  done
done
echo; echo "=== RERUN COMPLETE ==="
