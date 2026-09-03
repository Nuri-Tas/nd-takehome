#!/bin/bash
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=${1:-1}
CK=ckpt/sft_rope_clean.pt
P=".venv/bin/python"
LOG=clean_pipeline.log
{
echo "=== 4. expert iteration, seed 0 ==="
$P rl_expert_iteration.py --ckpt $CK \
    --rl_data data/rl_targets_hard.jsonl --transfer_data data/transfer_hard.jsonl \
    --n_targets 2000 --n_transfer 800 --rounds 5 --k 32 --block 512 \
    --relabel --out runs/clean_seed0 --seed 0 2>/dev/null

echo "=== 5. Stage 3 table ==="
$P eval_all.py --ckpts stage1=$CK final=runs/clean_seed0/final.pt \
    --transfer data/transfer_hard.jsonl --passk 32 --n_heldout 1500 \
    --out numbers_stage3_clean.json 2>/dev/null

echo "=== 6. TEST SET, one greedy run per file per model ==="
for spec in "stage1:$CK" "final:runs/clean_seed0/final.pt"; do
  nm=${spec%%:*}; ck=${spec##*:}
  for f in short long; do
    $P prove.py --ckpt $ck --in targets/test_${f}_prompts.jsonl \
       --out test_${f}_${nm}_clean.jsonl --greedy 2>/dev/null
    printf "%-7s %-5s  " "$nm" "$f"; $P score_test.py test_${f}_${nm}_clean.jsonl
  done
done
echo "=== PIPELINE COMPLETE ==="
} > $LOG 2>&1
