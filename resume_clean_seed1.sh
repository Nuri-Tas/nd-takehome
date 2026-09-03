#!/bin/bash
cd /home/nuri/nd-takehome || exit 1
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=1
.venv/bin/python rl_expert_iteration.py --ckpt ckpt/sft_rope_clean.pt \
    --rl_data data/rl_targets_hard.jsonl --transfer_data data/transfer_hard.jsonl \
    --n_targets 2000 --n_transfer 800 --rounds 5 --k 32 --block 512 \
    --relabel --out runs/clean_seed1 --seed 1 > clean_seed1.log 2>&1
echo "SEED1 COMPLETE" >> clean_seed1.log
