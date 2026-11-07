#!/usr/bin/env bash
# Full fine-tuning of pi0.5 on the SO-101 ball-bowl demos, starting from Project-IRA's 008000.
# train_config.json is the config LeRobot saved with the 010000 checkpoint (batch 16, 10k steps,
# episodes 0-39, AdamW peak lr 2.5e-5, 1k warmup, cosine schedule planned over 30k steps).
# Only the machine-specific paths are overridden here; they are placeholders.
#
# Run make_ira_train.py first: it writes so101_ira_train.py, a copy of the LeRobot 0.6.1 trainer
# that keeps the base checkpoint's normalization stats instead of recomputing them from this dataset.
set -euo pipefail
export HF_HUB_OFFLINE=1

BASE=/path/to/Project-IRA/pi05_6gpu_fsdp_V2/checkpoints/008000/pretrained_model
DATA=/path/to/so101_ball_bowl_pilot

python so101_ira_train.py \
  --config_path=train_config.json \
  --policy.path="$BASE" \
  --dataset.root="$DATA" \
  --output_dir=runs/ira_ball_batch16
