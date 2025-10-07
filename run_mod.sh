#! /bin/bash

# Check if --resume parameter is provided
if [[ "$*" == *"--resume"* ]]; then
    # Auto-detect latest checkpoint for resume
    RESUME_FROM=$(ls -lt logs/*/*.pt | head -n 1 | awk '{print $9}')
    echo "RESUME_FROM: $RESUME_FROM"
    RESUME_CMD="--resume_from $RESUME_FROM"
else
    echo "No --resume specified, starting fresh"
    RESUME_CMD=""
fi

torchrun --standalone --nproc_per_node=1 train_gpt2_mod.py \
  $RESUME_CMD \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --output_dir pylog124M \
  --model d12_mlp3x \
  --batch_size 12 \
  --grad_accumulation_steps 42 \
  --sequence_length 1024 \
  --val_loss_every 128 \
  --val_batch_size 8 \
  --num_iterations 4768 \
  --weight_decay 0.1 \
  --learning_rate 0.0018 \
  --warmup_iters 256 \
  --warmdown_iters 1024 \
  --log_wandb

  # --model d12_post_norm_qk_norm \