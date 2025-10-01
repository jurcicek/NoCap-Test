# Checkpoint Resumption Guide

## Overview

Training can now be resumed from any saved checkpoint. This is useful when:
- Training gets interrupted
- You want to train for more iterations
- You need to continue from a specific point

## What's Saved in Checkpoints

Each checkpoint now contains:
- `model`: Model state dict
- `optimizer`: Optimizer state dict  
- `step`: Training step number
- `code`: Training script code
- `args`: Training arguments
- `run_id`: Unique run identifier

## Checkpoint Locations

Checkpoints are saved in `logs/<run_id>/`:
- **Periodic checkpoints**: `model_step000256.pt`, `model_step000512.pt`, etc.
  - Saved every `--save_every` steps (default: 5000)
- **Final checkpoint**: `final.pt`
  - Saved at the end of training

## How to Resume Training

### 1. Find Your Checkpoint

```bash
# List available checkpoints
ls -lh logs/
ls -lh logs/<run_id>/
```

### 2. Resume from Checkpoint

Add the `--resume_from` argument to your training command:

```bash
torchrun --standalone --nproc_per_node=1 train_gpt2_mod.py \
  --resume_from logs/<run_id>/model_step000256.pt \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --output_dir pylog124M \
  --model d12 \
  --batch_size 12 \
  --sequence_length 1024 \
  --num_iterations 4768 \
  --learning_rate 0.0018 \
  --log_wandb
```

### 3. Resume from Final Checkpoint

```bash
torchrun --standalone --nproc_per_node=1 train_gpt2_mod.py \
  --resume_from logs/<run_id>/final.pt \
  --num_iterations 10000 \
  --other_args...
```

## Important Notes

### ✅ What Happens When Resuming

1. **Model state restored**: Continues with the same weights
2. **Optimizer state restored**: Maintains momentum/Adam states
3. **Training continues**: Starts from `step + 1` of the checkpoint
4. **Same run_id**: Logs continue in the same directory
5. **Log file appended**: Existing logs are preserved

### ⚠️ Parameter Validation

The script checks that critical parameters match the checkpoint:
- `model` type (d12, d24, etc.)
- `batch_size`
- `sequence_length`
- `grad_accumulation_steps`

**Warnings are shown if parameters don't match**, but training proceeds.

### 🔄 Training Range

If resuming from step 256 with `--num_iterations 4768`:
- Training will run from step **257 to 4768**
- If you want to train for **additional** iterations, set `--num_iterations` to a higher value than the checkpoint step

### 📊 WandB Logging

When resuming:
- If using `--log_wandb`, logs will continue with the same run_id
- Step numbers in WandB will be consistent
- Previous training data is preserved

## Examples

### Example 1: Training Got Interrupted

```bash
# Original training (interrupted at step 500)
torchrun --standalone --nproc_per_node=1 train_gpt2_mod.py \
  --model d12 \
  --num_iterations 4768 \
  --save_every 256

# Resume from last checkpoint
torchrun --standalone --nproc_per_node=1 train_gpt2_mod.py \
  --resume_from logs/691bd736-d91a-48bf-a7a9-4b17d3ec0733/model_step000256.pt \
  --model d12 \
  --num_iterations 4768
```

### Example 2: Train for More Iterations

```bash
# Original training completed at step 4768
# Now train for 2000 more iterations (total: 6768)

torchrun --standalone --nproc_per_node=1 train_gpt2_mod.py \
  --resume_from logs/691bd736-d91a-48bf-a7a9-4b17d3ec0733/final.pt \
  --model d12 \
  --num_iterations 6768
```

### Example 3: Resume with Different Hardware

```bash
# Can resume with different number of GPUs
# (batch processing adjusts automatically)

torchrun --standalone --nproc_per_node=2 train_gpt2_mod.py \
  --resume_from logs/<run_id>/model_step001024.pt \
  --model d12 \
  --num_iterations 4768
```

## Troubleshooting

### Issue: "KeyError: 'optimizer'"

**Cause**: Checkpoint was created before checkpoint resumption feature was added.

**Solution**: These old checkpoints cannot be resumed. Start a new training run or wait for the next checkpoint.

### Issue: Parameter Mismatch Warnings

**Cause**: Training arguments differ from the checkpoint.

**Solution**: 
- Review warnings carefully
- Ensure critical parameters match (model, batch_size, sequence_length)
- Some differences (learning_rate, num_iterations) are OK

### Issue: CUDA Out of Memory

**Cause**: Checkpoint may have been created with different hardware.

**Solution**: Adjust `--batch_size` or `--sequence_length` if needed.

## Testing

To verify checkpoint resumption works:

```bash
# 1. Train for a few steps
torchrun --standalone --nproc_per_node=1 train_gpt2_mod.py \
  --model d12 \
  --num_iterations 10 \
  --save_every 5

# 2. Resume from checkpoint
torchrun --standalone --nproc_per_node=1 train_gpt2_mod.py \
  --resume_from logs/<run_id>/model_step000005.pt \
  --model d12 \
  --num_iterations 10

# 3. Check logs show continuation from step 6
```

