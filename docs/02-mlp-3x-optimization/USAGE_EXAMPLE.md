# Quick Usage Example: d12_mlp3x

## Quick Start

Train with the optimized 3× MLP expansion configuration:

```bash
# Activate environment
source activate_env.sh

# Run training with d12_mlp3x
torchrun --standalone --nproc_per_node=1 train_gpt2_mod.py \
    --model=d12_mlp3x \
    --batch_size=8 \
    --sequence_length=1024 \
    --num_iterations=5000 \
    --val_loss_every=250 \
    --learning_rate=1e-4 \
    --warmup_iters=100 \
    --output_dir=logs \
    --save_every=500
```

## Key Configuration Details

The `d12_mlp3x` model has:

- **n_embd:** 832 (vs 768 in baseline)
- **n_head:** 13 (vs 12 in baseline)
- **head_dim:** 64 (same as baseline)
- **mlp_expansion_factor:** 3 (vs 4 in baseline)
- **MLP hidden dim:** 2496 = 3 × 832 (vs 3072 = 4 × 768 in baseline)
- **Total parameters:** ~125M (similar to baseline ~124M)

## Expected Performance

Based on theoretical analysis:

### Speed
- **~15% faster training** per step
- **~25% fewer MLP FLOPs**
- **Faster gradient computation** due to smaller matrices

### Memory
- **~10-15% lower peak memory** usage
- **Smaller activations** in MLP layers
- Can fit larger batch sizes or longer sequences

### Quality
- **Similar validation loss** to baseline d12
- Compensated capacity through larger n_embd
- Enhanced with post_norm and qk_norm

## Comparison Test

Compare baseline vs optimized:

```bash
# Train baseline d12 for comparison
torchrun --standalone --nproc_per_node=1 train_gpt2_mod.py \
    --model=d12 \
    --batch_size=8 \
    --sequence_length=1024 \
    --num_iterations=2000 \
    --val_loss_every=250 \
    --output_dir=logs/d12_baseline

# Train optimized d12_mlp3x
torchrun --standalone --nproc_per_node=1 train_gpt2_mod.py \
    --model=d12_mlp3x \
    --batch_size=8 \
    --sequence_length=1024 \
    --num_iterations=2000 \
    --val_loss_every=250 \
    --output_dir=logs/d12_mlp3x

# Compare the step_avg times in logs to measure speedup
# Compare validation losses to verify quality
```

## Custom Configurations

You can create custom configurations with different expansion factors:

### Example: 2× expansion with even larger n_embd

```python
# Add to base_configs in train_gpt2_mod.py
"d12_mlp2x": GPTConfig(
    vocab_size=num_vocab,
    n_layer=12,
    n_head=15,  # 960 / 15 = 64 head_dim
    n_embd=960,  # Larger to compensate for even smaller MLP
    mlp_expansion_factor=2,
    post_norm=True,
    qk_norm=True
)
```

### Example: 3× expansion with standard n_embd (no compensation)

```python
"d12_mlp3x_no_comp": GPTConfig(
    vocab_size=num_vocab,
    n_layer=12,
    n_head=12,
    n_embd=768,  # Keep standard size
    mlp_expansion_factor=3,
    post_norm=True,
    qk_norm=True
)
```

This will be faster but may have slightly worse quality.

## Multi-GPU Training

The configuration works seamlessly with DDP:

```bash
# 4 GPU training
torchrun --standalone --nproc_per_node=4 train_gpt2_mod.py \
    --model=d12_mlp3x \
    --batch_size=8 \
    --sequence_length=1024 \
    --num_iterations=10000 \
    --val_loss_every=500 \
    --output_dir=logs/d12_mlp3x_4gpu
```

## WandB Logging

Track experiments with Weights & Biases:

```bash
torchrun --standalone --nproc_per_node=1 train_gpt2_mod.py \
    --model=d12_mlp3x \
    --batch_size=8 \
    --sequence_length=1024 \
    --num_iterations=5000 \
    --val_loss_every=250 \
    --log_wandb \
    --output_dir=logs/d12_mlp3x_wandb
```

## Troubleshooting

### Out of Memory
Try reducing batch size or sequence length:
```bash
--batch_size=4 --sequence_length=512
```

### Slow First Steps
This is normal - PyTorch compilation takes time on first run. Performance will improve after ~10 steps.

### No Speedup Observed
1. Make sure torch.compile is enabled (it is by default)
2. Verify you're using bfloat16 (automatic in the script)
3. Measure after warmup (skip first 10-20 steps)

## Verification Test

Run the test suite to verify your installation:

```bash
python docs/02-mlp-3x-optimization/test_mlp3x.py
```

Expected output:
```
======================================================================
All tests passed! ✅
======================================================================

You can now train with: --model=d12_mlp3x
```

## Next Steps

After validating d12_mlp3x works well:

1. **Combine with GQA** for even more speedup (~30% total)
2. **Try SwiGLU** with 3× expansion for better convergence
3. **Experiment with 2× expansion** if you need maximum speed
4. **Profile your specific workload** to measure actual speedup

Happy training! 🚀

