#!/usr/bin/env python3
"""
Memory and performance comparison script for SlidingWindowAttention vs CausalSelfAttention

This script tests the updated SlidingWindowAttention implementation that uses
scaled_dot_product_attention with attention masks instead of flex_attention with block masks.

Features:
- Memory usage comparison between standard and sliding window attention
- Performance benchmarking
- Sliding window attention mask validation
- Layer-level memory analysis

Usage: python test_memory.py
"""

import torch
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from train_gpt2_mod import (
    GPTConfig, GPT
)

# set up a context manager following the desired dtype and device
ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)


def measure_attention_memory(attention_type='standard', batch_size=4, seq_len=1024):
    """Measure memory usage for different attention mechanisms."""
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Create config
    if attention_type == 'sliding_window':
        config = GPTConfig(
            vocab_size=50257, n_layer=6, n_head=6, n_embd=384,
            attention_type='sliding_window', window_size=64,
            post_norm=True, qk_norm=True
        )
    else:
        config = GPTConfig(
            vocab_size=50257, n_layer=6, n_head=6, n_embd=384,
            attention_type='standard'
        )
    
    print(f"{'='*60}")
    print(f"Creating model for {attention_type.upper()} attention (B={batch_size}, T={seq_len})")
    print(f"{'='*60}")
    # Create model
    model = GPT(config).cuda().to(dtype=torch.bfloat16).train()
    
    # Create input
    x = torch.randint(0, 50257, (batch_size, seq_len), device='cuda')
    y = torch.randint(0, 50257, (batch_size, seq_len), device='cuda')
    
    # Warmup
    print(f"\n{'='*60}")
    print(f"Testing {attention_type.upper()} attention (B={batch_size}, T={seq_len})")
    print(f"{'='*60}")
    
    mem_before = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"Memory before forward: {mem_before:.2f} MB")
    
    # Forward pass
    with ctx:
        logits, loss = model(x, y, return_logits=True)
    
    mem_after_fwd = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"Memory after forward: {mem_after_fwd:.2f} MB (+{mem_after_fwd - mem_before:.2f} MB)")
    
    # Backward pass
    loss.backward()
    
    mem_after_bwd = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"Memory after backward: {mem_after_bwd:.2f} MB (+{mem_after_bwd - mem_after_fwd:.2f} MB)")
    
    # Peak memory
    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"Peak memory: {peak_mem:.2f} MB")
    
    # Note: Sliding window attention now uses scaled_dot_product_attention with attn_mask
    # No block mask cache to report
    
    result = {
        'attention_type': attention_type,
        'mem_before': mem_before,
        'mem_after_fwd': mem_after_fwd,
        'mem_after_bwd': mem_after_bwd,
        'peak_mem': peak_mem,
        'fwd_delta': mem_after_fwd - mem_before,
        'bwd_delta': mem_after_bwd - mem_after_fwd,
        'total_delta': mem_after_bwd - mem_before
    }
    
    # Cleanup
    del model, x, y, logits, loss
    torch.cuda.empty_cache()
    
    return result


def compare_memory():
    """Compare memory usage between standard and sliding window attention."""
    
    configs = [
        ('standard', 4, 512),
        ('sliding_window', 4, 512),
        ('standard', 4, 1024),
        ('sliding_window', 4, 1024),
        # ('standard', 4, 2048),
        # ('sliding_window', 4, 2048),
    ]
    
    results = []
    for attn_type, batch_size, seq_len in configs:
        result = measure_attention_memory(attn_type, batch_size, seq_len)
        results.append(result)
    
    # Print comparison
    print(f"\n{'='*80}")
    print("MEMORY COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Attention':<20} {'Peak (MB)':<12} {'Fwd Δ (MB)':<12} {'Bwd Δ (MB)':<12} {'Total Δ (MB)':<12}")
    print(f"{'-'*80}")
    
    for r in results:
        print(f"{r['attention_type']:<20} {r['peak_mem']:>10.2f}  {r['fwd_delta']:>10.2f}  "
              f"{r['bwd_delta']:>10.2f}  {r['total_delta']:>10.2f}")
    
    # Calculate differences
    if len(results) >= 2:
        print(f"\n{'='*80}")
        print("MEMORY OVERHEAD (Sliding Window vs Standard)")
        print(f"{'='*80}")
        
        for i in range(0, len(results), 2):
            if i+1 < len(results):
                std = results[i]
                slide = results[i+1]
                overhead = slide['peak_mem'] - std['peak_mem']
                overhead_pct = (overhead / std['peak_mem']) * 100
                print(f"Batch {std['attention_type']}: +{overhead:.2f} MB ({overhead_pct:.1f}% overhead)")


def test_sliding_window_attention_mask():
    """Test the sliding window attention mask creation."""
    print(f"\n{'='*60}")
    print("SLIDING WINDOW ATTENTION MASK TEST")
    print(f"{'='*60}")
    
    from train_gpt2_mod import create_sliding_window_attn_mask
    
    # Test different window sizes and sequence lengths
    test_cases = [
        (10, 3),   # Short sequence, small window
        (20, 5),   # Medium sequence, small window
        (50, 10),  # Long sequence, medium window
        (100, 20), # Very long sequence, large window
    ]
    
    for seq_len, window_size in test_cases:
        print(f"\nTesting: T={seq_len}, window_size={window_size}")
        
        # Create mask
        mask = create_sliding_window_attn_mask(seq_len, window_size, 'cpu')
        
        # Verify mask properties
        # 1. Should be causal (upper triangular)
        causal_violations = 0
        for i in range(seq_len):
            for j in range(i+1, seq_len):
                if mask[i, j]:  # Should be False above diagonal
                    causal_violations += 1
        
        # 2. Should respect window size
        window_violations = 0
        for i in range(seq_len):
            for j in range(seq_len):
                if abs(i - j) > window_size and mask[i, j]:  # Should be False outside window
                    window_violations += 1
        
        # 3. Count valid attention positions
        valid_positions = mask.sum().item()
        expected_positions = sum(min(i+1, window_size+1) for i in range(seq_len))
        
        print(f"  Causal violations: {causal_violations}")
        print(f"  Window violations: {window_violations}")
        print(f"  Valid positions: {valid_positions}/{expected_positions}")
        print(f"  Mask density: {valid_positions/(seq_len*seq_len)*100:.1f}%")
        
        assert causal_violations == 0, f"Causal constraint violated: {causal_violations} violations"
        assert window_violations == 0, f"Window constraint violated: {window_violations} violations"
        assert valid_positions == expected_positions, f"Expected {expected_positions} positions, got {valid_positions}"
    
    print("✓ All sliding window mask tests passed!")


def detailed_layer_analysis():
    """Analyze memory at layer level."""
    print(f"\n{'='*60}")
    print("DETAILED LAYER-LEVEL ANALYSIS")
    print(f"{'='*60}")
    
    config = GPTConfig(
        vocab_size=50257, n_layer=1, n_head=12, n_embd=768,
        attention_type='sliding_window', window_size=64,
        post_norm=True, qk_norm=True
    )
    
    model = GPT(config).cuda().to(dtype=torch.bfloat16).train()
    x_input = torch.randint(0, 50257, (1, 1024), device='cuda')
    
    # Hook to measure memory per layer
    def make_hook(name):
        def hook(module, input, output):
            torch.cuda.synchronize()
            mem = torch.cuda.memory_allocated() / 1024 / 1024
            print(f"  {name}: {mem:.2f} MB")
        return hook
    
    # Register hooks
    handles = []
    for name, module in model.named_modules():
        if 'attn' in name or 'mlp' in name:
            h = module.register_forward_hook(make_hook(name))
            handles.append(h)
    
    print("\nMemory during forward pass:")
    with ctx:
        _ = model(x_input, return_logits=False)
    
    # Cleanup
    for h in handles:
        h.remove()


def test_attention_performance():
    """Test performance comparison between attention types."""
    print(f"\n{'='*60}")
    print("ATTENTION PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    
    import time
    
    configs = [
        ('standard', 'Standard Causal Attention'),
        ('sliding_window', 'Sliding Window Attention (SDPA + mask)'),
    ]
    
    seq_len = 1024
    batch_size = 4
    
    for attn_type, description in configs:
        print(f"\nTesting {description}...")
        
        config = GPTConfig(
            vocab_size=50257, n_layer=1, n_head=12, n_embd=768,
            attention_type=attn_type, window_size=64 if attn_type == 'sliding_window' else None
        )
        
        model = GPT(config).cuda().to(dtype=torch.bfloat16).eval()
        x = torch.randint(0, 50257, (batch_size, seq_len), device='cuda')
        
        # Warmup
        with torch.no_grad(), ctx:
            for _ in range(3):
                _ = model(x, return_logits=False)
        
        torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(10):
            torch.cuda.synchronize()
            start = time.time()
            
            with torch.no_grad(), ctx:
                _ = model(x, return_logits=False)
            
            torch.cuda.synchronize()
            end = time.time()
            times.append(end - start)
        
        avg_time = sum(times) / len(times)
        print(f"  Average time: {avg_time*1000:.2f} ms")
        print(f"  Throughput: {batch_size*seq_len/avg_time:.0f} tokens/sec")
        
        del model, x
        torch.cuda.empty_cache()


if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name()}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Test sliding window attention mask
        test_sliding_window_attention_mask()
        
        # Run memory comparison
        compare_memory()
        
        # Performance comparison
        test_attention_performance()
        
        # Detailed analysis
        detailed_layer_analysis()
        
    else:
        print("CUDA not available. This script requires a GPU.")
