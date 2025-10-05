#!/usr/bin/env python3
"""
Grouped Query Attention (GQA) and Multi-Query Attention (MQA) Testing and Benchmarking

This file contains tests and benchmarks for the GQA and MQA implementations
that have been moved to the main train_gpt2_mod.py file.

GQA is a key optimization used in modern LLMs like Llama-2, Mistral, and Gemma.
It provides 20-30% speedup with minimal quality loss by reducing KV computation.

Key idea: Instead of having one K,V head per Q head, multiple Q heads share K,V heads.
- Standard MHA: 12 Q heads, 12 K heads, 12 V heads
- GQA (4 groups): 12 Q heads, 4 K heads, 4 V heads  → 3× less KV computation
- MQA (extreme):  12 Q heads, 1 K head,  1 V head   → 12× less KV computation

Benefits:
1. Faster training: Less KV computation
2. Faster inference: Smaller KV cache
3. Proven quality: Used in production models

NOTE: The actual GQA and MQA implementations are now in train_gpt2_mod.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from train_gpt2_mod import (
    Rotary, apply_rotary_emb, rmsnorm, CausalSelfAttention, 
    MultiHeadLatentAttention, SlidingWindowAttention,
    GroupedQueryAttention, MultiQueryAttention
)


# ============================================================================
# Example usage and benchmarking
# ============================================================================

def benchmark_attention_variants():
    """Benchmark different attention mechanisms."""
    import time
    from dataclasses import dataclass
    
    @dataclass
    class Config:
        n_embd: int = 768
        n_head: int = 12
        n_kv_head: int = 4
        n_latent: int = 480  # For MultiHeadLatentAttention (480 = 12 * 40, divisible by n_head)
        window_size: int = 64  # For SlidingWindowAttention
        qk_norm: bool = True
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = Config()
    
    B, T = 12, 1024
    x = torch.randn(B, T, config.n_embd, device=device, dtype=torch.bfloat16)
    
    print("\n" + "="*80)
    print("ATTENTION MECHANISM COMPARISON")
    print("="*80)
    print(f"Config: {config}")
    print(f"Input shape: (B={B}, T={T}, C={config.n_embd})")
    print()
    
    # Test standard attention (for comparison, would need to import it)
    # Here we'll just compare GQA variants
    
    # Test different attention mechanisms
    variants = {}
    
    # Baseline: Standard CausalSelfAttention (equivalent to GQA with 12 KV heads)
    variants["CausalSelfAttention (baseline)"] = CausalSelfAttention(config)
    
    # MultiHeadLatentAttention (reduced latent space)
    variants["MultiHeadLatentAttention"] = MultiHeadLatentAttention(config)
    
    # SlidingWindowAttention (FlashAttention-2 with sliding window)
    variants["SlidingWindowAttention"] = SlidingWindowAttention(config)
    
    # GQA with 12 KV heads (standard MHA - all heads independent)
    config_12kv = Config(n_kv_head=12)
    variants["GQA (12 KV heads)"] = GroupedQueryAttention(config_12kv)
    
    # GQA with 4 KV heads (current default)
    variants["GQA (4 KV heads)"] = GroupedQueryAttention(config)
    
    # GQA with 2 KV heads
    config_2kv = Config(n_kv_head=2)
    variants["GQA (2 KV heads)"] = GroupedQueryAttention(config_2kv)
    
    # MQA with 1 KV head (extreme case)
    variants["MQA (1 KV head)"] = MultiQueryAttention(config)
    
    results = {}
    
    for name, attn in variants.items():
        attn = attn.to(device).to(torch.bfloat16)
        
        # Warmup
        for _ in range(10):
            _ = attn(x)
        
        torch.cuda.synchronize() if device == "cuda" else None
        
        # Measure
        start = time.perf_counter()
        for _ in range(100):
            y = attn(x)
        
        torch.cuda.synchronize() if device == "cuda" else None
        elapsed = time.perf_counter() - start
        
        avg_time_ms = (elapsed / 100) * 1000
        
        # Count parameters
        n_params = sum(p.numel() for p in attn.parameters())
        
        results[name] = {
            "time_ms": avg_time_ms,
            "params": n_params,
        }
        
        print(f"{name}:")
        print(f"  Time: {avg_time_ms:.4f} ms")
        print(f"  Params: {n_params:,}")
        print()
    
    # Print comparison
    print("-"*80)
    print("COMPARISON (vs CausalSelfAttention baseline):")
    print("-"*80)
    baseline_time = results["CausalSelfAttention (baseline)"]["time_ms"]
    baseline_params = results["CausalSelfAttention (baseline)"]["params"]
    
    for name, metrics in results.items():
        speedup = baseline_time / metrics["time_ms"] if name != "CausalSelfAttention (baseline)" else 1.0
        param_ratio = metrics["params"] / baseline_params
        print(f"{name}:")
        print(f"  Speedup: {speedup:.2f}×")
        print(f"  Param ratio: {param_ratio:.2f}×")
        print()


if __name__ == "__main__":
    print("Testing Grouped Query Attention implementation")
    print()
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        benchmark_attention_variants()
    else:
        print("⚠ CUDA not available, skipping benchmark")

