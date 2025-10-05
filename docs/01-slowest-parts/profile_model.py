#!/usr/bin/env python3
"""
Profile the GPT model to identify performance bottlenecks.
Run this to see exactly which operations are slowest.
"""

import torch
import torch.profiler
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from train_gpt2_mod import GPT, GPTConfig

def profile_model_forward_backward():
    """Profile a single forward+backward pass of the model."""
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # config = GPTConfig(vocab_size=50257, n_layer=12, n_head=12, n_embd=768)
    config = GPTConfig(vocab_size=50257, n_layer=12, n_head=12, n_embd=768, attention_type="grouped_query", n_kv_head=4)
    model = GPT(config).to(device).to(torch.bfloat16)
    
    # Create dummy batch
    B, T = 12, 1024  # batch size, sequence length
    x = torch.randint(0, config.vocab_size, (B, T), device=device)
    y = torch.randint(0, config.vocab_size, (B, T), device=device)
    
    # Warmup
    print("Warming up...")
    for _ in range(3):
        logits, loss = model(x, y)
        loss.backward()
    
    torch.cuda.synchronize()
    
    # Profile
    print("\nProfiling model (forward + backward pass)...")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        logits, loss = model(x, y)
        loss.backward()
    
    torch.cuda.synchronize()
    
    # Print results
    print("\n" + "="*100)
    print("TOP 20 OPERATIONS BY CUDA TIME")
    print("="*100)
    print(prof.key_averages().table(
        sort_by="cuda_time_total", 
        row_limit=20,
        max_name_column_width=80
    ))
    
    print("\n" + "="*100)
    print("TOP 20 OPERATIONS BY MEMORY USAGE")
    print("="*100)
    print(prof.key_averages().table(
        sort_by="cuda_memory_usage", 
        row_limit=20,
        max_name_column_width=80
    ))
    
    # Export for visualization
    print("\nExporting trace for visualization...")
    prof.export_chrome_trace("model_profile_trace.json")
    print("✓ Saved to model_profile_trace.json")
    print("  View at: chrome://tracing")


def compare_attention_mechanisms():
    """Compare different attention mechanisms."""
    import time
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, T = 12, 1024
    
    configs = {
        "CausalSelfAttention": GPTConfig(
            vocab_size=50257, n_layer=12, n_head=12, n_embd=768,
            attention_type="standard"
        ),
        "Sliding Window (w=64)": GPTConfig(
            vocab_size=50257, n_layer=12, n_head=12, n_embd=768,
            attention_type="sliding_window", window_size=64
        ),
        "Multi-Head Latent": GPTConfig(
            vocab_size=50257, n_layer=12, n_head=12, n_embd=768,
            attention_type="latent", n_latent=480
        ),
        "Grouped Query Attention (4 KV)": GPTConfig(
            vocab_size=50257, n_layer=12, n_head=12, n_embd=768,
            attention_type="grouped_query", n_kv_head=4
        ),
        "Multi-Query Attention": GPTConfig(
            vocab_size=50257, n_layer=12, n_head=12, n_embd=768,
            attention_type="multi_query"
        ),
    }
    
    print("\n" + "="*100)
    print("ATTENTION MECHANISM COMPARISON")
    print("="*100)
    print(f"Batch size: {B}, Sequence length: {T}")
    print()
    
    results = {}
    
    for name, config in configs.items():
        print(f"Testing {name}...")
        model = GPT(config).to(device).to(torch.bfloat16)
        x = torch.randint(0, config.vocab_size, (B, T), device=device)
        y = torch.randint(0, config.vocab_size, (B, T), device=device)
        
        # Warmup
        for _ in range(5):
            logits, loss = model(x, y)
            loss.backward()
        
        torch.cuda.synchronize()
        
        # Measure
        start = time.perf_counter()
        for _ in range(10):
            logits, loss = model(x, y)
            loss.backward()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        avg_time_ms = (elapsed / 10) * 1000
        throughput = (B * T * 10) / elapsed
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        results[name] = {
            "time_ms": avg_time_ms,
            "throughput": throughput,
            "memory_mb": memory_mb,
        }
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Print comparison table
    print("\n" + "-"*100)
    print(f"{'Mechanism':<30}  {'Time (ms)':>10}  {'Speedup':>8}  {'Throughput (tok/s)':>15}  {'Memory (MB)':>10}")
    print("-"*100)
    
    baseline = results["CausalSelfAttention"]["time_ms"]
    for name, metrics in results.items():
        speedup = baseline / metrics["time_ms"]
        speedup_str = f"({speedup:.2f}×)" if name != "CausalSelfAttention" else ""
        print(f"{name:<30}  {metrics['time_ms']:>10.2f}  {speedup_str:>8}  {metrics['throughput']:>15.0f}  {metrics['memory_mb']:>10.1f}")
    print("-"*100)


def layer_by_layer_analysis():
    """Analyze time spent in each layer type."""
    import time
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = GPTConfig(vocab_size=50257, n_layer=12, n_head=12, n_embd=768)
    model = GPT(config).to(device).to(torch.bfloat16)
    
    B, T = 12, 1024
    x = torch.randint(0, config.vocab_size, (B, T), device=device)
    
    print("\n" + "="*100)
    print("LAYER-BY-LAYER TIME BREAKDOWN")
    print("="*100)
    
    # Profile individual layers
    model.eval()
    with torch.no_grad():
        # Embedding
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            _ = model.transformer.wte(x)
        torch.cuda.synchronize()
        emb_time = (time.perf_counter() - start) / 100 * 1000
        
        # Get embeddings for subsequent tests
        hidden = model.transformer.wte(x)
        
        # Single attention layer
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            _ = model.transformer.h[0].attn(hidden)
        torch.cuda.synchronize()
        attn_time = (time.perf_counter() - start) / 100 * 1000
        
        # Single MLP layer
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            _ = model.transformer.h[0].mlp(hidden)
        torch.cuda.synchronize()
        mlp_time = (time.perf_counter() - start) / 100 * 1000
        
        # LM head
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            _ = model.lm_head(hidden)
        torch.cuda.synchronize()
        lm_head_time = (time.perf_counter() - start) / 100 * 1000
    
    total_per_layer = attn_time + mlp_time
    total_all_layers = total_per_layer * 12
    total_forward = emb_time + total_all_layers + lm_head_time
    
    print(f"\n{'Component':<25} {'Time (ms)':<15} {'% of Forward Pass':<20}")
    print("-"*60)
    print(f"{'Embedding':<25} {emb_time:>10.4f}      {emb_time/total_forward*100:>10.1f}%")
    print(f"{'Attention (per layer)':<25} {attn_time:>10.4f}      {attn_time/total_per_layer*100:>10.1f}% of layer")
    print(f"{'MLP (per layer)':<25} {mlp_time:>10.4f}      {mlp_time/total_per_layer*100:>10.1f}% of layer")
    print(f"{'All 12 layers':<25} {total_all_layers:>10.4f}      {total_all_layers/total_forward*100:>10.1f}%")
    print(f"{'LM Head':<25} {lm_head_time:>10.4f}      {lm_head_time/total_forward*100:>10.1f}%")
    print("-"*60)
    print(f"{'TOTAL FORWARD':<25} {total_forward:>10.4f}      100.0%")
    print("-"*60)
    
    print("\n📊 Key Insights:")
    if attn_time > mlp_time:
        ratio = attn_time / mlp_time
        print(f"   • Attention is {ratio:.2f}× slower than MLP per layer")
        print(f"   • Attention takes {attn_time/total_per_layer*100:.1f}% of each layer's time")
    else:
        ratio = mlp_time / attn_time
        print(f"   • MLP is {ratio:.2f}× slower than Attention per layer")
        print(f"   • MLP takes {mlp_time/total_per_layer*100:.1f}% of each layer's time")
    
    print(f"   • Total layers take {total_all_layers/total_forward*100:.1f}% of forward pass")
    print(f"   • Overhead (embed + LM head) is {(emb_time+lm_head_time)/total_forward*100:.1f}%")


if __name__ == "__main__":
    print("GPU Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU Name:", torch.cuda.get_device_name(0))
        print("GPU Memory:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")
    
    print("\n" + "="*100)
    print("GPT-2 MODEL PROFILING")
    print("="*100)
    
    # Run analyses
    try:
        layer_by_layer_analysis()
    except Exception as e:
        print(f"Layer-by-layer analysis failed: {e}")
    
    try:
        compare_attention_mechanisms()
    except Exception as e:
        print(f"Attention comparison failed: {e}")
    
    try:
        profile_model_forward_backward()
    except Exception as e:
        print(f"Detailed profiling failed: {e}")
    
    print("\n✓ Profiling complete!")

