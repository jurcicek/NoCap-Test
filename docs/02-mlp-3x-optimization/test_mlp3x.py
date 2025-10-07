#!/usr/bin/env python3
"""Quick test script to verify d12_mlp3x configuration works correctly."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
from train_gpt2_mod import GPT, GPTConfig

def test_mlp_expansion_factor():
    """Test that MLP expansion factor is correctly applied."""
    print("Testing MLP expansion factor configuration...")
    
    # Test 1: Standard 4× expansion
    config_4x = GPTConfig(
        vocab_size=50257,
        n_layer=2,
        n_head=4,
        n_embd=256,
        mlp_expansion_factor=4
    )
    model_4x = GPT(config_4x)
    
    # Count parameters in first MLP layer
    mlp_fc_params_4x = sum(p.numel() for p in model_4x.transformer.h[0].mlp.c_fc.parameters())
    expected_4x = 256 * (4 * 256)  # n_embd × (4 × n_embd)
    assert mlp_fc_params_4x == expected_4x, f"4× MLP size mismatch: {mlp_fc_params_4x} != {expected_4x}"
    print(f"✓ 4× expansion: {mlp_fc_params_4x:,} parameters")
    
    # Test 2: 3× expansion
    config_3x = GPTConfig(
        vocab_size=50257,
        n_layer=2,
        n_head=4,
        n_embd=256,
        mlp_expansion_factor=3
    )
    model_3x = GPT(config_3x)
    
    mlp_fc_params_3x = sum(p.numel() for p in model_3x.transformer.h[0].mlp.c_fc.parameters())
    expected_3x = 256 * (3 * 256)  # n_embd × (3 × n_embd)
    assert mlp_fc_params_3x == expected_3x, f"3× MLP size mismatch: {mlp_fc_params_3x} != {expected_3x}"
    print(f"✓ 3× expansion: {mlp_fc_params_3x:,} parameters")
    
    # Verify reduction
    reduction_pct = (1 - mlp_fc_params_3x / mlp_fc_params_4x) * 100
    print(f"✓ Parameter reduction: {reduction_pct:.1f}%")
    assert abs(reduction_pct - 25.0) < 0.1, "Expected 25% reduction"
    
    print("\nAll MLP expansion factor tests passed! ✓")


def test_d12_mlp3x_config():
    """Test the d12_mlp3x configuration."""
    print("\nTesting d12_mlp3x configuration...")
    
    config = GPTConfig(
        vocab_size=50257,
        n_layer=12,
        n_head=13,
        n_embd=832,
        mlp_expansion_factor=3,
        post_norm=True,
        qk_norm=True
    )
    
    # Check head dimension
    head_dim = config.n_embd // config.n_head
    assert head_dim == 64, f"Head dimension should be 64, got {head_dim}"
    print(f"✓ n_embd={config.n_embd}, n_head={config.n_head}, head_dim={head_dim}")
    
    # Create model
    model = GPT(config)
    
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Total parameters: {total_params:,}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 64
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        logits, loss = model(x, x)
    
    assert logits.shape == (batch_size, seq_len, config.vocab_size), \
        f"Unexpected output shape: {logits.shape}"
    print(f"✓ Forward pass successful: output shape {logits.shape}")
    
    print("\nd12_mlp3x configuration test passed! ✓")


def compare_model_sizes():
    """Compare parameter counts between d12 and d12_mlp3x."""
    print("\nComparing model sizes...")
    
    # Standard d12
    config_d12 = GPTConfig(
        vocab_size=50257,
        n_layer=12,
        n_head=12,
        n_embd=768,
        mlp_expansion_factor=4
    )
    model_d12 = GPT(config_d12)
    params_d12 = sum(p.numel() for p in model_d12.parameters())
    
    # Optimized d12_mlp3x
    config_mlp3x = GPTConfig(
        vocab_size=50257,
        n_layer=12,
        n_head=13,
        n_embd=832,
        mlp_expansion_factor=3,
        post_norm=True,
        qk_norm=True
    )
    model_mlp3x = GPT(config_mlp3x)
    params_mlp3x = sum(p.numel() for p in model_mlp3x.parameters())
    
    print(f"\nd12 baseline:      {params_d12:,} parameters")
    print(f"d12_mlp3x:         {params_mlp3x:,} parameters")
    
    diff = params_mlp3x - params_d12
    diff_pct = (diff / params_d12) * 100
    print(f"Difference:        {diff:+,} ({diff_pct:+.2f}%)")
    
    # MLP-specific comparison
    mlp_params_d12 = sum(p.numel() for p in model_d12.transformer.h[0].mlp.parameters())
    mlp_params_mlp3x = sum(p.numel() for p in model_mlp3x.transformer.h[0].mlp.parameters())
    
    mlp_reduction = (1 - mlp_params_mlp3x / mlp_params_d12) * 100
    print(f"\nMLP layer reduction: {mlp_reduction:.1f}%")
    
    print("\nModel size comparison complete! ✓")


def test_backward_compatibility():
    """Test that default behavior matches old 4× expansion."""
    print("\nTesting backward compatibility...")
    
    # Config without explicit mlp_expansion_factor
    config_default = GPTConfig(
        vocab_size=50257,
        n_layer=2,
        n_head=4,
        n_embd=256
    )
    model = GPT(config_default)
    
    # Should default to 4× expansion
    mlp_fc_params = sum(p.numel() for p in model.transformer.h[0].mlp.c_fc.parameters())
    expected = 256 * (4 * 256)
    assert mlp_fc_params == expected, \
        f"Default should be 4× expansion: {mlp_fc_params} != {expected}"
    
    print("✓ Default mlp_expansion_factor=4 works correctly")
    print("✓ Backward compatibility maintained")


if __name__ == "__main__":
    print("=" * 70)
    print("MLP 3× Expansion Factor - Test Suite")
    print("=" * 70)
    
    test_mlp_expansion_factor()
    test_d12_mlp3x_config()
    compare_model_sizes()
    test_backward_compatibility()
    
    print("\n" + "=" * 70)
    print("All tests passed! ✅")
    print("=" * 70)
    print("\nYou can now train with: --model=d12_mlp3x")

