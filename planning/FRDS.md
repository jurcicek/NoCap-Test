# FRD 1: Post-Normalization Implementation

## Overview
Implement post-normalization in `train_gpt2_mod.py` following the approach described in the OLMo 2 paper. Reference: https://sebastianraschka.com/blog/2025/the-big-llm-architecture-comparison.html#21-normalization-layer-placement

## Implementation Requirements
- **Modify `Block.forward()`** to support post-normalization
- **Add configuration option**: `post_norm` to the model config
- **Create a new config**: `d12_post_norm` with `post_norm` enabled

## Additional Requirement
You want the pre norm keep using even when post norm is enabled.

## Design Questions

### Normalization Layer Placement Strategy
**How should we handle the transition between pre-norm and post-norm architectures?**

- **Option 2**: Keep pre-norm as default and add `post_norm` flag for new behavior

### Implementation Approach
**What is the most maintainable way to implement this architectural change?**

- **Option 2**: Modify existing `Block` class with conditional logic based on config flag

# FRD 2: Query-Key Normalization Implementation

## Overview
Implement Query-Key (QK) normalization in `train_gpt2_mod.py` following the approach described in the OLMo 2 paper. Reference: https://sebastianraschka.com/blog/2025/the-big-llm-architecture-comparison.html#21-normalization-layer-placement

## Implementation Requirements
- **Modify `CausalSelfAttention`** to support QK normalization
- **Add configuration option**: `qk_norm` to the model config
- **Create a new config**: `d12_post_norm_qk_norm` with `post_norm` and `qk_norm` enabled

## Additional Requirement
You want the pre-norm to keep using even when QK normalization is enabled.

## Design Questions

### Query-Key Normalization Strategy
**How should we handle the integration of QK normalization with existing attention mechanisms?**

- **Option 2**: Integrate QK normalization directly into the attention computation pipeline

### Normalization Integration Approach
**What is the most effective way to combine QK normalization with pre-norm layers?**

- **Option 1**: Apply QK normalization before applying rotary embedding

# FRD 3: FlashMLA Implementation

## Overview
Implement FlashMLA (FlashAttention with Multi-Head Latent Attention) in `train_gpt2_mod.py` to improve attention efficiency and performance. All new attention mechanisms should optionally use `post_norm` and `qk_norm` configurations. For FlashAttention, use the `flash-attn` PyTorch package.

## Architecture Components
- **Create a separate FlashAttention class** for optimized attention computation
- **Create a separate Multi-Head Latent Attention class** for reduced complexity
- **Create a combined FlashMLA class** that integrates both approaches

## Implementation Requirements
- **Integrate FlashAttention-2** for faster attention computation with better parallelism
- **Implement Multi-Head Latent Attention** to reduce computational complexity
- **Add configuration option**: `use_flash_mla` to the model config
- **Create a new config**: `d12_flash_mla` with FlashMLA enabled

## Design Questions

### FlashAttention Integration Strategy
**How should we integrate FlashAttention-2 with existing attention mechanisms?**

- **Option 2**: Create a new `FlashCausalSelfAttention` class and use configuration to switch between implementations

### Multi-Head Latent Attention Implementation
**What is the most effective way to implement Multi-Head Latent Attention while maintaining compatibility with existing normalization options?**

- **Option 1**: Implement as a separate attention head type that can be combined with other attentions

### Configuration and Backward Compatibility
**How should we handle the transition to FlashMLA while maintaining existing model configurations?**

- **Option 1**: Add `use_flash_mla` as an optional flag, keeping existing models unchanged by default

## References
- FlashAttention-2: [Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) (July 17, 2023)
- Multi-Head Latent Attention: [The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)
- Research highlights: [FlashAttention-2 Research Summary](https://magazine.sebastianraschka.com/p/research-highlights-in-three-sentences)


