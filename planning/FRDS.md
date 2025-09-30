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

# FRD 2

Implement QK norm  in `train_gpt2_mod.py` following the approach described in the OLMo 2 paper. Reference: https://sebastianraschka.com/blog/2025/the-big-llm-architecture-comparison.html#21-normalization-layer-placement

