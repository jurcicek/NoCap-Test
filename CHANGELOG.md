# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [1.0.3] - 2025-09-30

### Added
- **Memory Profiling**: Added memory tracking to block mask creation in SlidingWindowAttention
- **Memory Analysis Documentation**: Created comprehensive MEMORY_ANALYSIS_SLIDING_WINDOW.md documenting memory overhead causes
- **Memory Testing Script**: Added test_memory.py for comparing memory usage between attention mechanisms

### Changed
- **Block Mask Logging**: Improved logging to show memory delta during block mask creation
- **Attention Comments**: Added explanatory comments about memory characteristics of flex_attention

### Documented
- **Memory Overhead Analysis**: Identified that SlidingWindowAttention uses ~350MB more memory than CausalSelfAttention due to:
  - Block mask metadata storage (~50-200 MB)
  - FlexAttention intermediate tensors (~50-100 MB)  
  - Compiled kernel overhead (~100-200 MB)
  - Cache broadcasting inefficiencies (~10-50 MB)

### Notes
- SlidingWindowAttention currently uses more memory than standard attention despite being designed for efficiency
- Root cause: flex_attention with block masks materializes sparse structures vs FlashAttention's O(1) memory
- Recommended solution: Use flash-attn library with native sliding window support instead of flex_attention

## [1.0.2] - 2025-09-29

### Added
- **Query-Key Normalization**: Implemented QK normalization in attention mechanism following OLMo 2 paper approach
- **QK Normalization Configuration**: Added `qk_norm` parameter to GPTConfig for enabling Query-Key normalization
- **Combined Model Configuration**: Added `d12_post_norm_qk_norm` model variant with both post-normalization and QK normalization
- **Enhanced Command Line Support**: Extended CLI to support `d12_post_norm_qk_norm` model configuration

### Changed
- **Attention Architecture**: Modified CausalSelfAttention.forward() method to support QK normalization before rotary embedding
- **Model Configurations**: Enhanced GPTConfig with qk_norm boolean flag for attention mechanism flexibility

### Enhanced
- **Attention Stability**: QK normalization improves attention weight distributions and training dynamics
- **Composability**: QK normalization can be combined with post-normalization for enhanced training stability

## [1.0.1] - 2025-09-29

### Added
- **Hybrid Pre/Post-Normalization**: Implemented hybrid normalization architecture in GPT-2 model following OLMo 2 paper approach
- **Post-Normalization Configuration**: Added `post_norm` parameter to GPTConfig for enabling hybrid normalization
- **New Model Configuration**: Added `d12_post_norm` model variant with hybrid pre/post-normalization enabled
- **Command Line Support**: Extended CLI to support `d12_post_norm` model configuration

### Changed
- **Block Architecture**: Modified Block.forward() method to support conditional hybrid normalization
- **Model Configurations**: Enhanced GPTConfig with post_norm boolean flag for architectural flexibility

### Enhanced
- **Training Stability**: Hybrid normalization combines pre-norm stability with post-norm gradient flow benefits
- **Backward Compatibility**: Default behavior unchanged, maintaining existing pre-norm functionality

