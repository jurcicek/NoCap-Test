# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.3] - 2025-09-30

### Added
- **FlashCausalSelfAttention Class**: Implemented FlashAttention-2 based attention mechanism with optional post_norm and qk_norm support for optimized computation
- **MultiHeadLatentAttention Class**: Added reduced complexity attention through latent space projection achieving O(n×d) complexity with optional FlashAttention support
- **Flash-attn Integration**: Integrated `flash-attn` library for 2-4x faster attention computation with better parallelism
- **FlashAttention Configuration**: Added `use_flash_attention` parameter to GPTConfig for enabling FlashAttention-2 based attention
- **Latent Attention Configuration**: Added `use_latent_attention` and `n_latent` parameters to GPTConfig for reduced complexity attention
- **Enhanced Documentation**: Added comprehensive docstrings to CausalSelfAttention, FlashCausalSelfAttention, and MultiHeadLatentAttention classes
- **Time Module Import**: Added time import for performance benchmarking capabilities

### Changed
- **Block Architecture**: Modified Block class to support FlashAttention and Multi-Head Latent Attention with configuration-driven selection
- **CausalSelfAttention Enhancement**: Enhanced CausalSelfAttention with proper QK normalization support and improved documentation
- **Model Configurations**: Enhanced GPTConfig with FlashAttention and latent attention parameters for flexible attention mechanism selection
- **Code Formatting**: Improved code formatting and comments for better readability

### Enhanced
- **Attention Performance**: FlashAttention-2 provides 2-4x faster attention computation with better parallelism
- **Memory Efficiency**: Reduced memory usage for attention operations, especially beneficial for long sequences  
- **Scalability**: Multi-Head Latent Attention enables better performance on very long sequences through reduced complexity
- **Composability**: FlashAttention can be combined with latent attention for optimal performance on different sequence lengths
- **Backward Compatibility**: All existing model configurations remain unchanged, FlashAttention and latent attention are opt-in

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

