# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.3] - 2025-01-29

### Added
- **FlashMLA Implementation**: Implemented FlashAttention-2 with Multi-Head Latent Attention for improved performance and efficiency
- **FlashCausalSelfAttention Class**: New attention mechanism using FlashAttention-2 with optional post_norm and qk_norm support
- **MultiHeadLatentAttention Class**: Reduced complexity attention through latent space projection achieving O(n×d) complexity
- **Combined FlashMLA Class**: Integration of FlashAttention-2 with Multi-Head Latent Attention for optimal performance
- **FlashMLA Configuration**: Added `use_flash_mla`, `use_latent_attention`, and `latent_dim` parameters to GPTConfig
- **New Model Configuration**: Added `d12_flash_mla` model variant with FlashMLA attention mechanism enabled
- **Command Line Support**: Extended CLI with `--use-flash-mla`, `--use-latent-attention`, and `--latent-dim` options
- **Performance Benchmarking**: Added benchmarking tools to measure and compare different attention mechanisms
- **Memory Usage Tracking**: Implemented memory usage measurement for different model configurations

### Changed
- **Block Architecture**: Modified Block class to support FlashMLA attention mechanism with configuration-driven selection
- **Model Configurations**: Enhanced GPTConfig with FlashMLA-specific parameters for flexible attention mechanism selection
- **Training Scripts**: Updated run_mod_v01.sh to use d12_flash_mla model configuration by default

### Enhanced
- **Attention Performance**: FlashAttention-2 provides 2-4x faster attention computation with better parallelism
- **Memory Efficiency**: Reduced memory usage for attention operations, especially beneficial for long sequences
- **Scalability**: Multi-Head Latent Attention enables better performance on very long sequences through reduced complexity
- **Backward Compatibility**: All existing model configurations remain unchanged, FlashMLA is opt-in

## [1.0.2] - 2025-01-29

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

## [1.0.1] - 2025-01-29

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

