# Changelog

All notable changes to this project will be documented in this file.

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
