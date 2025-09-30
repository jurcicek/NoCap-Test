# Tasks for FRD 1: Post-Normalization Implementation

## Context
This task plan implements hybrid pre/post-normalization in the GPT-2 model following the OLMo 2 paper approach. The implementation will modify the existing `Block.forward()` method to support pre-norm only (default) or hybrid pre/post-norm architectures through a configuration flag, and create a new `d12_post_norm` configuration. When post-norm is enabled, both pre-norm and post-norm are applied for enhanced stability and gradient flow. Reference: [FRD 1 in FRDS.md](FRDS.md)

---

## Task 1: Extend GPTConfig with post_norm Parameter

**Objective**: Add the `post_norm` configuration option to support both pre-norm and post-norm architectures.

### Implementation Details
- **File**: `train_gpt2_mod.py`
- **Class**: `GPTConfig` (line ~131)
- **Action**: Add `post_norm: bool = False` parameter to the dataclass

### Code Changes
```python
@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    post_norm: bool = False  # New parameter
```

### Benefits
- **Backward Compatibility**: Default `False` maintains existing pre-norm behavior
- **Clean Configuration**: Single boolean flag for architectural choice
- **Type Safety**: Dataclass ensures proper type validation

---

## Task 2: Modify Block.forward() Method for Hybrid Pre/Post-Normalization

**Objective**: Update the `Block.forward()` method to apply both pre-norm and post-norm when post-normalization is enabled.

### Implementation Details
- **File**: `train_gpt2_mod.py`
- **Class**: `Block` (line ~112)
- **Method**: `forward()` (line ~120)
- **Algorithm**: Hybrid normalization - pre-norm always applied, post-norm added when enabled

### Code Changes
```python
def forward(self, x):
    if self.config.post_norm:
        # Hybrid: Apply pre-norm (input normalization) + post-norm (output normalization)
        x = x + self.attn_scale * rmsnorm(self.attn(rmsnorm(x)))
        x = x + rmsnorm(self.mlp(rmsnorm(x)))
    else:
        # Pre-norm only: Apply normalization first (existing behavior)
        x = x + self.attn_scale * self.attn(rmsnorm(x))
        x = x + self.mlp(rmsnorm(x))
    return x
```

### Algorithm Explanation
- **Pre-norm only** (default): `x = x + attention(rmsnorm(x))` - Normalize input before attention
- **Hybrid Pre/Post-norm**: `x = x + rmsnorm(attention(rmsnorm(x)))` - Normalize both input and output
- **Benefits**: Combines stability of pre-norm with improved gradient flow of post-norm

### Libraries Used
- **PyTorch**: `torch.nn.Module` for conditional logic
- **Existing `rmsnorm`**: Reuse the existing RMS normalization function

---

## Task 3: Add d12_post_norm Configuration

**Objective**: Create a new model configuration that enables post-normalization for the d12 model.

### Implementation Details
- **File**: `train_gpt2_mod.py`
- **Location**: Model configurations dictionary (line ~441)
- **Action**: Add new configuration entry

### Code Changes
```python
model_configs = {
    "d12": GPTConfig(
        vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768
    ),  # 124M GPT-2
    "d12_post_norm": GPTConfig(
        vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768, post_norm=True
    ),  # 124M GPT-2 with post-normalization
    "d24": GPTConfig(vocab_size=num_vocab, n_layer=24, n_head=16, n_embd=1024),
    "d36": GPTConfig(vocab_size=num_vocab, n_layer=36, n_head=20, n_embd=1280),
    "d48": GPTConfig(vocab_size=num_vocab, n_layer=48, n_head=25, n_embd=1600),
}
```

### Command Line Support
- **Update argument parser**: Add `d12_post_norm` to valid model choices (line ~390)
- **Help text**: Update help text to include new option

### Benefits
- **Easy Testing**: Simple command-line flag to test post-norm vs pre-norm
- **Reproducible**: Clear configuration for experiments
- **Scalable**: Pattern can be extended to other model sizes

---

## Task 4: Update Command Line Interface

**Objective**: Extend the argument parser to support the new `d12_post_norm` model configuration.

### Implementation Details
- **File**: `train_gpt2_mod.py`
- **Location**: Argument parser setup and validation
- **Action**: Add `d12_post_norm` to valid model choices

### Code Changes
```python
# Update help text (around line 322)
parser.add_argument("--model", type=str, default="d12", 
                   help="d12|d12_post_norm|d24|d36|d48")

# Update validation (around line 390)
assert args.model in {"d12", "d12_post_norm", "d24", "d36", "d48"}
```

### Testing Commands
```bash
# Test pre-norm (default d12)
python train_gpt2_mod.py --model d12

# Test post-norm (new configuration)
python train_gpt2_mod.py --model d12_post_norm
```

---

## Task 5: Add Block Configuration Access

**Objective**: Ensure the Block class has access to the configuration for the conditional logic.

### Implementation Details
- **File**: `train_gpt2_mod.py`
- **Class**: `Block` (line ~112)
- **Action**: Store config reference in Block constructor

### Code Changes
```python
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # Store config reference
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.attn_scale = 1 / math.sqrt(2 * config.n_layer)
```

### Benefits
- **Configuration Access**: Block can access `post_norm` flag
- **Consistency**: Follows existing pattern in other classes
- **Maintainability**: Clear dependency on configuration

---

## Task 6: Add Documentation and Comments

**Objective**: Document the post-normalization implementation for future maintainability.

### Implementation Details
- **File**: `train_gpt2_mod.py`
- **Location**: Block class and configuration
- **Action**: Add comprehensive docstrings and comments

### Documentation Additions
```python
@dataclass
class GPTConfig:
    """Configuration for GPT model architecture.
    
    Args:
        post_norm: If True, use hybrid pre/post-normalization (OLMo 2 style + input norm).
                  If False, use pre-normalization only (GPT-2 style).
    """
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    post_norm: bool = False

class Block(nn.Module):
    """Transformer block with conditional hybrid pre/post normalization.
    
    Supports pre-norm only (GPT-2 style) or hybrid pre/post-norm architectures
    based on config.post_norm flag. Hybrid mode applies both input and output
    normalization for enhanced training stability.
    """
```

---

## Task 7: Validation and Testing

**Objective**: Ensure the implementation works correctly and maintains backward compatibility.

### Testing Strategy
1. **Backward Compatibility Test**: Verify `d12` model works unchanged
2. **Hybrid Norm Test**: Verify `d12_post_norm` model initializes and runs with both pre and post normalization
3. **Gradient Flow Test**: Compare gradient norms between pre-norm only vs hybrid pre/post-norm
4. **Memory Usage Test**: Ensure no significant memory overhead from additional normalization
5. **Normalization Verification**: Confirm both input and output normalization are applied in hybrid mode

### Test Commands
```bash
# Quick initialization test
python -c "
from train_gpt2_mod import GPT, GPTConfig
config = GPTConfig(post_norm=True)
model = GPT(config)
print('Hybrid pre/post-norm model initialized successfully')
"

# Training smoke test
python train_gpt2_mod.py --model d12_post_norm --max_iters 10 --batch_size 1
```

### Expected Outcomes
- **No Errors**: Model initializes and trains without errors
- **Enhanced Stability**: Hybrid normalization shows improved gradient flow compared to pre-norm only
- **Performance**: Better or similar training dynamics with enhanced numerical stability

---

## Implementation Order

1. **Task 1**: Extend GPTConfig with post_norm parameter
2. **Task 5**: Add Block configuration access  
3. **Task 2**: Modify Block.forward() method
4. **Task 3**: Add d12_post_norm configuration
5. **Task 4**: Update command line interface
6. **Task 6**: Add documentation and comments
7. **Task 7**: Validation and testing

---

## Success Criteria

- [ ] `d12` model works unchanged (backward compatibility)
- [ ] `d12_post_norm` model initializes successfully
- [ ] Both models can be trained without errors
- [ ] Command line interface supports both configurations
- [ ] Code is well-documented and maintainable
- [ ] Implementation follows existing code patterns

---

## References

- **OLMo 2 Paper**: Post-normalization architecture reference
- **Sebastian Raschka's Blog**: Normalization layer placement comparison
- **Existing Code**: `train_gpt2_mod.py` for implementation patterns
- **PyTorch Documentation**: For conditional logic and configuration patterns
