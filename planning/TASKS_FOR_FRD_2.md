# Tasks for FRD 2: Query-Key Normalization Implementation

## Context
This task plan implements Query-Key (QK) normalization in the GPT-2 model following the OLMo 2 paper approach. The implementation will modify the existing `CausalSelfAttention` class to support QK normalization before applying rotary embeddings, and create a new `d12_post_norm_qk_norm` configuration that combines both post-normalization and QK normalization. When QK normalization is enabled, it will be applied to both query and key matrices before rotary embedding to improve attention stability and training dynamics. Reference: [FRD 2 in FRDS.md](FRDS.md)

---

## Task 1: Extend GPTConfig with qk_norm Parameter

**Objective**: Add the `qk_norm` configuration option to support Query-Key normalization in attention mechanisms.

### Implementation Details
- **File**: `train_gpt2_mod.py`
- **Class**: `GPTConfig` (line ~131)
- **Action**: Add `qk_norm: bool = False` parameter to the dataclass

### Code Changes
```python
@dataclass
class GPTConfig:
    """Configuration for GPT model architecture.
    
    Args:
        post_norm: If True, use hybrid pre/post-normalization (OLMo 2 style + input norm).
                  If False, use pre-normalization only (GPT-2 style).
        qk_norm: If True, apply Query-Key normalization before rotary embedding.
                If False, use standard attention without QK normalization.
    """
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    post_norm: bool = False  # Hybrid pre/post-normalization
    qk_norm: bool = False    # Query-Key normalization
```

### Benefits
- **Backward Compatibility**: Default `False` maintains existing attention behavior
- **Clean Configuration**: Single boolean flag for QK normalization choice
- **Type Safety**: Dataclass ensures proper type validation
- **Composability**: Can be combined with post_norm for enhanced training

---

## Task 2: Modify CausalSelfAttention for QK Normalization

**Objective**: Update the `CausalSelfAttention.forward()` method to apply Query-Key normalization before rotary embedding.

### Implementation Details
- **File**: `train_gpt2_mod.py`
- **Class**: `CausalSelfAttention` (line ~60)
- **Method**: `forward()` (line ~74)
- **Algorithm**: Apply RMS normalization to Q and K matrices before rotary embedding

### Code Changes
```python
def forward(self, x):
    B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
    # calculate query, key, values for all heads in batch and move head forward to be the batch dim
    qkv = self.c_attn(x)
    q, k, v = qkv.split(self.n_embd, dim=2)
    k = k.view(B, T, self.n_head, self.head_dim)
    q = q.view(B, T, self.n_head, self.head_dim)
    v = v.view(B, T, self.n_head, self.head_dim)
    
    # Apply QK normalization before rotary embedding if enabled
    if self.config.qk_norm:
        q = rmsnorm(q, eps=1e-6)
        k = rmsnorm(k, eps=1e-6)
    
    cos, sin = self.rotary(q)
    q = apply_rotary_emb(q, cos, sin)
    k = apply_rotary_emb(k, cos, sin)
    y = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
    )
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    # output projection
    y = self.c_proj(y)
    return y
```

### Algorithm Explanation
- **Standard attention**: Q and K matrices used directly with rotary embedding
- **QK normalization**: RMS normalization applied to Q and K matrices before rotary embedding
- **Benefits**: Improves attention stability, reduces gradient variance, enhances training dynamics

### Libraries Used
- **PyTorch**: `torch.nn.Module` for conditional logic
- **Existing `rmsnorm`**: Reuse the existing RMS normalization function
- **Existing `Rotary`**: Leverage existing rotary embedding implementation

---

## Task 3: Add CausalSelfAttention Configuration Access

**Objective**: Ensure the CausalSelfAttention class has access to the configuration for the conditional logic.

### Implementation Details
- **File**: `train_gpt2_mod.py`
- **Class**: `CausalSelfAttention` (line ~60)
- **Action**: Store config reference in CausalSelfAttention constructor

### Code Changes
```python
class CausalSelfAttention(nn.Module):
    """Causal self-attention with optional Query-Key normalization.
    
    Supports standard attention or Query-Key normalized attention based on
    config.qk_norm flag. QK normalization is applied before rotary embedding
    to improve attention stability and training dynamics.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config  # Store config reference for conditional logic
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)
```

### Benefits
- **Configuration Access**: CausalSelfAttention can access `qk_norm` flag
- **Consistency**: Follows existing pattern established in Block class
- **Maintainability**: Clear dependency on configuration

---

## Task 4: Add d12_post_norm_qk_norm Configuration

**Objective**: Create a new model configuration that enables both post-normalization and QK normalization.

### Implementation Details
- **File**: `train_gpt2_mod.py`
- **Location**: Model configurations dictionary (line ~461)
- **Action**: Add new configuration entry with both flags enabled

### Code Changes
```python
model_configs = {
    "d12": GPTConfig(
        vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768
    ),  # 124M GPT-2
    "d12_post_norm": GPTConfig(
        vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768, post_norm=True
    ),  # 124M GPT-2 with hybrid pre/post-normalization
    "d12_post_norm_qk_norm": GPTConfig(
        vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768, post_norm=True, qk_norm=True
    ),  # 124M GPT-2 with hybrid normalization + QK normalization
    "d24": GPTConfig(vocab_size=num_vocab, n_layer=24, n_head=16, n_embd=1024),
    "d36": GPTConfig(vocab_size=num_vocab, n_layer=36, n_head=20, n_embd=1280),
    "d48": GPTConfig(vocab_size=num_vocab, n_layer=48, n_head=25, n_embd=1600),
}
```

### Command Line Support
- **Update argument parser**: Add `d12_post_norm_qk_norm` to valid model choices (line ~390)
- **Help text**: Update help text to include new option

### Benefits
- **Combined Features**: Tests both post-normalization and QK normalization together
- **Reproducible**: Clear configuration for comprehensive experiments
- **Scalable**: Pattern can be extended to other model sizes

---

## Task 5: Update Command Line Interface

**Objective**: Extend the argument parser to support the new `d12_post_norm_qk_norm` model configuration.

### Implementation Details
- **File**: `train_gpt2_mod.py`
- **Location**: Argument parser setup and validation
- **Action**: Add `d12_post_norm_qk_norm` to valid model choices

### Code Changes
```python
# Update help text (around line 343)
parser.add_argument(
    "--model",
    type=str,
    default="d12",
    help="d12|d12_post_norm|d12_post_norm_qk_norm|d24|d36|d48",
)

# Update validation (around line 410)
assert args.model in {"d12", "d12_post_norm", "d12_post_norm_qk_norm", "d24", "d36", "d48"}
```

### Testing Commands
```bash
# Test standard model (d12)
python train_gpt2_mod.py --model d12

# Test post-norm model (d12_post_norm)
python train_gpt2_mod.py --model d12_post_norm

# Test combined model (d12_post_norm_qk_norm)
python train_gpt2_mod.py --model d12_post_norm_qk_norm
```

---

## Task 6: Add Documentation and Comments

**Objective**: Document the QK normalization implementation for future maintainability.

### Implementation Details
- **File**: `train_gpt2_mod.py`
- **Location**: CausalSelfAttention class and configuration
- **Action**: Add comprehensive docstrings and comments

### Documentation Additions
```python
@dataclass
class GPTConfig:
    """Configuration for GPT model architecture.
    
    Args:
        post_norm: If True, use hybrid pre/post-normalization (OLMo 2 style + input norm).
                  If False, use pre-normalization only (GPT-2 style).
        qk_norm: If True, apply Query-Key normalization before rotary embedding.
                If False, use standard attention without QK normalization.
    """
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    post_norm: bool = False  # Hybrid pre/post-normalization
    qk_norm: bool = False    # Query-Key normalization

class CausalSelfAttention(nn.Module):
    """Causal self-attention with optional Query-Key normalization.
    
    Supports standard attention or Query-Key normalized attention based on
    config.qk_norm flag. QK normalization is applied before rotary embedding
    to improve attention stability and training dynamics.
    
    The normalization is applied to both query and key matrices using RMS
    normalization before rotary positional embeddings are applied.
    """
```

---

## Task 7: Validation and Testing

**Objective**: Ensure the implementation works correctly and maintains backward compatibility.

### Testing Strategy
1. **Backward Compatibility Test**: Verify `d12` model works unchanged
2. **QK Norm Test**: Verify `d12_post_norm_qk_norm` model initializes and runs with QK normalization
3. **Combined Features Test**: Verify both post-normalization and QK normalization work together
4. **Attention Stability Test**: Compare attention weight distributions between standard and QK normalized models
5. **Memory Usage Test**: Ensure no significant memory overhead from QK normalization

### Test Commands
```bash
# Quick initialization test
python -c "
from train_gpt2_mod import GPT, GPTConfig
config = GPTConfig(qk_norm=True)
model = GPT(config)
print('QK normalized model initialized successfully')
"

# Combined features test
python -c "
from train_gpt2_mod import GPT, GPTConfig
config = GPTConfig(post_norm=True, qk_norm=True)
model = GPT(config)
print('Combined post-norm + QK norm model initialized successfully')
"

# Training smoke test
python train_gpt2_mod.py --model d12_post_norm_qk_norm --max_iters 10 --batch_size 1
```

### Expected Outcomes
- **No Errors**: Model initializes and trains without errors
- **Enhanced Stability**: QK normalization shows improved attention weight distributions
- **Performance**: Better or similar training dynamics with enhanced numerical stability
- **Combined Benefits**: Both normalization techniques work synergistically

---

## Implementation Order

1. **Task 1**: Extend GPTConfig with qk_norm parameter
2. **Task 3**: Add CausalSelfAttention configuration access  
3. **Task 2**: Modify CausalSelfAttention.forward() method
4. **Task 4**: Add d12_post_norm_qk_norm configuration
5. **Task 5**: Update command line interface
6. **Task 6**: Add documentation and comments
7. **Task 7**: Validation and testing

---

## Success Criteria

- [ ] `d12` model works unchanged (backward compatibility)
- [ ] `d12_post_norm_qk_norm` model initializes successfully
- [ ] QK normalization is applied before rotary embedding
- [ ] Both post-normalization and QK normalization work together
- [ ] Command line interface supports new configuration
- [ ] Code is well-documented and maintainable
- [ ] Implementation follows existing code patterns

---

## References

- **OLMo 2 Paper**: Query-Key normalization architecture reference
- **Sebastian Raschka's Blog**: Attention mechanism improvements
- **Existing Code**: `train_gpt2_mod.py` for implementation patterns
- **PyTorch Documentation**: For conditional logic and configuration patterns
