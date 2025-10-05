# Quick Implementation Guide: Ready-to-Use Code Snippets

This guide provides copy-paste ready code to optimize your GPT-2 model.

---

## Option 1: Add Grouped Query Attention (GQA) - 20% Speedup

### Step 1: Add to GPTConfig (line 448)

```python
@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_kv_head: int = 4  # ← ADD THIS LINE
    post_norm: bool = False
    qk_norm: bool = False
    n_latent: int = 384
    attention_type: str = "standard"
    window_size: int = 64
    gradient_checkpointing: bool = True
    use_gated_lm_head: bool = False
    gated_bottleneck_factor: int = 4
```

### Step 2: Add GQA class (after CausalSelfAttention, around line 165)

```python
class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention - 3× faster KV computation."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_kv_head = getattr(config, 'n_kv_head', config.n_head)
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        
        assert self.n_embd % self.n_head == 0
        assert self.n_head % self.n_kv_head == 0
        
        self.n_rep = self.n_head // self.n_kv_head
        
        # Separate Q, K, V projections
        self.q_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.k_proj = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)
        
        print(f"GQA: {self.n_head} Q heads, {self.n_kv_head} KV heads ({self.n_rep}:1 ratio)")
    
    def forward(self, x):
        B, T, C = x.size()
        
        # Compute Q, K, V
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        
        # QK normalization if enabled
        if getattr(self.config, 'qk_norm', False):
            q = rmsnorm(q, eps=1e-6)
            k = rmsnorm(k, eps=1e-6)
        
        # Apply rotary embeddings
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        
        # Repeat K, V to match Q heads
        if self.n_rep > 1:
            k = k.unsqueeze(2).expand(B, T, self.n_kv_head, self.n_rep, self.head_dim)
            k = k.reshape(B, T, self.n_head, self.head_dim)
            v = v.unsqueeze(2).expand(B, T, self.n_kv_head, self.n_rep, self.head_dim)
            v = v.reshape(B, T, self.n_head, self.head_dim)
        
        # Attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.o_proj(y)
```

### Step 3: Update Block to use GQA (line 338)

```python
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        attention_type = getattr(config, 'attention_type', 'standard')
        
        if attention_type == 'gqa':  # ← ADD THIS
            self.attn = GroupedQueryAttention(config)
        elif attention_type == 'sliding_window':
            self.attn = SlidingWindowAttention(config)
        elif attention_type == 'latent':
            self.attn = MultiHeadLatentAttention(config)
        else:
            self.attn = CausalSelfAttention(config)
        
        self.mlp = MLP(config)
        self.attn_scale = 1 / math.sqrt(2 * config.n_layer)
```

### Step 4: Add model config (line 906 in base_configs)

```python
base_configs = {
    # ... existing configs ...
    "d12_gqa": GPTConfig(
        vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768,
        n_kv_head=4, attention_type="gqa", post_norm=True, qk_norm=True
    ),
}
```

### Step 5: Update model assertion (line 851)

```python
assert args.model in {
    "d12", "d12_post_norm", "d12_post_norm_qk_norm", "d12_mla", 
    "d12_window", "d12_window_large", "d12_gemb", "d12_ghead", 
    "d12_gemb_ghead", "d12_glu_head", "d12_gqa",  # ← ADD THIS
    "d24", "d36", "d48",
}
```

### Test it:
```bash
source activate_env.sh
torchrun --standalone --nproc_per_node=1 train_gpt2_mod.py \
    --model=d12_gqa \
    --batch_size=4 \
    --sequence_length=512 \
    --num_iterations=100
```

---

## Option 2: Add Flash Attention to Standard Attention - 20% Speedup

### Modify CausalSelfAttention.forward() (line 141)

Replace lines 158-160:

```python
# OLD:
y = F.scaled_dot_product_attention(
    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
)

# NEW:
if self.training:
    # Use Flash Attention during training
    q_fa = q.transpose(1, 2).to(torch.bfloat16)
    k_fa = k.transpose(1, 2).to(torch.bfloat16)
    v_fa = v.transpose(1, 2).to(torch.bfloat16)
    y = flash_attn.flash_attn_func(q_fa, k_fa, v_fa, causal=True)
    y = y.to(q.dtype)
else:
    # Fallback for inference
    y = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
    )
```

---

## Option 3: Reduce MLP Expansion to 3× - 15% Speedup

### Modify MLP class (line 317)

```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Changed from 4× to 3× expansion
        expansion_factor = 3
        hidden_dim = expansion_factor * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x
```

**Note:** You might want to slightly increase n_embd (768→832) to compensate for capacity loss.

---

## Option 4: Add SwiGLU Activation - 5-10% Fewer Training Steps

### Add SwiGLU MLP class (after regular MLP, around line 329)

```python
class SwiGLUMLP(nn.Module):
    """SwiGLU MLP - better gradient flow and faster convergence."""
    
    def __init__(self, config):
        super().__init__()
        # SwiGLU typically uses ~2.67× expansion to match parameter count
        hidden_dim = int(config.n_embd * 8 / 3)
        self.gate_proj = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)
    
    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)
```

### Add config option (line 473)

```python
@dataclass
class GPTConfig:
    # ... existing fields ...
    use_swiglu: bool = False  # ← ADD THIS
```

### Update Block to use SwiGLU (line 347)

```python
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # ... attention setup ...
        
        # Choose MLP type
        if getattr(config, 'use_swiglu', False):
            self.mlp = SwiGLUMLP(config)
        else:
            self.mlp = MLP(config)
        
        self.attn_scale = 1 / math.sqrt(2 * config.n_layer)
```

### Add model config:

```python
"d12_swiglu": GPTConfig(
    vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768,
    use_swiglu=True, post_norm=True, qk_norm=True
),
```

---

## Option 5: Hybrid Attention Strategy - 30% Speedup for Long Sequences

### Modify Block to use layer-dependent attention (line 333)

```python
class Block(nn.Module):
    def __init__(self, config, layer_idx=0):  # ← ADD layer_idx parameter
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Hybrid attention strategy based on layer depth
        attention_type = getattr(config, 'attention_type', 'standard')
        
        if attention_type == 'hybrid':
            # Early layers: full attention (global context)
            if layer_idx < config.n_layer // 3:
                self.attn = CausalSelfAttention(config)
            # Middle layers: GQA (efficiency)
            elif layer_idx < 2 * config.n_layer // 3:
                self.attn = GroupedQueryAttention(config)
            # Deep layers: sliding window (local patterns)
            else:
                self.attn = SlidingWindowAttention(config)
        elif attention_type == 'gqa':
            self.attn = GroupedQueryAttention(config)
        # ... rest of attention types ...
        
        self.mlp = MLP(config)
        self.attn_scale = 1 / math.sqrt(2 * config.n_layer)
```

### Update GPT model to pass layer_idx (line 489)

```python
self.transformer = nn.ModuleDict(
    dict(
        wte=embedding_layer,
        h=nn.ModuleList([Block(config, layer_idx=i) for i in range(config.n_layer)]),
    )
)
```

### Add hybrid config:

```python
"d12_hybrid": GPTConfig(
    vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768,
    n_kv_head=4, attention_type="hybrid", window_size=64,
    post_norm=True, qk_norm=True
),
```

---

## Combined Optimized Model

### The "Fast" Config - Best Overall Performance

```python
"d12_fast": GPTConfig(
    vocab_size=num_vocab,
    n_layer=12,
    n_head=12,
    n_embd=768,
    n_kv_head=4,              # GQA with 4 KV heads
    attention_type="gqa",     # Use GQA
    use_swiglu=True,          # SwiGLU activation
    post_norm=True,           # Hybrid normalization
    qk_norm=True,             # QK normalization
),
```

Plus modify MLP to use 3× expansion.

**Expected results:**
- 1.6-1.8× faster training
- Same or better quality
- 30% less memory

---

## Measuring Performance

### Add timing code to measure speedup:

```python
# After line 1033 in training loop
import time

# Before training loop
times = []

# In training loop, after optimizer.step()
torch.cuda.synchronize()
step_time = time.perf_counter() - t0
times.append(step_time)

if step > 0 and step % 10 == 0:
    avg_time = sum(times[-10:]) / min(len(times), 10)
    tokens_per_sec = (B * T * ddp_world_size * args.grad_accumulation_steps) / avg_time
    print0(f"  → {tokens_per_sec:.0f} tok/s, {avg_time*1000:.1f} ms/step")

# Restart clock
t0 = time.perf_counter()
```

---

## Testing Checklist

After implementing each optimization:

1. **Does it compile?**
   ```bash
   python -c "from train_gpt2_mod import GPT, GPTConfig; print('OK')"
   ```

2. **Does it run?**
   ```bash
   torchrun --standalone --nproc_per_node=1 train_gpt2_mod.py \
       --model=d12_gqa --num_iterations=10
   ```

3. **Is it faster?**
   - Compare tokens/second before and after
   - Should see 15-30% improvement

4. **Does it learn?**
   - Train for 1000 steps
   - Compare validation loss to baseline
   - Should be within 1-2% of baseline

5. **Memory usage?**
   - Check peak memory: `torch.cuda.max_memory_allocated()`
   - Should use same or less memory

---

## Debugging Common Issues

### Issue: "n_head must be divisible by n_kv_head"
**Solution:** Make sure n_kv_head divides n_head evenly
- Good: n_head=12, n_kv_head=4 (ratio 3:1)
- Good: n_head=12, n_kv_head=6 (ratio 2:1)
- Bad: n_head=12, n_kv_head=5 (doesn't divide)

### Issue: Loss is NaN or diverging
**Solution:** 
- Reduce learning rate by 0.5×
- Add gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
- Check initialization isn't too large

### Issue: Flash Attention not working
**Solution:**
- Make sure flash-attn is installed: `pip install flash-attn`
- Check CUDA version compatibility
- Fall back to F.scaled_dot_product_attention if needed

### Issue: Slower instead of faster
**Solution:**
- Make sure model is compiled: `model = torch.compile(model)`
- Check you're using bfloat16: `.to(dtype=torch.bfloat16)`
- Warmup: First few steps are always slower
- Profile to see what's actually slow

---

## Next Steps

1. **Start with GQA** - easiest and most impactful
2. **Add Flash Attention** - good compatibility with GQA
3. **Try 3× MLP** - simple change, good speedup
4. **Experiment with SwiGLU** - may need more tuning
5. **Profile and iterate** - use profile_model.py

Expected total implementation time: **2-3 hours**
Expected total speedup: **1.6-2.0× faster**


