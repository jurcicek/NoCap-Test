# Performance Analysis: GPT-2 Model Bottlenecks and Optimization Strategies

## Current Model Architecture Overview

**Base Model (d12 - 124M parameters):**
- 12 transformer blocks
- 12 attention heads
- 768 embedding dimensions
- 64 head dimension (768/12)
- MLP expansion: 4x (768 → 3072 → 768)

---

## 🔴 SLOWEST PARTS OF THE MODEL

### 1. **Attention Mechanism - O(n²) Complexity** ⚠️ PRIMARY BOTTLENECK

**Location:** Lines 117-164 (`CausalSelfAttention`)

**Cost Breakdown:**
- **QKV Projection:** `Linear(768 → 2304)` - Moderate cost
- **Attention Computation:** `O(T² × d)` where T=sequence_length - **HIGHEST COST**
- **Output Projection:** `Linear(768 → 768)` - Moderate cost

**Why it's slow:**
- Quadratic complexity: For T=1024, you compute 1M attention scores per head
- Memory bandwidth: Large attention matrices need to be materialized
- For 12 heads × 12 layers = 144 attention operations per forward pass

**Memory Usage:** ~`B × n_heads × T² × 4 bytes` per layer

---

### 2. **MLP Layers - Large Linear Projections** ⚠️ SECONDARY BOTTLENECK

**Location:** Lines 317-328 (`MLP`)

**Cost Breakdown:**
- **Up projection:** `Linear(768 → 3072)` - **HIGH COST**
- **GELU activation:** Moderate
- **Down projection:** `Linear(3072 → 768)` - **HIGH COST**

**Why it's slow:**
- 4× expansion means 12× more parameters than embedding dimension
- Each layer: 2.36M parameters (768×3072 + 3072×768)
- 12 layers × 2.36M = ~28M parameters just in MLPs

---

### 3. **Normalization Operations (RMSNorm)**

**Location:** Lines 111-114 (`rmsnorm`)

**Cost:** Low individually, but called 24 times per block (2 per block × 12 blocks)

---

### 4. **Rotary Position Embeddings**

**Location:** Lines 78-108 (`Rotary` + `apply_rotary_emb`)

**Cost:** Moderate, but called for every attention layer (12 times)

---

## 🚀 SPECIFIC ARCHITECTURAL CHANGES TO SPEED UP THE MODEL

### **Strategy 1: Grouped-Query Attention (GQA)** ⭐ RECOMMENDED

**What:** Reduce KV heads while keeping Q heads
- Instead of 12 Q, 12 K, 12 V heads → use 12 Q, 4 K, 4 V heads
- Q heads share K,V across groups (3 Q heads per KV pair)

**Benefits:**
- **3× faster KV computation** (4 heads instead of 12)
- **3× less KV cache memory** during inference
- **Minimal quality loss** (proven in Llama-2, Mistral)

**Implementation:**
```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head  # 12 query heads
        self.n_kv_head = config.n_kv_head  # 4 KV heads
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        
        # Separate Q, K, V projections
        self.q_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.k_proj = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
```

**Expected speedup:** 20-30% faster attention, 15-20% overall training speedup

---

### **Strategy 2: Multi-Query Attention (MQA)** ⭐ MOST AGGRESSIVE

**What:** Single set of K,V shared across ALL query heads
- 12 Q heads → 1 K head, 1 V head

**Benefits:**
- **12× faster KV computation**
- **12× less KV memory**
- **Best for inference speed**

**Tradeoff:** Slightly lower quality (~1-2% perplexity increase)

**Expected speedup:** 30-40% faster attention, 20-25% overall training speedup

---

### **Strategy 3: Sliding Window Attention** ⭐ ALREADY IMPLEMENTED

**Location:** Lines 236-315 (`SlidingWindowAttention`)

**What:** Limit attention to local window (e.g., 64 tokens)
- Reduces O(n²) → O(n × window_size)

**Benefits:**
- For T=1024, window=64: **16× less computation**
- Enables longer sequences
- Uses FlashAttention-2 for efficiency

**Current implementation:** Available via `attention_type='sliding_window'`

**To enable:** Use `d12_window` or `d12_window_large` model configs

**Expected speedup:** 40-60% for long sequences (T>512)

---

### **Strategy 4: Reduce MLP Expansion Factor**

**Current:** 4× expansion (768 → 3072)
**Proposal:** 3× or 2.67× expansion (768 → 2304 or 2048)

**Benefits:**
- **25-33% fewer MLP parameters**
- Faster forward/backward pass
- Less memory

**Tradeoff:** May need slightly wider model to compensate

**Implementation:**
```python
class EfficientMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        expansion_factor = 3.0  # Instead of 4.0
        hidden_dim = int(config.n_embd * expansion_factor)
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)
```

**Expected speedup:** 10-15% overall training speedup

---

### **Strategy 5: SwiGLU Activation (Better Learning Efficiency)**

**What:** Replace GELU with SwiGLU activation
- Used in Llama, PaLM, Chinchilla
- Better gradient flow = faster convergence

**Implementation:**
```python
class SwiGLUMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(config.n_embd * 8/3)  # Adjust for same params
        self.gate_proj = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)
    
    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)
```

**Benefits:**
- **Faster convergence** (5-10% fewer steps to same loss)
- Better gradient properties
- Slight compute overhead (~10%) but worth it for learning speed

**Expected speedup:** 5-10% fewer training steps to reach target loss

---

### **Strategy 6: Shared Input-Output Embeddings** ✅ ALREADY DONE

**Location:** Lines 501-513 (weight tying)

**Status:** Already implemented - good!

---

### **Strategy 7: Flash Attention** ⭐ PARTIALLY IMPLEMENTED

**Current status:** Only in `SlidingWindowAttention`

**Proposal:** Add FlashAttention to standard attention
- Memory-efficient attention computation
- Fused kernel = faster

**Benefits:**
- **20-30% faster attention**
- **5-10× less memory** for attention matrices
- Enables larger batch sizes

**Implementation:**
```python
def forward(self, x):
    # ... QKV projection ...
    # Use FlashAttention instead of F.scaled_dot_product_attention
    y = flash_attn.flash_attn_func(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        causal=True
    )
```

---

### **Strategy 8: Multi-Head Latent Attention (MLA)** ⚠️ EXPERIMENTAL

**Location:** Lines 167-234 (`MultiHeadLatentAttention`)

**What:** Compress attention to lower dimension
- Full dim: 768
- Latent dim: 320
- Attention computed in 320D space

**Benefits:**
- Lower memory
- Potentially faster

**Concerns:**
- More complex implementation
- May hurt quality
- Current config uses MORE parameters (n_embd=1280 vs 768)

**Status:** Available as `d12_mla` but needs tuning

---

## 🎯 RECOMMENDED IMPLEMENTATION PRIORITY

### **High Priority (Implement First):**

1. **Grouped-Query Attention (GQA)** - Best balance of speed/quality
   - 20-30% speedup with minimal quality loss
   
2. **Flash Attention for all layers** - Easy win
   - 20-30% speedup, especially with longer sequences
   
3. **Reduce MLP expansion to 3×** - Simple change
   - 10-15% speedup

### **Medium Priority:**

4. **SwiGLU activation** - Better learning efficiency
   - Fewer steps to convergence
   
5. **Sliding window for deep layers** - Hybrid approach
   - Use full attention in early layers, sliding window in deep layers

### **Low Priority (Experimental):**

6. **Multi-Query Attention** - Most aggressive
   - Only if you prioritize speed over quality
   
7. **Multi-Head Latent Attention** - Needs more tuning

---

## 📊 EXPECTED COMBINED SPEEDUP

**Conservative estimate (GQA + Flash + 3× MLP):**
- **1.6-1.8× overall training speedup**
- Same or slightly better model quality

**Aggressive estimate (MQA + Flash + 3× MLP + SwiGLU):**
- **2.0-2.5× overall training speedup**
- May need more tuning to match quality

---

## 🔬 HOW TO MEASURE BOTTLENECKS

Add profiling code:

```python
import torch.profiler

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
) as prof:
    logits, loss = model(x, y)
    loss.backward()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

This will show you exactly which operations are slowest.

---

## 💡 LEARNING SPEED vs EXECUTION SPEED

**Two different optimizations:**

1. **Execution speed:** Faster per-step (attention optimizations, MLP reduction)
2. **Learning speed:** Fewer steps to same loss (better optimization, architecture improvements)

**For learning speed:**
- Better optimizers (Lion, Sophia, etc.)
- Better learning rate schedules
- Auxiliary losses (next-token prediction at multiple positions)
- Better initialization
- SwiGLU activation
- Deeper but narrower models


