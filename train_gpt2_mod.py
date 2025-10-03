import os
import sys
import uuid
import math
import glob
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import flash_attn

import time


# Global cache for sliding window attention masks
_sliding_window_mask_cache = {}
_MAX_CACHE_SIZE = 10

def create_sliding_window_attn_mask(T, window_size, device):
    """Create a sliding window attention mask for scaled_dot_product_attention.
    
    Args:
        T: Sequence length
        window_size: Window size for sliding window attention
        device: Device to create the mask on
        
    Returns:
        Attention mask tensor of shape (T, T) where True means attend, False means mask out
    """
    # Check cache first
    cache_key = (T, window_size, str(device))
    if cache_key in _sliding_window_mask_cache:
        return _sliding_window_mask_cache[cache_key]
    
    # Create position indices
    q_pos = torch.arange(T, device=device).unsqueeze(1)  # (T, 1)
    k_pos = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
    
    # Create sliding window mask: within window and causal
    within_window = torch.abs(q_pos - k_pos) <= window_size
    causal_mask = k_pos <= q_pos  # causal: can only see previous tokens
    
    # Combine: attend if within window AND causal
    attn_mask = within_window & causal_mask
    
    # Cache the mask (with size limit)
    if len(_sliding_window_mask_cache) < _MAX_CACHE_SIZE:
        _sliding_window_mask_cache[cache_key] = attn_mask
    
    return attn_mask

def clear_sliding_window_mask_cache():
    """Clear the sliding window attention mask cache to free memory."""
    global _sliding_window_mask_cache
    _sliding_window_mask_cache.clear()

def clear_all_attention_caches():
    """Clear all attention mask caches to free memory."""
    clear_sliding_window_mask_cache()
    # Clear any other attention caches here if needed

with open(sys.argv[0]) as f:
    code = f.read()

torch.set_float32_matmul_precision('high')

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        # Ensure cos and sin match the dtype of input x for FlashAttention compatibility
        cos = self.cos_cached[None, :, None, :].to(dtype=x.dtype)
        sin = self.sin_cached[None, :, None, :].to(dtype=x.dtype)
        return cos, sin


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


def rmsnorm(x0, eps=1e-6):
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)


class CausalSelfAttention(nn.Module):
    """Causal self-attention with optional Query-Key normalization.
    
    Supports standard attention or Query-Key normalized attention based on
    config.qk_norm flag. QK normalization is applied before rotary embedding
    to improve attention stability and training dynamics.
    
    The normalization is applied to both query and key matrices using RMS
    normalization before rotary positional embeddings are applied.
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


class MultiHeadLatentAttention(nn.Module):
    """Multi-Head Latent Attention for reduced complexity attention computation.
    
    Implements attention in a reduced latent space. Supports post_norm and qk_norm.
    
    Args:
        config: Model configuration containing attention parameters
        n_latent: Dimension of latent space (if None, uses n_embd)
        post_norm: If True, apply layer normalization after attention
        qk_norm: If True, apply Query-Key normalization before rotary embedding
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.n_latent = config.n_latent
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        
        # Attention projections in latent space
        self.latent_head_dim = self.n_latent // self.n_head
        assert self.n_latent % self.n_head == 0
        
        # Fused input projection: directly produce QKV in latent space from input
        self.c_attn_fused = nn.Linear(self.n_embd, 3 * self.n_latent, bias=False)
        # Fused output projection: map latent output back to model dimension
        self.out_proj_fused = nn.Linear(self.n_latent, self.n_embd, bias=False)
        self.rotary = Rotary(self.latent_head_dim)

        print(f"Applying Multi-Head Latent Attention {self.n_embd=} {self.n_head=} {self.head_dim=}")
        print(f"Applying Multi-Head Latent Attention {self.n_latent=} {self.latent_head_dim=}")

    def forward(self, x, attention_mask=None):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality
        
        # Calculate query, key, values in latent space (fused input projection)
        qkv = self.c_attn_fused(x)
        q, k, v = qkv.split(self.n_latent, dim=2)
        
        # Reshape for multi-head attention in latent space
        q = q.view(B, T, self.n_head, self.latent_head_dim)
        k = k.view(B, T, self.n_head, self.latent_head_dim)
        v = v.view(B, T, self.n_head, self.latent_head_dim)
        
        # Apply QK normalization before rotary embedding if enabled
        if self.config.qk_norm:
            q = rmsnorm(q, eps=1e-6)
            k = rmsnorm(k, eps=1e-6)
        
        # Apply rotary position embeddings in latent space
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        
        # Compute attention in latent space using standard PyTorch scaled dot product attention
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), 
            is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_latent)
        
        # Fused output projection to original space
        y = self.out_proj_fused(y)  # (B, T, n_embd)
        
        return y


class SlidingWindowAttention(nn.Module):
    """Sliding window attention implementation using FlashAttention-2.
    
    Features:
    - FlashAttention-2 with sliding window support (assumes always available)
    - Fixed window size for consistent performance
    - Optimized memory usage
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.window_size = getattr(config, 'window_size', 64)
                
        assert self.n_embd % self.n_head == 0
        
        # Key, query, value projections
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        # Output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)
        
        print(f"SlidingWindowAttention: window_size={self.window_size}, n_head={self.n_head}, head_dim={self.head_dim}")

    def forward(self, x):
        B, T, C = x.size()
        
        # Calculate Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        
        # Apply QK normalization if enabled
        if getattr(self.config, 'qk_norm', False):
            q = rmsnorm(q, eps=1e-6)
            k = rmsnorm(k, eps=1e-6)
        
        # Apply rotary embeddings
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        
        # Use FlashAttention sliding window with fixed window size
        if self.training and getattr(self.config, 'gradient_checkpointing', False):
            y = torch.utils.checkpoint.checkpoint(
                self._flash_attention_sliding_window, q, k, v, self.window_size, use_reentrant=False
            )
        else:
            y = self._flash_attention_sliding_window(q, k, v, self.window_size)
        
        # Output projection
        y = self.c_proj(y)
        return y
    
    def _flash_attention_sliding_window(self, q, k, v, window_size):
        """Use FlashAttention-2 with sliding window support.
        Assumes FlashAttention-2 is always available.
        """
        # Ensure inputs are in the correct dtype for FlashAttention
        q_fa = q.transpose(1, 2).half()  # (B, H, T, D) - FlashAttention requires fp16/bf16
        k_fa = k.transpose(1, 2).half()
        v_fa = v.transpose(1, 2).half()
        
        # FlashAttention-2 with sliding window
        y = flash_attn.flash_attn_func(
            q_fa, k_fa, v_fa,
            causal=True,
            window_size=(window_size, 0)  # Sliding window support
        )
        
        # Convert back to original dtype and shape
        return y.transpose(1, 2).view(q.shape[0], q.shape[1], self.n_embd).to(q.dtype)


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config  # Store config reference for conditional logic
        
        # Initialize attention mechanism based on configuration
        attention_type = getattr(config, 'attention_type', 'standard')
        
        if attention_type == 'sliding_window':
            self.attn = SlidingWindowAttention(config)
        elif attention_type == 'latent':
            self.attn = MultiHeadLatentAttention(config)
        else:  # standard or default
            self.attn = CausalSelfAttention(config)
        
        self.mlp = MLP(config)
        self.attn_scale = 1 / math.sqrt(2 * config.n_layer)

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

class GatedEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, bottleneck_factor=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        bottleneck_dim = embed_dim // bottleneck_factor
        self.gate_down = nn.Linear(embed_dim, bottleneck_dim, bias=False)
        self.gate_up = nn.Linear(bottleneck_dim, embed_dim, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Get embeddings
        emb = self.embedding(x)

        # Compute gate
        gate_hidden = self.gate_down(emb)
        gate_values = self.sigmoid(self.gate_up(gate_hidden))

        # Apply gating
        gated_emb = gate_values * emb

        return gated_emb

class GatedLMHead(nn.Module):
    def __init__(self, embed_dim, vocab_size, bottleneck_factor=4):
        super().__init__()
        bottleneck_dim = embed_dim // bottleneck_factor
        self.gate_down = nn.Linear(embed_dim, bottleneck_dim, bias=False)
        self.gate_up = nn.Linear(bottleneck_dim, embed_dim, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x):
        # Compute gate with bottleneck
        gate_hidden = self.gate_down(x)
        gate_values = self.sigmoid(self.gate_up(gate_hidden))

        # Apply gating
        gated_values = x * gate_values

        # Get logits
        logits = self.lm_head(gated_values)

        return logits


# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    """Configuration for GPT model architecture.
    
    Args:
        post_norm: If True, use hybrid pre/post-normalization (OLMo 2 style + input norm).
                   If False, use pre-normalization only (GPT-2 style).
        qk_norm: If True, apply Query-Key normalization before rotary embedding.
                 If False, use standard attention without QK normalization.
        n_latent: Dimension of latent space for latent attention.
        attention_type: Type of attention mechanism to use.
                        Options: 'standard', 'latent', 'sliding_window'
        window_size: Window size for sliding window attention (only used if attention_type='sliding_window').
        gradient_checkpointing: If True, use gradient checkpointing for memory efficiency.
        use_gated_embedding: If True, use gated embeddings instead of standard embeddings.
    """
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    post_norm: bool = False  # Hybrid pre/post-normalization
    qk_norm: bool = False    # Query-Key normalization
    n_latent: int = 384  # Latent space dimension
    attention_type: str = "standard"  # Attention mechanism type
    window_size: int = 64  # Window size for sliding window attention
    # gradient_checkpointing: bool = False  # Gradient checkpointing for memory efficiency
    gradient_checkpointing: bool = True  # Gradient checkpointing for memory efficiency
    use_gated_embedding: bool = False  # Use gated embeddings
    use_gated_lm_head: bool = False # Use gated lm_head
    gated_embedding_bottleneck_factor: int = 4 # Bottleneck factor for GatedEmbedding
    gated_lm_head_bottleneck_factor: int = 4 # Bottleneck factor for GatedLMHead


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Choose embedding type based on configuration
        if config.use_gated_embedding:
            embedding_layer = GatedEmbedding(config.vocab_size, config.n_embd,
                                             bottleneck_factor=config.gated_embedding_bottleneck_factor)
        else:
            embedding_layer = nn.Embedding(config.vocab_size, config.n_embd)

        self.transformer = nn.ModuleDict(
            dict(
                wte=embedding_layer,
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            )
        )
        if config.use_gated_lm_head:
            self.lm_head = GatedLMHead(config.n_embd, config.vocab_size,
                                       bottleneck_factor=config.gated_lm_head_bottleneck_factor)
        else:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Tie weights for both embedding types
        if config.use_gated_embedding and not config.use_gated_lm_head:
            self.transformer.wte.embedding.weight = self.lm_head.weight
        elif config.use_gated_lm_head and not config.use_gated_embedding:
            self.transformer.wte.weight = self.lm_head.lm_head.weight
        elif config.use_gated_embedding and config.use_gated_lm_head:
            self.transformer.wte.embedding.weight = self.lm_head.lm_head.weight
        else:
            self.transformer.wte.weight = (
                self.lm_head.weight
            )  # https://paperswithcode.com/method/weight-tying

    def forward(self, idx, targets=None, return_logits=True):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)  # shape (t)

        # forward the GPT model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        # print(f"torch.cuda.memory_allocated() / 1024 / 1024 = {torch.cuda.memory_allocated() / 1024 / 1024}")
        for block in self.transformer.h:
            # print(f"torch.cuda.memory_allocated() / 1024 / 1024 = {torch.cuda.memory_allocated() / 1024 / 1024}")
            x = block(x)
        x = rmsnorm(x)

        # print(f"torch.cuda.memory_allocated() / 1024 / 1024 = {torch.cuda.memory_allocated() / 1024 / 1024}")

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas
        )
        return optimizer
    
    def clear_attention_caches(self):
        """Clear all attention mask caches to free memory."""
        clear_all_attention_caches()


# -----------------------------------------------------------------------------
# Performance Benchmarking Functions

def benchmark_attention_mechanisms(model, input_data, mechanisms=['standard', 'mla']):
    """Benchmark different attention mechanisms for performance comparison.
    
    Args:
        model: The GPT model to benchmark
        input_data: Input tensor for benchmarking
        mechanisms: List of attention mechanisms to compare
            - 'standard': Standard CausalSelfAttention
            - 'mla': Multi-Head Latent Attention
        
    Returns:
        dict: Performance metrics for each mechanism
    """
    results = {}
    device = input_data.device
    
    for mechanism in mechanisms:
        if mechanism == 'standard':
            # Use standard CausalSelfAttention
            config = model.config
            config.attention_type = 'standard'
        elif mechanism == 'mla':
            # Use Multi-Head Latent Attention
            config = model.config
            config.attention_type = 'latent'
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(input_data, return_logits=False)
        
        # Benchmark
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_data, return_logits=False)
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        avg_time = (end_time - start_time) / 10
        results[mechanism] = {
            'avg_time_ms': avg_time * 1000,
            'throughput_tokens_per_sec': input_data.numel() / avg_time
        }
    
    return results


def measure_memory_usage(model, input_data):
    """Measure memory usage for different model configurations.
    
    Args:
        model: The GPT model to measure
        input_data: Input tensor for measurement
        
    Returns:
        dict: Memory usage metrics
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Measure memory usage during forward pass
    with torch.no_grad():
        _ = model(input_data, return_logits=False)
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    current_memory = torch.cuda.memory_allocated() / 1024 / 1024   # MB
    
    return {
        'peak_memory_mb': peak_memory,
        'current_memory_mb': current_memory,
        'input_tokens': input_data.numel(),
        'memory_per_token_mb': peak_memory / input_data.numel()
    }


# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader


def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print(
            "---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README"
        )
        print(
            "---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try"
        )
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2]  # number of tokens (claimed)
    return ntok  # for now just return the number of tokens


def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2]  # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens


class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert (
            len(self.files) > 0
        ), f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = np.int64(0)
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print0(
            f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files"
        )

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self):  # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()


# -----------------------------------------------------------------------------
# int main

VAL_TOKENS = 1_048_576  # how many tokens of validation data. It's important to keep this fixed for consistent comparisons


def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)


if __name__ == "__main__":
    import time
    import argparse

    print0(f"Running pytorch {torch.version.__version__}")

    parser = argparse.ArgumentParser()
    # file system input / output
    parser.add_argument(
        "--input_bin",
        type=str,
        default="data/fineweb10B/fineweb_train_*.bin",
        help="input .bin to train on",
    )
    parser.add_argument(
        "--input_val_bin",
        type=str,
        default="data/fineweb10B/fineweb_val_*.bin",
        help="input .bin to eval validation loss on",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="output directory to which to write logs and checkpoints",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="d12",
            help=
            "d12|"
            "d12_post_norm|d12_post_norm_qk_norm|d12_mla|"
            "d12_window|d12_window_large|"
            "d12_gemb|d12_ghead|d12_gemb_ghead|"
            "d24|d36|d48",
    )
    # token layout for each step of the optimization
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="batch size, in units of #batch dimensions",
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        type=int,
        default=1,
        help="number of gradient accumulation steps",
    )
    parser.add_argument(
        "--sequence_length", type=int, default=64, help="sequence length"
    )
    # workload (number of steps)
    parser.add_argument(
        "--num_iterations", type=int, default=10, help="number of iterations to run"
    )
    # optimization
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="learning rate warmup iterations",
    )
    parser.add_argument(
        "--warmup_iters", type=int, default=0, help="learning rate warmup iterations"
    )
    parser.add_argument(
        "--warmdown_iters",
        type=int,
        default=0,
        help="learning rate warmdown iterations",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    # evaluation
    parser.add_argument(
        "--val_loss_every",
        type=int,
        default=0,
        help="every how mant steps to evaluate val loss?",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=16,
        help="how many batches of val to average?",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=5000,
        help="every how many steps to save the checkpoint",
    )
    parser.add_argument(
        "--log_wandb",
        action="store_true",
        help="log to wandb",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default="",
        help="path to checkpoint to resume training from",
    )
    args = parser.parse_args()

    # args error checking and convenience variables
    B, T = args.batch_size, args.sequence_length
    assert args.model in {
        "d12", "d12_post_norm", "d12_post_norm_qk_norm", "d12_mla", 
        "d12_window", "d12_window_large", "d12_gemb", "d12_ghead", "d12_gemb_ghead",
        "d24", "d36", "d48",
    }
    # set up DDP (distributed data parallel). torchrun sets this env variable
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    assert (
        args.grad_accumulation_steps % ddp_world_size == 0
    ), "grad_accumulation_steps must be divisible by world size"
    args.grad_accumulation_steps //= (
        ddp_world_size  # each gpu does its fraction of the work
    )
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = 0  # each process gets the exact same seed
    print(f"using device: {device}")

    if args.log_wandb and master_process:
        import wandb
        import datetime

        start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        wandb.init(project="benchmark_gpt2", name=f"gpt2-{args.model} {start_time}")
        wandb.config.update(args)
        wandb.save("train_gpt2_mod.py")
        wandb.save("run_mod.sh")

    tokens_per_iter = B * T * ddp_world_size * args.grad_accumulation_steps
    print0(f"tokens per iteration: {tokens_per_iter:,}")

    # set up a context manager following the desired dtype and device
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    # load tokens
    train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
    val_loader = None
    tokens_per_iter_val = args.val_batch_size * T * ddp_world_size
    assert VAL_TOKENS % tokens_per_iter_val == 0
    val_steps = VAL_TOKENS // tokens_per_iter_val

    val_loader = DistributedDataLoader(
        args.input_val_bin, args.val_batch_size, T, ddp_rank, ddp_world_size
    )
    x, y = train_loader.next_batch()

    # init the model from scratch
    num_vocab = 50257
    
    # Get base model configuration
    base_configs = {
        "d12": GPTConfig(
            vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768
        ),  # 124M GPT-2
        "d12_post_norm": GPTConfig(
            vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768, post_norm=True
        ),  # 124M GPT-2 with hybrid pre/post-normalization
        "d12_post_norm_qk_norm": GPTConfig(
            vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768, post_norm=True, qk_norm=True
        ),  # 124M GPT-2 with hybrid normalization + QK normalization
        "d12_mla": GPTConfig(
            vocab_size=num_vocab, n_layer=12, n_head=20, n_embd=1280,
            post_norm=True, qk_norm=True, attention_type="latent", n_latent=320
        ),  # 124M GPT-2 with MLA + post-norm + QK norm
        "d12_window": GPTConfig(
            vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768,
            # vocab_size=num_vocab, n_layer=12, n_head=8, n_embd=512,
            post_norm=True, qk_norm=True, attention_type="sliding_window", window_size=32
        ),  # 124M GPT-2 with sliding window attention
        "d12_window_large": GPTConfig(
            vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768,
            post_norm=True, qk_norm=True, attention_type="sliding_window", window_size=64
        ),  # 124M GPT-2 with large sliding window attention
        "d12_gemb": GPTConfig(
            vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768, use_gated_embedding=True
        ),  # 124M GPT-2 with gated embeddings
        "d12_ghead": GPTConfig(
            vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768, use_gated_lm_head=True
        ),  # 124M GPT-2 with gated lm_head
        "d12_gemb_ghead": GPTConfig(
            vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768, 
            use_gated_lm_head=True, use_gated_embedding=True
        ),  # 124M GPT-2 with gated lm_head and gated embeddings
        "d24": GPTConfig(vocab_size=num_vocab, n_layer=24, n_head=16, n_embd=1024),
        "d36": GPTConfig(vocab_size=num_vocab, n_layer=36, n_head=20, n_embd=1280),
        "d48": GPTConfig(vocab_size=num_vocab, n_layer=48, n_head=25, n_embd=1600),
    }
    
    model_config = base_configs[args.model]
    
    model = GPT(model_config)
    model = model.train().cuda().to(dtype=torch.bfloat16)
    if hasattr(config, "coordinate_descent_tuning"):
        config.coordinate_descent_tuning = True  # suggested by @Chillee
    print0("compiling the model...")
    model = torch.compile(
        model
    )  # NOTE: this might cause issues depending on your GPU, consider turning it off

    # here we wrap model into DDP container
    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module  # always contains the "raw" unwrapped model

    # init the optimizer
    optimizer = raw_model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
        device_type=device,
    )

    # checkpoint resumption
    start_step = 0
    run_id = str(uuid.uuid4())
    
    if args.resume_from:
        print0(f"Resuming training from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        
        # Load model state
        raw_model.load_state_dict(checkpoint["model"])
        print0("✓ Loaded model state")
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint["optimizer"])
        print0("✓ Loaded optimizer state")
        
        # Resume from the next step after the checkpoint
        start_step = checkpoint.get("step", 0) + 1
        print0(f"✓ Resuming from step {start_step}")
        
        # Use the same run_id to continue logging to the same directory
        if "run_id" in checkpoint:
            run_id = checkpoint["run_id"]
            print0(f"✓ Continuing run_id: {run_id}")
        
        # Verify that critical training parameters match
        checkpoint_args = checkpoint.get("args", {})
        critical_params = ["model", "batch_size", "sequence_length", "grad_accumulation_steps"]
        for param in critical_params:
            checkpoint_val = checkpoint_args.get(param)
            current_val = getattr(args, param)
            if checkpoint_val is not None and checkpoint_val != current_val:
                print0(f"WARNING: {param} mismatch - checkpoint: {checkpoint_val}, current: {current_val}")
        
        del checkpoint  # Free memory
        torch.cuda.empty_cache()

    # learning rate decay scheduler (linear warmup and warmdown)
    def get_lr(it):
        assert it <= args.num_iterations
        # 1) linear warmup for warmup_iters steps
        if it < args.warmup_iters:
            return args.learning_rate * (it + 1) / args.warmup_iters
        # 2) constant lr for a while
        elif it < args.num_iterations - args.warmdown_iters:
            return args.learning_rate
        # 3) linear warmdown
        else:
            decay_ratio = (args.num_iterations - it) / args.warmdown_iters
            return args.learning_rate * decay_ratio

    # create the logging directory if it does not exist
    logfile = None
    if master_process and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        logfile = os.path.join(args.output_dir, "%s.log" % run_id)
        # create the log file or append if resuming
        if not args.resume_from:
            # wipe it clean for new runs
            with open(logfile, "w") as f:
                pass
        else:
            print0(f"Appending to existing log file: {logfile}")

    training_time_ms = 0.0
    # start the clock
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # begin training
    for step in range(start_step, args.num_iterations + 1):
        last_step = step == args.num_iterations

        # once in a while evaluate the validation dataset
        if args.val_loss_every > 0 and (step % args.val_loss_every == 0 or last_step):
            # stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            model.eval()
            val_loader.reset()  # reset the val loader so that it starts from the beginning
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(val_steps):  # always fiexed number of validation steps
                    x_val, y_val = val_loader.next_batch()
                    _, loss = model(x_val, y_val, return_logits=False)
                    val_loss += loss
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
                val_loss /= val_steps
            # log to console and to file
            print0(f"step:{step}/{args.num_iterations} | val loss {val_loss:.6f}")
            if master_process:
                if args.log_wandb:
                    wandb.log({"val_loss": val_loss}, step=step * tokens_per_iter)
                    wandb.log({"time": training_time_ms}, step=step * tokens_per_iter)
                if logfile is not None:
                    with open(logfile, "a") as f:
                        f.write("s:%d val:%f\n" % (step, val_loss))

            # restart the clock
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        # bit confusing: we want to make sure to eval on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        if last_step:
            break

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        train_loss = torch.zeros(1, device=device)
        for micro_step in range(args.grad_accumulation_steps):
            model.require_backward_grad_sync = (
                micro_step == args.grad_accumulation_steps - 1
            )  # sync only on last micro step to avoid overhead
            # forward pass
            with ctx:
                _, loss = model(x, y, return_logits=False)
                loss = (
                    loss / args.grad_accumulation_steps
                )  # scale loss for gradient accumulation
                train_loss += loss.detach()
            # advance the dataset for the next batch
            x, y = train_loader.next_batch()
            # backward pass
            loss.backward()

        train_loss /= (
            args.grad_accumulation_steps
        )  # average the loss over all micro steps

        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        # step the optimizer
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.

        torch.cuda.synchronize()
        # time and print
        approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
        # the 0th iteration is often an outlier (much slower) => skip logging it
        # tokens_per_second = ddp_world_size * B * T / (t1-t0)
        dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
        lossf = train_loss.item()  # keep track of the mean loss
        print0(
            f"step:{step}/{args.num_iterations} | loss {lossf:.6f} | train_time:{approx_training_time_ms/1000:.2f}s | step_avg:{approx_training_time_ms/(step+1):.2f}ms"
        )
        # log to logile
        if master_process:
            if args.log_wandb:
                wandb.log({"train_loss": lossf}, step=step * tokens_per_iter)
            if logfile is not None:
                with open(logfile, "a") as f:
                    f.write("s:%d trn:%f\n" % (step, lossf))

        if master_process and (step + 1) % args.save_every == 0:
            log = dict(
                model=raw_model.state_dict(),
                optimizer=optimizer.state_dict(),
                step=step,
                code=code,
                args=args.__dict__,
                run_id=run_id,
            )
            os.makedirs("logs/%s" % run_id, exist_ok=True)
            torch.save(log, "logs/%s/model_step%06d.pt" % (run_id, step))

    print0(
        f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB"
    )

    # -------------------------------------------------------------------------

    if master_process:
        log = dict(
            model=raw_model.state_dict(),
            optimizer=optimizer.state_dict(),
            step=args.num_iterations,
            code=code,
            args=args.__dict__,
            run_id=run_id,
        )
        os.makedirs("logs/%s" % run_id, exist_ok=True)
        torch.save(log, "logs/%s/final.pt" % run_id)

    # -------------------------------------------------------------------------
    # clean up nice
    destroy_process_group()
