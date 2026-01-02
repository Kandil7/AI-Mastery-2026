"""Quick verification of attention module functionality."""
import sys
sys.path.insert(0, '.')

import torch
from src.llm.attention import (
    CausalSelfAttention,
    AttentionWithRoPE,
    GroupedQueryAttention,
    FlashAttention,
    create_attention_mechanism
)

print("=" * 50)
print("ATTENTION MODULE VERIFICATION")
print("=" * 50)

# Test input
batch_size, seq_len, d_model = 2, 16, 64
x = torch.randn(batch_size, seq_len, d_model)

# 1. Causal Self-Attention
print("\n[1] CausalSelfAttention...")
causal = CausalSelfAttention(d_model, num_heads=4, block_size=32)
out = causal(x)
print(f"    Input: {x.shape} -> Output: {out.shape} ✓")

# 2. RoPE Attention
print("\n[2] AttentionWithRoPE...")
rope = AttentionWithRoPE(d_model, num_heads=4, max_len=128)
out = rope(x)
print(f"    Input: {x.shape} -> Output: {out.shape} ✓")

# 3. Grouped Query Attention
print("\n[3] GroupedQueryAttention...")
gqa = GroupedQueryAttention(d_model, num_heads=8, num_kv_heads=2)
out = gqa(x)
print(f"    Input: {x.shape} -> Output: {out.shape} ✓")

# 4. Flash Attention
print("\n[4] FlashAttention...")
flash = FlashAttention(d_model, num_heads=4)
out = flash(x)
print(f"    Input: {x.shape} -> Output: {out.shape} ✓")

# 5. Factory function
print("\n[5] create_attention_mechanism factory...")
attn = create_attention_mechanism('rope', d_model=d_model, num_heads=4)
print(f"    Created: {type(attn).__name__} ✓")

print("\n" + "=" * 50)
print("ALL ATTENTION MECHANISMS VERIFIED ✓")
print("=" * 50)
