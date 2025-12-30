#!/usr/bin/env python
# Test script to verify attention module functionality

import torch
import numpy as np
from src.llm.attention import (
    scaled_dot_product_attention,
    MultiHeadAttention,
    TransformerBlock,
    FeedForwardNetwork,
    LayerNorm
)

def test_scaled_dot_product_attention():
    """Test scaled dot product attention function."""
    print("Testing scaled_dot_product_attention...")
    Q = torch.randn(2, 10, 64)  # batch=2, seq=10, d_k=64
    K = torch.randn(2, 15, 64)  # batch=2, seq=15, d_k=64
    V = torch.randn(2, 15, 64)  # batch=2, seq=15, d_v=64

    output, attention_weights = scaled_dot_product_attention(Q, K, V)

    assert output.shape == (2, 10, 64), f"Expected (2, 10, 64), got {output.shape}"
    assert attention_weights.shape == (2, 10, 15), f"Expected (2, 10, 15), got {attention_weights.shape}"
    print("✓ scaled_dot_product_attention test passed")


def test_multi_head_attention():
    """Test MultiHeadAttention class."""
    print("Testing MultiHeadAttention...")
    mha = MultiHeadAttention(d_model=512, num_heads=8)

    # Self-attention input
    X = torch.randn(4, 20, 512)  # batch=4, seq=20, d_model=512

    output = mha(X, X, X)

    assert output.shape == (4, 20, 512), f"Expected (4, 20, 512), got {output.shape}"
    print("✓ MultiHeadAttention test passed")


def test_multi_head_attention_num_heads_divisibility():
    """Test that d_model must be divisible by num_heads."""
    print("Testing MultiHeadAttention divisibility constraint...")
    try:
        mha = MultiHeadAttention(d_model=511, num_heads=8)  # 511 not divisible by 8
        assert False, "Should have raised an assertion error"
    except AssertionError:
        print("✓ MultiHeadAttention divisibility test passed")


def test_feed_forward_network():
    """Test FeedForwardNetwork class."""
    print("Testing FeedForwardNetwork...")
    ffn = FeedForwardNetwork(d_model=256, d_ff=512, activation='relu')

    X = torch.randn(3, 10, 256)  # batch=3, seq=10, d_model=256

    output = ffn(X)

    assert output.shape == (3, 10, 256), f"Expected (3, 10, 256), got {output.shape}"
    print("✓ FeedForwardNetwork test passed")


def test_layer_norm():
    """Test LayerNorm class."""
    print("Testing LayerNorm...")
    ln = LayerNorm(d_model=128)

    X = torch.randn(5, 20, 128)  # batch=5, seq=20, d_model=128

    output = ln(X)

    assert output.shape == (5, 20, 128), f"Expected (5, 20, 128), got {output.shape}"
    print("✓ LayerNorm test passed")


def test_transformer_block():
    """Test TransformerBlock class."""
    print("Testing TransformerBlock...")
    block = TransformerBlock(d_model=256, num_heads=8, d_ff=512)

    X = torch.randn(2, 15, 256)  # batch=2, seq=15, d_model=256

    output = block(X)

    assert output.shape == (2, 15, 256), f"Expected (2, 15, 256), got {output.shape}"
    print("✓ TransformerBlock test passed")


if __name__ == "__main__":
    print("Running attention module tests...")
    
    test_scaled_dot_product_attention()
    test_multi_head_attention()
    test_multi_head_attention_num_heads_divisibility()
    test_feed_forward_network()
    test_layer_norm()
    test_transformer_block()
    
    print("\nAll attention module tests passed! ✓")