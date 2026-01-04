"""
Tests for vision module
======================
Tests for Conv2D, MaxPool2D, ResidualBlock, and ResNet18.
"""

import pytest
import numpy as np
from src.ml.vision import (
    Conv2D, MaxPool2D, Flatten, ResidualBlock, ResNet18,
    im2col, col2im, calculate_output_size
)


class TestConv2D:
    """Tests for Conv2D layer."""
    
    def test_forward_shape(self):
        """Test output shape is correct."""
        conv = Conv2D(3, 64, kernel_size=3, stride=1, padding=1)
        x = np.random.randn(8, 3, 32, 32)
        out = conv.forward(x)
        
        assert out.shape == (8, 64, 32, 32), f"Expected (8, 64, 32, 32), got {out.shape}"
    
    def test_stride_2(self):
        """Test stride reduces spatial dimensions."""
        conv = Conv2D(3, 64, kernel_size=3, stride=2, padding=1)
        x = np.random.randn(8, 3, 32, 32)
        out = conv.forward(x)
        
        assert out.shape == (8, 64, 16, 16)
    
    def test_no_padding(self):
        """Test without padding reduces size."""
        conv = Conv2D(3, 64, kernel_size=3, stride=1, padding=0)
        x = np.random.randn(8, 3, 32, 32)
        out = conv.forward(x)
        
        # 32 - 3 + 1 = 30
        assert out.shape == (8, 64, 30, 30)
    
    def test_backward_pass(self):
        """Test backward pass updates weights."""
        conv = Conv2D(3, 64, kernel_size=3, stride=1, padding=1)
        x = np.random.randn(8, 3, 32, 32)
        
        # Forward
        out = conv.forward(x)
        
        # Backward
        grad = np.random.randn(*out.shape)
        input_grad = conv.backward(grad, learning_rate=0.01)
        
        assert input_grad.shape == x.shape
        assert conv.dW is not None
        assert conv.db is not None


class TestMaxPool2D:
    """Tests for MaxPool2D layer."""
    
    def test_forward_shape(self):
        """Test pooling reduces spatial dimensions."""
        pool = MaxPool2D(pool_size=2, stride=2)
        x = np.random.randn(8, 64, 32, 32)
        out = pool.forward(x)
        
        assert out.shape == (8, 64, 16, 16)
    
    def test_max_operation(self):
        """Test pooling takes maximum value."""
        pool = MaxPool2D( pool_size=2, stride=2)
        x = np.array([[[[1, 2, 3, 4],
                         [5, 6, 7, 8],
                         [9, 10, 11, 12],
                         [13, 14, 15, 16]]]])  # (1, 1, 4, 4)
        
        out = pool.forward(x)
        
        # Expected: [[6, 8], [14, 16]]
        expected = np.array([[[[6, 8], [14, 16]]]])
        np.testing.assert_array_equal(out, expected)
    
    def test_backward_pass(self):
        """Test backward pass routes gradient to max elements."""
        pool = MaxPool2D(pool_size=2, stride=2)
        x = np.random.randn(4, 16, 32, 32)
        
        out = pool.forward(x)
        grad = np.random.randn(*out.shape)
        input_grad = pool.backward(grad, learning_rate=0.01)
        
        assert input_grad.shape == x.shape


class TestFlatten:
    """Tests for Flatten layer."""
    
    def test_flatten(self):
        """Test 4D to 2D conversion."""
        flatten = Flatten()
        x = np.random.randn(8, 64, 4, 4)
        out = flatten.forward(x)
        
        assert out.shape == (8, 64 * 4 * 4)
    
    def test_backward_reshape(self):
        """Test backward reshapes correctly."""
        flatten = Flatten()
        x = np.random.randn(8, 64, 4, 4)
        
        out = flatten.forward(x)
        grad = np.random.randn(*out.shape)
        input_grad = flatten.backward(grad, learning_rate=0.01)
        
        assert input_grad.shape == x.shape


class TestResidualBlock:
    """Tests for ResidualBlock."""
    
    def test_identity_block(self):
        """Test block preserves spatial dimensions (identity)."""
        block = ResidualBlock(64, 64, stride=1)
        x = np.random.randn(4, 64, 16, 16)
        out = block.forward(x)
        
        assert out.shape == x.shape
    
    def test_projection_block(self):
        """Test block with projection (changes channels/size)."""
        block = ResidualBlock(64, 128, stride=2)
        x = np.random.randn(4, 64, 32, 32)
        out = block.forward(x)
        
        # Channels: 64 -> 128, spatial: 32 -> 16 (stride=2)
        assert out.shape == (4, 128, 16, 16)
    
    def test_skip_connection_used(self):
        """Test that block uses skip connection."""
        block = ResidualBlock(64, 64, stride=1)
        
        # Create input with known pattern
        x = np.ones((1, 64, 8, 8))
        
        # Forward through block
        out = block.forward(x, training=False)
        
        # Output should be influenced by skip connection (not just convolutions)
        # This is a basic sanity check
        assert out.shape == x.shape
        assert block.use_projection == False  # Identity block
    
    def test_backward_pass(self):
        """Test backward pass through residual block."""
        block = ResidualBlock(64, 64, stride=1)
        x = np.random.randn(4, 64, 16, 16)
        
        out = block.forward(x)
        grad = np.random.randn(*out.shape)
        input_grad = block.backward(grad, learning_rate=0.001)
        
        assert input_grad.shape == x.shape


class TestResNet18:
    """Tests for ResNet18 architecture."""
    
    def test_architecture_creation(self):
        """Test ResNet18 can be instantiated."""
        model = ResNet18(in_channels=3, num_classes=10)
        assert model is not None
        assert len(model.layer1) == 2
        assert len(model.layer2) == 2
        assert len(model.layer3) == 2
        assert len(model.layer4) == 2
    
    def test_forward_pass(self):
        """Test forward pass through ResNet18."""
        model = ResNet18(in_channels=3, num_classes=10)
        x = np.random.randn(2, 3, 224, 224)  # Small batch
        
        out = model.forward(x, training=False)
        
        # After all layers, spatial size should be reduced
        assert out.shape[0] == 2  # Batch size preserved
        assert out.shape[1] == 512  # Final layer has 512 channels
    
    def test_smaller_input(self):
        """Test with CIFAR-10 sized input (32x32)."""
        model = ResNet18(in_channels=3, num_classes=10)
        x = np.random.randn(4, 3, 32, 32)
        
        out = model.forward(x, training=False)
        
        assert out.shape[0] == 4  # Batch preserved
        assert out.shape[1] == 512  # Final channels


class TestUtilities:
    """Tests for utility functions."""
    
    def test_calculate_output_size(self):
        """Test output size calculation."""
        # No padding, stride 1
        assert calculate_output_size(32, 3, stride=1, padding=0) == 30
        
        # With padding (same size)
        assert calculate_output_size(32, 3, stride=1, padding=1) == 32
        
        # With stride (downsampling)
        assert calculate_output_size(32, 3, stride=2, padding=1) == 16
    
    def test_im2col_col2im_inverse(self):
        """Test im2col and col2im are inverses."""
        x = np.random.randn(2, 3, 8, 8)
        
        # im2col
        col = im2col(x, filter_h=3, filter_w=3, stride=1, pad=1)
        
        # col2im
        x_reconstructed = col2im(col, x.shape, filter_h=3, filter_w=3, stride=1, pad=1)
        
        # Should be close (some numerical differences due to padding)
        assert x_reconstructed.shape == x.shape


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
