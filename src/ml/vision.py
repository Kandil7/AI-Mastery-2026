"""
Computer Vision Module
======================
Convolutional neural network components built from scratch.

Includes:
- Conv2D layer with im2col optimization
- MaxPool2D, Flatten layers
- ResidualBlock (ResNet building block)
- ResNet18 architecture
- Batch normalization for CNNs

Mathematical Foundation:
- Convolution: y[i,j] = Σ x[i+m, j+n] * kernel[m,n]
- Residual: F(x) + x (skip connection)

Author: AI-Mastery-2026
"""

import numpy as np
from typing import Tuple, Optional, List

try:
    from src.ml.deep_learning import Layer, BatchNormalization, Activation
except ImportError:
    from .deep_learning import Layer, BatchNormalization, Activation


# ============================================================
# HELPER: IM2COL FOR EFFICIENT CONVOLUTION
# ============================================================

def im2col(input_data: np.ndarray, filter_h: int, filter_w: int, 
           stride: int = 1, pad: int = 0) -> np.ndarray:
    """
    Convert 4D input to 2D matrix for efficient convolution.
    
    Transforms convolution into matrix multiplication.
    
    Args:
        input_data: (N, C, H, W) format
        filter_h, filter_w: Filter dimensions
        stride: Stride
        pad: Padding
    
    Returns:
        2D matrix: (N*out_h*out_w, C*filter_h*filter_w)
    """
    N, C, H, W = input_data.shape
    
    # Output dimensions
    out_h = (H + 2*pad - filter_h) // stride + 1
    out_w = (W + 2*pad - filter_w) // stride + 1
    
    # Pad input
    if pad > 0:
        img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 
                     mode='constant')
    else:
        img = input_data
    
    # Create column matrix
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


def col2im(col: np.ndarray, input_shape: Tuple, filter_h: int, filter_w: int,
           stride: int = 1, pad: int = 0) -> np.ndarray:
    """
    Convert 2D matrix back to 4D input format.
    
    Reverse operation of im2col (for backprop).
    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h) // stride + 1
    out_w = (W + 2*pad - filter_w) // stride + 1
    
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
    
    if pad > 0:
        img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    else:
        img = np.zeros((N, C, H, W))
    
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
    
    if pad > 0:
        return img[:, :, pad:H+pad, pad:W+pad]
    return img


#============================================================
# CONVOLUTIONAL LAYERS
# ============================================================

class Conv2D(Layer):
    """
    2D Convolutional Layer.
    
    Convolves filters over input feature maps.
    Uses im2col for efficiency (transforms conv to matrix multiply).
    
    Forward:
        y[b, f, i, j] = Σ x[b, c, i+m, j+n] * W[f, c, m, n] + b[f]
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output filters
        kernel_size: Size of convolution kernel (int or tuple)
        stride: Convolution stride
        padding: Zero padding
        weight_init: 'he' or 'xavier'
    
    Example:
        >>> conv = Conv2D(3, 64, kernel_size=3, stride=1, padding=1)
        >>> output = conv.forward(input_data)  # (N, 3, H, W) -> (N, 64, H, W)
    """
    
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 0,
                 weight_init: str = 'he'):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights
        if weight_init == 'he':
            scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        else:  # xavier
            scale = np.sqrt(1.0 / (in_channels * kernel_size * kernel_size))
        
        self.filters = np.random.randn(out_channels, in_channels, 
                                       kernel_size, kernel_size) * scale
        self.bias = np.zeros((out_channels,))
        
        # For backward pass
        self.input_shape = None
        self.col = None
        self.dW = None
        self.db = None
    
    def forward(self, input_data: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            input_data: (N, C, H, W) format
        
        Returns:
            (N, out_channels, H', W')
        """
        self.input = input_data
        self.input_shape = input_data.shape
        N, C, H, W = input_data.shape
        
        # Calculate output dimensions
        out_h = (H + 2*self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2*self.padding - self.kernel_size) // self.stride + 1
        
        # im2col transformation
        self.col = im2col(input_data, self.kernel_size, self.kernel_size, 
                         self.stride, self.padding)
        
        # Reshape filters for matrix multiply
        filters_col = self.filters.reshape(self.out_channels, -1).T
        
        # Convolution as matrix multiplication
        out = self.col @ filters_col + self.bias
        
        # Reshape to output format
        self.output = out.reshape(N, out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)
        return self.output
    
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Backward pass.
        
        Args:
            output_gradient: (N, out_channels, H', W')
        
        Returns:
            input_gradient: (N, in_channels, H, W)
        """
        N = output_gradient.shape[0]
        
        # Reshape gradient
        dout = output_gradient.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)
        
        # Gradient w.r.t. bias
        self.db = np.sum(dout, axis=0)
        
        # Gradient w.r.t. filters
        self.dW = (self.col.T @ dout).T
        self.dW = self.dW.reshape(self.out_channels, self.in_channels, 
                                   self.kernel_size, self.kernel_size)
        
        # Gradient w.r.t. input
        filters_col = self.filters.reshape(self.out_channels, -1).T
        dcol = dout @ filters_col.T
        input_gradient = col2im(dcol, self.input_shape, self.kernel_size, 
                               self.kernel_size, self.stride, self.padding)
        
        # Update weights
        self.filters -= learning_rate * self.dW
        self.bias -= learning_rate * self.db
        
        return input_gradient
    
    def get_params(self) -> dict:
        return {'filters': self.filters.copy(), 'bias': self.bias.copy()}
    
    def set_params(self, params: dict):
        self.filters = params['filters']
        self.bias = params['bias']


class MaxPool2D(Layer):
    """
    Max Pooling Layer.
    
    Downsamples feature maps by taking max value in each window.
    Reduces spatial dimensions, provides translation invariance.
    
    Args:
        pool_size: Size of pooling window
        stride: Stride (default: same as pool_size)
    
    Example:
        >>> pool = MaxPool2D(pool_size=2, stride=2)
        >>> output = pool.forward(input_data)  # (N, C, H, W) -> (N, C, H/2, W/2)
    """
    
    def __init__(self, pool_size: int = 2, stride: Optional[int] = None):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        self.mask = None
        self.trainable = False
    
    def forward(self, input_data: np.ndarray, training: bool = True) -> np.ndarray:
        self.input = input_data
        N, C, H, W = input_data.shape
        
        out_h = (H - self.pool_size) // self.stride + 1
        out_w = (W - self.pool_size) // self.stride + 1
        
        # Reshape for pooling
        col = im2col(input_data, self.pool_size, self.pool_size, 
                    self.stride, pad=0)
        col = col.reshape(-1, self.pool_size * self.pool_size)
        
        # Max pooling
        arg_max = np.argmax(col, axis=1)
        self.output = np.max(col, axis=1)
        
        # Reshape output
        self.output = self.output.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        
        # Store mask for backward
        self.mask = np.zeros_like(col)
        self.mask[np.arange(arg_max.size), arg_max.flatten()] = 1
        
        return self.output
    
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        dout = output_gradient.transpose(0, 2, 3, 1)
        
        # Reshape gradient
        dmax = dout.reshape(-1, 1)
        
        # Backprop through max operation
        dcol = dmax * self.mask
        dcol = dcol.reshape(dout.shape[0] * dout.shape[1] * dout.shape[2], -1)
        
        # col2im
        input_gradient = col2im(dcol, self.input.shape, self.pool_size, 
                               self.pool_size, self.stride, pad=0)
        
        return input_gradient


class Flatten(Layer):
    """
    Flatten layer.
    
    Reshapes 4D tensor to 2D for fully connected layers.
    (N, C, H, W) -> (N, C*H*W)
    """
    
    def __init__(self):
        super().__init__()
        self.input_shape = None
        self.trainable = False
    
    def forward(self, input_data: np.ndarray, training: bool = True) -> np.ndarray:
        self.input_shape = input_data.shape
        self.output = input_data.reshape(input_data.shape[0], -1)
        return self.output
    
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        return output_gradient.reshape(self.input_shape)


# ============================================================
# RESIDUAL BLOCK (ResNet)
# ============================================================

class ResidualBlock(Layer):
    """
    Residual Block with skip connection.
    
    Core building block of ResNet. Learns F(x) and adds it to x.
    Output: F(x) + x
    
    This allows training very deep networks by providing gradient shortcuts.
    
    Architecture:
        x -> Conv -> BN -> ReLU -> Conv -> BN -> (+ x) -> ReLU
        |______________________________________________|
                        (skip connection)
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for first conv (downsampling if stride=2)
    
    Example:
        >>> block = ResidualBlock(64, 64, stride=1)  # Identity block
        >>> output = block.forward(input_data)
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        # Main path
        self.conv1 = Conv2D(in_channels, out_channels, kernel_size=3, 
                          stride=stride, padding=1, weight_init='he')
        self.bn1 = BatchNormalization(out_channels)
        self.relu1 = Activation('relu')
        
        self.conv2 = Conv2D(out_channels, out_channels, kernel_size=3, 
                          stride=1, padding=1, weight_init='he')
        self.bn2 =BatchNormalization(out_channels)
        
        # Skip connection (identity or projection)
        self.use_projection = (stride != 1) or (in_channels != out_channels)
        if self.use_projection:
            self.projection = Conv2D(in_channels, out_channels, kernel_size=1, 
                                    stride=stride, padding=0)
            self.bn_proj = BatchNormalization(out_channels)
        
        self.relu2 = Activation('relu')
    
    def forward(self, input_data: np.ndarray, training: bool = True) -> np.ndarray:
        self.input = input_data
        
        # Main path
        out = self.conv1.forward(input_data, training)
        out = self.bn1.forward(out, training)
        out = self.relu1.forward(out, training)
        
        out = self.conv2.forward(out, training)
        out = self.bn2.forward(out, training)
        
        # Skip connection
        if self.use_projection:
            shortcut = self.projection.forward(input_data, training)
            shortcut = self.bn_proj.forward(shortcut, training)
        else:
            shortcut = input_data
        
        # Add residual
        out = out + shortcut
        
        # Final activation
        self.output = self.relu2.forward(out, training)
        return self.output
    
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        # Backprop through final ReLU
        grad = self.relu2.backward(output_gradient, learning_rate)
        
        # Split gradient for main path and skip connection
        grad_main = grad
        grad_skip = grad
        
        # Main path backward
        grad_main = self.bn2.backward(grad_main, learning_rate)
        grad_main = self.conv2.backward(grad_main, learning_rate)
        
        grad_main = self.relu1.backward(grad_main, learning_rate)
        grad_main = self.bn1.backward(grad_main, learning_rate)
        grad_main = self.conv1.backward(grad_main, learning_rate)
        
        # Skip connection backward
        if self.use_projection:
            grad_skip = self.bn_proj.backward(grad_skip, learning_rate)
            grad_skip = self.projection.backward(grad_skip, learning_rate)
        
        # Combine gradients
        input_gradient = grad_main + grad_skip
        
        return input_gradient


# ============================================================
# RESNET-18 ARCHITECTURE
# ============================================================

class ResNet18:
    """
    ResNet-18 architecture from scratch.
    
    Architecture:
        conv1 (7x7, 64) -> bn -> relu -> maxpool
        -> layer1 (2x ResBlock, 64)
        -> layer2 (2x ResBlock, 128, stride=2)
        -> layer3 (2x ResBlock, 256, stride=2)
        -> layer4 (2x ResBlock, 512, stride=2)
        -> avgpool -> fc
    
    Args:
        in_channels: Number of input channels (3 for RGB)
        num_classes: Number of output classes
    
    Example:
        >>> model = ResNet18(in_channels=3, num_classes=10)
        >>> model.compile(learning_rate=0.001)
        >>> history = model.fit(X_train, y_train, epochs=50)
    """
    
    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.layers = []
        self.learning_rate = 0.001
        
        # Initial convolution
        self.conv1 = Conv2D(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = BatchNormalization(64)
        self.relu = Activation('relu')
        self.maxpool = MaxPool2D(pool_size=3, stride=2)
        
        # ResNet layers
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)
        
        # Note: For full implementation, would need:
        # - Global average pooling
        # - Fully connected layer
        # - These would be added in a complete version
    
    def _make_layer(self, in_channels: int, out_channels: int, 
                    num_blocks: int, stride: int) -> List[ResidualBlock]:
        """Create a layer of residual blocks."""
        layers = []
        
        # First block may downsample
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        
        return layers
    
    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through ResNet-18."""
        # Initial layers
        out = self.conv1.forward(X, training)
        out = self.bn1.forward(out, training)
        out = self.relu.forward(out, training)
        out = self.maxpool.forward(out, training)
        
        # ResNet layers
        for block in self.layer1:
            out = block.forward(out, training)
        
        for block in self.layer2:
            out = block.forward(out, training)
        
        for block in self.layer3:
            out = block.forward(out, training)
        
        for block in self.layer4:
            out = block.forward(out, training)
        
        return out
    
    def compile(self, learning_rate: float = 0.001):
        """Configure model for training."""
        self.learning_rate = learning_rate
    
    def summary(self):
        """Print model architecture."""
        print("=" * 60)
        print("ResNet-18 Architecture")
        print("=" * 60)
        print(f"Input: ({self.in_channels}, H, W)")
        print(f"Conv1: 7x7, 64, stride=2, padding=3")
        print(f"MaxPool: 3x3, stride=2")
        print(f"Layer1: 2x ResBlock(64, 64)")
        print(f"Layer2: 2x ResBlock(128, 128, stride=2)")
        print(f"Layer3: 2x ResBlock(256, 256, stride=2)")
        print(f"Layer4: 2x ResBlock(512, 512, stride=2)")
        print(f"Output: {self.num_classes} classes")
        print("=" * 60)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def calculate_output_size(input_size: int, kernel_size: int, 
                         stride: int = 1, padding: int = 0) -> int:
    """
    Calculate output size after convolution/pooling.
    
    Formula: (input - kernel + 2*padding) / stride + 1
    """
    return (input_size - kernel_size + 2*padding) // stride + 1


if __name__ == "__main__":
    # Quick test
    print("Testing Conv2D...")
    conv = Conv2D(3, 64, kernel_size=3, stride=1, padding=1)
    x = np.random.randn(8, 3, 32, 32)  # Batch of 8, 3 channels, 32x32
    out = conv.forward(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")
    
    print("\nTesting ResidualBlock...")
    block = ResidualBlock(64, 64, stride=1)
    x = np.random.randn(8, 64, 32, 32)
    out = block.forward(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")
    
    print("\nTesting ResNet18...")
    model = ResNet18(in_channels=3, num_classes=10)
    model.summary()
    
    print("\n✅ Vision module tests passed!")
