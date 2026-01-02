"""
Pure Python Implementation of Optimization Algorithms ("White-Box").
Uses src.core.linear_algebra.Vector instead of NumPy.
"""
import math
from typing import Callable, List, Tuple
from src.core.linear_algebra import Vector

class WhiteBoxOptimizer:
    def __init__(self, learning_rate: float = 0.01):
        self.lr = learning_rate

    def step(self, params: Vector, gradients: Vector) -> Vector:
        raise NotImplementedError

class SGD(WhiteBoxOptimizer):
    """Stochastic Gradient Descent"""
    def step(self, params: Vector, gradients: Vector) -> Vector:
        # params = params - lr * gradients
        return params - (gradients * self.lr)

class Adam(WhiteBoxOptimizer):
    """
    Adam Optimizer from scratch.
    m_t = beta1 * m_{t-1} + (1-beta1) * g
    v_t = beta2 * v_{t-1} + (1-beta2) * g^2
    """
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params: Vector, gradients: Vector) -> Vector:
        if self.m is None:
            self.m = Vector([0.0] * params.size)
            self.v = Vector([0.0] * params.size)
        
        self.t += 1
        
        # 1. Update biased first moment estimate
        # m = beta1 * m + (1 - beta1) * g
        self.m = (self.m * self.beta1) + (gradients * (1 - self.beta1))
        
        # 2. Update biased second raw moment estimate
        # v = beta2 * v + (1 - beta2) * g^2
        # We need element-wise square for gradients.
        g_squared = Vector([g**2 for g in gradients.data])
        self.v = (self.v * self.beta2) + (g_squared * (1 - self.beta2))
        
        # 3. Compute bias-corrected first moment estimate
        # m_hat = m / (1 - beta1^t)
        m_hat = self.m * (1.0 / (1 - self.beta1 ** self.t))
        
        # 4. Compute bias-corrected second raw moment estimate
        # v_hat = v / (1 - beta2^t)
        v_hat = self.v * (1.0 / (1 - self.beta2 ** self.t))
        
        # 5. Update parameters
        # theta = theta - lr * m_hat / (sqrt(v_hat) + epsilon)
        # Element-wise division/sqrt
        update_data = []
        for mh, vh in zip(m_hat.data, v_hat.data):
             update_data.append(self.lr * mh / (math.sqrt(vh) + self.epsilon))
             
        return params - Vector(update_data)

def numerical_gradient(func: Callable[[Vector], float], params: Vector, h: float = 1e-5) -> Vector:
    """Compute gradient using Finite Difference Method."""
    grads = []
    for i in range(params.size):
        # f(x + h)
        orig_val = params.data[i]
        params.data[i] = orig_val + h
        pos = func(params)
        
        # f(x - h)
        params.data[i] = orig_val - h
        neg = func(params)
        
        # Central difference: (f(x+h) - f(x-h)) / 2h
        grads.append((pos - neg) / (2 * h))
        
        # Reset
        params.data[i] = orig_val
        
    return Vector(grads)
