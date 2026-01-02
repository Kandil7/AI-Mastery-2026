"""
Integration in Energy-Efficient Machine Learning Systems

This module implements energy-aware integration methods for resource-constrained
environments like IoT devices, mobile, and edge computing.

Energy consumption model:
E_total = E_compute + E_memory + E_communication

Key insight: Reduce integration operations while maintaining accuracy
to minimize power consumption.

Industrial Case Study: Google DeepMind for Data Centers
- Challenge: Data centers consume 1-2% of global electricity
- Solution: Energy-efficient integration in predictive models
- Results: 40% cooling reduction, $150M/year saved
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
import math


@dataclass
class IntegrationResult:
    """Result from energy-efficient integration."""
    value: float
    n_evaluations: int
    execution_time: float
    energy_cost: float
    method: str
    error_estimate: float = 0.0


@dataclass
class DeviceProfile:
    """Energy profile for different device types."""
    name: str
    compute_power_watt: float
    memory_power_watt: float
    comm_power_watt: float
    max_operations: int
    
    def estimate_energy(self, n_ops: int, mem_accesses: int, 
                        data_mb: float = 0.0) -> float:
        """Estimate energy consumption for a workload."""
        time_per_op = 1e-6  # 1 microsecond per operation
        time_per_mem = 1e-7  # 100ns per memory access
        time_per_mb = 0.01   # 10ms per MB transferred
        
        compute_time = n_ops * time_per_op
        memory_time = mem_accesses * time_per_mem
        comm_time = data_mb * time_per_mb
        
        energy = (
            self.compute_power_watt * compute_time +
            self.memory_power_watt * memory_time +
            self.comm_power_watt * comm_time
        )
        
        return energy


# Standard device profiles
DEVICE_PROFILES = {
    'iot': DeviceProfile(
        name='IoT Sensor',
        compute_power_watt=0.1,
        memory_power_watt=0.05,
        comm_power_watt=0.15,
        max_operations=10000
    ),
    'mobile': DeviceProfile(
        name='Mobile Phone',
        compute_power_watt=1.0,
        memory_power_watt=0.3,
        comm_power_watt=0.5,
        max_operations=100000
    ),
    'edge': DeviceProfile(
        name='Edge Device',
        compute_power_watt=5.0,
        memory_power_watt=1.0,
        comm_power_watt=0.8,
        max_operations=500000
    ),
    'desktop': DeviceProfile(
        name='Desktop',
        compute_power_watt=50.0,
        memory_power_watt=10.0,
        comm_power_watt=2.0,
        max_operations=10000000
    ),
    'server': DeviceProfile(
        name='Server',
        compute_power_watt=200.0,
        memory_power_watt=40.0,
        comm_power_watt=10.0,
        max_operations=100000000
    )
}


class EnergyEfficientIntegrator:
    """
    Energy-efficient numerical integration.
    
    Optimizes integration methods for power-constrained devices.
    
    Example:
        >>> integrator = EnergyEfficientIntegrator(device='iot')
        >>> result = integrator.integrate(lambda x: x**2, 0, 1)
        >>> print(f"Integral: {result.value:.4f}, Energy: {result.energy_cost:.6f} Wh")
    """
    
    def __init__(self, device: str = 'mobile', 
                 energy_budget: float = None):
        """
        Initialize energy-efficient integrator.
        
        Args:
            device: Device type ('iot', 'mobile', 'edge', 'desktop', 'server')
            energy_budget: Maximum energy budget in watt-hours
        """
        if device not in DEVICE_PROFILES:
            raise ValueError(f"Unknown device: {device}. Use one of {list(DEVICE_PROFILES.keys())}")
        
        self.device = DEVICE_PROFILES[device]
        self.energy_budget = energy_budget
        self.total_energy_used = 0.0
    
    def trapezoidal(self, f: Callable[[float], float],
                    a: float, b: float,
                    n: int = 100) -> IntegrationResult:
        """
        Simple trapezoidal rule (low energy).
        
        Energy complexity: O(n)
        """
        start_time = time.time()
        
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = np.array([f(xi) for xi in x])
        
        integral = h * (y[0]/2 + np.sum(y[1:-1]) + y[-1]/2)
        
        exec_time = time.time() - start_time
        energy = self.device.estimate_energy(n + 1, n + 1)
        self.total_energy_used += energy
        
        return IntegrationResult(
            value=integral,
            n_evaluations=n + 1,
            execution_time=exec_time,
            energy_cost=energy,
            method='trapezoidal'
        )
    
    def simpson(self, f: Callable[[float], float],
                a: float, b: float,
                n: int = 100) -> IntegrationResult:
        """
        Simpson's rule (moderate energy, better accuracy).
        
        Energy complexity: O(n)
        """
        if n % 2 == 1:
            n += 1
        
        start_time = time.time()
        
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = np.array([f(xi) for xi in x])
        
        integral = h/3 * (y[0] + 4*np.sum(y[1::2]) + 2*np.sum(y[2:-1:2]) + y[-1])
        
        exec_time = time.time() - start_time
        energy = self.device.estimate_energy(n + 1, n + 1)
        self.total_energy_used += energy
        
        return IntegrationResult(
            value=integral,
            n_evaluations=n + 1,
            execution_time=exec_time,
            energy_cost=energy,
            method='simpson'
        )
    
    def gauss_legendre(self, f: Callable[[float], float],
                       a: float, b: float,
                       n: int = 5) -> IntegrationResult:
        """
        Gauss-Legendre quadrature (high accuracy, low evaluations).
        
        Best for smooth functions with limited energy.
        """
        start_time = time.time()
        
        # Gauss-Legendre nodes and weights for n points
        nodes, weights = np.polynomial.legendre.leggauss(n)
        
        # Transform from [-1, 1] to [a, b]
        x = 0.5 * (b - a) * nodes + 0.5 * (b + a)
        w = 0.5 * (b - a) * weights
        
        y = np.array([f(xi) for xi in x])
        integral = np.sum(w * y)
        
        exec_time = time.time() - start_time
        energy = self.device.estimate_energy(n * 2, n)  # More ops per point
        self.total_energy_used += energy
        
        return IntegrationResult(
            value=integral,
            n_evaluations=n,
            execution_time=exec_time,
            energy_cost=energy,
            method='gauss_legendre'
        )
    
    def sparse_grid(self, f: Callable[[float], float],
                    a: float, b: float,
                    level: int = 3) -> IntegrationResult:
        """
        Sparse grid integration (efficient for high dimensions).
        
        Uses Clenshaw-Curtis nodes with sparse combination.
        """
        start_time = time.time()
        
        total = 0.0
        n_evals = 0
        
        for l in range(level + 1):
            # Clenshaw-Curtis points
            n_points = max(1, 2**l - 1 + 2)
            if l == 0:
                nodes = [0.5 * (a + b)]
                weights = [b - a]
            else:
                nodes = [0.5 * (a + b) + 0.5 * (b - a) * np.cos(np.pi * k / (n_points - 1)) 
                         for k in range(n_points)]
                weights = [1.0] * n_points
                weights[0] = weights[-1] = 0.5
                weights = [w * (b - a) / (n_points - 1) for w in weights]
            
            for node, weight in zip(nodes, weights):
                level_contribution = weight * f(node) / (level + 1)
                total += level_contribution
                n_evals += 1
        
        exec_time = time.time() - start_time
        energy = self.device.estimate_energy(n_evals, n_evals * 2)
        self.total_energy_used += energy
        
        return IntegrationResult(
            value=total,
            n_evaluations=n_evals,
            execution_time=exec_time,
            energy_cost=energy,
            method='sparse_grid'
        )
    
    def adaptive_quadrature(self, f: Callable[[float], float],
                             a: float, b: float,
                             tol: float = 1e-6,
                             max_depth: int = 10) -> IntegrationResult:
        """
        Adaptive quadrature with energy awareness.
        
        Refines only where needed, saving energy.
        """
        start_time = time.time()
        n_evals = [0]
        
        def recursive_integrate(a, b, fa, fb, fc, depth):
            if depth > max_depth:
                return (b - a) * (fa + 4*fc + fb) / 6
            
            mid = (a + b) / 2
            n_evals[0] += 2
            
            fl = f((a + mid) / 2)
            fr = f((mid + b) / 2)
            
            # Simpson estimates
            left_simpson = (mid - a) * (fa + 4*fl + fc) / 6
            right_simpson = (b - mid) * (fc + 4*fr + fb) / 6
            whole_simpson = (b - a) * (fa + 4*fc + fb) / 6
            
            error = abs(left_simpson + right_simpson - whole_simpson) / 15
            
            if error < tol or depth >= max_depth:
                return left_simpson + right_simpson
            else:
                return (recursive_integrate(a, mid, fa, fc, fl, depth + 1) +
                        recursive_integrate(mid, b, fc, fb, fr, depth + 1))
        
        fa, fb, fc = f(a), f(b), f((a + b) / 2)
        n_evals[0] = 3
        
        integral = recursive_integrate(a, b, fa, fb, fc, 0)
        
        exec_time = time.time() - start_time
        energy = self.device.estimate_energy(n_evals[0], n_evals[0] * 2)
        self.total_energy_used += energy
        
        return IntegrationResult(
            value=integral,
            n_evaluations=n_evals[0],
            execution_time=exec_time,
            energy_cost=energy,
            method='adaptive',
            error_estimate=tol
        )
    
    def integrate(self, f: Callable[[float], float],
                  a: float, b: float,
                  accuracy: str = 'medium') -> IntegrationResult:
        """
        Auto-select integration method based on energy/accuracy tradeoff.
        
        Args:
            f: Function to integrate
            a, b: Integration bounds
            accuracy: 'low', 'medium', 'high'
        """
        if accuracy == 'low':
            # Minimum energy: few Gauss points
            return self.gauss_legendre(f, a, b, n=3)
        elif accuracy == 'high':
            # Maximum accuracy: adaptive
            return self.adaptive_quadrature(f, a, b, tol=1e-8)
        else:
            # Balance: moderate Simpson
            return self.simpson(f, a, b, n=50)
    
    def optimize_for_energy_budget(self, f: Callable[[float], float],
                                    a: float, b: float,
                                    energy_budget: float) -> IntegrationResult:
        """
        Find best integration given an energy budget.
        
        Args:
            f: Function to integrate
            a, b: Bounds
            energy_budget: Maximum energy in watt-hours
            
        Returns:
            Best result within budget
        """
        methods = [
            ('gauss_3', lambda: self.gauss_legendre(f, a, b, n=3)),
            ('gauss_5', lambda: self.gauss_legendre(f, a, b, n=5)),
            ('simpson_20', lambda: self.simpson(f, a, b, n=20)),
            ('simpson_50', lambda: self.simpson(f, a, b, n=50)),
            ('adaptive_1e-4', lambda: self.adaptive_quadrature(f, a, b, tol=1e-4)),
        ]
        
        best_result = None
        
        for name, method in methods:
            # Save state
            prev_energy = self.total_energy_used
            
            result = method()
            
            if result.energy_cost <= energy_budget:
                if best_result is None or result.n_evaluations > best_result.n_evaluations:
                    best_result = result
            
            # Restore state
            self.total_energy_used = prev_energy
        
        if best_result is None:
            # Fall back to minimum energy method
            best_result = self.gauss_legendre(f, a, b, n=3)
        
        return best_result
    
    def compare_methods(self, f: Callable[[float], float],
                        a: float, b: float,
                        true_value: float = None) -> Dict[str, IntegrationResult]:
        """
        Compare all integration methods.
        
        Returns dictionary of method -> result.
        """
        methods = {
            'trapezoidal_10': lambda: self.trapezoidal(f, a, b, n=10),
            'trapezoidal_100': lambda: self.trapezoidal(f, a, b, n=100),
            'simpson_10': lambda: self.simpson(f, a, b, n=10),
            'simpson_50': lambda: self.simpson(f, a, b, n=50),
            'gauss_3': lambda: self.gauss_legendre(f, a, b, n=3),
            'gauss_5': lambda: self.gauss_legendre(f, a, b, n=5),
            'sparse_grid_2': lambda: self.sparse_grid(f, a, b, level=2),
            'adaptive_1e-4': lambda: self.adaptive_quadrature(f, a, b, tol=1e-4),
        }
        
        results = {}
        for name, method in methods.items():
            result = method()
            if true_value is not None:
                result.error_estimate = abs(result.value - true_value)
            results[name] = result
        
        return results


def energy_efficient_demo():
    """
    Demonstrate energy-efficient integration.
    
    Industrial Case Study: Google DeepMind Data Centers
    - 40% cooling energy reduction
    - $150M/year savings
    - 300,000 tons CO2 reduction annually
    """
    print("=" * 60)
    print("Integration in Energy-Efficient ML Systems")
    print("=" * 60)
    print("\nIndustrial Case Study: Google DeepMind Data Centers")
    print("- Challenge: 1-2% of global electricity for data centers")
    print("- Solution: Energy-efficient predictive integration")
    print("- Results: 40% cooling reduction, $150M/year saved\n")
    
    # Example: Building energy consumption over a day
    def building_energy(t):
        """Energy consumption pattern (kW) over 24 hours."""
        base = 2.0
        time_factor = 0.5 + 0.5 * np.sin(2 * np.pi * t / 24 - np.pi/2)
        temp_factor = 1.0 + 0.3 * max(0, np.cos(2 * np.pi * t / 24) - 0.5)
        return base * time_factor * temp_factor
    
    # True integral
    from scipy.integrate import quad
    true_value, _ = quad(building_energy, 0, 24)
    print(f"True daily energy consumption: {true_value:.2f} kWh\n")
    
    # Compare across devices
    print("-" * 60)
    print("Device Comparison")
    print("-" * 60)
    
    for device_type in ['iot', 'mobile', 'edge']:
        integrator = EnergyEfficientIntegrator(device=device_type)
        result = integrator.integrate(building_energy, 0, 24, accuracy='medium')
        
        error = abs(result.value - true_value)
        print(f"\n{integrator.device.name}:")
        print(f"  Result: {result.value:.2f} kWh (error: {error:.4f})")
        print(f"  Evaluations: {result.n_evaluations}")
        print(f"  Energy: {result.energy_cost:.8f} Wh")
    
    # Method comparison for IoT
    print("\n" + "-" * 60)
    print("Method Comparison (IoT Device)")
    print("-" * 60)
    
    integrator = EnergyEfficientIntegrator(device='iot')
    results = integrator.compare_methods(building_energy, 0, 24, true_value)
    
    print(f"\n{'Method':<20} {'Value':<10} {'Error':<10} {'Evals':<8} {'Energy (Wh)':<12}")
    print("-" * 60)
    
    for name, result in sorted(results.items(), key=lambda x: x[1].energy_cost):
        print(f"{name:<20} {result.value:<10.2f} {result.error_estimate:<10.4f} "
              f"{result.n_evaluations:<8} {result.energy_cost:<12.8f}")
    
    # Energy budget optimization
    print("\n" + "-" * 60)
    print("Energy Budget Optimization")
    print("-" * 60)
    
    budgets = [1e-6, 5e-6, 1e-5, 5e-5]
    for budget in budgets:
        integrator = EnergyEfficientIntegrator(device='iot')
        result = integrator.optimize_for_energy_budget(building_energy, 0, 24, budget)
        error = abs(result.value - true_value) / true_value * 100
        
        print(f"Budget {budget:.0e} Wh: {result.method} -> "
              f"error {error:.2f}%, used {result.energy_cost:.2e} Wh")
    
    return results


# Module exports
__all__ = [
    'EnergyEfficientIntegrator',
    'DeviceProfile',
    'IntegrationResult',
    'DEVICE_PROFILES',
    'energy_efficient_demo',
]
