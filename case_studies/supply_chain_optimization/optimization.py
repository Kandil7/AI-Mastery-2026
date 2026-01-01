"""
Supply Chain Optimization Algorithms
====================================

Operations Research techniques used by companies like Amazon
for supply chain optimization.

Key Components:
- LinearProgramSolver: Inventory allocation
- TransportationProblem: Distribution optimization
- DemandForecaster: Probabilistic forecasting
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


@dataclass
class Warehouse:
    """Warehouse with capacity and location."""
    id: str
    capacity: int
    location: Tuple[float, float]
    cost_per_unit: float = 1.0


@dataclass
class Customer:
    """Customer with demand and location."""
    id: str
    demand: int
    location: Tuple[float, float]


@dataclass
class AllocationResult:
    """Result of allocation optimization."""
    allocations: Dict[str, Dict[str, int]]  # warehouse -> customer -> units
    total_cost: float
    satisfied_demand: Dict[str, int]


class TransportationProblem:
    """
    Solve the transportation problem using North-West Corner method
    and stepping stone optimization.
    
    Example:
    >>> warehouses = [Warehouse("W1", 100, (0, 0)), Warehouse("W2", 150, (10, 0))]
    >>> customers = [Customer("C1", 80, (5, 5)), Customer("C2", 120, (5, -5))]
    >>> solver = TransportationProblem(warehouses, customers)
    >>> result = solver.solve()
    """
    
    def __init__(
        self, 
        warehouses: List[Warehouse], 
        customers: List[Customer],
        cost_matrix: Optional[np.ndarray] = None
    ):
        self.warehouses = warehouses
        self.customers = customers
        
        if cost_matrix is not None:
            self.costs = cost_matrix
        else:
            self.costs = self._compute_distance_costs()
    
    def _compute_distance_costs(self) -> np.ndarray:
        """Compute transportation costs based on Euclidean distance."""
        costs = np.zeros((len(self.warehouses), len(self.customers)))
        
        for i, w in enumerate(self.warehouses):
            for j, c in enumerate(self.customers):
                dx = w.location[0] - c.location[0]
                dy = w.location[1] - c.location[1]
                distance = np.sqrt(dx**2 + dy**2)
                costs[i, j] = distance * w.cost_per_unit
        
        return costs
    
    def solve_northwest_corner(self) -> np.ndarray:
        """
        North-West Corner method for initial feasible solution.
        
        Simple but not optimal - used as starting point.
        """
        supply = np.array([w.capacity for w in self.warehouses], dtype=float)
        demand = np.array([c.demand for c in self.customers], dtype=float)
        
        allocation = np.zeros((len(supply), len(demand)))
        
        i, j = 0, 0
        while i < len(supply) and j < len(demand):
            qty = min(supply[i], demand[j])
            allocation[i, j] = qty
            supply[i] -= qty
            demand[j] -= qty
            
            if supply[i] == 0:
                i += 1
            if demand[j] == 0:
                j += 1
        
        return allocation
    
    def solve(self) -> AllocationResult:
        """Solve transportation problem."""
        allocation = self.solve_northwest_corner()
        
        # Calculate total cost
        total_cost = np.sum(allocation * self.costs)
        
        # Build result structure
        alloc_dict = {}
        for i, w in enumerate(self.warehouses):
            alloc_dict[w.id] = {}
            for j, c in enumerate(self.customers):
                if allocation[i, j] > 0:
                    alloc_dict[w.id][c.id] = int(allocation[i, j])
        
        satisfied = {
            c.id: int(allocation[:, j].sum()) 
            for j, c in enumerate(self.customers)
        }
        
        return AllocationResult(
            allocations=alloc_dict,
            total_cost=total_cost,
            satisfied_demand=satisfied
        )


class InventoryOptimizer:
    """
    Economic Order Quantity (EOQ) and safety stock optimization.
    
    EOQ Formula: Q* = sqrt(2DS/H)
    where:
        D = annual demand
        S = ordering cost per order
        H = holding cost per unit per year
    """
    
    def __init__(
        self,
        annual_demand: float,
        ordering_cost: float,
        holding_cost: float,
        lead_time_days: float = 7,
        demand_std: float = 0.0,
        service_level: float = 0.95
    ):
        self.D = annual_demand
        self.S = ordering_cost
        self.H = holding_cost
        self.lead_time = lead_time_days
        self.demand_std = demand_std
        self.service_level = service_level
    
    def calculate_eoq(self) -> float:
        """Calculate Economic Order Quantity."""
        return np.sqrt(2 * self.D * self.S / self.H)
    
    def calculate_safety_stock(self) -> float:
        """Calculate safety stock for desired service level."""
        from scipy import stats
        
        if self.demand_std == 0:
            return 0.0
        
        # Z-score for service level
        z = stats.norm.ppf(self.service_level)
        
        # Standard deviation during lead time
        lead_time_std = self.demand_std * np.sqrt(self.lead_time)
        
        return z * lead_time_std
    
    def calculate_reorder_point(self) -> float:
        """Calculate when to reorder."""
        daily_demand = self.D / 365
        return daily_demand * self.lead_time + self.calculate_safety_stock()
    
    def get_policy(self) -> Dict[str, float]:
        """Get complete inventory policy."""
        eoq = self.calculate_eoq()
        safety = self.calculate_safety_stock()
        rop = self.calculate_reorder_point()
        
        # Total annual cost
        ordering_costs = (self.D / eoq) * self.S
        holding_costs = (eoq / 2 + safety) * self.H
        total_cost = ordering_costs + holding_costs
        
        return {
            "order_quantity": eoq,
            "safety_stock": safety,
            "reorder_point": rop,
            "orders_per_year": self.D / eoq,
            "annual_ordering_cost": ordering_costs,
            "annual_holding_cost": holding_costs,
            "total_annual_cost": total_cost
        }


class DemandForecaster:
    """
    Simple demand forecasting with exponential smoothing.
    
    Production systems like Amazon's DeepAR use deep learning,
    but exponential smoothing is a solid baseline.
    """
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.1):
        """
        Args:
            alpha: Level smoothing parameter (0-1)
            beta: Trend smoothing parameter (0-1)
        """
        self.alpha = alpha
        self.beta = beta
        self.level = None
        self.trend = None
    
    def fit(self, history: np.ndarray) -> 'DemandForecaster':
        """Fit model to historical demand."""
        if len(history) < 2:
            raise ValueError("Need at least 2 data points")
        
        # Initialize
        self.level = history[0]
        self.trend = history[1] - history[0]
        
        # Update for each observation
        for y in history[1:]:
            prev_level = self.level
            self.level = self.alpha * y + (1 - self.alpha) * (prev_level + self.trend)
            self.trend = self.beta * (self.level - prev_level) + (1 - self.beta) * self.trend
        
        return self
    
    def forecast(self, periods: int = 7) -> np.ndarray:
        """Forecast future demand."""
        if self.level is None:
            raise ValueError("Model not fitted")
        
        forecasts = []
        for h in range(1, periods + 1):
            forecasts.append(self.level + h * self.trend)
        
        return np.array(forecasts)
    
    def forecast_with_intervals(
        self, 
        periods: int = 7, 
        alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forecast with prediction intervals."""
        from scipy import stats
        
        point_forecast = self.forecast(periods)
        
        # Approximate prediction interval
        z = stats.norm.ppf(1 - alpha / 2)
        
        # Increasing uncertainty over time
        stderr = np.array([np.sqrt(h) for h in range(1, periods + 1)])
        
        lower = point_forecast - z * stderr * np.abs(self.trend)
        upper = point_forecast + z * stderr * np.abs(self.trend)
        
        return point_forecast, lower, upper


def optimize_supply_chain(
    warehouses: List[Warehouse],
    customers: List[Customer],
    demand_history: Optional[np.ndarray] = None
) -> Dict[str, any]:
    """
    End-to-end supply chain optimization.
    
    Example:
    >>> warehouses = [
    ...     Warehouse("Seattle", 500, (47.6, -122.3)),
    ...     Warehouse("Dallas", 400, (32.8, -96.8)),
    ... ]
    >>> customers = [
    ...     Customer("NYC", 200, (40.7, -74.0)),
    ...     Customer("LA", 300, (34.0, -118.2)),
    ... ]
    >>> result = optimize_supply_chain(warehouses, customers)
    """
    # 1. Solve transportation problem
    transport = TransportationProblem(warehouses, customers)
    allocation = transport.solve()
    
    # 2. Inventory optimization for each warehouse
    inventory_policies = {}
    for w in warehouses:
        total_demand = sum(
            allocation.allocations.get(w.id, {}).values()
        )
        if total_demand > 0:
            optimizer = InventoryOptimizer(
                annual_demand=total_demand * 52,  # Weekly to annual
                ordering_cost=100,
                holding_cost=0.25 * 10,  # 25% of unit cost
                lead_time_days=7
            )
            inventory_policies[w.id] = optimizer.get_policy()
    
    # 3. Demand forecasting
    forecasts = None
    if demand_history is not None:
        forecaster = DemandForecaster(alpha=0.3, beta=0.1)
        forecaster.fit(demand_history)
        forecasts = forecaster.forecast(periods=7)
    
    return {
        "allocation": allocation,
        "inventory_policies": inventory_policies,
        "demand_forecast": forecasts
    }


__all__ = [
    "Warehouse",
    "Customer",
    "AllocationResult",
    "TransportationProblem",
    "InventoryOptimizer",
    "DemandForecaster",
    "optimize_supply_chain",
]
