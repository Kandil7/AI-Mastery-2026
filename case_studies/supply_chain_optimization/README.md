# Supply Chain Optimization Case Study

How Amazon uses Operations Research and ML to optimize their supply chain.

---

## 1. The Problem

**Challenge**: Deliver millions of products to customers efficiently while minimizing:
- Transportation costs
- Inventory holding costs
- Delivery time
- Environmental impact

---

## 2. Mathematical Formulation

### Linear Programming for Inventory Allocation

**Objective**: Minimize total cost
```
minimize: Σᵢ Σⱼ cᵢⱼ × xᵢⱼ

where:
  xᵢⱼ = units shipped from warehouse i to location j
  cᵢⱼ = cost per unit
```

**Constraints**:
```
Supply:   Σⱼ xᵢⱼ ≤ Sᵢ        (warehouse capacity)
Demand:   Σᵢ xᵢⱼ ≥ Dⱼ        (customer demand)
Non-neg:  xᵢⱼ ≥ 0
```

---

## 3. Python Implementation

```python
import numpy as np
from scipy.optimize import linprog

def optimize_allocation(costs, supply, demand):
    """
    Optimal allocation using linear programming.
    
    Args:
        costs: n_warehouses × n_locations cost matrix
        supply: warehouse capacities
        demand: location demands
    
    Returns:
        Optimal allocation matrix
    """
    n_warehouses, n_locations = costs.shape
    
    # Flatten cost matrix for linprog
    c = costs.flatten()
    
    # Supply constraints: Σⱼ xᵢⱼ ≤ Sᵢ
    A_supply = np.zeros((n_warehouses, len(c)))
    for i in range(n_warehouses):
        A_supply[i, i*n_locations:(i+1)*n_locations] = 1
    b_supply = supply
    
    # Demand constraints: Σᵢ xᵢⱼ ≥ Dⱼ (as ≤ with negation)
    A_demand = np.zeros((n_locations, len(c)))
    for j in range(n_locations):
        for i in range(n_warehouses):
            A_demand[j, i*n_locations + j] = -1
    b_demand = -demand
    
    # Combine constraints
    A = np.vstack([A_supply, A_demand])
    b = np.concatenate([b_supply, b_demand])
    
    # Solve
    result = linprog(c, A_ub=A, b_ub=b, method='highs')
    
    return result.x.reshape(costs.shape), result.fun

# Example: 3 warehouses, 4 cities
costs = np.array([
    [8, 6, 10, 9],   # From warehouse 1
    [9, 12, 13, 7],  # From warehouse 2
    [14, 9, 16, 5]   # From warehouse 3
])

supply = np.array([35, 50, 40])  # Warehouse capacities
demand = np.array([45, 20, 30, 30])  # City demands

allocation, total_cost = optimize_allocation(costs, supply, demand)
print(f"Optimal allocation:\n{allocation}")
print(f"Minimum total cost: ${total_cost:.2f}")
```

---

## 4. Demand Forecasting with DeepAR

Amazon's approach to predicting future demand:

```python
# DeepAR-style probabilistic forecasting (simplified)
import torch
import torch.nn as nn

class SimpleLSTMForecaster(nn.Module):
    """LSTM-based demand forecaster."""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        self.fc_sigma = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        
        # Predict mean and std (probabilistic)
        mu = self.fc_mu(last_hidden)
        sigma = nn.functional.softplus(self.fc_sigma(last_hidden))
        
        return mu, sigma
    
    def predict(self, x, n_samples=100):
        """Sample from predictive distribution."""
        mu, sigma = self.forward(x)
        samples = torch.normal(
            mu.unsqueeze(1).expand(-1, n_samples, -1),
            sigma.unsqueeze(1).expand(-1, n_samples, -1)
        )
        return samples

# Usage
model = SimpleLSTMForecaster(input_dim=10, hidden_dim=64, output_dim=7)
# Predict next 7 days with uncertainty
```

---

## 5. MILP for Scheduling

Mixed Integer Linear Programming for delivery scheduling:

```python
# Simplified MILP-style scheduling
def schedule_deliveries(orders, trucks, time_windows):
    """
    Assign orders to trucks respecting time windows.
    
    This is a simplified version - production uses 
    commercial solvers like Gurobi or OR-Tools.
    """
    from itertools import permutations
    
    best_assignment = None
    best_cost = float('inf')
    
    # Brute force for small problems
    for perm in permutations(range(len(orders))):
        assignment = {}
        truck_loads = {t: 0 for t in trucks}
        cost = 0
        
        for order_idx in perm:
            order = orders[order_idx]
            # Find best truck
            for truck in trucks:
                if truck_loads[truck] + order['weight'] <= truck['capacity']:
                    assignment[order['id']] = truck['id']
                    truck_loads[truck] += order['weight']
                    cost += order['distance'] * truck['cost_per_mile']
                    break
        
        if len(assignment) == len(orders) and cost < best_cost:
            best_cost = cost
            best_assignment = assignment
    
    return best_assignment, best_cost
```

---

## 6. Results & Impact

| Metric | Before OR | After OR | Improvement |
|--------|-----------|----------|-------------|
| Delivery cost | $2.50/pkg | $1.80/pkg | 28% ↓ |
| Inventory waste | 8% | 3% | 63% ↓ |
| On-time delivery | 89% | 97% | 9% ↑ |
| Route efficiency | - | +25% | - |

---

## 7. Key Takeaways

1. **Linear Programming** optimizes continuous decisions (how much)
2. **Integer Programming** handles discrete choices (which route)
3. **Probabilistic Forecasting** quantifies uncertainty
4. **Hybrid Approaches** combine optimization with ML

**Real-world note**: Amazon uses custom solvers running on thousands of cores to solve these problems at scale.
