"""
Pure Python Implementation of Causal Inference Concepts ("White-Box").
Focuses on Average Treatment Effect (ATE) estimation logic.
"""
from typing import List, Dict, Callable

class CausalDataset:
    def __init__(self, data: List[Dict[str, float]]):
        """
        data: List of dicts, e.g., [{'T': 1, 'Y': 5, 'X': 0.5}, ...]
        """
        self.data = data

    def estimate_ate_naive(self) -> float:
        """Naive ATE: E[Y|T=1] - E[Y|T=0] (Biased if confounding exists)"""
        t1 = [d['Y'] for d in self.data if d['T'] == 1]
        t0 = [d['Y'] for d in self.data if d['T'] == 0]
        
        mean_t1 = sum(t1) / len(t1) if t1 else 0
        mean_t0 = sum(t0) / len(t0) if t0 else 0
        
        return mean_t1 - mean_t0

    def estimate_ate_ipw(self, propensity_score_func: Callable[[Dict], float]) -> float:
        """
        Inverse Probability Weighting (IPW) for ATE.
        ATE = E[ (T*Y)/e(X) - ((1-T)*Y)/(1-e(X)) ]
        """
        n = len(self.data)
        weighted_sum = 0.0
        
        for row in self.data:
            t = row['T']
            y = row['Y']
            e_x = propensity_score_func(row) # P(T=1 | X)
            
            # Avoid division by zero
            e_x = max(min(e_x, 0.99), 0.01)
            
            if t == 1:
                weighted_sum += y / e_x
            else:
                weighted_sum -= y / (1 - e_x)
                
        return weighted_sum / n

def simulation_confounder_example():
    """
    Simulate Data: 
    X = Confounder (e.g., Age)
    T = Treatment (e.g., Drug) -> depends on X (Rich/Old take drug more)
    Y = Outcome (e.g., Recovery) -> depends on T and X (Young recover faster naturally)
    """
    import random
    data = []
    for _ in range(1000):
        x = random.uniform(0, 1) # Age: 0=Young, 1=Old
        
        # Propensity: Older people more likely to take drug
        prob_t = 0.2 + 0.6 * x 
        t = 1 if random.random() < prob_t else 0
        
        # Outcome: Drug helps (+0.5), Age hurts (-1.0*x)
        # Truth ATE = 0.5
        y = 0.5 * t - 1.0 * x + random.normalvariate(0, 0.1)
        
        data.append({'T': t, 'Y': y, 'X': x})
        
    ds = CausalDataset(data)
    naive = ds.estimate_ate_naive()
    
    # We know the true propensity used to generate data
    def true_propensity(row):
        return 0.2 + 0.6 * row['X']
        
    ipw = ds.estimate_ate_ipw(true_propensity)
    
    return naive, ipw

if __name__ == "__main__":
    naive, ipw = simulation_confounder_example()
    print(f"Naive ATE (Biased): {naive:.4f}")
    print(f"IPW ATE (Corrected): {ipw:.4f}")
    print(f"True ATE: 0.5000")
