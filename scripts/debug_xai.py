
import sys
import os
import traceback
import numpy as np

sys.path.insert(0, '.')

try:
    print("Importing ExplainableModel...")
    from src.core.explainable_ai import ExplainableModel
    
    print("Initializing model...")
    model = ExplainableModel()
    
    print("Generating data...")
    data = model.generate_medical_data(100)
    
    print("Training model...")
    model.train(data['X'], data['y'], data['feature_names'])
    
    print("Predicting with explanation...")
    exp = model.predict_with_explanation(data['X'][:1], num_samples=10)
    
    print(f"Success! Result length: {len(exp)}")
    
except Exception:
    traceback.print_exc()
