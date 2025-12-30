# scripts/create_dummy_model.py

import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import os

def create_and_save_model():
    """
    Creates a simple linear regression model, trains it on toy data,
    and saves it to the models/ directory.
    """
    # 1. Create toy data
    # y = 2x_1 + 3x_2 + 5
    X = np.random.rand(100, 2) * 10
    noise = np.random.randn(100) * 0.5
    y = 2 * X[:, 0] + 3 * X[:, 1] + 5 + noise

    # 2. Train a model
    model = LinearRegression()
    model.fit(X, y)

    # 3. Save the model
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'default_model.joblib')
    joblib.dump(model, model_path)
    
    print(f"Model trained and saved to {model_path}")
    print(f"Model coefficients: {model.coef_}")
    print(f"Model intercept: {model.intercept_}")

if __name__ == "__main__":
    create_and_save_model()
