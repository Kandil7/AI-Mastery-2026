# Classical Machine Learning Module
"""
Classical machine learning algorithms implemented from scratch.
Includes linear models, tree-based methods, ensemble methods, and SVM.
"""

from .linear_regression import LinearRegression, RidgeRegression, LassoRegression
from .logistic_regression import LogisticRegression, MultinomialLogisticRegression
from .decision_trees import Node, DecisionTree, DecisionTreeClassifier, DecisionTreeRegressor
from .ensemble import RandomForest, RandomForestRegressor, AdaBoostClassifier, GradientBoostingRegressor
from .svm import SVM, SVR

__all__ = [
    # Linear Models
    "LinearRegression",
    "RidgeRegression",
    "LassoRegression",
    "LogisticRegression",
    "MultinomialLogisticRegression",
    # Tree-based
    "Node",
    "DecisionTree",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    # Ensemble
    "RandomForest",
    "RandomForestRegressor",
    "AdaBoostClassifier",
    "GradientBoostingRegressor",
    # SVM
    "SVM",
    "SVR",
]
