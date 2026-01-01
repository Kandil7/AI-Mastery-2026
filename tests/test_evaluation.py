"""
Unit Tests for Evaluation Module

Tests for classification, regression, ranking metrics, and RAG evaluation.
"""

import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.evaluation.evaluation import (
    EvaluationConfig,
    Metric,
    MetricType,
    # Classification metrics
    accuracy,
    precision,
    recall,
    f1_score,
    confusion_matrix,
    # Regression metrics
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
    # Ranking metrics
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    # Evaluators
    RAGEvaluator,
    MLEvaluator,
    LLMEvaluator,
    BenchmarkRunner,
    EvaluationReport,
)


class TestClassificationMetrics(unittest.TestCase):
    """Tests for classification metrics."""
    
    def setUp(self):
        """Set up test data."""
        # Binary classification
        self.y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
        self.y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 0])
        
    def test_accuracy(self):
        """Test accuracy calculation."""
        acc = accuracy(self.y_true, self.y_pred)
        
        self.assertAlmostEqual(acc, 0.8, places=2)  # 8/10 correct
    
    def test_precision(self):
        """Test precision calculation."""
        prec = precision(self.y_true, self.y_pred)
        
        # TP=4, FP=1, Precision = 4/5 = 0.8
        self.assertAlmostEqual(prec, 0.8, places=2)
    
    def test_recall(self):
        """Test recall calculation."""
        rec = recall(self.y_true, self.y_pred)
        
        # TP=4, FN=1, Recall = 4/5 = 0.8
        self.assertAlmostEqual(rec, 0.8, places=2)
    
    def test_f1_score(self):
        """Test F1 score calculation."""
        f1 = f1_score(self.y_true, self.y_pred)
        
        # F1 = 2 * (0.8 * 0.8) / (0.8 + 0.8) = 0.8
        self.assertAlmostEqual(f1, 0.8, places=2)
    
    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y = np.array([1, 0, 1, 0])
        
        self.assertEqual(accuracy(y, y), 1.0)
        self.assertEqual(precision(y, y), 1.0)
        self.assertEqual(recall(y, y), 1.0)
        self.assertEqual(f1_score(y, y), 1.0)
    
    def test_confusion_matrix(self):
        """Test confusion matrix calculation."""
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        self.assertEqual(cm.shape, (2, 2))
        self.assertEqual(cm.sum(), len(self.y_true))


class TestRegressionMetrics(unittest.TestCase):
    """Tests for regression metrics."""
    
    def setUp(self):
        """Set up test data."""
        self.y_true = np.array([3.0, 5.0, 2.5, 7.0, 4.5])
        self.y_pred = np.array([2.5, 5.0, 2.0, 8.0, 4.0])
    
    def test_mean_squared_error(self):
        """Test MSE calculation."""
        mse = mean_squared_error(self.y_true, self.y_pred)
        
        # Expected: mean of [0.25, 0, 0.25, 1.0, 0.25] = 0.35
        self.assertAlmostEqual(mse, 0.35, places=2)
    
    def test_root_mean_squared_error(self):
        """Test RMSE calculation."""
        rmse = root_mean_squared_error(self.y_true, self.y_pred)
        
        self.assertAlmostEqual(rmse, np.sqrt(0.35), places=2)
    
    def test_mean_absolute_error(self):
        """Test MAE calculation."""
        mae = mean_absolute_error(self.y_true, self.y_pred)
        
        # Expected: mean of [0.5, 0, 0.5, 1.0, 0.5] = 0.5
        self.assertAlmostEqual(mae, 0.5, places=2)
    
    def test_r2_score(self):
        """Test R² score calculation."""
        r2 = r2_score(self.y_true, self.y_pred)
        
        self.assertLessEqual(r2, 1.0)
        self.assertGreater(r2, 0.0)  # Should be positive for reasonable predictions
    
    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y = np.array([1.0, 2.0, 3.0])
        
        self.assertEqual(mean_squared_error(y, y), 0.0)
        self.assertEqual(mean_absolute_error(y, y), 0.0)
        self.assertEqual(r2_score(y, y), 1.0)


class TestRankingMetrics(unittest.TestCase):
    """Tests for ranking metrics."""
    
    def test_mean_reciprocal_rank(self):
        """Test MRR calculation."""
        y_true = [[0, 1], [2]]  # Relevant documents for each query
        y_pred = [[1, 0, 2], [0, 1, 2]]  # Ranked lists
        
        mrr = mean_reciprocal_rank(y_true, y_pred)
        
        # Query 1: first relevant at rank 1 → 1/1 = 1
        # Query 2: first relevant at rank 3 → 1/3
        # MRR = (1 + 1/3) / 2 = 0.667
        self.assertAlmostEqual(mrr, (1.0 + 1/3) / 2, places=2)
    
    def test_precision_at_k(self):
        """Test P@K calculation."""
        y_true = [0, 2, 4]  # Relevant doc indices
        y_pred = [0, 1, 2, 3, 4]  # Ranked list
        
        p_at_3 = precision_at_k(y_true, y_pred, k=3)
        
        # Top 3: [0, 1, 2], relevant: [0, 2], P@3 = 2/3
        self.assertAlmostEqual(p_at_3, 2/3, places=2)
    
    def test_recall_at_k(self):
        """Test R@K calculation."""
        y_true = [0, 2, 4]  # Relevant
        y_pred = [0, 1, 2, 3, 4]  # Ranked
        
        r_at_3 = recall_at_k(y_true, y_pred, k=3)
        
        # Top 3 has [0, 2] of [0, 2, 4] relevant, R@3 = 2/3
        self.assertAlmostEqual(r_at_3, 2/3, places=2)
    
    def test_ndcg_at_k(self):
        """Test nDCG@K calculation."""
        y_true = [3, 2, 1, 0]  # Relevance scores
        y_pred = [0, 1, 2, 3]  # Indices of ranked documents
        
        ndcg = ndcg_at_k(y_true, y_pred, k=4)
        
        # Perfect ranking → nDCG = 1.0
        self.assertAlmostEqual(ndcg, 1.0, places=2)


class TestRAGEvaluator(unittest.TestCase):
    """Tests for RAGEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = RAGEvaluator()
    
    def test_evaluate_faithfulness(self):
        """Test faithfulness evaluation."""
        answer = "Machine learning is a type of AI."
        contexts = ["Machine learning is a subset of artificial intelligence."]
        
        score = self.evaluator.evaluate_faithfulness(answer, contexts)
        
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_evaluate_relevance(self):
        """Test relevance evaluation."""
        question = "What is machine learning?"
        answer = "Machine learning is a branch of artificial intelligence."
        
        score = self.evaluator.evaluate_relevance(question, answer)
        
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_evaluate_context_precision(self):
        """Test context precision evaluation."""
        question = "What is AI?"
        contexts = ["AI is artificial intelligence.", "Weather today is sunny."]
        ground_truth = "AI stands for artificial intelligence."
        
        score = self.evaluator.evaluate_context_precision(question, contexts, ground_truth)
        
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_full_evaluation(self):
        """Test full RAG evaluation."""
        questions = ["What is ML?"]
        answers = ["ML is machine learning, a type of AI."]
        contexts = [["Machine learning is a subset of AI."]]
        ground_truths = ["Machine learning is AI that learns from data."]
        
        results = self.evaluator.evaluate(questions, answers, contexts, ground_truths)
        
        self.assertIn("faithfulness", results)
        self.assertIn("relevance", results)
        self.assertIn("context_precision", results)


class TestMLEvaluator(unittest.TestCase):
    """Tests for MLEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = MLEvaluator()
    
    def test_evaluate_classification(self):
        """Test classification evaluation."""
        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 0])
        
        results = self.evaluator.evaluate_classification(y_true, y_pred)
        
        self.assertIn("accuracy", results)
        self.assertIn("precision", results)
        self.assertIn("recall", results)
        self.assertIn("f1", results)
    
    def test_evaluate_regression(self):
        """Test regression evaluation."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.2, 2.9])
        
        results = self.evaluator.evaluate_regression(y_true, y_pred)
        
        self.assertIn("mse", results)
        self.assertIn("rmse", results)
        self.assertIn("mae", results)
        self.assertIn("r2", results)


class TestLLMEvaluator(unittest.TestCase):
    """Tests for LLMEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = LLMEvaluator()
    
    def test_evaluate_fluency(self):
        """Test fluency evaluation."""
        text = "This is a well-written sentence. It flows nicely."
        
        score = self.evaluator.evaluate_fluency(text)
        
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_evaluate_coherence(self):
        """Test coherence evaluation."""
        text = "First, we discuss AI. Furthermore, we explore its applications."
        
        score = self.evaluator.evaluate_coherence(text)
        
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_evaluate_relevance(self):
        """Test relevance evaluation."""
        prompt = "Explain machine learning"
        response = "Machine learning is a field of AI that uses algorithms to learn from data."
        
        score = self.evaluator.evaluate_relevance(prompt, response)
        
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_full_evaluation(self):
        """Test full LLM evaluation."""
        prompt = "What is AI?"
        response = "AI is artificial intelligence."
        
        results = self.evaluator.evaluate(prompt, response)
        
        self.assertIn("fluency", results)
        self.assertIn("coherence", results)
        self.assertIn("relevance", results)


class TestEvaluationReport(unittest.TestCase):
    """Tests for EvaluationReport class."""
    
    def test_add_metrics(self):
        """Test adding metrics to report."""
        report = EvaluationReport(title="Test Report")
        
        report.add_metrics("Classification", {"accuracy": 0.9, "f1": 0.85})
        
        self.assertIn("Classification", report.sections)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        report = EvaluationReport(title="Test")
        report.add_metrics("Test", {"value": 1.0})
        
        d = report.to_dict()
        
        self.assertIn("title", d)
        self.assertIn("sections", d)
    
    def test_to_json(self):
        """Test conversion to JSON."""
        report = EvaluationReport(title="Test")
        report.add_metrics("Test", {"value": 1.0})
        
        json_str = report.to_json()
        
        self.assertIn("title", json_str)
        self.assertIn("Test", json_str)
    
    def test_to_markdown(self):
        """Test conversion to Markdown."""
        report = EvaluationReport(title="Test Report")
        report.add_metrics("Metrics", {"accuracy": 0.9})
        
        md = report.to_markdown()
        
        self.assertIn("# Test Report", md)
        self.assertIn("accuracy", md)


class TestMetric(unittest.TestCase):
    """Tests for Metric dataclass."""
    
    def test_metric_creation(self):
        """Test creating a metric."""
        metric = Metric(
            name="accuracy",
            value=0.95,
            metric_type=MetricType.CLASSIFICATION
        )
        
        self.assertEqual(metric.name, "accuracy")
        self.assertEqual(metric.value, 0.95)
    
    def test_metric_to_dict(self):
        """Test converting metric to dict."""
        metric = Metric(name="f1", value=0.85)
        
        d = metric.to_dict()
        
        self.assertIn("name", d)
        self.assertIn("value", d)


if __name__ == "__main__":
    unittest.main()
