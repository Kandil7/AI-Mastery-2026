import unittest
import sys
import os
import torch
import pandas as pd
import tempfile
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.data_preprocessing.generate_synthetic_data import SyntheticDataGenerator
from src.llm.attention import create_attention_layer, AttentionConfig

class TestFeatureCompleteness(unittest.TestCase):
    
    def setUp(self):
        self.generator = SyntheticDataGenerator(seed=42)
        self.config = AttentionConfig(
            hidden_size=64,
            num_attention_heads=4,
            head_dim=16,
            attention_dropout=0.0
        )

    def test_synthetic_data_generation(self):
        """Test that synthetic data generators produce correct English outputs"""
        print("\nTesting Synthetic Data Generation...")
        
        # Customer Data
        customer_df = self.generator.generate_customer_data(n_samples=10)
        self.assertEqual(len(customer_df), 10)
        self.assertIn("New York", customer_df['city'].unique())
        print("- Customer data OK")

        # Medical Records (ensure English terms)
        medical_df = self.generator.generate_medical_records(n_records=50)
        # Check if at least one row has some condition (Hypertension/Diabetes etc)
        # Use a broader check to avoid randomness failure
        english_conditions = ["Hypertension", "Diabetes", "Asthma", "Heart Disease"]
        found_english = any(any(cond in record for cond in english_conditions) for record in medical_df['medical_conditions'])
        self.assertTrue(found_english, "No English medical conditions found in 50 samples")
        print("- Medical records OK")

        # Legal Docs
        legal_docs = self.generator.generate_legal_documents(n_documents=5)
        self.assertTrue("Clause" in legal_docs[0]['content'])
        print("- Legal documents OK")

    def test_attention_mechanisms(self):
        """Test new attention implementations"""
        print("\nTesting Attention Mechanisms...")
        
        # Input tensor [batch, seq, hidden]
        x = torch.randn(2, 10, 64)
        
        # 1. Standard (RoPE implicit)
        attn = create_attention_layer(self.config, "standard")
        out, _, _ = attn(x)
        self.assertEqual(out.shape, (2, 10, 64))
        print("- Standard Attention OK")
        
        # 2. Grouped Query Attention
        attn_gqa = create_attention_layer(self.config, "grouped_query")
        out_gqa, _, _ = attn_gqa(x)
        self.assertEqual(out_gqa.shape, (2, 10, 64))
        print("- GQA OK")
        
        # 3. Multi Query Attention
        attn_mqa = create_attention_layer(self.config, "multi_query")
        out_mqa, _, _ = attn_mqa(x)
        self.assertEqual(out_mqa.shape, (2, 10, 64))
        print("- MQA OK")

    def test_rag_notebook_existence(self):
        """Verify RAG notebook was created"""
        print("\nTesting RAG Notebook...")
        path = "k:\\learning\\technical\\ai-ml\\AI-Mastery-2026\\notebooks\\week_11\\03_rag_advanced_techniques.ipynb"
        self.assertTrue(os.path.exists(path))
        
        with open(path, 'r', encoding='utf-8') as f:
            content = json.load(f)
            # Check for English content
            cells = content['cells']
            found_english = False
            for cell in cells:
                if "Advanced RAG" in "".join(cell['source']):
                    found_english = True
                    break
            self.assertTrue(found_english, "Notebook does not contain expected English header")
        print("- Notebook OK")

if __name__ == '__main__':
    unittest.main()
