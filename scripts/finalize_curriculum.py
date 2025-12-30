
import os
import json
import textwrap

# Curriculum Definition based on full_plan_imp.md and README.md
CURRICULUM = {
    "week_05": {
        "title": "Production Engineering",
        "notebooks": [
            {
                "name": "01_model_serving_fastapi.ipynb",
                "title": "Model Serving with FastAPI",
                "content": """
import requests
from src.production.api import create_app
from fastapi.testclient import TestClient

# 1. Initialize the App
# In this notebook, we verify our production API wrapper.
app = create_app()
client = TestClient(app)

# 2. Health Check
response = client.get("/health")
print(f"Health Status: {response.json()}")

# 3. Simulate Prediction
# payload = {"features": [1.0, 2.0, 3.0]}
# response = client.post("/predict", json=payload)
# print(response.json())
"""
            },
            {
                "name": "02_drift_detection.ipynb",
                "title": "Drift Detection & Monitoring",
                "content": """
import numpy as np
from src.production.monitoring import DriftDetector

# 1. Setup Reference Data (Training)
reference_data = np.random.normal(0, 1, 1000)

# 2. Initialize Detector
detector = DriftDetector()
detector.fit(reference_data)

# 3. Detect Drift (Normal Data)
production_data_normal = np.random.normal(0, 1, 100)
is_drift, p_val = detector.detect(production_data_normal)
print(f"Normal Data Drift: {is_drift} (p={p_val:.4f})")

# 4. Detect Drift (Shifted Data)
production_data_drift = np.random.normal(1, 1, 100)
is_drift, p_val = detector.detect(production_data_drift)
print(f"Shifted Data Drift: {is_drift} (p={p_val:.4f})")
"""
            }
        ]
    },
    "week_06": {
        "title": "LLM Foundations",
        "notebooks": [
            {
                "name": "01_attention_mechanisms.ipynb",
                "title": "Attention from Scratch",
                "content": """
import torch
from src.llm.attention import MultiHeadAttention

# 1. Initialize MHA
d_model = 512
num_heads = 8
mha = MultiHeadAttention(d_model, num_heads)

# 2. Create Dummy Inputs (Batch, Seq, Dim)
x = torch.randn(2, 10, d_model)

# 3. Forward Pass
output, weights = mha(x, x, x)
print(f"Input Shape: {x.shape}")
print(f"Output Shape: {output.shape}")
"""
            }
        ]
    },
    "week_07": {
        "title": "System Design",
        "notebooks": [
            {
                "name": "01_system_design_framework.ipynb",
                "title": "AI System Design Framework",
                "content": """
# AI System Design Framework

In this week, we study the Case Studies located in `case_studies/`.

## Key Components

1. **Requirements Analysis**: Functional vs Non-Functional
2. **Data Pipeline**: Ingestion, Processing, Feature Engineering
3. **Model Architecture**: Selection, Training, Evaluation
4. **Serving Infrastructure**: API, Batch, Edge
5. **Monitoring**: Drift, Performance, Business Metrics

## Case Study Verification

Run the following scripts to see the architectures in action:

```python
# !python case_studies/medical_diagnosis_agent/run_demo.py
```
"""
            }
        ]
    },
    "week_08": {
        "title": "Advanced RAG",
        "notebooks": [
            {
                "name": "01_rag_pipeline.ipynb",
                "title": "Building a RAG Pipeline",
                "content": """
from src.llm.rag import RAGPipeline, Document

# 1. Setup Pipeline
rag = RAGPipeline(embedding_dim=384)

# 2. Index Documents
docs = [
    Document(content="The white-box approach emphasizes understanding foundations.", metadata={"source": "manifesto"}),
    Document(content="Transformers use self-attention mechanisms.", metadata={"source": "lecture"})
]
rag.add_documents(docs)

# 3. Retrieve
query = "What is the white-box approach?"
results = rag.retrieve(query, k=1)
print(f"Retrieved: {results[0].content}")
"""
            }
        ]
    },
    "week_09": {
        "title": "Fine-Tuning & PEFT",
        "notebooks": [
            {
                "name": "01_lora_implementation.ipynb",
                "title": "LoRA from Scratch",
                "content": """
import torch
from src.llm.fine_tuning import LoRALayer

# 1. Create Base Layer simulating a dense layer weight
in_features, out_features = 768, 768
lora = LoRALayer(in_features, out_features, r=8, alpha=16)

# 2. Forward Pass
x = torch.randn(1, in_features)
output = lora(x)

print(f"LoRA adapted output shape: {output.shape}")
"""
            }
        ]
    },
    # Add placeholders for remaining weeks to ensure structure exists
    "week_10": {"title": "Generative Models", "notebooks": [{"name": "01_intro.ipynb", "title": "Generative Models", "content": "# Generative Models\nSee src.ml.deep_learning for implementations."}]},
    "week_11": {"title": "Computer Vision", "notebooks": [{"name": "01_cnn_basics.ipynb", "title": "CNNs", "content": "# CNNs\nSee src.ml.deep_learning for Conv2D implementations."}]},
    "week_13": {"title": "Deployment at Scale", "notebooks": [{"name": "01_scaling.ipynb", "title": "Scaling", "content": "# Scaling\nRefer to src.production.deployment."}]},
    "week_14": {"title": "Security & Privacy", "notebooks": [{"name": "01_privacy.ipynb", "title": "Privacy", "content": "# Privacy\nSee case_studies/medical_diagnosis_agent for PII filtering."}]},
    "week_15": {"title": "Final Project", "notebooks": [{"name": "01_capstone.ipynb", "title": "Capstone", "content": "# Capstone Project\nIntegrate all modules to build a custom solution."}]},
}

def create_notebook(path, title, content_code):
    nb_structure = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"# {title}\n", "\n", "Generated Demonstration Notebook"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": content_code.strip().split("\n")
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb_structure, f, indent=2)
    print(f"Created: {path}")

def main():
    base_dir = "k:\\learning\\technical\\ai-ml\\AI-Mastery-2026\\notebooks"
    
    for week, data in CURRICULUM.items():
        week_dir = os.path.join(base_dir, week)
        
        # Create directory if missing
        if not os.path.exists(week_dir):
            os.makedirs(week_dir)
            print(f"Created directory: {week_dir}")
        elif not os.listdir(week_dir):
            print(f"Directory empty, populating: {week_dir}")
        else:
            print(f"Directory exists and not empty, checking contents: {week_dir}")
            # Optional: Add missing files if they don't exist?
            # For now, only fill if empty or specifically missing key files is safer 
            # to avoid overwriting user work.
            # BUT, for the purpose of "Complete Project", we'll ensure these demo notebooks exist.
        
        for nb in data['notebooks']:
            nb_path = os.path.join(week_dir, nb['name'])
            if not os.path.exists(nb_path):
                create_notebook(nb_path, nb['title'], nb['content'])
            else:
                print(f"  Skipping {nb['name']} (already exists)")

if __name__ == "__main__":
    main()
