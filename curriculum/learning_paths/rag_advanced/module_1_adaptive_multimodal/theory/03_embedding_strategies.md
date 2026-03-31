# Theory 3: Embedding Strategies for Multimodal RAG

## 3.1 Overview

This section covers advanced embedding strategies for multimodal content, including model selection, fine-tuning approaches, and optimization techniques for production systems.

## 3.2 Text Embedding Models

### 3.2.1 Model Comparison

| Model | Dimensions | Max Tokens | MTEB Score | Best For |
|-------|------------|------------|------------|----------|
| text-embedding-3-large | 3072 | 8191 | 64.6 | General purpose |
| text-embedding-3-small | 1536 | 8191 | 62.3 | Cost-effective |
| bge-large-en-v1.5 | 1024 | 512 | 64.2 | High accuracy |
| bge-base-en-v1.5 | 768 | 512 | 62.8 | Balanced |
| e5-large-v2 | 1024 | 512 | 63.7 | Query-document |
| jina-embeddings-v2 | 1024 | 8192 | 63.0 | Long documents |

### 3.2.2 Implementation Example

```python
from openai import AsyncOpenAI
from typing import List
import numpy as np

class TextEmbedder:
    def __init__(self, model: str = "text-embedding-3-large"):
        self.client = AsyncOpenAI()
        self.model = model
    
    async def embed(self, texts: List[str]) -> np.ndarray:
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return np.array([d.embedding for d in response.data])
```

## 3.3 Image Embedding Models

### 3.3.1 CLIP-Based Models

```python
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image

class ImageEmbedder:
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
    
    def embed(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        return features.numpy()[0]
```

### 3.3.2 Model Options

| Model | Dimensions | Training Data | Strengths |
|-------|------------|---------------|-----------|
| CLIP ViT-L/14 | 768 | 400M image-text | General purpose |
| CLIP ViT-B/32 | 512 | 400M image-text | Fast inference |
| SigLIP | 1152 | Web-scale | Better zero-shot |
| BLIP-2 | 768 | Image-caption | Caption generation |

## 3.4 Code Embedding Models

### 3.4.1 Specialized Code Models

| Model | Dimensions | Languages | Best For |
|-------|------------|-----------|----------|
| CodeBERT | 768 | Python, Java, etc. | Code understanding |
| GraphCodeBERT | 768 | Multiple | Code + structure |
| CodeT5 | 768 | Multiple | Code generation |
| StarCoder | 4096 | 80+ languages | Large context |

### 3.4.2 Implementation

```python
from transformers import AutoTokenizer, AutoModel
import torch

class CodeEmbedder:
    def __init__(self, model_name: str = "microsoft/graphcodebert-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
    
    def embed(self, code: str) -> np.ndarray:
        inputs = self.tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].numpy()[0]
```

## 3.5 Unified Embedding Spaces

### 3.5.1 Cross-Modal Alignment

For unified retrieval across modalities:

```python
class UnifiedEmbedder:
    """Maps all modalities to shared embedding space."""
    
    def __init__(self):
        # CLIP provides natural text-image alignment
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        # Project code embeddings to CLIP space
        self.code_projector = self._load_code_projector()
    
    def embed_text(self, text: str) -> np.ndarray:
        inputs = self.clip_processor(text=text, return_tensors="pt", padding=True)
        features = self.clip_model.get_text_features(**inputs)
        return features.numpy()[0]
    
    def embed_image(self, image: Image.Image) -> np.ndarray:
        inputs = self.clip_processor(images=image, return_tensors="pt")
        features = self.clip_model.get_image_features(**inputs)
        return features.numpy()[0]
    
    def embed_code(self, code: str) -> np.ndarray:
        # First get code embedding, then project to CLIP space
        code_emb = self._get_code_embedding(code)
        return self.code_projector(code_emb)
```

## 3.6 Embedding Optimization

### 3.6.1 Batch Processing

```python
async def batch_embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
    """Process embeddings in batches for efficiency."""
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = await self.embed(batch)
        all_embeddings.append(embeddings)
    
    return np.vstack(all_embeddings)
```

### 3.6.2 Caching Strategy

```python
import hashlib
from typing import Optional
import redis
import pickle

class EmbeddingCache:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url)
        self.ttl = 86400  # 24 hours
    
    def _get_cache_key(self, content: str, model: str) -> str:
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return f"embedding:{model}:{content_hash}"
    
    def get(self, content: str, model: str) -> Optional[np.ndarray]:
        key = self._get_cache_key(content, model)
        cached = self.redis.get(key)
        if cached:
            return pickle.loads(cached)
        return None
    
    def set(self, content: str, model: str, embedding: np.ndarray):
        key = self._get_cache_key(content, model)
        self.redis.setex(key, self.ttl, pickle.dumps(embedding))
```

## 3.7 Summary

Key points:
- Choose embedding models based on modality and use case
- CLIP provides natural cross-modal alignment
- Batch processing and caching essential for production
- Consider fine-tuning for domain-specific applications
