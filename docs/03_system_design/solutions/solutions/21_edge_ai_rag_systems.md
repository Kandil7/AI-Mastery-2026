# System Design Solution: Edge AI RAG for IoT Applications

## Problem Statement

Design an Edge AI Retrieval-Augmented Generation (RAG) system that can:
- Operate efficiently on resource-constrained IoT devices
- Provide low-latency responses for real-time applications
- Maintain privacy by processing data locally
- Handle intermittent connectivity scenarios
- Scale across distributed IoT networks
- Optimize energy consumption for battery-powered devices

## Solution Overview

This system design presents a comprehensive architecture for Edge AI RAG that brings RAG processing closer to IoT devices, reducing latency and bandwidth usage while enhancing privacy. The solution addresses the unique challenges of resource-constrained environments by implementing model optimization, efficient retrieval mechanisms, and adaptive processing strategies.

## 1. High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    └─────────────────┐
│   IoT Sensors   │────│  Edge RAG       │────│  Local          │
│   (Cameras,     │    │  Processing     │    │  Knowledge      │
│   Environmental,│    │  Unit          │    │  Base           │
│   Wearables)    │    │  (Embedded)    │    │  (Flash/SSD)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Data Preproc.  │────│  Lightweight    │────│  Vector         │
│  & Compression │    │  RAG Model      │    │  Database       │
│  (On-device)   │    │  (Distilled)    │    │  (FAISS/SPTAG) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
┌────────────────────────────────┼─────────────────────────────────┐
│                    Edge RAG Query Processing                    │
│  ┌─────────────────┐    ┌──────────────────┐    ┌──────────┐  │
│  │  Query         │────│  Local          │────│  Response│  │
│  │  Preprocessing │    │  Retrieval      │    │  Gen.   │  │
│  │  (Sensor Data) │    │  (Similarity)   │    │  (Text/ │  │
│  └─────────────────┘    └──────────────────┘    │  Action)│  │
└───────────────────────────────────────────────────└──────────┘──┘
```

## 2. Core Components

### 2.1 Edge-Optimized RAG Core
```python
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional
import onnxruntime as ort
import faiss
import json
import os
from transformers import AutoTokenizer, AutoModel
import gc
import time
import asyncio
from dataclasses import dataclass
from enum import Enum

@dataclass
class DeviceCapabilities:
    """Represents the capabilities of an edge device"""
    cpu_cores: int
    memory_mb: int
    storage_mb: int
    gpu_available: bool
    gpu_memory_mb: int
    power_source: str  # "battery", "plugged", "solar"
    thermal_limit: float  # max temperature in Celsius

class PowerMode(Enum):
    """Power management modes for edge devices"""
    PERFORMANCE = "performance"
    BALANCED = "balanced"
    POWER_SAVE = "power_save"
    ULTRA_LOW_POWER = "ultra_low_power"

class EdgeRAGCore:
    """
    Edge-optimized RAG model for IoT applications
    """
    def __init__(self, 
                 model_path: str, 
                 tokenizer_path: str, 
                 device_capabilities: DeviceCapabilities,
                 power_mode: PowerMode = PowerMode.BALANCED):
        self.device_caps = device_capabilities
        self.power_mode = power_mode
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        
        # Initialize based on device capabilities
        self.model = self._initialize_model()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Initialize power management
        self.power_manager = PowerManager(self.device_caps, self.power_mode)
        
        # Initialize components for edge optimization
        self.quantized_model = None
        self.compiled_model = None
        self.cache = QueryCache(max_size=self._get_cache_size())
        
    def _initialize_model(self):
        """
        Initialize model based on device capabilities
        """
        # Determine model size based on device memory
        if self.device_caps.memory_mb < 512:
            # Use tiny model for very constrained devices
            model_name = "prajjwal1/bert-tiny"
        elif self.device_caps.memory_mb < 1024:
            # Use mini model for constrained devices
            model_name = "prajjwal1/bert-mini"
        elif self.device_caps.memory_mb < 2048:
            # Use small model for moderately constrained devices
            model_name = "prajjwal1/bert-small"
        else:
            # Use base model for more capable devices
            model_name = "distilbert-base-uncased"
        
        model = AutoModel.from_pretrained(
            model_name,
            torchscript=True,
            low_cpu_mem_usage=True
        )
        
        # Move to appropriate device
        if self.device_caps.gpu_available and self.device_caps.gpu_memory_mb > 128:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def _get_cache_size(self) -> int:
        """
        Determine cache size based on device capabilities
        """
        if self.device_caps.memory_mb < 512:
            return 10  # Small cache for constrained devices
        elif self.device_caps.memory_mb < 1024:
            return 50  # Medium cache
        else:
            return 100  # Larger cache for capable devices
    
    def encode_query(self, query: str, max_length: int = 128) -> torch.Tensor:
        """
        Encode query for retrieval using the edge-optimized model
        """
        # Tokenize input
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling to get sentence embedding
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.cpu()
    
    def generate_response(self, context: str, query: str, max_new_tokens: int = 100) -> str:
        """
        Generate response using the edge-optimized model
        """
        # Combine context and query
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generate response with power management
        with self.power_manager.manage_inference():
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    # Limit generation to conserve resources
                    min_length=min(10, max_new_tokens // 2)
                )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the answer part
        if "Answer:" in response:
            answer = response.split("Answer:")[-1].strip()
        else:
            answer = response[len(prompt):].strip()
        
        return answer

class PowerManager:
    """
    Manages power consumption for edge devices
    """
    def __init__(self, device_caps: DeviceCapabilities, power_mode: PowerMode):
        self.device_caps = device_caps
        self.power_mode = power_mode
        self.inference_count = 0
        self.energy_consumed = 0.0
        
    def manage_inference(self):
        """
        Context manager for inference with power management
        """
        return PowerManagedInference(self)
    
    def get_power_settings(self) -> Dict[str, Any]:
        """
        Get power management settings based on mode
        """
        if self.power_mode == PowerMode.PERFORMANCE:
            return {
                'batch_size': 16,
                'precision': 'fp32',
                'max_workers': self.device_caps.cpu_cores
            }
        elif self.power_mode == PowerMode.BALANCED:
            return {
                'batch_size': 8,
                'precision': 'fp16',
                'max_workers': max(1, self.device_caps.cpu_cores // 2)
            }
        elif self.power_mode == PowerMode.POWER_SAVE:
            return {
                'batch_size': 4,
                'precision': 'int8',
                'max_workers': 1
            }
        else:  # ULTRA_LOW_POWER
            return {
                'batch_size': 1,
                'precision': 'int8',
                'max_workers': 1
            }

class PowerManagedInference:
    """
    Context manager for power-managed inference
    """
    def __init__(self, power_manager: PowerManager):
        self.power_manager = power_manager
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        # Apply power settings
        settings = self.power_manager.get_power_settings()
        # In practice, this would configure the model with these settings
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        # Calculate energy consumption (simplified)
        power_draw = self._estimate_power_draw()
        self.power_manager.energy_consumed += power_draw * duration
        self.power_manager.inference_count += 1
    
    def _estimate_power_draw(self) -> float:
        """
        Estimate power draw based on device capabilities and settings
        """
        # Simplified power estimation
        base_power = 0.5  # Base power in watts
        if self.power_manager.device_caps.gpu_available:
            base_power += 1.0  # GPU power
        
        # Adjust based on power mode
        if self.power_manager.power_mode == PowerMode.PERFORMANCE:
            multiplier = 1.0
        elif self.power_manager.power_mode == PowerMode.BALANCED:
            multiplier = 0.7
        elif self.power_manager.power_mode == PowerMode.POWER_SAVE:
            multiplier = 0.4
        else:  # ULTRA_LOW_POWER
            multiplier = 0.2
        
        return base_power * multiplier

class QueryCache:
    """
    Cache for frequently asked queries
    """
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache = {}
        self.access_times = {}
        self.usage_counts = {}
    
    def get(self, query: str) -> Optional[str]:
        """
        Get cached response for query
        """
        if query in self.cache:
            # Update access time and usage count
            self.access_times[query] = time.time()
            self.usage_counts[query] = self.usage_counts.get(query, 0) + 1
            return self.cache[query]
        return None
    
    def put(self, query: str, response: str):
        """
        Put query-response pair in cache
        """
        if len(self.cache) >= self.max_size:
            # Remove least recently used item
            lru_query = min(self.access_times.items(), key=lambda x: x[1])[0]
            del self.cache[lru_query]
            del self.access_times[lru_query]
            if lru_query in self.usage_counts:
                del self.usage_counts[lru_query]
        
        self.cache[query] = response
        self.access_times[query] = time.time()
        self.usage_counts[query] = 1
    
    def get_hit_rate(self) -> float:
        """
        Calculate cache hit rate
        """
        if not self.usage_counts:
            return 0.0
        return len(self.cache) / sum(self.usage_counts.values())
```

### 2.2 IoT Data Processing Pipeline
```python
import cv2
import librosa
import pandas as pd
from PIL import Image
import struct
import zlib
import io
from datetime import datetime

class IoTDataProcessor:
    """
    Processes IoT sensor data for RAG system
    """
    def __init__(self, device_caps: DeviceCapabilities):
        self.device_caps = device_caps
        self.compression_level = self._get_compression_level()
        self.max_image_size = self._get_image_size()
        self.audio_sample_rate = 16000  # Standard for edge models
        
    def _get_compression_level(self) -> int:
        """
        Determine compression level based on device capabilities
        """
        if self.device_caps.memory_mb < 512:
            return 9  # Maximum compression
        elif self.device_caps.memory_mb < 1024:
            return 7  # High compression
        else:
            return 6  # Balanced compression
    
    def _get_image_size(self) -> tuple:
        """
        Determine appropriate image size based on device capabilities
        """
        if self.device_caps.memory_mb < 512:
            return (112, 112)  # Small for constrained devices
        elif self.device_caps.memory_mb < 1024:
            return (224, 224)  # Standard for moderate devices
        else:
            return (336, 336)  # Larger for capable devices
    
    def preprocess_camera_data(self, image_data: bytes) -> Dict[str, Any]:
        """
        Preprocess camera data from IoT devices
        """
        try:
            # Convert bytes to image
            image = Image.open(io.BytesIO(image_data))
            
            # Resize image based on device capabilities
            image = image.resize(self.max_image_size, Image.Resampling.LANCZOS)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert back to bytes for processing
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG', quality=85)
            compressed_data = img_byte_arr.getvalue()
            
            # Extract features for RAG
            features = self._extract_visual_features(image)
            
            return {
                'type': 'image',
                'data': compressed_data,
                'features': features,
                'size_original': len(image_data),
                'size_compressed': len(compressed_data),
                'compression_ratio': len(image_data) / len(compressed_data) if len(compressed_data) > 0 else 0,
                'processing_time': time.time()
            }
        except Exception as e:
            return {
                'type': 'image',
                'error': str(e),
                'processing_time': time.time()
            }
    
    def preprocess_sensor_data(self, sensor_readings: Dict[str, float]) -> Dict[str, Any]:
        """
        Preprocess environmental sensor data
        """
        try:
            # Normalize readings based on sensor type and device capabilities
            normalized_readings = {}
            for key, value in sensor_readings.items():
                # Apply normalization based on sensor type
                if 'temperature' in key.lower():
                    normalized_readings[key] = self._normalize_temperature(value)
                elif 'humidity' in key.lower():
                    normalized_readings[key] = self._normalize_humidity(value)
                elif 'pressure' in key.lower():
                    normalized_readings[key] = self._normalize_pressure(value)
                elif 'light' in key.lower() or 'lux' in key.lower():
                    normalized_readings[key] = self._normalize_light(value)
                else:
                    normalized_readings[key] = value  # Assume already normalized
            
            # Create textual description
            description = self._create_sensor_description(normalized_readings)
            
            return {
                'type': 'sensor',
                'readings': normalized_readings,
                'description': description,
                'timestamp': time.time(),
                'processing_time': time.time()
            }
        except Exception as e:
            return {
                'type': 'sensor',
                'error': str(e),
                'processing_time': time.time()
            }
    
    def preprocess_audio_data(self, audio_data: bytes) -> Dict[str, Any]:
        """
        Preprocess audio data from IoT devices
        """
        try:
            # Load audio
            audio_np, sr = librosa.load(io.BytesIO(audio_data), sr=self.audio_sample_rate)
            
            # Extract features
            mfccs = librosa.feature.mfcc(y=audio_np, sr=sr, n_mfcc=13)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_np, sr=sr)[0]
            chroma = librosa.feature.chroma_stft(y=audio_np, sr=sr)
            
            # Create textual description
            description = self._describe_audio_features(mfccs, spectral_centroids, chroma)
            
            return {
                'type': 'audio',
                'features': {
                    'mfccs': mfccs.tolist(),
                    'spectral_centroids': spectral_centroids.tolist(),
                    'chroma': chroma.tolist()
                },
                'description': description,
                'duration': len(audio_np) / sr,
                'processing_time': time.time()
            }
        except Exception as e:
            return {
                'type': 'audio',
                'error': str(e),
                'processing_time': time.time()
            }
    
    def _normalize_temperature(self, temp: float) -> float:
        """
        Normalize temperature reading to 0-1 range
        """
        # Assuming range of -50°C to 50°C for general IoT applications
        return max(0.0, min(1.0, (temp + 50) / 100))
    
    def _normalize_humidity(self, humidity: float) -> float:
        """
        Normalize humidity reading to 0-1 range
        """
        # Humidity is already 0-100%, normalize to 0-1
        return max(0.0, min(1.0, humidity / 100))
    
    def _normalize_pressure(self, pressure: float) -> float:
        """
        Normalize pressure reading to 0-1 range
        """
        # Assuming range of 950-1050 hPa
        return max(0.0, min(1.0, (pressure - 950) / 100))
    
    def _normalize_light(self, lux: float) -> float:
        """
        Normalize light reading to 0-1 range
        """
        # Assuming range of 0-100,000 lux (from moonlight to bright sun)
        return max(0.0, min(1.0, lux / 100000))
    
    def _create_sensor_description(self, readings: Dict[str, float]) -> str:
        """
        Create textual description of sensor readings
        """
        descriptions = []
        
        for key, value in readings.items():
            if 'temperature' in key.lower():
                descriptions.append(f"Temperature: {value*100-50:.1f}°C")
            elif 'humidity' in key.lower():
                descriptions.append(f"Humidity: {value*100:.1f}%")
            elif 'pressure' in key.lower():
                descriptions.append(f"Pressure: {950 + value*100:.1f} hPa")
            elif 'light' in key.lower() or 'lux' in key.lower():
                descriptions.append(f"Light: {value*100000:.0f} lux")
            else:
                descriptions.append(f"{key}: {value}")
        
        return ", ".join(descriptions)
    
    def _describe_audio_features(self, mfccs: np.ndarray, spectral_centroids: np.ndarray, chroma: np.ndarray) -> str:
        """
        Create textual description of audio features
        """
        # Calculate averages
        avg_mfcc = np.mean(mfccs, axis=1)
        avg_spectral = np.mean(spectral_centroids)
        avg_chroma = np.mean(chroma, axis=1)
        
        # Create description
        return f"Audio features: MFCC avg={avg_mfcc[:3]}, Spectral centroid avg={avg_spectral:.2f}, Chroma avg={avg_chroma[:3]}"
    
    def _extract_visual_features(self, image: Image.Image) -> Dict[str, Any]:
        """
        Extract visual features from image
        """
        # Convert to numpy array
        img_array = np.array(image)
        
        # Calculate basic statistics
        mean_color = np.mean(img_array, axis=(0, 1))
        std_color = np.std(img_array, axis=(0, 1))
        
        # Calculate texture features using gray-level co-occurrence matrix
        gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        texture_features = self._calculate_texture_features(gray_img)
        
        return {
            'mean_color': mean_color.tolist(),
            'std_color': std_color.tolist(),
            'texture': texture_features,
            'dominant_colors': self._find_dominant_colors(img_array)
        }
    
    def _calculate_texture_features(self, gray_img: np.ndarray) -> Dict[str, float]:
        """
        Calculate texture features using GLCM
        """
        # Simplified texture calculation
        contrast = np.std(gray_img)
        homogeneity = 1.0 / (1.0 + np.var(gray_img))
        
        return {
            'contrast': float(contrast),
            'homogeneity': float(homogeneity)
        }
    
    def _find_dominant_colors(self, img_array: np.ndarray, k: int = 5) -> List[List[float]]:
        """
        Find dominant colors in image using K-means clustering
        """
        # Reshape image to pixels
        pixels = img_array.reshape(-1, 3)
        
        # Apply K-means clustering (simplified)
        # In practice, use sklearn for full K-means
        # For this example, we'll use a simple approach
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        
        # Get top k colors by frequency
        top_indices = np.argsort(counts)[-k:][::-1]
        dominant_colors = unique_colors[top_indices].tolist()
        
        return dominant_colors
```

### 2.3 Edge Vector Database
```python
class EdgeVectorDatabase:
    """
    Optimized vector database for edge devices
    """
    def __init__(self, dimension: int = 128, device_caps: DeviceCapabilities = None):
        self.dimension = dimension
        self.device_caps = device_caps
        self.index = self._create_index()
        self.documents = {}  # Maps index to document content
        self.metadata = {}   # Maps index to document metadata
        self.doc_id_to_idx = {}  # Maps document ID to index
        
        # Determine storage strategy based on device capabilities
        self.storage_strategy = self._determine_storage_strategy()
        
        # Normalize for cosine similarity
        faiss.normalize_L2(self.index.d)
    
    def _determine_storage_strategy(self) -> str:
        """
        Determine storage strategy based on device capabilities
        """
        if self.device_caps and self.device_caps.storage_mb < 100:
            return "memory_only"
        elif self.device_caps and self.device_caps.storage_mb < 500:
            return "memory_with_swap"
        else:
            return "disk_optimized"
    
    def _create_index(self):
        """
        Create FAISS index optimized for edge devices
        """
        if self.device_caps and self.device_caps.memory_mb < 512:
            # Use smaller index for constrained devices
            index = faiss.IndexFlatIP(self.dimension)
        elif self.device_caps and self.device_caps.memory_mb < 1024:
            # Use IVF index for moderate devices
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, 50)  # 50 clusters
        else:
            # Use more sophisticated index for capable devices
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFPQ(quantizer, self.dimension, 64, 8)  # 64 clusters, 8 bits per sub-vector
        
        return index
    
    def add_documents(self, embeddings: np.ndarray, contents: List[str], 
                     metadata_list: List[Dict] = None, doc_ids: List[str] = None):
        """
        Add documents to the vector database
        """
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to FAISS index
        start_idx = self.index.ntotal
        self.index.add(embeddings.astype('float32'))
        
        # Store documents and metadata
        for i, (content, meta, doc_id) in enumerate(zip(contents, 
                                                        metadata_list or [{}]*len(contents),
                                                        doc_ids or [f"doc_{start_idx+j}" for j in range(len(contents))])):
            idx = start_idx + i
            self.documents[idx] = self._sanitize_content(content)
            self.metadata[idx] = meta
            self.doc_id_to_idx[doc_id] = idx
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Search for similar documents
        """
        # Normalize query
        faiss.normalize_L2(query_embedding.reshape(1, -1))
        
        # Perform search
        scores, indices = self.index.search(query_embedding.reshape(1, -1), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx in self.documents:
                results.append({
                    'id': self._get_doc_id(idx),
                    'score': float(score),
                    'content': self.documents[idx],
                    'metadata': self.metadata[idx]
                })
        
        return results
    
    def _sanitize_content(self, content: str) -> str:
        """
        Sanitize content to reduce storage requirements
        """
        # Truncate very long content
        max_length = 500 if self.device_caps and self.device_caps.storage_mb < 500 else 1000
        if len(content) > max_length:
            return content[:max_length] + "..."
        
        return content
    
    def _get_doc_id(self, idx: int) -> str:
        """
        Get document ID for a given index
        """
        for doc_id, stored_idx in self.doc_id_to_idx.items():
            if stored_idx == idx:
                return doc_id
        return f"unknown_{idx}"
    
    def save_index(self, filepath: str):
        """
        Save the index to disk
        """
        faiss.write_index(self.index, f"{filepath}.index")
        
        # Save documents and metadata
        with open(f"{filepath}_docs.json", 'w') as f:
            json.dump({
                'documents': {str(k): v for k, v in self.documents.items()},
                'metadata': {str(k): v for k, v in self.metadata.items()},
                'doc_id_to_idx': self.doc_id_to_idx
            }, f)
    
    def load_index(self, filepath: str):
        """
        Load the index from disk
        """
        self.index = faiss.read_index(f"{filepath}.index")
        
        # Load documents and metadata
        with open(f"{filepath}_docs.json", 'r') as f:
            data = json.load(f)
            self.documents = {int(k): v for k, v in data['documents'].items()}
            self.metadata = {int(k): v for k, v in data['metadata'].items()}
            self.doc_id_to_idx = data['doc_id_to_idx']

class AdaptiveRetriever:
    """
    Adaptive retrieval system that adjusts based on device conditions
    """
    def __init__(self, vector_db: EdgeVectorDatabase, device_caps: DeviceCapabilities):
        self.vector_db = vector_db
        self.device_caps = device_caps
        self.retrieval_strategy = self._determine_strategy()
        self.performance_history = []
    
    def _determine_strategy(self) -> str:
        """
        Determine retrieval strategy based on device capabilities
        """
        if self.device_caps.memory_mb < 512:
            return "exact_search"
        elif self.device_caps.memory_mb < 1024:
            return "approximate_search"
        else:
            return "hybrid_search"
    
    def retrieve(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Retrieve documents using adaptive strategy
        """
        start_time = time.time()
        
        if self.retrieval_strategy == "exact_search":
            results = self._exact_search(query_embedding, k)
        elif self.retrieval_strategy == "approximate_search":
            results = self._approximate_search(query_embedding, k)
        else:  # hybrid_search
            results = self._hybrid_search(query_embedding, k)
        
        end_time = time.time()
        
        # Record performance
        self.performance_history.append({
            'timestamp': time.time(),
            'strategy': self.retrieval_strategy,
            'query_time': end_time - start_time,
            'results_returned': len(results)
        })
        
        return results
    
    def _exact_search(self, query_embedding: np.ndarray, k: int) -> List[Dict]:
        """
        Perform exact search (more accurate but slower)
        """
        return self.vector_db.search(query_embedding, k)
    
    def _approximate_search(self, query_embedding: np.ndarray, k: int) -> List[Dict]:
        """
        Perform approximate search (faster but less accurate)
        """
        # In practice, this would use approximate nearest neighbor methods
        # For this example, we'll use the exact search but with a smaller k
        return self.vector_db.search(query_embedding, min(k, 3))
    
    def _hybrid_search(self, query_embedding: np.ndarray, k: int) -> List[Dict]:
        """
        Perform hybrid search (balance of accuracy and speed)
        """
        return self.vector_db.search(query_embedding, k)
```

### 2.4 Edge RAG Orchestrator
```python
class EdgeRAGOrchestrator:
    """
    Orchestrates the edge RAG system for IoT applications
    """
    def __init__(self, 
                 model_path: str, 
                 tokenizer_path: str, 
                 device_caps: DeviceCapabilities,
                 power_mode: PowerMode = PowerMode.BALANCED):
        self.edge_model = EdgeRAGCore(model_path, tokenizer_path, device_caps, power_mode)
        self.vector_db = EdgeVectorDatabase(device_caps=device_caps)
        self.data_processor = IoTDataProcessor(device_caps)
        self.adaptive_retriever = AdaptiveRetriever(self.vector_db, device_caps)
        self.query_cache = self.edge_model.cache
        self.energy_monitor = EnergyMonitor(device_caps)
        self.connectivity_manager = ConnectivityManager()
        
        # Initialize components
        self._initialize_components()
        
    def _initialize_components(self):
        """
        Initialize all components with device-appropriate settings
        """
        # Quantize model if device is constrained
        if self.edge_model.device_caps.memory_mb < 1024:
            self.edge_model.quantize_model()
    
    def process_iot_query(self, iot_data: Dict[str, Any], query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Process a query from IoT device with sensor data context
        """
        start_time = time.time()
        start_energy = self.energy_monitor.get_energy_consumption()
        
        # Check cache first
        cache_key = f"{query}_{str(iot_data)}"
        cached_result = self.query_cache.get(cache_key)
        if cached_result:
            return {
                'response': cached_result,
                'source': 'cache',
                'query_time_ms': 0,  # Negligible cache time
                'energy_consumed': 0,  # No processing energy
                'power_efficiency': float('inf'),
                'cache_hit': True
            }
        
        # Preprocess IoT data
        processed_data = self._preprocess_iot_data(iot_data)
        
        # Encode query
        query_embedding = self.edge_model.encode_query(query)
        
        # Retrieve relevant context
        retrieved_docs = self.adaptive_retriever.retrieve(query_embedding, top_k)
        
        # Combine IoT context with retrieved documents
        combined_context = self._combine_context(processed_data, retrieved_docs)
        
        # Generate response
        response = self.edge_model.generate_response(combined_context, query)
        
        end_time = time.time()
        end_energy = self.energy_monitor.get_energy_consumption()
        
        result = {
            'response': response,
            'retrieved_docs': retrieved_docs,
            'iot_context': processed_data,
            'query_time_ms': (end_time - start_time) * 1000,
            'energy_consumed': end_energy - start_energy,
            'power_efficiency': (end_time - start_time) / (end_energy - start_energy) if (end_energy - start_energy) > 0 else float('inf'),
            'cache_hit': False
        }
        
        # Cache the result
        self.query_cache.put(cache_key, response)
        
        return result
    
    def _preprocess_iot_data(self, iot_data: Dict[str, Any]) -> str:
        """
        Preprocess IoT data into textual context
        """
        if iot_data['type'] == 'sensor':
            return f"IoT Sensor Data: {iot_data['description']}"
        elif iot_data['type'] == 'image':
            features = iot_data['features']
            return f"IoT Camera Data: Image with dominant colors {features['dominant_colors'][:3]}, texture contrast {features['texture']['contrast']:.2f}"
        elif iot_data['type'] == 'audio':
            return f"IoT Audio Data: {iot_data['description']}"
        else:
            return f"IoT Data: {str(iot_data)}"
    
    def _combine_context(self, iot_context: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Combine IoT context with retrieved documents
        """
        context_parts = [iot_context]
        
        for doc in retrieved_docs:
            context_parts.append(f"Related Info: {doc['content'][:200]}...")  # Limit length
        
        return " ".join(context_parts)
    
    def batch_process_queries(self, queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple queries efficiently
        """
        results = []
        
        for query_data in queries:
            iot_data = query_data.get('iot_data', {})
            query = query_data.get('query', '')
            top_k = query_data.get('top_k', 3)
            
            result = self.process_iot_query(iot_data, query, top_k)
            results.append(result)
        
        return results
    
    def update_knowledge_base(self, new_documents: List[Dict[str, Any]]):
        """
        Update the local knowledge base with new information
        """
        for doc in new_documents:
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            doc_id = doc.get('id', None)
            
            # Generate embedding for the content
            embedding = self.edge_model.encode_query(content)
            
            # Add to vector database
            self.vector_db.add_documents(
                embeddings=embedding.numpy().reshape(1, -1),
                contents=[content],
                metadata_list=[metadata],
                doc_ids=[doc_id] if doc_id else None
            )

class EnergyMonitor:
    """
    Monitor energy consumption of edge RAG system
    """
    def __init__(self, device_caps: DeviceCapabilities):
        self.device_caps = device_caps
        self.base_power = self._calculate_base_power()
        self.active_power = self._calculate_active_power()
        self.energy_consumed = 0.0
        self.start_time = time.time()
    
    def _calculate_base_power(self) -> float:
        """
        Calculate base power consumption based on device capabilities
        """
        # Base power for different device classes
        if self.device_caps.memory_mb < 512:
            return 0.1  # Very low power device
        elif self.device_caps.memory_mb < 1024:
            return 0.3  # Low power device
        elif self.device_caps.memory_mb < 2048:
            return 0.5  # Medium power device
        else:
            return 0.8  # Higher power device
    
    def _calculate_active_power(self) -> float:
        """
        Calculate additional power when actively processing
        """
        if self.device_caps.gpu_available:
            return 1.5  # GPU processing
        else:
            return 1.0  # CPU processing
    
    def get_energy_consumption(self) -> float:
        """
        Get current estimated energy consumption in Joules
        """
        elapsed_time = time.time() - self.start_time
        current_power = self.base_power + self.active_power
        return current_power * elapsed_time  # Power * Time = Energy
    
    def estimate_processing_energy(self, duration_ms: float) -> float:
        """
        Estimate energy used during processing
        """
        duration_hours = duration_ms / (1000 * 3600)  # Convert ms to hours
        return (self.base_power + self.active_power) * duration_hours * 3600  # Convert to joules

class ConnectivityManager:
    """
    Manage connectivity for edge devices
    """
    def __init__(self):
        self.connection_status = "disconnected"  # "connected", "limited", "disconnected"
        self.bandwidth_mbps = 0
        self.last_sync_time = 0
    
    def check_connectivity(self) -> Dict[str, Any]:
        """
        Check current connectivity status
        """
        # In practice, this would check actual network status
        # For this example, we'll return mock data
        return {
            'status': self.connection_status,
            'bandwidth_mbps': self.bandwidth_mbps,
            'last_sync': self.last_sync_time,
            'can_sync': self.connection_status == "connected"
        }
    
    def sync_with_cloud(self, local_changes: List[Dict[str, Any]]) -> bool:
        """
        Sync local changes with cloud when connectivity is available
        """
        if self.connection_status != "connected":
            return False
        
        # In practice, this would upload changes to cloud
        # For this example, we'll just return success
        self.last_sync_time = time.time()
        return True
```

## 3. IoT-Specific Components

### 3.1 IoT-Specific RAG Extensions
```python
class IoTSpecificRAG:
    """
    IoT-specific extensions to the RAG system
    """
    def __init__(self, orchestrator: EdgeRAGOrchestrator):
        self.orchestrator = orchestrator
        self.anomaly_detector = IoTAnomalyDetector()
        self.action_generator = IoTActionGenerator()
        self.context_awareness = ContextAwarenessEngine()
        
    def process_smart_home_query(self, sensor_data: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Process queries specific to smart home IoT applications
        """
        # Detect anomalies in sensor data
        anomalies = self.anomaly_detector.detect(sensor_data)
        
        # Process with edge RAG
        iot_data = {
            'type': 'sensor',
            'readings': sensor_data,
            'anomalies': anomalies
        }
        
        result = self.orchestrator.process_iot_query(iot_data, query)
        
        # Generate IoT-specific actions
        actions = self.action_generator.generate_actions(result['response'], sensor_data)
        
        result['actions'] = actions
        result['anomalies_detected'] = anomalies
        
        return result
    
    def process_industrial_iot_query(self, equipment_data: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Process queries specific to industrial IoT applications
        """
        # Perform industrial-specific anomaly detection
        anomalies = self.anomaly_detector.detect_equipment_anomalies(equipment_data)
        
        # Process with edge RAG
        iot_data = {
            'type': 'equipment',
            'readings': equipment_data,
            'anomalies': anomalies
        }
        
        result = self.orchestrator.process_iot_query(iot_data, query)
        
        # Generate industrial-specific actions
        actions = self.action_generator.generate_maintenance_actions(result['response'], equipment_data)
        
        result['actions'] = actions
        result['anomalies_detected'] = anomalies
        result['maintenance_recommendations'] = self._generate_maintenance_insights(equipment_data)
        
        return result
    
    def process_wearable_health_query(self, health_data: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Process queries specific to wearable health monitoring
        """
        # Detect health anomalies
        anomalies = self.anomaly_detector.detect_health_anomalies(health_data)
        
        # Process with edge RAG
        iot_data = {
            'type': 'health',
            'readings': health_data,
            'anomalies': anomalies
        }
        
        result = self.orchestrator.process_iot_query(iot_data, query)
        
        # Generate health-specific recommendations
        recommendations = self.action_generator.generate_health_recommendations(result['response'], health_data)
        
        result['recommendations'] = recommendations
        result['anomalies_detected'] = anomalies
        result['health_insights'] = self._generate_health_insights(health_data)
        
        return result
    
    def _generate_maintenance_insights(self, equipment_data: Dict[str, Any]) -> List[str]:
        """
        Generate maintenance insights based on equipment data
        """
        insights = []
        
        # Example insights based on common industrial parameters
        if 'temperature' in equipment_data and equipment_data['temperature'] > 80:
            insights.append("Equipment temperature elevated - recommend inspection")
        
        if 'vibration' in equipment_data and equipment_data['vibration'] > 2.5:
            insights.append("Abnormal vibration detected - possible misalignment")
        
        if 'pressure' in equipment_data and equipment_data['pressure'] < 1.0:
            insights.append("Low pressure detected - check for leaks")
        
        return insights
    
    def _generate_health_insights(self, health_data: Dict[str, Any]) -> List[str]:
        """
        Generate health insights based on wearable data
        """
        insights = []
        
        # Example insights based on common health parameters
        if 'heart_rate' in health_data and health_data['heart_rate'] > 100:
            insights.append("Elevated heart rate detected - consider rest or medical attention")
        
        if 'step_count' in health_data and health_data['step_count'] < 5000:
            insights.append("Low activity level - consider increasing physical activity")
        
        if 'sleep_duration' in health_data and health_data['sleep_duration'] < 6:
            insights.append("Insufficient sleep duration - aim for 7-9 hours")
        
        return insights

class IoTAnomalyDetector:
    """
    Detect anomalies in IoT sensor data
    """
    def __init__(self):
        self.baseline_stats = {}
        self.anomaly_threshold = 2.0  # Standard deviations
        self.health_thresholds = {
            'heart_rate': (60, 100),
            'body_temperature': (36.0, 37.5),
            'blood_pressure_sys': (90, 140),
            'blood_pressure_dia': (60, 90)
        }
    
    def detect(self, sensor_data: Dict[str, float]) -> Dict[str, bool]:
        """
        Detect anomalies in sensor readings
        """
        anomalies = {}
        
        for sensor_name, reading in sensor_data.items():
            if sensor_name not in self.baseline_stats:
                # Initialize baseline with first reading
                self.baseline_stats[sensor_name] = {
                    'mean': reading,
                    'std': 0.1,  # Initial standard deviation
                    'count': 1
                }
                continue
            
            # Update baseline statistics incrementally
            stats = self.baseline_stats[sensor_name]
            old_mean = stats['mean']
            stats['mean'] = (stats['mean'] * stats['count'] + reading) / (stats['count'] + 1)
            stats['std'] = np.sqrt(((stats['std']**2) * stats['count'] + (reading - old_mean) * (reading - stats['mean'])) / (stats['count'] + 1))
            stats['count'] += 1
            
            # Check if reading is anomalous
            z_score = abs(reading - stats['mean']) / (stats['std'] + 1e-8)  # Add small value to avoid division by zero
            anomalies[sensor_name] = z_score > self.anomaly_threshold
        
        return anomalies
    
    def detect_equipment_anomalies(self, equipment_data: Dict[str, float]) -> Dict[str, bool]:
        """
        Detect anomalies specific to industrial equipment
        """
        anomalies = {}
        
        # Industrial-specific thresholds
        thresholds = {
            'temperature': (10, 90),  # Min, Max acceptable range
            'pressure': (0.5, 10.0),
            'vibration': (0.1, 3.0),
            'rpm': (100, 5000)
        }
        
        for param, reading in equipment_data.items():
            if param in thresholds:
                min_val, max_val = thresholds[param]
                anomalies[param] = not (min_val <= reading <= max_val)
            else:
                # Use statistical method for unknown parameters
                if param not in self.baseline_stats:
                    self.baseline_stats[param] = {
                        'mean': reading,
                        'std': 0.1,
                        'count': 1
                    }
                    continue
                
                stats = self.baseline_stats[param]
                old_mean = stats['mean']
                stats['mean'] = (stats['mean'] * stats['count'] + reading) / (stats['count'] + 1)
                stats['std'] = np.sqrt(((stats['std']**2) * stats['count'] + (reading - old_mean) * (reading - stats['mean'])) / (stats['count'] + 1))
                stats['count'] += 1
                
                z_score = abs(reading - stats['mean']) / (stats['std'] + 1e-8)
                anomalies[param] = z_score > self.anomaly_threshold
        
        return anomalies
    
    def detect_health_anomalies(self, health_data: Dict[str, float]) -> Dict[str, bool]:
        """
        Detect anomalies in health data
        """
        anomalies = {}
        
        for param, reading in health_data.items():
            if param in self.health_thresholds:
                min_val, max_val = self.health_thresholds[param]
                anomalies[param] = not (min_val <= reading <= max_val)
            else:
                # Use statistical method for unknown parameters
                if param not in self.baseline_stats:
                    self.baseline_stats[param] = {
                        'mean': reading,
                        'std': 0.1,
                        'count': 1
                    }
                    continue
                
                stats = self.baseline_stats[param]
                old_mean = stats['mean']
                stats['mean'] = (stats['mean'] * stats['count'] + reading) / (stats['count'] + 1)
                stats['std'] = np.sqrt(((stats['std']**2) * stats['count'] + (reading - old_mean) * (reading - stats['mean'])) / (stats['count'] + 1))
                stats['count'] += 1
                
                z_score = abs(reading - stats['mean']) / (stats['std'] + 1e-8)
                anomalies[param] = z_score > self.anomaly_threshold
        
        return anomalies

class IoTActionGenerator:
    """
    Generate IoT-specific actions based on RAG responses
    """
    def __init__(self):
        pass
    
    def generate_actions(self, response: str, sensor_data: Dict[str, float]) -> List[str]:
        """
        Generate actions for smart home IoT devices
        """
        actions = []
        
        # Parse response for actionable items
        response_lower = response.lower()
        
        if 'turn on lights' in response_lower or 'illuminate' in response_lower:
            actions.append('{"device": "lights", "action": "turn_on", "location": "main_room"}')
        
        if 'adjust temperature' in response_lower or 'climate control' in response_lower:
            target_temp = self._extract_temperature(response)
            if target_temp:
                actions.append(f'{{"device": "thermostat", "action": "set_temperature", "value": {target_temp}}}')
        
        if 'lock doors' in response_lower:
            actions.append('{"device": "door_lock", "action": "lock", "location": "front_door"}')
        
        if 'security' in response_lower or 'surveillance' in response_lower:
            actions.append('{"device": "security_cameras", "action": "start_recording"}')
        
        return actions
    
    def generate_maintenance_actions(self, response: str, equipment_data: Dict[str, Any]) -> List[str]:
        """
        Generate maintenance actions for industrial IoT
        """
        actions = []
        
        response_lower = response.lower()
        
        if 'inspect' in response_lower or 'check' in response_lower:
            actions.append('{"equipment": "motor", "action": "visual_inspection", "priority": "high"}')
        
        if 'calibrate' in response_lower:
            actions.append('{"equipment": "sensors", "action": "recalibrate", "priority": "medium"}')
        
        if 'replace' in response_lower or 'repair' in response_lower:
            actions.append('{"equipment": "filter", "action": "replace", "priority": "critical"}')
        
        if 'schedule maintenance' in response_lower:
            actions.append('{"action": "create_work_order", "priority": "medium", "due_date": "next_week"}')
        
        return actions
    
    def generate_health_recommendations(self, response: str, health_data: Dict[str, Any]) -> List[str]:
        """
        Generate health recommendations for wearable devices
        """
        recommendations = []
        
        response_lower = response.lower()
        
        if 'rest' in response_lower or 'sleep' in response_lower:
            recommendations.append('{"action": "rest", "duration": "30_minutes", "type": "active_recovery"}')
        
        if 'exercise' in response_lower or 'activity' in response_lower:
            recommendations.append('{"action": "physical_activity", "type": "cardio", "duration": "20_minutes"}')
        
        if 'hydration' in response_lower or 'water' in response_lower:
            recommendations.append('{"action": "hydrate", "amount_ml": 250, "reminder": true}')
        
        if 'medical' in response_lower or 'doctor' in response_lower:
            recommendations.append('{"action": "consult_professional", "urgency": "medium", "notes": "See response for details"}')
        
        return recommendations
    
    def _extract_temperature(self, response: str) -> Optional[float]:
        """
        Extract temperature value from response
        """
        import re
        # Look for temperature patterns like "22 degrees" or "72°F"
        temp_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:degrees|°|deg)?F?', response)
        if temp_match:
            return float(temp_match.group(1))
        return None

class ContextAwarenessEngine:
    """
    Provide context awareness for IoT RAG system
    """
    def __init__(self):
        self.location_context = {}
        self.time_context = {}
        self.device_context = {}
    
    def enrich_context(self, query: str, device_info: Dict[str, Any]) -> str:
        """
        Enrich query with contextual information
        """
        enriched_query = query
        
        # Add location context
        if 'location' in device_info:
            enriched_query += f" [Location: {device_info['location']}]"
        
        # Add time context
        current_time = datetime.now()
        enriched_query += f" [Time: {current_time.strftime('%H:%M %A')}]"
        
        # Add device context
        if 'device_type' in device_info:
            enriched_query += f" [Device: {device_info['device_type']}]"
        
        return enriched_query
```

## 4. Performance and Evaluation

### 4.1 Edge IoT-Specific Evaluation Metrics
```python
class EdgeIoTEvaluationFramework:
    """
    Evaluation framework for edge IoT RAG systems
    """
    def __init__(self, device_caps: DeviceCapabilities):
        self.device_caps = device_caps
        self.metrics = [
            'energy_efficiency',
            'response_time',
            'accuracy',
            'bandwidth_usage',
            'privacy_preservation',
            'connectivity_resilience'
        ]
    
    def evaluate_system(self, system: EdgeRAGOrchestrator, 
                       test_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate the edge IoT RAG system
        """
        results = {
            'queries': [],
            'responses': [],
            'performance_metrics': {
                'energy_efficiency': [],  # mW per query
                'response_time': [],      # ms
                'accuracy': [],           # 0-1 scale
                'bandwidth_usage': [],    # bytes per query
                'cache_hit_rate': []      # 0-1 scale
            },
            'iot_specific_metrics': {
                'anomaly_detection_accuracy': [],
                'action_generation_success': [],
                'context_awareness_score': []
            }
        }
        
        total_energy = 0
        total_time = 0
        total_queries = len(test_queries)
        
        for query_data in test_queries:
            iot_data = query_data.get('iot_data', {})
            query = query_data.get('query', '')
            
            start_time = time.time()
            start_energy = system.energy_monitor.get_energy_consumption()
            
            # Process query
            response_data = system.process_iot_query(iot_data, query)
            
            end_time = time.time()
            end_energy = system.energy_monitor.get_energy_consumption()
            
            # Calculate metrics for this query
            query_time = (end_time - start_time) * 1000  # Convert to ms
            query_energy = end_energy - start_energy
            
            results['queries'].append(query_data)
            results['responses'].append(response_data)
            
            # Add to metrics
            results['performance_metrics']['energy_efficiency'].append(query_energy)
            results['performance_metrics']['response_time'].append(query_time)
            results['performance_metrics']['cache_hit_rate'].append(1.0 if response_data['cache_hit'] else 0.0)
            
            # Calculate accuracy if expected output provided
            expected_output = query_data.get('expected_output')
            if expected_output:
                accuracy = self._calculate_accuracy(response_data['response'], expected_output)
                results['performance_metrics']['accuracy'].append(accuracy)
            
            # Calculate bandwidth usage (simplified)
            # In practice, this would measure actual network usage
            bandwidth_usage = len(query.encode()) + len(response_data['response'].encode())
            results['performance_metrics']['bandwidth_usage'].append(bandwidth_usage)
            
            # IoT-specific metrics
            if 'anomalies_detected' in response_data:
                anomaly_accuracy = self._calculate_anomaly_accuracy(
                    response_data['anomalies_detected'],
                    query_data.get('expected_anomalies', {})
                )
                results['iot_specific_metrics']['anomaly_detection_accuracy'].append(anomaly_accuracy)
            
            # Update totals
            total_energy += query_energy
            total_time += query_time
        
        # Calculate aggregate metrics
        results['aggregate_metrics'] = {
            'avg_energy_efficiency': np.mean(results['performance_metrics']['energy_efficiency']),
            'avg_response_time': np.mean(results['performance_metrics']['response_time']),
            'avg_accuracy': np.mean(results['performance_metrics']['accuracy']) if results['performance_metrics']['accuracy'] else 0,
            'avg_bandwidth_usage': np.mean(results['performance_metrics']['bandwidth_usage']),
            'avg_cache_hit_rate': np.mean(results['performance_metrics']['cache_hit_rate']),
            'total_energy_consumed': total_energy,
            'total_processing_time': total_time,
            'queries_per_second': total_queries / (total_time / 1000) if total_time > 0 else 0,
            'energy_per_second': total_energy / (total_time / 1000) if total_time > 0 else 0
        }
        
        return results
    
    def _calculate_accuracy(self, generated: str, expected: str) -> float:
        """
        Calculate accuracy between generated and expected responses
        """
        # Simple character-level accuracy
        min_len = min(len(generated), len(expected))
        if min_len == 0:
            return 1.0 if generated == expected else 0.0
        
        matches = sum(1 for a, b in zip(generated, expected) if a == b)
        return matches / max(len(generated), len(expected))
    
    def _calculate_anomaly_accuracy(self, detected: Dict[str, bool], expected: Dict[str, bool]) -> float:
        """
        Calculate accuracy of anomaly detection
        """
        if not expected:
            return 0.5  # Unknown accuracy if no expected values
        
        correct_detections = 0
        total_comparisons = 0
        
        for param, expected_anomaly in expected.items():
            if param in detected:
                total_comparisons += 1
                if detected[param] == expected_anomaly:
                    correct_detections += 1
        
        return correct_detections / total_comparisons if total_comparisons > 0 else 0.0
```

## 5. Deployment Architecture

### 5.1 Edge Device Deployment
```yaml
# docker-compose.yml for edge IoT RAG system
version: '3.8'

services:
  # Edge RAG service for resource-constrained devices
  edge-rag-service:
    build: 
      context: .
      dockerfile: Dockerfile.edge
    image: edge-rag:tiny
    container_name: edge-rag-${DEVICE_ID}
    ports:
      - "${EDGE_PORT}:8000"
    environment:
      - DEVICE_ID=${DEVICE_ID}
      - MODEL_SIZE=tiny
      - POWER_MODE=${POWER_MODE:-balanced}
      - MAX_MEMORY_MB=${MAX_MEMORY:-256}
      - STORAGE_PATH=/data
    volumes:
      - edge_data_${DEVICE_ID}:/data
      - ./models:/models:ro
    # Resource constraints for edge devices
    deploy:
      resources:
        limits:
          memory: ${MAX_MEMORY:-256}M
          cpus: '0.5'
    restart: unless-stopped

  # Lightweight vector database for edge
  edge-vector-db:
    image: faiss-server:light
    environment:
      - INDEX_DIMENSION=128
      - MAX_MEMORY=128M
    volumes:
      - edge_vector_data_${DEVICE_ID}:/indexes
    restart: unless-stopped

  # IoT data preprocessing service
  iot-processor:
    build:
      context: .
      dockerfile: Dockerfile.iot-processor
    environment:
      - DEVICE_CAPABILITIES_FILE=/config/device_caps.json
      - COMPRESSION_LEVEL=9
    volumes:
      - ./config:/config:ro
    restart: unless-stopped

  # Connectivity manager for sync
  connectivity-manager:
    image: connectivity-manager:edge
    environment:
      - CLOUD_ENDPOINT=${CLOUD_ENDPOINT}
      - SYNC_INTERVAL=300  # 5 minutes
      - LOCAL_ONLY_MODE=${LOCAL_ONLY:-false}
    restart: unless-stopped

volumes:
  edge_data_${DEVICE_ID}:
  edge_vector_data_${DEVICE_ID}:

# Example device_caps.json:
# {
#   "cpu_cores": 4,
#   "memory_mb": 512,
#   "storage_mb": 1024,
#   "gpu_available": false,
#   "gpu_memory_mb": 0,
#   "power_source": "battery",
#   "thermal_limit": 70
# }
```

## 6. Security and Privacy

### 6.1 Edge IoT Security Measures
```python
class EdgeIoTSecurityManager:
    """
    Security manager for edge IoT RAG system
    """
    def __init__(self, device_caps: DeviceCapabilities):
        self.device_caps = device_caps
        self.encryption_manager = EdgeEncryptionManager()
        self.access_control = EdgeAccessControl()
        self.privacy_preserver = PrivacyPreserver()
        self.audit_logger = EdgeAuditLogger()
    
    def secure_process_query(self, iot_data: Dict[str, Any], query: str, 
                           user_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Securely process a query with privacy preservation
        """
        # Verify user permissions
        if not self.access_control.verify_permission(user_context, 'query'):
            raise PermissionError("User not authorized for queries")
        
        # Log the request
        request_id = self.audit_logger.log_request(user_context, query, iot_data)
        
        try:
            # Sanitize and anonymize IoT data
            sanitized_iot_data = self.privacy_preserver.anonymize_iot_data(iot_data)
            
            # Process query through secure pipeline
            result = self._secure_query_processing(sanitized_iot_data, query)
            
            # Log successful processing
            self.audit_logger.log_success(request_id, result)
            
            return result
            
        except Exception as e:
            # Log failure
            self.audit_logger.log_failure(request_id, str(e))
            raise e
    
    def _secure_query_processing(self, iot_data: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Process query through secure pipeline
        """
        # In practice, this would call the actual edge RAG system
        # For this example, we'll simulate the processing
        return {
            'response': f"Secure response to query: '{query[:50]}...' processed on device",
            'processing_time_ms': 150,  # Simulated time
            'privacy_preserved': True,
            'data_anonymized': True
        }

class EdgeEncryptionManager:
    """
    Encryption manager for edge devices
    """
    def __init__(self):
        import secrets
        self.key = secrets.token_bytes(32)  # 256-bit key
    
    def encrypt_data(self, data: str) -> bytes:
        """
        Encrypt data using AES-GCM (simplified for this example)
        """
        # In practice, use proper encryption like AES-GCM
        # For this example, we'll use a simple XOR cipher
        data_bytes = data.encode('utf-8')
        encrypted = bytearray()
        
        for i, byte in enumerate(data_bytes):
            encrypted.append(byte ^ self.key[i % len(self.key)])
        
        return bytes(encrypted)
    
    def decrypt_data(self, encrypted_data: bytes) -> str:
        """
        Decrypt data
        """
        # Decrypt using the same XOR operation
        decrypted = bytearray()
        
        for i, byte in enumerate(encrypted_data):
            decrypted.append(byte ^ self.key[i % len(self.key)])
        
        return decrypted.decode('utf-8')

class PrivacyPreserver:
    """
    Preserves privacy in IoT data
    """
    def __init__(self):
        self.pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{3}-\d{3}-\d{4}\b',  # Phone
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        ]
    
    def anonymize_iot_data(self, iot_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anonymize IoT data to preserve privacy
        """
        anonymized_data = iot_data.copy()
        
        if 'location' in anonymized_data:
            # Generalize location to broader area
            anonymized_data['location'] = self._generalize_location(anonymized_data['location'])
        
        if 'description' in anonymized_data:
            # Remove PII from descriptions
            anonymized_data['description'] = self._remove_pii(anonymized_data['description'])
        
        return anonymized_data
    
    def _generalize_location(self, location: str) -> str:
        """
        Generalize location to preserve privacy
        """
        # For example, convert specific address to neighborhood or city
        # This is a simplified example
        return location.split(',')[0]  # Just keep the first part (e.g., neighborhood)
    
    def _remove_pii(self, text: str) -> str:
        """
        Remove personally identifiable information
        """
        import re
        
        for pattern in self.pii_patterns:
            text = re.sub(pattern, '[REDACTED]', text)
        
        return text

class EdgeAccessControl:
    """
    Access control for edge devices
    """
    def __init__(self):
        self.user_permissions = {}
        self.device_whitelist = set()
        self.rate_limits = {}
    
    def verify_permission(self, user_context: Dict[str, Any], operation: str) -> bool:
        """
        Verify if user has permission for the operation
        """
        user_id = user_context.get('user_id')
        device_id = user_context.get('device_id')
        
        # Check if device is whitelisted
        if device_id and device_id not in self.device_whitelist:
            return False
        
        # Check user permissions
        if user_id and operation in self.user_permissions.get(user_id, []):
            # Check rate limits
            return self._check_rate_limit(user_id, operation)
        
        return False
    
    def _check_rate_limit(self, user_id: str, operation: str) -> bool:
        """
        Check if user has exceeded rate limits
        """
        current_time = time.time()
        key = f"{user_id}:{operation}"
        
        if key not in self.rate_limits:
            self.rate_limits[key] = {'count': 1, 'window_start': current_time}
            return True
        
        # Reset window if needed (1 minute window)
        if current_time - self.rate_limits[key]['window_start'] > 60:
            self.rate_limits[key] = {'count': 1, 'window_start': current_time}
            return True
        
        # Check if limit exceeded
        if self.rate_limits[key]['count'] >= 10:  # 10 requests per minute
            return False
        
        # Increment count
        self.rate_limits[key]['count'] += 1
        return True

class EdgeAuditLogger:
    """
    Audit logging for edge IoT systems
    """
    def __init__(self):
        import json
        self.log_file = "edge_audit.log"
    
    def log_request(self, user_context: Dict[str, Any], query: str, iot_data: Dict[str, Any]) -> str:
        """
        Log a query request
        """
        import uuid
        request_id = str(uuid.uuid4())
        
        log_entry = {
            'timestamp': time.time(),
            'request_id': request_id,
            'user_id': user_context.get('user_id'),
            'device_id': user_context.get('device_id'),
            'query_preview': query[:100] + "..." if len(query) > 100 else query,
            'iot_data_types': list(iot_data.keys()) if isinstance(iot_data, dict) else [],
            'event_type': 'query_request'
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        return request_id
    
    def log_success(self, request_id: str, response: Dict[str, Any]):
        """
        Log successful query processing
        """
        log_entry = {
            'timestamp': time.time(),
            'request_id': request_id,
            'event_type': 'query_success',
            'response_length': len(str(response.get('response', ''))),
            'processing_time_ms': response.get('query_time_ms', 0),
            'energy_consumed': response.get('energy_consumed', 0)
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_failure(self, request_id: str, error_message: str):
        """
        Log query processing failure
        """
        log_entry = {
            'timestamp': time.time(),
            'request_id': request_id,
            'event_type': 'query_failure',
            'error': error_message
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
```

## 7. Performance Benchmarks

### 7.1 Expected Performance Metrics
| Metric | Target | Device Class | Notes |
|--------|--------|--------------|-------|
| Energy Efficiency | < 0.1 J/query | Constrained (512MB RAM) | For battery-powered devices |
| Response Time | < 200ms | Moderate (1GB RAM) | For real-time applications |
| Accuracy | > 80% | Capable (2GB+ RAM) | Compared to cloud-based systems |
| Bandwidth Usage | < 1KB/query | All classes | Minimized for limited connectivity |
| Privacy Preservation | 100% | All classes | Data stays on device |
| Cache Hit Rate | > 60% | All classes | Reduces processing needs |

## 8. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-4)
- Implement basic edge RAG core for resource-constrained devices
- Develop IoT data preprocessing pipeline
- Create optimized vector database for edge
- Implement power management features

### Phase 2: IoT Integration (Weeks 5-8)
- Add IoT-specific RAG extensions
- Implement anomaly detection for sensor data
- Develop action generation for IoT devices
- Add context awareness engine

### Phase 3: Optimization (Weeks 9-12)
- Optimize for different device classes
- Implement advanced compression techniques
- Add connectivity management features
- Performance tuning and testing

### Phase 4: Production (Weeks 13-16)
- Deploy to actual IoT devices
- Implement monitoring and alerting
- Security and privacy validation
- Documentation and deployment guides

## 9. Conclusion

This Edge AI RAG system for IoT applications demonstrates how to bring RAG processing closer to data sources while addressing the unique constraints of IoT devices. The system prioritizes energy efficiency, low latency, and privacy preservation while maintaining reasonable accuracy for IoT applications.

The architecture is designed to be adaptable to different classes of IoT devices, from highly constrained sensors to more capable edge gateways. By implementing model optimization, efficient retrieval mechanisms, and adaptive processing strategies, the system achieves the goal of enabling intelligent IoT applications without relying on cloud connectivity.

While challenges remain in balancing performance with resource constraints, the fundamental approach of edge-based RAG processing shows great promise for creating responsive, private, and efficient IoT applications that can operate independently of cloud infrastructure.