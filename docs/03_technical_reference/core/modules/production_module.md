# Guide: Module `src/production`

The `src/production` module addresses the critical final step in the machine learning lifecycle: deploying and maintaining models in a real-world environment. This module provides from-scratch and production-ready components for model serving, monitoring, and vector search.

## 1. `api.py`

This file contains a production-grade API server built with **FastAPI**. It is designed to serve machine learning models efficiently and reliably.

### Key Features

*   **FastAPI Framework**: Built on FastAPI for high performance, asynchronous request handling, and automatic OpenAPI (Swagger) documentation.
*   **Pydantic Validation**: All incoming requests (`/predict`, `/chat/completions`) are validated using Pydantic models, ensuring data integrity and providing clear error messages.
*   **Model Management**:
    *   A `ModelCache` singleton is used to load models into memory once at startup, avoiding the latency of loading a model on every request.
    *   The API provides endpoints (`/models`) to list what's currently loaded.
*   **Asynchronous Endpoints**: Uses `async def` for all routes, allowing the server to handle many concurrent connections efficiently, which is crucial for I/O-bound tasks like waiting for model predictions.
*   **LLM Streaming with SSE**:
    *   The `/chat/completions` endpoint supports `stream=True`.
    *   It uses **Server-Sent Events (SSE)** to stream token-by-token responses from a language model, providing a real-time, ChatGPT-like user experience.
*   **Production-Ready Endpoints**:
    *   `/health`: A simple health check for load balancers.
    *   `/ready`: A readiness probe that checks if the main model is loaded, telling a container orchestrator (like Kubernetes) when the app is ready to receive traffic.
    *   `/metrics`: An endpoint for exposing key performance indicators (latency, request count, error rate) in a Prometheus-compatible format.

## 2. `monitoring.py`

This file provides tools to monitor models in production, which is essential for detecting when a model's performance is degrading.

### Key Components

*   **Statistical Drift Detection**:
    *   `ks_test`: Implements the **Kolmogorov-Smirnov (KS) test** to determine if the distribution of a feature in production has shifted significantly from its distribution in the training data.
    *   `psi`: Implements the **Population Stability Index (PSI)**, another common technique for measuring distribution shift by binning data.
    *   `chi_square_test`: For detecting drift in categorical features.
*   **`DriftDetector` Class**:
    *   A wrapper that allows you to set a "reference" dataset (your training data) and then compare incoming production data against it.
    *   It runs the chosen statistical test (`ks` or `psi`) on each feature and returns a list of `DriftResult` objects, making it easy to see which features have drifted.
*   **`PerformanceMonitor` Class**:
    *   Tracks model performance over a sliding window of recent predictions.
    *   If ground truth labels are available, it computes standard metrics like `accuracy`, `precision`, `recall`, `f1_score`, `MSE`, and `MAE`.
    *   It also tracks prediction `latency` (p50, p95, p99) and overall error rates.
*   **`AlertManager` Class**:
    *   A simple, extensible system for sending alerts when monitoring thresholds are breached (e.g., if drift is detected or accuracy drops). It's designed to be integrated with tools like Slack or PagerDuty.

## 3. `vector_db.py`

This file contains from-scratch implementations of vector database indexes, which are fundamental to modern semantic search and RAG systems.

### Key Implementations

*   **`BruteForceIndex`**:
    *   Performs an exact nearest neighbor search by calculating the distance from the query vector to every other vector in the index.
    *   While perfectly accurate, it is slow (`O(n)`) and serves as a baseline for comparison.

*   **`HNSW` (Hierarchical Navigable Small World)**:
    *   A from-scratch implementation of the state-of-the-art **Approximate Nearest Neighbor (ANN)** search algorithm.
    *   It builds a multi-layer graph structure that allows for extremely fast searching (`O(log n)`) with high recall. This is the same type of algorithm used by most commercial vector databases.
    *   The implementation details the core logic of inserting nodes at random levels and greedily traversing the graph from a high-level entry point down to the dense base layer.

*   **`LSH` (Locality-Sensitive Hashing)**:
    *   An alternative ANN algorithm that uses random projections to "hash" similar vectors into the same buckets.
    *   It's particularly effective for cosine similarity and provides another perspective on how to solve the nearest neighbor problem without brute-force search.

*   **`VectorIndex` Factory**:
    *   A simple factory class that provides a unified interface for creating any of the implemented index types (`VectorIndex.create('hnsw', dim=...)`).

---

The `src/production` module demonstrates how to wrap AI/ML models in a robust, observable, and scalable service, completing the journey from mathematical theory to a deployable artifact.
