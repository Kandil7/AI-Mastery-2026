# AI-Mastery-2026: Project Completion Plan

This document outlines the tasks required to transform the `AI-Mastery-2026` repository from an educational toolkit into a fully functional, end-to-end AI platform.

---

### Phase 1: Foundational Backend and MLOps Integration

- [ ] **Task 1.1: Integrate a Real Pre-trained Model**
    - [ ] Choose and save a pre-trained `scikit-learn` model to the `./models` directory.
    - [ ] Modify `src/production/api.py` to load the model into the `ModelCache` on startup.
    - [ ] Update the `/predict` endpoint to use the loaded model for inference instead of dummy logic.
    - [ ] Verify the endpoint returns real predictions.

- [ ] **Task 1.2: Set Up PostgreSQL Database**
    - [ ] Create a Python script (`scripts/setup_database.py`) to initialize the DB schema (e.g., for logging predictions or experiment tracking).
    - [ ] Modify the `docker-compose.yml` to run this script automatically.
    - [ ] Add a new endpoint to the API to log prediction results to the database.

- [ ] **Task 1.3: Implement CI/CD Pipeline**
    - [ ] Create a `.github/workflows/ci.yml` file.
    - [ ] Define a GitHub Actions workflow that triggers on pull requests.
    - [ ] The workflow should:
        - [ ] Install dependencies (`make install`).
        - [ ] Run linters (`make lint`).
        - [ ] Run tests (`make test`).

- [ ] **Task 1.4: Configure Monitoring Stack**
    - [ ] Create a `config/prometheus.yml` to scrape the API's `/metrics` endpoint.
    - [ ] Create a `config/grafana/provisioning` configuration to automatically set up Prometheus as a data source.
    - [ ] Create a default Grafana dashboard JSON file to visualize key metrics (latency, requests, error rate).

---

### Phase 2: Full-Stack Application and End-to-End RAG

- [ ] **Task 2.1: Integrate Open-Source LLM and Embedding Models**
    - [ ] Choose and download a small, efficient embedding model (e.g., from `sentence-transformers`).
    - [ ] Choose and download a small, efficient LLM (e.g., a quantized version of Llama or Mistral).
    - [ ] Modify `src/llm/rag.py` to use these real models instead of the dummy implementations.

- [ ] **Task 2.2: Build Data Ingestion Pipeline**
    - [ ] Create a script (`scripts/ingest_data.py`) that reads documents from a directory.
    - [ ] The script should use the `TextChunker` and `EmbeddingModel` from the RAG pipeline.
    - [ ] It should then add the embedded chunks to the `HNSW` vector index and save the index to disk.

- [ ] **Task 2.3: Create a Web Front-End**
    - [ ] Create a new file `app/main.py`.
    - [ ] Use `Streamlit` to build a simple UI with a title, a text input for questions, and an area to display the response.

- [ ] **Task 2.4: Connect Front-End to Backend**
    - [ ] Add `streamlit` and `requests` to `requirements.txt`.
    - [ ] In `app/main.py`, make an HTTP request to the backend's `/chat/completions` endpoint.
    - [ ] Implement logic to handle the streaming response and display the tokens as they arrive.
    - [ ] Add a new service to `docker-compose.yml` for the Streamlit app.

---

### Phase 3: Enhancing Core AI Capabilities

- [ ] **Task 3.1: Implement Support Vector Machine (SVM)**
    - [ ] Add a `SVMScratch` class to `src/ml/classical.py`.
    - [ ] Implement the training logic using the Hinge loss and a gradient-based optimizer.
    - [ ] Add a corresponding test file in `tests/`.

- [ ] **Task 3.2: Implement Advanced Deep Learning Layers**
    - [ ] Add an `LSTM` layer to `src/ml/deep_learning.py`, including the forward and backward passes for all gates.
    - [ ] Add a `Conv2D` layer that handles multiple channels and uses the `conv2d_single` primitive.

- [ ] **Task 3.3: Optimize Numerical Code**
    - [ ] Identify a computationally intensive, from-scratch function (e.g., `matrix_multiply`).
    - [ ] Add `numba` to `requirements.txt`.
    - [ ] Use the `@numba.jit` decorator to accelerate the function.
    - [ ] Add a benchmark in a notebook to show the performance improvement.

---

### Phase 4: Finalization and Polish

- [ ] **Task 4.1: Create End-to-End Example Notebooks**
    - [ ] Create a new notebook in `research/` that demonstrates a full MLOps cycle.
    - [ ] The notebook should show how to:
        - [ ] Train a model from `src/ml`.
        - [ ] Save the model.
        - [ ] Run the Docker-based API to serve it.
        - [ ] Send a request to the API and get a prediction.

- [ ] **Task 4.2: Write Capstone Project Guide**
    - [ ] Create `docs/guide/10_capstone_project.md`.
    - [ ] Write a tutorial that guides the user through building a novel application (e.g., a "GitHub Issue Classifier") using the repository's components.

- [ ] **Task 4.3: Final Documentation Review**
    - [ ] Read through all generated documentation in `docs/guide/`.
    - [ ] Check for clarity, consistency, and correctness.
    - [ ] Add any missing details or examples.
