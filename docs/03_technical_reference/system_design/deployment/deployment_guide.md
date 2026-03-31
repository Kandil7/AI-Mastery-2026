# Guide: Deployment

This guide covers how to deploy the `AI-Mastery-2026` application using Docker and Docker Compose. This is the recommended method for running the application in a production-like environment, as it ensures consistency and encapsulates all dependencies.

## 1. Prerequisites

*   **Docker and Docker Compose:** You must have Docker and Docker Compose installed. The easiest way to get them is by installing [Docker Desktop](https://www.docker.com/products/docker-desktop/).

## 2. Understanding the Docker Setup

The deployment is orchestrated via two key files:

*   **`Dockerfile`**: This is a blueprint for building a single container image for the main application. It handles:
    *   Starting from a slim Python base image.
    *   Installing system and Python dependencies.
    *   Copying the project code into the container.
    *   Creating a non-root user (`appuser`) for enhanced security.
    *   Exposing port `8000` for the API.
    *   Setting up a `HEALTHCHECK` to ensure the API is running correctly.
    *   Defining the default command to start the `uvicorn` server.

*   **`docker-compose.yml`**: This file defines and orchestrates a multi-service application. It sets up a complete, interconnected environment consisting of:
    *   **`api`**: The main FastAPI application, built from the `Dockerfile`.
    *   **`jupyter`**: A Jupyter Lab instance for running notebooks, using the same base image.
    *   **`redis`**: A Redis container for caching.
    *   **`postgres`**: A PostgreSQL database, potentially for storing metadata or experiment tracking.
    *   **`prometheus`**: A Prometheus server configured to scrape metrics from the API's `/metrics` endpoint.
    *   **`grafana`**: A Grafana instance for visualizing the metrics collected by Prometheus.

## 3. Running the Application with Docker Compose

The `Makefile` provides convenient commands to manage the Docker Compose environment.

### Step 3.1: Build and Start All Services

To build the necessary Docker images and start all the services defined in `docker-compose.yml`, run:

```bash
make docker-run
```

This command will:
1.  Build the image for the `api` and `jupyter` services using the `Dockerfile`.
2.  Pull the required images for `redis`, `postgres`, `prometheus`, and `grafana` from Docker Hub.
3.  Start all containers in detached mode (`-d`), meaning they will run in the background.

### Step 3.2: Accessing the Services

Once the containers are running, you can access the different parts of the application at their default ports:

*   **FastAPI Application:** `http://localhost:8000`
*   **API Docs (Swagger UI):** `http://localhost:8000/docs`
*   **Jupyter Lab:** `http://localhost:8888`
*   **Prometheus UI:** `http://localhost:9090`
*   **Grafana Dashboard:** `http://localhost:3000` (Login with `admin`/`admin`)

### Step 3.3: Managing the Services

*   **Viewing Logs:**
    To view the logs from all running services in real-time, use:
    ```bash
    make docker-logs
    ```

*   **Stopping Services:**
    To stop all the running containers:
    ```bash
    make docker-stop
    ```
    This command gracefully stops the containers but does not remove them or their data volumes.

## 4. Cleaning Up

If you want to completely remove the containers, their associated networks, and data volumes, you can use the cleanup targets in the `Makefile`.

*   **Clean Docker Resources:**
    This command will stop and remove the containers, networks, and anonymous volumes.
    ```bash
# This is an alias for 'docker-compose down -v'
make docker-clean
    ```

*   **Full Cleanup:**
    To remove everything, including build artifacts and Python cache files from your host machine, run:
    ```bash
    make clean-all
    ```

## 5. Configuration via Environment Variables

The `docker-compose.yml` file is configured to allow overriding default ports via environment variables. For example, to run the API on port `8001` and Grafana on port `3001`, you can execute the command like this:

```bash
API_PORT=8001 GRAFANA_PORT=3001 make docker-run
```

This provides flexibility when deploying in an environment where default ports may already be in use.
