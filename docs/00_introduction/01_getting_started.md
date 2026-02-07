# Guide: Getting Started

This guide provides detailed steps to set up, install, and verify the `AI-Mastery-2026` project on your local machine.

## 1. Prerequisites

*   **Python:** You need Python `3.10` or newer. You can download it from [python.org](https://www.python.org/downloads/).
*   **Git:** The project is managed with Git. Instructions for installing Git are available [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).
*   **Docker (Recommended):** For easy deployment and environment management, Docker is recommended. Get Docker Desktop from the [official website](https://www.docker.com/products/docker-desktop/).

## 2. Installation

Follow these steps to clone the repository and set up the environment.

### Step 2.1: Clone the Repository

Open your terminal and clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/AI-Mastery-2026.git
cd AI-Mastery-2026
```
*(Replace `yourusername` with the actual repository owner's username or organization)*

### Step 2.2: Create a Virtual Environment

It is a best practice to use a virtual environment to manage project-specific dependencies.

```bash
# This command creates a new virtual environment in the `.venv` directory
python -m venv .venv
```

### Step 2.3: Activate the Virtual Environment

Before installing dependencies, you must activate the environment.

*   **On Windows:**
    ```bash
    .venv\Scripts\activate
    ```

*   **On macOS and Linux:**
    ```bash
    source .venv/bin/activate
    ```

Once activated, your terminal prompt will typically be prefixed with `(.venv)`.

### Step 2.4: Install Dependencies

The project's dependencies are managed through the `Makefile`, which simplifies the installation process.

```bash
# This command reads requirements.txt and installs all necessary packages
make install
```
This command ensures that all packages for core functionality, API development, and machine learning are installed into your virtual environment.

## 3. Verification

After installation, it's crucial to verify that everything is set up correctly by running the test suite.

### Run All Tests

The project uses `pytest` for testing. You can run all tests using the Makefile:

```bash
make test
```

You should see a series of passing tests. This confirms that the core logic is working as expected and the environment is configured correctly.

## 4. Running the Application

The primary runnable component is the FastAPI server.

### Run in Development Mode

For development, you can run the server with auto-reload enabled, which automatically restarts the server when you make code changes.

```bash
make run
```
The API will be accessible at `http://localhost:8000`. You can access the auto-generated documentation at `http://localhost:8000/docs`.

## 5. Using Docker (Alternative Setup)

If you have Docker installed, you can build and run the entire environment in containers. This is the recommended approach for ensuring consistency.

### Step 5.1: Build and Run Containers

The `docker-compose.yml` file is configured to set up all services, including the FastAPI server and a Jupyter Lab instance.

```bash
# This single command builds the images and starts the containers in detached mode
make docker-run
```

### Step 5.2: Access Services

Once the containers are running, you can access the services:
*   **API Server:** [http://localhost:8000](http://localhost:8000)
*   **Jupyter Lab:** [http://localhost:8888](http://localhost:8888)

### Step 5.3: Stop Containers

To stop all running services:
```bash
make docker-stop
```
---

You have now successfully set up the `AI-Mastery-2026` project. You can proceed to explore the [Core Concepts](./02_core_concepts.md) or dive into the module documentation.
