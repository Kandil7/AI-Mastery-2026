# Docker Fundamentals: A Complete Guide from First Principles

## Introduction: Why Do We Need Docker?

### The Problem: "It Works on My Machine"

Imagine you're a developer working on a Python application. You've spent weeks perfecting your code on your laptop, and it works flawlessly. You write beautiful code, install some packages, configure your database, and everything is great.

Now it's time to deploy to production. You send your code to the ops team, and... it doesn't work.

**Why?**
- Your laptop has Python 3.9, production has Python 3.7
- You have a Mac, production runs Linux
- You installed a package that requires system libraries not present on the server
- Your configuration files point to `/Users/yourname/data`, but production uses `/var/data`
- The database version differs slightly, causing compatibility issues

This is the infamous "it works on my machine" problem.

### The Traditional Solution (and why it's painful)

**Approach 1: Manual Configuration Documentation**
```
Developer writes a 50-page document:
1. Install Python 3.9.2 (not 3.9.1 or 3.9.3 - specifically 3.9.2)
2. Install these 47 system packages
3. Configure these 12 environment variables
4. Set file permissions exactly like this
5. Pray nothing changes
```

**Problems:**
- Time-consuming to write and follow
- Easy to miss one small detail
- Different environments (dev/staging/prod) drift over time
- Hard to reproduce exactly
- Doesn't scale to many machines

**Approach 2: Virtual Machines**
Create a complete virtual computer with everything installed.

**Problems:**
- Each VM is 10-20GB (includes entire operating system)
- Slow to start (minutes)
- Resource heavy (each VM needs RAM, CPU cores)
- Expensive to run many VMs

### The Docker Solution

**Think of Docker like a shipping container:**

In the 1950s, shipping was chaotic. Goods were loaded in sacks, barrels, boxes of different sizes. Dock workers had to manually handle everything, and it took weeks to load a ship.

Then came the shipping container - a standardized box that could hold anything. The container is the same size whether it holds toys, electronics, or food. Ships, trains, and trucks are all designed to handle these standard containers.

**Docker is the shipping container for software:**
- Package your application AND all its dependencies into a standardized "container"
- The container includes: code, runtime, system tools, libraries, settings
- Run the same container on any machine that supports Docker
- It works identically everywhere

**Benefits:**
1. **Consistency**: Dev, staging, and production run identical containers
2. **Isolation**: Your app can't interfere with other apps on the same machine
3. **Portability**: Run on laptop, server, or cloud without changes
4. **Efficiency**: Containers share the host OS kernel (unlike VMs)
5. **Scalability**: Start new containers in seconds

---

## Part 1: Understanding Docker Architecture

### The Docker Ecosystem

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Architecture                      │
│                                                             │
│  ┌──────────────┐                                          │
│  │ Docker CLI   │  ← You type commands here                │
│  │   (docker)   │                                          │
│  └──────┬───────┘                                          │
│         │                                                   │
│         │ REST API                                         │
│         ▼                                                   │
│  ┌──────────────┐                                          │
│  │ Docker       │  ← The brain that manages everything     │
│  │ Daemon       │                                          │
│  │  (dockerd)   │                                          │
│  └──────┬───────┘                                          │
│         │                                                   │
│    ┌────┴────┬────────┐                                    │
│    ▼         ▼        ▼                                    │
│ ┌─────┐  ┌──────┐ ┌────────┐                              │
│ │Images│  │Containers│ │Registry│                              │
│ └─────┘  └──────┘ └────────┘                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Docker Daemon (dockerd)**: The background service that manages Docker objects (images, containers, networks, storage).

**Docker CLI (docker)**: The command-line tool you use to talk to the daemon.

**Docker Registries**: Storage for Docker images (like Docker Hub, AWS ECR, GCR).

### Core Concepts

#### 1. Images (The Blueprint)

**Analogy**: Think of an image like a recipe + ingredients list for a cake.

An image is a read-only template that contains:
- Application code
- Runtime environment (Python, Node.js, etc.)
- System libraries
- Configuration files
- Environment variables

**Key characteristics:**
- Immutable (can't be changed once created)
- Layered (built in layers, like an onion)
- Shareable (can be pushed to registries)
- Versioned (tagged with versions like `v1.0`, `latest`)

**Viewing images:**
```bash
# List all images on your machine
docker images

# Output shows:
# REPOSITORY    TAG       IMAGE ID       CREATED        SIZE
# rag-engine    latest    abc123def456   2 hours ago    450MB
# python        3.9       123456789abc   5 days ago     885MB
```

#### 2. Containers (The Running Instance)

**Analogy**: If an image is a recipe, a container is the actual cake you baked.

A container is a runnable instance of an image. You can:
- Start it (bake the cake)
- Stop it (put cake in fridge)
- Delete it (eat the cake)
- Create many from the same image (bake multiple cakes)

**Key characteristics:**
- Isolated (thinks it has its own filesystem, network, etc.)
- Ephemeral (can be created and destroyed easily)
- Stateful (can save data in volumes)
- Lightweight (shares host OS kernel)

**Container lifecycle:**
```
Image → Create → Start → Run → Stop → Delete
            ↓
        Pause/Resume
            ↓
        Restart
```

**Working with containers:**
```bash
# Create and start a container
docker run -d --name my-app nginx

# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# Stop a container
docker stop my-app

# Start a stopped container
docker start my-app

# Delete a container
docker rm my-app
```

#### 3. Layers (How Images Are Built)

Docker images are built in layers. Each instruction in a Dockerfile creates a new layer.

**Analogy**: Think of layers like transparent sheets in an overhead projector. Each sheet adds something to the final image.

**How layers work:**
```
Layer 1: Base OS (Ubuntu 22.04)
    ↓
Layer 2: Install Python 3.9
    ↓
Layer 3: Install requirements.txt
    ↓
Layer 4: Copy application code
    ↓
Layer 5: Set environment variables
    ↓
Layer 6: Define startup command
```

**Benefits of layering:**
1. **Caching**: If Layer 1-3 don't change, Docker reuses them
2. **Sharing**: Multiple images can share base layers
3. **Efficiency**: Only changed layers need to be transferred

**Example:**
```bash
# Build an image
docker build -t my-app .

# See the layers
docker history my-app

# Output shows each layer with size
```

---

## Part 2: Docker in Practice

### Installing Docker

**Why you need Docker:**
- Develop applications in isolated environments
- Test applications in production-like settings
- Deploy applications consistently
- Learn modern DevOps practices

**Installation (Ubuntu/Debian):**
```bash
# Update package index
sudo apt-get update

# Install prerequisites
sudo apt-get install apt-transport-https ca-certificates curl gnupg lsb-release

# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Add Docker repository
echo \
  "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Verify installation
sudo docker run hello-world
```

**Installation (macOS):**
```bash
# Using Homebrew
brew install --cask docker

# Or download from https://docs.docker.com/desktop/install/mac-install/
```

**Installation (Windows):**
```powershell
# Using Chocolatey
choco install docker-desktop

# Or download from https://docs.docker.com/desktop/install/windows-install/
```

### Your First Container

Let's run a simple container to understand the basics.

**Step 1: Run a basic container**
```bash
# Run a container that prints "Hello, World!" and exits
docker run hello-world
```

**What happens:**
1. Docker checks if `hello-world` image exists locally
2. If not, it downloads from Docker Hub (registry)
3. Docker creates a new container from the image
4. Container runs and prints the message
5. Container exits

**Step 2: Run an interactive container**
```bash
# Run Ubuntu and get a shell inside it
docker run -it ubuntu:22.04 bash

# -i = interactive (keep STDIN open)
# -t = allocate a pseudo-TTY (terminal)
# ubuntu:22.04 = image name and tag
# bash = command to run inside container
```

**Inside the container, try:**
```bash
# See what OS this is
cat /etc/os-release

# Check current user
whoami

# Check current directory
pwd

# Create a file
echo "Hello from container" > /tmp/test.txt

# Read the file
cat /tmp/test.txt

# Exit the container
exit
```

**Key observations:**
- The container has its own filesystem (creating `/tmp/test.txt` doesn't affect your host)
- The container has its own process space (processes inside don't show on host)
- When you exit, the container stops (but still exists)

**Step 3: List containers**
```bash
# See running containers (should be empty now)
docker ps

# See all containers (including stopped)
docker ps -a

# You'll see something like:
# CONTAINER ID   IMAGE           COMMAND   CREATED         STATUS                     PORTS     NAMES
# 123456789abc   ubuntu:22.04    "bash"    2 minutes ago   Exited (0) 1 minute ago             stoic_einstein
```

**Step 4: Understanding container persistence**
```bash
# Start the stopped container again
docker start -ai stoic_einstein

# Try to find your file
cat /tmp/test.txt

# It's still there! Containers preserve their state between stops and starts

# Exit again
exit
```

**Step 5: Clean up**
```bash
# Remove the container
docker rm stoic_einstein

# Verify it's gone
docker ps -a
```

### Understanding Ports and Networking

**The Problem:**
Your application runs inside a container and listens on port 8000. How do you access it from your browser?

**Container networking basics:**
- Each container gets its own network namespace (like its own network stack)
- By default, container ports are isolated from the host
- You must explicitly "publish" ports to make them accessible

**Analogy**: Think of a container as a hotel room. The room has a phone (port 8000), but you can't call it from outside unless you know the hotel's main number and the room extension.

**Port mapping:**
```
Host Port 8080 → Container Port 80
```

**Practical example:**
```bash
# Run nginx web server, map host port 8080 to container port 80
docker run -d -p 8080:80 --name my-nginx nginx

# -d = detached mode (run in background)
# -p 8080:80 = map port 8080 on host to port 80 in container
# --name my-nginx = give the container a name
# nginx = the image to use
```

**Test it:**
```bash
# Check it's running
docker ps

# Access the web server
curl http://localhost:8080
# Or open http://localhost:8080 in your browser

# You'll see the nginx welcome page!
```

**Understanding port binding:**
```bash
# You can map multiple ports
docker run -p 8080:80 -p 8443:443 nginx

# Or let Docker choose a random host port
docker run -d -p 80 nginx

# Check which port was assigned
docker port <container_id>
```

**Clean up:**
```bash
# Stop the container
docker stop my-nginx

# Remove it
docker rm my-nginx
```

### Understanding Volumes and Persistence

**The Problem:**
By default, when a container is deleted, all its data is lost. How do we persist data?

**Analogy**: A container is like a temporary workspace. When you leave (delete the container), the cleaning crew throws everything away. Volumes are like permanent lockers - they persist even when you leave.

**Three types of storage:**

1. **Volumes** (Managed by Docker)
   - Stored in Docker's internal storage area
   - Easiest to back up and migrate
   - Can be shared between containers

2. **Bind Mounts** (Direct host filesystem access)
   - Maps a host directory to a container directory
   - Great for development (live code editing)
   - Direct performance (no Docker abstraction)

3. **tmpfs Mounts** (In-memory storage)
   - Stored in host's memory
   - Very fast, but lost when container stops
   - Good for sensitive data (never written to disk)

**Practical example with volumes:**
```bash
# Create a named volume
docker volume create my-data

# Run container with the volume
docker run -d \
  -v my-data:/data \
  --name data-container \
  ubuntu:22.04 \
  sleep infinity

# Write data to the volume
docker exec data-container bash -c "echo 'Important data' > /data/file.txt"

# Stop and remove the container
docker stop data-container
docker rm data-container

# The data is still in the volume!
docker volume inspect my-data

# Create new container with same volume
docker run -d \
  -v my-data:/data \
  --name new-container \
  ubuntu:22.04 \
  sleep infinity

# Verify data persists
docker exec new-container cat /data/file.txt
# Output: Important data
```

**Practical example with bind mounts (development):**
```bash
# Your project structure:
# /home/user/my-project/
#   ├── app.py
#   ├── requirements.txt
#   └── data/

# Run container with bind mount
docker run -d \
  -v /home/user/my-project:/app \
  -w /app \
  python:3.9 \
  python app.py

# Now any changes you make to /home/user/my-project on your host
# are immediately visible inside the container at /app
```

**Clean up:**
```bash
# Stop and remove containers
docker stop new-container
docker rm new-container

# Remove volume
docker volume rm my-data
```

---

## Part 3: Building Custom Images with Dockerfile

### What is a Dockerfile?

A Dockerfile is a text file with instructions for building a Docker image. Think of it as a recipe for your application.

**Why use Dockerfiles?**
- Reproducible builds (same Dockerfile always produces same image)
- Version controlled (track changes to your environment)
- Shareable (others can build your exact environment)
- Automated (CI/CD can build images automatically)

### Basic Dockerfile Structure

```dockerfile
# Start with a base image
FROM ubuntu:22.04

# Set metadata (optional)
LABEL maintainer="developer@example.com"
LABEL version="1.0"

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy application files
COPY requirements.txt .
COPY app.py .

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Set environment variables
ENV APP_ENV=production
ENV PORT=8000

# Expose port (documentation only, still need -p to publish)
EXPOSE 8000

# Define command to run when container starts
CMD ["python3", "app.py"]
```

### Understanding Dockerfile Instructions

#### FROM - The Foundation
```dockerfile
# Use official Python image as base
FROM python:3.9-slim

# Why use official images?
# - Maintained by Python team
# - Optimized for Python applications
# - Security updates applied regularly
# - Available in multiple variants:
#   python:3.9       - Full version with many tools
#   python:3.9-slim  - Minimal version (smaller)
#   python:3.9-alpine - Based on Alpine Linux (tiny)
```

**Best practice**: Use specific versions, not `latest`
```dockerfile
# Good - reproducible builds
FROM python:3.9.18-slim

# Bad - can change unexpectedly
FROM python:latest
```

#### RUN - Execute Commands
```dockerfile
# Install packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Why chain commands with &&?
# - Creates fewer layers (smaller image)
# - If any command fails, the whole RUN fails
# - Cleaning up in the same layer reduces size

# Create multiple layers only when needed for caching
RUN pip install --no-cache-dir -r requirements.txt
```

#### COPY vs ADD
```dockerfile
# COPY - Simple file copying (preferred)
COPY requirements.txt .
COPY app.py /app/
COPY ./src/ /app/src/

# ADD - Additional features (use sparingly)
ADD https://example.com/data/file.txt /data/
ADD archive.tar.gz /app/  # Automatically extracts

# Best practice: Use COPY unless you need ADD's special features
```

#### WORKDIR - Set Working Directory
```dockerfile
# Create and set working directory
WORKDIR /app

# Why use WORKDIR instead of cd?
# - Creates the directory if it doesn't exist
# - All subsequent commands use this directory
# - More readable than RUN cd /app

# Equivalent to:
# RUN mkdir -p /app
# WORKDIR /app
```

#### ENV - Environment Variables
```dockerfile
# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV APP_ENV=production
ENV DATABASE_URL=postgresql://localhost/rag_engine

# Can also be set at runtime (overrides Dockerfile)
# docker run -e APP_ENV=staging my-image
```

#### EXPOSE - Document Ports
```dockerfile
# Document which ports the application uses
EXPOSE 8000
EXPOSE 8080/tcp

# Note: This is documentation only!
# You still need -p flag to actually publish ports
```

#### CMD vs ENTRYPOINT

**CMD** - Default command (can be overridden):
```dockerfile
CMD ["python", "app.py"]

# Can be overridden at runtime:
# docker run my-image python --version
```

**ENTRYPOINT** - Fixed command (always runs):
```dockerfile
ENTRYPOINT ["python", "app.py"]

# Arguments are appended:
# docker run my-image --debug
# Runs: python app.py --debug
```

**Combination** (common pattern):
```dockerfile
ENTRYPOINT ["python"]
CMD ["app.py"]

# Default: python app.py
# Override: docker run my-image manage.py migrate
# Runs: python manage.py migrate
```

### Building Your First Image

**Project structure:**
```
my-docker-app/
├── Dockerfile
├── requirements.txt
└── app.py
```

**app.py:**
```python
from flask import Flask
import os

app = Flask(__name__)
port = int(os.environ.get('PORT', 8000))

@app.route('/')
def hello():
    return f"Hello from Docker! Running on port {port}"

@app.route('/health')
def health():
    return {"status": "healthy"}, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)
```

**requirements.txt:**
```
flask==2.3.0
gunicorn==21.2.0
```

**Dockerfile:**
```dockerfile
# Use official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Expose port
EXPOSE 8000

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "app:app"]
```

**Build the image:**
```bash
# Navigate to project directory
cd my-docker-app

# Build the image
docker build -t my-flask-app:1.0 .

# -t = tag (name and version)
# . = current directory (where Dockerfile is)

# Check the built image
docker images
```

**Run the container:**
```bash
# Run the container
docker run -d \
  -p 8080:8000 \
  --name my-flask-container \
  my-flask-app:1.0

# Test it
curl http://localhost:8080
curl http://localhost:8080/health
```

### Multi-Stage Builds (Optimization)

**The Problem:**
Your build process needs compilers, build tools, and dev dependencies. But these shouldn't be in your final image.

**Solution: Multi-stage builds**

```dockerfile
# Stage 1: Builder
FROM python:3.9 as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y gcc

# Install Python packages
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.9-slim

WORKDIR /app

# Copy only the installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY app.py .

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
```

**Benefits:**
- Final image is much smaller (no build tools)
- More secure (fewer packages = smaller attack surface)
- Faster deployments (smaller images transfer faster)

---

## Part 4: Docker Compose - Multi-Container Applications

### Why Docker Compose?

Real applications need multiple containers:
- Web application container
- Database container
- Cache container
- Background worker container

**The Problem:**
Managing multiple containers manually is tedious:
```bash
# Start database
docker run -d --name db -v db-data:/var/lib/postgresql/data postgres

# Start redis
docker run -d --name redis redis

# Start web app (linked to database and redis)
docker run -d --name web -p 8000:8000 \
  --link db:postgres \
  --link redis:redis \
  my-app

# Remember to start them in right order!
# Remember to connect them properly!
# What a mess!
```

**Docker Compose Solution:**
Define everything in a YAML file, start with one command.

### Docker Compose Basics

**docker-compose.yml structure:**
```yaml
version: '3.8'  # Docker Compose file format version

services:
  # Define your containers here
  web:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - db
      - redis
  
  db:
    image: postgres:14
    volumes:
      - db-data:/var/lib/postgresql/data
  
  redis:
    image: redis:7

volumes:
  db-data:
```

### Complete RAG Engine Example

**Project structure:**
```
rag-engine/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── config.py
└── .env
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  # RAG Engine API
  api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: rag-engine-api
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=${ENVIRONMENT:-production}
      - DEBUG=${DEBUG:-false}
      - DB_HOST=postgres
      - DB_PORT=5432
      - DB_NAME=${DB_NAME:-rag_engine}
      - DB_USER=${DB_USER:-rag_user}
      - DB_PASSWORD=${DB_PASSWORD:-changeme}
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - REDIS_PASSWORD=${REDIS_PASSWORD:-changeme}
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
    volumes:
      - ./data/documents:/app/data/documents
      - ./logs:/app/logs
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      qdrant:
        condition: service_started
    networks:
      - rag-engine-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # PostgreSQL Database
  postgres:
    image: postgres:14-alpine
    container_name: rag-engine-postgres
    restart: unless-stopped
    environment:
      POSTGRES_DB: ${DB_NAME:-rag_engine}
      POSTGRES_USER: ${DB_USER:-rag_user}
      POSTGRES_PASSWORD: ${DB_PASSWORD:-changeme}
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - rag-engine-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER:-rag_user} -d ${DB_NAME:-rag_engine}"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis Cache
  redis:
    image: redis:7-alpine
    container_name: rag-engine-redis
    restart: unless-stopped
    command: redis-server --requirepass ${REDIS_PASSWORD:-changeme}
    volumes:
      - redis-data:/data
    networks:
      - rag-engine-network
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Qdrant Vector Database
  qdrant:
    image: qdrant/qdrant:latest
    container_name: rag-engine-qdrant
    restart: unless-stopped
    volumes:
      - qdrant-data:/qdrant/storage
    networks:
      - rag-engine-network

  # Nginx Reverse Proxy
  nginx:
    image: nginx:alpine
    container_name: rag-engine-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - api
    networks:
      - rag-engine-network

# Named volumes for data persistence
volumes:
  postgres-data:
    driver: local
  redis-data:
    driver: local
  qdrant-data:
    driver: local

# Custom network for service communication
networks:
  rag-engine-network:
    driver: bridge
```

**Using Docker Compose:**
```bash
# Start all services
docker-compose up

# Start in background (detached mode)
docker-compose up -d

# View logs
docker-compose logs

# View logs for specific service
docker-compose logs -f api

# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes data!)
docker-compose down -v

# Rebuild images after code changes
docker-compose up -d --build

# Scale a service
docker-compose up -d --scale api=3

# Execute command in running container
docker-compose exec api python manage.py migrate

# Check service status
docker-compose ps
```

---

## Part 5: Best Practices and Production Considerations

### 1. Security Best Practices

**Don't run as root:**
```dockerfile
# Create non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Change ownership
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser
```

**Use minimal base images:**
```dockerfile
# Good - minimal attack surface
FROM python:3.9-alpine

# Avoid - large attack surface
FROM ubuntu:latest  # Has many unnecessary packages
```

**Scan for vulnerabilities:**
```bash
# Use Docker Scout or Trivy
# docker scout quickview my-image
# trivy image my-image
```

**Don't embed secrets:**
```dockerfile
# BAD - Never do this!
ENV DATABASE_PASSWORD=mysecretpassword

# GOOD - Use environment variables at runtime
ENV DATABASE_PASSWORD=
# Set at runtime: docker run -e DATABASE_PASSWORD=secret my-image
```

### 2. Image Optimization

**Minimize layers:**
```dockerfile
# Good - single layer
RUN apt-get update && apt-get install -y \
    package1 \
    package2 \
    && rm -rf /var/lib/apt/lists/*

# Bad - multiple layers
RUN apt-get update
RUN apt-get install -y package1
RUN apt-get install -y package2
```

**Use .dockerignore:**
```
# .dockerignore
.git
__pycache__
*.pyc
.env
node_modules
.vscode
.idea
*.md
!README.md
```

**Leverage build cache:**
```dockerfile
# Copy requirements first (rarely changes)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy code last (changes frequently)
COPY . .
```

### 3. Health Checks and Monitoring

```dockerfile
# Dockerfile
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
```

### 4. Logging Best Practices

```dockerfile
# Log to stdout/stderr (Docker collects these)
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

# Don't log to files inside container
# (use docker logs or centralized logging instead)
```

---

## Part 6: Common Patterns and Recipes

### Pattern 1: Development vs Production

**docker-compose.yml** (development):
```yaml
version: '3.8'
services:
  api:
    build: .
    volumes:
      - .:/app  # Mount code for live editing
    environment:
      - DEBUG=true
      - LOG_LEVEL=debug
    command: python app.py  # Development server
```

**docker-compose.prod.yml** (production):
```yaml
version: '3.8'
services:
  api:
    build: .
    restart: always
    environment:
      - DEBUG=false
      - LOG_LEVEL=info
      - WORKERS=4
    command: gunicorn -w 4 -b 0.0.0.0:8000 app:app
    deploy:
      replicas: 3
```

**Usage:**
```bash
# Development
docker-compose up

# Production
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Pattern 2: Database Migrations

```yaml
# docker-compose.yml
services:
  api:
    build: .
    # ...
  
  migrate:
    build: .
    command: python manage.py migrate
    depends_on:
      - postgres
    profiles:
      - migrate
```

```bash
# Run migrations
docker-compose --profile migrate run --rm migrate
```

### Pattern 3: One-off Commands

```bash
# Run tests
docker-compose run --rm api pytest

# Open database shell
docker-compose exec postgres psql -U rag_user -d rag_engine

# Backup database
docker-compose exec postgres pg_dump -U rag_user rag_engine > backup.sql
```

---

## Part 7: Troubleshooting Common Issues

### Issue 1: Container Exits Immediately

**Symptoms:**
```bash
docker run my-image
# Container starts and exits immediately
docker ps -a
# Shows container with status "Exited (0)"
```

**Causes and Solutions:**

1. **No long-running process:**
   ```dockerfile
   # BAD - echo finishes immediately
   CMD ["echo", "Hello"]
   
   # GOOD - keep process running
   CMD ["python", "app.py"]
   # Or for debugging:
   CMD ["sleep", "infinity"]
   ```

2. **Application crashes:**
   ```bash
   # Check logs
   docker logs <container_id>
   ```

### Issue 2: Port Already in Use

**Symptoms:**
```
Bind for 0.0.0.0:8080 failed: port is already allocated
```

**Solution:**
```bash
# Find what's using the port
sudo lsof -i :8080
# or
sudo netstat -tulpn | grep 8080

# Stop the conflicting service
# Or use a different port
docker run -p 8081:80 nginx
```

### Issue 3: Permission Denied

**Symptoms:**
```
Error: EACCES: permission denied, mkdir '/app/data'
```

**Solutions:**

1. **Fix host directory permissions:**
   ```bash
   sudo chown -R $USER:$USER ./data
   ```

2. **Run container with your user ID:**
   ```bash
   docker run -u $(id -u):$(id -g) -v $(pwd)/data:/app/data my-image
   ```

3. **Adjust Dockerfile:**
   ```dockerfile
   RUN mkdir -p /app/data && chmod 777 /app/data
   ```

### Issue 4: Container Can't Reach Internet

**Symptoms:**
```
curl: (6) Could not resolve host: example.com
```

**Causes:**
1. Docker daemon networking issue
2. DNS configuration problem
3. Corporate proxy blocking traffic

**Solutions:**
```bash
# Check Docker network
docker network ls
docker network inspect bridge

# Restart Docker service
sudo systemctl restart docker

# Check DNS in container
docker run busybox nslookup google.com
```

### Issue 5: Image Build is Slow

**Solutions:**

1. **Use build cache effectively:**
   - Copy requirements before code
   - Put frequently changing instructions last

2. **Use .dockerignore:**
   - Don't copy unnecessary files

3. **Use smaller base images:**
   ```dockerfile
   FROM python:3.9-alpine  # 50MB
   # vs
   FROM python:3.9        # 900MB
   ```

4. **Parallelize builds:**
   ```bash
   DOCKER_BUILDKIT=1 docker build .
   ```

---

## Summary

### Key Takeaways

1. **Docker solves the "it works on my machine" problem** by packaging applications with all their dependencies

2. **Images are blueprints, containers are running instances** - think recipe vs. baked cake

3. **Layers make Docker efficient** - changed layers are rebuilt, unchanged layers are cached

4. **Dockerfile is your recipe** - it defines how to build your image

5. **Docker Compose orchestrates multi-container apps** - one command starts your entire stack

6. **Security matters** - don't run as root, minimize image size, scan for vulnerabilities

7. **Volumes persist data** - without them, data disappears when containers are removed

### Quick Command Reference

```bash
# Images
docker images                    # List images
docker build -t name:tag .      # Build image
docker rmi image_id             # Delete image

# Containers
docker ps                       # List running containers
docker ps -a                    # List all containers
docker run -d -p 8080:80 name   # Run container
docker stop container_id        # Stop container
docker start container_id       # Start container
docker rm container_id          # Delete container
docker logs container_id        # View logs
docker exec -it container_id sh # Shell into container

# Volumes
docker volume ls                # List volumes
docker volume create name       # Create volume
docker volume rm name           # Delete volume

# Docker Compose
docker-compose up               # Start all services
docker-compose up -d            # Start in background
docker-compose down             # Stop all services
docker-compose logs             # View logs
docker-compose ps               # List services
docker-compose build            # Rebuild images

# System
docker system df                # Check disk usage
docker system prune             # Clean up unused data
```

### Next Steps

1. **Practice**: Build a simple Flask app with Docker
2. **Explore**: Try different base images (alpine, slim, full)
3. **Optimize**: Implement multi-stage builds
4. **Secure**: Run containers as non-root user
5. **Scale**: Learn Docker Swarm or Kubernetes

### Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Hub](https://hub.docker.com/) - Find official images
- [Dockerfile Best Practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)

---

**You now understand Docker from first principles! 🎉**

Next: Learn [Kubernetes Fundamentals](02-kubernetes-fundamentals.md) to orchestrate containers at scale.
