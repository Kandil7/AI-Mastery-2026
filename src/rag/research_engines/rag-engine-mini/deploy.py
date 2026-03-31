#!/usr/bin/env python3
"""
RAG Engine Mini - Production Deployment Script
==============================================
Sets up and runs the RAG Engine in production mode.
"""

import os
import sys
import subprocess
import time
from pathlib import Path


def print_header():
    """Print deployment header."""
    print("=" * 79)
    print("RAG ENGINE MINI - PRODUCTION DEPLOYMENT")
    print("=" * 79)
    print()


def check_prerequisites():
    """Check if Python is available."""
    print("[1/6] Checking prerequisites...")

    try:
        result = subprocess.run(["python", "--version"], capture_output=True, text=True, check=True)
        print(f"[OK] Python found: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[ERROR] Python is not installed or not in PATH")
        print("Please install Python 3.11+ from https://python.org")
        sys.exit(1)

    print()


def setup_environment():
    """Create .env file if it doesn't exist."""
    print("[2/6] Setting up environment...")

    env_file = Path(".env")
    env_example = Path(".env.example")

    if not env_file.exists() and env_example.exists():
        print("[!] .env file not found. Creating from .env.example...")
        env_file.write_text(env_example.read_text())
        print("[!] Please edit .env file to add your API keys before continuing")

    print("[OK] Environment configured")
    print()


def setup_virtual_env():
    """Create and activate virtual environment."""
    print("[3/6] Setting up Python virtual environment...")

    venv_path = Path(".venv")

    if not venv_path.exists():
        print("Creating virtual environment...")
        subprocess.run(["python", "-m", "venv", ".venv"], check=True)

    print("[OK] Virtual environment ready")
    print()


def install_dependencies():
    """Install Python dependencies."""
    print("[4/6] Installing Python dependencies...")

    # Determine pip path based on OS
    if os.name == "nt":  # Windows
        pip_path = ".venv\\Scripts\\pip.exe"
        python_path = ".venv\\Scripts\\python.exe"
    else:  # Unix/Linux/Mac
        pip_path = ".venv/bin/pip"
        python_path = ".venv/bin/python"

    # Upgrade pip first
    subprocess.run([python_path, "-m", "pip", "install", "--upgrade", "pip"], capture_output=True)

    # Install requirements
    result = subprocess.run(
        [pip_path, "install", "-r", "requirements.txt"], capture_output=True, text=True
    )

    if result.returncode != 0:
        print("[!] Some dependencies failed to install:")
        print(result.stderr)
        print("Continuing anyway...")
    else:
        print("[OK] Dependencies installed")

    print()


def start_infrastructure():
    """Start Docker infrastructure services if available."""
    print("[5/6] Checking for Docker...")

    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True, check=True)
        print(f"[OK] Docker found: {result.stdout.strip()}")

        # Try to start infrastructure
        print("Starting infrastructure services...")
        infra_result = subprocess.run(
            ["docker-compose", "-f", "docker-compose.infrastructure.yml", "up", "-d"],
            capture_output=True,
            text=True,
        )

        if infra_result.returncode == 0:
            print("[*] Waiting for services to start...")
            time.sleep(10)
            print("[OK] Infrastructure services started")
            return True
        else:
            print("[!] Could not start Docker services:")
            print(infra_result.stderr)
            return False

    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[!] Docker not found. Running without infrastructure services.")
        print("[!] You need to provide your own PostgreSQL, Redis, and Qdrant instances.")
        print()
        print("To use Docker infrastructure:")
        print("  1. Install Docker Desktop from https://docker.com")
        print("  2. Run: docker-compose -f docker-compose.infrastructure.yml up -d")
        print()
        return False


def start_application():
    """Start the RAG Engine API server."""
    print("[6/6] Starting RAG Engine API...")
    print()
    print("=" * 79)
    print("RAG ENGINE IS NOW RUNNING!")
    print("=" * 79)
    print()
    print("SERVICE ENDPOINTS:")
    print()
    print("  API Server:       http://localhost:8000")
    print("  API Docs:         http://localhost:8000/docs")
    print("  Health Check:     http://localhost:8000/health")
    print("  GraphQL:          http://localhost:8000/graphql")
    print()
    print("  PostgreSQL:       localhost:5432")
    print("  Redis:            localhost:6379")
    print("  Qdrant:           http://localhost:6333")
    print()
    print("=" * 79)
    print("TO STOP: Press Ctrl+C")
    print("=" * 79)
    print()

    # Determine Python path
    if os.name == "nt":  # Windows
        python_path = ".venv\\Scripts\\python.exe"
    else:  # Unix/Linux/Mac
        python_path = ".venv/bin/python"

    # Start the API server
    try:
        subprocess.run(
            [
                python_path,
                "-m",
                "uvicorn",
                "src.main:app",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
                "--reload",
            ],
            check=True,
        )
    except KeyboardInterrupt:
        print("\n\nGoodbye! Shutting down RAG Engine...")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Failed to start API: {e}")
        sys.exit(1)


def main():
    """Main deployment function."""
    print_header()

    # Run all setup steps
    check_prerequisites()
    setup_environment()
    setup_virtual_env()
    install_dependencies()
    start_infrastructure()

    # Start the application
    start_application()


if __name__ == "__main__":
    main()
