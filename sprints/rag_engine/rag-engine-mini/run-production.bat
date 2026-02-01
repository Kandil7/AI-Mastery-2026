@echo off
REM =============================================================================
REM RAG Engine Mini - Windows Production Deployment
REM =============================================================================
REM This script sets up the RAG Engine for production on Windows
REM 
REM INFRASTRUCTURE (Docker):
REM   - PostgreSQL on port 5432
REM   - Redis on port 6379
REM   - Qdrant on port 6333
REM
REM APPLICATION (Local Python):
REM   - FastAPI API on port 8000
REM =============================================================================

echo ==============================================================================
echo 🚀 RAG ENGINE MINI - WINDOWS PRODUCTION DEPLOYMENT
echo ==============================================================================
echo.

REM =============================================================================
REM STEP 1: Check Prerequisites
REM =============================================================================
echo [1/6] Checking prerequisites...

python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.11+ from https://python.org
    exit /b 1
)

echo ✅ Python found
echo.

REM =============================================================================
REM STEP 2: Create Environment File
REM =============================================================================
echo [2/6] Setting up environment...

if not exist .env (
    echo ⚠️  .env file not found. Creating from .env.example...
    copy .env.example .env
    echo ⚠️  Please edit .env file to add your API keys before continuing
)

echo ✅ Environment configured
echo.

REM =============================================================================
REM STEP 3: Create Virtual Environment
REM =============================================================================
echo [3/6] Setting up Python virtual environment...

if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
)

echo ✅ Virtual environment ready
echo.

REM =============================================================================
REM STEP 4: Install Dependencies
REM =============================================================================
echo [4/6] Installing Python dependencies...

call .venv\Scripts\activate.bat
pip install -q -r requirements.txt

echo ✅ Dependencies installed
echo.

REM =============================================================================
REM STEP 5: Start Infrastructure Services (if Docker is available)
REM =============================================================================
echo [5/6] Checking for Docker...

docker --version >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Docker not found. Running without infrastructure services.
    echo ⚠️  You need to provide your own PostgreSQL, Redis, and Qdrant instances.
    echo.
    echo To use Docker infrastructure:
    echo   1. Install Docker Desktop from https://docker.com
    echo   2. Run: docker-compose -f docker-compose.infrastructure.yml up -d
    echo.
) else (
    echo ✅ Docker found
    echo Starting infrastructure services...
    docker-compose -f docker-compose.infrastructure.yml up -d
    echo.
    echo ⏳ Waiting for services to start...
    timeout /t 10 /nobreak >nul
    echo ✅ Infrastructure services started
)

echo.

REM =============================================================================
REM STEP 6: Start Application
REM =============================================================================
echo [6/6] Starting RAG Engine API...
echo.
echo ==============================================================================
echo 🚀 RAG ENGINE IS NOW RUNNING!
echo ==============================================================================
echo.
echo 📋 SERVICE ENDPOINTS:
echo.
echo   API Server:       http://localhost:8000
echo   API Docs:         http://localhost:8000/docs
echo   Health Check:     http://localhost:8000/health
echo   GraphQL:          http://localhost:8000/graphql
echo.
if errorlevel 0 (
    echo   PostgreSQL:       localhost:5432
echo   Redis:            localhost:6379
echo   Qdrant:           http://localhost:6333
echo.
)
echo ==============================================================================
echo 🛑 TO STOP: Press Ctrl+C
echo ==============================================================================
echo.

REM Start the API server
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
