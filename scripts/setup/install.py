#!/usr/bin/env python3
"""
AI-Mastery-2026 Installation Script
====================================

Automated setup for development environment.

Usage:
------
    # Basic installation
    python scripts/setup/install.py
    
    # Full development setup
    python scripts/setup/install.py --dev
    
    # Production installation
    python scripts/setup/install.py --prod
    
    # Verify installation
    python scripts/setup/install.py --verify
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import Optional, List
import argparse
from datetime import datetime


# Colors for output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str) -> None:
    """Print header text."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}\n")


def print_success(text: str) -> None:
    """Print success message."""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_error(text: str) -> None:
    """Print error message."""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_warning(text: str) -> None:
    """Print warning message."""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def print_info(text: str) -> None:
    """Print info message."""
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")


def run_command(cmd: List[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """Run a shell command."""
    try:
        if capture:
            return subprocess.run(cmd, check=check, capture_output=True, text=True)
        return subprocess.run(cmd, check=check)
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {' '.join(cmd)}")
        if e.stderr:
            print_error(f"Error: {e.stderr}")
        raise


def check_python_version() -> bool:
    """Check Python version."""
    print_info("Checking Python version...")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print_success(f"Python {version.major}.{version.minor}.{version.micro} is supported")
        return True
    else:
        print_error(f"Python 3.10+ required, found {version.major}.{version.minor}")
        return False


def check_pip() -> bool:
    """Check pip installation."""
    print_info("Checking pip...")
    
    try:
        result = run_command([sys.executable, "-m", "pip", "--version"], capture=True)
        print_success(f"pip is installed: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError:
        print_error("pip is not installed")
        return False


def install_dependencies(extras: Optional[List[str]] = None) -> bool:
    """Install Python dependencies."""
    print_header("Installing Dependencies")
    
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "pip"]
    run_command(cmd)
    
    # Install main package
    cmd = [sys.executable, "-m", "pip", "install", "-e", "."]
    if extras:
        cmd[-1] += f"[{','.join(extras)}]"
    
    print_info(f"Running: {' '.join(cmd)}")
    try:
        run_command(cmd)
        print_success("Main package installed")
        return True
    except subprocess.CalledProcessError:
        print_error("Failed to install main package")
        return False


def install_dev_dependencies() -> bool:
    """Install development dependencies."""
    print_info("Installing development dependencies...")
    
    dev_req = Path("requirements-dev.txt")
    if dev_req.exists():
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(dev_req)]
        try:
            run_command(cmd)
            print_success("Development dependencies installed")
            return True
        except subprocess.CalledProcessError:
            print_warning("Some development dependencies may have failed to install")
            return False
    else:
        print_warning("requirements-dev.txt not found")
        return False


def setup_pre_commit() -> bool:
    """Install pre-commit hooks."""
    print_info("Installing pre-commit hooks...")
    
    pre_commit_config = Path(".pre-commit-config.yaml")
    if not pre_commit_config.exists():
        print_warning(".pre-commit-config.yaml not found")
        return False
    
    try:
        run_command(["pre-commit", "install"])
        print_success("Pre-commit hooks installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_warning("pre-commit not installed, skipping...")
        print_info("Install with: pip install pre-commit")
        return False


def setup_jupyter_kernel() -> bool:
    """Register Jupyter kernel."""
    print_info("Registering Jupyter kernel...")
    
    try:
        run_command([
            sys.executable, "-m", "ipykernel", "install",
            "--user",
            "--name", "ai-mastery-2026",
            "--display-name", "AI-Mastery-2026",
        ])
        print_success("Jupyter kernel registered")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_warning("ipykernel not installed, skipping...")
        print_info("Install with: pip install ipykernel")
        return False


def create_directories() -> bool:
    """Create necessary directories."""
    print_info("Creating directories...")
    
    directories = [
        "data",
        "models",
        "logs",
        "results",
        "notebooks",
    ]
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print_success(f"Created directory: {directory}")
    
    return True


def create_env_file() -> bool:
    """Create .env file if it doesn't exist."""
    print_info("Checking .env file...")
    
    env_file = Path(".env")
    if not env_file.exists():
        env_example = Path(".env.example")
        if env_example.exists():
            import shutil
            shutil.copy(env_example, env_file)
            print_success("Created .env from .env.example")
            print_warning("Please update .env with your configuration")
        else:
            # Create minimal .env
            with open(env_file, "w") as f:
                f.write("""# AI-Mastery-2026 Environment Configuration
# Copy this file to .env and update with your values

# Application
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/ai_mastery
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ai_mastery
DB_USER=postgres
DB_PASSWORD=password

# Redis
REDIS_URL=redis://localhost:6379
REDIS_HOST=localhost
REDIS_PORT=6379

# LLM
LLM_PROVIDER=openai
LLM_API_KEY=your-api-key-here
LLM_MODEL=gpt-4

# Embeddings
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL=all-MiniLM-L6-v2

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Security (CHANGE IN PRODUCTION!)
JWT_SECRET_KEY=dev-secret-key-change-in-production
""")
            print_success("Created minimal .env file")
            print_warning("Please update .env with your configuration")
    else:
        print_success(".env file already exists")
    
    return True


def verify_installation() -> bool:
    """Verify installation by importing key modules."""
    print_header("Verifying Installation")
    
    test_imports = [
        ("numpy", "NumPy"),
        ("torch", "PyTorch"),
        ("fastapi", "FastAPI"),
        ("pydantic", "Pydantic"),
        ("src.core", "src.core"),
        ("src.ml", "src.ml"),
        ("src.llm", "src.llm"),
        ("src.utils", "src.utils"),
    ]
    
    all_passed = True
    
    for module, name in test_imports:
        try:
            __import__(module)
            print_success(f"{name}")
        except ImportError as e:
            print_error(f"{name}: {e}")
            all_passed = False
    
    return all_passed


def print_summary(success: bool) -> None:
    """Print installation summary."""
    print_header("Installation Summary")
    
    if success:
        print_success("Installation completed successfully!")
        print("")
        print_info("Next steps:")
        print("  1. Update .env with your configuration")
        print("  2. Run 'make verify-install' to verify")
        print("  3. Run 'make test' to run tests")
        print("  4. Run 'make run-api' to start the API")
        print("")
        print_info("Documentation: http://localhost:8000/docs")
    else:
        print_error("Installation completed with errors")
        print("")
        print_info("Please review the errors above and try again")


def main() -> int:
    """Main installation function."""
    parser = argparse.ArgumentParser(description="AI-Mastery-2026 Installation Script")
    parser.add_argument("--dev", action="store_true", help="Install development dependencies")
    parser.add_argument("--prod", action="store_true", help="Production installation")
    parser.add_argument("--verify", action="store_true", help="Verify installation only")
    parser.add_argument("--no-pre-commit", action="store_true", help="Skip pre-commit setup")
    parser.add_argument("--no-jupyter", action="store_true", help="Skip Jupyter kernel setup")
    
    args = parser.parse_args()
    
    print_header("AI-Mastery-2026 Installation")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Python: {sys.version}")
    print(f"Directory: {Path.cwd()}")
    
    # Verify only mode
    if args.verify:
        success = verify_installation()
        return 0 if success else 1
    
    # Check prerequisites
    print_header("Prerequisites")
    
    if not check_python_version():
        return 1
    
    if not check_pip():
        return 1
    
    # Create directories
    create_directories()
    
    # Create .env file
    create_env_file()
    
    # Install dependencies
    extras = ["all"] if not args.prod else []
    if not install_dependencies(extras):
        return 1
    
    # Install dev dependencies
    if args.dev and not args.prod:
        install_dev_dependencies()
    
    # Setup pre-commit
    if not args.no_pre_commit:
        setup_pre_commit()
    
    # Setup Jupyter kernel
    if not args.no_jupyter:
        setup_jupyter_kernel()
    
    # Verify installation
    success = verify_installation()
    
    # Print summary
    print_summary(success)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
