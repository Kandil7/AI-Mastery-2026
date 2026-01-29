#!/bin/bash

set -e

ENV_MANAGER="auto"
CUDA_VERSION=""
INSTALL_JUPYTER=0
RUN_TEST=0
MINIMAL=0
ENV_NAME="ai-mastery-2026"
ENV_FILE="environment.full.yml"
VENV_PATH=".venv"

usage() {
  cat <<EOF
Usage: ./setup.sh [--conda|--venv|--auto] [--cuda VERSION] [--jupyter] [--test] [--minimal]
  --conda     Use conda environment (default if conda is available)
  --venv      Use python venv
  --auto      Auto-detect conda, fallback to venv (default)
  --cuda      Install CUDA build of PyTorch (conda only)
  --jupyter   Register Jupyter kernel
  --test      Run a quick import smoke test
  --minimal   Use requirements-minimal.txt (venv only)
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --conda) ENV_MANAGER="conda" ;;
    --venv) ENV_MANAGER="venv" ;;
    --auto) ENV_MANAGER="auto" ;;
    --cuda) CUDA_VERSION="$2"; shift ;;
    --jupyter) INSTALL_JUPYTER=1 ;;
    --test) RUN_TEST=1 ;;
    --minimal) MINIMAL=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
  shift
done

if [ "$ENV_MANAGER" = "auto" ]; then
  if command -v conda >/dev/null 2>&1; then
    ENV_MANAGER="conda"
  else
    ENV_MANAGER="venv"
  fi
fi

if [ "$ENV_MANAGER" = "conda" ]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found. Install Miniconda/Anaconda or use --venv."
    exit 1
  fi

  echo "Setting up conda environment: $ENV_NAME"
  if conda env list | awk '{print $1}' | grep -q "^${ENV_NAME}\$"; then
    conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
  else
    conda env create -n "$ENV_NAME" -f "$ENV_FILE"
  fi

  if [ -n "$CUDA_VERSION" ]; then
    echo "Installing CUDA PyTorch build (pytorch-cuda=$CUDA_VERSION)"
    conda install -n "$ENV_NAME" pytorch torchvision torchaudio pytorch-cuda="$CUDA_VERSION" -c pytorch -c nvidia -y
  fi

  if [ "$INSTALL_JUPYTER" -eq 1 ]; then
    conda run -n "$ENV_NAME" python -m ipykernel install --user --name "$ENV_NAME" --display-name "AI-Mastery-2026"
  fi

  if [ "$RUN_TEST" -eq 1 ]; then
    conda run -n "$ENV_NAME" python -c "import numpy, torch, fastapi; print('deps ok')"
  fi

  echo "Done. Activate with: conda activate $ENV_NAME"
  exit 0
fi

echo "Setting up venv at $VENV_PATH"
if [ ! -d "$VENV_PATH" ]; then
  python -m venv "$VENV_PATH"
fi

VENV_PY="$VENV_PATH/bin/python"
"$VENV_PY" -m pip install --upgrade pip

if [ "$MINIMAL" -eq 1 ]; then
  "$VENV_PY" -m pip install -r requirements-minimal.txt
else
  "$VENV_PY" -m pip install -r requirements.txt
fi

if [ -n "$CUDA_VERSION" ]; then
  echo "CUDA requested, but venv install is not automated. Install a matching torch wheel manually."
fi

if [ "$INSTALL_JUPYTER" -eq 1 ]; then
  "$VENV_PY" -m ipykernel install --user --name "$ENV_NAME" --display-name "AI-Mastery-2026"
fi

if [ "$RUN_TEST" -eq 1 ]; then
  "$VENV_PY" -c "import numpy, torch, fastapi; print('deps ok')"
fi

echo "Done. Activate with: source $VENV_PATH/bin/activate"
