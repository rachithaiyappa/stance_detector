#!/bin/bash
set -e

echo "🔧 Creating or updating the virtual environment..."
uv venv stance-detector
source stance-detector/bin/activate

echo "📦 Syncing standard dependencies from pyproject.toml..."
uv sync

echo "Installing the stance-detector package in an editable mode..."
uv pip install -e .

echo "⚙️ Installing CUDA-specific PyTorch packages..."
uv pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 \
    --extra-index-url https://download.pytorch.org/whl/cu116

echo "✅ Setup complete"