#!/usr/bin/env bash
set -euo pipefail

# ACE-Step P100 Environment Installer
# Installs Python 3.12 venv with PyTorch 2.4 + CUDA 11.8 for Tesla P100

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Step 1: Check Ubuntu ─────────────────────────────────────────────
echo "=== Checking Ubuntu version ==="
if [ -f /etc/os-release ]; then
    UBUNTU_VERSION=$(. /etc/os-release && echo "$VERSION_ID")
    echo "Ubuntu version: $UBUNTU_VERSION"
else
    echo "WARNING: Could not determine Ubuntu version"
fi

# ── Step 2: Check NVIDIA GPU ─────────────────────────────────────────
echo "=== Checking NVIDIA GPU ==="
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "Detected GPU: $GPU_INFO"
else
    echo "ERROR: nvidia-smi not found. NVIDIA driver required."
    exit 1
fi

# ── Step 3: Create virtual environment ───────────────────────────────
echo "=== Creating Python 3.12 virtual environment ==="
cd "$SCRIPT_DIR/.."
python3.12 -m venv .venv-p100
echo "Virtual environment created at .venv-p100"

# ── Step 4: Install uv (if not present) ──────────────────────────────
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# ── Step 5: Install PyTorch 2.4 + CUDA 11.8 ─────────────────────────
echo "=== Installing PyTorch 2.4 with CUDA 11.8 ==="
uv pip install --python .venv-p100/bin/python \
    torch==2.4.0+cu118 \
    torchvision==0.19.0+cu118 \
    torchaudio==2.4.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# ── Step 6: Install project requirements ─────────────────────────────
echo "=== Installing project dependencies ==="
uv pip install --python .venv-p100/bin/python -r "$SCRIPT_DIR/requirements.p100.lock.txt"

# ── Step 7: Verify CUDA ──────────────────────────────────────────────
echo "=== Verifying CUDA availability ==="
.venv-p100/bin/python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version (compiled): {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Compute capability: {torch.cuda.get_device_capability(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
    print('SUCCESS: CUDA environment verified!')
else:
    print('FAILURE: CUDA not available!')
    exit(1)
"

echo "=== Installation complete ==="
echo "Activate with: source .venv-p100/bin/activate"
