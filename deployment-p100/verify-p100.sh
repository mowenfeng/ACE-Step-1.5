#!/usr/bin/env bash
set -euo pipefail

# ACE-Step P100 Environment Verification
# Checks that the GPU environment is properly configured

VENV_PATH=".venv-p100/bin/python"
PASS_COUNT=0
FAIL_COUNT=0

echo "=== ACE-Step P100 Environment Verification ==="
echo ""

# Check Python
PYTHON_VER=$($VENV_PATH -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
if [[ "$PYTHON_VER" == 3.12* ]]; then
    echo "PASS: Python version $PYTHON_VER"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "FAIL: Expected Python 3.12, got $PYTHON_VER"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# Check CUDA availability
CUDA_AVAIL=$($VENV_PATH -c "import torch; print(torch.cuda.is_available())")
if [ "$CUDA_AVAIL" = "True" ]; then
    echo "PASS: CUDA is available"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "FAIL: CUDA is NOT available"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# Check GPU name
GPU_NAME=$($VENV_PATH -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
if [ -n "$GPU_NAME" ]; then
    echo "PASS: GPU detected: $GPU_NAME"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "FAIL: Could not detect GPU name"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# Check compute capability
CAPABILITY=$($VENV_PATH -c "import torch; cap = torch.cuda.get_device_capability(0); print(f'{cap[0]}.{cap[1]}')" 2>/dev/null)
if [ "$CAPABILITY" = "6.0" ]; then
    echo "PASS: Compute capability $CAPABILITY (P100)"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "WARN: Compute capability is $CAPABILITY (expected 6.0 for P100)"
    PASS_COUNT=$((PASS_COUNT + 1))
fi

# Check PyTorch version
TORCH_VER=$($VENV_PATH -c "import torch; print(torch.__version__)" 2>/dev/null)
if [[ "$TORCH_VER" == 2.4* ]]; then
    echo "PASS: PyTorch version $TORCH_VER"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "FAIL: Expected PyTorch 2.4.x, got $TORCH_VER"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# Check CUDA compiled version
CUDA_VER=$($VENV_PATH -c "import torch; print(torch.version.cuda)" 2>/dev/null)
if [[ "$CUDA_VER" == 11.8* ]]; then
    echo "PASS: CUDA toolkit $CUDA_VER"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "FAIL: Expected CUDA 11.8, got $CUDA_VER"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# Check bf16 support (should be False for P100)
BF16_SUPPORT=$($VENV_PATH -c "import torch; print(torch.cuda.is_bf16_supported())" 2>/dev/null)
if [ "$BF16_SUPPORT" = "False" ]; then
    echo "PASS: BF16 not supported (correct for P100)"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "INFO: BF16 reported as supported ($BF16_SUPPORT) - will use fp16 fallback"
    PASS_COUNT=$((PASS_COUNT + 1))
fi

echo ""
echo "=== Results: $PASS_COUNT passed, $FAIL_COUNT failed ==="

if [ "$FAIL_COUNT" -gt 0 ]; then
    echo "FAIL: Environment verification failed"
    exit 1
else
    echo "PASS: Environment verified successfully"
    exit 0
fi
