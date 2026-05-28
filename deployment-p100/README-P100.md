# ACE-Step Tesla P100 Deployment Environment

## Environment Specifications

| Component | Version / Value |
|-----------|---------------|
| **Python** | 3.12.13 |
| **PyTorch** | 2.4.0+cu118 |
| **CUDA Toolkit** | 11.8 |
| **GPU** | Tesla P100-PCIE-16GB |
| **Compute Capability** | 6.0 (Pascal) |
| **diffusers** | 0.31.0 |
| **transformers** | 4.54.1 |
| **accelerate** | 1.1.1 |

## GPU Compatibility

This environment is specifically configured for **NVIDIA Pascal architecture GPUs** (compute capability 6.0).

**Supported GPUs:**
- Tesla P100 (16GB)
- GTX 1080 Ti
- GTX 1080
- GTX 1070
- GTX 1060 (6GB/3GB)
- Other Pascal-based GPUs

## Known Limitations

### Tesla P100 Specific
- **Compute capability 6.0** — not supported by PyTorch 2.5+ with CUDA 12.x
- **FP16 only** — bf16 is not natively supported on Pascal architecture
- **No flash attention** — flash-attn and xformers require sm_70+ (Volta+)
- **No tensor cores** — Pascal lacks mixed-precision tensor cores for fp16 acceleration
- **SDPA softmax overflow risk** — eager attention mode used for numerical stability with fp16

### Software Constraints
- **PyTorch pinned to 2.4.0+cu118** — newer versions drop sm_60 support
- **torchao incompatible** — requires PyTorch 2.5+
- **torchcodec incompatible** — requires PyTorch 2.5+
- **Python 3.12 required** — PyTorch 2.4 cu118 wheels not available for Python 3.13+

### Operational Constraints
- **No automatic model downloads** — models must be manually downloaded to `checkpoints/`
- **Memory management** — 16GB VRAM requires careful batch sizing and tiled VAE decoding
- **CPU fallback** — generation on CPU is extremely slow; not recommended for production

## Quick Start

### 1. Verify Environment

```bash
cd deployment-p100
chmod +x verify-p100.sh
./verify-p100.sh
```

### 2. Fresh Installation

```bash
cd deployment-p100
chmod +x install-p100.sh
./install-p100.sh
```

### 3. Run Inference

```bash
source ../.venv-p100/bin/activate
python ../test_inference.py
```

## File Inventory

| File | Purpose |
|------|---------|
| `requirements.p100.lock.txt` | Exact package versions (pip freeze output) |
| `env.p100.summary.txt` | Environment summary (Python, PyTorch, GPU details) |
| `DO_NOT_CHANGE_P100_ENV.txt` | Critical constraints reminder |
| `install-p100.sh` | Automated environment setup script |
| `verify-p100.sh` | Environment verification script |
| `README-P100.md` | This documentation file |

## Troubleshooting

### CUDA not available
```bash
# Check NVIDIA driver
nvidia-smi

# Check PyTorch CUDA detection
.venv-p100/bin/python -c "import torch; print(torch.cuda.is_available())"
```

### Out of memory during generation
- Reduce batch size to 1
- Reduce generation duration
- Use fewer inference steps
- Ensure model is loaded in fp16, not fp32

### "no kernel image" error
This indicates PyTorch was compiled without sm_60 support. Ensure you're using:
- PyTorch 2.4.0+cu118 (NOT 2.5+)
- CUDA 11.8 runtime (NOT 12.x)
