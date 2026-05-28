#!/bin/bash
# Sets up Python 3.10 venv with xmgn deps inside WSL Ubuntu via apt.
# Run as root inside Ubuntu: wsl -d Ubuntu-24.04 --user root -- bash setup_wsl_apt.sh
# No conda — uses deadsnakes PPA for Python 3.10 + standard venv + pip.

set -e
set -x

export DEBIAN_FRONTEND=noninteractive

# --- 1. apt update + install Python 3.10 from deadsnakes ---
echo "==> Updating apt cache..."
apt-get update -qq

echo "==> Installing prereqs for adding PPAs..."
apt-get install -y -qq software-properties-common curl ca-certificates

echo "==> Adding deadsnakes PPA (for Python 3.10 on Ubuntu 24.04)..."
add-apt-repository -y ppa:deadsnakes/ppa

echo "==> Updating apt cache with PPA..."
apt-get update -qq

echo "==> Installing Python 3.10..."
apt-get install -y -qq python3.10 python3.10-venv python3.10-dev

# --- 2. Create venv at /root/xmgn-env ---
VENV=/root/xmgn-env
if [ ! -d "$VENV" ]; then
    echo "==> Creating venv at $VENV..."
    python3.10 -m venv "$VENV"
else
    echo "==> Venv already exists at $VENV"
fi

PY="$VENV/bin/python"

echo "==> Upgrading pip..."
$PY -m pip install --upgrade pip --quiet

# --- 3. Install xmgn requirements + physicsnemo ---
echo "==> Installing xmgn requirements (torch 2.4.0 + PyG extensions, ~2.5 GB download)..."
$PY -m pip install -r /mnt/e/NVIDIA/reservoir_simulation/xmgn/requirements.txt

echo "==> Installing nvidia-physicsnemo + its dependencies..."
$PY -m pip install nvidia-physicsnemo s3fs einops onnx timm treelib xarray zarr

# --- 4. Verification ---
echo "==> Verifying torch + CUDA + PyG + physicsnemo..."
$PY -c "
import torch
print(f'torch:       {torch.__version__}')
print(f'CUDA built:  {torch.version.cuda}')
print(f'CUDA avail:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:         {torch.cuda.get_device_name(0)}')
    print(f'Compute cap: {torch.cuda.get_device_capability(0)}')

import torch_scatter, torch_sparse, torch_cluster
print(f'torch_scatter: {torch_scatter.__version__}')
print(f'torch_sparse:  {torch_sparse.__version__}')
print(f'torch_cluster: {torch_cluster.__version__}')

from physicsnemo.distributed import DistributedManager
print('physicsnemo: OK')
from physicsnemo.models.meshgraphnet import MeshGraphNet
print('MeshGraphNet: OK')

# Quick CUDA scatter test (the one that crashed on Windows)
if torch.cuda.is_available():
    src = torch.randn(10, device='cuda')
    idx = torch.tensor([0,0,1,1,2,2,3,3,4,4], device='cuda')
    out = torch_scatter.scatter_add(src, idx, dim=0)
    print(f'CUDA scatter test: OK (output shape {tuple(out.shape)})')
"

echo ""
echo "==> Setup complete. Activate via:  source $VENV/bin/activate"
