#!/bin/bash
# Sets up a clean Python 3.10 conda env in WSL Ubuntu, installs xmgn requirements
# + physicsnemo. No sudo required — everything goes into ~/miniconda3/ and conda env.

set -e   # exit on any error
set -x   # echo each command for visibility

cd "$HOME"

# --- 1. Install Miniconda (~150 MB) if not already present ---
if [ ! -d "$HOME/miniconda3" ]; then
    echo "==> Downloading Miniconda installer..."
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda_installer.sh
    echo "==> Installing Miniconda to ~/miniconda3 (batch mode, no prompts)..."
    bash miniconda_installer.sh -b -p "$HOME/miniconda3"
    rm miniconda_installer.sh
else
    echo "==> Miniconda already installed at ~/miniconda3"
fi

# --- 2. Activate conda + create Python 3.10 env ---
source "$HOME/miniconda3/bin/activate"

if ! conda env list | grep -q "^xmgn "; then
    echo "==> Creating Python 3.10 environment 'xmgn' from conda-forge (no Anaconda ToS)..."
    conda create -n xmgn python=3.10 -c conda-forge --override-channels -y
else
    echo "==> Conda env 'xmgn' already exists"
fi

conda activate xmgn

# Conda-forge python may not ship the `pip` console script — use `python -m pip`
# throughout. Also ensure pip is actually present in the env.
PY="$HOME/miniconda3/envs/xmgn/bin/python"

# --- 3. Install xmgn requirements + physicsnemo ---
echo "==> Ensuring pip module is installed in the env..."
$PY -m ensurepip --upgrade 2>/dev/null || conda install -n xmgn -c conda-forge pip -y

echo "==> Upgrading pip..."
$PY -m pip install --upgrade pip --quiet

echo "==> Installing xmgn requirements (torch + PyG extensions for CUDA 12.1)..."
$PY -m pip install -r /mnt/e/NVIDIA/reservoir_simulation/xmgn/requirements.txt

echo "==> Installing nvidia-physicsnemo + its dependencies..."
$PY -m pip install nvidia-physicsnemo s3fs einops onnx timm treelib xarray zarr

# --- 4. Verify the stack works ---
echo "==> Verifying torch + CUDA + PyG extensions..."
python -c "
import torch
print(f'torch:       {torch.__version__}')
print(f'CUDA built:  {torch.version.cuda}')
print(f'CUDA avail:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:         {torch.cuda.get_device_name(0)}')

import torch_scatter, torch_sparse, torch_cluster
print(f'torch_scatter: {torch_scatter.__version__}')
print(f'torch_sparse:  {torch_sparse.__version__}')
print(f'torch_cluster: {torch_cluster.__version__}')

from physicsnemo.distributed import DistributedManager
print('physicsnemo: OK')
from physicsnemo.models.meshgraphnet import MeshGraphNet
print('MeshGraphNet: OK')
"

echo ""
echo "==> Setup complete. To activate the env:"
echo "    source ~/miniconda3/bin/activate xmgn"
