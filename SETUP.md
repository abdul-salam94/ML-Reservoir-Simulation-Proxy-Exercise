# Environment Setup

PhysicsNeMo officially supports **Ubuntu 24.04** (and Windows **only via WSL2** —
native Windows is not supported). This guide gets you from a fresh machine to a
working training environment.

> **TL;DR**: Linux or WSL2 Ubuntu + Python 3.10 + CUDA 12.1 GPU + the pinned
> `requirements.txt`. Keep the dataset on a native Linux filesystem, not a
> Windows-mounted `/mnt/` path.

---

## 1. Operating system

| Your machine | Do this |
|---|---|
| Native Linux (Ubuntu 24.04) | Use it directly |
| Windows 10/11 | Install **WSL2 + Ubuntu** (below). Do NOT run on native Windows |
| macOS | Not supported (no NVIDIA CUDA) — use a Linux GPU box or cloud |

### Windows → WSL2 Ubuntu

```powershell
# Install Ubuntu 24.04 into WSL2. --location puts the VM's virtual disk on a
# drive with plenty of space (NOT C:, which fills up fast — see note below).
# --no-launch avoids the interactive first-run that can hang in automation.
wsl --install -d Ubuntu-24.04 --no-launch --location D:\WSL\Ubuntu

# Then either launch the Ubuntu app once to set a username/password,
# or work as root:  wsl -d Ubuntu-24.04 --user root
```

> **⚠️ Why `--location` matters**: WSL stores the entire Linux filesystem in one
> growing `ext4.vhdx` file, by default on `C:`. Installing torch + PyG +
> physicsnemo (~10 GB) plus the dataset can fill `C:` and put the Linux
> filesystem into a read-only state mid-install. Put the vdisk on a drive with
> 100+ GB free.

### Keep WSL alive for long training jobs

WSL2 shuts down the VM after `vmIdleTimeout` ms with no foreground `wsl.exe`
process — which kills background training. Create `C:\Users\<you>\.wslconfig`:

```ini
[wsl2]
# 500 hours — effectively never auto-shutdown during multi-day training
vmIdleTimeout=1800000000
```

Then `wsl --shutdown` once to apply. (Takes effect on next start.)

---

## 2. Python 3.10

PhysicsNeMo + the pinned `torch==2.4.0` need **Python 3.10 or 3.11** (torch 2.4
has no wheels for 3.12+). Ubuntu 24.04 ships 3.12, so install 3.10 explicitly.

### Option A — Miniforge / conda (no sudo, recommended)

```bash
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o mc.sh
bash mc.sh -b -p ~/miniconda3 && rm mc.sh
source ~/miniconda3/bin/activate
# Use conda-forge to avoid Anaconda channel Terms-of-Service prompts:
conda create -n xmgn python=3.10 -c conda-forge --override-channels -y
conda activate xmgn
```

### Option B — deadsnakes PPA + venv (needs sudo)

```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-dev
python3.10 -m venv ~/xmgn-env && source ~/xmgn-env/bin/activate
```

---

## 3. Install dependencies

```bash
# From the repo root, with your Python 3.10 env active:
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux, `torch==2.4.0` from PyPI is already the CUDA 12.1 build — no special
index URL needed. (On native Windows it would be CPU-only; another reason to use
WSL2.)

---

## 4. Verify the stack

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
import torch_scatter, torch_sparse, torch_cluster   # compiled extensions
from physicsnemo.distributed import DistributedManager
from physicsnemo.models.meshgraphnet import MeshGraphNet
print("All imports OK")
PY
```

Expected:
```
torch: 2.4.0+cu121 | CUDA: True
GPU: NVIDIA GeForce ...
All imports OK
```

If `torch.cuda.is_available()` is False, your GPU/driver isn't visible to the
env — check `nvidia-smi` works inside WSL/Linux first.

---

## 5. Dataset placement (performance-critical)

Put the preprocessed dataset on a **native Linux filesystem** (e.g.
`~/dataset/` or `/root/...`), NOT on a Windows-mounted `/mnt/d/...` path.

> **Why**: the WSL 9p filesystem bridge to Windows drives adds ~50-150 ms latency
> per file open. Training reads thousands of small partition files, so 9p starves
> the GPU (we measured 0% GPU utilization reading from `/mnt/d`, vs 90% from
> native ext4 — a ~5-10× throughput difference).

See `data/README.md` for how to obtain and place the dataset.

---

## 6. Run the pipeline

```bash
cd xmgn
# Preprocess simulation outputs → partitioned graphs
python src/preprocessor.py --config-name=<your-config>
# Train (single GPU). Set distributed env vars for the DistributedManager:
RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
    python src/train.py --config-name=<your-config>
# Inference on the test split
python src/inference.py --config-name=<your-config>
```

For long runs, detach so they survive terminal/session changes:

```bash
nohup python -u src/train.py --config-name=<your-config> > ~/train.log 2>&1 &
disown
```

See `TROUBLESHOOTING.md` if anything misbehaves.
