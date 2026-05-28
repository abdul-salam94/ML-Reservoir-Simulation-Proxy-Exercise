# Troubleshooting

The code-level bugs we found are already patched (see `PATCHES.md`). What remains
are mostly **environment** gotchas. Each entry is **Symptom → Cause → Fix**.

---

### Symptom: `ModuleNotFoundError: No module named 'torch'` / "No matching distribution for torch==2.4.0"
**Cause**: wrong Python version. torch 2.4.0 has no wheels for Python 3.12+ (Ubuntu 24.04's default).
**Fix**: use Python 3.10 (see `SETUP.md` §2).

---

### Symptom: `torch.cuda.is_available()` returns `False`
**Cause (Linux/WSL)**: GPU not visible to the env, or you installed the CPU torch build.
**Fix**:
- Confirm `nvidia-smi` works inside WSL/Linux first.
- On Linux, `pip install torch==2.4.0` is already cu121. If you somehow got `+cpu`, reinstall: `pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu121 torch==2.4.0 torchaudio==2.4.0 torchvision==0.19.0`.

---

### Symptom: `FileNotFoundError: _scatter_cuda.pyd` / PyG extensions fail to import
**Cause**: torch and the PyG extensions (`torch-scatter` etc.) were built for different torch/CUDA versions. The extensions are ABI-locked to torch 2.4.0+cu121.
**Fix**: ensure torch is `2.4.0+cu121` (not `+cpu`), then reinstall the extensions from the matching index in `requirements.txt`.

---

### Symptom: log output crashes with `UnicodeEncodeError: 'charmap' codec can't encode character '→'`
**Cause**: Windows-native stdio defaults to cp1252; xmgn's logger uses `→` characters. (Should not happen on Linux/WSL.)
**Fix**: `export PYTHONUTF8=1` before running, or just use WSL2 (UTF-8 by default).

---

### Symptom: preprocessing/training process killed with no Python traceback (exit 137 / SIGKILL)
**Cause**: out-of-memory. The streaming-statistics patch (PATCHES.md #2) fixes the known Step-3 OOM, but very large datasets or low WSL RAM can still hit limits.
**Fix**:
- Check available RAM: `free -h` inside WSL. WSL2 defaults to ~50% of host RAM; raise it in `.wslconfig` with `memory=24GB` if needed.
- Reduce `num_partitions` or `hidden_dim` in the config to lower peak memory.

---

### Symptom: GPU utilization stuck near 0% during training, training crawls
**Cause**: dataset is on a Windows-mounted `/mnt/d/...` path — the WSL 9p filesystem starves the dataloader.
**Fix**: copy the preprocessed dataset to a native Linux filesystem (`~/dataset/...`) and point `sim_dir` there. See `SETUP.md` §5. We measured 0% → 90% GPU after this.

---

### Symptom: after moving the dataset, training still reads from the old location
**Cause**: `dataset_metadata.json` (written by the preprocessor) stores **absolute paths**. Moving the files doesn't update it.
**Fix**: rewrite the paths in `<dataset>.dataset/<job>/dataset_metadata.json`, e.g. `sed -i 's|/old/path|/new/path|g' dataset_metadata.json`. Or re-run preprocessing from the new location.

---

### Symptom: training dies after a few hours with no error, WSL distro shows "Stopped"
**Cause**: WSL2 idle-timeout killed the VM (no foreground `wsl.exe` for `vmIdleTimeout` ms).
**Fix**: set `vmIdleTimeout=1800000000` in `.wslconfig` (SETUP §1) and `wsl --shutdown` to apply. For a current run, keep any `wsl` terminal open as a heartbeat.

---

### Symptom: `wsl` commands hang indefinitely; even `wsl --terminate` doesn't return
**Cause**: a stuck WSL session from a prior interrupted command holds locks.
**Fix** (PowerShell): `Get-Process wsl,wslhost,wslservice | Stop-Process -Force; wsl --shutdown`, then retry.

---

### Symptom: preprocessing errors like `FileNotFoundError: .../INCLUDE/SUMMARY/summary.EGRID`
**Cause**: xmgn's case discovery uses a recursive `**/*.DATA` glob. If your dataset directory contains Eclipse `INCLUDE/` folders (which have their own `.DATA`-suffixed sub-files), the glob picks them up as bogus "cases".
**Fix**: lay out the dataset *flat* — one folder per case with only the case-level output files (`.DATA`, `.EGRID`, `.INIT`, `.UNRST`, `.UNSMRY`, `.SMSPEC`), no `INCLUDE/` subtree. See `tools/link_completed.py` for an example that builds this layout.

---

### Symptom: a "completed" simulation case has a truncated/partial `.UNRST` (much smaller than peers)
**Cause**: the simulation was interrupted mid-write (e.g., power/PC restart). A naive "file exists" resume check counts it as done.
**Fix**: verify completeness by file size (a complete Norne `.UNRST` is ~470 MB) or by parsing the final timestep. Delete the partial outputs (keep `.DATA` + `INCLUDE/`) and re-run that case. `tools/rerun_partial_cases.ps1` shows the pattern.

---

### Symptom: conda `create` fails with `CondaToSNonInteractiveError: Terms of Service have not been accepted`
**Cause**: Anaconda's default channels now require ToS acceptance (2025 policy).
**Fix**: use conda-forge: `conda create -n xmgn python=3.10 -c conda-forge --override-channels -y`.
