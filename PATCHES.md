# Patches Applied to the NVIDIA X-MeshGraphNet Example

This package ships NVIDIA's `examples/reservoir_simulation/xmgn` (Apache-2.0)
with a small number of fixes pre-applied so the pipeline runs end-to-end on a
standard Linux/WSL2 + CUDA setup. Each patch is documented below with **what**
changed, **why**, and **the engineering lesson** — because understanding these
is part of the assessment.

> The upstream example is genuinely excellent ML, but — like much research code —
> it was demoed at smaller scale on the authors' own (Linux, big-GPU) environment.
> The gaps below only surface when you run it at real scale on different hardware.

---

## Patch 1 — PhysicsNeMo ≥ 1.3.0 module relocation

**Files**: `xmgn/src/train.py`, `xmgn/src/inference.py`

**What changed**: import paths updated from the old `physicsnemo.utils.*`
hierarchy to the new `physicsnemo.launch.*` hierarchy:

```python
# Before (xmgn as shipped — targets an older physicsnemo):
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.utils.logging.mlflow import initialize_mlflow
from physicsnemo.utils import load_checkpoint, save_checkpoint

# After (works with nvidia-physicsnemo==1.3.0):
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.launch.logging.mlflow import initialize_mlflow
from physicsnemo.launch.utils.checkpoint import load_checkpoint, save_checkpoint
```

**Why**: PhysicsNeMo was reorganized (post-Modulus-rebrand). The current pip
package `nvidia-physicsnemo==1.3.0` no longer has `physicsnemo.utils.logging`;
those modules moved under `physicsnemo.launch`. The xmgn example was written
against the older layout, so it fails at import with
`ModuleNotFoundError: No module named 'physicsnemo.utils.logging'`.

**Lesson**: framework version skew is a first-class hazard. Research examples
rarely pin the exact framework version they were written against, so an example
"that worked when published" breaks against the latest release. Always pin the
framework version (we pin `nvidia-physicsnemo==1.3.0` in `requirements.txt`).

---

## Patch 2 — Streaming statistics (memory scaling)

**File**: `xmgn/src/data/dataloader.py`, function `compute_global_statistics`

**What changed**: replaced the original implementation, which concatenated every
per-node feature from every graph into one giant tensor before computing
mean/std, with a **streaming Welford's algorithm** that maintains running
`(count, mean, M2)` accumulators and processes one graph at a time.

**Why**: the original does, in effect:

```python
all_node_features = [g.x for g in all_graphs]   # holds EVERYTHING in RAM
all_nodes = torch.cat(all_node_features, dim=0)  # one ~9 GB allocation for 60 cases
node_mean = all_nodes.mean(dim=0)
```

Memory scales **linearly with dataset size**:
- 60 cases  → ~9.2 GB  → OOMs a 16 GB box / WSL2 default RAM
- 500 cases → ~72 GB  → OOMs essentially any workstation

We observed a hard SIGKILL (OOM) at Step 3 on a 60-case dataset. The streaming
version uses O(num_features) memory (a few KB) regardless of dataset size, and
produces a **mathematically identical** result (Welford is exact, not an
approximation).

**Lesson**: the classic "research code doesn't scale" trap. Accumulating
everything in a list then concatenating is fine for a 10-case demo, catastrophic
at 500. Before scaling any pipeline, audit the memory complexity of each stage —
anything O(dataset_size) in RAM is a red flag. Welford / streaming reductions are
the standard fix.

---

## Patch 3 — Crash-resilient resume default

**File**: `xmgn/src/preprocessor.py`, function `check_existing_data`

**What changed**: in a non-interactive environment, the preprocessor now
defaults to **"use existing data, run only the missing steps"** instead of
**"overwrite everything from scratch"**.

```python
# Before: non-interactive auto-selects 'y' (overwrite all), redoing Steps 1-2
#         from scratch even after a crash in Step 3.
# After:  non-interactive auto-selects 'n' (keep completed steps), so re-running
#         after a crash resumes instead of restarting a ~40-minute job.
```

**Why**: preprocessing has 5 sequential steps (graph build → partition → split →
statistics → metadata). Steps 1-2 take ~40 min on a 60-case set. If Step 3
crashes (e.g., the OOM that Patch 2 fixes), the original would redo Steps 1-2 on
the next run. Defaulting to "use existing" lets the built-in resume logic skip
completed steps.

**Lesson**: long multi-stage pipelines should be idempotent and resumable.
"Start fresh" as the non-interactive default punishes every crash with a full
restart. Prefer "resume what's done" and make overwrite explicit.

---

## NOT applied here (Windows-only, intentionally excluded)

These fixes were needed when we briefly attempted native-Windows execution. They
are **not** in this package because the supported platform is Linux/WSL2, where
they're unnecessary — and one is actively harmful on Linux.

- **SimplePartition bypass** (`preprocessor.py`): on native Windows, PyG's
  `ClusterData`/METIS C-extension segfaults on Norne-scale graphs, so we forced
  the pure-Python `SimplePartition` fallback. On Linux this is **reverted** —
  Linux PyG's METIS works correctly and produces better-balanced partitions.
  (The upstream try/except METIS→SimplePartition fallback is left intact.)
- **`torch+cu121` explicit index** (install-time): PyPI's Windows torch wheel is
  CPU-only; Linux's is already CUDA. See `SETUP.md`.
- **`PYTHONUTF8=1`**: Windows stdio defaults to cp1252 and crashes on the `→`
  characters in xmgn's log messages. Linux defaults to UTF-8. See
  `TROUBLESHOOTING.md`.

---

## Summary table

| # | File | Change | Class | Applied on Linux? |
|---|------|--------|-------|-------------------|
| 1 | train.py, inference.py | physicsnemo.utils.* → physicsnemo.launch.* | Version compat | ✅ Yes |
| 2 | dataloader.py | torch.cat → streaming Welford | Memory scaling | ✅ Yes |
| 3 | preprocessor.py | non-interactive resume default | Pipeline robustness | ✅ Yes |
| — | preprocessor.py | SimplePartition bypass | Windows-only | ❌ No (reverted) |
