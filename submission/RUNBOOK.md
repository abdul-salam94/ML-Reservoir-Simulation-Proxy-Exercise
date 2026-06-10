# RUNBOOK — X-MeshGraphNet Norne Surrogate (Post-Doc Assessment)

Execution plan with a **verification gate** at every phase. Nothing is declared
"done" without the evidence line passing. Driven over SSH; commands run
inside `conda activate xmgn` on the RunPod RTX 4090 box.

Pre-computed seeded split (seed 42, confirm against `dataset_metadata.json`):
- **TEST (6):** NORNE_002, 008, 016, 018, 041, 048
- **VAL  (6):** NORNE_007, 009, 015, 044, 052, 057
- **TRAIN (48):** the remaining cases

---

## Phase 0 — Prep (local, DONE)
- [x] `conf/config_postdoc.yaml` — final config, mandated split, documented changes
- [x] `scripts/setup.sh` — turnkey env
- [x] This runbook + writeup skeleton

## Phase 1 — Box + environment
1. Fork `abdul-salam94/ML-Reservoir-Simulation-Proxy-Exercise` on GitHub (montaserramadan acct).
2. `bash scripts/setup.sh`
   - **GATE:** prints `All imports OK`, `CUDA available: True`, GPU = RTX 4090, torch 2.4.0.
3. Copy `config_postdoc.yaml` → `$REPO_DIR/xmgn/conf/`.
4. Get the dataset onto `/workspace/data/NORNE_LHS` (native ext4):
   - Path A: `rclone` SharePoint remote → pull the 30 GB bundle, extract.
   - Path B: download on PC → `scp`/`rsync` to box.
   - **GATE:** `ls /workspace/data/NORNE_LHS` shows 60 `NORNE_xxx/` folders, each with
     `.DATA .EGRID .INIT .UNRST .UNSMRY .SMSPEC`; a `.UNRST` is ~470 MB.
   - **GATE:** layout is **flat** (no nested `INCLUDE/` with stray `.DATA` — TROUBLESHOOTING glob trap).
     Use `tools/link_completed.py` if needed.
   - **GATE:** `df -h /workspace` shows ≥150 GB free.

## Phase 2 — Preprocess
5. **Smoke test (5 cases):** set `num_samples: 5` in config → `python src/preprocessor.py --config-name=config_postdoc`
   - **GATE:** completes all 5 steps (graphs→partition→split→stats→metadata), no exit 137.
6. Re-comment `num_samples`; **full preprocess** of 60 cases (~1–3 h).
   - **GATE:** `<sim_dir>.dataset/<job>/dataset_metadata.json` exists; split lists match the
     pre-computed assignment above. Record the actual assignment for the writeup.
7. `python tools/inspect_graphs.py ...`
   - **GATE:** node count ~44K/case, edges include NNCs, feature dims sane, no NaNs.

## Phase 3 — Train
8. Launch (detached, survives SSH drop):
   ```bash
   cd $REPO_DIR/xmgn
   RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
     nohup python -u src/train.py --config-name=config_postdoc > ~/train_run1.log 2>&1 &
   disown
   ```
   - **GATE:** loss decreasing in `train_run1.log`; MLflow logging; GPU util ~80–95% (`nvidia-smi`).
     If GPU ~0% → dataset on slow path; fix before wasting hours.
9. Monitor MLflow; let early-stopping/cosine finish. ~12–20 h.
   - **GATE:** `outputs/XMGN_Norne_PostDoc/best_checkpoints/` has a best `.pt` + `.mdlus`.
10. **Methodology experiment (run #2)** — ONE hypothesis-driven change (candidate: loss weights,
    or `prev_timesteps`, or hidden_dim). Documented in writeup with before/after val metrics.

## Phase 4 — Inference + metrics + visualization
11. `python src/inference.py --config-name=config_postdoc`
    - **GATE:** log shows `Processing case: NORNE_xxx (62 timesteps)` (autoregressive rollout),
      NOT 1-timestep-per-case (Patch 5 sanity).
    - **GATE:** every output HDF5 has `f.attrs["ordering"] == "natural"` (Patch 7 sanity).
12. `python tools/build_accuracy_matrix.py ...` → per-case + per-timestep RMSE/MAE.
    - Convert normalized → physical units (bar for PRESSURE, fraction for SWAT) using the
      stats in `global_stats.json`. Report BOTH in the writeup.
13. `python tools/pick_representative_cases.py ...` → best / median / **worst** test case.
14. Visualization on the worst test case:
    ```bash
    python tools/hdf5_to_unrst.py ...        # writes _PRED.UNRST (auto-clips SWAT, natural order)
    python tools/final_timestep_matrix.py ...# TRUE / PRED / DIFF panel figure
    ```
    - **GATE:** spatial pattern is physical (smooth water front, error concentrated near
      faults/wells), not scrambled splotches.

## Phase 5 — Writeup + PR
15. Fill `writeup/REPORT.md` (4–6 pages): setup notes, training (epochs/time/hardware/loss curves),
    test metrics (normalized + physical), failure-mode analysis + hypothesis, config changes +
    reasoning, FNO-vs-X-MGN bonus, "what next".
16. Export to PDF.
17. Commit to fork; open PR including: best checkpoint, `config_postdoc.yaml`, inference HDF5s,
    the TRUE/PRED/DIFF figure, and the writeup.
    - **GATE:** PR contains all 5 mandated deliverables (TASK.md §4).

---

### Standing safety rules
- Never run on a `/mnt/...` path. Never let disk drop below ~20 GB free.
- Keep a `wsl`/SSH heartbeat or `nohup`+`disown` for long runs.
- Back up `best_checkpoints/` + `global_stats.json` off-box once training converges
  (these are irreplaceable without re-training).
- Every metric reported in the writeup must trace to a file we can re-open.
