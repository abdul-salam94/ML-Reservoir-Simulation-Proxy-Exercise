# Submission — X-MeshGraphNet Norne Surrogate

Candidate: **Dr. Montaser Ramadan**

This folder contains the deliverables for the technical assessment (TASK.md §4).

## Deliverables

| # | Deliverable | Location |
|---|---|---|
| 1 | Trained checkpoint (best) | `best_checkpoints/MeshGraphNet.0.150.mdlus` (+ `checkpoint.0.150.pt`, `global_stats.json`) |
| 2 | Writeup (4–6 pp) | `writeup/REPORT.md` and `writeup/REPORT.pdf` |
| 3 | Final config | `../xmgn/conf/config_postdoc.yaml` |
| 4 | Inference output (HDF5) | `inference/NORNE_0*.hdf5` (6 held-out test cases) |
| 5 | TRUE/PRED/DIFF visualization | `inference/final_timestep_matrix.png` |

## Supporting artifacts
- `inference/per_case_metrics.csv`, `inference/error_vs_timestep.csv`, `inference/summary_matrix.txt` — accuracy matrix (physical units) + autoregressive-drift curve.
- `inference/representative_cases.json` — best/median/worst test cases.
- `inference/heatmap_PRESSURE.png`, `inference/heatmap_SWAT.png` — per-case error heatmaps.
- `verify_artifacts.py` — read-only checker that re-validates every gate (ordering, timestep count, active-cell count, physical ranges, mandated split). Run: `python submission/verify_artifacts.py`.
- `RUNBOOK.md` — phased execution plan with verification gates.

## Headline results (held-out test split, seed 42: NORNE_002/008/016/018/041/048)

| Regime | PRESSURE RMSE | SWAT RMSE |
|---|---|---|
| Single-step (validation) | 0.89 bar | 0.0029 |
| Autoregressive 62-step (test) | 9.27 bar (≈3.3 % of ~278 bar) | 0.0128 |

See `writeup/REPORT.md` for methodology, failure-mode analysis, and the FNO-vs-X-MGN discussion.
