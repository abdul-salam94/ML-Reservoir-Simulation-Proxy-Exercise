"""Final artifact verification for the X-MGN Norne submission.
Checks every required deliverable against the gates in TASK.md / PATCHES.md /
TROUBLESHOOTING.md. Pure read-only; runs on the local backup copies.
"""
import os, glob, json
import h5py
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
INF = os.path.join(ROOT, "inference")
CKPT = os.path.join(ROOT, "best_checkpoints")
TEST = ["NORNE_002", "NORNE_008", "NORNE_016", "NORNE_018", "NORNE_041", "NORNE_048"]

ok = []
bad = []
def check(name, cond, detail=""):
    (ok if cond else bad).append(f"{'PASS' if cond else 'FAIL'} | {name} {detail}")

# 1) checkpoint + stats present
check("checkpoint .mdlus present", os.path.exists(os.path.join(CKPT, "MeshGraphNet.0.150.mdlus")))
check("checkpoint .pt present", os.path.exists(os.path.join(CKPT, "checkpoint.0.150.pt")))
stats_p = os.path.join(CKPT, "global_stats.json")
check("global_stats.json present", os.path.exists(stats_p))

# 2) the 6 test HDF5s, gates from PATCHES 5 & 7
files = sorted(glob.glob(os.path.join(INF, "NORNE_*.hdf5")))
check("6 test HDF5 files", len(files) == 6, f"(found {len(files)})")
got_cases = []
for fp in files:
    with h5py.File(fp, "r") as f:
        case = str(f.attrs["case_name"]); got_cases.append(case)
        ordering = str(f.attrs.get("ordering", "MISSING"))
        # PATCH 7 gate
        check(f"{case}: ordering==natural", ordering == "natural", f"(got '{ordering}')")
        gP = f["predictions"]["PRESSURE"]; tP = f["targets"]["PRESSURE"]
        gS = f["predictions"]["SWAT"];     tS = f["targets"]["SWAT"]
        ks = sorted(gP.keys())
        # PATCH 5 gate: full autoregressive rollout, not 1-step
        check(f"{case}: 62 timesteps", len(ks) == 62, f"(got {len(ks)})")
        nodes = gP[ks[0]].shape[0]
        check(f"{case}: 44431 active cells", nodes == 44431, f"(got {nodes})")
        # non-degenerate, physical-range guards (TROUBLESHOOTING all-zero bug)
        p_last = gP[ks[-1]][:].astype("f8")
        s_last = gS[ks[-1]][:].astype("f8")
        check(f"{case}: PRESSURE pred non-zero", np.abs(p_last).mean() > 1.0,
              f"(mean|P|={np.abs(p_last).mean():.1f} bar)")
        check(f"{case}: PRESSURE in physical range", 50 < p_last.mean() < 600,
              f"(meanP={p_last.mean():.1f} bar)")
        # SWAT: a tiny unconstrained-head overshoot (~1.001) is EXPECTED and documented
        # (TROUBLESHOOTING.md); hdf5_to_unrst.py clips to [0, 1-SGAS] at the UNRST stage.
        check(f"{case}: SWAT in physical range (small overshoot OK)",
              s_last.min() >= -0.01 and s_last.max() <= 1.02,
              f"(min={s_last.min():.4f} max={s_last.max():.4f})")
        # accuracy vs target (denormalized)
        seP = seS = n = 0.0
        for k in ks:
            dP = gP[k][:].astype("f8") - tP[k][:].astype("f8"); seP += (dP**2).sum()
            dS = gS[k][:].astype("f8") - tS[k][:].astype("f8"); seS += (dS**2).sum()
            n += dP.size
        rmseP = (seP/n)**0.5; rmseS = (seS/n)**0.5
        check(f"{case}: PRESSURE RMSE sane (<25 bar)", rmseP < 25, f"(={rmseP:.2f} bar)")
        check(f"{case}: SWAT RMSE sane (<0.05)", rmseS < 0.05, f"(={rmseS:.4f})")

check("test cases == mandated seed-42 set", sorted(got_cases) == sorted(TEST),
      f"(got {sorted(got_cases)})")

# 3) supporting artifacts
for fn in ["per_case_metrics.csv", "error_vs_timestep.csv", "summary_matrix.txt",
           "final_timestep_matrix.png", "representative_cases.json",
           "heatmap_PRESSURE.png", "heatmap_SWAT.png"]:
    check(f"artifact: {fn}", os.path.exists(os.path.join(INF, fn)))

print("\n".join(ok))
print("\n".join(bad) if bad else "\n--- ALL CHECKS PASSED ---")
print(f"\nSUMMARY: {len(ok)} passed, {len(bad)} failed")
