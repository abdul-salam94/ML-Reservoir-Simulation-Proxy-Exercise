# Dataset

The simulation dataset is **not** stored in git (the binary outputs are tens to
hundreds of GB). It's delivered separately.

## What the dataset is

Norne field simulations (grid 46×112×22, ~44K active cells), each a distinct
realization with **fault transmissibility multipliers** varied via Latin
Hypercube Sampling. Each case was run through the Eclipse simulator and provides:

```
NORNE_<id>/
├── NORNE_<id>.DATA      # simulation deck
├── NORNE_<id>.EGRID     # grid geometry (binary)
├── NORNE_<id>.INIT      # static properties: PERMX, PORV, TRAN... (binary)
├── NORNE_<id>.UNRST     # dynamic state per timestep: PRESSURE, SWAT... (~470 MB)
├── NORNE_<id>.UNSMRY    # well/field summary time series (binary)
└── NORNE_<id>.SMSPEC    # summary metadata (binary)
```

~65 timesteps per case spanning ~9 years of simulated production.

## How to obtain it

> **TODO (instructor)**: choose and document the delivery method, e.g.:
> - [ ] Download link / shared cloud bucket: `<URL>`
> - [ ] Institutional shared drive: `<path>`
> - [ ] Physical drive shipped on request
> - [ ] Subset (60 cases, ~32 GB) vs full (500 cases, ~250 GB)

Until then, the dataset can be **regenerated from scratch** (see below).

## Placement (important for performance)

Put the cases on a **native Linux filesystem**, not a Windows-mounted `/mnt/`
path. Then point your config at them:

```yaml
# conf/<your-config>.yaml
dataset:
  sim_dir: /home/<you>/data/NORNE_LHS    # native ext4, NOT /mnt/d/...
```

For a flat, glob-safe layout (one folder per case, no nested `INCLUDE/`), use
`tools/link_completed.py` as a template. See `TROUBLESHOOTING.md` for why the
flat layout and native filesystem matter.

## Regenerating the dataset from scratch

If you have an Eclipse (or OPM Flow) license, you can generate cases yourself:

```bash
# 1. Generate N LHS-sampled cases from the Norne base deck
python tools/generate_lhs_cases.py --n 500 --output_dir ../dataset/NORNE_LHS \
    --base ../norne_base --seed 42

# 2. Run them through Eclipse (resumable, parallel)
python tools/run_eclipse_batch.py --dataset_dir ../dataset/NORNE_LHS --parallel 4
```

The LHS design (which fault multipliers per case) is written to
`lhs_design.csv` for reproducibility — fixed seed gives identical samples.

> Generating 500 cases takes ~1-2 days of Eclipse compute on a 4-parallel
> workstation and ~250 GB of disk. The 60-case subset is enough for a meaningful
> pipeline test.
