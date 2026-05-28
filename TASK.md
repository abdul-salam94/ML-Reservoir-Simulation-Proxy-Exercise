# Postdoc Technical Assessment — Neural Reservoir Surrogate (X-MeshGraphNet on Norne)

---

## 1. Overview

You'll train a **graph neural network surrogate** that predicts how physical
fields (pressure, water saturation) evolve through time inside a 3D faulted
reservoir. The framework is NVIDIA's [X-MeshGraphNet](https://arxiv.org/pdf/2411.17164)
(X-MGN) on PhysicsNeMo; the data is the **Norne field**, a real Norwegian-Sea
reservoir, simulated under many geological scenarios.

In ML terms: **autoregressive next-step prediction on an irregular 3D mesh**.
Nodes are grid cells, edges are inter-cell flow connections, and you predict each
cell's next-timestep state from its current state + neighborhood.

This mirrors a real commercial workflow — Stone Ridge Technology runs the same
class of surrogate on PhysicsNeMo at AWS scale for uncertainty quantification and
field optimization ([NVIDIA spotlight](https://developer.nvidia.com/blog/spotlight-stone-ridge-technology-accelerates-reservoir-simulation-workflows-with-nvidia-physicsnemo-on-aws/)).

---

## 2. Domain context (everything you need)

| Petroleum term | What it is, in ML terms |
|---|---|
| **Reservoir / grid** | A 3D structured grid (Norne: 46×112×22, ~44K *active* cells) = the graph's nodes |
| **Permeability (PERMX), porosity (PORV)** | Static per-cell scalars (rock properties) — node features. Permeability spans ~6 orders of magnitude (use log scale) |
| **Transmissibility (TRAN)** | Static per-*edge* scalar — how easily fluid flows between two cells. Edge weight |
| **Fault** | A geological fracture that impedes/enables flow. Norne has ~47, encoded as **Non-Neighbor Connections (NNCs)** — extra graph edges that break the regular grid structure. This is where graph nets beat structured-grid methods (CNNs/FNOs) |
| **MULTFLT (fault multiplier)** | A per-fault scalar (0=sealing, 1=baseline, >1=enhanced). **The uncertainty axis** — each simulation case varies these |
| **Pressure / SWAT** | The dynamic state you predict: pressure (bar) and water saturation (0-1) per cell, evolving over ~65 timesteps spanning ~9 years |
| **Case / realization** | One full simulation run with a particular set of fault multipliers. Each yields ~65 timesteps = ~62 training samples |

That's all the domain you need. The framework reads the binary simulation files
for you — you never write reservoir-specific code.

---

## 3. What you receive

```
xmgn/          X-MGN framework (fixes pre-applied — see PATCHES.md)
sim_utils/     Eclipse-format reader
tools/         LHS generator, batch runner, dataset-prep, graph inspector
norne_base/    The Norne reference deck (text inputs)
requirements.txt, SETUP.md, PATCHES.md, TROUBLESHOOTING.md
data/          (dataset obtained separately — see data/README.md)
```

The dataset is a set of Norne simulations (varying fault multipliers via Latin
Hypercube Sampling), already run through the Eclipse simulator. Each case
provides the binary outputs X-MGN consumes.

---

## 4. Your task

### Required deliverables

1. **A trained checkpoint** — your best X-MGN model from
   `outputs/<job>/best_checkpoints/`.
2. **A writeup** (~3-6 pages, PDF or Markdown):
   - Setup notes + any issues you hit
   - Training run: epochs, time, hardware, loss curves (MLflow screenshot fine)
   - **Test-set metrics** — RMSE/MAE for PRESSURE and SWAT, both normalized and
     in physical units (bar, saturation fraction), on your held-out test split
   - **At least one failure-mode analysis** — a case/region the model predicts
     poorly, with a hypothesis why
   - Any config changes you made and your reasoning
   - What you'd do next with more time/compute
3. **Your final config** (`conf/*.yaml`).
4. **One inference run** — `python src/inference.py --config-name=<your-config>`,
   submit the resulting `outputs/<job>/inference/` (HDF5 + GRDECL).

### Bonus (optional, not required)

- **FNO vs X-MGN**: in your writeup, discuss when a Fourier Neural Operator
  (structured-grid) would beat or lose to X-MGN (graph) for *faulted* reservoirs
  like Norne. (Hint: think about what NNCs do to grid regularity.)
- Stratify your error metrics by something meaningful and analyze the pattern.
- Propose one concrete architecture/loss/data improvement and justify it.

### Out of scope

- Implementing a model from scratch — use the provided X-MGN.
- Re-running the reservoir simulations — the data is provided.
- Hitting any specific accuracy number — we evaluate methodology, not leaderboard wins.

---

## 5. How we evaluate

| Criterion | Weight | What we look for |
|---|---|---|
| Pipeline correctness | 20% | Model trains, converges, produces sensible outputs |
| ML methodology | 25% | Sound validation strategy; documented hyperparameter choices; diagnostics |
| Held-out performance | 20% | Per-variable RMSE on the test split (physical units) |
| Failure analysis & insight | 20% | The most informative part of strong writeups — what breaks and why |
| Communication | 15% | Clarity, structure, brevity |

We are **not** comparing you to a target accuracy. We want to see how you
*engineer and reason about* a real surrogate-modeling pipeline.

---

## 6. Suggested workflow

```
Day 0   Environment (SETUP.md). Verify torch+CUDA+physicsnemo import.
Day 0   Skim xmgn/README.md, src/preprocessor.py, src/train.py.
Day 1   Preprocess a small subset (num_samples: 10) → sanity check.
Day 1   Full preprocessing; inspect graphs (tools/inspect_graphs.py).
Day 1-N Train; monitor with MLflow; iterate on hyperparameters.
Day N   Inference on test split; collect metrics; write up.
```

If you hit a blocker, check `TROUBLESHOOTING.md` first. If something is genuinely
broken on our end, email us. For research/design judgment calls ("should I do X
or Y?"), make the call and document your reasoning — that's part of what we
evaluate.

---

## 7. A note on what was pre-fixed for you

We pre-applied a handful of patches so you start from a working pipeline rather
than debugging framework version-skew (see `PATCHES.md`). These are infrastructure
fixes — the *ML* work is entirely yours. We mention them because reading
`PATCHES.md` is a quick lesson in the kind of "research code at scale" issues
you'll deal with on the team.

Good luck.
