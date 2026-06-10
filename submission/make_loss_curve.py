"""Plot train/val loss + LR vs epoch from the run-1 training log."""
import re, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = os.path.join(os.path.dirname(__file__), "backup_checkpoints", "train_run1.log")
OUT = os.path.join(os.path.dirname(__file__), "inference", "loss_curve.png")

tr_ep, tr_loss, lr_ep, lr = [], [], [], []
va_ep, va_loss = [], []
for line in open(LOG, encoding="utf-8", errors="ignore"):
    m = re.search(r"\[train\].*Epoch (\d+) Metrics\s*:\s*train_loss\s*=\s*([\d.eE+-]+).*learning_rate\s*=\s*([\d.eE+-]+)", line)
    if m:
        e = int(m.group(1)); tr_ep.append(e); tr_loss.append(float(m.group(2)))
        lr_ep.append(e); lr.append(float(m.group(3)))
        continue
    m = re.search(r"\[valid\].*Epoch (\d+) Metrics\s*:\s*val_loss\s*=\s*([\d.eE+-]+)", line)
    if m:
        va_ep.append(int(m.group(1))); va_loss.append(float(m.group(2)))

fig, ax1 = plt.subplots(figsize=(7.0, 4.2))
ax1.plot(tr_ep, tr_loss, color="#1f77b4", lw=1.4, label="train loss")
ax1.plot(va_ep, va_loss, color="#d62728", lw=1.4, marker="o", ms=3, label="val loss")
ax1.set_yscale("log")
ax1.set_xlabel("epoch")
ax1.set_ylabel("loss (normalized, log scale)")
ax1.grid(True, which="both", ls=":", alpha=0.4)

ax2 = ax1.twinx()
ax2.plot(lr_ep, lr, color="#2ca02c", lw=1.0, ls="--", alpha=0.7, label="learning rate")
ax2.set_yscale("log")
ax2.set_ylabel("learning rate (log scale)", color="#2ca02c")
ax2.tick_params(axis="y", labelcolor="#2ca02c")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
ax1.set_title("Run-1 training: train/val loss and cosine LR schedule (150 epochs)", fontsize=10)
fig.tight_layout()
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print("wrote", OUT)
print(f"train epochs: {len(tr_ep)}  val points: {len(va_ep)}")
print(f"final train_loss={tr_loss[-1]:.3e}  best val_loss={min(va_loss):.3e}  final LR={lr[-1]:.2e}")
