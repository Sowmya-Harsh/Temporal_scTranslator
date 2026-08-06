"""
12_plot_loss_curves.py

Purpose: Parse the training log text you already have (the exact console
output format confirmed from your paste: "Epoch  NNN: train_loss=X.XXXX
val_loss=X.XXXX lr=X.XXXXXX") for each of the three architectures (MLP,
FiLM-pre, FiLM-post), and plot overlaid train/val loss curves.

Since these logs are only printed to console (not saved as CSV per your
answer), save each run's full console output to a .txt file first, then
point this script at all three files.

Usage:
    # 1. Save each run's console log to a file, e.g.:
    #    (if you still have the output in your terminal scrollback or a
    #    SLURM .out file, just paste/copy it into these three files)
    #    mlp_log.txt, film_pre_log.txt, film_post_log.txt

    python 12_plot_loss_curves.py \
        --logs mlp_log.txt film_pre_log.txt film_post_log.txt \
        --labels "MLP" "FiLM-pre" "FiLM-post" \
        --out loss_curves.png
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPOCH_LINE_RE = re.compile(
    r"Epoch\s+(\d+):\s+train_loss=([\d.]+)\s+val_loss=([\d.]+)"
)


def parse_log(path: Path):
    epochs, train_losses, val_losses = [], [], []
    with open(path) as f:
        for line in f:
            m = EPOCH_LINE_RE.search(line)
            if m:
                epochs.append(int(m.group(1)))
                train_losses.append(float(m.group(2)))
                val_losses.append(float(m.group(3)))
    if not epochs:
        raise ValueError(
            f"No epoch lines matched in {path}. Check that the file contains lines like "
            f"'Epoch    1: train_loss=0.2706 val_loss=0.0050 lr=0.001000' -- "
            f"if your log format differs, adjust EPOCH_LINE_RE."
        )
    return epochs, train_losses, val_losses


def main(log_paths, labels, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    for i, (log_path, label) in enumerate(zip(log_paths, labels)):
        epochs, train_losses, val_losses = parse_log(Path(log_path))
        color = colors[i % len(colors)]
        axes[0].plot(epochs, train_losses, label=label, color=color)
        axes[1].plot(epochs, val_losses, label=label, color=color)
        print(f"{label}: {len(epochs)} epoch entries parsed, "
              f"final train_loss={train_losses[-1]:.4f}, final val_loss={val_losses[-1]:.4f}")

    for ax, title in zip(axes, ["Training loss", "Validation loss"]):
        ax.set_xlabel("Epoch", color="black")
        ax.set_ylabel("Loss", color="black")
        ax.set_title(title, color="black")
        ax.tick_params(axis="x", colors="black")
        ax.tick_params(axis="y", colors="black")
        for spine in ax.spines.values():
            spine.set_color("black")
        legend = ax.legend()
        for text in legend.get_texts():
            text.set_color("black")

    fig.suptitle("Training and validation loss curves by architecture", color="black", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved figure to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=str, nargs="+", required=True)
    parser.add_argument("--labels", type=str, nargs="+", required=True)
    parser.add_argument("--out", type=str, default="loss_curves.png")
    args = parser.parse_args()
    if len(args.logs) != len(args.labels):
        raise ValueError("--logs and --labels must have the same length")
    main(args.logs, args.labels, args.out)
