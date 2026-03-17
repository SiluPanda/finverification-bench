"""Generate Figure 2: Model Comparison grouped bar chart (9 models + baseline)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Data: Accuracy, Precision, Recall, F1, 100-FPR (Specificity)
metrics = ["Accuracy", "Precision", "Recall", "F1", "Specificity\n(100\u2013FPR)"]
systems = {
    "Rule-based":        [53.8, 100.0, 52.8, 69.1,  100.0],
    "Claude Sonnet 4":   [98.1, 100.0, 96.9, 98.4,  100.0],
    "Gemma 3 27B":       [62.0, 78.6,  50.8, 61.7,  79.1],
    "Qwen 3 235B":       [66.7, 65.6,  93.8, 77.2,  25.6],
    "GPT-4.1":           [62.0, 61.3,  100.0, 76.0,  4.7],
    "DeepSeek R1":       [61.1, 61.0,  98.5, 75.3,  4.7],
    "Llama 4 Maverick":  [60.2, 66.7,  67.7, 67.2,  48.8],
    "Llama 4 Scout":     [58.3, 60.9,  86.2, 71.3,  16.3],
    "Claude Opus 4.6":   [60.2, 60.2,  100.0, 75.1,  0.0],
    "Claude Sonnet 4.6": [60.2, 60.2,  100.0, 75.1,  0.0],
    "DeepSeek V3.2":     [60.2, 60.2,  100.0, 75.1,  0.0],
    "MiniMax M2.5":      [58.9, 58.9,  100.0, 74.1,  0.0],
    "Gemini 2.5 Pro":    [58.8, 58.8,  100.0, 74.1,  0.0],
}

colors = {
    "Rule-based":        "#888888",
    "Claude Sonnet 4":   "#2166AC",
    "Gemma 3 27B":       "#B8860B",
    "Qwen 3 235B":       "#984EA3",
    "GPT-4.1":           "#4DAF4A",
    "DeepSeek R1":       "#CD853F",
    "Llama 4 Maverick":  "#A0522D",
    "Llama 4 Scout":     "#D2691E",
    "Claude Opus 4.6":   "#6BAED6",
    "Claude Sonnet 4.6": "#4A90D9",
    "DeepSeek V3.2":     "#8B4513",
    "MiniMax M2.5":      "#E41A1C",
    "Gemini 2.5 Pro":    "#FF7F00",
}

# Style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.linewidth": 0.8,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})

n_metrics = len(metrics)
n_systems = len(systems)
bar_width = 0.062
x = np.arange(n_metrics)

fig, ax = plt.subplots(figsize=(12.0, 5.5))

for i, (name, vals) in enumerate(systems.items()):
    offset = (i - (n_systems - 1) / 2) * bar_width
    bars = ax.bar(
        x + offset, vals, bar_width,
        label=name, color=colors[name],
        edgecolor="white", linewidth=0.3, zorder=3,
    )
    # Value labels – only show on bars tall enough to read; skip 0-height bars
    for bar, val in zip(bars, vals):
        if val < 1.0:
            # Place "0" just above the baseline so they don't overlap
            ax.text(
                bar.get_x() + bar.get_width() / 2, 2.0,
                "0",
                ha="center", va="bottom", fontsize=5.5, rotation=90, color="#666666",
            )
        else:
            y_pos = bar.get_height() + 0.8
            ax.text(
                bar.get_x() + bar.get_width() / 2, y_pos,
                f"{val:.0f}" if val == int(val) else f"{val:.1f}",
                ha="center", va="bottom", fontsize=5.5, rotation=90,
            )

ax.set_ylabel("Score (%)")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 120)
ax.set_yticks(np.arange(0, 110, 20))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.25, linewidth=0.5, zorder=0)

ax.legend(
    loc="upper center", bbox_to_anchor=(0.5, 1.22),
    ncol=5, frameon=True, fancybox=False,
    edgecolor="#CCCCCC", framealpha=0.95,
    columnspacing=0.6, handletextpad=0.3,
)

fig.tight_layout(pad=0.5)
fig.savefig("/Users/silupanda/Downloads/finverification-bench/paper/figures/figure_model_comparison_updated.pdf",
            bbox_inches="tight", pad_inches=0.05)
plt.close(fig)
print("Saved figure_model_comparison_updated.pdf")
