"""Generate Figure 1: Precision-Recall Trade-off scatter plot (9 models + baseline)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

# Data – all 9 models + rule-based baseline
# recall_plot / precision_plot are used for display (with jitter for overlapping points).
# The true values are in the comments.
systems = {
    "Rule-based":        {"precision": 100.0, "recall": 52.8,  "marker": "s", "color": "#888888",  "size": 140},
    "Claude Sonnet 4":   {"precision": 100.0, "recall": 96.9,  "marker": "*", "color": "#2166AC",  "size": 260},
    "Gemma 3 27B":       {"precision": 78.6,  "recall": 50.8,  "marker": "d", "color": "#B8860B",  "size": 140},
    "Qwen 3 235B":       {"precision": 65.6,  "recall": 93.8,  "marker": "v", "color": "#984EA3",  "size": 140},
    "GPT-4.1":           {"precision": 61.3,  "recall": 100.0, "marker": "D", "color": "#4DAF4A",  "size": 130},
    "DeepSeek R1":       {"precision": 61.0,  "recall": 98.5,  "marker": "8", "color": "#CD853F",  "size": 130},
    "Llama 4 Maverick":  {"precision": 66.7,  "recall": 67.7,  "marker": "h", "color": "#A0522D",  "size": 180},
    "Llama 4 Scout":     {"precision": 60.9,  "recall": 86.2,  "marker": ">", "color": "#D2691E",  "size": 130},
    "Claude Opus 4.6":   {"precision": 60.2,  "recall": 100.0, "marker": "X", "color": "#6BAED6",  "size": 140},
    "Claude Sonnet 4.6": {"precision": 60.2,  "recall": 100.0, "marker": "P", "color": "#4A90D9",  "size": 140},
    "DeepSeek V3.2":     {"precision": 60.2,  "recall": 100.0, "marker": "H", "color": "#8B4513",  "size": 140},
    "MiniMax M2.5":      {"precision": 58.9,  "recall": 100.0, "marker": "^", "color": "#E41A1C",  "size": 130},
    "Gemini 2.5 Pro":    {"precision": 58.8,  "recall": 100.0, "marker": "p", "color": "#FF7F00",  "size": 140},
}

# Apply small jitter to overlapping models so they don't stack.
jitter = {
    "GPT-4.1":           (0.0,   +0.6),
    "DeepSeek R1":       (-0.8,  -0.6),
    "Claude Opus 4.6":   (-0.9,  +0.5),
    "Claude Sonnet 4.6": (+0.9,  +0.5),
    "DeepSeek V3.2":     (0.0,   -0.5),
    "MiniMax M2.5":      (-0.8,  0.0),
    "Gemini 2.5 Pro":    (+0.8,  0.0),
}

# Style (matching existing paper style)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.linewidth": 0.8,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
})

fig, ax = plt.subplots(figsize=(8.0, 6.5))

# Random classifier baseline (precision = class prevalence ~50%)
prevalence = 50.0
ax.axhline(y=prevalence, color="#CCCCCC", linestyle="--", linewidth=1.0, zorder=1)
ax.text(37, prevalence + 1.2, "Random classifier", fontsize=8.5, color="#999999", style="italic")

# Shaded region: "Error Detection Bias Zone" – encloses the 5+1 biased models at
# recall ~100, precision 57-63.  Use a rectangle with rounded corners.
bias_rect = FancyBboxPatch(
    (96.0, 55.5), 8.0, 9.0,
    boxstyle="round,pad=0.8", fc="#FFE0E0", ec="#CC4444",
    alpha=0.35, linewidth=0.8, linestyle="--", zorder=0,
)
ax.add_patch(bias_rect)
# Bias zone label
ax.annotate(
    "Error detection\nbias zone",
    xy=(98.5, 56.0),
    xytext=(82.0, 46.0),
    fontsize=8.5, color="#CC4444", style="italic", ha="center",
    bbox=dict(boxstyle="round,pad=0.25", fc="#FFE0E0", ec="none", alpha=0.6),
    arrowprops=dict(arrowstyle="->", color="#CC4444", lw=0.8),
)

# Plot each system (with jitter for overlapping points)
for name, d in systems.items():
    dx, dy = jitter.get(name, (0.0, 0.0))
    ax.scatter(
        d["recall"] + dx, d["precision"] + dy,
        marker=d["marker"], s=d["size"], c=d["color"],
        edgecolors="black", linewidths=0.6, zorder=3, label=name,
    )

# Label positions – carefully arranged to avoid overlap.
label_positions = {
    "Rule-based":        {"xytext": (37.0, 96.0),  "ha": "left"},
    "Claude Sonnet 4":   {"xytext": (76.0, 105.0), "ha": "left"},
    "Gemma 3 27B":       {"xytext": (37.0, 82.0),  "ha": "left"},
    "Qwen 3 235B":       {"xytext": (76.0, 69.0),  "ha": "left"},
    "Llama 4 Maverick":  {"xytext": (48.0, 72.0),  "ha": "left"},
    "Llama 4 Scout":     {"xytext": (68.0, 53.0),  "ha": "left"},
    "GPT-4.1":           {"xytext": (37.0, 65.0),  "ha": "left"},
    "DeepSeek R1":       {"xytext": (37.0, 59.0),  "ha": "left"},
    "Claude Opus 4.6":   {"xytext": (37.0, 76.0),  "ha": "left"},
    "Claude Sonnet 4.6": {"xytext": (37.0, 53.0),  "ha": "left"},
    "DeepSeek V3.2":     {"xytext": (37.0, 47.0),  "ha": "left"},
    "MiniMax M2.5":      {"xytext": (76.0, 47.0),  "ha": "left"},
    "Gemini 2.5 Pro":    {"xytext": (76.0, 53.0),  "ha": "left"},
}
for name, d in systems.items():
    cfg = label_positions[name]
    dx, dy = jitter.get(name, (0.0, 0.0))
    ax.annotate(
        name,
        xy=(d["recall"] + dx, d["precision"] + dy),
        xytext=cfg["xytext"],
        textcoords="data",
        fontsize=8.5,
        ha=cfg["ha"],
        arrowprops=dict(arrowstyle="-", color="#999999", lw=0.5),
    )

ax.set_xlabel("Recall (%)")
ax.set_ylabel("Precision (%)")
ax.set_xlim(35, 108)
ax.set_ylim(38, 110)
ax.set_xticks(np.arange(40, 110, 10))
ax.set_yticks(np.arange(50, 110, 10))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, alpha=0.25, linewidth=0.5)

ax.legend(
    loc="upper left", frameon=True, fancybox=False,
    edgecolor="#CCCCCC", framealpha=0.95,
    ncol=2, fontsize=7.5,
    bbox_to_anchor=(0.0, 1.0),
)

fig.tight_layout(pad=0.5)
fig.savefig("/Users/silupanda/Downloads/finverification-bench/paper/figures/figure_precision_recall.pdf",
            bbox_inches="tight", pad_inches=0.05)
plt.close(fig)
print("Saved figure_precision_recall.pdf")
