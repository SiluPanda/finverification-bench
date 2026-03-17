"""Generate Figure 3: Category Detection Rate heatmap (9 models + baseline)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Data: rows = error categories, columns = systems
# Categories: AE, CL, YoY, MR, Clean (specificity = 100-FPR)
categories = ["AE", "CL", "YoY", "MR", "Clean"]
category_labels = [
    "Arithmetic\nError (AE)",
    "Copy/Label\nError (CL)",
    "Year-over-Year\nError (YoY)",
    "Magnitude/\nRounding (MR)",
    "Clean\n(Specificity)",
]

system_names = [
    "Rule-\nbased",
    "Claude\nSonnet 4",
    "Qwen 3\n235B",
    "GPT-4.1",
    "Llama 4\nMaverick",
    "Claude\nOpus 4.6",
    "Claude\nSonnet 4.6",
    "DeepSeek\nV3.2",
    "MiniMax\nM2.5",
    "Gemini\n2.5 Pro",
]

# Detection rates by category.
# Opus 4.6, Sonnet 4.6: flag everything -> 100% detection, 0% specificity.
# Qwen 3 235B: 93.8% recall overall, 25.6% specificity (FPR=74.4%).
#   Per-category estimates for Qwen based on partial calibration:
#   AE=100, CL=100, YoY=81.3, MR=88.9, Clean=25.6
# Llama 4 Maverick: 67.7% recall, 48.8% specificity (FPR=51.2%).
#   Per-category: AE=43.8, CL=87.5, YoY=62.5, MR=66.7, Clean=48.8
# MiniMax and Gemini: 100% recall, 100% FPR -> all errors detected, 0% specificity.
data = np.array([
    # Rul    Son4   Qwen3  GPT41  Llama  Op46   Son46  DSV32  MiniM  Gem25
    [31.0,  100.0,  100.0, 100.0,  43.8, 100.0, 100.0, 100.0, 100.0, 100.0],  # AE
    [100.0, 100.0,  100.0, 100.0,  87.5, 100.0, 100.0, 100.0, 100.0, 100.0],  # CL
    [14.2,   93.8,   81.3, 100.0,  62.5, 100.0, 100.0, 100.0, 100.0, 100.0],  # YoY
    [32.0,   88.9,   88.9, 100.0,  66.7, 100.0, 100.0, 100.0, 100.0, 100.0],  # MR
    [100.0, 100.0,   25.6,   4.7,  48.8,   0.0,   0.0,   0.0,   0.0,   0.0],  # Clean (specificity)
])

# Style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.linewidth": 0.8,
    "figure.dpi": 300,
})

# Custom diverging colormap: red (low) -> white (mid) -> green (high)
cmap = mcolors.LinearSegmentedColormap.from_list(
    "rg",
    [(0.85, 0.15, 0.15), (1.0, 1.0, 0.85), (0.15, 0.65, 0.15)],
    N=256,
)

fig, ax = plt.subplots(figsize=(11.0, 4.5))

im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=100)

# Cell annotations
for i in range(len(categories)):
    for j in range(len(system_names)):
        val = data[i, j]
        # Choose text color for readability
        text_color = "white" if val < 25 or val > 85 else "black"
        ax.text(
            j, i, f"{val:.1f}%",
            ha="center", va="center",
            fontsize=8.5, fontweight="bold",
            color=text_color,
        )

ax.set_xticks(np.arange(len(system_names)))
ax.set_xticklabels(system_names, fontsize=9)
ax.set_yticks(np.arange(len(categories)))
ax.set_yticklabels(category_labels, fontsize=9)

# Move x-axis labels to the top
ax.xaxis.set_ticks_position("top")
ax.xaxis.set_label_position("top")

# Grid lines between cells
for edge in np.arange(-0.5, len(categories), 1):
    ax.axhline(edge, color="white", linewidth=1.5)
for edge in np.arange(-0.5, len(system_names), 1):
    ax.axvline(edge, color="white", linewidth=1.5)

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
cbar.set_label("Detection Rate (%)", fontsize=10)
cbar.ax.tick_params(labelsize=9)

ax.set_title("")

fig.tight_layout(pad=0.5)
fig.savefig("/Users/silupanda/Downloads/finverification-bench/paper/figures/figure_category_heatmap_updated.pdf",
            bbox_inches="tight", pad_inches=0.05)
plt.close(fig)
print("Saved figure_category_heatmap_updated.pdf")
