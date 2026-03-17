"""Generate publication-quality dataset figures for the FinVerBench paper.

Produces four figures:
  1. Dataset composition: error category bar chart + magnitude distribution
  2. Company sectors: horizontal bar chart of 43 companies grouped by sector
  3. Error taxonomy: heatmap of error type x magnitude instance counts
  4. Example statement: side-by-side clean vs. error-injected financial statement

All figures are saved as PDF to ``paper/figures/``.

Usage:
    PYTHONPATH=src python -m src.analysis.generate_figures
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "benchmark"
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CATEGORY_ORDER = ["AE", "CL", "YoY", "MR", "Clean"]
CATEGORY_LABELS_FULL = {
    "AE": "Arithmetic\nErrors (AE)",
    "CL": "Cross-Statement\nLinkage (CL)",
    "YoY": "Year-over-Year\nContinuity (YoY)",
    "MR": "Magnitude\nReasoning (MR)",
    "Clean": "Clean\nInstances",
}

MAGNITUDE_ORDER = ["<1%", "1-5%", "5-20%", ">20%"]

ERROR_TYPE_ORDER = [
    "AE_ROW_SUM",
    "AE_COLUMN_SUM",
    "CL_NET_INCOME_TO_RE",
    "CL_NET_INCOME_TO_CFS",
    "CL_ENDING_CASH",
    "YOY_OPENING_BALANCE",
    "YOY_COMPUTED_CHANGE",
    "MR_MINOR",
    "MR_MODERATE",
    "MR_SIGNIFICANT",
    "MR_EXTREME",
]

ERROR_TYPE_LABELS = {
    "AE_ROW_SUM": "Row Sum",
    "AE_COLUMN_SUM": "Column Sum",
    "CL_NET_INCOME_TO_RE": "NI to Ret. Earnings",
    "CL_NET_INCOME_TO_CFS": "NI to Cash Flow",
    "CL_ENDING_CASH": "Ending Cash",
    "YOY_OPENING_BALANCE": "Opening Balance",
    "YOY_COMPUTED_CHANGE": "Computed Change",
    "MR_MINOR": "Minor",
    "MR_MODERATE": "Moderate",
    "MR_SIGNIFICANT": "Significant",
    "MR_EXTREME": "Extreme",
}

ERROR_TYPE_TO_CATEGORY = {
    "AE_ROW_SUM": "AE",
    "AE_COLUMN_SUM": "AE",
    "CL_NET_INCOME_TO_RE": "CL",
    "CL_NET_INCOME_TO_CFS": "CL",
    "CL_ENDING_CASH": "CL",
    "YOY_OPENING_BALANCE": "YoY",
    "YOY_COMPUTED_CHANGE": "YoY",
    "MR_MINOR": "MR",
    "MR_MODERATE": "MR",
    "MR_SIGNIFICANT": "MR",
    "MR_EXTREME": "MR",
}

# Company-to-sector mapping for the 43 S&P 500 companies in the benchmark.
COMPANY_SECTOR: Dict[str, str] = {
    "Apple Inc.": "Technology",
    "MICROSOFT CORPORATION": "Technology",
    "ALPHABET INC.": "Technology",
    "AMAZON COM INC": "Technology",
    "Meta Platforms, Inc.": "Technology",
    "NVIDIA CORP": "Technology",
    "Broadcom Ltd": "Technology",
    "ADOBE INC.": "Technology",
    "Salesforce, Inc.": "Technology",
    "CISCO SYSTEMS, INC.": "Technology",
    "INTEL CORPORATION": "Technology",
    "Oracle Corporation": "Technology",
    "TEXAS INSTRUMENTS INC": "Technology",
    "Johnson & Johnson": "Healthcare",
    "ELI LILLY AND COMPANY": "Healthcare",
    "UNITEDHEALTH GROUP INCORPORATED": "Healthcare",
    "PFIZER INC": "Healthcare",
    "ABBOTT LABORATORIES": "Healthcare",
    "AMGEN INC": "Healthcare",
    "Merck & Co., Inc.": "Healthcare",
    "Merck\xa0& Co., Inc.": "Healthcare",
    "THERMO FISHER SCIENTIFIC INC.": "Healthcare",
    "Walmart Inc.": "Consumer",
    "COSTCO WHOLESALE CORP /NEW": "Consumer",
    "HOME DEPOT, INC.": "Consumer",
    "LOWES COMPANIES INC": "Consumer",
    "McDONALD'S CORPORATION": "Consumer",
    "McDONALD\u2019S\xa0CORPORATION": "Consumer",
    "NIKE, Inc.": "Consumer",
    "COCA COLA CO": "Consumer",
    "PepsiCo, Inc.": "Consumer",
    "PROCTER & GAMBLE CO": "Consumer",
    "Philip Morris International Inc.": "Consumer",
    "Walt Disney Co": "Consumer",
    "Comcast Corporation": "Consumer",
    "BOEING CO": "Industrials",
    "HONEYWELL INTERNATIONAL INC": "Industrials",
    "GENERAL ELECTRIC CAPITAL CORP": "Industrials",
    "United Parcel Service, Inc.": "Industrials",
    "Chevron Corp": "Energy",
    "Exxon Mobil Corporation": "Energy",
    "JPMorgan Chase & Co": "Financial",
    "Bank of America Corporation": "Financial",
    "BERKSHIRE HATHAWAY INC": "Financial",
    "VISA INC.": "Financial",
}

SECTOR_ORDER = [
    "Technology",
    "Healthcare",
    "Consumer",
    "Industrials",
    "Energy",
    "Financial",
]

# Colorblind-friendly palette (Tableau 10 subset).
PALETTE = {
    "AE": "#4e79a7",
    "CL": "#f28e2b",
    "YoY": "#e15759",
    "MR": "#76b7b2",
    "Clean": "#59a14f",
}

SECTOR_COLORS = {
    "Technology": "#4e79a7",
    "Healthcare": "#e15759",
    "Consumer": "#f28e2b",
    "Industrials": "#76b7b2",
    "Energy": "#edc948",
    "Financial": "#b07aa1",
}

MAGNITUDE_COLORS = {
    "<1%": "#a0cbe8",
    "1-5%": "#4e79a7",
    "5-20%": "#f28e2b",
    ">20%": "#e15759",
}


def _apply_publication_style() -> None:
    """Configure matplotlib for publication-quality output."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_benchmark(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_stats(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _magnitude_bin(pct: float) -> str:
    """Assign a magnitude percentage to a bin label."""
    pct = abs(pct)
    if pct < 1.0:
        return "<1%"
    elif pct < 5.0:
        return "1-5%"
    elif pct < 20.0:
        return "5-20%"
    else:
        return ">20%"


def _short_company(name: str) -> str:
    """Shorten and normalize company names for publication display."""
    import unicodedata
    import re
    # Normalize unicode: replace non-breaking spaces, curly quotes, etc.
    s = unicodedata.normalize("NFKC", name).strip()
    s = s.replace("\u2019", "'")

    # Full-name overrides for tricky cases (keys must be UPPERCASE).
    overrides: Dict[str, str] = {
        "COCA COLA CO": "Coca-Cola Co.",
        "MCDONALD'S CORPORATION": "McDonald's Corp.",
        "ELI LILLY AND COMPANY": "Eli Lilly & Co.",
        "AMAZON COM INC": "Amazon.com Inc.",
        "COSTCO WHOLESALE CORP /NEW": "Costco Wholesale Corp.",
        "GENERAL ELECTRIC CAPITAL CORP": "GE Capital Corp.",
        "LOWES COMPANIES INC": "Lowe's Cos. Inc.",
    }
    # Check overrides using normalized uppercase form.
    s_upper = re.sub(r"\s+", " ", s.upper().strip(" .,"))
    for key, val in overrides.items():
        if key == s_upper:
            return val

    # Abbreviation replacements applied word-by-word.
    abbrevs = {
        "CORPORATION": "Corp.",
        "INCORPORATED": "Inc.",
        "INTERNATIONAL": "Intl.",
        "COMPANY": "Co.",
        "COMPANIES": "Cos.",
        "CORP": "Corp.",
        "INC": "Inc.",
        "CO": "Co.",
        "LTD": "Ltd.",
    }

    # Remove trailing " /NEW" etc.
    s = re.sub(r"\s*/NEW\s*$", "", s, flags=re.IGNORECASE)

    parts = s.split()
    result = []
    for p in parts:
        stripped = p.upper().rstrip(".,")
        if stripped in abbrevs:
            result.append(abbrevs[stripped])
        elif p.isupper():
            # Title-case fully uppercase words.
            result.append(p.title())
        else:
            result.append(p)
    return " ".join(result)


# ---------------------------------------------------------------------------
# Figure 1: Dataset composition (2-panel)
# ---------------------------------------------------------------------------

def plot_dataset_composition(
    stats: Dict[str, Any],
    output_path: Path,
) -> None:
    """2-panel figure: bar chart of categories + bar chart of magnitude distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0),
                                     gridspec_kw={"width_ratios": [1.15, 1]})

    # --- Left panel: instances by error category ---
    cat_counts_raw = stats["instances_per_category"]
    clean_count = stats["clean_instances"]

    full_labels = [
        "Arithmetic\n(AE)",
        "Cross-Stmt\n(CL)",
        "Year-over-Year\n(YoY)",
        "Magnitude\n(MR)",
        "Clean",
    ]

    categories = CATEGORY_ORDER
    counts = []
    colors = []
    for cat in categories:
        if cat == "Clean":
            counts.append(clean_count)
        elif cat == "YoY":
            counts.append(cat_counts_raw.get("YOY", 0))
        else:
            counts.append(cat_counts_raw.get(cat, 0))
        colors.append(PALETTE[cat])

    x = np.arange(len(categories))
    bars = ax1.bar(x, counts, color=colors, edgecolor="white", linewidth=0.6, width=0.6)

    for bar, count in zip(bars, counts):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 15,
            f"{count:,}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(full_labels, fontsize=8.5, linespacing=0.9)
    ax1.set_ylabel("Number of Instances")
    ax1.set_title("(a) Instances by Error Category")
    ax1.set_ylim(0, max(counts) * 1.18)

    # --- Right panel: magnitude distribution (stacked horizontal bar) ---
    mag_dist = stats["magnitude_distribution"]
    mag_vals = [mag_dist[m] for m in MAGNITUDE_ORDER]
    total = sum(mag_vals)
    mag_pcts = [v / total * 100 for v in mag_vals]

    bar_left = 0.0
    for i, (mag, val, pct) in enumerate(zip(MAGNITUDE_ORDER, mag_vals, mag_pcts)):
        bar = ax2.barh(
            0, pct, left=bar_left, height=0.5,
            color=MAGNITUDE_COLORS[mag], edgecolor="white", linewidth=0.6,
        )
        # Label inside bar
        cx = bar_left + pct / 2
        label_text = f"{mag}\n{val:,}\n({pct:.0f}%)"
        ax2.text(cx, 0, label_text, ha="center", va="center", fontsize=8, fontweight="bold")
        bar_left += pct

    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Percentage of Error Instances")
    ax2.set_title("(b) Error Magnitude Distribution")
    ax2.set_yticks([])
    ax2.spines["left"].set_visible(False)

    fig.tight_layout(w_pad=3.0)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info("Saved figure_dataset_composition to %s", output_path)


# ---------------------------------------------------------------------------
# Figure 2: Company sectors
# ---------------------------------------------------------------------------

def plot_company_sectors(
    stats: Dict[str, Any],
    output_path: Path,
) -> None:
    """Horizontal bar chart of 43 companies grouped by sector."""
    instances_per_company = stats["instances_per_company"]

    # Build a normalized lookup to handle non-breaking spaces and curly quotes.
    def _normalize(s: str) -> str:
        import unicodedata
        return unicodedata.normalize("NFKD", s).replace("\xa0", " ")

    normalized_sector: Dict[str, str] = {}
    for k, v in COMPANY_SECTOR.items():
        normalized_sector[_normalize(k)] = v

    # Group companies by sector and sort within each sector.
    sector_companies: Dict[str, List[Tuple[str, int]]] = {s: [] for s in SECTOR_ORDER}
    for company, count in instances_per_company.items():
        sector = COMPANY_SECTOR.get(company)
        if sector is None:
            sector = normalized_sector.get(_normalize(company), "Other")
        if sector not in sector_companies:
            logger.warning("Unknown sector '%s' for company '%s'", sector, company)
            continue
        sector_companies[sector].append((company, count))

    for sector in sector_companies:
        sector_companies[sector].sort(key=lambda x: x[1], reverse=True)

    # Build ordered lists for plotting (sectors top-to-bottom, reversed for barh).
    labels: List[str] = []
    values: List[int] = []
    bar_colors: List[str] = []
    sector_boundaries: List[Tuple[int, str]] = []

    for sector in reversed(SECTOR_ORDER):
        start_idx = len(labels)
        for company, count in sector_companies[sector]:
            labels.append(_short_company(company))
            values.append(count)
            bar_colors.append(SECTOR_COLORS[sector])
        if sector_companies[sector]:
            sector_boundaries.append((start_idx + len(sector_companies[sector]) // 2, sector))

    fig_height = max(6, len(labels) * 0.32 + 1.5)
    fig, ax = plt.subplots(figsize=(7, fig_height))

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, height=0.7, color=bar_colors, edgecolor="white", linewidth=0.4)

    # Value labels at end of bars.
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            str(val),
            ha="left", va="center", fontsize=7,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("Number of Benchmark Instances")
    ax.set_title("Companies by Sector (43 S&P 500 Companies)")
    ax.set_xlim(0, max(values) * 1.15)

    # Sector legend.
    legend_patches = [
        mpatches.Patch(color=SECTOR_COLORS[s], label=f"{s} ({len(sector_companies[s])})")
        for s in SECTOR_ORDER
    ]
    ax.legend(
        handles=legend_patches, loc="lower right", framealpha=0.9,
        fontsize=8, title="Sector", title_fontsize=9,
    )

    # Light horizontal separator lines between sectors.
    cumulative = 0
    for sector in reversed(SECTOR_ORDER):
        n = len(sector_companies[sector])
        cumulative += n
        if cumulative < len(labels):
            ax.axhline(y=cumulative - 0.5, color="gray", linewidth=0.4, linestyle="-", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info("Saved figure_company_sectors to %s", output_path)


# ---------------------------------------------------------------------------
# Figure 3: Error taxonomy heatmap
# ---------------------------------------------------------------------------

def plot_error_taxonomy(
    benchmark: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Heatmap of error type (rows) x magnitude level (columns)."""
    # Count instances per (error_type, magnitude_bin).
    counts: Dict[Tuple[str, str], int] = {}
    for inst in benchmark:
        gt = inst["ground_truth"]
        if not gt.get("has_error"):
            continue
        etype = gt.get("error_type")
        mag = gt.get("error_magnitude_pct")
        if etype is None or mag is None:
            continue
        mbin = _magnitude_bin(mag)
        key = (etype, mbin)
        counts[key] = counts.get(key, 0) + 1

    # Build matrix.
    rows = ERROR_TYPE_ORDER
    cols = MAGNITUDE_ORDER
    matrix = np.zeros((len(rows), len(cols)), dtype=int)
    for i, etype in enumerate(rows):
        for j, mbin in enumerate(cols):
            matrix[i, j] = counts.get((etype, mbin), 0)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Use a sequential colormap.
    cmap = plt.cm.Blues
    # Mask zeros for cleaner display.
    masked = np.ma.masked_where(matrix == 0, matrix)
    cmap_copy = cmap.copy()
    cmap_copy.set_bad(color="#f5f5f5")

    vmax = matrix.max() if matrix.max() > 0 else 1
    im = ax.imshow(masked, cmap=cmap_copy, vmin=0, vmax=vmax, aspect="auto")

    # Annotate cells.
    for i in range(len(rows)):
        for j in range(len(cols)):
            val = matrix[i, j]
            if val > 0:
                text_color = "white" if val > vmax * 0.6 else "black"
                ax.text(
                    j, i, str(val),
                    ha="center", va="center", fontsize=10,
                    fontweight="bold", color=text_color,
                )
            else:
                ax.text(j, i, "--", ha="center", va="center", fontsize=9, color="#bbb")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks(range(len(rows)))
    # Composite labels: prepend category abbreviation to each row.
    ytick_labels = []
    for e in rows:
        cat = ERROR_TYPE_TO_CATEGORY[e]
        ytick_labels.append(f"{ERROR_TYPE_LABELS[e]}")
    ax.set_yticklabels(ytick_labels, fontsize=9)
    ax.set_xlabel("Error Magnitude")
    ax.set_ylabel("Error Type")
    ax.set_title("Error Taxonomy: Instance Counts by Type and Magnitude")

    # Add gridlines between cells.
    for i in range(len(rows) + 1):
        ax.axhline(y=i - 0.5, color="white", linewidth=1.5)
    for j in range(len(cols) + 1):
        ax.axvline(x=j - 0.5, color="white", linewidth=1.5)

    # Add horizontal separator lines between category groups.
    prev_cat = None
    for i, etype in enumerate(rows):
        cat = ERROR_TYPE_TO_CATEGORY[etype]
        if prev_cat is not None and cat != prev_cat:
            ax.axhline(y=i - 0.5, color="#999", linewidth=1.0, linestyle="-")
        prev_cat = cat

    # Place colorbar with enough padding to leave room for bracket labels.
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.14, shrink=0.8)
    cbar.set_label("Number of Instances", fontsize=9)

    fig.tight_layout()

    # Category bracket labels placed using axes-fraction coordinates (via
    # transAxes) so they sit between the heatmap right edge and the colorbar,
    # regardless of data coordinate ranges. Must be done after tight_layout.
    cat_spans: Dict[str, Tuple[int, int]] = {}
    for i, etype in enumerate(rows):
        cat = ERROR_TYPE_TO_CATEGORY[etype]
        if cat not in cat_spans:
            cat_spans[cat] = (i, i)
        else:
            cat_spans[cat] = (cat_spans[cat][0], i)

    n_rows = len(rows)
    for cat, (start, end) in cat_spans.items():
        color = PALETTE.get(cat, "black")
        # Convert row indices to axes fraction: row 0 is at top, row n-1 at bottom.
        # imshow places row i at y=i in data coords; axes fraction maps inversely.
        # We use the data-to-axes transform.
        y_top_data = start - 0.35
        y_bot_data = end + 0.35
        mid_data = (start + end) / 2

        # Place bracket and label at axes x=1.02..1.08 (just past right edge).
        bracket_x_ax = 1.02
        label_x_ax = 1.05

        # Transform data y to axes y.
        y_top_ax = ax.transData.transform((0, y_top_data))
        y_bot_ax = ax.transData.transform((0, y_bot_data))
        y_mid_ax = ax.transData.transform((0, mid_data))

        # Convert to axes coordinates.
        inv = ax.transAxes.inverted()
        y_top_frac = inv.transform(y_top_ax)[1]
        y_bot_frac = inv.transform(y_bot_ax)[1]
        y_mid_frac = inv.transform(y_mid_ax)[1]

        # Vertical bracket line in axes coordinates.
        ax.plot(
            [bracket_x_ax, bracket_x_ax],
            [y_top_frac, y_bot_frac],
            color=color, linewidth=2.5, clip_on=False,
            solid_capstyle="round", transform=ax.transAxes,
        )
        # Category label.
        ax.text(
            label_x_ax, y_mid_frac, cat,
            ha="left", va="center", fontsize=10, fontweight="bold",
            color=color, clip_on=False, transform=ax.transAxes,
        )

    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info("Saved figure_error_taxonomy to %s", output_path)


# ---------------------------------------------------------------------------
# Figure 4: Example statement comparison
# ---------------------------------------------------------------------------

def plot_example_statement(
    benchmark: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Side-by-side comparison of clean vs. error-injected financial statement."""
    # Find a clean instance and a matching error instance for Apple.
    clean_inst = None
    error_inst = None
    for inst in benchmark:
        if inst["instance_id"] == "apple_inc__clean":
            clean_inst = inst
        elif inst["instance_id"] == "apple_inc__AE_ROW_SUM_0.5pct_0":
            error_inst = inst
    if clean_inst is None or error_inst is None:
        logger.warning("Could not find Apple clean/error instances for example figure.")
        return

    gt = error_inst["ground_truth"]
    error_location = gt["error_location"]
    original_val = gt["original_value"]
    modified_val = gt["modified_value"]
    error_desc = gt["description"]
    magnitude = gt["error_magnitude_pct"]

    # Extract just the cash flow statement section.
    def _extract_cfs(text: str) -> str:
        lines = text.split("\n")
        start = None
        for i, line in enumerate(lines):
            if "CASH FLOW STATEMENT" in line:
                start = i
                break
        if start is None:
            return text
        return "\n".join(lines[start:])

    clean_cfs = _extract_cfs(clean_inst["formatted_statements"])
    error_cfs = _extract_cfs(error_inst["formatted_statements"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

    mono_fontsize = 7.2
    text_top = 0.93
    text_left = 0.05

    # Compute line height in axes fraction.
    # Monospace text at 7.2pt: use renderer to measure, or approximate.
    # With 17 lines of CFS text in ~0.88 axes height, each line ~ 0.052.
    error_lines = error_cfs.split("\n")
    n_lines = len(error_lines)
    # The text height occupied is approximately text_top - 0.05 = 0.88
    text_height = 0.88
    line_height = text_height / max(n_lines, 1)

    def _render_panel(
        ax: plt.Axes,
        title: str,
        text: str,
        bg_color: str,
        border_color: str,
        highlight_line_idx: int = -1,
    ) -> None:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

        bg = mpatches.FancyBboxPatch(
            (0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.02",
            facecolor=bg_color, edgecolor=border_color, linewidth=1,
        )
        ax.add_patch(bg)

        lines = text.split("\n")
        for i, line in enumerate(lines):
            y = text_top - i * line_height
            is_target = (i == highlight_line_idx)

            if is_target:
                # Draw highlight rectangle behind this line.
                rect = mpatches.FancyBboxPatch(
                    (0.03, y - line_height * 0.35),
                    0.94, line_height * 0.85,
                    boxstyle="round,pad=0.003",
                    facecolor="#e15759", edgecolor="none", alpha=0.18,
                    transform=ax.transAxes,
                )
                ax.add_patch(rect)

            ax.text(
                text_left, y, line,
                transform=ax.transAxes, fontsize=mono_fontsize,
                fontfamily="monospace",
                verticalalignment="top", horizontalalignment="left",
                fontweight="bold" if is_target else "normal",
                color="#c41e3a" if is_target else "black",
            )

    # --- Left panel: Clean statement ---
    _render_panel(ax1, "(a) Clean Statement", clean_cfs, "#f8f9fa", "#dee2e6")

    # --- Right panel: Error-injected statement ---
    # Find which line index contains the modified value.
    target_idx = -1
    for i, line in enumerate(error_lines):
        if "112,039.41" in line or f"{modified_val}" in line:
            target_idx = i
            break

    _render_panel(
        ax2, "(b) Error-Injected Statement", error_cfs,
        "#fff5f5", "#e15759", highlight_line_idx=target_idx,
    )

    # Arrow annotation pointing to the highlighted line.
    if target_idx >= 0:
        target_y = text_top - target_idx * line_height - line_height * 0.1
        ax2.annotate(
            f"Modified: {original_val:,.0f} -> {modified_val:,.2f}\n({magnitude}% deviation)",
            xy=(0.88, target_y),
            xytext=(0.62, target_y - 0.20),
            transform=ax2.transAxes,
            fontsize=8, fontweight="bold", color="#c41e3a",
            arrowprops=dict(
                arrowstyle="-|>", color="#c41e3a", linewidth=1.3,
                connectionstyle="arc3,rad=-0.25",
            ),
            bbox=dict(
                boxstyle="round,pad=0.35", facecolor="white",
                edgecolor="#c41e3a", linewidth=1, alpha=0.95,
            ),
        )

    # Add error metadata below the figure.
    meta_text = (
        f"Error Type: AE_ROW_SUM  |  "
        f"Location: {error_location}  |  "
        f"Magnitude: {magnitude}%  |  "
        f"Difficulty: easy"
    )
    fig.text(
        0.5, -0.01, meta_text,
        ha="center", va="top", fontsize=8, fontstyle="italic", color="#666",
    )

    fig.tight_layout(w_pad=1.5)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info("Saved figure_example_statement to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    _apply_publication_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    benchmark_path = DATA_DIR / "benchmark.json"
    stats_path = DATA_DIR / "benchmark_stats.json"

    logger.info("Loading benchmark data from %s", benchmark_path)
    benchmark = _load_benchmark(benchmark_path)
    logger.info("Loaded %d benchmark instances", len(benchmark))

    logger.info("Loading stats from %s", stats_path)
    stats = _load_stats(stats_path)

    # Figure 1: Dataset composition
    plot_dataset_composition(stats, FIGURES_DIR / "figure_dataset_composition.pdf")

    # Figure 2: Company sectors
    plot_company_sectors(stats, FIGURES_DIR / "figure_company_sectors.pdf")

    # Figure 3: Error taxonomy heatmap
    plot_error_taxonomy(benchmark, FIGURES_DIR / "figure_error_taxonomy.pdf")

    # Figure 4: Example statement
    plot_example_statement(benchmark, FIGURES_DIR / "figure_example_statement.pdf")

    logger.info("All figures generated successfully in %s", FIGURES_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
