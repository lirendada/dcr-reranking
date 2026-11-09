"""Render two saved-data figure previews at the manuscript's text width."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.text import Text


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "rendered"
OUT.mkdir(exist_ok=True)
MM = 1 / 25.4
WIDTH = 122
INK = "#243042"
MUTED = "#626C79"
RULE = "#D9DEE5"
GRAY = "#F5F6F8"
BLUE = "#357AB8"
ORANGE = "#CC6A50"
BLUE_TEXT = "#286697"
ORANGE_TEXT = "#A9523A"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 7,
    "text.color": INK,
    "axes.labelcolor": MUTED,
    "axes.edgecolor": RULE,
    "axes.linewidth": 0.5,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 6.4,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "figure.dpi": 160,
    "savefig.dpi": 600,
    "text.usetex": False,
    "axes.unicode_minus": True,
})


def text(canvas, x, y, label, size=7, color=INK, weight="normal", ha="left"):
    return canvas.text(x, y, label, fontsize=size, color=color,
                       fontweight=weight, ha=ha, va="center")


def page(height):
    fig = plt.figure(figsize=(WIDTH * MM, height * MM))
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set(xlim=(0, WIDTH), ylim=(height, 0))
    canvas.set_axis_off()
    canvas.patch.set_visible(False)
    return fig, canvas


def section(canvas, letter, top, title, subtitle):
    text(canvas, 3, top, letter, size=8.5, weight="bold")
    text(canvas, 9, top, title, size=8, weight="bold")
    text(canvas, 9, top + 4.6, subtitle, size=6.5, color=MUTED)
    canvas.plot([3, 119], [top + 8.5, top + 8.5], color=RULE, lw=0.5)


def shade(fig, top, height):
    full_height = fig.get_size_inches()[1] / MM
    fig.add_artist(Rectangle((3 / WIDTH, 1 - (top + height) / full_height),
                             116 / WIDTH, height / full_height,
                             transform=fig.transFigure, facecolor=GRAY,
                             edgecolor="none", zorder=-1))


def forest(fig, left, top, width, height, xlim, ticks):
    full_height = fig.get_size_inches()[1] / MM
    ax = fig.add_axes([left / WIDTH, 1 - (top + height) / full_height,
                       width / WIDTH, height / full_height])
    ax.set(xlim=xlim, ylim=(height, 0), yticks=[], xticks=ticks)
    ax.set_facecolor("none")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", length=2.3, width=0.5, pad=2)
    ax.axvline(0, color="#9AA3AE", lw=0.65, ls=(0, (3, 3)), zorder=1)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=RULE, lw=0.4, alpha=0.65)
    return ax


def point(ax, y, row, color, marker):
    ax.errorbar(row["est"], y,
                xerr=[[row["est"] - row["lo"]], [row["hi"] - row["est"]]],
                fmt=marker, markersize=5, color=color,
                markeredgecolor="white", markeredgewidth=0.65,
                elinewidth=1.15, capsize=2.3, capthick=0.8, zorder=3)


def interval(row):
    return (f"{row['est']:+.2f}  [{row['lo']:+.2f}, {row['hi']:+.2f}]")


def export(fig, canvas, axes, stem):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = []
    outside = []
    for item in fig.findobj(Text):
        if not item.get_visible() or not item.get_text().strip():
            continue
        bounds = item.get_window_extent(renderer)
        if not fig.bbox.contains(bounds.x0, bounds.y0) or not fig.bbox.contains(bounds.x1, bounds.y1):
            outside.append(item.get_text())
        boxes.append((item.get_text(), bounds))
    overlaps = []
    for index, (label, bounds) in enumerate(boxes):
        for other_label, other_bounds in boxes[index + 1:]:
            dx = min(bounds.x1, other_bounds.x1) - max(bounds.x0, other_bounds.x0)
            dy = min(bounds.y1, other_bounds.y1) - max(bounds.y0, other_bounds.y0)
            if dx > 0.8 and dy > 0.8:
                overlaps.append([label, other_label])
    geometry = {"text_outside_canvas": outside, "text_overlap_candidates": overlaps,
                "width_mm": WIDTH, "height_mm": float(fig.get_size_inches()[1] / MM)}
    (OUT / f"{stem}.geometry.json").write_text(json.dumps(geometry, indent=2) + "\n")
    assert not outside, outside
    assert not overlaps, overlaps
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches=None)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches=None)
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches=None)
    plt.close(fig)


def figure_one():
    data = json.loads((ROOT / "source_data/fig1_input_effects_and_fusion.json").read_text())
    fig, canvas = page(125)
    section(canvas, "a", 4, "Input effects within a fixed candidate head",
            "ExcluIR · 3,452 queries · request minus topic · 95% intervals")
    first_axis = forest(fig, 40, 15.5, 65, 39.5, (-1, 28), [0, 10, 20])
    labels = ["MiniLM", "Qwen3", "Input-effect interaction"]
    details = ["L6-v2", "Reranker-4B", "Qwen minus MiniLM · primary"]
    for index, row in enumerate(data["panel_A"]["rows"]):
        y = 18 + 13 * index
        color = [BLUE, ORANGE, INK][index]
        if index != 1:
            shade(fig, y - 3.2, 12)
        label_color = [BLUE_TEXT, ORANGE_TEXT, INK][index]
        text(canvas, 4, y, labels[index], size=7.2, color=label_color, weight="bold")
        text(canvas, 4, y + 3.2, details[index], size=6.4, color=MUTED)
        text(canvas, 4, y + 6.4, interval(row), size=6.5)
        point(first_axis, y + 3.2 - 15.5, row, color, ["o", "s", "D"][index])
    first_axis.set_xlabel("P@1 change (percentage points)", fontsize=7, labelpad=4)

    section(canvas, "b", 73, "Corrections and losses after fusion",
            "Top-1 outcomes vs fixed Qwen-request · same head of 20")
    second_axis = forest(fig, 40, 87, 65, 24, (-150, 225), [-150, -75, 0, 75, 150, 225])
    second_axis.grid(False)
    second_axis.set_xticklabels(["150", "75", "0", "75", "150", "225"])
    text(canvas, 49, 84.5, "Losses", size=6.5, color=ORANGE_TEXT)
    text(canvas, 83, 84.5, "Rescues", size=6.5, color=BLUE_TEXT)
    text(canvas, 118, 84.5, "Net", size=6.5, color=MUTED, ha="right")
    for index, row in enumerate(data["panel_B"]["rows"]):
        y = 91 + 8 * index
        text(canvas, 4, y, row["label"], size=7.2)
        second_axis.barh(y - 87, -row["losses"], height=3.4, color=ORANGE, zorder=3)
        second_axis.barh(y - 87, row["rescued"], height=3.4, color=BLUE, zorder=3)
        second_axis.annotate(str(row["losses"]), (-row["losses"], y - 87),
                             xytext=(-3, 0), textcoords="offset points",
                             ha="right", va="center", fontsize=6.8)
        second_axis.annotate(str(row["rescued"]), (row["rescued"], y - 87),
                             xytext=(3, 0), textcoords="offset points",
                             ha="left", va="center", fontsize=6.8)
        text(canvas, 118, y, f"{row['net']:+d}", size=7, weight="bold", ha="right")
    second_axis.set_xlabel("Queries", fontsize=7, labelpad=4)
    export(fig, canvas, [first_axis, second_axis], "figure_1_white")


def figure_two():
    data = json.loads((ROOT / "source_data/fig2_cross_collection.json").read_text())
    fig, canvas = page(110)
    section(canvas, "a", 4, "Input effects across collections",
            "Request minus topic · fixed MiniLM-topic head of 20")
    ax = forest(fig, 43, 20, 75, 70, (-20, 30), [-20, -10, 0, 10, 20, 30])
    counts = ["3,452 queries", "1,718 requests", "86 requests"]
    for group_index, group in enumerate(data["groups"]):
        y = 16 + 26 * group_index
        if group_index != 1:
            shade(fig, y - 2.5, 25)
        text(canvas, 4, y, group["collection"], size=7.5, weight="bold")
        text(canvas, 118, y, counts[group_index], size=6.5, color=MUTED, ha="right")
        for row_index, row in enumerate(group["rows"]):
            yy = y + 6 + 9 * row_index
            color = [BLUE, ORANGE][row_index]
            label_color = [BLUE_TEXT, ORANGE_TEXT][row_index]
            text(canvas, 4, yy, row["label"], size=7, color=label_color, weight="bold")
            text(canvas, 4, yy + 3.3, interval(row), size=6.5)
            point(ax, yy + 1.4 - 20, row, color, ["o", "s"][row_index])
    ax.set_xlabel("P@1 change (percentage points)", fontsize=7, labelpad=4)
    text(canvas, 4, 105, "95% intervals: component resampling (ExcluIR), topic resampling (IFIR).",
         size=6.4, color=MUTED)
    export(fig, canvas, [ax], "figure_2_white")


if __name__ == "__main__":
    figure_one()
    figure_two()
