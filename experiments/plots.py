"""Draws the report figures from the CSV results of the training and the experiments.

Usage: python experiments/plots.py   (writes PNG files to docs/images/)
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "docs" / "images"
TRAINING = ROOT / "training" / "results"
RESULTS = ROOT / "experiments" / "results"
# Categorical slots in fixed order (validated for colour-vision deficiencies), and neutral inks.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
ENGINES = ("v1", "v2", "v3", "v4", "v5", "cloudict")

plt.rcParams.update({
    "font.family": "Segoe UI", "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": MUTED,
    "axes.titlesize": 10, "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "axes.axisbelow": True, "lines.linewidth": 2, "figure.dpi": 150, "savefig.bbox": "tight",
})


def read(path):
    """Rows of a CSV file as dictionaries (empty if the experiment has not been run)."""
    if not path.exists():
        return []
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def save(figure, name):
    """Write a figure to docs/images/."""
    IMAGES.mkdir(parents=True, exist_ok=True)
    figure.savefig(IMAGES / name)
    plt.close(figure)
    print("wrote", IMAGES / name)


def ga_evolution():
    """Fitness of the genetic algorithm and the evolution of the weights of its best individual."""
    evolution, progress = read(TRAINING / "evolution.csv"), read(TRAINING / "progress.csv")
    if not evolution:
        return
    generations = [int(row["generation"]) for row in evolution]
    figure, (left, right) = plt.subplots(1, 2, figsize=(9, 3.2))
    left.plot(generations, [float(r["best_fitness"]) for r in evolution], color=SERIES[0], label="best (relative)")
    left.plot(generations, [float(r["mean_fitness"]) for r in evolution], color=SERIES[1], label="mean (relative)")
    if progress:
        left.plot(generations, [float(r["score_vs_hand_tuned"]) for r in progress], color=SERIES[2],
                  marker="o", markersize=4, label="best vs hand-tuned (absolute)")
    left.set(title="Fitness", xlabel="generation", ylabel="share of points", ylim=(0.3, 0.9))
    left.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8)

    features = [key for key in evolution[0] if key not in ("generation", "best_fitness", "mean_fitness")]
    hand = (1, 4, 16, -1, -4, -16, -128)
    for i, (feature, start) in enumerate(zip(features, hand)):
        right.plot(generations, [float(row[feature]) / start for row in evolution], color=SERIES[i], label=feature)
    right.axhline(1, color=MUTED, linewidth=0.8)
    right.set(title="Weights of the best individual", xlabel="generation", ylabel="weight / hand-tuned weight")
    right.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=8)
    save(figure, "ga_evolution.png")


def battery():
    """Positions solved by each version against Cloudict: attacks (forced wins) and defences."""
    rows = [row for path in sorted(RESULTS.glob("battery*.csv")) for row in read(path)]
    if not rows:
        return
    engines = [e for e in ENGINES if any(r["engine"] == e for r in rows)]
    total = sum(1 for r in rows if r["engine"] == engines[0] and r["kind"] == "attack")
    figure, axis = plt.subplots(figsize=(6, 2.8))
    width = 0.38
    for i, kind in enumerate(("attack", "defence")):
        solved = [sum(int(r["success"]) for r in rows if r["engine"] == e and r["kind"] == kind) for e in engines]
        bars = axis.bar([x + (i - 0.5) * width for x in range(len(engines))], solved, width - 0.04,
                        color=SERIES[i], label=f"{kind}s solved (of {total})")
        axis.bar_label(bars, color=INK, fontsize=8)
    axis.set(xticks=range(len(engines)), xticklabels=engines, ylim=(0, total + 1.5), ylabel="positions",
             title="Test battery against Cloudict")
    axis.legend(loc="upper left", fontsize=8, ncol=2)
    save(figure, "battery.png")


def version_metrics():
    """Mean decision time, depth, expanded and pruned nodes per move of each version.

    v1-v3 search a fixed depth of 2 without reporting it, so their depth is shown as 2.
    """
    # v1-v4 from the round robin, v5 from the final tournament (v4 is not counted twice).
    rows = [r for r in read(RESULTS / "tournament_moves.csv") if r["engine"] != "cloudict"]
    rows += [r for r in read(RESULTS / "final_moves.csv") if r["engine"] == "v5"]
    rows = [r for r in rows if int(r["nodes"])]  # Opening book moves involve no search.
    if not rows:
        return
    engines = [e for e in ENGINES if any(r["engine"] == e for r in rows)]
    by_engine = defaultdict(list)
    for row in rows:
        by_engine[row["engine"]].append(row)
    metrics = (("seconds", "decision time (s)"), ("depth", "depth reached"),
               ("nodes", "expanded nodes"), ("pruned", "pruned nodes"))
    figure, axes = plt.subplots(1, 4, figsize=(10, 2.6))
    for axis, (column, label) in zip(axes, metrics):
        means = [sum(float(r[column]) or (2.0 if column == "depth" else 0.0) for r in by_engine[e]) / len(by_engine[e])
                 for e in engines]
        bars = axis.bar(engines, means, 0.6, color=[SERIES[i] for i in range(len(engines))])
        axis.bar_label(bars, fmt={"seconds": "%.2f", "depth": "%.1f"}.get(column, "%.0f"), color=INK, fontsize=7)
        axis.set_title(f"Mean {label}" + (" (v1-v3: fixed)" if column == "depth" else ""))
    save(figure, "version_metrics.png")


def tournament():
    """Points (2 per win, 1 per draw) of each engine in the round robin, as a share of the maximum."""
    rows = read(RESULTS / "tournament_games.csv")
    if not rows:
        return
    engines = [e for e in ENGINES if any(e in (r["black"], r["white"]) for r in rows)]
    shares = []
    for engine in engines:
        games = [r for r in rows if engine in (r["black"], r["white"])]
        points = sum(2 if r["winner"] == engine else 1 if r["winner"] == "draw" else 0 for r in games)
        shares.append(points / (2 * len(games)))
    figure, axis = plt.subplots(figsize=(5, 2.6))
    bars = axis.bar(engines, shares, 0.6, color=[SERIES[i] for i in range(len(engines))])
    axis.bar_label(bars, labels=[f"{s:.0%}" for s in shares], color=INK, fontsize=8)
    axis.set(ylim=(0, 1.05), ylabel="share of points", title="Round-robin tournament")
    save(figure, "tournament.png")


def time_limit():
    """Depth, nodes and correct decisions on the certified positions as the time limit grows."""
    rows = [r for r in read(RESULTS / "sweeps.csv") if r["parameter"] == "time limit (s)"]
    if not rows:
        return
    limits = list(dict.fromkeys(r["value"] for r in rows))
    group = {v: [r for r in rows if r["value"] == v] for v in limits}
    figure, axes = plt.subplots(1, 3, figsize=(9, 2.6))
    panels = (("depth", "Mean depth reached", "%.1f"), ("nodes", "Mean expanded nodes", "%.0f"),
              ("correct", "Correct decisions (of 12)", "%.0f"))
    for axis, (column, title, fmt) in zip(axes, panels):
        if column == "correct":
            data = [sum(int(r[column]) for r in group[v]) for v in limits]
        else:
            data = [sum(float(r[column]) for r in group[v]) / len(group[v]) for v in limits]
        axis.plot(limits, data, color=SERIES[0], marker="o", markersize=5)
        for x, y in zip(limits, data):
            axis.annotate(fmt % y, (x, y), textcoords="offset points", xytext=(0, 6), ha="center",
                          fontsize=7, color=INK)
        axis.set(title=title, xlabel="time limit (s)")
        if column == "correct":
            axis.yaxis.set_major_locator(MaxNLocator(integer=True))
        axis.margins(y=0.25)
    save(figure, "time_limit.png")


def duels():
    """Score of each non-default setting in games against the default engine (0.5 = equal)."""
    rows = read(RESULTS / "duels.csv")
    if not rows:
        return
    labels = [f"{r['parameter']} = {r['value']}" for r in rows]
    scores = [float(r["score"]) for r in rows]
    figure, axis = plt.subplots(figsize=(6, 0.4 * len(rows) + 0.8))
    bars = axis.barh(labels, scores, 0.6, color=SERIES[0])
    axis.bar_label(bars, labels=[f"{s:.2f}  ({r['wins']}W {r['draws']}D {r['losses']}L)" for s, r in zip(scores, rows)],
                   color=INK, fontsize=7, padding=3)
    axis.axvline(0.5, color=MUTED, linewidth=0.8)
    axis.set(xlim=(0, 1), xlabel="score against the default engine", title="Duels: one setting changed")
    axis.invert_yaxis()
    axis.grid(axis="y", visible=False)
    save(figure, "duels.png")


if __name__ == "__main__":
    ga_evolution()
    battery()
    version_metrics()
    tournament()
    time_limit()
    duels()
