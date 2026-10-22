"""Compute training/evaluation statistics from the saved CSVs and draw the report plots.

Usage:
    python3 analyze_results.py

Reads results/baseline_* and results/custom_e* (created by run_project.py) and writes
results/training_stats.csv plus PNG plots in results/plots/.
"""
import csv
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
RESULTS = HERE / "results"
PLOTS = RESULTS / "plots"
SEEDS = (0, 1, 2)

BASELINE = [("baseline_SW", "S-W", 18), ("baseline_LD", "L-D", 22), ("baseline_HP", "H-P", 16)]
CUSTOM = [("custom_e005", "epsilon = 0.05", 26), ("custom_e020", "epsilon = 0.2", 26),
          ("custom_e050", "epsilon = 0.5", 26)]
COLORS = ["#2a6fdb", "#e0832b", "#2f9e6e"]


def read_csv(filename):
    with open(filename, newline="") as source:
        return list(csv.DictReader(source))


def load(folder):
    training = [read_csv(RESULTS / folder / ("training_seed%d.csv" % s)) for s in SEEDS]
    evaluation = [read_csv(RESULTS / folder / ("evaluation_seed%d.csv" % s)) for s in SEEDS]
    return training, evaluation


def block_means(run, field, block):
    values = [float(r[field]) for r in run]
    return [statistics.mean(values[i:i + block]) for i in range(0, len(values), block)]


def first_success_streak(run, length=20):
    """First episode that starts a run of `length` consecutive successful episodes."""
    streak = 0
    for i, row in enumerate(run):
        streak = streak + 1 if row["success"] == "1" else 0
        if streak == length:
            return i - length + 2
    return "NA"


def first_optimal(run, optimal):
    for row in run:
        if int(row["steps"]) == optimal:
            return int(row["episode"])
    return "never"


def stats_row(folder, label, optimal):
    training, evaluation = load(folder)
    row = {"run": folder, "setting": label, "optimal_length": optimal}
    per_seed = []
    for run, ev in zip(training, evaluation):
        steps = [int(r["steps"]) for r in run]
        per_seed.append({
            "train_failures": sum(1 for r in run if r["success"] == "0"),
            "first_20_success_streak": first_success_streak(run),
            "first_optimal_episode": first_optimal(run, optimal),
            "train_steps_first100": statistics.mean(steps[:100]),
            "train_steps_last500": statistics.mean(steps[-500:]),
            "train_return_last500": statistics.mean(int(r["return"]) for r in run[-500:]),
            "total_train_actions": sum(steps),
            "eval_success_rate": statistics.mean(int(r["success"]) for r in ev),
            "eval_mean_steps": statistics.mean(int(r["steps"]) for r in ev),
            "eval_mean_return": statistics.mean(int(r["return"]) for r in ev),
        })
    for key in per_seed[0]:
        values = [p[key] for p in per_seed]
        if all(isinstance(v, (int, float)) for v in values):
            row[key + "_mean"] = round(statistics.mean(values), 2)
            row[key + "_sd"] = round(statistics.stdev(values), 2)
        else:
            row[key + "_mean"] = "/".join(str(v) for v in values)
            row[key + "_sd"] = ""
    return row


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_baseline():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    block = 10
    for (folder, label, optimal), color in zip(BASELINE, COLORS):
        training, _ = load(folder)
        curves = [block_means(run, "steps", block) for run in training]
        mean_curve = [statistics.mean(v) for v in zip(*curves)]
        x = [(i + 1) * block for i in range(len(mean_curve))]
        axes[0].plot(x[:40], mean_curve[:40], color=color, linewidth=2, label=label)
        axes[0].axhline(optimal, color=color, linestyle=":", linewidth=1)
        late = [statistics.mean(v) for v in zip(*[block_means(run, "steps", 100) for run in training])]
        axes[1].plot([(i + 1) * 100 for i in range(1, len(late))], late[1:], color=color,
                     linewidth=2, marker="o", markersize=3, label=label + " (optimal %d)" % optimal)
        axes[1].axhline(optimal, color=color, linestyle=":", linewidth=1.2)
    axes[0].set_title("Early training (first 400 episodes)", fontsize=11)
    axes[0].set_xlabel("Training episode")
    axes[0].set_ylabel("Mean actions per episode\n(10-episode blocks, 3 seeds)")
    axes[1].set_title("Episodes 101-2000 (100-episode blocks)", fontsize=11)
    axes[1].set_xlabel("Training episode")
    axes[1].set_ylabel("Mean actions per episode")
    axes[1].set_ylim(4, 32)
    for ax in axes:
        style(ax)
    axes[1].legend(frameon=False, fontsize=8.5, loc="lower right", ncol=1)
    axes[0].text(395, 160, "dotted lines = BFS optimum", fontsize=8, ha="right", color="#555")
    fig.suptitle("Part B: Q-learning on emu_campus.txt (alpha 0.3, gamma 0.95, epsilon 0.2)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS / "baseline_training_steps.png", dpi=200)
    plt.close(fig)


def plot_custom():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
    block = 100
    for (folder, label, optimal), color in zip(CUSTOM, COLORS):
        training, evaluation = load(folder)
        curves = [block_means(run, "return", block) for run in training]
        mean_curve = [statistics.mean(v) for v in zip(*curves)]
        sd_curve = [statistics.stdev(v) for v in zip(*curves)]
        x = [(i + 1) * block for i in range(len(mean_curve))]
        axes[0].plot(x, mean_curve, color=color, linewidth=2, marker="o", markersize=3, label=label)
        axes[0].fill_between(x, [m - s for m, s in zip(mean_curve, sd_curve)],
                             [m + s for m, s in zip(mean_curve, sd_curve)], color=color, alpha=0.18)
    axes[0].axhline(40 - 26, color="#555", linestyle=":", linewidth=1)
    axes[0].text(2000, 15.2, "best possible return = 14", fontsize=8, ha="right", color="#555")
    axes[0].set_title("Training return (100-episode blocks, mean +/- SD over 3 seeds)", fontsize=10)
    axes[0].set_xlabel("Training episode")
    axes[0].set_ylabel("Mean return per episode")
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")

    labels, train_late, eval_steps = [], [], []
    for folder, label, optimal in CUSTOM:
        training, evaluation = load(folder)
        labels.append(label.replace("epsilon = ", ""))
        train_late.append(statistics.mean(statistics.mean(int(r["steps"]) for r in run[-500:])
                                          for run in training))
        eval_steps.append(statistics.mean(statistics.mean(int(r["steps"]) for r in ev)
                                          for ev in evaluation))
    width = 0.38
    pos = range(len(labels))
    b1 = axes[1].bar([p - width / 2 for p in pos], train_late, width, color="#9bbbe8",
                     label="training, last 500 episodes")
    b2 = axes[1].bar([p + width / 2 for p in pos], eval_steps, width, color="#2a6fdb",
                     label="greedy evaluation")
    for bars in (b1, b2):
        for bar in bars:
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                         "%.1f" % bar.get_height(), ha="center", fontsize=8)
    axes[1].set_xticks(list(pos))
    axes[1].set_xticklabels(labels)
    axes[1].set_xlabel("Exploration rate epsilon")
    axes[1].set_ylabel("Mean actions per delivery")
    axes[1].set_title("Training vs. evaluation steps", fontsize=10)
    axes[1].set_ylim(0, 65)
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    for ax in axes:
        style(ax)
    fig.suptitle("Part C: effect of epsilon on my_campus.txt, S-W (alpha 0.3, gamma 0.95)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS / "custom_epsilon.png", dpi=200)
    plt.close(fig)


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    rows = [stats_row(*spec) for spec in BASELINE + CUSTOM]
    with open(RESULTS / "training_stats.csv", "w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    plot_baseline()
    plot_custom()
    print("Wrote results/training_stats.csv and plots in", PLOTS)


if __name__ == "__main__":
    main()
