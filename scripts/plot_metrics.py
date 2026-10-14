"""Plot learning results and separate diagnostics from actual experiment files.

Default: aggregate completed Counterexample runs. --run_dir selects one run;
repeating --run_dir selects an explicit set. Missing metrics remain missing.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

COLORS = {"train": "#3269c8", "val": "#16877b", "base": "#a3adbd",
          "test": "#6355b1", "zero": "#c36b59", "mixed": "#8b73b5"}


def load_runs(paths):
    runs = []
    for path in paths:
        root = Path(path)
        records = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()
                   if line.strip()]
        if not records:
            raise ValueError(f"No metrics in {root}")
        by_step = {}
        for record in records:
            step = record["step"]
            if step in by_step:
                raise ValueError(f"Duplicate step {step} in {root}; select an unambiguous run")
            by_step[step] = record
        manifest = root / "run_manifest.json"
        config = json.loads(manifest.read_text()).get("config", {}) if manifest.exists() else {}
        config.setdefault("dynamic_resampling", False)
        report = root / "final_test_report.json"
        runs.append({"root": root, "records": by_step, "config": config,
                     "report": json.loads(report.read_text()) if report.exists() else {},
                     "label": f"Seed {config['seed']}" if "seed" in config else root.name})
    if not runs:
        raise ValueError("No runs selected")
    for key in ("model_name", "task", "group_size", "anchor_size", "problems_per_update",
                "max_new_tokens", "temperature", "top_p", "dynamic_resampling"):
        values = {json.dumps(run["config"].get(key), sort_keys=True) for run in runs}
        if len(values) != 1:
            raise ValueError(f"Cannot combine runs with different {key}")
    return runs


def metric_matrix(runs, key, scale=1):
    steps = sorted({step for run in runs for step in run["records"]})
    values = np.full((len(runs), len(steps)), np.nan)
    for i, run in enumerate(runs):
        for j, step in enumerate(steps):
            value = run["records"].get(step, {}).get(key)
            if value is not None:
                values[i, j] = float(value) * scale
    return np.asarray(steps), values


def mean_std(values):
    """Column-wise mean and sample SD; an unavailable SD stays NaN."""
    means = np.full(values.shape[1], np.nan)
    stds = np.full(values.shape[1], np.nan)
    for j in range(values.shape[1]):
        finite = values[:, j][np.isfinite(values[:, j])]
        if len(finite):
            means[j] = finite.mean()
        if len(finite) > 1:
            stds[j] = finite.std(ddof=1)
    return means, stds


def draw_series(ax, runs, key, color, label=None, scale=1, linestyle="-", bounded=False):
    steps, values = metric_matrix(runs, key, scale)
    mean, std = mean_std(values)
    if not np.isfinite(mean).any():
        return False
    ax.plot(steps, mean, color=color, label=label, linewidth=2.1, linestyle=linestyle)
    low, high = mean - std, mean + std
    if bounded:
        low, high = np.clip(low, 0, 100), np.clip(high, 0, 100)
    ax.fill_between(steps, low, high, color=color, alpha=.12, linewidth=0)
    return True


def style_axis(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=13, color="#263449")
    ax.set_ylabel(ylabel, fontsize=10, color="#546174", labelpad=8)
    ax.set_xlabel("Global step", fontsize=10, color="#546174", labelpad=8)
    ax.spines[["top", "right"]].set_visible(False)
    for name in ("bottom", "left"):
        ax.spines[name].set_color("#d7dee8")
    ax.tick_params(colors="#68758a", labelsize=9, length=0, pad=6)
    ax.grid(axis="y", color="#e8edf4", linewidth=.7)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))


def test_values(runs, kind):
    values = []
    for run in runs:
        result = run["report"].get("blind_test_evaluation", {})
        if kind == "base":
            value = (result.get("base") or {}).get("problem_pass_rate")
        else:
            selected = (result.get("checkpoint_best") or {}).get(kind, {})
            value = selected.get("problem_pass_rate")
        if value is not None:
            values.append(float(value) * 100)
    return np.asarray(values)


def draw_test(ax, runs):
    ax.set_title("Held-out test", loc="left", fontsize=12, fontweight="bold", pad=13, color="#263449")
    ax.set_xlim(0, 105); ax.set_ylim(-.5, 3.3)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#d7dee8")
    ax.tick_params(axis="x", colors="#68758a", labelsize=9, length=0, pad=6)
    ax.xaxis.set_major_formatter(ticker.PercentFormatter())
    ax.xaxis.set_major_locator(ticker.MultipleLocator(25))
    ax.set_xlabel("Solved problems", fontsize=10, color="#546174", labelpad=8)
    ax.grid(axis="x", color="#e8edf4", linewidth=.7); ax.set_axisbelow(True)
    for y, key, label, color in ((2.6, "base", "Base / greedy", COLORS["base"]),
                                (1.4, "greedy", "GRPO / greedy", COLORS["test"]),
                                (.2, "sample", "GRPO / sample@4", COLORS["val"])):
        values = test_values(runs, key)
        ax.text(0, y+.36, label, fontsize=10, color="#546174")
        if not len(values):
            ax.text(100, y+.36, "Not evaluated", ha="right", fontsize=10, color="#68758a")
            continue
        mean = values.mean(); std = values.std(ddof=1) if len(values) > 1 else None
        ax.barh(y, mean, height=.28, color=color, alpha=.85, zorder=3)
        if std is not None:
            ax.errorbar(mean, y, xerr=std, fmt="none", ecolor="#263449", capsize=4, linewidth=1.3, zorder=5)
        offsets = np.linspace(-.075, .075, len(values)) if len(values) > 1 else np.zeros(1)
        ax.scatter(values, y+offsets, color="white", edgecolors=color, s=25, linewidth=.9, zorder=6)
        score = f"{mean:.1f}%" + (f" ± {std:.1f}%" if std is not None else "")
        if len(values) != len(runs):
            score += f" (n={len(values)})"
        ax.text(100, y+.36, score, ha="right", fontsize=10, fontweight="bold", color="#263449")


def figure_header(fig, runs, title):
    model = runs[0]["config"].get("model_name", "Model not recorded").split("/")[-1]
    if all("seed" in run["config"] for run in runs):
        labels = ", ".join(str(run["config"]["seed"]) for run in runs)
        seed_word = "seed" if len(runs) == 1 else "seeds"
        run_label = f"{len(runs)} training {seed_word}: {labels}"
    else:
        run_label = f"{len(runs)} training run(s)"
    fig.text(.07, .955, title, fontsize=19, fontweight="bold", color="#1d2d43", va="top")
    fig.text(.07, .893, f"{model}  ·  {run_label}",
             fontsize=10, color="#68758a", va="top")


def save_figure(fig, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, facecolor="white")
    # Provide a vector version for export as well as the README's PNG.
    if output.suffix.lower() != ".svg":
        fig.savefig(output.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    print(f"Saved {output}")


def plot_runs(paths, output_path, diagnostics_path=None):
    runs = load_runs(paths)
    with plt.rc_context({"font.family": "DejaVu Sans", "svg.fonttype": "path"}):
        fig = plt.figure(figsize=(12, 4.9))
        axes = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.6, 1]})
        fig.subplots_adjust(left=.07, right=.97, bottom=.21, top=.77, wspace=.24)
        figure_header(fig, runs, "nano-GRPO / Learning to find counterexamples")
        ax = axes[0]
        style_axis(ax, "Learning curves", "Success rate")
        draw_series(ax, runs, "anchor_pass_rate", COLORS["train"], "Train / sampled candidates", 100, bounded=True)
        draw_series(ax, runs, "val_acc", COLORS["val"], "Validation / greedy", 100, "--", bounded=True)
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter())
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper left", frameon=False, fontsize=9, handlelength=2)
        draw_test(axes[1], runs)
        curve_note = "Curves: mean ± sample SD across available runs." if len(runs) > 1 else "Curves: single-run observations."
        fig.text(.07, .07, curve_note + " Test: validation-selected checkpoint-best.",
                 fontsize=9, color="#68758a")
        test_note = "Test dots: individual runs. Whiskers: sample SD." if len(runs) > 1 else "Test dots: one run; across-run SD unavailable."
        fig.text(.07, .035, test_note + " sample@4 uses four attempts per problem.",
                 fontsize=9, color="#68758a")
        save_figure(fig, output_path)

        output = Path(output_path)
        diagnostics = diagnostics_path or output.with_name(output.stem + "_diagnostics" + output.suffix)
        fig, axes = plt.subplots(2, 3, figsize=(12, 7.2))
        fig.subplots_adjust(left=.07, right=.97, bottom=.13, top=.8, wspace=.33, hspace=.55)
        figure_header(fig, runs, "nano-GRPO / Training diagnostics")
        specs = [("Reward groups", "Problems", None, None),
                 ("Gradient norm", "Norm before clipping", "grad_norm", "#ba8050"),
                 ("GRPO loss", "Token-normalized loss", "avg_loss", COLORS["train"]),
                 ("Reference KL", "Sampled k3 estimate", "approx_kl", COLORS["test"]),
                 ("Completion length", "Tokens per candidate", "avg_len", COLORS["val"]),
                 ("Reference KL / individual runs", "Sampled k3 estimate", None, None)]
        for ax, (title, ylabel, key, color) in zip(axes.flat, specs):
            style_axis(ax, title, ylabel)
            if key:
                present = draw_series(ax, runs, key, color)
                if not present:
                    ax.text(.5, .5, "Not recorded", transform=ax.transAxes, ha="center", color="#68758a")
                if key == "avg_loss":
                    ax.axhline(0, color="#b6c0cc", linewidth=.8, linestyle="--")
                else:
                    ax.set_ylim(bottom=0)
        for key, color, label in (("all_one_problems", COLORS["val"], "All correct"),
                                   ("all_zero_problems", COLORS["zero"], "All wrong"),
                                   ("effective_problems", COLORS["mixed"], "Mixed rewards")):
            draw_series(axes[0, 0], runs, key, color, label)
        axes[0, 0].set_ylim(bottom=0)
        problem_count = runs[0]["config"].get("anchor_size")
        if problem_count:
            axes[0, 0].set_ylim(0, problem_count)
        if axes[0, 0].get_legend_handles_labels()[0]:
            axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
        steps, values = metric_matrix(runs, "approx_kl")
        for i, run in enumerate(runs):
            if np.isfinite(values[i]).any():
                axes[1, 2].plot(steps, values[i], label=run["label"], linewidth=1.7)
        if axes[1, 2].get_legend_handles_labels()[0]:
            axes[1, 2].legend(frameon=False, fontsize=8, loc="upper left")
        else:
            axes[1, 2].text(.5, .5, "Not recorded", transform=axes[1, 2].transAxes,
                            ha="center", color="#68758a")
        axes[1, 2].set_ylim(bottom=0)
        diagnostic_note = "Shading: ±1 sample SD; individual KL traces expose divergence hidden by the mean." if len(runs) > 1 else "Single-run observations; no across-run standard deviation."
        fig.text(.07, .04, diagnostic_note + " No smoothing or dual axes.",
                 fontsize=9, color="#68758a")
        save_figure(fig, diagnostics)
    return runs


def plot_single_run(run_dir, output_path):
    return plot_runs([run_dir], output_path)


def plot_multi_seed(runs, output_path):
    return plot_runs(runs, output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", action="append", help="Select run directory; repeat to compare runs")
    parser.add_argument("--output", default="assets/training_curves.png")
    parser.add_argument("--diagnostics_output", help="Optional separate diagnostic figure path")
    args = parser.parse_args()
    paths = args.run_dir or [p for p in sorted(Path("checkpoints_counterexample").glob("20*"))
                            if (p / "final_test_report.json").is_file()]
    plot_runs(paths, args.output, args.diagnostics_output)


if __name__ == "__main__":
    main()
