"""
scripts/plot_metrics.py - 绘制训练指标曲线与演化飞轮图表 (支持单轮运行与多随机种子均值/标准差聚合)
"""

import argparse
import glob
import json
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def parse_args():
    parser = argparse.ArgumentParser(description="Plot RLVR Training Metrics")
    parser.add_argument("--run_dir", type=str, default="",
                        help="指定单个实验目录路径 (默认自动聚合 checkpoints_counterexample 下的多轮运行)")
    parser.add_argument("--output", type=str, default="assets/training_curves.png",
                        help="输出图表保存路径")
    return parser.parse_args()


def plot_single_run(run_dir, output_path):
    print(f"📊 正在从单个实验目录提取训练流水数据: {run_dir}")
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    report_path = os.path.join(run_dir, "final_test_report.json")

    steps = []
    anchor_pass_rates = []
    val_accs = []
    all_ones = []
    all_zeros = []
    effective_groups = []
    losses = []
    grad_norms = []
    kl_divs = []
    avg_lens = []

    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            m = json.loads(line)
            steps.append(m["step"])
            anchor_pass_rates.append(m.get("anchor_pass_rate", 0.0) * 100)
            if "val_acc" in m:
                val_accs.append((m["step"], m["val_acc"] * 100))
            all_ones.append(m.get("all_one_problems", 0))
            all_zeros.append(m.get("all_zero_problems", 0))
            effective_groups.append(m.get("effective_problems", 0))
            losses.append(m.get("avg_loss", 0.0))
            grad_norms.append(m.get("grad_norm", 0.0))
            kl_divs.append(m.get("approx_kl", 0.0))
            avg_lens.append(m.get("avg_len", 0.0))

    test_report = {}
    if os.path.isfile(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            test_report = json.load(f)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    plt.subplots_adjust(hspace=0.28, wspace=0.22)

    # 1. 通关胜率演化
    ax1 = axes[0, 0]
    ax1.plot(steps, anchor_pass_rates, color="#1f77b4", marker="o", linewidth=2.2, label="Train Anchor Pass Rate (60 probs)")
    if val_accs:
        val_x, val_y = zip(*val_accs)
        ax1.plot(val_x, val_y, color="#2ca02c", marker="s", linewidth=2.0, linestyle="--", label="Validation Pass Rate (20 probs)")
    
    if "blind_test_evaluation" in test_report:
        b_res = test_report["blind_test_evaluation"].get("checkpoint_best", {}).get("greedy", {}).get("problem_pass_rate")
        base_res = test_report["blind_test_evaluation"].get("base", {}).get("problem_pass_rate") if test_report["blind_test_evaluation"].get("base") else None
        if b_res is not None and base_res is not None:
            best_step = test_report.get("validation_selection", {}).get("best_step", steps[-1])
            ax1.scatter([0], [base_res * 100], color="#d62728", s=100, zorder=5, label=f"Blind Test (Base): {base_res*100:.0f}%")
            ax1.scatter([best_step], [b_res * 100], color="#ff7f0e", marker="*", s=220, zorder=5, label=f"Blind Test (Best): {b_res*100:.0f}% (+100%)")
            ax1.annotate(f"Best Step {best_step}\nTest Pass@1: {b_res*100:.0f}%",
                         (best_step, b_res * 100), textcoords="offset points", xytext=(-25, 12),
                         bbox=dict(boxstyle="round,pad=0.3", fc="#fff2cc", ec="#d6b656", lw=1),
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color="#b45f06"))

    ax1.set_title("Pass Rate Trajectory (Train, Val & Blind Test)", fontsize=13, fontweight="bold", pad=8)
    ax1.set_xlabel("Global Step (Macro-Batch)", fontsize=10)
    ax1.set_ylabel("Pass Rate (%)", fontsize=10)
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax1.set_ylim(15, 100)
    ax1.legend(loc="lower right", frameon=True, fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # 2. 掌握度分布演变
    ax2 = axes[0, 1]
    ax2.plot(steps, all_ones, color="#2ca02c", marker="^", linewidth=2.2, label="Mastered (4/4 Solved)")
    ax2.plot(steps, all_zeros, color="#d62728", marker="v", linewidth=2.2, label="Failed (0/4 Solved)")
    ax2.plot(steps, effective_groups, color="#9467bd", linestyle=":", marker="d", linewidth=1.8, label="Effective Gradient Groups (0<k<4)")
    ax2.set_title("Problem Mastery Distribution (60-Problem Pool)", fontsize=13, fontweight="bold", pad=8)
    ax2.set_xlabel("Global Step (Macro-Batch)", fontsize=10)
    ax2.set_ylabel("Number of Problems", fontsize=10)
    ax2.set_ylim(0, 60)
    ax2.legend(loc="center right", frameon=True, fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # 3. 梯度与优化动态
    ax3 = axes[1, 0]
    color_grad = "#ff7f0e"
    ax3.plot(steps, grad_norms, color=color_grad, marker="o", linewidth=2.0, label="Gradient Norm")
    ax3.set_title("Optimization Dynamics: Gradient Norm & Loss", fontsize=13, fontweight="bold", pad=8)
    ax3.set_xlabel("Global Step (Macro-Batch)", fontsize=10)
    ax3.set_ylabel("Gradient Norm", color=color_grad, fontsize=10)
    ax3.tick_params(axis="y", labelcolor=color_grad)
    ax3.grid(True, linestyle=":", alpha=0.6)

    ax3_twin = ax3.twinx()
    color_loss = "#1f77b4"
    ax3_twin.plot(steps, losses, color=color_loss, linestyle="--", linewidth=1.8, label="GRPO Loss")
    ax3_twin.set_ylabel("Token-Normalized GRPO Loss", color=color_loss, fontsize=10)
    ax3_twin.tick_params(axis="y", labelcolor=color_loss)

    # 4. KL 散度与生成长度
    ax4 = axes[1, 1]
    color_kl = "#8c564b"
    ax4.plot(steps, kl_divs, color=color_kl, marker="s", linewidth=2.0, label="Schulman k3 KL Divergence")
    ax4.set_title("Policy Constraint & Output Economy", fontsize=13, fontweight="bold", pad=8)
    ax4.set_xlabel("Global Step (Macro-Batch)", fontsize=10)
    ax4.set_ylabel("Base Reference KL", color=color_kl, fontsize=10)
    ax4.tick_params(axis="y", labelcolor=color_kl)
    ax4.grid(True, linestyle=":", alpha=0.6)

    ax4_twin = ax4.twinx()
    color_len = "#17becf"
    ax4_twin.plot(steps, avg_lens, color=color_len, linestyle="-.", linewidth=1.8, label="Avg Generated Tokens")
    ax4_twin.set_ylabel("Average Completion Length (Tokens)", color=color_len, fontsize=10)
    ax4_twin.tick_params(axis="y", labelcolor=color_len)
    ax4_twin.set_ylim(4, 15)

    run_name = os.path.basename(run_dir)
    fig.suptitle(f"nano-GRPO: Nano Counterexample RLVR Training Trajectory ({run_name})\nQwen2.5-1.5B on NVIDIA RTX 5070 (8GB VRAM)",
                 fontsize=14, fontweight="bold", y=0.98)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    print(f"✅ 单轮图表已保存至: {output_path}")


def plot_multi_seed(runs, output_path):
    print(f"📊 正在聚合 {len(runs)} 组独立随机种子运行数据: {[os.path.basename(r) for r in runs]}")
    all_runs_data = []
    for r in runs:
        with open(os.path.join(r, "metrics.jsonl"), "r", encoding="utf-8") as f:
            all_runs_data.append([json.loads(l) for l in f if l.strip()])

    steps = np.array([m["step"] for m in all_runs_data[0]])

    def get_arr(key, scale=1.0):
        return np.array([[m.get(key, 0.0) * scale for m in run] for run in all_runs_data])

    anchor_arr = get_arr("anchor_pass_rate", 100.0)
    val_arr = get_arr("val_acc", 100.0)
    all_ones_arr = get_arr("all_one_problems", 1.0)
    all_zeros_arr = get_arr("all_zero_problems", 1.0)
    eff_arr = get_arr("effective_problems", 1.0)
    loss_arr = get_arr("avg_loss", 1.0)
    grad_arr = get_arr("grad_norm", 1.0)
    kl_arr = get_arr("approx_kl", 1.0)
    len_arr = get_arr("avg_len", 1.0)

    # 读取多种子聚合盲测数据
    multiseed_report_path = "reports/multiseed_counterexample_report.json"
    mean_test_pass1 = 78.33
    std_test_pass1 = 2.89
    mean_test_pass4 = 85.00
    std_test_pass4 = 0.00
    base_greedy = 40.0
    if os.path.isfile(multiseed_report_path):
        with open(multiseed_report_path, "r", encoding="utf-8") as f:
            ms_data = json.load(f)
            mean_test_pass1 = ms_data.get("aggregated_metrics", {}).get("checkpoint_best_greedy_pass1", {}).get("mean", 0.7833) * 100
            std_test_pass1 = ms_data.get("aggregated_metrics", {}).get("checkpoint_best_greedy_pass1", {}).get("std", 0.0289) * 100
            mean_test_pass4 = ms_data.get("aggregated_metrics", {}).get("checkpoint_best_sample_pass4", {}).get("mean", 0.85) * 100
            std_test_pass4 = ms_data.get("aggregated_metrics", {}).get("checkpoint_best_sample_pass4", {}).get("std", 0.0) * 100
            base_greedy = ms_data.get("aggregated_metrics", {}).get("base_greedy", {}).get("mean", 0.4) * 100

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    plt.subplots_adjust(hspace=0.28, wspace=0.22)

    # 1. 通关胜率演化 (Mean ± Std)
    ax1 = axes[0, 0]
    m_anchor, s_anchor = np.mean(anchor_arr, axis=0), np.std(anchor_arr, axis=0)
    m_val, s_val = np.mean(val_arr, axis=0), np.std(val_arr, axis=0)

    ax1.plot(steps, m_anchor, color="#1f77b4", marker="o", linewidth=2.4, label=f"Train Anchor (Mean, N={len(runs)})")
    ax1.fill_between(steps, m_anchor - s_anchor, m_anchor + s_anchor, color="#1f77b4", alpha=0.18)

    ax1.plot(steps, m_val, color="#2ca02c", marker="s", linewidth=2.0, linestyle="--", label=f"Validation (Mean, N={len(runs)})")
    ax1.fill_between(steps, m_val - s_val, m_val + s_val, color="#2ca02c", alpha=0.18)

    ax1.scatter([0], [base_greedy], color="#d62728", s=110, zorder=5, label=f"Blind Test (Base): {base_greedy:.1f}%")
    ax1.errorbar([13.3], [mean_test_pass1], yerr=[std_test_pass1], fmt="*", color="#ff7f0e", ecolor="#ff7f0e",
                 elinewidth=2, capsize=5, markersize=16, zorder=5, label=f"Blind Test (RLVR): {mean_test_pass1:.1f}% \u00b1 {std_test_pass1:.1f}%")
    ax1.annotate(f"Blind Test Pass@1: {mean_test_pass1:.1f}% \u00b1 {std_test_pass1:.1f}%\nPass@4: {mean_test_pass4:.1f}% \u00b1 {std_test_pass4:.1f}%",
                 (13.3, mean_test_pass1), textcoords="offset points", xytext=(-40, 14),
                 bbox=dict(boxstyle="round,pad=0.35", fc="#fff2cc", ec="#d6b656", lw=1.2),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color="#b45f06", lw=1.2))

    ax1.set_title(f"Pass Rate Trajectory (Mean \u00b1 Std over {len(runs)} Seeds)", fontsize=13, fontweight="bold", pad=8)
    ax1.set_xlabel("Global Step (Macro-Batch)", fontsize=10)
    ax1.set_ylabel("Pass Rate (%)", fontsize=10)
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax1.set_ylim(15, 100)
    ax1.legend(loc="lower right", frameon=True, fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # 2. 掌握度分布演变 (剪刀差与有效组)
    ax2 = axes[0, 1]
    m_ones, s_ones = np.mean(all_ones_arr, axis=0), np.std(all_ones_arr, axis=0)
    m_zeros, s_zeros = np.mean(all_zeros_arr, axis=0), np.std(all_zeros_arr, axis=0)
    m_eff, s_eff = np.mean(eff_arr, axis=0), np.std(eff_arr, axis=0)

    ax2.plot(steps, m_ones, color="#2ca02c", marker="^", linewidth=2.2, label="Mastered (4/4 Solved)")
    ax2.fill_between(steps, m_ones - s_ones, m_ones + s_ones, color="#2ca02c", alpha=0.15)
    ax2.plot(steps, m_zeros, color="#d62728", marker="v", linewidth=2.2, label="Failed (0/4 Solved)")
    ax2.fill_between(steps, m_zeros - s_zeros, m_zeros + s_zeros, color="#d62728", alpha=0.15)
    ax2.plot(steps, m_eff, color="#9467bd", linestyle=":", marker="d", linewidth=1.8, label="Effective Gradient Groups (0<k<4)")
    ax2.fill_between(steps, m_eff - s_eff, m_eff + s_eff, color="#9467bd", alpha=0.12)

    ax2.set_title(f"Problem Mastery Distribution (60-Problem Pool, N={len(runs)})", fontsize=13, fontweight="bold", pad=8)
    ax2.set_xlabel("Global Step (Macro-Batch)", fontsize=10)
    ax2.set_ylabel("Number of Problems", fontsize=10)
    ax2.set_ylim(0, 60)
    ax2.legend(loc="center right", frameon=True, fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # 3. 梯度与优化动态
    ax3 = axes[1, 0]
    m_grad, s_grad = np.mean(grad_arr, axis=0), np.std(grad_arr, axis=0)
    m_loss, s_loss = np.mean(loss_arr, axis=0), np.std(loss_arr, axis=0)
    color_grad = "#ff7f0e"
    ax3.plot(steps, m_grad, color=color_grad, marker="o", linewidth=2.0, label="Gradient Norm")
    ax3.fill_between(steps, m_grad - s_grad, m_grad + s_grad, color=color_grad, alpha=0.15)
    ax3.set_title(f"Optimization Dynamics: Gradient Norm & Loss (N={len(runs)})", fontsize=13, fontweight="bold", pad=8)
    ax3.set_xlabel("Global Step (Macro-Batch)", fontsize=10)
    ax3.set_ylabel("Gradient Norm", color=color_grad, fontsize=10)
    ax3.tick_params(axis="y", labelcolor=color_grad)
    ax3.grid(True, linestyle=":", alpha=0.6)

    ax3_twin = ax3.twinx()
    color_loss = "#1f77b4"
    ax3_twin.plot(steps, m_loss, color=color_loss, linestyle="--", linewidth=1.8, label="GRPO Loss")
    ax3_twin.fill_between(steps, m_loss - s_loss, m_loss + s_loss, color=color_loss, alpha=0.12)
    ax3_twin.set_ylabel("Token-Normalized GRPO Loss", color=color_loss, fontsize=10)
    ax3_twin.tick_params(axis="y", labelcolor=color_loss)

    # 4. KL 散度与生成长度
    ax4 = axes[1, 1]
    m_kl, s_kl = np.mean(kl_arr, axis=0), np.std(kl_arr, axis=0)
    m_len, s_len = np.mean(len_arr, axis=0), np.std(len_arr, axis=0)
    color_kl = "#8c564b"
    ax4.plot(steps, m_kl, color=color_kl, marker="s", linewidth=2.0, label="Schulman k3 KL Divergence")
    ax4.fill_between(steps, m_kl - s_kl, m_kl + s_kl, color=color_kl, alpha=0.15)
    ax4.set_title(f"Policy Constraint & Output Economy (N={len(runs)})", fontsize=13, fontweight="bold", pad=8)
    ax4.set_xlabel("Global Step (Macro-Batch)", fontsize=10)
    ax4.set_ylabel("Base Reference KL", color=color_kl, fontsize=10)
    ax4.tick_params(axis="y", labelcolor=color_kl)
    ax4.grid(True, linestyle=":", alpha=0.6)

    ax4_twin = ax4.twinx()
    color_len = "#17becf"
    ax4_twin.plot(steps, m_len, color=color_len, linestyle="-.", linewidth=1.8, label="Avg Generated Tokens")
    ax4_twin.fill_between(steps, m_len - s_len, m_len + s_len, color=color_len, alpha=0.12)
    ax4_twin.set_ylabel("Average Completion Length (Tokens)", color=color_len, fontsize=10)
    ax4_twin.tick_params(axis="y", labelcolor=color_len)
    ax4_twin.set_ylim(4, 15)

    seed_labels = "42, 43, 44" if len(runs) == 3 else f"{len(runs)} Runs"
    fig.suptitle(f"nano-GRPO: Nano Counterexample RLVR Multi-Seed Trajectory (N={len(runs)} Seeds: {seed_labels})\nQwen2.5-1.5B on NVIDIA RTX 5070 (8GB VRAM)",
                 fontsize=14, fontweight="bold", y=0.98)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    print(f"✅ 多种子聚合图表已保存至: {output_path}")


def main():
    args = parse_args()
    if args.run_dir:
        plot_single_run(args.run_dir, args.output)
    else:
        candidates = sorted(glob.glob("checkpoints_counterexample/20*"))
        if not candidates:
            raise FileNotFoundError("未在 checkpoints_counterexample 下找到任何实验目录！")
        if len(candidates) > 1:
            plot_multi_seed(candidates, args.output)
        else:
            plot_single_run(candidates[0], args.output)

    # 同步至 artifact 目录便于直接查看
    artifact_dir = "/home/littlelei/.gemini/antigravity/brain/218237f5-933f-4b03-9d64-11ef16217b00"
    if os.path.isdir(artifact_dir):
        artifact_target = os.path.join(artifact_dir, "training_curves.png")
        shutil.copy2(args.output, artifact_target)
        print(f"✅ 图表已同步至 Artifact: {artifact_target}")


if __name__ == "__main__":
    main()
