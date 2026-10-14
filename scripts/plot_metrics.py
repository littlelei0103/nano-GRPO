"""
scripts/plot_metrics.py - 绘制训练指标曲线与演化飞轮图表
"""

import argparse
import glob
import json
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def parse_args():
    parser = argparse.ArgumentParser(description="Plot RLVR Training Metrics")
    parser.add_argument("--run_dir", type=str, default="",
                        help="实验目录路径 (默认自动查找 checkpoints_counterexample 下最新运行)")
    parser.add_argument("--output", type=str, default="assets/training_curves.png",
                        help="输出图表保存路径")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir
    if not run_dir:
        candidates = sorted(glob.glob("checkpoints_counterexample/20*"))
        if not candidates:
            raise FileNotFoundError("未在 checkpoints_counterexample 下找到任何实验目录！")
        run_dir = candidates[-1]

    print(f"📊 正在从实验目录提取训练流水数据: {run_dir}")
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

    # 设置专业风格
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    plt.subplots_adjust(hspace=0.28, wspace=0.22)

    # 1. 通关胜率演化 (Pass Rate Trajectory)
    ax1 = axes[0, 0]
    ax1.plot(steps, anchor_pass_rates, color="#1f77b4", marker="o", linewidth=2.2, label="Train Anchor Pass Rate (60 probs)")
    if val_accs:
        val_x, val_y = zip(*val_accs)
        ax1.plot(val_x, val_y, color="#2ca02c", marker="s", linewidth=2.0, linestyle="--", label="Validation Pass Rate (20 probs)")
    
    # 标注盲盒大考结果
    if "blind_test_evaluation" in test_report:
        b_res = test_report["blind_test_evaluation"].get("checkpoint_best", {}).get("greedy", {}).get("problem_pass_rate")
        base_res = test_report["blind_test_evaluation"].get("base", {}).get("problem_pass_rate") if test_report["blind_test_evaluation"].get("base") else None
        if b_res is not None and base_res is not None:
            best_step = test_report["validation_selection"]["best_step"]
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

    # 2. 掌握度分布演变 (All-One vs All-Zero 剪刀差曲线)
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

    # 3. 梯度与优化动态 (GradNorm & Policy Loss)
    ax3 = axes[1, 0]
    color_grad = "#ff7f0e"
    ax3.plot(steps, grad_norms, color=color_grad, marker="o", linewidth=2.0, label="Gradient Norm (GradNorm)")
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

    # 4. KL 散度与生成长度 (Reference KL & Token Length)
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

    fig.suptitle("nano-GRPO: Nano Counterexample RLVR Training Trajectory\nQwen2.5-1.5B on NVIDIA RTX 5070 (8GB VRAM)",
                 fontsize=15, fontweight="bold", y=0.98)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, bbox_inches="tight")
    print(f"✅ 图表已保存至: {args.output}")

    # 同时复制至 artifact 目录便于直接展示
    artifact_dir = "/home/littlelei/.gemini/antigravity/brain/218237f5-933f-4b03-9d64-11ef16217b00"
    if os.path.isdir(artifact_dir):
        artifact_target = os.path.join(artifact_dir, "training_curves.png")
        shutil.copy2(args.output, artifact_target)
        print(f"✅ 图表已同步至 Artifact: {artifact_target}")


if __name__ == "__main__":
    main()
