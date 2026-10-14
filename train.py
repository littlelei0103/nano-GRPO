"""
train.py - 基于在线流式微步探索的可验证奖励强化学习主入口 (Online Streaming GRPO)

使用示例:
  # 启动 Nano Counterexample 在线流式强化学习训练 (默认任务与模型)
  python train.py --task counterexample --total_steps 100 --lr 2e-5

  # 启动 BugHunter 代码漏洞反例审计挑战集训练
  python train.py --task bug_hunter --total_steps 100 --lr 2e-5
"""

import argparse
import torch
from trainer.evaluation import evaluate_benchmark


def parse_args():
    parser = argparse.ArgumentParser(description="nano-GRPO: Online Streaming RLVR Training Engine")

    # 任务与模型
    parser.add_argument("--task", type=str, default="counterexample", choices=["counterexample", "bug_hunter", "bug_differential", "code_exec"],
                        help="选择训练任务: counterexample (差分程序反例挖掘), bug_hunter, bug_differential (输入差分执行), code_exec")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B",
                        help="基座模型名称或本地权重路径 (默认 Qwen/Qwen2.5-1.5B)")
    parser.add_argument("--resume_checkpoint", type=str, default="",
                        help="恢复已有 LoRA 权重的目录路径 (精确续训同一实验)")
    parser.add_argument("--init_adapter", type=str, default="",
                        help="初始化已有预训练 LoRA 适配器权重目录 (跨任务迁移学习热启动，开启新实验)")
    parser.add_argument("--data_path", type=str, default="",
                        help="自定义题目数据集 JSON 路径 (默认自动加载对应轨道数据集)")
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                        help="检查点保存目录")

    # 强化学习与训练超参数
    parser.add_argument("--group_size", type=int, default=8,
                        help="每个 Prompt 采样的候选数量 G (默认 8)")
    parser.add_argument("--micro_batch_size", type=int, default=8,
                        help="采样与前向微批次大小，设为 8 实现单次前向极速吞吐 (默认 8)")
    parser.add_argument("--problems_per_update", type=int, default=1,
                        help="每次更新覆盖的题目数 (流式训练默认为 1)")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="AdamW 学习率 (默认 2e-5)")
    parser.add_argument("--kl_coef", type=float, default=0.01,
                        help="KL 散度惩罚系数 beta (默认 0.01)")
    parser.add_argument("--clip_eps", type=float, default=0.2,
                        help="PPO 策略裁剪范围 epsilon (默认 0.2)")
    parser.add_argument("--max_new_tokens", type=int, default=32,
                        help="自回归生成的最大 Token 长度 (默认 32)")

    parser.add_argument("--temperature", type=float, default=0.7,
                        help="采样温度 (默认 0.7)")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="完整支持集采样，训练要求 top_p=1.0")

    parser.add_argument("--seed", type=int, default=42, help="训练随机种子")

    # 步数与记录控制
    parser.add_argument("--total_steps", type=int, default=200,
                        help="总训练步数 (默认 200)")
    parser.add_argument("--eval_interval", type=int, default=20,
                        help="诊断切片与检查点保存周期 (默认 20)")
    parser.add_argument("--skip_baseline", action="store_true",
                        help="是否跳过初始 Step 0 的零样本基线评测 (用于快速测试)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="运行设备 (cuda 或 cpu)")

    parser.add_argument("--dynamic_resampling", action="store_true",
                        help="启用 DAPO 风格动态零方差重采样 (Dynamic Variance Rescue)")
    parser.add_argument("--max_resample_rounds", type=int, default=2,
                        help="零方差组最大重采样轮数 (默认 2)")
    parser.add_argument("--resample_temp_boost", type=float, default=0.0, choices=[0.0],
                        help="兼容参数，只允许 0.0；重采样强制使用原策略温度")

    return parser.parse_args()


def evaluate_val_benchmark(model, tokenizer, task, max_new_tokens=300, device="cuda"):
    result = evaluate_benchmark(model, tokenizer, task.get_val_dataset(), task, max_new_tokens, device)
    return result["problem_pass_rate"] or 0.0, result["solved_problems"], result["total_problems"]


def main():
    from trainer.run import run_training
    run_training(parse_args(), macro=False)


if __name__ == "__main__":
    main()
