"""
train_mimo.py - 基于宏步梯度累积的可验证奖励强化学习主入口 (Macro-Batch RLVR Training)

核心架构:
  1. 纯净三相数据隔离: Train (60 题锚点池), Val (20 题验证选模), Test (20 题独立盲盒测试集)
  2. 真实基座参考 KL 正则化与 Token-normalized GRPO 联合损失
  3. 双重验证机制: Validation 集用于筛选最优权重 (checkpoint-best)，Test 集全流程隔离仅在训练终期执行无偏泛化评测
  4. 多维指标体系: 确定性贪心解 Pass@1 与 随机采样 Pass@k / Avg Pass Rate
  5. 完备状态持久化: 统一保存 LoRA 适配器、优化器动量、随机数发生器与实验运行清单
"""

import argparse
import torch
from trainer.evaluation import evaluate_benchmark


def parse_args():
    parser = argparse.ArgumentParser(description="nano-GRPO: Xiaomi MiMo Macro-Batch RLVR Engine")

    parser.add_argument("--task", type=str, default="counterexample",
                        choices=["counterexample", "bug_hunter", "bug_differential", "code_exec"],
                        help="选择训练任务: counterexample (差分程序反例挖掘), bug_hunter, bug_differential (输入差分执行), code_exec")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B",
                        help="基座模型名称或本地权重路径 (默认 Qwen/Qwen2.5-1.5B)")
    parser.add_argument("--data_path", type=str, default="",
                        help="自定义题目数据集 JSON 路径 (默认自动加载对应轨道数据集)")
    parser.add_argument("--anchor_size", type=int, default=60,
                        help="训练集固定锚点池题目规模 (默认 60 题)")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_counterexample",
                        help="检查点保存目录")
    parser.add_argument("--resume_checkpoint", type=str, default="",
                        help="恢复已有 LoRA 与优化器断点的目录路径 (精确续训同一实验)")
    parser.add_argument("--init_adapter", type=str, default="",
                        help="初始化已有预训练 LoRA 适配器权重目录 (跨任务迁移学习热启动，开启新实验)")

    # 算法超参数
    parser.add_argument("--global_steps", type=int, default=20,
                        help="MiMo 全局宏步总轮次 (默认 20 轮)")
    parser.add_argument("--group_size", type=int, default=4,
                        help="每道题采样的候选回答数 G (默认 4)")
    parser.add_argument("--micro_batch_size", type=int, default=2,
                        help="前向与反传微批次大小 (默认 2)")
    parser.add_argument("--problems_per_update", type=int, default=12,
                        help="每次更新覆盖的题目数 (默认 12，遍历 60 题更新 5 次)")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="AdamW 学习率 (默认 2e-5)")
    parser.add_argument("--kl_coef", type=float, default=0.04,
                        help="基座参考模型 KL 散度惩罚系数 (默认 0.04)")
    parser.add_argument("--clip_eps", type=float, default=0.2,
                        help="PPO 策略裁剪范围 (默认 0.2)")
    parser.add_argument("--max_new_tokens", type=int, default=32,
                        help="最大生成 Token 长度 (counterexample 极速生成默认 32)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="探索采样温度 (默认 0.7)")

    parser.add_argument("--top_p", type=float, default=1.0,
                        help="完整支持集采样，训练要求 top_p=1.0")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="运行设备")
    parser.add_argument("--skip_baseline", action="store_true",
                        help="是否跳过初始 Step 0 的零样本基线评测 (用于快速测试)")

    parser.add_argument("--dynamic_resampling", action="store_true",
                        help="启用 DAPO 风格动态零方差重采样 (Dynamic Variance Rescue)")
    parser.add_argument("--max_resample_rounds", type=int, default=2,
                        help="零方差组最大重采样轮数 (默认 2)")
    parser.add_argument("--resample_temp_boost", type=float, default=0.0, choices=[0.0],
                        help="兼容参数，只允许 0.0；重采样强制使用原策略温度")

    parser.add_argument("--seed", type=int, default=42, help="训练随机种子")
    return parser.parse_args()



def main():
    from trainer.run import run_training
    run_training(parse_args(), macro=True)


if __name__ == "__main__":
    main()
