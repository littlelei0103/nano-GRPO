"""
demo.py - Nano Counterexample 旗舰前后对比交互式演示脚本

使用方法:
  # 运行经典盲盒测试集抽题对比
  python demo.py

  # 抽选指定数量的题目对比
  python demo.py --num_cases 5

  # 交互式查看指定题目 ID 或索引
  python demo.py --prob_idx 0
"""

import argparse
import glob
import os
import random
import torch

from tasks.counterexample import CounterexampleTask, extract_input_value, evaluate_functions_on_input
from trainer.grpo_loss import rollout_generation_config, generate_with_policy
from trainer.utils import setup_lora_and_model


def parse_args():
    parser = argparse.ArgumentParser(description="nano-GRPO Side-by-Side Comparison Demo")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B",
                        help="基座模型名称或本地权重路径")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="训练最佳 LoRA checkpoint 路径 (默认自动查找最新实验的 checkpoint-best)")
    parser.add_argument("--num_cases", type=int, default=3,
                        help="展示对比的测试集题目数量 (默认 3)")
    parser.add_argument("--prob_idx", type=int, default=-1,
                        help="指定评测的题目索引 (0~19)，若指定则仅对比该题")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="运行设备")
    return parser.parse_args()


# ANSI 颜色格式化
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def main():
    args = parse_args()

    # 1. 自动定位 checkpoint-best
    ckpt_path = args.checkpoint
    if not ckpt_path:
        candidates = sorted(glob.glob("checkpoints_counterexample/20*/checkpoint-best"))
        if not candidates:
            raise FileNotFoundError("未找到任何 checkpoint-best！请先运行训练或指定 --checkpoint")
        ckpt_path = candidates[-1]

    print("=" * 72)
    print(f"{BOLD}nano-GRPO: Program Differential Testing & Counterexample Mining Demo{RESET}")
    print(f"Base Model: {CYAN}{args.model_name}{RESET} | Device: {args.device}")
    print(f"RLVR Checkpoint: {GREEN}{ckpt_path}{RESET}")
    print("=" * 72)

    # 2. 加载模型与 LoRA
    print(f"[Model Init] Loading base model and LoRA adapter weights...")
    tokenizer, model = setup_lora_and_model(args.model_name, device=args.device, resume_checkpoint=ckpt_path)
    model.eval()
    print("[Model Init] Base policy and RL adapter ready.\n")

    # 3. 加载独立盲盒测试集
    task = CounterexampleTask()
    test_problems = task.get_test_dataset()
    if not test_problems:
        test_problems = task.train_problems

    if args.prob_idx >= 0 and args.prob_idx < len(test_problems):
        selected_problems = [test_problems[args.prob_idx]]
    else:
        # 挑选经典代表性案例
        selected_problems = test_problems[:args.num_cases]

    cfg = rollout_generation_config(tokenizer, max_new_tokens=24, temperature=0.7, do_sample=False)

    for case_no, prob in enumerate(selected_problems, 1):
        f_a = prob["func_a_src"]
        f_b = prob["func_b_src"]
        ces = prob["counterexamples"]
        input_dim = prob.get("input_dim", 1)
        if input_dim == 2 and "domain_x" in prob and "domain_y" in prob:
            domain = [tuple(prob["domain_x"]), tuple(prob["domain_y"])]
        else:
            domain = tuple(prob["domain"])
        prompt_inputs = tokenizer(prob["prompt"], return_tensors="pt").to(args.device)
        prompt_len = prompt_inputs["input_ids"].shape[1]

        # --- A. 基座模型采样 (关闭 Adapter) ---
        with model.disable_adapter():
            with torch.no_grad():
                out_base = generate_with_policy(model, cfg, **prompt_inputs)
        raw_base = tokenizer.decode(out_base[0][prompt_len:], skip_special_tokens=True).strip()
        val_base, cat_base, _ = extract_input_value("<input>" + raw_base, domain, input_dim=input_dim)
        diff_base, res_a_base, res_b_base, _ = evaluate_functions_on_input(f_a, f_b, val_base) if val_base is not None else (False, None, None, None)

        # --- B. RL 强化模型采样 (启用 Adapter) ---
        with torch.no_grad():
            out_rl = generate_with_policy(model, cfg, **prompt_inputs)
        raw_rl = tokenizer.decode(out_rl[0][prompt_len:], skip_special_tokens=True).strip()
        val_rl, cat_rl, _ = extract_input_value("<input>" + raw_rl, domain, input_dim=input_dim)
        diff_rl, res_a_rl, res_b_rl, _ = evaluate_functions_on_input(f_a, f_b, val_rl) if val_rl is not None else (False, None, None, None)

        # --- C. 打印双栏对抗卡片 ---
        print(f"\n{BOLD}{'─'*72}{RESET}")
        print(f"{BOLD}[Case {case_no}] Problem ID: {CYAN}{prob['id']}{RESET} | Tier: {YELLOW}{prob['level']}{RESET}")
        print(f"Specification: {prob['spec']}")
        print(f"Domain: {prob['domain']} | Counterexamples: {BOLD}{ces[:6]}{RESET} (Sparsity: {prob['sparsity']:.2%})")
        print(f"{'─'*72}")

        print(f"【Program Pair Specification】:")
        code_lines_a = f_a.splitlines()
        code_lines_b = f_b.splitlines()
        max_lines = max(len(code_lines_a), len(code_lines_b))
        for line_i in range(max_lines):
            l_a = code_lines_a[line_i] if line_i < len(code_lines_a) else ""
            l_b = code_lines_b[line_i] if line_i < len(code_lines_b) else ""
            print(f"  {l_a:<34s} │  {l_b}")
        print(f"{'─'*72}")

        # 结果判定
        status_base = f"{GREEN}Divergence Confirmed (+1.0){RESET}" if diff_base else f"{RED}Outputs Match (0.0){RESET}"
        status_rl = f"{GREEN}Divergence Confirmed (+1.0){RESET}" if diff_rl else f"{RED}Outputs Match (0.0){RESET}"

        arg_str_base = f"({val_base[0]}, {val_base[1]})" if isinstance(val_base, (list, tuple)) else f"({val_base})"
        arg_str_rl = f"({val_rl[0]}, {val_rl[1]})" if isinstance(val_rl, (list, tuple)) else f"({val_rl})"

        print(f"  {BOLD}Zero-Shot Base Policy{RESET}                 │  {BOLD}RLVR Optimized Policy{RESET}")
        print(f"  Raw: {repr(raw_base):<28s} │  Raw: {repr(raw_rl)}")
        print(f"  Parsed Input: {str(val_base):<20s} │  Parsed Input: {str(val_rl)}")
        print(f"  f_A{arg_str_base:<16s} = {str(res_a_base):<8s} │  f_A{arg_str_rl:<16s} = {str(res_a_rl)}")
        print(f"  f_B{arg_str_base:<16s} = {str(res_b_base):<8s} │  f_B{arg_str_rl:<16s} = {str(res_b_rl)}")
        print(f"  Verification: {status_base:<31s} │  Verification: {status_rl}")

    print(f"\n{'='*72}\n")


if __name__ == "__main__":
    main()
