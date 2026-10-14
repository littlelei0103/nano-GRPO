import os
import sys
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tasks.bug_hunter import BugHunterTask
from trainer.grpo_loss import rollout_generation_config, generate_with_policy


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-metric evaluation: Greedy Pass@1 vs Sampling Avg@8 vs Pass@8")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--checkpoints", nargs="+", default=["base"], help="Explicit model adapters to compare")
    parser.add_argument("--k_samples", type=int, default=8, help="Number of rollouts per problem for sampling benchmark")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=350)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_file", type=str, default="logs/multimetric_benchmark_report.json")
    return parser.parse_args()


def evaluate_checkpoint(model, tokenizer, test_items, task, args):
    model.eval()
    model.config.use_cache = True

    # 1. 纯贪心 Pass@1 (T=0.0)
    greedy_correct = 0
    total = len(test_items)

    print("   [1/2] 正在评估确定性纯贪心 Pass@1 (T=0.0)...", flush=True)
    for item in test_items:
        prompt = item["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
        plen = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            out = generate_with_policy(model, rollout_generation_config(tokenizer, args.max_new_tokens, args.temperature, do_sample=False), **inputs)
        comp = tokenizer.decode(out[0][plen:], skip_special_tokens=True)
        r = task.compute_rewards([prompt], [comp], [item])[0]
        if r >= 1.0:
            greedy_correct += 1

    greedy_pass1 = greedy_correct / total

    # 2. 小米规范采样 Avg@8 和 Pass@8 (T=0.7, top_p=0.95, K=8)
    print(f"   [2/2] 正在评估小米规范采样 Avg@{args.k_samples} 与 Pass@{args.k_samples} (T={args.temperature})...", flush=True)
    total_sampled_hits = 0
    total_sampled_trajectories = total * args.k_samples
    pass_at_k_solved = 0

    for idx, item in enumerate(test_items):
        prompt = item["prompt"]
        inputs = tokenizer([prompt] * args.k_samples, return_tensors="pt", padding=True).to(args.device)
        plen = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            out = generate_with_policy(model, rollout_generation_config(tokenizer, args.max_new_tokens, args.temperature, top_p=args.top_p), **inputs)
        comps = [tokenizer.decode(seq[plen:], skip_special_tokens=True) for seq in out]
        rewards = task.compute_rewards([prompt] * args.k_samples, comps, [item] * args.k_samples)

        hits = sum(1 for r in rewards if r >= 1.0)
        total_sampled_hits += hits
        if hits > 0:
            pass_at_k_solved += 1

    avg_at_k = total_sampled_hits / total_sampled_trajectories
    pass_at_k = pass_at_k_solved / total

    return {
        "greedy_pass1": greedy_pass1,
        "greedy_correct": greedy_correct,
        "avg_at_k": avg_at_k,
        "total_hits": total_sampled_hits,
        "total_rollouts": total_sampled_trajectories,
        "pass_at_k": pass_at_k,
        "pass_k_solved": pass_at_k_solved,
        "total_problems": total
    }


def main():
    args = parse_args()
    print("=" * 85)
    print("🔬 nano-GRPO 独立盲盒测试集 (50 题) 多维度全息大考 (Greedy vs Avg@8 vs Pass@8)")
    print(f"   基座模型:      {args.base_model}")
    print(f"   待评测检查点:  {args.checkpoints}")
    print(f"   采样参数:      K={args.k_samples}, Temp={args.temperature}, Top-p={args.top_p}")
    print(f"   计算设备:      {args.device}")
    print("=" * 85, flush=True)

    task = BugHunterTask()
    test_items = task.get_test_dataset() if hasattr(task, 'get_test_dataset') and task.get_test_dataset() else task.get_val_dataset()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    results = []

    for ckpt in args.checkpoints:
        name = os.path.basename(ckpt) if ckpt != "base" else "Base Model (Step 0)"
        print(f"\n🔍 >>> 正在大考检查点: {name} ({ckpt}) ...", flush=True)

        # 加载基座模型
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map=args.device,
            trust_remote_code=True
        )

        if ckpt == "base":
            active_model = base_model
        else:
            if not os.path.exists(ckpt):
                print(f"⚠️ 检查点路径不存在: {ckpt}，跳过！")
                del base_model
                torch.cuda.empty_cache()
                continue
            active_model = PeftModel.from_pretrained(base_model, ckpt)

        metrics = evaluate_checkpoint(active_model, tokenizer, test_items, task, args)
        metrics["checkpoint"] = ckpt
        metrics["name"] = name
        results.append(metrics)

        print(f"   🎯 [得分公布] {name}:")
        print(f"      • 确定性纯贪心 Pass@1 (T=0.0): {metrics['greedy_pass1']*100:.1f}% ({metrics['greedy_correct']}/{metrics['total_problems']})")
        print(f"      • 小米规范采样 Avg@8 (T=0.7):   {metrics['avg_at_k']*100:.1f}% ({metrics['total_hits']}/{metrics['total_rollouts']})")
        print(f"      • 能力上限天花板 Pass@8:        {metrics['pass_at_k']*100:.1f}% ({metrics['pass_k_solved']}/{metrics['total_problems']})")

        del active_model
        if ckpt != "base":
            del base_model
        torch.cuda.empty_cache()

    # 打印最终对比大表
    print("\n" + "=" * 85)
    print("🏆 最终全维度全息竞技场总榜单 (Leaderboard)")
    print("=" * 85)
    print(f"{'Checkpoint Name':<32} | {'Greedy Pass@1':<14} | {'Sampling Avg@8':<15} | {'Ceiling Pass@8':<14}")
    print("-" * 85)
    for res in results:
        g = f"{res['greedy_pass1']*100:.1f}% ({res['greedy_correct']}/{res['total_problems']})"
        a = f"{res['avg_at_k']*100:.1f}%"
        p = f"{res['pass_at_k']*100:.1f}% ({res['pass_k_solved']}/{res['total_problems']})"
        print(f"{res['name']:<32} | {g:<14} | {a:<15} | {p:<14}")
    print("=" * 85)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n📁 详细全息评测报告已持久化保存至: {args.output_file}\n")


if __name__ == "__main__":
    main()
