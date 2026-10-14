"""
scripts/diagnose_checkpoint.py - 深度拆解模型在任务中的失败原因与有效组统计

统计维度:
  1. 细粒度失败分类 (format_or_truncated, precondition_violation, golden_failed, missed_bug, execution_error, success)
  2. 组方差与有效梯度组统计 (effective_problems, all_zero_problems, all_one_problems)
  3. 各失败类别的代表性采样实例与错误信息
"""

import argparse
import json
import os
import sys
from collections import defaultdict
import torch

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks import BugHunterTask
from trainer.utils import setup_lora_and_model
from trainer.grpo_loss import rollout_generation_config, generate_with_policy


def diagnose(args):
    device = args.device
    print(f"🔬 正在加载任务与检查点: {args.checkpoint} (Device: {device})...")

    task = BugHunterTask(data_path=args.data_path or None)
    if args.split == "val":
        dataset = task.get_val_dataset()
    elif args.split == "test":
        dataset = task.get_test_dataset()
    else:
        dataset = task.train_problems[:args.num_problems] if args.num_problems else task.train_problems

    print(f"📋 诊断数据集: {args.split} (共 {len(dataset)} 道题目), 每题采样 {args.samples_per_problem} 个候选 (温度: {args.temperature})")

    tokenizer, model = setup_lora_and_model(
        args.model_name,
        device=device,
        resume_checkpoint=args.checkpoint if args.checkpoint else None
    )
    model.eval()
    model.config.use_cache = True

    gen_config = rollout_generation_config(
        tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )

    category_counts = defaultdict(int)
    reason_counts = defaultdict(int)
    samples_by_category = defaultdict(list)

    group_stats = {
        "effective_problems": 0,    # 0 < solved < G (方差 > 0，有有效策略梯度)
        "all_zero_problems": 0,     # solved == 0 (全错，策略优势全部归 0)
        "all_one_problems": 0,      # solved == G (全对，策略优势全部归 0)
    }

    problem_results = []

    for idx, prob in enumerate(dataset, 1):
        prompt_str = prob["prompt"]
        prompt_inputs = tokenizer(prompt_str, return_tensors="pt").to(device)
        prompt_len = prompt_inputs["input_ids"].shape[1]

        # 批量重复 prompt 进行采样
        repeated_ids = prompt_inputs["input_ids"].repeat(args.samples_per_problem, 1)
        repeated_mask = prompt_inputs["attention_mask"].repeat(args.samples_per_problem, 1)

        with torch.no_grad():
            outputs = generate_with_policy(
                model,
                input_ids=repeated_ids,
                attention_mask=repeated_mask,
                generation_config=gen_config
            )

        completions = []
        for i in range(args.samples_per_problem):
            comp_text = tokenizer.decode(outputs[i][prompt_len:], skip_special_tokens=True)
            completions.append(comp_text)

        rewards, details = task.compute_rewards_with_details(
            prompts=[prompt_str] * args.samples_per_problem,
            completions=completions,
            metadata=[prob] * args.samples_per_problem
        )

        solved_in_group = sum(1 for r in rewards if r >= 1.0)
        if 0 < solved_in_group < args.samples_per_problem:
            group_stats["effective_problems"] += 1
        elif solved_in_group == 0:
            group_stats["all_zero_problems"] += 1
        else:
            group_stats["all_one_problems"] += 1

        for c_text, det in zip(completions, details):
            cat = det["category"]
            category_counts[cat] += 1
            reason_counts[det["reason"]] += 1
            if len(samples_by_category[cat]) < 3:
                samples_by_category[cat].append({
                    "problem_id": prob.get("problem_id", prob.get("id")),
                    "spec": prob["spec"],
                    "assertion": det.get("assertion"),
                    "reason": det.get("reason"),
                    "completion_snippet": c_text[:300] + ("..." if len(c_text) > 300 else "")
                })

        problem_results.append({
            "problem_id": prob.get("problem_id", prob.get("id")),
            "rewards": rewards,
            "solved": solved_in_group,
            "categories": [d["category"] for d in details]
        })

        if idx % 5 == 0 or idx == len(dataset):
            print(f"  进度: [{idx}/{len(dataset)}] 题诊断完成", flush=True)

    total_candidates = len(dataset) * args.samples_per_problem
    total_solved = category_counts["success"]

    summary = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "num_problems": len(dataset),
        "samples_per_problem": args.samples_per_problem,
        "total_candidates": total_candidates,
        "total_solved": total_solved,
        "overall_pass_rate": total_solved / max(1, total_candidates),
        "category_breakdown": {
            cat: {
                "count": count,
                "percentage": f"{count / total_candidates * 100:.1f}%"
            }
            for cat, count in category_counts.items()
        },
        "group_variance_stats": {
            "effective_problems (0 < solved < G)": {
                "count": group_stats["effective_problems"],
                "ratio": f"{group_stats['effective_problems'] / len(dataset) * 100:.1f}%"
            },
            "all_zero_problems (solved == 0)": {
                "count": group_stats["all_zero_problems"],
                "ratio": f"{group_stats['all_zero_problems'] / len(dataset) * 100:.1f}%"
            },
            "all_one_problems (solved == G)": {
                "count": group_stats["all_one_problems"],
                "ratio": f"{group_stats['all_one_problems'] / len(dataset) * 100:.1f}%"
            }
        },
        "samples_by_category": dict(samples_by_category),
        "problem_results": problem_results
    }

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n💾 详细诊断报告已保存至: {args.output}")

    print("\n" + "=" * 60)
    print("📊 诊断结果摘要:")
    print(f"  总候选生成数: {total_candidates} (通过: {total_solved}, 通过率: {total_solved/total_candidates*100:.1f}%)")
    print("\n[细粒度分类统计]:")
    for cat in ["success", "format_or_truncated", "precondition_violation", "golden_failed", "missed_bug", "execution_error"]:
        cnt = category_counts.get(cat, 0)
        pct = cnt / total_candidates * 100
        print(f"  - {cat:24s}: {cnt:4d} / {total_candidates:4d} ({pct:5.1f}%)")

    print("\n[组内方差与有效信号统计]:")
    print(f"  - 有效策略梯度组 (0 < solved < G): {group_stats['effective_problems']:2d} / {len(dataset):2d} ({group_stats['effective_problems']/len(dataset)*100:5.1f}%)")
    print(f"  - 零方差全错组 (solved == 0)      : {group_stats['all_zero_problems']:2d} / {len(dataset):2d} ({group_stats['all_zero_problems']/len(dataset)*100:5.1f}%)")
    print(f"  - 零方差全对组 (solved == G)      : {group_stats['all_one_problems']:2d} / {len(dataset):2d} ({group_stats['all_one_problems']/len(dataset)*100:5.1f}%)")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nano-GRPO 检查点细粒度失败归因与有效组诊断器")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--checkpoint", type=str, required=True, help="待诊断的检查点路径")
    parser.add_argument("--data_path", type=str, default="tasks/data/bug_hunter_dataset.json")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--num_problems", type=int, default=None)
    parser.add_argument("--samples_per_problem", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=350)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    diagnose(args)
