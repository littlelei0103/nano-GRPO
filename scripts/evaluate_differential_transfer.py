"""
scripts/evaluate_differential_transfer.py - 差分缺陷审计轨道基线、截断对齐与跨任务迁移评测

核心改进:
  1. 准确标记截断 (is_truncated): 当生成达到 max_new_tokens 且未命中 EOS 时，向判题器显式传递 is_truncated=True；
  2. 严谨支持逐题明细持久化 (detailed_results): 记录每题的完整生成文本、提取结果、执行归因与截断状态；
  3. 固定可复现随机采样种子 (RNG seed): 对多样本采样按 (seed + p_idx * 1000 + s_idx) 进行确定性控制；
  4. 支持验证集/测试集隔离 (--split val/test): 默认在验证集 (Val) 上进行提示词调优与诊断；
  5. 支持直接模式与推理模式对照 (--prompt_mode direct/reasoning 或 --compare_prompts)。
"""

import os
import sys
import json
import time
import argparse
from collections import defaultdict
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tasks.bug_differential import DifferentialBugHunterTask
from trainer.grpo_loss import rollout_generation_config, generate_with_policy


def evaluate_model_on_task(
    model,
    tokenizer,
    problems,
    task,
    max_new_tokens=160,
    device="cuda",
    do_sample=False,
    temperature=0.7,
    samples_per_problem=1,
    base_seed=42,
    tag="Model"
):
    model.eval()
    model.config.use_cache = True
    gen_config = rollout_generation_config(tokenizer, max_new_tokens, temperature, do_sample=do_sample)

    solved_count = 0
    total_hits = 0
    category_counts = defaultdict(int)
    detailed_results = []

    print(f"\n[{tag}] Starting evaluation: {len(problems)} problems, samples={samples_per_problem}, do_sample={do_sample}, max_new_tokens={max_new_tokens}...", flush=True)
    t0 = time.time()

    for p_idx, prob in enumerate(problems, 1):
        prompt = prob["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]

        prob_hits = 0
        completions = []
        eval_details = []

        for s_idx in range(samples_per_problem):
            if do_sample:
                seed = base_seed + p_idx * 1000 + s_idx
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

            with torch.no_grad():
                out = generate_with_policy(model, gen_config, **inputs)

            gen_len = out.shape[1] - prompt_len
            is_truncated = (gen_len >= max_new_tokens) and (out[0, -1].item() != tokenizer.eos_token_id)
            gen_text = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)

            meta = dict(prob, is_truncated=is_truncated)
            res = task.evaluate_single_completion(gen_text, meta)

            completions.append(gen_text)
            eval_details.append({
                "sample_idx": s_idx,
                "gen_length": gen_len,
                "is_truncated": is_truncated,
                "reward": res["reward"],
                "category": res["category"],
                "reason": res["reason"],
                "input_args": res.get("input_args"),
                "format_compliant": res.get("format_compliant", False)
            })
            cat = res["category"]
            category_counts[cat] += 1
            if res["reward"] >= 1.0:
                prob_hits += 1

        total_hits += prob_hits
        is_solved = (prob_hits > 0)
        if is_solved:
            solved_count += 1

        detailed_results.append({
            "problem_id": prob.get("id", f"prob_{p_idx}"),
            "is_solved": is_solved,
            "hits": prob_hits,
            "total_samples": samples_per_problem,
            "completions": completions,
            "details": eval_details
        })

        if p_idx % 5 == 0 or p_idx == len(problems):
            print(f"  [{tag}] Progress: {p_idx:02d}/{len(problems)} | Current Solved: {solved_count}/{p_idx} ({solved_count/p_idx*100:.1f}%) | Elapsed: {time.time()-t0:.1f}s", flush=True)

    elapsed = time.time() - t0
    total_problems = len(problems)
    total_samples = total_problems * samples_per_problem

    return {
        "problem_pass_rate": solved_count / max(1, total_problems),
        "sample_pass_rate": total_hits / max(1, total_samples),
        "solved_problems": solved_count,
        "total_problems": total_problems,
        "total_samples": total_samples,
        "correct_samples": total_hits,
        "category_breakdown": dict(category_counts),
        "elapsed_sec": elapsed,
        "detailed_results": detailed_results
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Differential Bug Hunter Transfer & Prompts")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--adapter_path", type=str, default="checkpoints_counterexample/20261014T102327Z-d1694cb8/checkpoint-best")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"],
                        help="选择评测集 (推荐使用 val 探索提示词，严格保留 test 作为盲测)")
    parser.add_argument("--prompt_mode", type=str, default="direct", choices=["direct", "reasoning"],
                        help="提示词模式: direct (直接输出 <input>) 或 reasoning (含 <think> 思考链)")
    parser.add_argument("--compare_prompts", action="store_true",
                        help="同时比对 direct 模式与 reasoning 模式在所选集上的表现")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="最大生成 Token 数量")
    parser.add_argument("--k_samples", type=int, default=4,
                        help="Pass@k 采样轮数")
    parser.add_argument("--seed", type=int, default=42,
                        help="基础随机种子")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_file", type=str, default="",
                        help="自定义输出 JSON 路径")
    return parser.parse_args()


def run_benchmark_for_prompt_mode(args, base_model, model_transfer, tokenizer, prompt_mode):
    max_tokens = 64 if prompt_mode == "direct" else args.max_new_tokens
    task = DifferentialBugHunterTask(prompt_mode=prompt_mode)
    problems = task.get_val_dataset() if args.split == "val" else task.get_test_dataset()

    print(f"\n" + "=" * 70)
    print(f" 评测提示词模式: [{prompt_mode.upper()}] | 数据集: [{args.split.upper()} ({len(problems)} 题)] | max_new_tokens: {max_tokens}")
    print("=" * 70)

    # 1. Base Model 贪心与采样
    base_greedy = evaluate_model_on_task(
        base_model, tokenizer, problems, task,
        max_new_tokens=max_tokens, device=args.device, do_sample=False,
        samples_per_problem=1, base_seed=args.seed, tag=f"Base-{prompt_mode}-Greedy"
    )
    base_sample = evaluate_model_on_task(
        base_model, tokenizer, problems, task,
        max_new_tokens=max_tokens, device=args.device, do_sample=True,
        temperature=0.7, samples_per_problem=args.k_samples, base_seed=args.seed, tag=f"Base-{prompt_mode}-Sample@{args.k_samples}"
    )

    # 2. Transfer Model 贪心与采样
    trans_greedy = evaluate_model_on_task(
        model_transfer, tokenizer, problems, task,
        max_new_tokens=max_tokens, device=args.device, do_sample=False,
        samples_per_problem=1, base_seed=args.seed, tag=f"Transfer-{prompt_mode}-Greedy"
    )
    trans_sample = evaluate_model_on_task(
        model_transfer, tokenizer, problems, task,
        max_new_tokens=max_tokens, device=args.device, do_sample=True,
        temperature=0.7, samples_per_problem=args.k_samples, base_seed=args.seed, tag=f"Transfer-{prompt_mode}-Sample@{args.k_samples}"
    )

    return {
        "prompt_mode": prompt_mode,
        "max_new_tokens": max_tokens,
        "base_model": {
            "greedy": base_greedy,
            "sample": base_sample
        },
        "transfer_model": {
            "greedy": trans_greedy,
            "sample": trans_sample
        }
    }


def main():
    args = parse_args()
    device = args.device

    print("=" * 70)
    print(" nano-GRPO Differential Bug Hunter Evaluation & Audit")
    print(f" Model: {args.model_name} | Adapter: {args.adapter_path}")
    print(f" Split: {args.split} | Seed: {args.seed} | Device: {device}")
    print("=" * 70)

    # 加载 Tokenizer 与 Base Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    torch_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    ).to(device)

    # 加载 LoRA 适配器
    if not os.path.isdir(args.adapter_path):
        raise FileNotFoundError(f"Adapter not found at: {args.adapter_path}")
    model_transfer = PeftModel.from_pretrained(base_model, args.adapter_path).to(device)

    modes_to_test = ["direct", "reasoning"] if args.compare_prompts else [args.prompt_mode]
    results_by_mode = {}

    for mode in modes_to_test:
        results_by_mode[mode] = run_benchmark_for_prompt_mode(args, base_model, model_transfer, tokenizer, mode)

    # 输出汇总报告
    output_path = args.output_file or f"reports/differential_evaluation_{args.split}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    summary_data = {
        "benchmark": "DifferentialBugHunterTask",
        "split": args.split,
        "seed": args.seed,
        "k_samples": args.k_samples,
        "model_name": args.model_name,
        "adapter_path": args.adapter_path,
        "results": results_by_mode
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(f" 评测汇总对比 (Split: {args.split.upper()})")
    print("=" * 80)
    print(f"{'Prompt Mode':<12} | {'Model':<16} | {'Greedy Pass@1':<15} | {'Sample Pass@4':<15} | {'Truncated':<10} | {'Malformed':<10}")
    print("-" * 80)
    for mode, data in results_by_mode.items():
        for m_tag, m_data in [("Base", data["base_model"]), ("Transfer", data["transfer_model"])]:
            g = m_data["greedy"]
            s = m_data["sample"]
            trunc = s["category_breakdown"].get("truncated", 0)
            malf = s["category_breakdown"].get("format_malformed", 0)
            print(f"{mode:<12} | {m_tag:<16} | {g['problem_pass_rate']*100:5.1f}% ({g['solved_problems']:02d}/{g['total_problems']}) | {s['problem_pass_rate']*100:5.1f}% ({s['solved_problems']:02d}/{s['total_problems']}) | {trunc:<10} | {malf:<10}")
    print("=" * 80)
    print(f"✅ Detailed report with per-problem records saved to: {output_path}")


if __name__ == "__main__":
    main()
