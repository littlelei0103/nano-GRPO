"""
scripts/check_learnability.py - Nano Counterexample 任务可学性体检脚本

核心目标:
  在正式开启强化学习训练前，对基座模型进行快速“可学性体检”:
  1. 格式合规率: 模型自回归输出 8~32 tokens 时，能否稳定输出定义域内的合法整数？
  2. 有效梯度组占比: 在 G=4 与 G=8 采样下，有多少题目出现 "0 < solved < G" (存在正负样本方差，能产生有效 GRPO 策略梯度)？
  3. 传统基线硬碰硬: 同等采样预算下，基座模型 vs 随机采样 (Random Search) vs 常量探测 (Constant Prober)。
  4. 难度分级表现: 各难度级别 (Level 1/2/3) 的可学性分布。
"""

import argparse
import json
import random
import time
from collections import defaultdict
from typing import Dict, List, Any

import torch
from tasks.counterexample import CounterexampleTask, HeuristicSolver, extract_input_integer
from trainer.grpo_loss import rollout_generation_config, generate_with_policy
from trainer.utils import setup_lora_and_model


def parse_args():
    parser = argparse.ArgumentParser(description="Nano Counterexample Learnability Healthcheck")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B",
                        help="基座模型名称或本地路径")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="运行设备")
    parser.add_argument("--num_problems", type=int, default=30,
                        help="体检抽取的题目数量 (默认 30 题，覆盖不同难度)")
    parser.add_argument("--group_size", type=int, default=4,
                        help="候选采样数 G (默认 4)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="采样温度 (默认 0.7)")
    parser.add_argument("--max_new_tokens", type=int, default=32,
                        help="最大生成 Token 长度 (默认 32，极短生成)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("🧪 Nano Counterexample 任务可学性体检启动")
    print(f"模型: {args.model_name} | 设备: {args.device} | 采样数 G: {args.group_size} | max_new_tokens: {args.max_new_tokens}")
    print("=" * 70)

    # 1. 准备任务与测试题目
    task = CounterexampleTask(seed=args.seed)
    all_train = task.train_problems
    # 选取均衡覆盖各难度层级的子集
    by_level = defaultdict(list)
    for p in all_train:
        by_level[p["level"]].append(p)

    selected_problems = []
    per_level = max(1, args.num_problems // len(by_level))
    for lvl in sorted(by_level.keys()):
        selected_problems.extend(by_level[lvl][:per_level])
    selected_problems = selected_problems[:args.num_problems]

    print(f"📊 体检测试集规模: {len(selected_problems)} 道题 | 涵盖等级: {list(by_level.keys())}\n")

    # 2. 传统启发式基线评测 (随机搜索 vs 常量探测)
    print("--- [基线 1: 纯随机采样 Random Search] ---")
    rng = random.Random(args.seed)
    random_solved_problems = 0
    random_total_hits = 0
    random_effective_groups = 0

    for prob in selected_problems:
        rand_inputs = HeuristicSolver.random_search(prob, budget=args.group_size, rng=rng)
        hits = sum(1 for x in rand_inputs if x in prob["counterexamples"])
        random_total_hits += hits
        if hits > 0:
            random_solved_problems += 1
        if 0 < hits < args.group_size:
            random_effective_groups += 1

    rand_prob_rate = random_solved_problems / len(selected_problems)
    rand_sample_rate = random_total_hits / (len(selected_problems) * args.group_size)
    print(f"  Random Search Pass@{args.group_size}: {rand_prob_rate:.1%} ({random_solved_problems}/{len(selected_problems)})")
    print(f"  Random Search 候选命中率: {rand_sample_rate:.1%} ({random_total_hits}/{len(selected_problems) * args.group_size})")
    print(f"  Random Search 有效组比例: {random_effective_groups / len(selected_problems):.1%} ({random_effective_groups}/{len(selected_problems)})\n")

    print("--- [基线 2: 常量边界探测 Constant Prober] ---")
    const_solved_problems = 0
    const_total_hits = 0
    const_effective_groups = 0

    for prob in selected_problems:
        const_inputs = HeuristicSolver.constant_and_neighbors(prob, budget=args.group_size)
        hits = sum(1 for x in const_inputs if x in prob["counterexamples"])
        const_total_hits += hits
        if hits > 0:
            const_solved_problems += 1
        if 0 < hits < args.group_size:
            const_effective_groups += 1

    const_prob_rate = const_solved_problems / len(selected_problems)
    const_sample_rate = const_total_hits / (len(selected_problems) * args.group_size)
    print(f"  Constant Prober Pass@{args.group_size}: {const_prob_rate:.1%} ({const_solved_problems}/{len(selected_problems)})")
    print(f"  Constant Prober 候选命中率: {const_sample_rate:.1%} ({const_total_hits}/{len(selected_problems) * args.group_size})")
    print(f"  Constant Prober 有效组比例: {const_effective_groups / len(selected_problems):.1%} ({const_effective_groups}/{len(selected_problems)})\n")

    # 3. 加载基座模型并评测
    print(f"⏳ 正在加载基座模型权重 {args.model_name}...")
    t0 = time.time()
    tokenizer, model = setup_lora_and_model(args.model_name, device=args.device)
    model.eval()
    print(f"   模型加载完成，耗时 {time.time() - t0:.1f} 秒\n")

    gen_config = rollout_generation_config(
        tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=1.0,
        do_sample=True
    )

    print(f"🚀 开始对模型执行采样体检 (共 {len(selected_problems)} 题 × {args.group_size} 候选)...")
    start_eval_time = time.time()

    model_solved_problems = 0
    model_total_hits = 0
    model_effective_groups = 0
    model_all_zero_groups = 0
    model_all_solved_groups = 0

    format_valid_count = 0
    total_candidates = len(selected_problems) * args.group_size
    level_stats = defaultdict(lambda: {"total": 0, "hits": 0, "solved_problems": 0, "problems": 0, "effective_groups": 0})

    sample_inspections = []

    for idx, prob in enumerate(selected_problems, 1):
        prompt_inputs = tokenizer(prob["prompt"], return_tensors="pt").to(args.device)
        prompt_len = prompt_inputs["input_ids"].shape[1]

        # 扩增 batch 进行并发前向采样
        micro_prompt_ids = prompt_inputs["input_ids"].repeat(args.group_size, 1)
        micro_attention_mask = prompt_inputs["attention_mask"].repeat(args.group_size, 1)

        with torch.no_grad():
            outputs = generate_with_policy(
                model,
                input_ids=micro_prompt_ids,
                attention_mask=micro_attention_mask,
                generation_config=gen_config,
            )

        prob_hits = 0
        completions = []
        parsed_values = []

        for g in range(args.group_size):
            comp_text = tokenizer.decode(outputs[g][prompt_len:], skip_special_tokens=True)
            completions.append(comp_text)

            val, cat, _ = extract_input_integer(comp_text, tuple(prob["domain"]))
            if val is not None and cat == "valid_extracted":
                format_valid_count += 1
                parsed_values.append(val)
                if val in prob["counterexamples"]:
                    prob_hits += 1
            else:
                parsed_values.append(None)

        model_total_hits += prob_hits
        if prob_hits > 0:
            model_solved_problems += 1

        if 0 < prob_hits < args.group_size:
            model_effective_groups += 1
        elif prob_hits == 0:
            model_all_zero_groups += 1
        else:
            model_all_solved_groups += 1

        # 统计分级数据
        lvl = prob["level"]
        level_stats[lvl]["problems"] += 1
        level_stats[lvl]["total"] += args.group_size
        level_stats[lvl]["hits"] += prob_hits
        if prob_hits > 0:
            level_stats[lvl]["solved_problems"] += 1
        if 0 < prob_hits < args.group_size:
            level_stats[lvl]["effective_groups"] += 1

        # 记录前 3 道题用于诊断报告
        if len(sample_inspections) < 3:
            sample_inspections.append({
                "id": prob["id"],
                "level": prob["level"],
                "spec": prob["spec"],
                "domain": prob["domain"],
                "counterexamples": prob["counterexamples"][:5],
                "completions": completions,
                "parsed_values": parsed_values,
                "hits": prob_hits
            })

    total_time = time.time() - start_eval_time

    # =========================================================================
    # 汇总输出诊断体检报告
    # =========================================================================
    print("\n" + "=" * 70)
    print("📋 [Nano Counterexample 任务可学性体检报告]")
    print("=" * 70)
    print(f"总计耗时: {total_time:.2f} 秒 (平均每题 {(total_time/len(selected_problems))*1000:.1f} 毫秒)")
    print(f"1. 格式提取合规率: {format_valid_count / total_candidates:.1%} ({format_valid_count}/{total_candidates})")
    print(f"2. 候选总体命中率: {model_total_hits / total_candidates:.1%} ({model_total_hits}/{total_candidates})")
    print(f"3. 题目胜率 Pass@{args.group_size}: {model_solved_problems / len(selected_problems):.1%} ({model_solved_problems}/{len(selected_problems)})")
    print(f"4. GRPO 策略梯度有效组: {model_effective_groups / len(selected_problems):.1%} ({model_effective_groups}/{len(selected_problems)}) [核心学习信号]")
    print(f"   ↳ 全错组 (0/{args.group_size}): {model_all_zero_groups}/{len(selected_problems)} ({model_all_zero_groups/len(selected_problems):.1%})")
    print(f"   ↳ 全对组 ({args.group_size}/{args.group_size}): {model_all_solved_groups}/{len(selected_problems)} ({model_all_solved_groups/len(selected_problems):.1%})")

    print("\n📈 [难度分级体检分布]")
    for lvl, s in sorted(level_stats.items()):
        pass_rate = s["solved_problems"] / s["problems"] if s["problems"] else 0
        eff_ratio = s["effective_groups"] / s["problems"] if s["problems"] else 0
        hit_rate = s["hits"] / s["total"] if s["total"] else 0
        print(f"  • {lvl:8s}: Pass@{args.group_size}={pass_rate:.1%} | 候选命中率={hit_rate:.1%} | 有效梯度组={eff_ratio:.1%} ({s['effective_groups']}/{s['problems']})")

    print("\n⚔️ [模型 vs 传统启发式硬碰硬对比]")
    print(f"  {'方法':24s} | {'Pass@' + str(args.group_size):12s} | {'候选命中率':12s} | {'有效梯度组':12s}")
    print(f"  {'-'*24} | {'-'*12} | {'-'*12} | {'-'*12}")
    print(f"  {'Random Search (随机)':22s} | {rand_prob_rate:11.1%} | {rand_sample_rate:11.1%} | {random_effective_groups/len(selected_problems):11.1%}")
    print(f"  {'Constant Prober (常量探测)':18s} | {const_prob_rate:11.1%} | {const_sample_rate:11.1%} | {const_effective_groups/len(selected_problems):11.1%}")
    print(f"  {'Qwen2.5-1.5B (Base ZeroShot)':18s} | {model_solved_problems/len(selected_problems):11.1%} | {model_total_hits/total_candidates:11.1%} | {model_effective_groups/len(selected_problems):11.1%}")

    print("\n🔍 [精选生成样本快照]")
    for i, s in enumerate(sample_inspections, 1):
        print(f"\n[Case {i}] ID: {s['id']} ({s['level']}) - {s['spec']}")
        print(f"  定义域: {s['domain']} | 有效反例: {s['counterexamples']}")
        for g in range(args.group_size):
            hit_flag = "✅ 命中" if s['parsed_values'][g] in s['counterexamples'] else "❌ 未命中"
            print(f"    候选 {g+1}: Raw='{s['completions'][g].strip()}' -> 解析值={s['parsed_values'][g]} ({hit_flag})")
    print("=" * 70)


if __name__ == "__main__":
    main()
