"""
scripts/evaluate_counterexample_baselines.py - Counterexample 主线启发式基准与对比评测

评测对象:
  1. 纯随机采样基线 (Random Search, budget=1 与 budget=4)
  2. AST 常量与边界邻域探测基线 (Constant & Neighbors Heuristic, budget=1 与 budget=4)
  3. 基座模型与 GRPO 训练权重结果对齐
"""

import os
import sys
import json
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks.counterexample import CounterexampleTask, HeuristicSolver


def evaluate_baselines(output_path="reports/counterexample_baselines_report.json"):
    task = CounterexampleTask(seed=42)
    test_probs = task.get_test_dataset()
    val_probs = task.get_val_dataset()

    report = {
        "task": "CounterexampleTask",
        "splits": {
            "test_count": len(test_probs),
            "val_count": len(val_probs),
        },
        "baselines": {}
    }

    for split_name, probs in [("val", val_probs), ("test", test_probs)]:
        report["baselines"][split_name] = {}
        for budget in [1, 4]:
            # 1. Random Search (固定随机种子以保证可复现)
            rand_solved = 0
            rand_hits = 0
            for p in probs:
                rng = random.Random(2026 + budget)
                guesses = HeuristicSolver.random_search(p, budget=budget, rng=rng)
                hits = sum(1 for g in guesses if g in p["counterexamples"])
                rand_hits += hits
                if hits > 0:
                    rand_solved += 1

            total_attempts = len(probs) * budget
            report["baselines"][split_name][f"random_budget_{budget}"] = {
                "solved_problems": rand_solved,
                "total_problems": len(probs),
                "problem_pass_rate": rand_solved / len(probs),
                "total_hits": rand_hits,
                "total_attempts": total_attempts,
                "sample_hit_rate": rand_hits / total_attempts,
            }

            # 2. Constant & Neighbors Heuristic
            const_solved = 0
            const_hits = 0
            for p in probs:
                guesses = HeuristicSolver.constant_and_neighbors(p, budget=budget)
                hits = sum(1 for g in guesses if g in p["counterexamples"])
                const_hits += hits
                if hits > 0:
                    const_solved += 1

            report["baselines"][split_name][f"constant_heuristic_budget_{budget}"] = {
                "solved_problems": const_solved,
                "total_problems": len(probs),
                "problem_pass_rate": const_solved / len(probs),
                "total_hits": const_hits,
                "total_attempts": total_attempts,
                "sample_hit_rate": const_hits / total_attempts,
            }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"✅ Baseline report generated at: {output_path}")
    return report


if __name__ == "__main__":
    evaluate_baselines()
