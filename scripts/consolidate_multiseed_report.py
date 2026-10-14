"""
scripts/consolidate_multiseed_report.py - 汇总 Counterexample 主线多随机种子实验报告
"""

import os
import sys
import json
import statistics

RUN_DIRS = [
    ("Seed 42", "checkpoints_counterexample/20261014T102327Z-d1694cb8"),
    ("Seed 43", "checkpoints_counterexample/20261014T132832Z-225a3239"),
    ("Seed 44", "checkpoints_counterexample/20261014T135458Z-022190f1"),
]


def consolidate():
    runs_data = []
    for label, r_dir in RUN_DIRS:
        report_path = os.path.join(r_dir, "final_test_report.json")
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        runs_data.append({
            "label": label,
            "dir": r_dir,
            "best_step": data["validation_selection"]["best_step"],
            "val_acc": data["validation_selection"]["best_val_acc"],
            "base_greedy": data["blind_test_evaluation"]["base"]["problem_pass_rate"],
            "test_greedy": data["blind_test_evaluation"]["checkpoint_best"]["greedy"]["problem_pass_rate"],
            "test_sample_p4": data["blind_test_evaluation"]["checkpoint_best"]["sample"]["problem_pass_rate"],
            "test_sample_hit": data["blind_test_evaluation"]["checkpoint_best"]["sample"]["sample_pass_rate"],
            "last_greedy": data["blind_test_evaluation"]["checkpoint_last"]["greedy"]["problem_pass_rate"],
            "last_sample_p4": data["blind_test_evaluation"]["checkpoint_last"]["sample"]["problem_pass_rate"],
        })

    def stats(key):
        vals = [r[key] for r in runs_data]
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return {"mean": round(m, 4), "std": round(s, 4), "values": vals}

    summary = {
        "benchmark": "CounterexampleTask (Blind Test 20 problems, Val 20 problems)",
        "num_seeds": len(runs_data),
        "seeds": [42, 43, 44],
        "aggregated_metrics": {
            "val_acc": stats("val_acc"),
            "base_greedy": stats("base_greedy"),
            "checkpoint_best_greedy_pass1": stats("test_greedy"),
            "checkpoint_best_sample_pass4": stats("test_sample_p4"),
            "checkpoint_best_candidate_hit_rate": stats("test_sample_hit"),
            "checkpoint_last_greedy_pass1": stats("last_greedy"),
            "checkpoint_last_sample_pass4": stats("last_sample_p4"),
        },
        "individual_runs": runs_data
    }

    out_file = "reports/multiseed_counterexample_report.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"✅ Multi-seed consolidated report generated at: {out_file}")
    return summary


if __name__ == "__main__":
    consolidate()
