"""
tests/test_counterexample_2d.py - Level 4 2D 多变量差分反例任务单元测试
"""

import random
import unittest
from tasks.counterexample import (
    extract_input_value,
    evaluate_functions_on_input,
    generate_2d_manhattan_problems,
    generate_2d_linear_problems,
    generate_2d_box_problems,
    generate_2d_piecewise_problems,
    generate_benchmark_suite,
    CounterexampleTask
)


class TestCounterexample2D(unittest.TestCase):
    """验证 Level 4 二维变量反例挖掘的输入抽取、函数执行与环境判题"""

    def test_2d_extraction_formats(self):
        domain_2d = [(0, 50), (0, 50)]

        # 1. 基础逗号分隔
        val, cat, _ = extract_input_value("<input>15, 30</input>", domain_2d, input_dim=2)
        self.assertEqual(val, (15, 30))
        self.assertEqual(cat, "valid_extracted")

        # 2. 命名变量匹配
        val, cat, _ = extract_input_value("<input>x = 7, y = 42</input>", domain_2d, input_dim=2)
        self.assertEqual(val, (7, 42))
        self.assertEqual(cat, "valid_extracted")

        # 3. 逆序命名变量匹配
        val, cat, _ = extract_input_value("<input>y = 20, x = 11</input>", domain_2d, input_dim=2)
        self.assertEqual(val, (11, 20))
        self.assertEqual(cat, "valid_extracted")

        # 4. 元组括号形式
        val, cat, _ = extract_input_value("<input>(12, 34)</input>", domain_2d, input_dim=2)
        self.assertEqual(val, (12, 34))
        self.assertEqual(cat, "valid_extracted")

        # 5. 超出定义域
        val, cat, _ = extract_input_value("<input>60, 20</input>", domain_2d, input_dim=2)
        self.assertEqual(cat, "out_of_domain")

        # 6. 维度不足 (仅输出一个整数)
        val, cat, _ = extract_input_value("<input>25</input>", domain_2d, input_dim=2)
        self.assertIsNone(val)
        self.assertEqual(cat, "format_malformed")

    def test_2d_generators_and_counterexample_validity(self):
        rng = random.Random(42)

        # Manhattan
        man_probs = generate_2d_manhattan_problems(rng, 3)
        self.assertEqual(len(man_probs), 3)
        for p in man_probs:
            self.assertEqual(p["level"], "level_4")
            self.assertEqual(p["input_dim"], 2)
            self.assertGreater(p["num_counterexamples"], 0)
            # 抽验反例在函数对上的差分表现
            ce = p["counterexamples"][0]
            diff, res_a, res_b, err = evaluate_functions_on_input(p["func_a_src"], p["func_b_src"], ce)
            self.assertIsNone(err)
            self.assertTrue(diff, f"Expected differing outputs for {ce}, got {res_a} == {res_b}")

        # Linear
        lin_probs = generate_2d_linear_problems(rng, 3)
        self.assertEqual(len(lin_probs), 3)
        for p in lin_probs:
            ce = p["counterexamples"][0]
            diff, res_a, res_b, err = evaluate_functions_on_input(p["func_a_src"], p["func_b_src"], ce)
            self.assertIsNone(err)
            self.assertTrue(diff)

        # Box
        box_probs = generate_2d_box_problems(rng, 3)
        self.assertEqual(len(box_probs), 3)
        for p in box_probs:
            ce = p["counterexamples"][0]
            diff, _, _, err = evaluate_functions_on_input(p["func_a_src"], p["func_b_src"], ce)
            self.assertIsNone(err)
            self.assertTrue(diff)

    def test_2d_task_evaluation_and_rewards(self):
        rng = random.Random(123)
        prob = generate_2d_manhattan_problems(rng, 1)[0]
        task = CounterexampleTask()

        ce = prob["counterexamples"][0]
        # 寻找一个非反例点
        non_ce = (0, 0) if (0, 0) not in [tuple(c) for c in prob["counterexamples"]] else (50, 50)

        completions = [
            f"<input>{ce[0]}, {ce[1]}</input>",       # 正确反例 -> 1.0
            f"<input>{non_ce[0]}, {non_ce[1]}</input>", # 常规非反例 -> 0.0 (missed_bug)
            "<input>99, 99</input>",                    # 越界输入 -> 0.0 (precondition_violation)
            "invalid output format"                     # 格式错误 -> 0.0 (format_malformed)
        ]

        rewards, details = task.compute_rewards_with_details(
            [""] * 4, completions, [prob] * 4
        )

        self.assertEqual(rewards, [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(details[0]["category"], "success")
        self.assertEqual(details[1]["category"], "missed_bug")
        self.assertEqual(details[2]["category"], "precondition_violation")
        self.assertEqual(details[3]["category"], "format_malformed")

    def test_mixed_benchmark_suite(self):
        suite = generate_benchmark_suite(num_train=40, num_val=10, num_test=10, include_2d=True, seed=99)
        self.assertEqual(len(suite["train"]), 40)
        self.assertEqual(len(suite["val"]), 10)
        self.assertEqual(len(suite["test"]), 10)

        train_dims = [p.get("input_dim", 1) for p in suite["train"]]
        self.assertIn(1, train_dims)
        self.assertIn(2, train_dims)


if __name__ == "__main__":
    unittest.main()
