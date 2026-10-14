import unittest
from tasks.counterexample import (
    CounterexampleTask,
    HeuristicSolver,
    extract_input_integer,
    evaluate_functions_on_input,
    generate_benchmark_suite
)


class TestCounterexampleTask(unittest.TestCase):
    def setUp(self):
        self.task = CounterexampleTask(seed=42)

    def test_dataset_splits_and_properties(self):
        self.assertEqual(len(self.task.train_problems), 60)
        self.assertEqual(len(self.task.val_problems), 20)
        self.assertEqual(len(self.task.test_problems), 20)

        for split_name, dataset in [("train", self.task.train_problems), ("val", self.task.val_problems), ("test", self.task.test_problems)]:
            for prob in dataset:
                self.assertIn("counterexamples", prob)
                self.assertGreater(len(prob["counterexamples"]), 0)
                low, high = prob["domain"]
                for ce in prob["counterexamples"]:
                    self.assertGreaterEqual(ce, low)
                    self.assertLessEqual(ce, high)
                    differs, _, _, err = evaluate_functions_on_input(prob["func_a_src"], prob["func_b_src"], ce)
                    self.assertIsNone(err)
                    self.assertTrue(differs, f"Failed on problem {prob['id']} with ce {ce}")

    def test_extraction_and_categorization(self):
        meta = {"domain": [0, 50], "counterexamples": [17], "func_a_src": "", "func_b_src": ""}

        # 1. 精准命中
        res = self.task.evaluate_single_completion("<input>17</input>", meta)
        self.assertEqual(res["reward"], 1.0)
        self.assertEqual(res["category"], "success")
        self.assertEqual(res["extracted_value"], 17)

        # 2. 未闭合标签或简写格式容错抽取
        res = self.task.evaluate_single_completion("Here is 17</input>", meta)
        self.assertEqual(res["reward"], 1.0)
        self.assertEqual(res["category"], "success")

        # 3. 错误候选 (在定义域内，但不是反例)
        res = self.task.evaluate_single_completion("<input>18</input>", meta)
        self.assertEqual(res["reward"], 0.0)
        self.assertEqual(res["category"], "missed_bug")
        self.assertEqual(res["extracted_value"], 18)

        # 4. 超出定义域
        res = self.task.evaluate_single_completion("<input>99</input>", meta)
        self.assertEqual(res["reward"], 0.0)
        self.assertEqual(res["category"], "precondition_violation")

        # 5. 无有效整数
        res = self.task.evaluate_single_completion("Nothing here", meta)
        self.assertEqual(res["reward"], 0.0)
        self.assertEqual(res["category"], "format_malformed")

        # 6. 真实截断标记
        res = self.task.evaluate_single_completion("Output: <input", dict(meta, is_truncated=True))
        self.assertEqual(res["reward"], 0.0)
        self.assertEqual(res["category"], "truncated")

    def test_heuristic_solvers(self):
        sample = self.task.train_problems[0]
        rand_res = HeuristicSolver.random_search(sample, budget=5)
        self.assertEqual(len(rand_res), 5)
        for x in rand_res:
            self.assertGreaterEqual(x, sample["domain"][0])
            self.assertLessEqual(x, sample["domain"][1])

        const_res = HeuristicSolver.constant_and_neighbors(sample, budget=5)
        self.assertLessEqual(len(const_res), 5)
        self.assertGreater(len(const_res), 0)


if __name__ == "__main__":
    unittest.main()
