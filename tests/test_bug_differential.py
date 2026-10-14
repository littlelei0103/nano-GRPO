"""
tests/test_bug_differential.py - 差分缺陷审计任务 (DifferentialBugHunterTask) 单元测试
"""

import unittest
from tasks.bug_differential import (
    DifferentialBugHunterTask,
    extract_differential_input,
    parse_argument_literals,
    evaluate_differential_exposure
)


class TestBugDifferentialTask(unittest.TestCase):
    """测试方向 C: 基于输入字面量与差分沙盒执行的缺陷捕获任务"""

    def setUp(self):
        self.golden_bs = (
            "def target_func(arr: list[int], target: int) -> int:\n"
            "    left, right = 0, len(arr) - 1\n"
            "    while left <= right:\n"
            "        mid = (left + right) // 2\n"
            "        if arr[mid] == target: return mid\n"
            "        elif arr[mid] < target: left = mid + 1\n"
            "        else: right = mid - 1\n"
            "    return -1"
        )
        self.buggy_bs = (
            "def target_func(arr: list[int], target: int) -> int:\n"
            "    left, right = 0, len(arr) - 1\n"
            "    while left < right:\n"
            "        mid = (left + right) // 2\n"
            "        if arr[mid] == target: return mid\n"
            "        elif arr[mid] < target: left = mid + 1\n"
            "        else: right = mid - 1\n"
            "    return -1"
        )

    def test_parse_argument_literals_valid(self):
        ok, val, err = parse_argument_literals("[1, 3, 5], 5")
        self.assertTrue(ok)
        self.assertEqual(val, ([1, 3, 5], 5))

        ok, val, err = parse_argument_literals("[]")
        self.assertTrue(ok)
        self.assertEqual(val, ([],))

        ok, val, err = parse_argument_literals("\"hello\", -10")
        self.assertTrue(ok)
        self.assertEqual(val, ("hello", -10))

    def test_parse_argument_literals_security_rejection(self):
        # 拦截调用
        ok, val, err = parse_argument_literals("target_func(1)")
        self.assertFalse(ok)
        self.assertIn("Unsupported AST node", err)

        # 拦截注入
        ok, val, err = parse_argument_literals("__import__('os').system('ls')")
        self.assertFalse(ok)

    def test_extract_differential_input(self):
        comp = "<think>Analyzing binary search...</think>\n<input>[1, 3, 5], 5</input>"
        extracted = extract_differential_input(comp)
        self.assertEqual(extracted, "[1, 3, 5], 5")

        # 兼容 direct 模式: 前缀在 prompt 中，生成结果直接以参数开头并闭合 </input>
        comp_prefixed = "[1, 3, 5], 5</input>"
        self.assertEqual(extract_differential_input(comp_prefixed), "[1, 3, 5], 5")

        self.assertIsNone(extract_differential_input("No tag here"))

    def test_truncation_handling(self):
        task_reasoning = DifferentialBugHunterTask(prompt_mode="reasoning")
        task_direct = DifferentialBugHunterTask(prompt_mode="direct")

        # 截断时且无闭合 input 标签
        res = task_reasoning.evaluate_single_completion("<think>Thinking long...", {"golden_code": "", "buggy_code": "", "is_truncated": True})
        self.assertEqual(res["category"], "truncated")
        self.assertEqual(res["reward"], 0.0)

        # 未截断时缺少 input 标签应为 format_malformed
        res_malformed = task_reasoning.evaluate_single_completion("<think>Thinking done.</think>", {"golden_code": "", "buggy_code": "", "is_truncated": False})
        self.assertEqual(res_malformed["category"], "format_malformed")

        # 检查 direct 模式下 prompt 以 Assistant: <input> 结尾
        sample_direct = task_direct.val_problems[0]
        self.assertTrue(sample_direct["prompt"].strip().endswith("Assistant: <input>"))
        sample_reasoning = task_reasoning.val_problems[0]
        self.assertTrue(sample_reasoning["prompt"].strip().endswith("Assistant: <think>"))

    def test_differential_exposure_success(self):
        # 输入击穿缺陷: 右边界查找 (target=5 在末位)
        exposed, cat, reason, args = evaluate_differential_exposure(
            raw_input_str="[1, 3, 5], 5",
            golden_code=self.golden_bs,
            buggy_code=self.buggy_bs,
            input_contract="bs_standard_right_edge"
        )
        self.assertTrue(exposed)
        self.assertEqual(cat, "success")
        self.assertIn("Output diverged", reason)

    def test_differential_exposure_missed(self):
        # 正常输入未击穿缺陷: target=3 在中间，两个实现均返回 1
        exposed, cat, reason, args = evaluate_differential_exposure(
            raw_input_str="[1, 3, 5], 3",
            golden_code=self.golden_bs,
            buggy_code=self.buggy_bs,
            input_contract="bs_standard_right_edge"
        )
        self.assertFalse(exposed)
        self.assertEqual(cat, "missed_bug")
        self.assertIn("Outputs identical", reason)

    def test_differential_exposure_contract_violation(self):
        # 传入未排序数组，触发输入合约拦截
        exposed, cat, reason, args = evaluate_differential_exposure(
            raw_input_str="[5, 1, 3], 3",
            golden_code=self.golden_bs,
            buggy_code=self.buggy_bs,
            input_contract="bs_standard_right_edge"
        )
        self.assertFalse(exposed)
        self.assertEqual(cat, "precondition_violation")

    def test_dict_reordering_is_not_false_difference(self):
        # 字典键顺序不同但内容相同，绝不能误判为击穿缺陷
        golden_dict = "def target_func(): return {'a': 1, 'b': 2}"
        buggy_dict = "def target_func(): return {'b': 2, 'a': 1}"
        exposed, cat, reason, args = evaluate_differential_exposure(
            raw_input_str="",
            golden_code=golden_dict,
            buggy_code=buggy_dict
        )
        self.assertFalse(exposed)
        self.assertEqual(cat, "missed_bug")

    def test_type_difference_is_diverged(self):
        # 1 与 1.0 或 True 与 1 属于类型不一致的缺陷
        golden_type = "def target_func(): return 1"
        buggy_type = "def target_func(): return 1.0"
        exposed, cat, reason, args = evaluate_differential_exposure(
            raw_input_str="",
            golden_code=golden_type,
            buggy_code=buggy_type
        )
        self.assertTrue(exposed)
        self.assertEqual(cat, "success")
        self.assertIn("int", reason)
        self.assertIn("float", reason)

    def test_nested_types_and_float_tolerance_end_to_end(self):
        cases = [
            ("[1]", "[1.0]", True),
            ("{'v': [True]}", "{'v': [1]}", True),
            ("{1}", "{True}", True),
            ("{1: 'v'}", "{True: 'v'}", True),
            ("[0.3]", "[0.1 + 0.2]", False),
            ("{'a': (0.3,), 'b': {2, 3}}",
             "{'b': {3, 2}, 'a': (0.1 + 0.2,)}", False),
            ("[0.3]", "[0.4]", True),
            ("[1]", "(1,)", True),
        ]
        for golden, buggy, expected in cases:
            with self.subTest(golden=golden, buggy=buggy):
                exposed, category, _, _ = evaluate_differential_exposure(
                    "", f"def target_func(): return {golden}",
                    f"def target_func(): return {buggy}")
                self.assertEqual(exposed, expected)
                self.assertEqual(category, "success" if expected else "missed_bug")

    def test_special_float_transport_end_to_end(self):
        cases = [
            ("float('nan')", "float('nan')", False),
            ("[float('nan'), float('inf')]", "[float('nan'), float('inf')]", False),
            ("float('inf')", "float('-inf')", True),
            ("float('nan')", "0.0", True),
            ("{'v': b'bytes'}", "{'v': b'bytes'}", False),
        ]
        for golden, buggy, expected in cases:
            with self.subTest(golden=golden, buggy=buggy):
                result = evaluate_differential_exposure(
                    "", f"def target_func(): return {golden}",
                    f"def target_func(): return {buggy}")
                self.assertEqual(result[0], expected)
                self.assertEqual(result[1], "success" if expected else "missed_bug")

    def test_unsupported_output_is_not_rewarded(self):
        result = evaluate_differential_exposure(
            "", "def target_func(): return 1", "def target_func(): return object()")
        self.assertFalse(result[0])
        self.assertEqual(result[1], "execution_error")

    def test_function_prints_do_not_corrupt_result_transport(self):
        result = evaluate_differential_exposure(
            "", "def target_func():\n    print('debug A')\n    return {'a': 1}",
            "def target_func():\n    print('debug B')\n    return {'a': 1}")
        self.assertFalse(result[0])
        self.assertEqual(result[1], "missed_bug")

    def test_task_lifecycle(self):
        task = DifferentialBugHunterTask()
        self.assertGreater(len(task.train_problems), 0)
        self.assertGreater(len(task.val_problems), 0)
        self.assertGreater(len(task.test_problems), 0)

        batch = task.get_batch(2)
        self.assertEqual(len(batch), 2)
        self.assertIn("<input>", batch[0]["prompt"])

        # 验证单样本评测
        comp_success = "<think>edge case at the end</think>\n<input>[1, 3, 5], 5</input>"
        res = task.evaluate_single_completion(comp_success, batch[0])
        # 若第 0 题是 bs_standard_right_edge 则成功
        if batch[0]["id"] == "bs_standard_right_edge":
            self.assertEqual(res["reward"], 1.0)
            self.assertEqual(res["category"], "success")


if __name__ == "__main__":
    unittest.main()
