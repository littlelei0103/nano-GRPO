"""
tasks/code_exec.py - 轨道二：Python 代码执行与单元测试客观验证 (RLVR Code Execution Track)

核心机制:
  1. 纯客观真理奖励 (Ground-Truth Execution Reward):
     提取模型生成的 Python 代码，置于安全隔离子进程中针对 3~5 组单元测试断言 (含边界用例) 实际执行。
  2. 毫秒级沙盒隔离 (Subprocess Sandbox with Timeout):
     严格限制执行时间 (1.0s)，强力阻断死循环 (while True)、语法错误与运行时崩溃。
  3. 全通过给 1.0，任一测试失败/报错/超时给 0.0。
"""

import os
import re
import sys
import subprocess
import concurrent.futures
from typing import List, Dict, Any, Optional

# 允许直接 python tasks/code_exec.py 独立自测
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tasks.base import BaseTask

SYSTEM_PROMPT = (
    "You are an expert Python programming assistant.\n"
    "Show your algorithm design, edge case analysis, and reasoning inside <think> and </think> tags.\n"
    "At the end, output your complete Python code implementation inside <code> and </code> tags.\n"
    "CRITICAL RULES:\n"
    "1. The entry function MUST be strictly named `solution`.\n"
    "2. Do NOT include any testing code or assertions inside the <code> block.\n"
    "3. Handle all standard boundary conditions (e.g., empty lists, zero, single element)."
)

# 精选 30 道经典算法与数据结构编程题 (覆盖数组、字符串、双指针、字典、边界用例)
CURATED_CODING_PROBLEMS = [
    {
        "id": "move_zeroes",
        "description": "Write a function `solution(nums: list[int]) -> list[int]` that moves all 0's to the end of the array while maintaining the relative order of the non-zero elements. Return a new or modified list.",
        "test_cases": [
            "assert solution([0, 1, 0, 3, 12]) == [1, 3, 12, 0, 0]",
            "assert solution([0]) == [0]",
            "assert solution([1, 2, 3]) == [1, 2, 3]",
            "assert solution([]) == []",
            "assert solution([0, 0, 1]) == [1, 0, 0]"
        ]
    },
    {
        "id": "longest_increasing_subarray",
        "description": "Write a function `solution(nums: list[int]) -> int` that returns the length of the longest contiguous subarray where all elements are strictly increasing. Return 0 for an empty list.",
        "test_cases": [
            "assert solution([1, 3, 5, 4, 7]) == 3",
            "assert solution([2, 2, 2, 2, 2]) == 1",
            "assert solution([]) == 0",
            "assert solution([5]) == 1",
            "assert solution([1, 2, 3, 4, 5]) == 5"
        ]
    },
    {
        "id": "is_palindrome_str",
        "description": "Write a function `solution(s: str) -> bool` that returns True if the string is a palindrome, considering only alphanumeric characters and ignoring cases. Return True for empty or single-character strings.",
        "test_cases": [
            "assert solution('A man, a plan, a canal: Panama') == True",
            "assert solution('race a car') == False",
            "assert solution(' ') == True",
            "assert solution('0P') == False",
            "assert solution('a.') == True"
        ]
    },
    {
        "id": "valid_parentheses",
        "description": "Write a function `solution(s: str) -> bool` that determines if the input string containing only '()[]{}' is valid (brackets must close in correct order and pairs).",
        "test_cases": [
            "assert solution('()') == True",
            "assert solution('()[]{}') == True",
            "assert solution('(]') == False",
            "assert solution('([)]') == False",
            "assert solution('{[]}') == True",
            "assert solution('') == True"
        ]
    },
    {
        "id": "two_sum_indices",
        "description": "Write a function `solution(nums: list[int], target: int) -> list[int]` that returns the 0-based indices of the two numbers such that they add up to target. Indices can be in any order. Assume exactly one valid solution exists.",
        "test_cases": [
            "assert sorted(solution([2, 7, 11, 15], 9)) == [0, 1]",
            "assert sorted(solution([3, 2, 4], 6)) == [1, 2]",
            "assert sorted(solution([3, 3], 6)) == [0, 1]"
        ]
    },
    {
        "id": "reverse_vowels",
        "description": "Write a function `solution(s: str) -> str` that reverses only all the vowels in the string (vowels are 'a', 'e', 'i', 'o', 'u', case-insensitive).",
        "test_cases": [
            "assert solution('hello') == 'holle'",
            "assert solution('leetcode') == 'leotcede'",
            "assert solution('aA') == 'Aa'",
            "assert solution('xyz') == 'xyz'",
            "assert solution('') == ''"
        ]
    },
    {
        "id": "first_unique_char",
        "description": "Write a function `solution(s: str) -> int` that finds the first non-repeating character in a string and returns its 0-based index. If it does not exist, return -1.",
        "test_cases": [
            "assert solution('leetcode') == 0",
            "assert solution('loveleetcode') == 2",
            "assert solution('aabb') == -1",
            "assert solution('') == -1",
            "assert solution('z') == 0"
        ]
    },
    {
        "id": "merge_sorted_lists",
        "description": "Write a function `solution(list1: list[int], list2: list[int]) -> list[int]` that merges two sorted lists into one sorted list.",
        "test_cases": [
            "assert solution([1, 2, 4], [1, 3, 4]) == [1, 1, 2, 3, 4, 4]",
            "assert solution([], []) == []",
            "assert solution([], [0]) == [0]",
            "assert solution([5], [1, 2, 3]) == [1, 2, 3, 5]"
        ]
    },
    {
        "id": "find_missing_number",
        "description": "Write a function `solution(nums: list[int]) -> int` that takes an array containing n distinct numbers in the range [0, n] and returns the only number in the range that is missing.",
        "test_cases": [
            "assert solution([3, 0, 1]) == 2",
            "assert solution([0, 1]) == 2",
            "assert solution([9, 6, 4, 2, 3, 5, 7, 0, 1]) == 8",
            "assert solution([0]) == 1"
        ]
    },
    {
        "id": "majority_element",
        "description": "Write a function `solution(nums: list[int]) -> int` that returns the majority element (the element that appears more than ⌊n / 2⌋ times). Assume that the majority element always exists.",
        "test_cases": [
            "assert solution([3, 2, 3]) == 3",
            "assert solution([2, 2, 1, 1, 1, 2, 2]) == 2",
            "assert solution([1]) == 1"
        ]
    },
    {
        "id": "max_subarray_sum",
        "description": "Write a function `solution(nums: list[int]) -> int` that finds the contiguous subarray with the largest sum and returns its sum. If empty, return 0.",
        "test_cases": [
            "assert solution([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6",
            "assert solution([1]) == 1",
            "assert solution([5, 4, -1, 7, 8]) == 23",
            "assert solution([-1, -2, -3]) == -1",
            "assert solution([]) == 0"
        ]
    },
    {
        "id": "count_primes",
        "description": "Write a function `solution(n: int) -> int` that returns the number of prime numbers strictly less than a non-negative number n.",
        "test_cases": [
            "assert solution(10) == 4",
            "assert solution(0) == 0",
            "assert solution(1) == 0",
            "assert solution(2) == 0",
            "assert solution(20) == 8"
        ]
    },
    {
        "id": "is_anagram",
        "description": "Write a function `solution(s: str, t: str) -> bool` that returns True if t is an anagram of s, and False otherwise.",
        "test_cases": [
            "assert solution('anagram', 'nagaram') == True",
            "assert solution('rat', 'car') == False",
            "assert solution('a', 'a') == True",
            "assert solution('ab', 'a') == False"
        ]
    },
    {
        "id": "string_compression",
        "description": "Write a function `solution(s: str) -> str` that performs basic string compression using the counts of repeated characters (e.g. 'aabcccccaaa' -> 'a2b1c5a3'). If the compressed string is not shorter than the original, return the original string.",
        "test_cases": [
            "assert solution('aabcccccaaa') == 'a2b1c5a3'",
            "assert solution('abcdef') == 'abcdef'",
            "assert solution('aabbcc') == 'aabbcc'",
            "assert solution('') == ''"
        ]
    },
    {
        "id": "find_peak_element",
        "description": "Write a function `solution(nums: list[int]) -> int` that returns the index of any peak element (an element strictly greater than its neighbors). Assume nums[-1] = nums[n] = -infinity.",
        "test_cases": [
            "assert solution([1, 2, 3, 1]) == 2",
            "assert solution([1, 2, 1, 3, 5, 6, 4]) in [1, 5]",
            "assert solution([1]) == 0",
            "assert solution([1, 2]) == 1"
        ]
    },
    {
        "id": "remove_duplicates_sorted",
        "description": "Write a function `solution(nums: list[int]) -> list[int]` that removes duplicate elements from a sorted list in-place/returning a new list such that each unique element appears only once in sorted order.",
        "test_cases": [
            "assert solution([1, 1, 2]) == [1, 2]",
            "assert solution([0, 0, 1, 1, 1, 2, 2, 3, 3, 4]) == [0, 1, 2, 3, 4]",
            "assert solution([]) == []",
            "assert solution([7]) == [7]"
        ]
    },
    {
        "id": "chunk_list",
        "description": "Write a function `solution(lst: list, size: int) -> list[list]` that splits a list into chunks of given size. The last chunk may contain fewer elements. If size <= 0 or list is empty, return [].",
        "test_cases": [
            "assert solution([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]",
            "assert solution([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]",
            "assert solution([], 3) == []",
            "assert solution([1, 2], 5) == [[1, 2]]",
            "assert solution([1, 2], 0) == []"
        ]
    },
    {
        "id": "fizz_buzz_list",
        "description": "Write a function `solution(n: int) -> list[str]` that returns a list of strings from 1 to n where multiples of 3 are 'Fizz', multiples of 5 are 'Buzz', multiples of both are 'FizzBuzz', otherwise the number as a string.",
        "test_cases": [
            "assert solution(3) == ['1', '2', 'Fizz']",
            "assert solution(5) == ['1', '2', 'Fizz', '4', 'Buzz']",
            "assert solution(15)[-1] == 'FizzBuzz'",
            "assert solution(1) == ['1']"
        ]
    },
    {
        "id": "climbing_stairs",
        "description": "Write a function `solution(n: int) -> int` that calculates how many distinct ways you can climb to the top of a staircase of n steps, taking either 1 or 2 steps each time. For n <= 0, return 0.",
        "test_cases": [
            "assert solution(2) == 2",
            "assert solution(3) == 3",
            "assert solution(4) == 5",
            "assert solution(1) == 1",
            "assert solution(0) == 0"
        ]
    },
    {
        "id": "single_number",
        "description": "Write a function `solution(nums: list[int]) -> int` where every element appears twice except for one. Find that single one.",
        "test_cases": [
            "assert solution([2, 2, 1]) == 1",
            "assert solution([4, 1, 2, 1, 2]) == 4",
            "assert solution([1]) == 1",
            "assert solution([-1, -1, -2]) == -2"
        ]
    },
    {
        "id": "power_of_two",
        "description": "Write a function `solution(n: int) -> bool` that returns True if n is a power of two (i.e. exists integer x such that n == 2^x), False otherwise.",
        "test_cases": [
            "assert solution(1) == True",
            "assert solution(16) == True",
            "assert solution(3) == False",
            "assert solution(0) == False",
            "assert solution(-16) == False"
        ]
    },
    {
        "id": "flatten_nested_list",
        "description": "Write a function `solution(nested: list) -> list` that flattens an arbitrarily nested list of integers into a single flat list.",
        "test_cases": [
            "assert solution([1, [2, [3, 4], 5], 6]) == [1, 2, 3, 4, 5, 6]",
            "assert solution([]) == []",
            "assert solution([[], [[], []]]) == []",
            "assert solution([1, 2, 3]) == [1, 2, 3]"
        ]
    },
    {
        "id": "reverse_words_in_string",
        "description": "Write a function `solution(s: str) -> str` that reverses the order of words in a string. Words are separated by at least one space. Return a string of words joined by a single space with no leading/trailing spaces.",
        "test_cases": [
            "assert solution('the sky is blue') == 'blue is sky the'",
            "assert solution('  hello world  ') == 'world hello'",
            "assert solution('a good   example') == 'example good a'",
            "assert solution('') == ''"
        ]
    },
    {
        "id": "product_except_self",
        "description": "Write a function `solution(nums: list[int]) -> list[int]` that returns an array such that output[i] is equal to the product of all the elements of nums except nums[i].",
        "test_cases": [
            "assert solution([1, 2, 3, 4]) == [24, 12, 8, 6]",
            "assert solution([-1, 1, 0, -3, 3]) == [0, 0, 9, 0, 0]",
            "assert solution([2, 3]) == [3, 2]"
        ]
    },
    {
        "id": "is_subsequence",
        "description": "Write a function `solution(s: str, t: str) -> bool` that checks if s is a subsequence of t.",
        "test_cases": [
            "assert solution('abc', 'ahbgdc') == True",
            "assert solution('axc', 'ahbgdc') == False",
            "assert solution('', 'ahbgdc') == True",
            "assert solution('abc', '') == False",
            "assert solution('', '') == True"
        ]
    },
    {
        "id": "gcd_of_two_numbers",
        "description": "Write a function `solution(a: int, b: int) -> int` that returns the greatest common divisor (GCD) of two non-negative integers a and b.",
        "test_cases": [
            "assert solution(48, 18) == 6",
            "assert solution(101, 103) == 1",
            "assert solution(0, 5) == 5",
            "assert solution(12, 0) == 12",
            "assert solution(7, 7) == 7"
        ]
    },
    {
        "id": "rotate_list",
        "description": "Write a function `solution(nums: list[int], k: int) -> list[int]` that rotates the array to the right by k steps, where k is non-negative.",
        "test_cases": [
            "assert solution([1, 2, 3, 4, 5, 6, 7], 3) == [5, 6, 7, 1, 2, 3, 4]",
            "assert solution([-1, -100, 3, 99], 2) == [3, 99, -1, -100]",
            "assert solution([1], 5) == [1]",
            "assert solution([], 3) == []",
            "assert solution([1, 2], 0) == [1, 2]"
        ]
    },
    {
        "id": "longest_common_prefix",
        "description": "Write a function `solution(strs: list[str]) -> str` that finds the longest common prefix string amongst an array of strings. If there is no common prefix, return empty string ''.",
        "test_cases": [
            "assert solution(['flower', 'flow', 'flight']) == 'fl'",
            "assert solution(['dog', 'racecar', 'car']) == ''",
            "assert solution([]) == ''",
            "assert solution(['alone']) == 'alone'",
            "assert solution(['ab', 'a']) == 'a'"
        ]
    },
    {
        "id": "sort_colors",
        "description": "Write a function `solution(nums: list[int]) -> list[int]` that sorts an array containing only 0s, 1s, and 2s in-place or returning a new sorted list.",
        "test_cases": [
            "assert solution([2, 0, 2, 1, 1, 0]) == [0, 0, 1, 1, 2, 2]",
            "assert solution([2, 0, 1]) == [0, 1, 2]",
            "assert solution([]) == []",
            "assert solution([0]) == [0]",
            "assert solution([1, 1]) == [1, 1]"
        ]
    },
    {
        "id": "roman_to_int",
        "description": "Write a function `solution(s: str) -> int` that converts a Roman numeral string ('I', 'V', 'X', 'L', 'C', 'D', 'M') into an integer.",
        "test_cases": [
            "assert solution('III') == 3",
            "assert solution('LVIII') == 58",
            "assert solution('MCMXCIV') == 1994",
            "assert solution('IX') == 9",
            "assert solution('IV') == 4"
        ]
    }
]


def extract_code_solution(completion_text: str) -> Optional[str]:
    """
    从模型的生成完成文本中精准抽取 Python 代码。
    优先匹配:
      1. <code>...</code>
      2. ```python ... ```
      3. ``` ... ```
    """
    # 1. 优先 <code>...</code>
    code_match = re.search(r"<code>(.*?)</code>", completion_text, flags=re.DOTALL)
    if code_match:
        return code_match.group(1).strip()

    # 2. 备用 ```python ... ```
    md_py = re.search(r"```python\s*(.*?)\s*```", completion_text, flags=re.DOTALL)
    if md_py:
        return md_py.group(1).strip()

    # 3. 备用 ``` ... ```
    md_generic = re.search(r"```\s*(.*?)\s*```", completion_text, flags=re.DOTALL)
    if md_generic:
        return md_generic.group(1).strip()

    return None


def execute_test_assertions(code_str: str, test_cases: List[str], timeout_sec: float = 1.0) -> bool:
    """Execute restricted algorithmic functions in a worker; judge results here.

    The worker receives literal inputs, never test assertions or expected outputs.
    It permits a small Python subset and selected algorithm libraries. It is not
    an execution service for arbitrary Python programs.
    """
    import ast
    import json
    from tasks.code_worker import validate_source

    if not code_str or not test_cases:
        return False
    try:
        validate_source(code_str)
        tests = []
        calls = []
        for case in test_cases:
            tree = ast.parse(case)
            if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assert):
                return False
            expression = tree.body[0].test
            count = 0
            for node in ast.walk(expression):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "solution":
                    if any(kw.arg is None for kw in node.keywords):
                        return False
                    args = [ast.literal_eval(arg) for arg in node.args]
                    kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
                    node.result_index = len(calls)
                    calls.append(repr((args, kwargs)))
                    count += 1
            if not count:
                return False
            tests.append(expression)
        worker_path = os.path.join(os.path.dirname(__file__), "code_worker.py")
        proc = subprocess.run(
            [sys.executable, "-I", "-S", worker_path],
            input=json.dumps({"source": code_str, "calls": calls}),
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=timeout_sec, text=True, cwd="/tmp",
            env={"PATH": os.defpath},
        )
        if proc.returncode != 0 or len(proc.stdout) > 1_000_000:
            return False
        results = json.loads(proc.stdout)
        if not isinstance(results, list) or len(results) != len(calls):
            return False
        values = [ast.literal_eval(value) for value in results]

        class SubstituteResults(ast.NodeTransformer):
            def visit_Call(self, node):
                if hasattr(node, "result_index"):
                    return ast.copy_location(ast.parse(repr(values[node.result_index]), mode="eval").body, node)
                return self.generic_visit(node)

        oracle_builtins = {"sum": sum, "sorted": sorted, "len": len, "min": min, "max": max, "abs": abs, "all": all, "any": any}
        for expression in tests:
            expression = ast.fix_missing_locations(ast.Expression(SubstituteResults().visit(expression)))
            if not eval(compile(expression, "<trusted-test>", "eval"), {"__builtins__": oracle_builtins}):
                return False
        return True
    except (Exception, subprocess.TimeoutExpired):
        return False


class CodeExecTask(BaseTask):
    """
    Python 代码执行与单元测试客观断言验证任务 (轨道二)。
    """

    def __init__(self, problems: Optional[List[Dict[str, Any]]] = None, val_split: int = 10):
        all_probs = problems or CURATED_CODING_PROBLEMS
        if len(all_probs) > val_split:
            self.train_problems = all_probs[:-val_split]
            self.val_problems = all_probs[-val_split:]
        else:
            self.train_problems = all_probs
            self.val_problems = all_probs[:max(1, len(all_probs) // 2)]
        self.sample_idx = 0

    def __len__(self) -> int:
        return len(self.train_problems)

    def get_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """获取一个批次的代码生成训练题目"""
        batch = []
        for _ in range(batch_size):
            prob = self.train_problems[self.sample_idx % len(self.train_problems)]
            self.sample_idx += 1

            prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"Problem:\n{prob['description']}\n\n"
                f"Assistant: <think>\n"
            )

            batch.append({
                "prompt": prompt,
                "problem_id": prob["id"],
                "description": prob["description"],
                "ground_truth": prob["test_cases"]
            })
        return batch

    def get_val_dataset(self) -> List[Dict[str, Any]]:
        """获取固定验证集 (用于贪心解码 Pass@1 确定性无偏基准评估)"""
        val_set = []
        for prob in self.val_problems:
            prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"Problem:\n{prob['description']}\n\n"
                f"Assistant: <think>\n"
            )
            val_set.append({
                "prompt": prompt,
                "problem_id": prob["id"],
                "description": prob["description"],
                "ground_truth": prob["test_cases"]
            })
        return val_set

    def compute_rewards(
        self,
        prompts: List[str],
        completions: List[str],
        metadata: List[Dict[str, Any]]
    ) -> List[float]:
        """
        并行在沙箱中执行各 completion 的测试断言，返回 1.0 (全过) 或 0.0 (未过)。
        """
        assert len(completions) == len(metadata), "completions 与 metadata 长度必须一致！"

        def _verify_single(comp: str, meta: Dict[str, Any]) -> float:
            # 门禁 1: 思考链必须显式闭合且内容非空
            if "</think>" not in comp:
                return 0.0
            think_body = comp.split("</think>")[0].strip()
            if len(think_body) < 15:
                return 0.0

            # 门禁 2: 提取代码块
            code_str = extract_code_solution(comp)
            if not code_str:
                return 0.0

            test_cases = meta.get("ground_truth", [])
            if not test_cases:
                return 0.0

            # 门禁 3: 真实沙箱运行断言
            passed = execute_test_assertions(code_str, test_cases, timeout_sec=1.0)
            return 1.0 if passed else 0.0

        # 并行沙箱执行 (8路并发验证，耗时 < 50ms)
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(completions), 8)) as executor:
            futures = [executor.submit(_verify_single, c, m) for c, m in zip(completions, metadata)]
            rewards = [f.result() for f in futures]

        return rewards


if __name__ == "__main__":
    print("=" * 60)
    print(">>> 正在自测 tasks/code_exec.py 沙箱执行与奖励模块...")
    task = CodeExecTask()
    print(f"题库规模: {len(task)} 道经典编程题")

    batch = task.get_batch(1)
    sample = batch[0]
    print(f"\n[Prompt 示例预览]:\n{sample['prompt'][:220]}...\n")
    print(f"题目 ID: {sample['problem_id']}")
    print(f"单元测试用例数: {len(sample['ground_truth'])}")

    # 构造 5 种边界 completion 测试沙箱执行器
    correct_comp = (
        "<think> We iterate through the array and move zeroes. </think>\n"
        "<code>\n"
        "def solution(nums: list[int]) -> list[int]:\n"
        "    non_zeros = [x for x in nums if x != 0]\n"
        "    zero_count = len(nums) - len(non_zeros)\n"
        "    return non_zeros + [0] * zero_count\n"
        "</code>"
    )

    wrong_name_comp = (
        "<think> Good logic but wrong function name. </think>\n"
        "<code>\n"
        "def move_zeroes(nums):\n"
        "    return nums\n"
        "</code>"
    )

    buggy_comp = (
        "<think> Returning wrong answer for zeroes. </think>\n"
        "<code>\n"
        "def solution(nums):\n"
        "    return nums\n"
        "</code>"
    )

    infinite_loop_comp = (
        "<think> Timeout trap. </think>\n"
        "<code>\n"
        "def solution(nums):\n"
        "    while True: pass\n"
        "</code>"
    )

    no_think_comp = (
        "<code>def solution(nums): return nums</code>"
    )

    comps = [correct_comp, wrong_name_comp, buggy_comp, infinite_loop_comp, no_think_comp]
    metas = [sample] * 5

    rewards = task.compute_rewards([""] * 5, comps, metas)
    print("\n[沙箱奖励审计测试结果]:")
    print(f"  Case 1 (标准正确通过): {rewards[0]} (期望 1.0)")
    print(f"  Case 2 (函数名写错):   {rewards[1]} (期望 0.0)")
    print(f"  Case 3 (逻辑Bug断言挂):{rewards[2]} (期望 0.0)")
    print(f"  Case 4 (死循环超时中断):{rewards[3]} (期望 0.0)")
    print(f"  Case 5 (未闭合think):  {rewards[4]} (期望 0.0)")

    assert rewards == [1.0, 0.0, 0.0, 0.0, 0.0], f"沙箱奖励断言失败: {rewards}"
    print("\n✅ tasks/code_exec.py 沙箱执行器所有单元测试全量通过！")
    print("=" * 60)
