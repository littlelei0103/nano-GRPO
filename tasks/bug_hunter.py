"""
tasks/bug_hunter.py - 变异反例审计与漏洞捕获轨道 (BugHunter RLVR Track)

全新范式:
  不同于传统“让模型当做题家写代码”的同质化思路，BugHunter 训练大模型作为“AI 安全审计员 / 红队 QA 架构师”。

核心机制:
  1. 给定函数功能规范 (Spec) 与一段暗含真实隐蔽边缘 Bug 的代码 (Mutated Implementation)。
  2. 模型在 <think> 中推演边界漏洞，在 <code> 中输出一条精确打击的断言:
     assert target_func(...) == expected_value
  3. 双向沙盒客观验证 (Bi-Directional Sandbox Verification):
     - 必须在 Golden Reference (标准参考实现) 上顺利通过 (证明测试用例合法、符合题意)；
     - 必须在 Buggy Code (待测缺陷代码) 上触发 AssertionError 或运行时异常 (证明成功击穿漏洞)！
  4. 满足上述双向客观真理判定则给 +1.0，否则给 0.0。
"""

import os
import re
import sys
import ast
import subprocess
import concurrent.futures
from typing import List, Dict, Any, Optional

# 允许直接独立测试调试
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tasks.base import BaseTask

SYSTEM_PROMPT = (
    "You are an expert adversarial software testing engineer and vulnerability auditor.\n"
    "Your objective is to find a subtle edge-case defect in the provided Python function `target_func`.\n\n"
    "Instructions:\n"
    "1. In <think> and </think> tags, analyze the implementation, locate the edge-case flaw, and deduce an input where the code fails.\n"
    "2. In <code> and </code> tags, output a single Python test assertion:\n"
    "   assert target_func(...) == expected_value\n\n"
    "CRITICAL RULES:\n"
    "- Your assertion must be TRUE according to the problem specification (it must pass on a correct implementation).\n"
    "- Your assertion must EXPOSE the defect (it must trigger an AssertionError or exception on the buggy implementation).\n"
    "- Output ONLY the `assert target_func(...) == ...` line inside the <code> block."
)

CURATED_BUG_HUNTER_PROBLEMS = [
    {
        "id": "binary_search_right_edge",
        "spec": "Given a sorted list of integers `arr` and a `target`, return the 0-based index of `target`, or -1 if not found.",
        "buggy_code": (
            "def target_func(arr: list[int], target: int) -> int:\n"
            "    left, right = 0, len(arr) - 1\n"
            "    while left < right:"
            "        mid = (left + right) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        elif arr[mid] < target:\n"
            "            left = mid + 1\n"
            "        else:\n"
            "            right = mid - 1\n"
            "    return -1"
        ),
        "golden_code": (
            "def target_func(arr: list[int], target: int) -> int:\n"
            "    left, right = 0, len(arr) - 1\n"
            "    while left <= right:\n"
            "        mid = (left + right) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        elif arr[mid] < target:\n"
            "            left = mid + 1\n"
            "        else:\n"
            "            right = mid - 1\n"
            "    return -1"
        )
    },
    {
        "id": "longest_increasing_empty_list",
        "spec": "Given a list of integers `nums`, return the length of the longest strictly increasing contiguous subarray. Return 0 for an empty list.",
        "buggy_code": (
            "def target_func(nums: list[int]) -> int:\n"
            ""
            "    cur, ans = 1, 1\n"
            "    for i in range(1, len(nums)):\n"
            "        if nums[i] > nums[i - 1]:\n"
            "            cur += 1\n"
            "        else:\n"
            "            cur = 1\n"
            "        ans = max(ans, cur)\n"
            "    return ans"
        ),
        "golden_code": (
            "def target_func(nums: list[int]) -> int:\n"
            "    if not nums: return 0\n"
            "    cur, ans = 1, 1\n"
            "    for i in range(1, len(nums)):\n"
            "        if nums[i] > nums[i - 1]:\n"
            "            cur += 1\n"
            "        else:\n"
            "            cur = 1\n"
            "        ans = max(ans, cur)\n"
            "    return ans"
        )
    },
    {
        "id": "merge_intervals_adjacent",
        "spec": "Given a list of intervals `[[start, end], ...]`, merge all overlapping intervals. Adjacent intervals sharing an endpoint like [1, 2] and [2, 3] MUST be merged into [1, 3].",
        "buggy_code": (
            "def target_func(intervals: list[list[int]]) -> list[list[int]]:\n"
            "    if not intervals: return []\n"
            "    intervals.sort(key=lambda x: x[0])\n"
            "    merged = [intervals[0]]\n"
            "    for cur in intervals[1:]:\n"
            "        prev = merged[-1]\n"
            "        if cur[0] < prev[1]:"
            "            prev[1] = max(prev[1], cur[1])\n"
            "        else:\n"
            "            merged.append(cur)\n"
            "    return merged"
        ),
        "golden_code": (
            "def target_func(intervals: list[list[int]]) -> list[list[int]]:\n"
            "    if not intervals: return []\n"
            "    intervals.sort(key=lambda x: x[0])\n"
            "    merged = [intervals[0]]\n"
            "    for cur in intervals[1:]:\n"
            "        prev = merged[-1]\n"
            "        if cur[0] <= prev[1]:\n"
            "            prev[1] = max(prev[1], cur[1])\n"
            "        else:\n"
            "            merged.append(cur)\n"
            "    return merged"
        )
    },
    {
        "id": "max_subarray_all_negative",
        "spec": "Given an integer array `nums`, find the contiguous subarray with the largest sum and return its sum. Must work correctly for all-negative arrays.",
        "buggy_code": (
            "def target_func(nums: list[int]) -> int:\n"
            ""
            "    max_so_far = 0\n"
            "    cur = 0\n"
            "    for x in nums:\n"
            "        cur = max(x, cur + x)\n"
            "        max_so_far = max(max_so_far, cur)\n"
            "    return max_so_far"
        ),
        "golden_code": (
            "def target_func(nums: list[int]) -> int:\n"
            "    max_so_far = nums[0]\n"
            "    cur = nums[0]\n"
            "    for x in nums[1:]:\n"
            "        cur = max(x, cur + x)\n"
            "        max_so_far = max(max_so_far, cur)\n"
            "    return max_so_far"
        )
    },
    {
        "id": "two_sum_same_element_reuse",
        "spec": "Given an array of integers `nums` with exactly one valid pair and `target`, return sorted 0-based indices [i, j] with i < j where nums[i] + nums[j] == target. Cannot reuse same element.",
        "precondition": "def validate(nums, target):\n    return isinstance(nums, list) and sum(1 for i in range(len(nums)) for j in range(i+1, len(nums)) if nums[i] + nums[j] == target) == 1",
        "buggy_code": (
            "def target_func(nums: list[int], target: int) -> list[int]:\n"
            "    lookup = {num: i for i, num in enumerate(nums)}\n"
            "    for i, num in enumerate(nums):\n"
            "        diff = target - num\n"
            "        if diff in lookup:"
            "            return sorted([i, lookup[diff]])\n"
            "    return []"
        ),
        "golden_code": (
            "def target_func(nums: list[int], target: int) -> list[int]:\n"
            "    lookup = {}\n"
            "    for i, num in enumerate(nums):\n"
            "        diff = target - num\n"
            "        if diff in lookup:\n"
            "            return sorted([lookup[diff], i])\n"
            "        lookup[num] = i\n"
            "    return []"
        )
    },
    {
        "id": "reverse_vowels_case_sensitivity",
        "spec": "Given a string `s`, reverse only all the vowels in the string. Vowels include 'a', 'e', 'i', 'o', 'u' in both lowercase AND uppercase.",
        "buggy_code": (
            "def target_func(s: str) -> str:\n"
            "    vowels = set('aeiou')"
            "    chars = list(s)\n"
            "    i, j = 0, len(s) - 1\n"
            "    while i < j:\n"
            "        if chars[i] not in vowels:\n"
            "            i += 1\n"
            "        elif chars[j] not in vowels:\n"
            "            j -= 1\n"
            "        else:\n"
            "            chars[i], chars[j] = chars[j], chars[i]\n"
            "            i += 1\n"
            "            j -= 1\n"
            "    return ''.join(chars)"
        ),
        "golden_code": (
            "def target_func(s: str) -> str:\n"
            "    vowels = set('aeiouAEIOU')\n"
            "    chars = list(s)\n"
            "    i, j = 0, len(s) - 1\n"
            "    while i < j:\n"
            "        if chars[i] not in vowels:\n"
            "            i += 1\n"
            "        elif chars[j] not in vowels:\n"
            "            j -= 1\n"
            "        else:\n"
            "            chars[i], chars[j] = chars[j], chars[i]\n"
            "            i += 1\n"
            "            j -= 1\n"
            "    return ''.join(chars)"
        )
    },
    {
        "id": "rotate_array_large_k",
        "spec": "Given an integer array `nums`, rotate the array to the right by `k` steps, where `k` is non-negative and can be greater than len(nums).",
        "buggy_code": (
            "def target_func(nums: list[int], k: int) -> list[int]:\n"
            "    if not nums: return []\n"
            ""
            "    return nums[-k:] + nums[:-k]"
        ),
        "golden_code": (
            "def target_func(nums: list[int], k: int) -> list[int]:\n"
            "    if not nums: return []\n"
            "    k = k % len(nums)\n"
            "    if k == 0: return nums[:]\n"
            "    return nums[-k:] + nums[:-k]"
        )
    },
    {
        "id": "is_subsequence_empty_pattern",
        "spec": "Given two strings `s` and `t`, return True if `s` is a subsequence of `t`. An empty string `s` is always a valid subsequence of any string.",
        "buggy_code": (
            "def target_func(s: str, t: str) -> bool:\n"
            "    if not s: return False"
            "    i, j = 0, 0\n"
            "    while i < len(s) and j < len(t):\n"
            "        if s[i] == t[j]:\n"
            "            i += 1\n"
            "        j += 1\n"
            "    return i == len(s)"
        ),
        "golden_code": (
            "def target_func(s: str, t: str) -> bool:\n"
            "    if not s: return True\n"
            "    i, j = 0, 0\n"
            "    while i < len(s) and j < len(t):\n"
            "        if s[i] == t[j]:\n"
            "            i += 1\n"
            "        j += 1\n"
            "    return i == len(s)"
        )
    },
    {
        "id": "count_primes_strict_inequality",
        "spec": "Given an integer `n`, return the number of prime numbers strictly less than `n`.",
        "buggy_code": (
            "def target_func(n: int) -> int:\n"
            "    if n <= 2: return 0\n"
            ""
            "    is_prime = [True] * (n + 1)\n"
            "    is_prime[0] = is_prime[1] = False\n"
            "    for p in range(2, int((n + 1)**0.5) + 1):\n"
            "        if is_prime[p]:\n"
            "            for multiple in range(p * p, n + 1, p):\n"
            "                is_prime[multiple] = False\n"
            "    return sum(is_prime)  # Fails when n itself is prime!"
        ),
        "golden_code": (
            "def target_func(n: int) -> int:\n"
            "    if n <= 2: return 0\n"
            "    is_prime = [True] * n\n"
            "    is_prime[0] = is_prime[1] = False\n"
            "    for p in range(2, int(n**0.5) + 1):\n"
            "        if is_prime[p]:\n"
            "            for multiple in range(p * p, n, p):\n"
            "                is_prime[multiple] = False\n"
            "    return sum(is_prime)"
        )
    },
    {
        "id": "power_of_two_zero_and_negative",
        "spec": "Given an integer `n`, return True if it is a power of two (i.e. exists integer x such that n == 2^x). Must return False for n <= 0.",
        "buggy_code": (
            "def target_func(n: int) -> bool:\n"
            ""
            "    return (n & (n - 1)) == 0"
        ),
        "golden_code": (
            "def target_func(n: int) -> bool:\n"
            "    return n > 0 and (n & (n - 1)) == 0"
        )
    },
    {
        "id": "string_compression_expansion",
        "spec": "Compress string using counts of repeated characters (e.g. 'aabcccccaaa' -> 'a2b1c5a3'). If the compressed string is NOT strictly shorter than original, return the original string.",
        "buggy_code": (
            "def target_func(s: str) -> str:\n"
            "    if not s: return ''\n"
            "    res = []\n"
            "    count = 1\n"
            "    for i in range(1, len(s)):\n"
            "        if s[i] == s[i - 1]:\n"
            "            count += 1\n"
            "        else:\n"
            "            res.append(f'{s[i - 1]}{count}')\n"
            "            count = 1\n"
            "    res.append(f'{s[-1]}{count}')\n"
            ""
            "    return ''.join(res)"
        ),
        "golden_code": (
            "def target_func(s: str) -> str:\n"
            "    if not s: return ''\n"
            "    res = []\n"
            "    count = 1\n"
            "    for i in range(1, len(s)):\n"
            "        if s[i] == s[i - 1]:\n"
            "            count += 1\n"
            "        else:\n"
            "            res.append(f'{s[i - 1]}{count}')\n"
            "            count = 1\n"
            "    res.append(f'{s[-1]}{count}')\n"
            "    comp = ''.join(res)\n"
            "    return comp if len(comp) < len(s) else s"
        )
    },
    {
        "id": "climbing_stairs_zero_steps",
        "spec": "Calculate distinct ways to climb n stairs taking 1 or 2 steps each time. For n <= 0, return 0.",
        "buggy_code": (
            "def target_func(n: int) -> int:\n"
            ""
            "    if n <= 1: return 1\n"
            "    a, b = 1, 1\n"
            "    for _ in range(2, n + 1):\n"
            "        a, b = b, a + b\n"
            "    return b"
        ),
        "golden_code": (
            "def target_func(n: int) -> int:\n"
            "    if n <= 0: return 0\n"
            "    if n == 1: return 1\n"
            "    a, b = 1, 1\n"
            "    for _ in range(2, n + 1):\n"
            "        a, b = b, a + b\n"
            "    return b"
        )
    },
    {
        "id": "first_unique_char_none_exists",
        "spec": "Return the 0-based index of the first non-repeating character in string `s`. If none exists or string is empty, return -1.",
        "buggy_code": (
            "def target_func(s: str) -> int:\n"
            "    from collections import Counter\n"
            "    counts = Counter(s)\n"
            "    for i, c in enumerate(s):\n"
            "        if counts[c] == 1:\n"
            "            return i\n"
            ""
            "    return 0"
        ),
        "golden_code": (
            "def target_func(s: str) -> int:\n"
            "    from collections import Counter\n"
            "    counts = Counter(s)\n"
            "    for i, c in enumerate(s):\n"
            "        if counts[c] == 1:\n"
            "            return i\n"
            "    return -1"
        )
    },
    {
        "id": "chunk_list_zero_step",
        "spec": "Split a list `lst` into chunks of given `size`. If size <= 0 or list is empty, return [].",
        "buggy_code": (
            "def target_func(lst: list, size: int) -> list[list]:\n"
            ""
            "    return [lst[i:i + size] for i in range(0, len(lst), size)]"
        ),
        "golden_code": (
            "def target_func(lst: list, size: int) -> list[list]:\n"
            "    if size <= 0 or not lst: return []\n"
            "    return [lst[i:i + size] for i in range(0, len(lst), size)]"
        )
    },
    {
        "id": "fizz_buzz_ordering_trap",
        "spec": "Return a list of strings from 1 to n. For multiples of 3 return 'Fizz', for multiples of 5 return 'Buzz', and for multiples of BOTH 3 and 5 return 'FizzBuzz'.",
        "buggy_code": (
            "def target_func(n: int) -> list[str]:\n"
            "    res = []\n"
            "    for i in range(1, n + 1):\n"
            ""
            "        if i % 3 == 0:\n"
            "            res.append('Fizz')\n"
            "        elif i % 5 == 0:\n"
            "            res.append('Buzz')\n"
            "        elif i % 15 == 0:\n"
            "            res.append('FizzBuzz')\n"
            "        else:\n"
            "            res.append(str(i))\n"
            "    return res"
        ),
        "golden_code": (
            "def target_func(n: int) -> list[str]:\n"
            "    res = []\n"
            "    for i in range(1, n + 1):\n"
            "        if i % 15 == 0:\n"
            "            res.append('FizzBuzz')\n"
            "        elif i % 3 == 0:\n"
            "            res.append('Fizz')\n"
            "        elif i % 5 == 0:\n"
            "            res.append('Buzz')\n"
            "        else:\n"
            "            res.append(str(i))\n"
            "    return res"
        )
    }
]


def is_pure_literal_node(node: ast.AST) -> bool:
    """递归验证 AST 节点是否为严格纯字面量 (禁止任意变量、属性、函数调用、赋值表达式与复杂逻辑)"""
    import ast
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return isinstance(node.operand, ast.Constant)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(is_pure_literal_node(elt) for elt in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            k is not None and is_pure_literal_node(k) and is_pure_literal_node(v)
            for k, v in zip(node.keys, node.values)
        )
    return False


def validate_and_parse_assertion(assertion_str: str) -> tuple[bool, str]:
    """
    通过 AST 抽象语法树对模型生成的测试断言进行严格结构白名单验证。
    必须满足结构:
      assert target_func(<pure_literal_args>, <pure_literal_kwargs>) <comp_op> <pure_literal_expected>

    严格拒绝:
      - 任意命名表达式 (如 NamedExpr :=)
      - 任意属性访问 (如 sys._getframe, f_code, co_consts, __dict__)
      - 任意死分支与短路逻辑 (如 target_func() and False)
      - 任意全局或局部变量引用 (只允许字面量)
    """
    if not assertion_str or not assertion_str.strip():
        return False, "Empty assertion string"

    import ast
    try:
        tree = ast.parse(assertion_str.strip())
    except SyntaxError as e:
        return False, f"SyntaxError in assertion: {e}"

    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assert):
        return False, "Completion must contain exactly one assert statement"

    stmt = tree.body[0]
    if stmt.msg is not None or sum(1 for _ in ast.walk(tree)) > 4096:
        return False, "Assertion messages and oversized expressions are disallowed"
    if not isinstance(stmt.test, ast.Compare):
        return False, "Assert test must be an explicit comparison (e.g. target_func(...) == expected)"

    # 1. 严格只允许单个比较操作符 (==, is, in)
    valid_ops = (ast.Eq, ast.Is, ast.In)
    if len(stmt.test.ops) != 1 or len(stmt.test.comparators) != 1:
        return False, "Comparison must have exactly one operator and one comparator"
    if not isinstance(stmt.test.ops[0], valid_ops):
        return False, "Comparison must use '==', 'is', or 'in'"

    # 2. 左侧必须且只能是直接调用 target_func
    left = stmt.test.left
    if not isinstance(left, ast.Call):
        return False, "Left side of comparison must be an explicit direct call to target_func(...)"
    if not isinstance(left.func, ast.Name) or left.func.id != "target_func":
        return False, "Called function must be exactly 'target_func'"

    # 3. 函数实参必须全部为纯字面量 (禁止变量、表达式或属性访问)
    for i, arg in enumerate(left.args):
        if not is_pure_literal_node(arg):
            return False, f"Argument {i+1} in target_func call is not a pure literal"

    for kw in left.keywords:
        if kw.arg is None or not is_pure_literal_node(kw.value):
            return False, f"Keyword argument '{kw.arg}' is not a pure literal"

    # 4. 右侧期望值也必须是纯字面量
    right = stmt.test.comparators[0]
    if not is_pure_literal_node(right):
        return False, "Expected comparison value must be a pure literal"

    # 5. 全树安全扫描，禁止任何属性访问、变量读取 (除 target_func 外) 或赋值表达式
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            return False, f"Attribute access '{node.attr}' is strictly disallowed"
        if isinstance(node, ast.NamedExpr):
            return False, "NamedExpr (:=) is strictly disallowed"
        if isinstance(node, ast.Name) and node.id != "target_func":
            return False, f"Variable '{node.id}' is disallowed; only literals permitted"

    return True, "Valid"


def extract_call_args_and_expected(assertion_str: str) -> tuple[list, dict, Any]:
    """从通过验证的断言字符串中安全解析出位置参数、关键字参数与期望返回值"""
    import ast
    tree = ast.parse(assertion_str.strip())
    call_node = tree.body[0].test.left
    comp_node = tree.body[0].test.comparators[0]

    args = [ast.literal_eval(a) for a in call_node.args]
    kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in call_node.keywords}
    expected = ast.literal_eval(comp_node)
    return args, kwargs, expected


def extract_assertion(completion_text: str) -> Optional[str]:
    """从模型输出中精准抽取符合严格语法的测试断言"""
    code_match = re.search(r"<code>(.*?)</code>", completion_text, flags=re.DOTALL)
    candidate = code_match.group(1).strip() if code_match else completion_text

    # 优先在每一行中寻找合规 assert target_func
    for line in candidate.split("\n"):
        line_clean = line.strip()
        if line_clean.startswith("assert") and "target_func" in line_clean:
            is_valid, _ = validate_and_parse_assertion(line_clean)
            if is_valid:
                return line_clean

    match = re.search(r"(assert\s+target_func\([^)]*\)\s*==\s*[^\n]+)", candidate)
    if match:
        cand_str = match.group(1).strip()
        is_valid, _ = validate_and_parse_assertion(cand_str)
        if is_valid:
            return cand_str

    return None


def verify_bug_exposure_detailed(
    assertion_str: str,
    golden_code: str,
    buggy_code: str,
    precondition: Optional[Any] = None,
    timeout_sec: float = 1.0,
    input_contract: Optional[str] = None,
) -> tuple[bool, str, str]:
    """
    在隔离子进程中进行严格受控的双向沙盒验证并返回细粒度诊断信息:
    返回: (exposed: bool, category: str, reason: str)
      category 可选值:
        - "success": 成功击穿漏洞
        - "format_malformed": AST 解析失败或不符合纯字面量规范
        - "precondition_violation": 输入违反题目前置条件/合约
        - "golden_failed": 预期答案错误或 Golden 执行失败
        - "missed_bug": 断言正确但未能击穿 Buggy 代码缺陷
        - "execution_error": 超时或其他执行异常
    """
    is_valid, err_msg = validate_and_parse_assertion(assertion_str)
    if not is_valid:
        return False, "format_malformed", f"AST validation failed: {err_msg}"

    if input_contract:
        from tasks.contracts import validate_input
        try:
            args, kwargs, _ = extract_call_args_and_expected(assertion_str)
            if not validate_input(input_contract, golden_code, args, kwargs):
                return False, "precondition_violation", f"Input violates input_contract '{input_contract}'"
        except (ValueError, TypeError) as e:
            return False, "precondition_violation", f"Failed to extract or validate input contract: {e}"

    # 验证输入域前置条件 (Precondition Validation)
    if precondition:
        try:
            args, kwargs, _ = extract_call_args_and_expected(assertion_str)
            if callable(precondition):
                if not precondition(*args, **kwargs):
                    return False, "precondition_violation", "Input violates callable precondition"
            elif isinstance(precondition, str):
                ns = {}
                exec(precondition, ns)
                val_fn = ns.get("validate") or ns.get("validate_input")
                if not callable(val_fn) or not val_fn(*args, **kwargs):
                    return False, "precondition_violation", "Input violates string precondition"
        except Exception as e:
            return False, "precondition_violation", f"Precondition execution error: {e}"

    harness = """
import sys

# 注入实现
{code}

# 隔离执行断言
try:
    {assertion}
except AssertionError:
    sys.exit(42)  # 成功诱发断言失败！
except Exception:
    sys.exit(43)  # 成功诱发运行时异常！
except BaseException:
    sys.exit(44)  # 捕获 SystemExit 等非法越狱！

sys.exit(0)  # 正常无异常通过
"""

    script_golden = harness.format(code=golden_code, assertion=assertion_str)
    script_buggy = harness.format(code=buggy_code, assertion=assertion_str)

    try:
        # 1. 验证合法性 (Golden 必须正常无异常通过，exit code 0)
        p_golden = subprocess.run(
            [sys.executable, "-c", script_golden],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            text=True
        )
        if p_golden.returncode != 0:
            err = p_golden.stderr.strip()[:150] or f"exit code {p_golden.returncode}"
            return False, "golden_failed", f"Golden implementation failed ({err})"

        # 2. 验证有效性 (Buggy 必须被击穿，返回 42 (AssertionError), 43 (Exception), 44 (BaseException/SystemExit))
        p_buggy = subprocess.run(
            [sys.executable, "-c", script_buggy],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            text=True
        )
        if p_buggy.returncode in (42, 43, 44):
            return True, "success", f"Buggy defect exposed (exit code {p_buggy.returncode})"
        elif p_buggy.returncode == 0:
            return False, "missed_bug", "Buggy passed with exit code 0 (missed defect)"
        else:
            err = p_buggy.stderr.strip()[:150] or f"exit code {p_buggy.returncode}"
            return False, "execution_error", f"Buggy returned unexpected code ({err})"
    except subprocess.TimeoutExpired:
        return False, "execution_error", "Subprocess execution timed out"
    except Exception as e:
        return False, "execution_error", f"Execution exception: {e}"


def verify_bug_exposure(
    assertion_str: str,
    golden_code: str,
    buggy_code: str,
    precondition: Optional[Any] = None,
    timeout_sec: float = 1.0,
    input_contract: Optional[str] = None,
) -> bool:
    """
    在隔离子进程中进行严格受控的双向沙盒验证:
      1. assertion_str 必须通过纯字面量 AST 严格结构合规检查；
      2. 若题目指定了可执行的前置条件 (precondition)，输入参数必须满足前置条件检验；
      3. assertion_str 在 golden_code 上必须正常执行完毕 (exit code 0)；
      4. assertion_str 在 buggy_code 上必须被捕获为 AssertionError (exit 42) 或运行时异常 (exit 43, 44)；
      5. 任何通过 SystemExit 试图提前退出的越狱均被判定为击穿或异常。
    """
    exposed, _, _ = verify_bug_exposure_detailed(
        assertion_str=assertion_str,
        golden_code=golden_code,
        buggy_code=buggy_code,
        precondition=precondition,
        timeout_sec=timeout_sec,
        input_contract=input_contract
    )
    return exposed


class BugHunterTask(BaseTask):
    """
    变异反例审计与漏洞捕获任务 (BugHunter RLVR Track)。
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        problems: Optional[Any] = None,
        val_split: int = 20,
        test_split: int = 20
    ):
        default_100_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bug_hunter_dataset.json")
        fallback_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bug_hunter_100.json")

        if data_path:
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"指定的题目数据集路径不存在: {data_path}")
            target_path = data_path
        else:
            target_path = default_100_path if os.path.exists(default_100_path) else (fallback_path if os.path.exists(fallback_path) else None)

        all_probs: Any = None
        if problems is not None:
            all_probs = problems
        elif target_path and os.path.exists(target_path):
            import json
            with open(target_path, "r", encoding="utf-8") as f:
                all_probs = json.load(f)
        else:
            all_probs = CURATED_BUG_HUNTER_PROBLEMS

        # 检查是否为标准三级切分字典 {"train": [...], "val": [...], "test": [...]}
        if isinstance(all_probs, dict) and "train" in all_probs:
            self.train_problems = all_probs["train"]
            self.val_problems = all_probs.get("val", [])
            self.test_problems = all_probs.get("test", [])
        else:
            probs_list = list(all_probs)
            n = len(probs_list)
            if test_split > 0 and val_split > 0 and (val_split + test_split < n):
                self.train_problems = probs_list[: n - val_split - test_split]
                self.val_problems = probs_list[n - val_split - test_split : n - test_split]
                self.test_problems = probs_list[n - test_split :]
            elif val_split > 0 and val_split < n:
                self.train_problems = probs_list[:-val_split]
                self.val_problems = probs_list[-val_split:]
                self.test_problems = []
            else:
                self.train_problems = probs_list
                self.val_problems = []
                self.test_problems = []

        # 预构建所有题目的 Prompt 与元数据
        self.train_problems = [self._build_item(p) for p in self.train_problems]
        self.val_problems = [self._build_item(p) for p in self.val_problems]
        self.test_problems = [self._build_item(p) for p in self.test_problems]

        self.sample_idx = 0

    def __len__(self) -> int:
        return len(self.train_problems)

    def _build_item(self, prob: Dict[str, Any]) -> Dict[str, Any]:
        from tasks.contracts import CONTRACTS
        problem_id = prob.get("id", prob.get("problem_id", ""))
        if "prompt" in prob:
            return prob
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Problem Specification:\n{prob['spec']}\n\n"
            f"Implementation to Audit:\n```python\n{prob['buggy_code']}\n```\n\n"
            f"Assistant: <think>\n"
        )
        return {
            "prompt": prompt,
            "id": prob.get("id", prob.get("problem_id", "")),
            "problem_id": prob.get("id", prob.get("problem_id", "")),
            "spec": prob["spec"],
            "buggy_code": prob["buggy_code"],
            "golden_code": prob["golden_code"],
            "precondition": prob.get("precondition", None),
            "input_contract": prob.get("input_contract", problem_id if problem_id in CONTRACTS else None),
            "ground_truth": prob.get("id", prob.get("problem_id", ""))
        }

    def get_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        batch = []
        for _ in range(batch_size):
            prob = self.train_problems[self.sample_idx % len(self.train_problems)]
            self.sample_idx += 1
            batch.append(prob)
        return batch

    def get_val_dataset(self) -> List[Dict[str, Any]]:
        return self.val_problems

    def get_test_dataset(self) -> List[Dict[str, Any]]:
        return self.test_problems

    def evaluate_single_completion(self, comp: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        对单个生成结果进行细粒度判题，解耦真实截断、格式合规与任务正确性。
        类别包括:
          - 'success': 1.0 (击穿缺陷，任务成功)
          - 'golden_failed': 0.0 (预期答案错误或正确实现无法通过)
          - 'missed_bug': 0.0 (断言合规且Golden通过，但未能击穿Buggy缺陷)
          - 'precondition_violation': 0.0 (输入参数违反题目前置条件/合约)
          - 'format_malformed': 0.0 (生成未截断但未产生合规assert语句或AST解析失败)
          - 'truncated': 0.0 (达到max_new_tokens上限且未输出完整断言)
          - 'execution_error': 0.0 (子进程超时或系统异常)
        """
        is_truncated = meta.get("is_truncated", False)
        format_compliant = ("</think>" in comp)

        assertion_str = extract_assertion(comp)
        if not assertion_str:
            if is_truncated:
                return {
                    "reward": 0.0,
                    "category": "truncated",
                    "reason": "Generation hit token limit before completing assertion",
                    "assertion": None,
                    "format_compliant": False
                }
            return {
                "reward": 0.0,
                "category": "format_malformed",
                "reason": "No valid assert target_func statement found in finished text",
                "assertion": None,
                "format_compliant": False
            }

        exposed, category, reason = verify_bug_exposure_detailed(
            assertion_str=assertion_str,
            golden_code=meta["golden_code"],
            buggy_code=meta["buggy_code"],
            precondition=meta.get("precondition"),
            input_contract=meta.get("input_contract"),
            timeout_sec=1.0
        )
        return {
            "reward": 1.0 if exposed else 0.0,
            "category": category,
            "reason": reason,
            "assertion": assertion_str,
            "format_compliant": format_compliant
        }

    def compute_rewards_with_details(
        self,
        prompts: List[str],
        completions: List[str],
        metadata: List[Dict[str, Any]]
    ) -> tuple[List[float], List[Dict[str, Any]]]:
        assert len(completions) == len(metadata)
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(completions), 8)) as executor:
            futures = [executor.submit(self.evaluate_single_completion, c, m) for c, m in zip(completions, metadata)]
            details = [f.result() for f in futures]
        rewards = [d["reward"] for d in details]
        return rewards, details

    def compute_rewards(
        self,
        prompts: List[str],
        completions: List[str],
        metadata: List[Dict[str, Any]]
    ) -> List[float]:
        rewards, _ = self.compute_rewards_with_details(prompts, completions, metadata)
        return rewards


if __name__ == "__main__":
    print("=" * 60)
    print(">>> 正在自测 tasks/bug_hunter.py 沙箱执行器...")
    task = BugHunterTask()
    print(f"训练题库: {len(task)} 道 | 验证集: {len(task.get_val_dataset())} 道")

    batch = task.get_batch(1)
    sample = batch[0]
    print(f"\n[题目 ID]: {sample['problem_id']}")
    print(f"[规格说明]: {sample['spec']}")

    # Case 1: 准确击穿边界 (测试最右侧边界元素，buggy 返回 -1，golden 返回 2)
    comp_success = (
        "<think> The loop uses left < right which terminates early when the target is at the right boundary. We test target at the end. </think>\n"
        "<code>\nassert target_func([1, 3, 5], 5) == 2\n</code>"
    )

    # Case 2: 无效断言 (与题目规格相悖，Golden 也会挂)
    comp_invalid = (
        "<think> Let's test random wrong value. </think>\n"
        "<code>\nassert target_func([1, 3, 5], 5) == 999\n</code>"
    )

    # Case 3: 命中正常分支 (Buggy 也能跑通，未能击穿 Bug)
    comp_miss = (
        "<think> Normal input at the middle where code passes. </think>\n"
        "<code>\nassert target_func([1, 3, 5], 3) == 1\n</code>"
    )

    rewards = task.compute_rewards([""] * 3, [comp_success, comp_invalid, comp_miss], [sample] * 3)
    print("\n[双向沙盒奖励审计结果]:")
    print(f"  Case 1 (精准反例击穿漏洞): {rewards[0]} (期望 1.0)")
    print(f"  Case 2 (荒谬断言Golden挂): {rewards[1]} (期望 0.0)")
    print(f"  Case 3 (常规测试未能捕获): {rewards[2]} (期望 0.0)")

    assert rewards == [1.0, 0.0, 0.0], f"沙箱断言失败: {rewards}"
    print("\n✅ tasks/bug_hunter.py 双向沙盒执行器全量单元测试通过！")
    print("=" * 60)
