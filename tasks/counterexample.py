"""
tasks/counterexample.py - Nano Counterexample (程序差分反例挖掘与边界测试任务)

核心任务:
  给定两个行为极其相似的短函数 f_a(x) 与 f_b(x)，寻找一个整数输入 x ∈ [min_val, max_val]，
  使得两者的输出结果不同: f_a(x) != f_b(x)。

设计特点:
  1. 高效短序列生成 (8 ~ 32 tokens): 策略模型聚焦于预测目标输入 (如 <input>7</input>)。
  2. 确定性客观奖励: 判题器通过直接执行 f_a(x) 与 f_b(x) 比对输出，无需模型推导预期值，杜绝幻觉与奖励黑客。
  3. 反例稀疏度可控: 离线穷举定义域，精确量化反例点集合与稀疏度 (Sparsity)，构建平滑难度阶梯。
  4. 对标基线支持: 内置随机抽样 (Random Search) 与常量边界探测 (Constant Prober) 启发式基线。
"""

import ast
import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from tasks.base import BaseTask


DEFAULT_DOMAIN = (0, 50)


SYSTEM_PROMPT = (
    "You are an expert software test engineer.\n"
    "Given two Python functions that are supposed to behave identically, "
    "find an integer input x in the specified domain where their return values differ: f_a(x) != f_b(x)."
)

SYSTEM_PROMPT_2D = (
    "You are an expert software test engineer.\n"
    "Given two Python functions that are supposed to behave identically, "
    "find an integer pair (x, y) in the specified domain where their return values differ: f_a(x, y) != f_b(x, y)."
)


def extract_input_integer(completion_text: str, domain: Tuple[int, int] = DEFAULT_DOMAIN) -> Tuple[Optional[int], str, str]:
    """
    从模型输出中抽取单整数输入 (用于 1D 题目)。
    返回: (value, category, reason)
      - category: "valid_extracted", "out_of_domain", "format_malformed"
    """
    low, high = domain

    # 1. 优先匹配 <input>...</input> 标签
    match = re.search(r"<input>\s*([+-]?\d+)\s*</input>", completion_text, flags=re.IGNORECASE)
    if not match:
        # 2. 匹配可能未闭合的 <input> 标签
        match = re.search(r"<input>\s*([+-]?\d+)", completion_text, flags=re.IGNORECASE)
    if not match:
        # 3. 匹配 x = 7 或 input = 7 形式
        match = re.search(r"(?:x|input|ans)\s*=\s*([+-]?\d+)", completion_text, flags=re.IGNORECASE)
    if not match:
        # 4. 如果文本较短，直接匹配出现的第一个独立整数
        stripped = completion_text.strip()
        int_match = re.search(r"\b([+-]?\d+)\b", stripped)
        if int_match:
            match = int_match

    if not match:
        return None, "format_malformed", "No integer found in completion"

    try:
        val = int(match.group(1))
    except ValueError:
        return None, "format_malformed", f"Failed to parse '{match.group(1)}' as integer"

    if val < low or val > high:
        return val, "out_of_domain", f"Integer {val} is outside domain [{low}, {high}]"

    return val, "valid_extracted", "Valid integer within domain"


def _validate_domain_2d(vx: int, vy: int, dom_x: Tuple[int, int], dom_y: Tuple[int, int]) -> Tuple[Tuple[int, int], str, str]:
    low_x, high_x = dom_x
    low_y, high_y = dom_y
    if vx < low_x or vx > high_x or vy < low_y or vy > high_y:
        return (vx, vy), "out_of_domain", f"Pair ({vx}, {vy}) is outside domain [{low_x}, {high_x}] x [{low_y}, {high_y}]"
    return (vx, vy), "valid_extracted", "Valid integer pair within domain"


def extract_input_value(completion_text: str, domain: Any, input_dim: int = 1) -> Tuple[Optional[Any], str, str]:
    """
    统一输入解析器: 支持 1D 标量整数与 2D 二元整数对抽取。
    """
    if input_dim == 1:
        dom_1d = domain[0] if (isinstance(domain, list) and len(domain) > 0 and isinstance(domain[0], (list, tuple))) else domain
        return extract_input_integer(completion_text, tuple(dom_1d))

    # 2D 输入解析
    if isinstance(domain, (list, tuple)) and len(domain) == 2 and isinstance(domain[0], (list, tuple)):
        dom_x, dom_y = domain[0], domain[1]
    else:
        dom_x, dom_y = DEFAULT_DOMAIN, DEFAULT_DOMAIN

    tag_match = re.search(r"<input>(.*?)(?:</input>|$)", completion_text, flags=re.IGNORECASE | re.DOTALL)
    search_str = tag_match.group(1) if tag_match else completion_text

    named = re.search(r"x\s*=\s*([+-]?\d+)[^\d+-]+y\s*=\s*([+-]?\d+)", search_str, flags=re.IGNORECASE)
    if named:
        vx, vy = int(named.group(1)), int(named.group(2))
        return _validate_domain_2d(vx, vy, dom_x, dom_y)

    named_rev = re.search(r"y\s*=\s*([+-]?\d+)[^\d+-]+x\s*=\s*([+-]?\d+)", search_str, flags=re.IGNORECASE)
    if named_rev:
        vy, vx = int(named_rev.group(1)), int(named_rev.group(2))
        return _validate_domain_2d(vx, vy, dom_x, dom_y)

    all_ints = re.findall(r"[+-]?\d+", search_str)
    if len(all_ints) >= 2:
        vx, vy = int(all_ints[0]), int(all_ints[1])
        return _validate_domain_2d(vx, vy, dom_x, dom_y)

    return None, "format_malformed", "Could not extract two integers for 2D input"


def evaluate_functions_on_input(f_a_src: str, f_b_src: str, inp: Any) -> Tuple[bool, Any, Any, Optional[str]]:
    """
    安全执行 f_a 与 f_b，判断 f_a(inp) != f_b(inp)。
    支持一元输入 x 或多元输入 (x, y)。
    返回: (differ, res_a, res_b, error_msg)
    """
    safe_builtins = {
        "abs": abs, "min": min, "max": max, "len": len,
        "range": range, "int": int, "bool": bool
    }
    ns_a = {"__builtins__": safe_builtins}
    ns_b = {"__builtins__": safe_builtins}

    try:
        exec(f_a_src, ns_a)
        exec(f_b_src, ns_b)
        if isinstance(inp, (list, tuple)):
            res_a = ns_a["f_a"](*inp)
            res_b = ns_b["f_b"](*inp)
        else:
            res_a = ns_a["f_a"](inp)
            res_b = ns_b["f_b"](inp)
        return (res_a != res_b), res_a, res_b, None
    except Exception as e:
        return False, None, None, str(e)


# =========================================================================
# 程序合成生成器 (Synthetic Problem Generators)
# =========================================================================

def _make_problem_item(
    prob_id: str,
    family: str,
    level: str,
    f_a_src: str,
    f_b_src: str,
    spec: str,
    domain: Tuple[int, int] = DEFAULT_DOMAIN
) -> Optional[Dict[str, Any]]:
    """验证并封装一道反例题目，预计算其全部有效反例与稀疏度"""
    low, high = domain
    counterexamples = []
    domain_size = high - low + 1

    for x in range(low, high + 1):
        differs, _, _, err = evaluate_functions_on_input(f_a_src, f_b_src, x)
        if err:
            return None
        if differs:
            counterexamples.append(x)

    if not counterexamples:
        return None  # 无反例，题目无效
    if len(counterexamples) == domain_size:
        return None  # 全反例，过于简单无意义

    sparsity = len(counterexamples) / domain_size

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Domain of x: [{low}, {high}]\n"
        f"Description: {spec}\n\n"
        f"Function A:\n```python\n{f_a_src}\n```\n\n"
        f"Function B:\n```python\n{f_b_src}\n```\n\n"
        f"Output format: <input>INTEGER</input>\n\n"
        f"Assistant: <input>"
    )


    return {
        "id": prob_id,
        "problem_id": prob_id,
        "family": family,
        "level": level,
        "spec": spec,
        "domain": [low, high],
        "func_a_src": f_a_src,
        "func_b_src": f_b_src,
        "counterexamples": counterexamples,
        "num_counterexamples": len(counterexamples),
        "sparsity": round(sparsity, 4),
        "prompt": prompt,
        "ground_truth": counterexamples[0]
    }


def generate_comparison_problems(rng: random.Random, count: int, domain: Tuple[int, int] = DEFAULT_DOMAIN) -> List[Dict[str, Any]]:
    """生成比较操作符 Off-by-one 差异题目 (例如 > vs >=, < vs <=)"""
    problems = []
    low, high = domain
    for i in range(count):
        c = rng.randint(low + 2, high - 2)
        op_type = rng.choice(["gt_gte", "lt_lte", "gte_gt_next", "lte_lt_prev"])

        if op_type == "gt_gte":
            f_a = f"def f_a(x: int) -> bool:\n    return x > {c}"
            f_b = f"def f_b(x: int) -> bool:\n    return x >= {c}"
            spec = f"Check if x is strictly greater than {c} vs at least {c}."
        elif op_type == "lt_lte":
            f_a = f"def f_a(x: int) -> bool:\n    return x < {c}"
            f_b = f"def f_b(x: int) -> bool:\n    return x <= {c}"
            spec = f"Check if x is strictly less than {c} vs at most {c}."
        elif op_type == "gte_gt_next":
            f_a = f"def f_a(x: int) -> bool:\n    return x >= {c}"
            f_b = f"def f_b(x: int) -> bool:\n    return x > {c}"
            spec = f"Boundary check at threshold {c}."
        else:
            f_a = f"def f_a(x: int) -> bool:\n    return x <= {c}"
            f_b = f"def f_b(x: int) -> bool:\n    return x < {c}"
            spec = f"Upper bound comparison at {c}."

        prob = _make_problem_item(f"cmp_{op_type}_{i}", "comparison", "level_1", f_a, f_b, spec, domain)
        if prob:
            problems.append(prob)
    return problems


def generate_interval_problems(rng: random.Random, count: int, domain: Tuple[int, int] = DEFAULT_DOMAIN) -> List[Dict[str, Any]]:
    """生成双端区间边界题目 (例如 a <= x <= b vs a < x <= b)"""
    problems = []
    low, high = domain
    for i in range(count):
        a = rng.randint(low + 5, high - 15)
        span = rng.randint(5, 15)
        b = a + span
        var_type = rng.choice(["left_open", "right_open", "both_open", "extended_right"])

        if var_type == "left_open":
            f_a = f"def f_a(x: int) -> bool:\n    return {a} <= x <= {b}"
            f_b = f"def f_b(x: int) -> bool:\n    return {a} < x <= {b}"
            spec = f"Interval inclusion: [{a}, {b}] vs ({a}, {b}]."
        elif var_type == "right_open":
            f_a = f"def f_a(x: int) -> bool:\n    return {a} <= x <= {b}"
            f_b = f"def f_b(x: int) -> bool:\n    return {a} <= x < {b}"
            spec = f"Interval inclusion: [{a}, {b}] vs [{a}, {b})."
        elif var_type == "both_open":
            f_a = f"def f_a(x: int) -> bool:\n    return {a} < x < {b}"
            f_b = f"def f_b(x: int) -> bool:\n    return {a} <= x <= {b}"
            spec = f"Strict interior ({a}, {b}) vs closed [{a}, {b}]."
        else:
            f_a = f"def f_a(x: int) -> bool:\n    return {a} <= x <= {b}"
            f_b = f"def f_b(x: int) -> bool:\n    return {a} <= x <= {b + 1}"
            spec = f"Interval endpoint comparison [{a}, {b}] vs [{a}, {b+1}]."

        prob = _make_problem_item(f"interval_{var_type}_{i}", "interval", "level_2", f_a, f_b, spec, domain)
        if prob:
            problems.append(prob)
    return problems


def generate_clamping_and_abs_problems(rng: random.Random, count: int, domain: Tuple[int, int] = DEFAULT_DOMAIN) -> List[Dict[str, Any]]:
    """生成 min/max 裁剪与绝对值距离题目"""
    problems = []
    low, high = domain
    for i in range(count):
        c = rng.randint(low + 5, high - 5)
        r = rng.randint(2, 8)
        kind = rng.choice(["min_clip", "max_clip", "abs_dist_strict", "abs_dist_diff"])

        if kind == "min_clip":
            f_a = f"def f_a(x: int) -> int:\n    return min(x, {c})"
            f_b = f"def f_b(x: int) -> int:\n    return min(x, {c + 1})"
            spec = f"Upper bound clamp using min(x, {c}) vs min(x, {c+1})."
        elif kind == "max_clip":
            f_a = f"def f_a(x: int) -> int:\n    return max(x, {c})"
            f_b = f"def f_b(x: int) -> int:\n    return max(x, {c - 1})"
            spec = f"Lower bound clamp using max(x, {c}) vs max(x, {c-1})."
        elif kind == "abs_dist_strict":
            f_a = f"def f_a(x: int) -> bool:\n    return abs(x - {c}) <= {r}"
            f_b = f"def f_b(x: int) -> bool:\n    return abs(x - {c}) < {r}"
            spec = f"Distance from center {c} <= {r} vs < {r}."
        else:
            f_a = f"def f_a(x: int) -> int:\n    return abs(x) if x != {c} else {c + 1}"
            f_b = f"def f_b(x: int) -> int:\n    return abs(x)"
            spec = f"Special point perturbation at {c}."

        prob = _make_problem_item(f"clamp_abs_{kind}_{i}", "clamping_abs", "level_2", f_a, f_b, spec, domain)
        if prob:
            problems.append(prob)
    return problems


def generate_piecewise_and_arithmetic_problems(rng: random.Random, count: int, domain: Tuple[int, int] = DEFAULT_DOMAIN) -> List[Dict[str, Any]]:
    """生成分段分支与算术/取模题目"""
    problems = []
    low, high = domain
    for i in range(count):
        c = rng.randint(low + 5, high - 5)
        mod_k = rng.choice([2, 3, 4, 5])
        kind = rng.choice(["sign_branch", "scale_branch", "mod_shift"])

        if kind == "sign_branch":
            f_a = f"def f_a(x: int) -> int:\n    return 1 if x > {c} else (-1 if x < {c} else 0)"
            f_b = f"def f_b(x: int) -> int:\n    return 1 if x >= {c} else -1"
            spec = f"Three-way sign split around {c} vs two-way split."
        elif kind == "scale_branch":
            f_a = f"def f_a(x: int) -> int:\n    return x * 2 if x > {c} else x"
            f_b = f"def f_b(x: int) -> int:\n    return x * 2 if x >= {c} else x"
            spec = f"Multiplier activation at threshold {c}."
        else:
            f_a = f"def f_a(x: int) -> bool:\n    return (x % {mod_k}) == 0"
            f_b = f"def f_b(x: int) -> bool:\n    return ((x + 1) % {mod_k}) == 0"
            spec = f"Divisibility by {mod_k} with phase shift."

        prob = _make_problem_item(f"piecewise_{kind}_{i}", "piecewise_arithmetic", "level_3", f_a, f_b, spec, domain)
        if prob:
            problems.append(prob)
    return problems


# =========================================================================
# Level 4: 二元变量与复杂几何边界生成器 (Multi-Variable 2D Generators)
# =========================================================================

def _make_problem_item_2d(
    prob_id: str,
    family: str,
    level: str,
    f_a_src: str,
    f_b_src: str,
    spec: str,
    domain_x: Tuple[int, int] = DEFAULT_DOMAIN,
    domain_y: Tuple[int, int] = DEFAULT_DOMAIN
) -> Optional[Dict[str, Any]]:
    """验证并封装一道二元输入 (x, y) 反例题目，预计算其在二维离散网格中的反例点集合"""
    low_x, high_x = domain_x
    low_y, high_y = domain_y
    counterexamples = []
    total_grid_points = (high_x - low_x + 1) * (high_y - low_y + 1)

    for x in range(low_x, high_x + 1):
        for y in range(low_y, high_y + 1):
            differs, _, _, err = evaluate_functions_on_input(f_a_src, f_b_src, (x, y))
            if err:
                return None
            if differs:
                counterexamples.append([x, y])

    if not counterexamples:
        return None
    if len(counterexamples) == total_grid_points:
        return None

    sparsity = len(counterexamples) / total_grid_points

    prompt = (
        f"{SYSTEM_PROMPT_2D}\n\n"
        f"Domain of x: [{low_x}, {high_x}]\n"
        f"Domain of y: [{low_y}, {high_y}]\n"
        f"Description: {spec}\n\n"
        f"Function A:\n```python\n{f_a_src}\n```\n\n"
        f"Function B:\n```python\n{f_b_src}\n```\n\n"
        f"Output format: <input>x, y</input>\n\n"
        f"Assistant: <input>"
    )

    return {
        "id": prob_id,
        "problem_id": prob_id,
        "family": family,
        "level": level,
        "input_dim": 2,
        "spec": spec,
        "domain": [low_x, high_x],
        "domain_x": [low_x, high_x],
        "domain_y": [low_y, high_y],
        "func_a_src": f_a_src,
        "func_b_src": f_b_src,
        "counterexamples": counterexamples,
        "num_counterexamples": len(counterexamples),
        "sparsity": round(sparsity, 4),
        "prompt": prompt,
        "ground_truth": counterexamples[0]
    }


def generate_2d_manhattan_problems(rng: random.Random, count: int, domain: Tuple[int, int] = DEFAULT_DOMAIN) -> List[Dict[str, Any]]:
    """生成 2D 曼哈顿距离与 L1 范数几何边界题目"""
    problems = []
    low, high = domain
    for i in range(count):
        cx = rng.randint(low + 10, high - 10)
        cy = rng.randint(low + 10, high - 10)
        r = rng.randint(3, 8)
        op = rng.choice(["lte_lt", "gte_gt"])

        if op == "lte_lt":
            f_a = f"def f_a(x: int, y: int) -> bool:\n    return abs(x - {cx}) + abs(y - {cy}) <= {r}"
            f_b = f"def f_b(x: int, y: int) -> bool:\n    return abs(x - {cx}) + abs(y - {cy}) < {r}"
            spec = f"Manhattan distance from center ({cx}, {cy}) <= {r} vs strictly < {r}."
        else:
            f_a = f"def f_a(x: int, y: int) -> bool:\n    return abs(x - {cx}) + abs(y - {cy}) >= {r}"
            f_b = f"def f_b(x: int, y: int) -> bool:\n    return abs(x - {cx}) + abs(y - {cy}) > {r}"
            spec = f"Manhattan distance exterior from ({cx}, {cy}) >= {r} vs strictly > {r}."

        prob = _make_problem_item_2d(f"manhattan_2d_{i}", "manhattan_2d", "level_4", f_a, f_b, spec, domain, domain)
        if prob:
            problems.append(prob)
    return problems


def generate_2d_linear_problems(rng: random.Random, count: int, domain: Tuple[int, int] = DEFAULT_DOMAIN) -> List[Dict[str, Any]]:
    """生成 2D 线性规划与斜率分离超平面题目 (如 2x + y >= C)"""
    problems = []
    low, high = domain
    for i in range(count):
        a_coeff = rng.choice([1, 2])
        b_coeff = rng.choice([1, 2])
        c = rng.randint(low * (a_coeff + b_coeff) + 20, high * (a_coeff + b_coeff) - 20)
        op = rng.choice(["gte_gt", "lte_lt"])

        if op == "gte_gt":
            f_a = f"def f_a(x: int, y: int) -> bool:\n    return {a_coeff} * x + {b_coeff} * y >= {c}"
            f_b = f"def f_b(x: int, y: int) -> bool:\n    return {a_coeff} * x + {b_coeff} * y > {c}"
            spec = f"Linear inequality boundary: {a_coeff}*x + {b_coeff}*y >= {c} vs > {c}."
        else:
            f_a = f"def f_a(x: int, y: int) -> bool:\n    return {a_coeff} * x + {b_coeff} * y <= {c}"
            f_b = f"def f_b(x: int, y: int) -> bool:\n    return {a_coeff} * x + {b_coeff} * y < {c}"
            spec = f"Linear inequality boundary: {a_coeff}*x + {b_coeff}*y <= {c} vs < {c}."

        prob = _make_problem_item_2d(f"linear_2d_{i}", "linear_2d", "level_4", f_a, f_b, spec, domain, domain)
        if prob:
            problems.append(prob)
    return problems


def generate_2d_box_problems(rng: random.Random, count: int, domain: Tuple[int, int] = DEFAULT_DOMAIN) -> List[Dict[str, Any]]:
    """生成 2D 矩形包围盒边界题目"""
    problems = []
    low, high = domain
    for i in range(count):
        x1 = rng.randint(low + 5, high - 20)
        x2 = x1 + rng.randint(8, 15)
        y1 = rng.randint(low + 5, high - 20)
        y2 = y1 + rng.randint(8, 15)

        f_a = f"def f_a(x: int, y: int) -> bool:\n    return {x1} <= x <= {x2} and {y1} <= y <= {y2}"
        f_b = f"def f_b(x: int, y: int) -> bool:\n    return {x1} <= x < {x2} and {y1} <= y <= {y2}"
        spec = f"2D box boundary inclusion: [{x1}, {x2}] x [{y1}, {y2}] vs right-open on x."

        prob = _make_problem_item_2d(f"box_2d_{i}", "box_2d", "level_4", f_a, f_b, spec, domain, domain)
        if prob:
            problems.append(prob)
    return problems


def generate_2d_piecewise_problems(rng: random.Random, count: int, domain: Tuple[int, int] = DEFAULT_DOMAIN) -> List[Dict[str, Any]]:
    """生成 2D 对角线条件与坐标复合题目"""
    problems = []
    low, high = domain
    for i in range(count):
        c = rng.randint(low + 15, high + 15)
        f_a = f"def f_a(x: int, y: int) -> int:\n    return max(x, y) if x + y <= {c} else min(x, y)"
        f_b = f"def f_b(x: int, y: int) -> int:\n    return max(x, y) if x + y < {c} else min(x, y)"
        spec = f"Piecewise activation at diagonal coordinate sum x + y == {c}."

        prob = _make_problem_item_2d(f"piecewise_2d_{i}", "piecewise_2d", "level_4", f_a, f_b, spec, domain, domain)
        if prob:
            problems.append(prob)
    return problems


def generate_benchmark_suite(
    num_train: int = 60,
    num_val: int = 20,
    num_test: int = 20,
    domain: Tuple[int, int] = DEFAULT_DOMAIN,
    seed: int = 42,
    include_2d: bool = False
) -> Dict[str, List[Dict[str, Any]]]:
    """
    生成完整的 Train / Val / Test 三相基准数据集。
    确保各集合结构平衡，支持可选 include_2d 生成 1D+2D 混合多维基准。
    """
    rng = random.Random(seed)
    total_needed = num_train + num_val + num_test

    if include_2d:
        per_fam = (total_needed // 8) + 10
        p_cmp = generate_comparison_problems(rng, per_fam, domain)
        p_int = generate_interval_problems(rng, per_fam, domain)
        p_clp = generate_clamping_and_abs_problems(rng, per_fam, domain)
        p_pcw = generate_piecewise_and_arithmetic_problems(rng, per_fam, domain)
        p_man = generate_2d_manhattan_problems(rng, per_fam, domain)
        p_lin = generate_2d_linear_problems(rng, per_fam, domain)
        p_box = generate_2d_box_problems(rng, per_fam, domain)
        p_p2d = generate_2d_piecewise_problems(rng, per_fam, domain)
        all_problems = p_cmp + p_int + p_clp + p_pcw + p_man + p_lin + p_box + p_p2d
    else:
        per_fam = (total_needed // 4) + 15
        p_cmp = generate_comparison_problems(rng, per_fam, domain)
        p_int = generate_interval_problems(rng, per_fam, domain)
        p_clp = generate_clamping_and_abs_problems(rng, per_fam, domain)
        p_pcw = generate_piecewise_and_arithmetic_problems(rng, per_fam, domain)
        all_problems = p_cmp + p_int + p_clp + p_pcw

    rng.shuffle(all_problems)

    # 去重
    seen_prompts = set()
    unique_problems = []
    for p in all_problems:
        sig = (p["func_a_src"], p["func_b_src"])
        if sig not in seen_prompts:
            seen_prompts.add(sig)
            unique_problems.append(p)

    if len(unique_problems) < total_needed:
        raise ValueError(f"Generated {len(unique_problems)} problems, but need {total_needed}")

    train = unique_problems[:num_train]
    val = unique_problems[num_train : num_train + num_val]
    test = unique_problems[num_train + num_val : num_train + num_val + num_test]

    return {"train": train, "val": val, "test": test}


# =========================================================================
# 启发式基线 (Heuristic Baselines for Benchmarking)
# =========================================================================

class HeuristicSolver:
    """传统启发式算法基线，用于对比模型是否真正学到了代码边界"""

    @staticmethod
    def random_search(problem: Dict[str, Any], budget: int, rng: Optional[random.Random] = None) -> List[Any]:
        """纯随机采样 budget 次输入 (支持 1D 整数与 2D 整数对)"""
        rng = rng or random.Random()
        if problem.get("input_dim", 1) == 2:
            low_x, high_x = problem.get("domain_x", (0, 50))
            low_y, high_y = problem.get("domain_y", (0, 50))
            return [[rng.randint(low_x, high_x), rng.randint(low_y, high_y)] for _ in range(budget)]
        low, high = problem["domain"]
        return [rng.randint(low, high) for _ in range(budget)]

    @staticmethod
    def constant_and_neighbors(problem: Dict[str, Any], budget: int) -> List[int]:
        """
        常量邻域启发式:
        提取代码中出现的整数常量 c，优先探测 {c, c-1, c+1, 0, 1, -1}。
        """
        src = problem["func_a_src"] + "\n" + problem["func_b_src"]
        literals = set()
        try:
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, int):
                    literals.add(node.value)
        except Exception:
            pass

        candidates = []
        for c in sorted(literals):
            candidates.extend([c, c - 1, c + 1])
        candidates.extend([0, 1, -1])

        # 去重且保留顺序
        seen = set()
        ordered = []
        low, high = problem["domain"]
        for x in candidates:
            if x not in seen and low <= x <= high:
                seen.add(x)
                ordered.append(x)

        # 若不足 budget，用邻近点填充
        curr = 2
        while len(ordered) < budget and curr <= high:
            if curr not in seen:
                ordered.append(curr)
                seen.add(curr)
            if -curr not in seen and -curr >= low:
                ordered.append(-curr)
                seen.add(-curr)
            curr += 1

        return ordered[:budget]


# =========================================================================
# Task 任务类封装 (对齐 BaseTask 接口)
# =========================================================================

class CounterexampleTask(BaseTask):
    """
    Nano Counterexample 差分反例挖掘与边界测试任务。
    可作为标准强化学习环境嵌入 nano-GRPO 训练管线。
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        num_train: int = 60,
        num_val: int = 20,
        num_test: int = 20,
        domain: Tuple[int, int] = DEFAULT_DOMAIN,
        seed: int = 42
    ):
        self.domain = domain
        self.sample_idx = 0

        default_dataset_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "counterexample_dataset.json"
        )

        if data_path and os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif os.path.exists(default_dataset_path):
            with open(default_dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            # 自动生成三相划分基准并持久化
            data = generate_benchmark_suite(num_train, num_val, num_test, domain, seed)
            os.makedirs(os.path.dirname(default_dataset_path), exist_ok=True)
            with open(default_dataset_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        self.train_problems = data["train"]
        self.val_problems = data.get("val", [])
        self.test_problems = data.get("test", [])

    def __len__(self) -> int:
        return len(self.train_problems)

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
        对单个生成文本进行细粒度差分判题。
        类别包括:
          - 'success': 1.0 (成功找到有效反例，使 f_a(x) != f_b(x))
          - 'missed_bug': 0.0 (输入有效，但 f_a(x) == f_b(x))
          - 'out_of_domain': 0.0 (输入的整数超出题目指定定义域)
          - 'format_malformed': 0.0 (未提取到整数)
          - 'truncated': 0.0 (达到最大 token 且未生成有效输入)
        """
        input_dim = meta.get("input_dim", 1)
        if input_dim == 2 and "domain_x" in meta and "domain_y" in meta:
            domain = [tuple(meta["domain_x"]), tuple(meta["domain_y"])]
        else:
            domain = tuple(meta.get("domain", self.domain))
        is_truncated = meta.get("is_truncated", False)

        val, cat, reason = extract_input_value(comp, domain, input_dim=input_dim)

        if val is None:
            if is_truncated:
                return {
                    "reward": 0.0,
                    "category": "truncated",
                    "reason": "Generation hit token limit before outputting valid input",
                    "extracted_value": None,
                    "format_compliant": False
                }
            return {
                "reward": 0.0,
                "category": "format_malformed",
                "reason": reason,
                "extracted_value": None,
                "format_compliant": False
            }

        if cat == "out_of_domain":
            return {
                "reward": 0.0,
                "category": "precondition_violation",
                "reason": reason,
                "extracted_value": val,
                "format_compliant": True
            }

        # 检查是否为有效反例 (优先使用预计算的 counterexamples 集合以实现 O(1) 瞬时结算)
        if "counterexamples" in meta:
            ce_set = meta["counterexamples"]
            if isinstance(val, (tuple, list)):
                is_counterexample = (list(val) in ce_set) or (tuple(val) in ce_set)
            else:
                is_counterexample = (val in ce_set)
            differs = is_counterexample
            err = None
        else:
            differs, _, _, err = evaluate_functions_on_input(meta["func_a_src"], meta["func_b_src"], val)

        if err:
            return {
                "reward": 0.0,
                "category": "execution_error",
                "reason": f"Execution error: {err}",
                "extracted_value": val,
                "format_compliant": True
            }

        if differs:
            return {
                "reward": 1.0,
                "category": "success",
                "reason": f"Counterexample found: f_a({val}) != f_b({val})",
                "extracted_value": val,
                "format_compliant": True
            }
        else:
            return {
                "reward": 0.0,
                "category": "missed_bug",
                "reason": f"f_a({val}) == f_b({val}) (failed to expose difference)",
                "extracted_value": val,
                "format_compliant": True
            }

    def compute_rewards_with_details(
        self,
        prompts: List[str],
        completions: List[str],
        metadata: List[Dict[str, Any]]
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        details = [self.evaluate_single_completion(c, m) for c, m in zip(completions, metadata)]
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
    print(">>> 正在自测 tasks/counterexample.py 数据生成器与判题器...")
    task = CounterexampleTask()
    print(f"训练集: {len(task)} 道 | 验证集: {len(task.get_val_dataset())} 道 | 测试集: {len(task.get_test_dataset())} 道")

    sample = task.train_problems[0]
    print(f"\n[题目 ID]: {sample['id']}")
    print(f"[规格说明]: {sample['spec']}")
    print(f"[定义域]: {sample['domain']}")
    print(f"[有效反例数量]: {sample['num_counterexamples']} 个 (稀疏度: {sample['sparsity']:.2%})")
    print(f"[部分反例示例]: {sample['counterexamples'][:5]}")

    # 测试测试用例: 正确反例 vs 错误常规输入 vs 格式错误
    comp_correct = f"<input>{sample['counterexamples'][0]}</input>"
    # 找一个不是反例的输入
    low, high = sample["domain"]
    non_ce = next(x for x in range(low, high + 1) if x not in sample["counterexamples"])
    comp_wrong = f"<input>{non_ce}</input>"
    comp_invalid = "I think the answer is none."

    rewards, details = task.compute_rewards_with_details(
        [""] * 3, [comp_correct, comp_wrong, comp_invalid], [sample] * 3
    )

    print("\n[判题结果验证]:")
    print(f"  正确反例: reward={rewards[0]}, category={details[0]['category']}")
    assert rewards == [1.0, 0.0, 0.0], f"判题逻辑异常: {rewards}"

    # 2D 题目生成与判题自测
    print("\n>>> 正在验证 Level 4 (2D 二元几何边界) 生成与判题...")
    rng = random.Random(42)
    p2d = generate_2d_manhattan_problems(rng, 1)[0]
    ce2d = p2d["counterexamples"][0]
    comp_2d_correct = f"<input>{ce2d[0]}, {ce2d[1]}</input>"
    comp_2d_named = f"<input>x={ce2d[0]}, y={ce2d[1]}</input>"
    comp_2d_wrong = f"<input>0, 0</input>"
    r_2d, det_2d = task.compute_rewards_with_details(
        [""] * 3, [comp_2d_correct, comp_2d_named, comp_2d_wrong], [p2d] * 3
    )
    print(f"  2D 坐标逗号分隔反例: reward={r_2d[0]}, category={det_2d[0]['category']}")
    print(f"  2D 命名变量反例:     reward={r_2d[1]}, category={det_2d[1]['category']}")
    print(f"  2D 错误非反例输入:   reward={r_2d[2]}, category={det_2d[2]['category']}")
    assert r_2d == [1.0, 1.0, 0.0], f"2D 判题逻辑异常: {r_2d}"

    print("\n✅ tasks/counterexample.py (1D + 2D) 自测全部通过！")
    print("=" * 60)
