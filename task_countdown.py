import ast
from collections import Counter
import itertools
import operator
import random
import re
from typing import Dict, List, Tuple, Optional

SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."""


OPS = [
    ('+', operator.add),
    ('-', operator.sub),
    ('*', operator.mul),
    ('/', lambda a, b: a / b if b != 0 else None)
]


def _fmt(val) -> str:
    return str(int(val)) if abs(val - int(val)) < 1e-5 else f"{val:.2f}"


def find_countdown_solution(numbers: List[int], target: int) -> Optional[Tuple[str, List[str]]]:
    """穷举搜索精确解，返回 (算式表达式, 分步运算轨迹)"""
    n = len(numbers)
    if n == 2:
        for perm in itertools.permutations(numbers):
            a, b = perm
            for op_s, op in OPS:
                try:
                    r = op(a, b)
                    if r is not None and abs(r - target) < 1e-5:
                        expr = f"({a} {op_s} {b})"
                        steps = [f"{a} {op_s} {b} = {_fmt(r)}"]
                        return expr, steps
                except (ZeroDivisionError, OverflowError):
                    pass

    elif n == 3:
        for perm in itertools.permutations(numbers):
            a, b, c = perm
            for (op1_s, op1), (op2_s, op2) in itertools.product(OPS, repeat=2):
                # 结构 1: (a op1 b) op2 c
                try:
                    r1 = op1(a, b)
                    if r1 is not None:
                        r2 = op2(r1, c)
                        if r2 is not None and abs(r2 - target) < 1e-5:
                            expr = f"(({a} {op1_s} {b}) {op2_s} {c})"
                            steps = [
                                f"{a} {op1_s} {b} = {_fmt(r1)}",
                                f"{_fmt(r1)} {op2_s} {c} = {_fmt(r2)}"
                            ]
                            return expr, steps
                except (ZeroDivisionError, OverflowError):
                    pass

                # 结构 2: a op1 (b op2 c)
                try:
                    r1 = op2(b, c)
                    if r1 is not None:
                        r2 = op1(a, r1)
                        if r2 is not None and abs(r2 - target) < 1e-5:
                            expr = f"({a} {op1_s} ({b} {op2_s} {c}))"
                            steps = [
                                f"{b} {op2_s} {c} = {_fmt(r1)}",
                                f"{a} {op1_s} {_fmt(r1)} = {_fmt(r2)}"
                            ]
                            return expr, steps
                except (ZeroDivisionError, OverflowError):
                    pass

    elif n == 4:
        for perm in itertools.permutations(numbers):
            a, b, c, d = perm
            for (op1_s, op1), (op2_s, op2), (op3_s, op3) in itertools.product(OPS, repeat=3):
                # 结构 1: ((a op1 b) op2 c) op3 d
                try:
                    r1 = op1(a, b)
                    if r1 is not None:
                        r2 = op2(r1, c)
                        if r2 is not None:
                            r3 = op3(r2, d)
                            if r3 is not None and abs(r3 - target) < 1e-5:
                                expr = f"((({a} {op1_s} {b}) {op2_s} {c}) {op3_s} {d})"
                                steps = [
                                    f"{a} {op1_s} {b} = {_fmt(r1)}",
                                    f"{_fmt(r1)} {op2_s} {c} = {_fmt(r2)}",
                                    f"{_fmt(r2)} {op3_s} {d} = {_fmt(r3)}"
                                ]
                                return expr, steps
                except (ZeroDivisionError, OverflowError):
                    pass

                # 结构 2: (a op1 (b op2 c)) op3 d
                try:
                    r1 = op2(b, c)
                    if r1 is not None:
                        r2 = op1(a, r1)
                        if r2 is not None:
                            r3 = op3(r2, d)
                            if r3 is not None and abs(r3 - target) < 1e-5:
                                expr = f"(({a} {op1_s} ({b} {op2_s} {c})) {op3_s} {d})"
                                steps = [
                                    f"{b} {op2_s} {c} = {_fmt(r1)}",
                                    f"{a} {op1_s} {_fmt(r1)} = {_fmt(r2)}",
                                    f"{_fmt(r2)} {op3_s} {d} = {_fmt(r3)}"
                                ]
                                return expr, steps
                except (ZeroDivisionError, OverflowError):
                    pass

                # 结构 3: (a op1 b) op2 (c op3 d)
                try:
                    r1 = op1(a, b)
                    r2 = op3(c, d)
                    if r1 is not None and r2 is not None:
                        r3 = op2(r1, r2)
                        if r3 is not None and abs(r3 - target) < 1e-5:
                            expr = f"(({a} {op1_s} {b}) {op2_s} ({c} {op3_s} {d}))"
                            steps = [
                                f"{a} {op1_s} {b} = {_fmt(r1)}",
                                f"{c} {op3_s} {d} = {_fmt(r2)}",
                                f"{_fmt(r1)} {op2_s} {_fmt(r2)} = {_fmt(r3)}"
                            ]
                            return expr, steps
                except (ZeroDivisionError, OverflowError):
                    pass

                # 结构 4: a op1 ((b op2 c) op3 d)
                try:
                    r1 = op2(b, c)
                    if r1 is not None:
                        r2 = op3(r1, d)
                        if r2 is not None:
                            r3 = op1(a, r2)
                            if r3 is not None and abs(r3 - target) < 1e-5:
                                expr = f"({a} {op1_s} (({b} {op2_s} {c}) {op3_s} {d}))"
                                steps = [
                                    f"{b} {op2_s} {c} = {_fmt(r1)}",
                                    f"{_fmt(r1)} {op3_s} {d} = {_fmt(r2)}",
                                    f"{a} {op1_s} {_fmt(r2)} = {_fmt(r3)}"
                                ]
                                return expr, steps
                except (ZeroDivisionError, OverflowError):
                    pass

                # 结构 5: a op1 (b op2 (c op3 d))
                try:
                    r1 = op3(c, d)
                    if r1 is not None:
                        r2 = op2(b, r1)
                        if r2 is not None:
                            r3 = op1(a, r2)
                            if r3 is not None and abs(r3 - target) < 1e-5:
                                expr = f"({a} {op1_s} ({b} {op2_s} ({c} {op3_s} {d})))"
                                steps = [
                                    f"{c} {op3_s} {d} = {_fmt(r1)}",
                                    f"{b} {op2_s} {_fmt(r1)} = {_fmt(r2)}",
                                    f"{a} {op1_s} {_fmt(r2)} = {_fmt(r3)}"
                                ]
                                return expr, steps
                except (ZeroDivisionError, OverflowError):
                    pass

    return None


def solve_countdown(numbers: List[int], target: int) -> bool:
    """验证数字和四则运算在括号树结构下是否存在精确解（支持 2, 3, 4 个数字）"""
    return find_countdown_solution(numbers, target) is not None


def generate_countdown_sample(num_count: int = 4, target_range: Tuple[int, int] = (10, 40)) -> Dict:
    """生成 100% 保证有解的 Countdown 样本，并搭载标准思维链引导 Prompt"""
    while True:
        numbers = [random.randint(1, 10) for _ in range(num_count)]
        target = random.randint(*target_range)
        if solve_countdown(numbers, target):
            break

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"User: Using the numbers {numbers}, create an equation that equals {target}.\n"
        f"You can use basic arithmetic operations (+, -, *, /) and each number can only be used once.\n"
        f"Assistant: <think>\n"
    )
    return {"numbers": numbers, "target": target, "prompt": prompt}

def extract_numbers_from_expr(expr_str: str) -> List[int]:
    """提取表达式中的所有正整数操作数"""
    return [int(n) for n in re.findall(r'\b\d+\b', expr_str)]

def parse_and_eval_safely(expr_str: str) -> float:
    """基于 AST 安全解析数学表达式，防止注入与危险调用"""
    allowed_nodes = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,  # 补上了 ast.FloorDiv
        ast.USub, ast.UAdd, ast.Pow
    )
    tree = ast.parse(expr_str.strip(), mode='eval')
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError("Disallowed expression syntax.")
    return float(eval(compile(tree, filename='', mode='eval')))


def compute_reward_old(solution_str: str, numbers: List[int], target: int) -> float:
    """
    自适应奖励：
    1. 优先提取 <answer>...</answer>
    2. 兜底提取中文 '答案:' 后的算式（帮助冷启动）
    """
    reward = 0.0
    expr_str = None

    # 1. 提取表达式：优先匹配标准标签，兜底匹配中英文答案标识
    answer_match = re.search(r"<answer>(.*?)</answer>", solution_str, flags=re.DOTALL)
    if answer_match:
        expr_str = answer_match.group(1).strip()
        reward += 0.2  # 成功使用标准 <answer> 标签加 0.2
    else:
        # 冷启动兜底：如果模型输出了类似 "答案: ..." 或 "Answer: ..."
        fallback_match = re.search(r"(?:答案|Answer)[:：]\s*([0-9\+\-\*\/\(\)\s\.]+)", solution_str)
        if fallback_match:
            expr_str = fallback_match.group(1).strip()
            reward += 0.05  # 兜底格式给一点微小的激励拉开差距

    if not expr_str:
        return 0.0

    # 2. 检查表达式合法性与数字约束
    try:
        # 去除末尾可能带有的等号或杂质
        expr_str = expr_str.split("=")[0].strip()
        used_numbers = extract_numbers_from_expr(expr_str)
        if sorted(used_numbers) == sorted(numbers):
            reward += 0.3  # 数字合法加 0.3
        else:
            return reward

        # 3. 结果计算
        result = parse_and_eval_safely(expr_str)
        if abs(result - target) < 1e-5:
            reward += 1.0  # 答案正确大奖 1.0
    except Exception:
        pass
    return reward

def compute_reward(solution_str: str, numbers: List[int], target: int) -> float:
    """
    升级版 Anti-Shortcut 平滑梯度奖励函数：
    1. 思考门槛 (Thinking Gate): 必须有闭合 </think> 且思考长度 >= 30 字符 (+0.1)
    2. 格式规范 (Format Gate): 严格包含 <answer> 标签 (+0.1)
    3. 数字约束平滑梯度 (Numbers Constraint Smooth Gradient):
       - 计算 AST 提取数字与目标数字的多重集交集 (Counter)
       - 匹配 N 个数字给 (N / len(numbers)) * 0.1 分 (用对 3 个得 0.075，2 个得 0.050，4 个满额 0.10)
       - 超额数字进行惩罚扣减
    4. 算术命中 (Jackpot Correctness): 仅在 4 个数字完全无误用完时触发，计算结果等于 Target (+2.5)
    """
    reward = 0.0

    # 1. 思考链门槛
    think_match = re.search(r"^(.*?)</think>", solution_str, flags=re.DOTALL)
    if think_match:
        thought_content = think_match.group(1).strip()
        if len(thought_content) >= 30:
            reward += 0.1

def verify_intermediate_equations(thought_content: str):
    """
    检查思考链中的所有算术等式 (形如 expr = number 或连等) 的真伪性。
    支持处理 Step 标号、链式等号、文字混排等复杂格式。
    返回: (valid_equations, fake_equations)
    """
    valid_equations = []
    fake_equations = []

    # 按行或标点符号切分语句
    sentences = re.split(r'[\n;\.]', thought_content)
    
    for sent in sentences:
        sent = sent.strip()
        if '=' not in sent:
            continue
        
        # 将句子按 '=' 分割
        parts = [p.strip() for p in sent.split('=')]
        if len(parts) < 2:
            continue
        
        # 遍历相邻对: part[i] == part[i+1]
        for i in range(len(parts) - 1):
            lhs_raw = parts[i]
            rhs_raw = parts[i+1]
            
            lhs_match = re.search(r'([0-9\s\+\-\*\/\(\)]+?)$', lhs_raw)
            rhs_match = re.search(r'^([0-9\s\+\-\*\/\(\)]+)', rhs_raw)
            if not lhs_match or not rhs_match:
                continue
            
            lhs_expr = lhs_match.group(1).strip()
            rhs_expr = rhs_match.group(1).strip()
            
            # 至少有一侧包含算术运算符 (+, -, *, /)，且两边都有数字
            has_op = any(op in lhs_expr for op in ['+', '-', '*', '/']) or any(op in rhs_expr for op in ['+', '-', '*', '/'])
            has_digit = any(c.isdigit() for c in lhs_expr) and any(c.isdigit() for c in rhs_expr)
            if not (has_op and has_digit):
                continue

            try:
                lhs_val = parse_and_eval_safely(lhs_expr)
                rhs_val = parse_and_eval_safely(rhs_expr)
                if abs(lhs_val - rhs_val) < 1e-4:
                    valid_equations.append((lhs_expr, rhs_expr))
                else:
                    fake_equations.append((lhs_expr, rhs_expr, lhs_val, rhs_val))
            except Exception:
                continue

    return valid_equations, fake_equations


def compute_reward(solution_str: str, numbers: List[int], target: int) -> float:
    """
    nano-GRPO 3.0: Anti-Cheating PRM (过程真实性与防作弊奖励体系)
    1. 思考推导提取与过程真实性检验 (PRM Gate):
       - 提取 <answer> 之前的真实思考内容
       - AST 逐行扫描所有形如 'A op B = C' 的心算等式
       - 若发现伪造算术等式 (例如 16 + 5 = 15): 一票否决恶意作弊，重罚 -0.30！
       - 若推导步步真实 (valid >= 2 且 fake == 0): 奖励真实心算 +0.15
       - 若包含回溯/自省词汇且步步真实: 额外奖励诚实探索 +0.10
    2. 答案格式规范 (Format Gate):
       - 严格包含 <answer>...</answer> (+0.10)
    3. 数字约束平滑分 (Numbers Constraint):
       - 多重集匹配比例 (+0.10)
    4. 终极真理暴击 (Exact Match Jackpot):
       - 4个数字完全无误用完 + 结果完全精确命中 target: +2.50 (满分可达 2.85)
    """
    # 1. 提取思考推导内容
    if "<answer>" in solution_str:
        thought_content = solution_str.split("<answer>")[0].replace("</think>", "").strip()
    elif "</think>" in solution_str:
        thought_content = solution_str.split("</think>")[0].strip()
    else:
        thought_content = solution_str.strip()

    valid_eqs, fake_eqs = verify_intermediate_equations(thought_content)

    # 🚨 Anti-Cheating 核心防御：只要抓包一步心算作弊，立即施加重罚，直接剥夺所有得分！
    if len(fake_eqs) > 0:
        return -0.30

    reward = 0.0

    # 鼓励真实心算推导
    if len(valid_eqs) >= 2:
        reward += 0.15
    elif len(valid_eqs) == 1:
        reward += 0.05
    elif len(thought_content) >= 30:
        reward += 0.05

    # 鼓励诚实回溯与探索（包含自我反思关键词且无假等式）
    backtrack_keywords = ["wait", "not equal", "backtrack", "try another", "retry", "instead"]
    if any(kw in thought_content.lower() for kw in backtrack_keywords):
        reward += 0.10

    # 2. 答案格式提取
    answer_match = re.search(r"<answer>(.*?)</answer>", solution_str, flags=re.DOTALL)
    if not answer_match:
        return reward

    expr_str = answer_match.group(1).strip()
    reward += 0.10

    # 3. 检查表达式合法性与数字约束平滑梯度
    try:
        expr_str = expr_str.split("=")[0].strip()
        used_numbers = extract_numbers_from_expr(expr_str)

        # 多重集交集匹配度
        target_counter = Counter(numbers)
        used_counter = Counter(used_numbers)
        matched_count = sum((target_counter & used_counter).values())
        excess_count = max(0, len(used_numbers) - len(numbers))
        effective_matched = max(0, matched_count - excess_count)

        # 平滑梯度比例分
        num_score = (effective_matched / len(numbers)) * 0.10
        reward += num_score

        # 4. 核心终极真理大奖：必须数字无误用齐且无冗余，且绝对精确命中
        if effective_matched == len(numbers) and sorted(used_numbers) == sorted(numbers):
            result = parse_and_eval_safely(expr_str)
            if abs(result - target) < 1e-5:
                reward += 2.50  # 绝对精确命中暴击！
    except Exception:
        pass

    return reward


compute_reward_smooth = compute_reward  # 别名兼容

if __name__ == "__main__":
    print("=" * 60)
    print(">>> 1. 测试随机样本生成")
    sample = generate_countdown_sample(num_count=4, target_range=(20, 30))
    print(f"题目数字: {sample['numbers']}")
    print(f"目标值:   {sample['target']}")
    print(f"Prompt 预览:\n{sample['prompt'][:120]}...\n")

    print("=" * 60)
    print(">>> 2. 测试不同解答下的奖励分值计算")
    test_nums = [3, 8, 3, 3]
    test_target = 24

    test_cases = [
        (
            "Case 1: 完美解答（满分 1.5）",
            """<think>
            Let's see: 8 * (3 - 3/3) = 8 * 2 = 16.
            How about (3 * 8) * (3 / 3) = 24 * 1 = 24!
            </think>
            <answer>
            (3 * 8) * (3 / 3)
            </answer>""",
            2.85
        ),
        (
            "Case 2: 思考完整且数字合法，但答案算错了",
            """<think>
            Maybe 3 + 8 + 3 + 3 = 17, not 24. Let's just output this.
            </think>
            <answer>
            3 + 8 + 3 + 3
            </answer>""",
            0.25
        ),
        (
            "Case 3: 心算公然作弊撒谎（应得 -0.30 严惩）",
            """<think>
            Step 1: 3 * 8 = 24.
            Step 2: 24 + 3 = 27.
            Step 3: 27 - 3 = 24.
            Wait, let's fake: 16 + 5 = 15.
            </think>
            <answer>
            (3 * 8) + (3 - 3)
            </answer>""",
            -0.30
        ),
        (
            "Case 4: 格式缺失，没有闭合标签",
            "The answer is (3 * 8) * (3 / 3) = 24.",
            0.0
        ),
        (
            "Case 5: 尝试非法代码注入（AST 安全拦截）",
            """<think>Hacking</think><answer>__import__('os').system('ls')</answer>""",
            0.0
        ),
    ]

    for desc, response, expected in test_cases:
        score = compute_reward(response, test_nums, test_target)
        status = "✅ PASS" if abs(score - expected) < 1e-4 else f"❌ FAIL (Got {score})"
        print(f"{desc:<45} -> 得分: {score:.2f} | 预期: {expected:.2f} [{status}]")
    print("=" * 60)