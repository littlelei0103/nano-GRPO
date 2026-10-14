"""
tasks/bug_differential.py - 基于输入差分执行的程序缺陷审计轨道 (Differential Bug Hunting Track)

设计理念 (方向 C):
  不同于原始 BugHunter 要求小模型同时完成“搜索输入”与“口算标准期望输出 (expected_value)”的双重认知负担，
  本任务将执行与判题完全交由沙盒环境的差分执行器 (Differential Oracle):
    1. 给定待测函数规范 (Spec) 与含缺陷代码 (Buggy Code)。
    2. 模型仅需推演能够触发缺陷的边界输入实参字面量，输出在 <input>...</input> 标签内。
       例如: <input>[1, 3, 5], 5</input> 或 <input>[]</input>
    3. 环境判题器 (Differential Oracle):
       - 使用 ast.literal_eval 安全解析输入实参 (严禁任意代码执行与函数调用注入)；
       - 验证输入契约与题目前置条件 (Precondition / Input Contract)；
       - 在安全沙盒中分别执行 Golden Reference 与 Buggy Code；
       - 若 Golden 正常执行无异常，且 Buggy 产生异常或两者输出不一致 (Golden != Buggy)，则判定为缺陷击穿 (+1.0)；
       - 若两者输出完全一致，则判定为漏报 (0.0, missed_bug)；
       - 若 Golden 执行失败或输入非法，则判定为前置条件违规 (0.0, precondition_violation / golden_failed)。

注意:
  本任务与原始断言任务属于不同定义。评测指标统一命名为 `differential_exposure_rate` (合法输入触发差异成功率)。
"""

import os
import re
import sys
import ast
import json
import subprocess
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tasks.base import BaseTask
from tasks.contracts import CONTRACTS, validate_input
from tasks.semantic_values import loads_value, are_semantic_values_equal

REASONING_SYSTEM_PROMPT = (
    "You are an expert adversarial software testing engineer.\n"
    "Your objective is to find an input argument where the buggy implementation behaves differently from the problem specification.\n\n"
    "Instructions:\n"
    "1. In <think> and </think> tags, briefly analyze the edge cases where the code fails.\n"
    "2. In <input> and </input> tags, output ONLY the argument literals to pass to `target_func(...)`.\n"
    "   Do not write code, assertions, or expected values. Output only comma-separated literals.\n\n"
    "Examples:\n"
    "  <input>[1, 3, 5], 5</input>\n"
    "  <input>[]</input>\n"
    "  <input>\"hello\", 0</input>"
)

DIRECT_SYSTEM_PROMPT = (
    "You are an expert adversarial software testing engineer.\n"
    "Your objective is to find an input argument where the buggy implementation behaves differently from the problem specification.\n\n"
    "Instructions:\n"
    "Output ONLY the argument literals to pass to `target_func(...)` inside <input> and </input> tags.\n"
    "Do not write explanations, thoughts, assertions, or expected values. Output only literals.\n\n"
    "Examples:\n"
    "  <input>[1, 3, 5], 5</input>\n"
    "  <input>[]</input>\n"
    "  <input>\"hello\", 0</input>"
)

SYSTEM_PROMPT = REASONING_SYSTEM_PROMPT


def extract_differential_input(completion: str) -> Optional[str]:
    """
    从模型生成中提取 <input>...</input> 标签内的实参字面量字符串。
    兼顾 reasoning 模式 (<input>...</input>) 与 direct 模式 (前缀已被 Prompt 吞入，仅剩 ...</input>)。
    """
    m = re.findall(r"<input>(.*?)</input>", completion, re.DOTALL | re.IGNORECASE)
    if m:
        return m[-1].strip()
    m_prefixed = re.findall(r"^(.*?)</input>", completion.strip(), re.DOTALL)
    if m_prefixed:
        return m_prefixed[0].strip()
    m_loose = re.findall(r"<input>(.*)", completion, re.DOTALL | re.IGNORECASE)
    if m_loose:
        return m_loose[-1].strip()
    return None


def parse_argument_literals(raw_str: str) -> Tuple[bool, tuple, Optional[str]]:
    """
    安全解析模型给出的参数字面量。
    只允许 AST 纯字面量类型 (Constant, List, Tuple, Dict, Set, 负号)，严禁任意调用或表达式注入。
    """
    if raw_str is None:
        return False, (), "Missing input content"
    raw_str = raw_str.strip()
    if not raw_str:
        return True, (), None

    allowed_nodes = (
        ast.Expression, ast.Tuple, ast.List, ast.Dict, ast.Set,
        ast.Constant, ast.UnaryOp, ast.USub, ast.UAdd, ast.Load
    )

    try:
        # 若以括号包裹且本身是合法 tuple 表达式
        if raw_str.startswith("(") and raw_str.endswith(")"):
            try:
                tree = ast.parse(raw_str, mode="eval")
                if isinstance(tree.body, ast.Tuple):
                    for node in ast.walk(tree):
                        if not isinstance(node, allowed_nodes):
                            return False, (), f"Unsupported AST node: {type(node).__name__}"
                    return True, ast.literal_eval(tree), None
            except Exception:
                pass

        # 统一以尾部逗号包裹为元组表达式以支持多参数解包
        tree = ast.parse(f"({raw_str},)", mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                return False, (), f"Unsupported AST node: {type(node).__name__}"
        parsed = ast.literal_eval(tree)
        return True, parsed, None
    except Exception as e:
        return False, (), f"Literal parse error: {e}"


def run_function_in_sandbox(code: str, args: tuple, timeout_sec: float = 1.0) -> Tuple[int, str, str]:
    """
    在隔离子进程中安全执行 target_func(*args)。
    返回: (returncode, stdout, stderr)
      0: 正常返回并输出类型化 JSON
      45: 返回值无法按支持的类型序列化
      42: 函数抛出预期异常
      44: 非法系统中断
    """
    harness = f"""
import sys
import math
import collections
import itertools
import functools
import bisect
import heapq
import re
import contextlib
import io
sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})
from semantic_values import dumps_value

{code}

args = {repr(args)}
ret_code = 0
try:
    with contextlib.redirect_stdout(io.StringIO()):
        res = target_func(*args)
except Exception as e:
    sys.stderr.write(str(e))
    ret_code = 42
except BaseException as be:
    ret_code = 44
if ret_code == 0:
    try:
        sys.stdout.write(dumps_value(res))
    except Exception as e:
        sys.stderr.write(str(e))
        ret_code = 45
sys.exit(ret_code)
"""
    try:
        p = subprocess.run(
            [sys.executable, "-c", harness],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            text=True
        )
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "Execution timed out"
    except Exception as e:
        return -2, "", str(e)


def evaluate_differential_exposure(
    raw_input_str: Optional[str],
    golden_code: str,
    buggy_code: str,
    input_contract: Optional[str] = None,
    precondition: Optional[Any] = None,
    timeout_sec: float = 1.0
) -> Tuple[bool, str, str, Optional[tuple]]:
    """
    执行差分判题:
    返回: (exposed: bool, category: str, reason: str, parsed_args)
    """
    if raw_input_str is None:
        return False, "format_malformed", "No <input> tag found in completion", None

    is_valid, args, err_msg = parse_argument_literals(raw_input_str)
    if not is_valid:
        return False, "format_malformed", f"Invalid literal arguments: {err_msg}", None

    # 输入契约校验 (Input Contract Validation)
    if input_contract:
        try:
            if not validate_input(input_contract, golden_code, args, {}):
                return False, "precondition_violation", f"Input violates input_contract '{input_contract}'", args
        except Exception as e:
            return False, "precondition_violation", f"Contract check failed: {e}", args

    # 前置条件校验 (Precondition Validation)
    if precondition:
        try:
            if callable(precondition):
                if not precondition(*args):
                    return False, "precondition_violation", "Input violates callable precondition", args
            elif isinstance(precondition, str):
                ns = {}
                exec(precondition, ns)
                val_fn = ns.get("validate") or ns.get("validate_input")
                if not callable(val_fn) or not val_fn(*args):
                    return False, "precondition_violation", "Input violates string precondition", args
        except Exception as e:
            return False, "precondition_violation", f"Precondition execution error: {e}", args

    # 1. 执行 Golden 参考实现 (必须成功无异常执行，以确保输入合法且符合题意)
    rc_g, out_g, err_g = run_function_in_sandbox(golden_code, args, timeout_sec)
    if rc_g != 0:
        err = err_g[:150] or f"exit code {rc_g}"
        return False, "golden_failed", f"Golden reference failed on input ({err})", args

    # 解析 Golden 返回值字面量
    try:
        val_g = loads_value(out_g)
    except Exception as e:
        return False, "golden_failed", f"Golden returned unparseable output literal: {e}", args

    # 2. 执行待测缺陷实现 (Buggy Code)
    rc_b, out_b, err_b = run_function_in_sandbox(buggy_code, args, timeout_sec)

    # 3. 差分判定
    if rc_b == 42 or rc_b == 44 or rc_b == -1:
        # Buggy 代码抛出未捕获异常、段错误或超时，成功暴露缺陷！
        return True, "success", f"Buggy crashed or raised exception ({err_b[:100]})", args

    if rc_b == 0:
        try:
            val_b = loads_value(out_b)
        except Exception:
            # 传输/序列化失败不能充当行为差异的证据
            return False, "execution_error", f"Buggy returned invalid result payload: {out_b[:100]}", args

        if not are_semantic_values_equal(val_g, val_b):
            # 两个实现的语义返回值或类型不一致，成功暴露行为缺陷！
            return True, "success", f"Output diverged: Golden={repr(val_g)} ({type(val_g).__name__}) vs Buggy={repr(val_b)} ({type(val_b).__name__})", args
        else:
            return False, "missed_bug", f"Outputs identical ({repr(val_g)})", args

    return False, "execution_error", f"Buggy execution error ({err_b[:100]})", args


class DifferentialBugHunterTask(BaseTask):
    """
    方向 C 差分程序缺陷审计任务 (Differential Bug Hunter Task)。
    输入格式: 模型仅输出纯实参字面量 <input>arg1, arg2</input>。
    判题机制: 差分真理执行器 (Differential Oracle)，无模型口算负担。
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        num_train: int = 60,
        num_val: int = 20,
        num_test: int = 20,
        prompt_mode: str = "reasoning"
    ):
        self.sample_idx = 0
        self.prompt_mode = prompt_mode
        default_dataset_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "bug_hunter_dataset.json"
        )
        target_path = data_path if data_path and os.path.exists(data_path) else default_dataset_path

        if os.path.exists(target_path):
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            raise FileNotFoundError(f"BugHunter dataset not found at: {target_path}")

        raw_train = data.get("train", [])
        raw_val = data.get("val", [])
        raw_test = data.get("test", [])

        self.train_problems = [self._build_item(p) for p in raw_train[:num_train]]
        self.val_problems = [self._build_item(p) for p in raw_val[:num_val]]
        self.test_problems = [self._build_item(p) for p in raw_test[:num_test]]

    def _build_item(self, prob: Dict[str, Any]) -> Dict[str, Any]:
        problem_id = prob.get("id", prob.get("problem_id", ""))
        if self.prompt_mode == "direct":
            prompt = (
                f"{DIRECT_SYSTEM_PROMPT}\n\n"
                f"Problem Specification:\n{prob['spec']}\n\n"
                f"Implementation to Audit:\n```python\n{prob['buggy_code']}\n```\n\n"
                f"Output format: <input>ARGUMENTS</input>\n\n"
                f"Assistant: <input>"
            )
        else:
            prompt = (
                f"{REASONING_SYSTEM_PROMPT}\n\n"
                f"Problem Specification:\n{prob['spec']}\n\n"
                f"Implementation to Audit:\n```python\n{prob['buggy_code']}\n```\n\n"
                f"Assistant: <think>\n"
            )
        return {
            "prompt": prompt,
            "id": problem_id,
            "problem_id": problem_id,
            "spec": prob["spec"],
            "buggy_code": prob["buggy_code"],
            "golden_code": prob["golden_code"],
            "precondition": prob.get("precondition", None),
            "input_contract": prob.get("input_contract", problem_id if problem_id in CONTRACTS else None),
            "ground_truth": problem_id
        }

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
        is_truncated = meta.get("is_truncated", False)
        raw_input = extract_differential_input(comp)

        if raw_input is None:
            if is_truncated:
                return {
                    "reward": 0.0,
                    "category": "truncated",
                    "reason": "Generation reached token limit before completing <input> tag",
                    "input_args": None,
                    "format_compliant": False
                }
            return {
                "reward": 0.0,
                "category": "format_malformed",
                "reason": "No valid <input>...</input> block found in completion",
                "input_args": None,
                "format_compliant": False
            }

        exposed, category, reason, parsed_args = evaluate_differential_exposure(
            raw_input_str=raw_input,
            golden_code=meta["golden_code"],
            buggy_code=meta["buggy_code"],
            input_contract=meta.get("input_contract"),
            precondition=meta.get("precondition"),
            timeout_sec=1.0
        )

        format_compliant = raw_input is not None
        if self.prompt_mode == "reasoning":
            format_compliant = format_compliant and ("</think>" in comp)

        return {
            "reward": 1.0 if exposed else 0.0,
            "category": category,
            "reason": reason,
            "input_args": repr(parsed_args) if parsed_args is not None else None,
            "format_compliant": format_compliant
        }

    def compute_rewards_with_details(
        self,
        prompts: List[str],
        completions: List[str],
        metadata: List[Dict[str, Any]]
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
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
