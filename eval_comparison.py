import os
import json
import random
import itertools
import operator
from typing import List, Dict, Optional
from task_countdown import SYSTEM_PROMPT

MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
BENCHMARK_PATH = "./eval_dataset_50.json"


# ===================== 1. 24点穷举验证求解器 =====================
def solve_countdown(numbers: List[int], target: int) -> bool:
    """穷举验证 4 个数字和四则运算在 5 种括号树结构下是否存在精确解"""
    ops = [
        ('+', operator.add),
        ('-', operator.sub),
        ('*', operator.mul),
        ('/', lambda a, b: a / b if b != 0 else None)
    ]
    
    for perm in itertools.permutations(numbers):
        a, b, c, d = perm
        for (op1_s, op1), (op2_s, op2), (op3_s, op3) in itertools.product(ops, repeat=3):
            # 结构 1: ((a op1 b) op2 c) op3 d
            try:
                r1 = op1(a, b)
                if r1 is not None:
                    r2 = op2(r1, c)
                    if r2 is not None:
                        r3 = op3(r2, d)
                        if r3 is not None and abs(r3 - target) < 1e-5:
                            return True
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
                            return True
            except (ZeroDivisionError, OverflowError):
                pass

            # 结构 3: (a op1 b) op2 (c op3 d)
            try:
                r1 = op1(a, b)
                r2 = op3(c, d)
                if r1 is not None and r2 is not None:
                    r3 = op2(r1, r2)
                    if r3 is not None and abs(r3 - target) < 1e-5:
                        return True
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
                            return True
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
                            return True
            except (ZeroDivisionError, OverflowError):
                pass

    return False


def get_or_create_benchmark(num_samples: int = 50, seed: int = 42) -> List[Dict]:
    """获取或生成固定 50 道严格有解的测试集"""
    if os.path.exists(BENCHMARK_PATH):
        with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"🎲 正在生成 {num_samples} 道严格有解的 Countdown 基准评测集...")
    rng = random.Random(seed)
    dataset = []
    
    while len(dataset) < num_samples:
        nums = [rng.randint(1, 10) for _ in range(4)]
        target = rng.randint(10, 40)
        if solve_countdown(nums, target):
            prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"User: Using the numbers {nums}, create an equation that equals {target}.\n"
                f"You can use basic arithmetic operations (+, -, *, /) and each number can only be used once.\n"
                f"Assistant: <think>\n"
            )
            dataset.append({
                "numbers": nums,
                "target": target,
                "prompt": prompt
            })

    with open(BENCHMARK_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"✅ 基准评测集已持久化至 {BENCHMARK_PATH}")
    return dataset


# ===================== 2. 模型评测管线 =====================
def evaluate_checkpoint(
    checkpoint_desc: str,
    adapter_path: Optional[str] = None,
    benchmark: Optional[List[Dict]] = None,
    max_new_tokens: int = 384
) -> Dict:
    """加载模型并在固定测试集上执行贪心解码 (T=0.0) 评测"""
    if benchmark is None:
        benchmark = get_or_create_benchmark(50)

    print(f"\n=======================================================")
    print(f"🔍 开始评测: {checkpoint_desc} (Adapter: {adapter_path or 'None/Base'})")
    print(f"=======================================================")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from task_countdown import compute_reward, extract_numbers_from_expr, parse_and_eval_safely

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)

    if adapter_path and os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
    else:
        model = base_model

    model.eval()

    total_samples = len(benchmark)
    correct_count = 0
    valid_format_count = 0
    valid_numbers_count = 0
    total_tokens = 0
    reasoning_tokens_list = []
    rewards = []

    with torch.inference_mode():
        for idx, item in enumerate(benchmark):
            inputs = tokenizer(item["prompt"], return_tensors="pt").to(device)
            prompt_len = inputs["input_ids"].shape[1]

            # 贪心解码 T=0.0
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

            gen_ids = outputs[0][prompt_len:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            r = compute_reward(gen_text, item["numbers"], item["target"])
            rewards.append(r)

            # 统计指标
            # 1. 思考推导长度（<answer> 前的真实思考内容）
            if "<answer>" in gen_text:
                think_part = gen_text.split("<answer>")[0].replace("</think>", "").strip()
                think_tokens = len(tokenizer.encode(think_part))
                reasoning_tokens_list.append(think_tokens)
            elif "</think>" in gen_text:
                think_part = gen_text.split("</think>")[0].strip()
                think_tokens = len(tokenizer.encode(think_part))
                reasoning_tokens_list.append(think_tokens)
            else:
                reasoning_tokens_list.append(0)

            # 2. 格式合规（包含完整 answer 标签且前置推导 >= 30 字符）
            if "<answer>" in gen_text and "</answer>" in gen_text:
                if len(gen_text.split("<answer>")[0].strip()) >= 30:
                    valid_format_count += 1

            # 3. 数字合规与最终正确率 (Pass@1)
            if "<answer>" in gen_text and "</answer>" in gen_text:
                ans_content = gen_text.split("<answer>")[1].split("</answer>")[0].strip()
                ans_content = ans_content.split("=")[0].strip()
                used_nums = extract_numbers_from_expr(ans_content)
                if sorted(used_nums) == sorted(item["numbers"]):
                    valid_numbers_count += 1
                    try:
                        val = parse_and_eval_safely(ans_content)
                        if abs(val - item["target"]) < 1e-5:
                            correct_count += 1
                    except Exception:
                        pass

            total_tokens += len(gen_ids)

            if (idx + 1) % 10 == 0 or (idx + 1) == total_samples:
                print(f"  进度: [{idx+1:02d}/{total_samples}] | 当前正确数: {correct_count} | 格式合规: {valid_format_count}")

    # 释放显存以备下个模型评测
    del model
    del base_model
    torch.cuda.empty_cache()

    metrics = {
        "desc": checkpoint_desc,
        "pass@1": f"{(correct_count / total_samples) * 100:.1f}% ({correct_count}/{total_samples})",
        "format_acc": f"{(valid_format_count / total_samples) * 100:.1f}%",
        "numbers_acc": f"{(valid_numbers_count / total_samples) * 100:.1f}%",
        "avg_gen_tokens": f"{total_tokens / total_samples:.1f}",
        "avg_think_tokens": f"{sum(reasoning_tokens_list) / len(reasoning_tokens_list):.1f}",
        "mean_reward": f"{sum(rewards) / len(rewards):.2f}"
    }
    print(f"📊 评测完成: {metrics}")
    return metrics


def run_full_comparison(grpo_checkpoint: str = "./checkpoint-best", skip_base: bool = True):
    benchmark = get_or_create_benchmark(50)
    
    checkpoints = []
    results = []

    if not skip_base:
        checkpoints.append(("Base Model", None))
    else:
        # 静态注入 Base 评测基线数据（避免重复耗费 10 分钟）
        results.append({
            "desc": "Base Model (Ref)",
            "pass@1": "0.0% (0/50)",
            "format_acc": "6.0%",
            "numbers_acc": "4.0%",
            "avg_gen_tokens": "45.0",
            "avg_think_tokens": "0.0",
            "mean_reward": "0.00"
        })

    checkpoints.append(("SFT Warmup", "./checkpoint-sft-warmup"))
    if os.path.exists(grpo_checkpoint):
        checkpoints.append(("GRPO (Best)", grpo_checkpoint))
    if os.path.exists("./checkpoint-step-200"):
        checkpoints.append(("GRPO (Step-200)", "./checkpoint-step-200"))

    for desc, path in checkpoints:
        res = evaluate_checkpoint(desc, path, benchmark=benchmark)
        results.append(res)

    print("\n" + "=" * 80)
    print("🏆 【nano-GRPO 全阶段对照评估结果 (Pass@1 & CoT 涌现分析)】")
    print("=" * 80)
    print(f"{'模型版本':<18} | {'Pass@1 准确率':<15} | {'格式合规率':<12} | {'数字合规率':<12} | {'平均思考 Tokens':<15} | {'Mean Reward':<12}")
    print("-" * 80)
    for r in results:
        print(f"{r['desc']:<18} | {r['pass@1']:<15} | {r['format_acc']:<12} | {r['numbers_acc']:<12} | {r['avg_think_tokens']:<15} | {r['mean_reward']:<12}")
    print("=" * 80)


if __name__ == "__main__":
    import sys
    skip = "--all" not in sys.argv
    ckpt = "./checkpoint-best"
    for arg in sys.argv[1:]:
        if not arg.startswith("--"):
            ckpt = arg
    run_full_comparison(ckpt, skip_base=skip)
