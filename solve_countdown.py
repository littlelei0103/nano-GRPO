import sys
import argparse
import torch
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from task_countdown import (
    SYSTEM_PROMPT,
    extract_numbers_from_expr,
    parse_and_eval_safely,
    verify_intermediate_equations
)

MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
DEFAULT_ADAPTER = "./checkpoint-best"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def solve_countdown_interactive(
    numbers: List[int],
    target: int,
    adapter_path: str = DEFAULT_ADAPTER,
    num_samples: int = 16,
    temperature: float = 0.7,
    max_new_tokens: int = 384
):
    print(f"\n🧠 [nano-GRPO 推理引擎] 正在加载模型权重 (Device: {DEVICE})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(DEVICE)

    try:
        model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
        print(f"✅ 成功挂载微调适配器: {adapter_path}")
    except Exception as e:
        print(f"⚠️ 无法加载适配器 ({e})，降级使用基座模型")
        model = base_model

    model.eval()

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"User: Using the numbers {numbers}, create an equation that equals {target}.\n"
        f"You can use basic arithmetic operations (+, -, *, /) and each number can only be used once.\n"
        f"Assistant: <think>\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    prompt_len = inputs["input_ids"].shape[1]

    expanded_ids = inputs["input_ids"].repeat(num_samples, 1)
    expanded_mask = inputs["attention_mask"].repeat(num_samples, 1)

    print(f"🎲 启动 Test-Time Compute (并发采样 N={num_samples}, T={temperature})...")
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=expanded_ids,
            attention_mask=expanded_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id
        )

    print("🔍 启动 AST 过程真实性与算术精确验证 (PRM Verifier)...")
    valid_solutions = []

    for i in range(num_samples):
        gen_text = tokenizer.decode(outputs[i][prompt_len:], skip_special_tokens=True)
        if "<answer>" not in gen_text or "</answer>" not in gen_text:
            continue

        expr_str = gen_text.split("<answer>")[1].split("</answer>")[0].strip()
        expr_clean = expr_str.split("=")[0].strip()
        used_nums = extract_numbers_from_expr(expr_clean)

        # 验证数字约束
        if sorted(used_nums) != sorted(numbers):
            continue

        # 验证运算值
        try:
            val = parse_and_eval_safely(expr_clean)
            if abs(val - target) < 1e-5:
                # 检查过程是否无作弊
                thought_part = gen_text.split("<answer>")[0].strip()
                v_eqs, f_eqs = verify_intermediate_equations(thought_part)
                valid_solutions.append({
                    "full_text": gen_text,
                    "equation": expr_clean,
                    "fake_count": len(f_eqs),
                    "valid_count": len(v_eqs)
                })
        except Exception:
            continue

    if valid_solutions:
        # 优先选择假等式为 0 的最诚实推导
        valid_solutions.sort(key=lambda x: (x["fake_count"], -x["valid_count"]))
        best = valid_solutions[0]
        print(f"\n🎯 【成功求解】共找到 {len(valid_solutions)}/{num_samples} 个有效解答！")
        print("=" * 60)
        print("💡 [思维链推导过程]:")
        print(best["full_text"].strip())
        print("=" * 60)
        print(f"🏆 [最终标准算式]: {best['equation']} = {target}\n")
    else:
        print(f"\n❌ 在 N={num_samples} 次尝试中未能命中完全合规的解。可尝试调大 --samples 或温度重试。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nano-GRPO 24-point Countdown Solver")
    parser.add_argument("--nums", type=int, nargs="+", default=[10, 7, 3, 2], help="输入的数字列表 (例如: 10 7 3 2)")
    parser.add_argument("--target", type=int, default=16, help="目标结果 Target (例如: 16)")
    parser.add_argument("--adapter", type=str, default=DEFAULT_ADAPTER, help="微调权重路径")
    parser.add_argument("--samples", type=int, default=16, help="Test-Time Compute 并发采样数 N")
    args = parser.parse_args()

    solve_countdown_interactive(
        numbers=args.nums,
        target=args.target,
        adapter_path=args.adapter,
        num_samples=args.samples
    )
