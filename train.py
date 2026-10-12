import os
import sys
import json
import random
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from task_countdown import generate_countdown_sample, compute_reward, verify_intermediate_equations


# ===================== 超参数配置 =====================
MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
GROUP_SIZE = 8                    # 每个 Prompt 采样的候选数 G
MAX_PROMPT_LEN = 256
MAX_NEW_TOKENS = 384
LR = 5e-6
CLIP_EPS = 0.2
BETA_KL = 0.01                    # KL 散度惩罚系数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===================== nano-GRPO 3.0 配置 =====================
TOTAL_STEPS = 200
START_STEP = 1
RESUME_CHECKPOINT = "./checkpoint-best-v2"  # 基于 2.0 黄金最佳权重继续强化，直击痛点


def get_per_token_logps(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """从 logits 中提取每个 token 的 log 概率 (Shift 1 位对齐)"""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    per_token_logps = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    return per_token_logps


def train():
    print(f"🚀 正在初始化 nano-GRPO 3.0 (Anti-Cheating PRM + 混元课程学习) [设备: {DEVICE}]...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. 加载基座模型
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(DEVICE)
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()

    # 2. 从 Checkpoint 恢复或初始化 LoRA
    if RESUME_CHECKPOINT and os.path.exists(RESUME_CHECKPOINT):
        print(f"🔄 正在从 nano-GRPO 2.0 黄金权重唤醒: {RESUME_CHECKPOINT}", flush=True)
        model = PeftModel.from_pretrained(base_model, RESUME_CHECKPOINT, is_trainable=True)
    else:
        print("🌱 未检测到恢复权重，初始化全新 LoRA 适配器...", flush=True)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(base_model, lora_config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    best_reward = -1.0

    print("✨ nano-GRPO 3.0 (Anti-Cheating PRM & Curriculum Mixing) 正式点火！\n", flush=True)

    for step in range(START_STEP, TOTAL_STEPS + 1):
        # ---------------- 步骤 1: 混元动态课程抽样 (Curriculum Mixing) ----------------
        # 动态退火比例: 前期 40% 3数稳固正反馈梯队，后期递减至 20%，始终保持成功信号活跃
        p_3num = max(0.20, 0.40 - 0.20 * (step / TOTAL_STEPS))
        if random.random() < p_3num:
            stage_name = "Mix (3 Nums)"
            sample = generate_countdown_sample(num_count=3, target_range=(10, 30))
        else:
            stage_name = "Mix (4 Nums)"
            sample = generate_countdown_sample(num_count=4, target_range=(10, 40))

        prompt = sample["prompt"]
        prompt_inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        prompt_len = prompt_inputs["input_ids"].shape[1]

        # 批量扩展采样：复制 G 份 Prompt 批量并发生成 (Group Rollout)
        expanded_prompt_ids = prompt_inputs["input_ids"].repeat(GROUP_SIZE, 1)
        expanded_attention_mask = prompt_inputs["attention_mask"].repeat(GROUP_SIZE, 1)

        with torch.inference_mode():
            outputs = model.generate(
                input_ids=expanded_prompt_ids,
                attention_mask=expanded_attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                use_cache=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # ---------------- 步骤 2: 规则判卷与组内相对优势计算 (PRM Anti-Cheating) ----------------
        rewards_list = []
        solved_count = 0
        cheating_count = 0
        for i in range(GROUP_SIZE):
            generated_text = tokenizer.decode(outputs[i][prompt_len:], skip_special_tokens=True)
            r = compute_reward(generated_text, sample["numbers"], sample["target"])
            rewards_list.append(r)
            if r <= -0.2:
                cheating_count += 1
            elif r >= 2.5:  # 算术精确命中
                solved_count += 1

        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=DEVICE)

        # 组内相对优势标准化 (Group Advantage Normalization)
        mean_r = rewards.mean()
        std_r = rewards.std() + 1e-8
        advantages = (rewards - mean_r) / std_r

        # ---------------- 步骤 3: 计算 Policy 与 Ref 的 Token Log-probs ----------------
        seq_len = outputs.shape[1]
        response_mask = torch.zeros((GROUP_SIZE, seq_len - 1), dtype=torch.bool, device=DEVICE)
        response_mask[:, prompt_len - 1:] = (outputs[:, 1:] != tokenizer.pad_token_id)[:, prompt_len - 1:]

        # A. 策略模型前向传播
        model.train()
        logits = model(input_ids=outputs).logits
        token_logprobs = get_per_token_logps(logits, outputs)

        # B. 参考模型前向传播 (关闭 LoRA，零显存占用)
        with torch.no_grad():
            with model.disable_adapter():
                ref_logits = model(input_ids=outputs).logits
                ref_token_logprobs = get_per_token_logps(ref_logits, outputs)

        # ---------------- 步骤 4: 计算 GRPO PPO 裁剪损失与 KL 散度 ----------------
        with torch.no_grad():
            old_token_logprobs = token_logprobs.detach()

        ratio = torch.exp(token_logprobs - old_token_logprobs)
        surr1 = ratio * advantages.unsqueeze(-1)
        surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantages.unsqueeze(-1)
        policy_loss = -torch.min(surr1, surr2)

        # DeepSeek-R1 无偏 KL 散度
        log_ratio = ref_token_logprobs - token_logprobs
        kl_div = torch.exp(log_ratio) - log_ratio - 1.0

        total_loss = policy_loss + BETA_KL * kl_div
        loss = (total_loss * response_mask).sum() / (response_mask.sum() + 1e-8)

        # ---------------- 步骤 5: 反向传播与优化 ----------------
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # ---------------- 步骤 6: 打印心跳与自检 ----------------
        if step % 5 == 0 or step == 1 or solved_count > 0:
            best_idx = torch.argmax(rewards).item()
            best_pred = tokenizer.decode(outputs[best_idx][prompt_len:], skip_special_tokens=True)
            v_eqs, f_eqs = verify_intermediate_equations(best_pred)
            audit_str = f"✅ 心算真伪通过 ({len(v_eqs)}等式)" if len(f_eqs) == 0 else f"🚨 检出假等式 ({len(f_eqs)}处: {f_eqs})"
            print(f"--- [Step {step} ({stage_name}) 组内最佳 (Reward: {rewards[best_idx].item():.3f}, 命中: {solved_count}/{GROUP_SIZE}, 抓包作弊: {cheating_count}/{GROUP_SIZE})] ---\n"
                  f"题目: {sample['numbers']} -> Target: {sample['target']} | PRM 审计: {audit_str}\n"
                  f"{best_pred.strip()}\n"
                  f"-------------------------------------------------------------", flush=True)

        avg_resp_len = (response_mask.sum(dim=-1).float().mean()).item()
        print(f"Step [{step:03d}/{TOTAL_STEPS}] [{stage_name}] | "
              f"Solved: {solved_count}/{GROUP_SIZE} | "
              f"Fakes: {cheating_count}/{GROUP_SIZE} | "
              f"Mean R: {mean_r.item():.2f} | "
              f"Max: {rewards.max().item():.2f} | "
              f"Loss: {loss.item():.4f} | "
              f"Len: {avg_resp_len:.1f}", flush=True)

        # 定期保存检查点
        if step % 20 == 0:
            model.save_pretrained(f"./checkpoint-step-{step}")
            tokenizer.save_pretrained(f"./checkpoint-step-{step}")

        # 刷新历史纪录保存黄金存档
        if mean_r.item() > best_reward:
            best_reward = mean_r.item()
            model.save_pretrained("./checkpoint-best-v3")
            model.save_pretrained("./checkpoint-best")
            print(f"🌟 刷新历史最佳 Mean Reward: {best_reward:.2f}，已更新至 ./checkpoint-best-v3 & ./checkpoint-best", flush=True)


if __name__ == "__main__":
    train()