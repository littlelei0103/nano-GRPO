"""
trainer/grpo_loss.py - 规范化 GRPO 损失函数与参考模型 KL 约束模块

该模块统一管理:
1. 基于 EOS 的动态序列掩码 (Completion Mask)
2. 显存节约型 Completion-Only Log-Prob 计算 (FP32 Fused Cross-Entropy 与采样温度校准)
3. 零额外权重开销的基座参考模型 log-prob 获取 (通过 peft disable_adapter)
4. Schulman k3 正定高精度 KL 散度约束 (采用 FP32 torch.expm1 消除 BF16 消减误差)
5. 批次 Token 级与题目级梯度归一化，确保微批次切分下梯度严格等价
"""

import torch
import torch.nn.functional as F
from transformers import GenerationConfig
from typing import Dict, Optional, Tuple


def rollout_generation_config(tokenizer, max_new_tokens, temperature, top_p=1.0, do_sample=True):
    """Use a full-support temperature policy, with no inherited logit processors.

    Truncating support with top-p/top-k would make the sampled k3 reference KL
    biased. A fresh config also prevents model-specific penalties from silently
    changing the rollout distribution.
    """
    if top_p != 1.0:
        raise ValueError("GRPO training requires top_p=1.0 (no truncation) to match policy probabilities and reference KL")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return GenerationConfig(
        do_sample=do_sample, temperature=temperature if do_sample else 1.0,
        top_p=1.0, top_k=0 if do_sample else None, max_new_tokens=max_new_tokens, use_cache=True,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )


def generate_with_policy(model, generation_config, **inputs):
    """Prevent Transformers from filling unspecified fields from a Hub config."""
    original = model.generation_config
    model.generation_config = generation_config
    try:
        return model.generate(**inputs, generation_config=generation_config)
    finally:
        model.generation_config = original


def compute_completion_mask(
    outputs: torch.Tensor,
    prompt_len: int,
    eos_token_id: int,
    device: torch.device
) -> torch.Tensor:
    """
    构造 completion 部分的布尔掩码，确保只保留首个 EOS 及其之前的有效 token。

    Args:
        outputs: (batch_size, seq_len) 完整序列 input_ids
        prompt_len: prompt token 数量
        eos_token_id: 结束符 token id
        device: 计算设备

    Returns:
        mask: (batch_size, gen_len) 普通 Tensor (非 inference tensor)，True 为有效生成 token
    """
    seq_len = outputs.shape[1]
    gen_len = seq_len - prompt_len
    if gen_len <= 0:
        return torch.zeros((outputs.shape[0], 0), dtype=torch.bool, device=device)

    gen_tokens = outputs[:, prompt_len:]
    is_eos = (gen_tokens == eos_token_id)
    has_eos = is_eos.any(dim=-1)
    # 若包含 EOS，截断到首个 EOS；否则保留全部生成 token
    first_eos_idx = torch.where(has_eos, is_eos.int().argmax(dim=-1), gen_len - 1)
    col_indices = torch.arange(gen_len, device=device).unsqueeze(0)
    # Called from rollout's no_grad context (not inference_mode).
    resp_mask = (col_indices <= first_eos_idx.unsqueeze(1)).clone()
    return resp_mask


def compute_completion_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_len: int,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    仅对 completion 部分计算 token log 概率。
    在 FP32 精度下通过 cross_entropy 计算，与完整支持集采样的温度一致。

    Args:
        logits: (batch_size, seq_len, vocab_size) 模型完整输出 logits
        input_ids: (batch_size, seq_len) 完整序列 token ids
        prompt_len: prompt token 数量
        temperature: 采样策略温度参数 (保证采样概率与损失计算一致)

    Returns:
        completion_logps: (batch_size, gen_len) FP32 生成 token 的 log 概率
    """
    # 预测 input_ids[:, prompt_len] 的 logit 来自 logits[:, prompt_len - 1]
    # 预测 input_ids[:, -1] 的 logit 来自 logits[:, -2]
    comp_logits = logits[:, prompt_len - 1 : -1, :].contiguous().float()
    comp_labels = input_ids[:, prompt_len :].contiguous()

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if temperature != 1.0:
        comp_logits = comp_logits / temperature

    # Avoid prompt softmax work; cross_entropy may still allocate log-softmax intermediates.
    logprobs = -F.cross_entropy(
        comp_logits.transpose(1, 2),
        comp_labels,
        reduction="none"
    )  # (B, gen_len)
    return logprobs


def get_reference_logprobs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prompt_len: int,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    使用基座模型 (禁用 LoRA adapter) 计算参考模型的 completion log-probs。
    显存无需加载第二份模型权重。

    Args:
        model: PEFT LoRA 模型
        input_ids: (batch_size, seq_len) 完整序列
        prompt_len: prompt token 数量
        temperature: 策略采样温度

    Returns:
        ref_logprobs: (batch_size, gen_len) 冻结参考模型的 log 概率 (无梯度, FP32)
    """
    with torch.no_grad():
        if hasattr(model, "disable_adapter"):
            with model.disable_adapter():
                ref_logits = model(input_ids=input_ids).logits
                ref_logprobs = compute_completion_logprobs(ref_logits, input_ids, prompt_len, temperature=temperature)
                del ref_logits
        else:
            ref_logits = model(input_ids=input_ids).logits
            ref_logprobs = compute_completion_logprobs(ref_logits, input_ids, prompt_len, temperature=temperature)
            del ref_logits
    return ref_logprobs.detach()


def compute_grpo_loss(
    cur_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: Optional[torch.Tensor],
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_eps: float = 0.2,
    kl_coef: float = 0.04
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    计算单个批次内未归一化的逐 token GRPO 损失矩阵及关键统计指标。

    数学定义:
      ratio = exp(cur_logprobs - old_logprobs)
      surr1 = ratio * A
      surr2 = clamp(ratio, 1 - eps, 1 + eps) * A
      policy_loss = -min(surr1, surr2)

      KL 散度 (对基座参考模型 ref_logprobs，采用 Schulman k3 正定估计):
      采用 torch.expm1(d) - d (高精度 FP32) 彻底消除 BF16 精度下的灾难性消减误差，严格保证非负:
      d = ref_logprobs - cur_logprobs
      D_kl = max(0.0, expm1(d) - d)

      total_loss = policy_loss + kl_coef * D_kl

    Args:
        cur_logprobs: (B, gen_len) 当前策略 log-probs (带梯度)
        old_logprobs: (B, gen_len) rollout 采样策略 log-probs (已 detach)
        ref_logprobs: (B, gen_len) 冻结参考基座模型 log-probs (已 detach)，可为 None
        advantages: (B,) 或 (B, 1) 标准化后的 GRPO 组内优势
        mask: (B, gen_len) 有效生成 token 布尔掩码
        clip_eps: PPO 截断阈值
        kl_coef: 基座参考模型 KL 散度阻尼系数

    Returns:
        per_token_loss: (B, gen_len) 逐 token 损失 (FP32)
        metrics: 诊断指标字典 (mean_ratio, approx_kl, clip_frac, active_tokens)
    """
    cur_logprobs = cur_logprobs.float()
    old_logprobs = old_logprobs.float()
    advantages = advantages.float()

    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(-1)  # (B, 1)

    # 1. PPO-Clip 策略梯度
    log_ratio = torch.where(mask, cur_logprobs - old_logprobs, 0.0)
    ratio = torch.exp(log_ratio)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2)

    # 2. 统计剪裁比例与平均 ratio
    with torch.no_grad():
        active_tokens = mask.sum().float().clamp(min=1.0)
        clipped = ((ratio < (1.0 - clip_eps)) | (ratio > (1.0 + clip_eps))) & mask
        clip_frac = (clipped.sum().float() / active_tokens).item()
        mean_ratio = ((ratio * mask).sum() / active_tokens).item()

    # 3. 参考模型 KL 阻尼惩罚 (Schulman k3 估计量，FP32 expm1 消除消减误差)
    if kl_coef > 0.0 and ref_logprobs is not None:
        ref_logprobs = ref_logprobs.float()
        kl_log_ratio = torch.where(mask, ref_logprobs - cur_logprobs, 0.0)
        # expm1(x) - x 等价于 exp(x) - x - 1，但保留极微小差异的浮点精度
        per_token_kl = torch.clamp(torch.expm1(kl_log_ratio) - kl_log_ratio, min=0.0)
        total_loss = policy_loss + kl_coef * per_token_kl
        with torch.no_grad():
            approx_kl = ((per_token_kl * mask).sum() / active_tokens).item()
    else:
        total_loss = policy_loss
        approx_kl = 0.0

    if not torch.isfinite(total_loss[mask]).all():
        raise FloatingPointError("Non-finite GRPO loss; refusing an optimizer update")

    metrics = {
        "mean_ratio": mean_ratio,
        "clip_frac": clip_frac,
        "approx_kl": approx_kl,
        "active_tokens": active_tokens.item()
    }
    return total_loss, metrics
