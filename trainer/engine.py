"""
trainer/engine.py - nano-GRPO 核心强化学习算法引擎 (流式单步 GRPO)

核心机制:
  1. Group Rollout: 多路随机采样 G 个候选生成，微批次切分严格保护 8GB 显存。
  2. Group Relative Advantage: A_i = (R_i - mean) / (std + 1e-8) 标准化相对优势。
  3. Zero Update Guard: 当组内得分为全对或全错 (std == 0) 时安全跳过反向传播。
  4. PPO-Clip 策略梯度: 记录 rollout 时真实的 old_logprobs，支持策略边界截断约束。
  5. 基座参考 KL 散度约束: 使用 PEFT 无额外参考权重 base 策略计算 ref_logprobs，采用 Schulman k3 非负 KL 约束抑制 Policy Drift。
  6. Token-normalized GRPO 全局 Token 级均值归一化: 确保微批次大小不影响优化目标与累积梯度。
"""

import os
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional

from trainer.grpo_loss import (
    compute_completion_mask,
    compute_completion_logprobs,
    get_reference_logprobs,
    compute_grpo_loss,
    rollout_generation_config,
    generate_with_policy
)
from trainer.utils import print_gpu_memory


class GRPOTrainer:
    """
    通用、解耦的轻量级 Group Relative Policy Optimization 训练器。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        task: Any,
        lr: float = 2e-5,
        group_size: int = 8,
        micro_batch_size: int = 4,
        kl_coef: float = 0.04,
        clip_eps: float = 0.2,
        max_new_tokens: int = 384,
        temperature: float = 0.7,
        top_p: float = 1.0,
        device: str = "cuda",
        dynamic_resampling: bool = False,
        max_resample_rounds: int = 2,
        resample_temp_boost: float = 0.0
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.task = task

        self.lr = lr
        self.group_size = group_size
        self.micro_batch_size = micro_batch_size
        self.kl_coef = kl_coef
        self.clip_eps = clip_eps
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.dynamic_resampling = dynamic_resampling
        self.max_resample_rounds = max(1, max_resample_rounds)
        if resample_temp_boost != 0.0:
            raise ValueError("resample_temp_boost must be 0: resampling uses the same policy temperature")
        self.resample_temp_boost = resample_temp_boost
        self.generation_config = rollout_generation_config(tokenizer, max_new_tokens, temperature, top_p)
        if dynamic_resampling:
            raise NotImplementedError(
                "Dynamic active resampling is currently only supported in macro-batch mode (MiMoGRPOTrainer). "
                "For reliable baseline experiments, leave dynamic_resampling=False."
            )
        self.device = device

        assert self.group_size % self.micro_batch_size == 0, (
            f"group_size ({self.group_size}) 必须被 micro_batch_size ({self.micro_batch_size}) 整除！"
        )
        self.num_micro_batches = self.group_size // self.micro_batch_size

        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr
        )

    def train_step(self, step_idx: int) -> Dict[str, Any]:
        """
        执行单个 GRPO 训练步:
          1. 从 Task 获取 1 道题目
          2. 分微批次采样 G 个候选 Completion 并记录 rollout old_logprobs 与 ref_logprobs
          3. 调用 Task 双向沙盒计算客观奖励
          4. 计算组内相对优势 (含 Zero Update Guard)
          5. PPO-Clip 前向、真实 KL 阻尼与 Token-normalized GRPO 全局 Token 归一化累积
          6. 优化器梯度裁剪并更新，返回指标快照
        """
        # ================= 步骤 1: 任务抽样与 Prompt 编码 =================
        batch = self.task.get_batch(batch_size=1)
        sample = batch[0]
        prompt = sample["prompt"]

        prompt_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = prompt_inputs["input_ids"].shape[1]

        # ================= 步骤 2: 分批并发自回归生成 (Rollout) =================
        self.model.eval()
        self.model.config.use_cache = True  # 推理阶段开 KV Cache 加速

        micro_outputs_list = []
        micro_masks_list = []
        micro_old_logprobs_list = []
        micro_ref_logprobs_list = []

        with torch.no_grad():
            for _ in range(self.num_micro_batches):
                micro_prompt_ids = prompt_inputs["input_ids"].repeat(self.micro_batch_size, 1)
                micro_attention_mask = prompt_inputs["attention_mask"].repeat(self.micro_batch_size, 1)

                m_outputs = generate_with_policy(self.model,
                    input_ids=micro_prompt_ids,
                    attention_mask=micro_attention_mask,
                    generation_config=self.generation_config,
                )
                micro_outputs_list.append(m_outputs)

                # 构造包含首个 EOS 的精确掩码 (普通张量，支持 autograd 暂存)
                m_mask = compute_completion_mask(
                    m_outputs,
                    prompt_len=prompt_len,
                    eos_token_id=self.tokenizer.eos_token_id,
                    device=self.device
                )
                micro_masks_list.append(m_mask)

                # 记录 rollout 策略的真实 old_logprobs (与采样温度保持一致)
                m_logits = self.model(input_ids=m_outputs).logits
                m_old_lp = compute_completion_logprobs(m_logits, m_outputs, prompt_len, temperature=self.temperature)
                del m_logits
                micro_old_logprobs_list.append(m_old_lp)

                # 记录基座参考模型 (禁用 LoRA adapter) 的 ref_logprobs
                if self.kl_coef > 0.0:
                    m_ref_lp = get_reference_logprobs(self.model, m_outputs, prompt_len, temperature=self.temperature)
                else:
                    m_ref_lp = None
                micro_ref_logprobs_list.append(m_ref_lp)

        # 解码并计算客观奖励
        completions_text = []
        for m_outputs in micro_outputs_list:
            for i in range(self.micro_batch_size):
                gen_text = self.tokenizer.decode(m_outputs[i][prompt_len:], skip_special_tokens=True)
                completions_text.append(gen_text)

        rewards_list = self.task.compute_rewards(
            prompts=[prompt] * self.group_size,
            completions=completions_text,
            metadata=[sample] * self.group_size
        )

        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=self.device)
        mean_r = rewards.mean().item()
        max_r = rewards.max().item()
        std_r = rewards.std().item()
        solved_count = sum(1 for r in rewards_list if r >= 1.0)
        best_idx = torch.argmax(rewards).item()

        # 计算整组全部候选的总有效 token 数
        total_active_tokens = sum(m_mask.sum().item() for m_mask in micro_masks_list)

        # ================= 步骤 3: 组内相对优势与 Zero Update Guard =================
        if std_r < 1e-6 and self.kl_coef == 0:
            return {
                "step": step_idx,
                "mean_reward": mean_r,
                "max_reward": max_r,
                "std_reward": std_r,
                "solved_count": solved_count,
                "loss": 0.0,
                "grad_norm": 0.0,
                "avg_len": total_active_tokens / self.group_size,
                "mean_ratio": 1.0,
                "approx_kl": 0.0,
                "clip_frac": 0.0,
                "skipped_zero_std": True,
                "best_completion": completions_text[best_idx],
                "sample_meta": sample
            }

        # 标准化组内相对优势
        advantages = torch.zeros_like(rewards) if std_r < 1e-6 else (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # ================= 步骤 4: PPO-Clip 策略梯度与反向传播 =================
        self.model.train()
        self.model.config.use_cache = False  # 反传阶段关闭 use_cache
        self.optimizer.zero_grad()

        total_loss_val = 0.0
        total_kl_val = 0.0
        total_clip_frac = 0.0
        total_ratio_val = 0.0

        for m_idx in range(self.num_micro_batches):
            m_outputs = micro_outputs_list[m_idx]
            m_mask = micro_masks_list[m_idx]
            m_old_logprobs = micro_old_logprobs_list[m_idx]
            m_ref_logprobs = micro_ref_logprobs_list[m_idx]
            m_adv = advantages[m_idx * self.micro_batch_size : (m_idx + 1) * self.micro_batch_size]

            # 当前策略前向计算 (温度保持与采样一致)
            m_logits = self.model(input_ids=m_outputs).logits
            m_cur_logprobs = compute_completion_logprobs(m_logits, m_outputs, prompt_len, temperature=self.temperature)
            del m_logits

            # 计算 GRPO 损失 (PPO-Clip + Schulman k3 参考基座 KL 惩罚)
            m_loss, m_metrics = compute_grpo_loss(
                cur_logprobs=m_cur_logprobs,
                old_logprobs=m_old_logprobs,
                ref_logprobs=m_ref_logprobs,
                advantages=m_adv,
                mask=m_mask,
                clip_eps=self.clip_eps,
                kl_coef=self.kl_coef
            )

            # 动态 Token 级归一化 (分母为当前题目全组有效 token 数 total_active_tokens，确保组内微批次权重无偏)
            m_norm_loss = (m_loss * m_mask).sum() / (total_active_tokens + 1e-8)
            m_norm_loss.backward()

            total_loss_val += (m_loss * m_mask).sum().item()
            total_kl_val += m_metrics["approx_kl"]
            total_clip_frac += m_metrics["clip_frac"]
            total_ratio_val += m_metrics["mean_ratio"]

        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, error_if_nonfinite=True).item()
        self.optimizer.step()

        return {
            "step": step_idx,
            "mean_reward": mean_r,
            "max_reward": max_r,
            "std_reward": std_r,
            "solved_count": solved_count,
            "loss": total_loss_val / (total_active_tokens + 1e-8),
            "grad_norm": grad_norm,
            "avg_len": total_active_tokens / self.group_size,
            "mean_ratio": total_ratio_val / self.num_micro_batches,
            "approx_kl": total_kl_val / self.num_micro_batches,
            "clip_frac": total_clip_frac / self.num_micro_batches,
            "skipped_zero_std": False,
            "best_completion": completions_text[best_idx],
            "sample_meta": sample
        }
