"""
trainer/mimo_engine.py - 基于宏步梯度累积的可验证奖励强化学习引擎 (Macro-Batch RLVR Engine)

核心机制 (对齐 Xiaomi MiMo 训练范式与 Token-normalized GRPO 优化理论):
  1. 锚点环境遍历 (Anchor Pool Traversal):
     在单个 Global Step 中遍历固定的代表性任务池 (如 60 道高质量测试样例)。
  2. 双向沙盒执行验证与组内相对优势:
     针对每个题目并发生成 G 路候选序列，由沙盒环境执行并判定客观奖励，计算组内标准化优势 A_i。
  3. 基座参考模型 KL 散度约束:
     利用 PEFT disable_adapter 机制零显存冗余获取参考策略分布 ref_logprobs，实施 Schulman k3 正则化抑制 Policy Drift。
  4. 宏步梯度累积与子批次参数更新:
     支持配置 problems_per_update，在题目子批次内完成多样本梯度累积后执行单步更新，兼顾更新频次与全池探索覆盖率。
  5. 严格 Token 级与题目级两级归一化:
     反向传播损失先在各题目序列有效 Token 上归一化，再按子批次题目数做均匀平均，确保损失与梯度对微批次大小严格不变。
"""

import os
import random
from collections import defaultdict
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


class MiMoGRPOTrainer:
    """
    小米 MiMo 宏步强化学习训练器 (Macro-Batch GRPO)。
    支持按 problems_per_update 分批累积与更新参数，保留全量覆盖与宏步两级归一化。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        task: Any,
        lr: float = 2e-5,
        group_size: int = 4,
        micro_batch_size: int = 2,
        kl_coef: float = 0.04,
        clip_eps: float = 0.2,
        max_new_tokens: int = 350,
        temperature: float = 0.7,
        top_p: float = 1.0,
        device: str = "cuda",
        problems_per_update: int = 12,
        seed: int = 42,
        dynamic_resampling: bool = False,
        max_resample_rounds: int = 2,
        resample_temp_boost: float = 0.0
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.task = task

        self.lr = lr
        self.group_size = group_size
        self.micro_batch_size = min(micro_batch_size, group_size)
        self.kl_coef = kl_coef
        self.clip_eps = clip_eps
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.device = device
        self.problems_per_update = max(1, problems_per_update)
        self.seed = seed
        self.dynamic_resampling = dynamic_resampling
        self.max_resample_rounds = max(1, max_resample_rounds)
        if resample_temp_boost != 0.0:
            raise ValueError("resample_temp_boost must be 0: resampling uses the same policy temperature")
        self.resample_temp_boost = resample_temp_boost

        self.generation_config = rollout_generation_config(tokenizer, max_new_tokens, temperature, top_p)
        if group_size < 2 or micro_batch_size < 1:
            raise ValueError("group_size must be >= 2 and micro_batch_size >= 1")
        self.device = device

        assert self.group_size % self.micro_batch_size == 0, (
            f"group_size ({self.group_size}) 必须被 micro_batch_size ({self.micro_batch_size}) 整除！"
        )
        self.num_micro_batches = self.group_size // self.micro_batch_size

        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr
        )

    def _sample_micro_batch(self, prompt_inputs, prompt_len, gen_config):
        micro_prompt_ids = prompt_inputs["input_ids"].repeat(self.micro_batch_size, 1)
        micro_attention_mask = prompt_inputs["attention_mask"].repeat(self.micro_batch_size, 1)

        m_outputs = generate_with_policy(
            self.model,
            input_ids=micro_prompt_ids,
            attention_mask=micro_attention_mask,
            generation_config=gen_config,
        )
        m_mask = compute_completion_mask(
            m_outputs,
            prompt_len=prompt_len,
            eos_token_id=self.tokenizer.eos_token_id,
            device=self.device
        )
        actual_temp = getattr(gen_config, "temperature", self.temperature)
        m_logits = self.model(input_ids=m_outputs).logits
        m_old_lp = compute_completion_logprobs(m_logits, m_outputs, prompt_len, temperature=actual_temp)
        del m_logits

        if self.kl_coef > 0.0:
            m_ref_lp = get_reference_logprobs(self.model, m_outputs, prompt_len, temperature=actual_temp)
        else:
            m_ref_lp = None

        return m_outputs, m_mask, m_old_lp, m_ref_lp

    def _eval_micro_batch(self, m_outputs, prompt_len, prob):
        completions_text = []
        candidate_metas = []
        for i in range(self.micro_batch_size):
            comp_ids = m_outputs[i][prompt_len:]
            gen_text = self.tokenizer.decode(comp_ids, skip_special_tokens=True)
            completions_text.append(gen_text)

            hit_eos = (self.tokenizer.eos_token_id is not None) and (self.tokenizer.eos_token_id in comp_ids)
            is_truncated = (not hit_eos) and (len(comp_ids) >= self.max_new_tokens)

            meta_i = dict(prob)
            meta_i["is_truncated"] = is_truncated
            candidate_metas.append(meta_i)

        if hasattr(self.task, "compute_rewards_with_details"):
            rewards_list, details_list = self.task.compute_rewards_with_details(
                prompts=[prob["prompt"]] * self.micro_batch_size,
                completions=completions_text,
                metadata=candidate_metas
            )
        else:
            rewards_list = self.task.compute_rewards(
                prompts=[prob["prompt"]] * self.micro_batch_size,
                completions=completions_text,
                metadata=candidate_metas
            )
            details_list = [{"category": "success" if r >= 1.0 else "failed", "reward": r} for r in rewards_list]

        return completions_text, candidate_metas, rewards_list, details_list

    def train_global_step(self, global_step_idx: int) -> Dict[str, Any]:
        """
        执行单个 MiMo 全局宏步 (按 problems_per_update 分批更新参数):
          1. 每轮打乱全部锚点池题目 (保证题目覆盖率并引入随机排列多样性)
          2. 将题目切分为大小为 problems_per_update 的子批次
          3. 每个子批次内累积各题梯度，完成后执行 optimizer.step() 与 zero_grad()
          4. 详细记录每道题的失败分类 (format_or_truncated, precondition_violation, golden_failed, missed_bug)
          5. 细粒度统计有效策略梯度组 (0 < solved < G) 与零方差组
        """
        training_problems = list(self.task.train_problems)
        num_problems = len(training_problems)
        chunk_size = min(self.problems_per_update, num_problems)

        # 1. 每轮打乱题目，保持全部覆盖但重组批次
        rng = random.Random(self.seed + global_step_idx)
        indices = list(range(num_problems))
        rng.shuffle(indices)
        shuffled_problems = [training_problems[i] for i in indices]

        chunks = [shuffled_problems[i : i + chunk_size] for i in range(0, num_problems, chunk_size)]
        total_updates = len(chunks)

        print(
            f"\n[Global Step {global_step_idx:02d}] 锚点池规模: {num_problems} | "
            f"子批次大小: {chunk_size} | 本轮参数更新次数: {total_updates}",
            flush=True
        )

        total_solved = 0
        total_candidates = num_problems * self.group_size
        total_tokens_generated = 0

        # 分类统计与组方差统计
        failure_breakdown = defaultdict(int)
        category_samples = defaultdict(list)
        effective_problems_count = 0  # 0 < solved < G (方差 > 0)
        all_zero_problems_count = 0   # solved == 0 (全错)
        all_one_problems_count = 0    # solved == G (全对)
        rescued_zero_variance_count = 0
        total_resample_attempts = 0
        total_initial_solved = 0
        total_initial_candidates = 0
        total_resampled_solved = 0
        total_resampled_candidates = 0
        total_tokens_all_rollouts = 0

        grad_norms = []
        chunk_losses = []
        total_kl_accum = 0.0
        total_clip_frac_accum = 0.0
        active_backward_problems = 0
        problem_progress = 0

        for chunk_idx, chunk in enumerate(chunks, 1):
            self.optimizer.zero_grad()
            current_chunk_size = len(chunk)
            chunk_effective_problems = 0
            chunk_rescued = 0

            for prob in chunk:
                problem_progress += 1
                prompt_str = prob["prompt"]
                prompt_inputs = self.tokenizer(prompt_str, return_tensors="pt").to(self.device)
                prompt_len = prompt_inputs["input_ids"].shape[1]

                # 1. 探索生成 (Rollout)
                self.model.eval()
                self.model.config.use_cache = True

                micro_outputs_list = []
                micro_masks_list = []
                micro_old_logprobs_list = []
                micro_ref_logprobs_list = []
                completions_text = []
                candidate_metas = []
                rewards_list = []
                details_list = []

                with torch.no_grad():
                    for _ in range(self.num_micro_batches):
                        m_out, m_mask, m_old_lp, m_ref_lp = self._sample_micro_batch(
                            prompt_inputs, prompt_len, self.generation_config
                        )
                        total_tokens_all_rollouts += int(m_mask.sum().item())
                        m_comps, m_metas, m_rws, m_dets = self._eval_micro_batch(
                            m_out, prompt_len, prob
                        )
                        micro_outputs_list.append(m_out)
                        micro_masks_list.append(m_mask)
                        micro_old_logprobs_list.append(m_old_lp)
                        micro_ref_logprobs_list.append(m_ref_lp)
                        completions_text.extend(m_comps)
                        candidate_metas.extend(m_metas)
                        rewards_list.extend(m_rws)
                        details_list.extend(m_dets)

                    initial_solved = sum(1 for r in rewards_list if r >= 1.0)
                    total_initial_solved += initial_solved
                    total_initial_candidates += len(rewards_list)

                    was_resampled = False
                    if self.dynamic_resampling and all(r < 1.0 for r in rewards_list):
                        # 保持重采样温度与基准策略一致，避免跨温度分布失真
                        resample_cfg = self.generation_config
                        for _ in range(self.max_resample_rounds):
                            total_resample_attempts += 1
                            total_resampled_candidates += self.micro_batch_size
                            r_out, r_mask, r_old_lp, r_ref_lp = self._sample_micro_batch(
                                prompt_inputs, prompt_len, resample_cfg
                            )
                            r_comps, r_metas, r_rws, r_dets = self._eval_micro_batch(
                                r_out, prompt_len, prob
                            )
                            r_solved = sum(1 for r in r_rws if r >= 1.0)
                            total_resampled_solved += r_solved
                            total_tokens_all_rollouts += int(r_mask.sum().item())

                            if any(r >= 1.0 for r in r_rws):
                                # 成功挖掘出反例！替换末尾微批次以注入正样本
                                last_m = self.num_micro_batches - 1
                                micro_outputs_list[last_m] = r_out
                                micro_masks_list[last_m] = r_mask
                                micro_old_logprobs_list[last_m] = r_old_lp
                                micro_ref_logprobs_list[last_m] = r_ref_lp

                                s_idx = last_m * self.micro_batch_size
                                for off in range(self.micro_batch_size):
                                    completions_text[s_idx + off] = r_comps[off]
                                    candidate_metas[s_idx + off] = r_metas[off]
                                    rewards_list[s_idx + off] = r_rws[off]
                                    details_list[s_idx + off] = r_dets[off]

                                was_resampled = True
                                break

                # 记录失败分类详情与样本
                for i, d in enumerate(details_list):
                    cat = d["category"]
                    failure_breakdown[cat] += 1
                    if len(category_samples[cat]) < 3:
                        category_samples[cat].append({
                            "problem_id": prob.get("id", prob.get("problem_id", "")),
                            "category": cat,
                            "reason": d.get("reason", ""),
                            "assertion": d.get("assertion", d.get("extracted_value")),
                            "completion": completions_text[i],
                            "format_compliant": d.get("format_compliant", False),
                            "reward": d.get("reward", 0.0)
                        })

                rewards = torch.tensor(rewards_list, dtype=torch.float32, device=self.device)
                std_r = rewards.std().item()
                solved_in_group = sum(1 for r in rewards_list if r >= 1.0)
                total_solved += solved_in_group

                # 组内方差判定: 仅当替换后实际具有非零方差时才计入成功挽救
                if 0 < solved_in_group < self.group_size and std_r > 1e-6:
                    effective_problems_count += 1
                    chunk_effective_problems += 1
                    if was_resampled:
                        rescued_zero_variance_count += 1
                        chunk_rescued += 1
                elif solved_in_group == 0:
                    all_zero_problems_count += 1
                else:
                    all_one_problems_count += 1

                problem_active_tokens = sum(m_mask.sum().item() for m_mask in micro_masks_list)
                total_tokens_generated += problem_active_tokens

                # 组内方差为 0 且无 KL 时直接跳过反向传播
                if std_r < 1e-6 and self.kl_coef == 0:
                    continue

                active_backward_problems += 1
                advantages = torch.zeros_like(rewards) if std_r < 1e-6 else (rewards - rewards.mean()) / (rewards.std() + 1e-8)

                # 3. PPO-Clip 策略梯度与反向传播
                self.model.train()
                self.model.config.use_cache = False

                prob_loss_sum = 0.0
                for m_idx in range(self.num_micro_batches):
                    m_outputs = micro_outputs_list[m_idx]
                    m_mask = micro_masks_list[m_idx]
                    m_old_logprobs = micro_old_logprobs_list[m_idx]
                    m_ref_logprobs = micro_ref_logprobs_list[m_idx]
                    m_adv = advantages[m_idx * self.micro_batch_size : (m_idx + 1) * self.micro_batch_size]

                    m_logits = self.model(input_ids=m_outputs).logits
                    m_cur_logprobs = compute_completion_logprobs(m_logits, m_outputs, prompt_len, temperature=self.temperature)
                    del m_logits

                    m_loss, m_metrics = compute_grpo_loss(
                        cur_logprobs=m_cur_logprobs,
                        old_logprobs=m_old_logprobs,
                        ref_logprobs=m_ref_logprobs,
                        advantages=m_adv,
                        mask=m_mask,
                        clip_eps=self.clip_eps,
                        kl_coef=self.kl_coef
                    )

                    # 动态归一化: 单题 token 归一化后，再按当前更新子批次题数 current_chunk_size 平均
                    m_norm_loss = (m_loss * m_mask).sum() / (problem_active_tokens + 1e-8)
                    m_update_loss = m_norm_loss / max(1, current_chunk_size)
                    m_update_loss.backward()

                    prob_loss_sum += m_norm_loss.item()
                    total_kl_accum += m_metrics["approx_kl"]
                    total_clip_frac_accum += m_metrics["clip_frac"]

                chunk_losses.append(prob_loss_sum)


            # 当前子批次全部累积完毕，更新一次参数！
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, error_if_nonfinite=True).item()
            self.optimizer.step()
            self.optimizer.zero_grad()
            grad_norms.append(norm)

            rescue_info = f" (零方差挽救: {chunk_rescued})" if self.dynamic_resampling else ""
            print(
                f"   ↳ [Sub-Update {chunk_idx}/{total_updates}] 样本进度: [{problem_progress:02d}/{num_problems}] | "
                f"求解成功率: {total_solved}/{problem_progress * self.group_size} ({total_solved/(problem_progress * self.group_size)*100:.1f}%) | "
                f"有效梯度组: {chunk_effective_problems}/{current_chunk_size}{rescue_info} | "
                f"GradNorm: {norm:.4f}",
                flush=True
            )

        total_rollout_candidates = total_candidates + total_resampled_candidates
        total_rollout_solved = total_initial_solved + total_resampled_solved

        anchor_pass_rate = total_solved / max(1, total_candidates)
        initial_pass_rate = total_initial_solved / max(1, total_initial_candidates)
        overall_rollout_pass_rate = total_rollout_solved / max(1, total_rollout_candidates)
        avg_loss = sum(chunk_losses) / max(1, len(chunk_losses))
        avg_grad = sum(grad_norms) / max(1, len(grad_norms))
        avg_kl = total_kl_accum / max(1, active_backward_problems * self.num_micro_batches)
        avg_clip_frac = total_clip_frac_accum / max(1, active_backward_problems * self.num_micro_batches)

        return {
            "global_step": global_step_idx,
            "anchor_pass_rate": anchor_pass_rate,
            "initial_pass_rate": initial_pass_rate,
            "overall_rollout_pass_rate": overall_rollout_pass_rate,
            "total_solved": total_solved,
            "total_candidates": total_candidates,
            "total_rollout_solved": total_rollout_solved,
            "total_rollout_candidates": total_rollout_candidates,
            "total_tokens_generated": total_tokens_all_rollouts,
            "retained_tokens_generated": total_tokens_generated,
            "active_problems": effective_problems_count,
            "effective_problems": effective_problems_count,
            "all_zero_problems": all_zero_problems_count,
            "all_one_problems": all_one_problems_count,
            "zero_variance_rescued": rescued_zero_variance_count,
            "resample_attempts": total_resample_attempts,
            "effective_problem_ratio": effective_problems_count / max(1, num_problems),
            "num_problems": num_problems,
            "num_updates": total_updates,
            "failure_breakdown": dict(failure_breakdown),
            "avg_loss": avg_loss,
            "grad_norm": avg_grad,
            "avg_len": total_tokens_all_rollouts / max(1, total_rollout_candidates),
            "retained_avg_len": total_tokens_generated / max(1, total_candidates),
            "approx_kl": avg_kl,
            "clip_frac": avg_clip_frac,
            "sample_failures": dict(category_samples)
        }

