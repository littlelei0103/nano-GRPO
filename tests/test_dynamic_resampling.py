"""
tests/test_dynamic_resampling.py - DAPO 动态零方差重采样机制单元测试
"""

import unittest
from unittest.mock import MagicMock, patch
import torch
from trainer.mimo_engine import MiMoGRPOTrainer


class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size=32):
        super().__init__()
        self.config = MagicMock()
        self.embed = torch.nn.Embedding(vocab_size, 16)
        self.linear = torch.nn.Linear(16, vocab_size)

    def forward(self, input_ids, **kwargs):
        x = self.embed(input_ids)
        logits = self.linear(x)
        return MagicMock(logits=logits)


from transformers.tokenization_utils_base import BatchEncoding


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text, return_tensors="pt"):
        return BatchEncoding({
            "input_ids": torch.tensor([[1, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        })

    def decode(self, token_ids, skip_special_tokens=True):
        return "<input>7</input>"


class TestDynamicResampling(unittest.TestCase):
    """测试 DAPO 范式下的动态零方差重采样与梯度拯救"""

    def setUp(self):
        self.model = DummyModel()
        self.tokenizer = DummyTokenizer()
        self.prob = {
            "id": "mock_prob_1",
            "prompt": "mock prompt",
            "domain": [0, 50],
            "counterexamples": [7]
        }
        self.task = MagicMock()
        self.task.train_problems = [self.prob]
        self.task.sample_idx = 0

    def test_dynamic_resampling_disabled_by_default(self):
        trainer = MiMoGRPOTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            task=self.task,
            group_size=4,
            micro_batch_size=2,
            problems_per_update=1,
            dynamic_resampling=False,
            device="cpu"
        )
        self.assertFalse(trainer.dynamic_resampling)

    def test_nonzero_temperature_boost_is_rejected(self):
        for boost in (0.15, -0.1, float('nan')):
            with self.subTest(boost=boost), self.assertRaisesRegex(ValueError, 'resample_temp_boost'):
                MiMoGRPOTrainer(self.model, self.tokenizer, self.task,
                                resample_temp_boost=boost, device='cpu')

    def test_actual_logprobs_and_all_rollout_tokens_with_replacement(self):
        # Real forward/logprob/loss path: two initial batches, failed retry,
        # then successful retry of different length replacing batch two.
        from trainer.grpo_loss import compute_grpo_loss
        trainer = MiMoGRPOTrainer(self.model, self.tokenizer, self.task,
                                 dynamic_resampling=True, device='cpu', group_size=4,
                                 micro_batch_size=2, max_resample_rounds=2)
        lengths = [5, 4, 2, 3]
        outputs = [torch.tensor([[1, 3, 4] + [7] * (n - 1) + [2]] * 2)
                   for n in lengths]
        evaluations = []
        for i in range(4):
            rewards = [1.0, 0.0] if i == 3 else [0.0, 0.0]
            evaluations.append((['result'] * 2, [dict(self.prob)] * 2, rewards,
                                [{'category': 'success' if r else 'missed_bug', 'reward': r}
                                 for r in rewards]))

        def check_loss(**kwargs):
            torch.testing.assert_close(kwargs['cur_logprobs'], kwargs['old_logprobs'])
            torch.testing.assert_close(kwargs['cur_logprobs'], kwargs['ref_logprobs'])
            return compute_grpo_loss(**kwargs)

        with patch('trainer.mimo_engine.generate_with_policy', side_effect=outputs) as generation, \
             patch.object(trainer, '_eval_micro_batch', side_effect=evaluations), \
             patch('trainer.mimo_engine.compute_grpo_loss', side_effect=check_loss):
            stats = trainer.train_global_step(1)

        self.assertEqual(generation.call_count, 4)
        for call in generation.call_args_list:
            self.assertIs(call.kwargs['generation_config'], trainer.generation_config)
        self.assertEqual(stats['total_tokens_generated'], 28)
        self.assertEqual(stats['retained_tokens_generated'], 16)
        self.assertEqual(stats['total_rollout_candidates'], 8)
        self.assertEqual(stats['initial_pass_rate'], 0)
        self.assertEqual(stats['overall_rollout_pass_rate'], 1 / 8)
        self.assertEqual(stats['anchor_pass_rate'], 1 / 4)
        self.assertEqual(stats['zero_variance_rescued'], 1)

    def test_zero_variance_rescued_when_counterexample_discovered(self):
        trainer = MiMoGRPOTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            task=self.task,
            group_size=4,
            micro_batch_size=2,
            problems_per_update=1,
            dynamic_resampling=True,
            max_resample_rounds=2,
            device="cpu"
        )

        call_count = [0]
        def mock_eval_micro_batch(m_outputs, prompt_len, prob):
            call_count[0] += 1
            # 初始 2 个 micro-batch (4 个候选) 全错 (0.0)
            if call_count[0] <= 2:
                return (
                    ["<input>0</input>", "<input>0</input>"],
                    [dict(prob), dict(prob)],
                    [0.0, 0.0],
                    [{"category": "missed_bug", "reward": 0.0}, {"category": "missed_bug", "reward": 0.0}]
                )
            else:
                # 动态重采样第 1 轮命中反例！返回一个 1.0
                return (
                    ["<input>7</input>", "<input>0</input>"],
                    [dict(prob), dict(prob)],
                    [1.0, 0.0],
                    [{"category": "success", "reward": 1.0}, {"category": "missed_bug", "reward": 0.0}]
                )

        trainer._eval_micro_batch = mock_eval_micro_batch

        # Mock sample micro batch to return valid dummy outputs
        dummy_outputs = torch.tensor([[1, 3, 4, 7, 2], [1, 3, 4, 0, 2]])
        dummy_mask = torch.tensor([[True, True], [True, True]])
        dummy_lp = torch.tensor([[-0.5, -0.5], [-0.5, -0.5]])
        trainer._sample_micro_batch = MagicMock(return_value=(dummy_outputs, dummy_mask, dummy_lp, dummy_lp))

        stats = trainer.train_global_step(1)

        # 验证该题目被成功挽救为有效方差梯度组
        self.assertEqual(stats["zero_variance_rescued"], 1)
        self.assertEqual(stats["effective_problems"], 1)
        self.assertEqual(stats["all_zero_problems"], 0)
        self.assertEqual(stats["total_solved"], 1)
        # 初始 2 次 + 重采样 1 次 = 3 次 eval 调用
        self.assertEqual(call_count[0], 3)

        # 验证指标口径精确解耦 (原始未重采样成功率 vs 最终保留组成功率 vs 全量采样成功率)
        self.assertEqual(stats["initial_pass_rate"], 0.0)  # 初始 0/4
        self.assertEqual(stats["anchor_pass_rate"], 0.25) # 保留组 1/4
        self.assertEqual(stats["total_rollout_candidates"], 6)  # 初始 4 + 额外重采样 2
        self.assertEqual(stats["total_rollout_solved"], 1)      # 发现 1 个反例
        self.assertAlmostEqual(stats["overall_rollout_pass_rate"], 1.0 / 6.0)
        self.assertEqual(stats["total_tokens_generated"], 12)
        self.assertEqual(stats["retained_tokens_generated"], 8)

    def test_zero_variance_not_rescued_if_budget_exhausted(self):
        trainer = MiMoGRPOTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            task=self.task,
            group_size=4,
            micro_batch_size=2,
            problems_per_update=1,
            dynamic_resampling=True,
            max_resample_rounds=2,
            device="cpu"
        )

        call_count = [0]
        def mock_eval_micro_batch(m_outputs, prompt_len, prob):
            call_count[0] += 1
            # 始终全错 (0.0)
            return (
                ["<input>0</input>", "<input>0</input>"],
                [dict(prob), dict(prob)],
                [0.0, 0.0],
                [{"category": "missed_bug", "reward": 0.0}, {"category": "missed_bug", "reward": 0.0}]
            )

        trainer._eval_micro_batch = mock_eval_micro_batch

        dummy_outputs = torch.tensor([[1, 3, 4, 0, 2], [1, 3, 4, 0, 2]])
        dummy_mask = torch.tensor([[True, True], [True, True]])
        dummy_lp = torch.tensor([[-0.5, -0.5], [-0.5, -0.5]])
        trainer._sample_micro_batch = MagicMock(return_value=(dummy_outputs, dummy_mask, dummy_lp, dummy_lp))

        stats = trainer.train_global_step(1)

        # 达到上限仍全错，未被挽救
        self.assertEqual(stats["zero_variance_rescued"], 0)
        self.assertEqual(stats["effective_problems"], 0)
        self.assertEqual(stats["all_zero_problems"], 1)
        self.assertEqual(stats["resample_attempts"], 2)
        # 初始 2 次 + 重采样 2 次 = 4 次 eval 调用
        self.assertEqual(call_count[0], 4)
        self.assertEqual(stats["total_rollout_candidates"], 8)  # 初始 4 + 额外重采样 4
        self.assertEqual(stats["total_rollout_solved"], 0)
        self.assertEqual(stats["total_tokens_generated"], 16)
        self.assertEqual(stats["retained_tokens_generated"], 8)

    def test_resume_rejects_altered_resampling_options(self):
        """测试续训时若修改 dynamic_resampling 等配置，必须严格拒绝续训"""
        from trainer.experiment import prepare_run
        import tempfile
        import os
        import json

        with tempfile.TemporaryDirectory() as root:
            run_dir = os.path.join(root, "test_run")
            os.makedirs(run_dir, exist_ok=True)
            manifest = {
                "run_id": "test_run",
                "run_dir": run_dir,
                "data_sha256": "mock_hash",
                "config": {
                    "model_name": "dummy", "training_mode": "macro", "task": "mock",
                    "anchor_size": 1, "group_size": 4, "lr": 2e-5, "temperature": 0.7, "top_p": 1.0,
                    "max_new_tokens": 10, "kl_coef": 0.04, "clip_eps": 0.2, "seed": 42,
                    "problems_per_update": 1, "dynamic_resampling": False,
                    "max_resample_rounds": 2, "resample_temp_boost": 0.0
                }
            }
            with open(os.path.join(run_dir, "run_manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f)
            # Create a mock checkpoint directory
            ckpt_dir = os.path.join(run_dir, "checkpoint-1")
            os.makedirs(ckpt_dir, exist_ok=True)
            with open(os.path.join(ckpt_dir, "checkpoint_meta.json"), "w", encoding="utf-8") as f:
                json.dump({"run_context": {"run_id": "test_run"}}, f)

            # 试图在续训时偷偷开启 dynamic_resampling
            new_cfg = dict(manifest["config"])
            new_cfg["dynamic_resampling"] = True

            with patch("trainer.experiment.task_fingerprint", return_value="mock_hash"):
                with self.assertRaises(ValueError) as cm:
                    prepare_run(root, new_cfg, self.task, resume_checkpoint=ckpt_dir)
                self.assertIn("dynamic_resampling", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
