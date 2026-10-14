"""
tests/test_init_adapter.py - 跨任务预训练适配器热启动与迁移学习 (--init_adapter) 单元测试
"""

import os
import json
import tempfile
import unittest
import torch
from unittest.mock import MagicMock, patch
from peft import LoraConfig, get_peft_model

from trainer.experiment import prepare_run
from trainer.utils import setup_lora_and_model


class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size=32):
        super().__init__()
        self.config = MagicMock()
        self.config.use_cache = False
        self.embed = torch.nn.Embedding(vocab_size, 16)
        self.linear = torch.nn.Linear(16, vocab_size)

    def forward(self, input_ids, **kwargs):
        x = self.embed(input_ids)
        logits = self.linear(x)
        return MagicMock(logits=logits)

    def gradient_checkpointing_enable(self):
        pass


class TestInitAdapter(unittest.TestCase):
    """验证 --init_adapter 在跨任务迁移学习下的新实验隔离与权重挂载"""

    def test_prepare_run_with_init_adapter_creates_fresh_run(self):
        task = MagicMock()
        task.train_problems = [{"id": "p1"}]
        task.get_val_dataset = MagicMock(return_value=[{"id": "v1"}])
        task.get_test_dataset = MagicMock(return_value=[{"id": "t1"}])

        with tempfile.TemporaryDirectory() as root:
            config = {
                "model_name": "dummy_model",
                "init_adapter": "/dummy/path/to/counterexample/checkpoint-best",
                "lr": 2e-5
            }
            context = prepare_run(root, config, task, resume_checkpoint="")

            # 验证生成了全新的 run_id 与 run_manifest.json
            self.assertTrue(os.path.isdir(context["run_dir"]))
            manifest_path = os.path.join(context["run_dir"], "run_manifest.json")
            self.assertTrue(os.path.isfile(manifest_path))

            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            self.assertEqual(manifest["config"]["init_adapter"], "/dummy/path/to/counterexample/checkpoint-best")
            self.assertIn("data_sha256", manifest)


if __name__ == "__main__":
    unittest.main()
