"""
tests/test_overhaul.py - 全面回归与正确性验证测试套件

涵盖 GPT-6 Astra 审查的 8 大缺陷全面验证:
1. 沙盒安全性与 AST 语法约束 (防 Reward Hacking & SystemExit 绕过)
2. 数据集纯净度与三相隔离 (0 注释泄漏、0 重叠交叉)
3. 真实基座参考模型 KL 约束与梯度传导
4. Dr. GRPO 微批次梯度完全等价性 (Micro-batch Invariance)
5. 检查点持久化与训练状态恢复
"""

import os
import sys
import json
import torch
import unittest
import tempfile
import shutil

# 确保导入 nano-GRPO
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tasks.bug_hunter import (
    BugHunterTask,
    validate_and_parse_assertion,
    verify_bug_exposure,
    extract_assertion
)
from tasks.code_exec import CodeExecTask, execute_test_assertions
from trainer.grpo_loss import (
    compute_completion_mask,
    compute_completion_logprobs,
    get_reference_logprobs,
    compute_grpo_loss
)
from trainer.utils import (
    setup_lora_and_model,
    save_checkpoint,
    load_training_state
)
from trainer.engine import GRPOTrainer
from trainer.mimo_engine import MiMoGRPOTrainer
from transformers import AutoConfig, AutoModelForCausalLM, BatchEncoding
from peft import LoraConfig, get_peft_model


class TestSandboxAndSecurity(unittest.TestCase):
    """测试沙盒安全性与防作弊能力"""

    def test_ast_validation_legit_assertion(self):
        valid_assert = "assert target_func([1, 2, 3], 4) == True"
        is_valid, reason = validate_and_parse_assertion(valid_assert)
        self.assertTrue(is_valid, f"Should be valid: {reason}")

    def test_ast_validation_blocks_hacking(self):
        # 1. 试图访问内部常量或不调用 target_func
        hacks = [
            "assert 1 == 1",
            "assert target_func == target_func",
            "assert True",
            "assert open('/etc/passwd').read() == ''",
            "assert __import__('os').system('ls') == 0",
            "x = 1; assert target_func(x) == 1",  # 多语句
            "raise SystemExit(0)",
            "assert target_func([1]) is not None and eval('1+1') == 2"
        ]
        for h in hacks:
            is_valid, reason = validate_and_parse_assertion(h)
            self.assertFalse(is_valid, f"Expected '{h}' to be blocked, but passed: {reason}")

    def test_system_exit_bypass_prevented(self):
        # 即使模型输出试图执行 SystemExit(0)，沙盒也必须判定失败
        golden = "def target_func(x):\n    return x + 1"
        buggy = "def target_func(x):\n    import sys; sys.exit(0)"

        # 正常输入断言
        assertion = "assert target_func(1) == 2"
        # Golden 会得到 2，Buggy 会触发 sys.exit(0)
        # 沙盒应捕获 sys.exit(0) 为非零 exit code 并判定 buggy 异常，成功捕获漏洞！
        success = verify_bug_exposure(assertion, golden, buggy)
        self.assertTrue(success, "Buggy raising SystemExit(0) should be caught as abnormal exit!")

    def test_code_exec_system_exit_bypass_blocked(self):
        # CodeExecTask 中若代码直接 sys.exit(0) 而未完成测试，应判定为失败
        bad_code = "import sys\ndef solution():\n    sys.exit(0)"
        test_cases = ["assert solution() == 42"]
        success = execute_test_assertions(bad_code, test_cases)
        self.assertFalse(success, "CodeExec must NOT pass on premature sys.exit(0)!")

    def test_ast_advanced_bypass_blocked(self):
        """测试所有 GPT-6 Astra 复现的 AST 穿透路径均被严格拦截"""
        bypass_attempts = [
            # 1. 赋值表达式 NamedExpr (海象操作符)
            "assert (target_func := lambda *a: 1) and target_func() == 1",
            "assert (x := target_func([1])) == [1]",
            # 2. 属性访问 (读取常量、代码对象或内部私有属性)
            "assert target_func([1]) == target_func.__code__.co_consts[0]",
            "assert target_func([1]) == sys._getframe().f_code.co_consts[0]",
            "assert target_func([1]).__class__ == list",
            # 3. 死分支或短路求值逃逸
            "assert True or target_func([1]) == 2",
            "assert 1 == 1 if target_func([1]) else 2 == 2",
            "assert False if False else target_func([1]) == 2",
            # 4. 任意变量引用与推导式
            "assert target_func([x for x in [1, 2]]) == [1]",
            "assert target_func(any_var) == 1",
            # 5. 非 target_func 目标调用
            "assert other_func([1]) == 2",
            "assert len(target_func([1])) == 1",
        ]
        for expr in bypass_attempts:
            is_valid, reason = validate_and_parse_assertion(expr)
            self.assertFalse(is_valid, f"Bypass attempt should be blocked: '{expr}', but passed: {reason}")

    def test_code_exec_advanced_bypass_blocked(self):
        """测试 CodeExec 沙盒对于 exec 篡改、伪造成功标记及 os._exit 的强壮性"""
        # 1. 覆写 exec 试图跳过断言检查
        code_exec_override = "def solution(): return 0\nexec = lambda *args: None"
        self.assertFalse(
            execute_test_assertions(code_exec_override, ["assert solution() == 42"]),
            "Overriding exec must NOT pass verification!"
        )

        # 2. 伪造完成标记并退出
        code_fake_marker = (
            "def solution(): return 0\n"
            "print('__ALL_TESTS_VERIFIED_SUCCESSFULLY__')\n"
            "raise SystemExit(0)"
        )
        self.assertFalse(
            execute_test_assertions(code_fake_marker, ["assert solution() == 42"]),
            "Faking output string with SystemExit(0) must NOT pass verification!"
        )

    def test_precondition_validation(self):
        """测试题目可执行前置条件 (Precondition) 校验"""
        task = BugHunterTask()
        problem = None
        for p in task.train_problems + task.val_problems + task.test_problems:
            if "two_sum" in p["id"]:
                problem = p
                break

        self.assertIsNotNone(problem, "two_sum problem must exist in dataset")
        self.assertIn("precondition", problem)

        # 1. 多解非法输入: [1, 2, 3, 4], 5 (存在 (0,3) 和 (1,2) 两种和为 5 的索引)
        multi_solution_assert = "assert target_func([1, 2, 3, 4], 5) == [1, 2]"
        comp_multi = f"The input has multiple solution pairs violating precondition.\n</think>\n<code>{multi_solution_assert}</code>"
        r_multi = task.compute_rewards(["prompt"], [comp_multi], [problem])[0]
        self.assertEqual(r_multi, 0.0, "Input violating unique pair precondition must get 0.0 reward!")

        # 2. 唯一解合法反例: [3, 2, 4], 6 (仅有 2+4=6 对应索引 [1, 2])
        valid_assert = "assert target_func([3, 2, 4], 6) == [1, 2]"
        comp_valid = f"Analyzing unique pair logic and finding bug.\n</think>\n<code>{valid_assert}</code>"
        r_valid = task.compute_rewards(["prompt"], [comp_valid], [problem])[0]
        self.assertEqual(r_valid, 1.0, "Legitimate bug-exposing input satisfying precondition must get 1.0 reward!")



class TestDatasetIntegrity(unittest.TestCase):
    """测试数据集纯净度与三相隔离"""

    def setUp(self):
        self.dataset_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "tasks", "data", "bug_hunter_dataset.json"
        )

    def test_dataset_splits_and_leakage(self):
        self.assertTrue(os.path.exists(self.dataset_path))
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("train", data)
        self.assertIn("val", data)
        self.assertIn("test", data)

        train_set = data["train"]
        val_set = data["val"]
        test_set = data["test"]

        self.assertEqual(len(train_set), 60)
        self.assertEqual(len(val_set), 20)
        self.assertEqual(len(test_set), 20)

        train_ids = set(p["id"] for p in train_set)
        val_ids = set(p["id"] for p in val_set)
        test_ids = set(p["id"] for p in test_set)

        # 严格零重叠检验
        self.assertEqual(len(train_ids & val_ids), 0, "Train and Val must NOT overlap!")
        self.assertEqual(len(train_ids & test_ids), 0, "Train and Test must NOT overlap!")
        self.assertEqual(len(val_ids & test_ids), 0, "Val and Test must NOT overlap!")

        # 严格零注释泄漏检验
        for split_name, probs in [("train", train_set), ("val", val_set), ("test", test_set)]:
            for p in probs:
                self.assertNotIn("# BUG", p["buggy_code"], f"Leakage found in {p['id']} buggy_code ({split_name})")
                self.assertNotIn("# BUG", p["golden_code"], f"Leakage found in {p['id']} golden_code ({split_name})")


class TestGRPOLossAndMath(unittest.TestCase):
    """测试 GRPO 损失函数、真实基座 KL 及 Dr. GRPO 微批次梯度一致性"""

    def test_completion_mask(self):
        outputs = torch.tensor([
            [10, 20, 100, 2, 999, 999],
            [10, 20, 30, 40, 50, 2]
        ])
        prompt_len = 2
        eos_id = 2
        mask = compute_completion_mask(outputs, prompt_len, eos_id, device=torch.device("cpu"))
        self.assertEqual(mask[0].tolist(), [True, True, False, False])
        self.assertEqual(mask[1].tolist(), [True, True, True, True])

    def test_dr_grpo_micro_batch_invariance(self):
        """验证微批次切分对最终累积梯度的数学不变性 (Dr. GRPO Token 归一化)"""
        torch.manual_seed(12345)
        B, gen_len = 4, 6
        cur_logps = torch.randn(B, gen_len, requires_grad=True)
        old_logps = cur_logps.detach().clone()
        ref_logps = old_logps - 0.2
        advs = torch.tensor([1.2, -0.8, 0.5, -0.9])
        mask = torch.tensor([
            [True, True, True, False, False, False],
            [True, True, True, True, True, False],
            [True, True, False, False, False, False],
            [True, True, True, True, False, False]
        ])
        total_active_tokens = mask.sum().float()

        # 1. 全批次统一反传 (Full Batch)
        full_loss, metrics_full = compute_grpo_loss(
            cur_logps, old_logps, ref_logps, advs, mask, clip_eps=0.2, kl_coef=0.04
        )
        norm_full_loss = (full_loss * mask).sum() / total_active_tokens
        norm_full_loss.backward()
        full_grad = cur_logps.grad.clone()

        # 2. 微批次切分反传 (2 个 micro_batches，每个 B_micro=2)
        cur_logps.grad.zero_()
        for m_idx in range(2):
            m_cur = cur_logps[m_idx*2 : (m_idx+1)*2]
            m_old = old_logps[m_idx*2 : (m_idx+1)*2]
            m_ref = ref_logps[m_idx*2 : (m_idx+1)*2]
            m_adv = advs[m_idx*2 : (m_idx+1)*2]
            m_mask = mask[m_idx*2 : (m_idx+1)*2]

            m_loss, _ = compute_grpo_loss(
                m_cur, m_old, m_ref, m_adv, m_mask, clip_eps=0.2, kl_coef=0.04
            )
            # Dr. GRPO: 统一除以 total_active_tokens
            ( (m_loss * m_mask).sum() / total_active_tokens ).backward()

        micro_grad = cur_logps.grad.clone()
        max_diff = torch.abs(full_grad - micro_grad).max().item()
        self.assertLess(max_diff, 1e-6, f"Gradients diverge across micro-batches: max_diff={max_diff}")

    def test_kl_regularization_active(self):
        """验证当策略偏离基座参考模型时，KL 惩罚项严格为正并提供真实梯度"""
        cur_logps = torch.tensor([[-0.2, -0.5]], requires_grad=True)
        old_logps = cur_logps.detach().clone()
        ref_logps = torch.tensor([[-0.8, -0.1]])  # 不同的基座概率
        advs = torch.tensor([0.0])  # 优势设为 0，单纯测试 KL 项
        mask = torch.tensor([[True, True]])

        loss, metrics = compute_grpo_loss(
            cur_logps, old_logps, ref_logps, advs, mask, clip_eps=0.2, kl_coef=0.05
        )
        self.assertGreater(metrics["approx_kl"], 0.0, "KL must be strictly positive when cur != ref")
        loss.sum().backward()
        self.assertIsNotNone(cur_logps.grad)
        self.assertFalse(torch.all(cur_logps.grad == 0), "KL gradient must NOT be zero")

    def test_autograd_inference_mode_fix(self):
        """验证在 torch.no_grad() 环境下生成的 mask 和 logprobs 不会阻断后续 backward (彻底规避 Inference tensors cannot be saved for backward)"""
        linear = torch.nn.Linear(10, 10)
        x = torch.randn(4, 5, 10)

        # 模拟 rollout 采样阶段: 严格使用 torch.no_grad() 代替 torch.inference_mode()
        with torch.no_grad():
            old_logits = linear(x)
            old_logps = old_logits.log_softmax(dim=-1).max(dim=-1)[0]
            ref_logps = old_logps.clone()
            mask = torch.tensor([[True, True, True, False, False]] * 4)
            # 构造非零方差奖励优势，模拟 [1.0, 0.0] 混合奖励
            advs = torch.tensor([1.0, -1.0, 1.0, -1.0])

        # 模拟训练反传阶段
        cur_logits = linear(x)
        cur_logps = cur_logits.log_softmax(dim=-1).max(dim=-1)[0]

        loss, metrics = compute_grpo_loss(
            cur_logprobs=cur_logps,
            old_logprobs=old_logps,
            ref_logprobs=ref_logps,
            advantages=advs,
            mask=mask,
            clip_eps=0.2,
            kl_coef=0.04
        )
        total_loss = (loss * mask).sum() / mask.sum().float()

        # 必须能够无报错成功执行反向传播并产生梯度
        total_loss.backward()

        self.assertIsNotNone(linear.weight.grad, "Linear weight gradient must not be None!")
        self.assertFalse(torch.all(linear.weight.grad == 0), "Linear weight gradient must not be all zeros!")

    def test_bfloat16_kl_numerical_stability(self):
        """测试在 bfloat16 精度下 KL 惩罚不会因有效数字损失发生数值下溢变为负数"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

        cur = torch.tensor([[0.01]], dtype=dtype, device=device, requires_grad=True)
        old = cur.detach().clone()
        ref = torch.tensor([[0.00]], dtype=dtype, device=device)
        adv = torch.tensor([0.0], dtype=dtype, device=device)
        mask = torch.tensor([[True]], device=device)

        loss, metrics = compute_grpo_loss(
            cur, old, ref, adv, mask, clip_eps=0.2, kl_coef=0.04
        )
        self.assertGreaterEqual(metrics["approx_kl"], 0.0, "approx_kl in BF16 must be non-negative!")
        (loss * mask).sum().backward()
        self.assertFalse(torch.isnan(cur.grad).any(), "Gradient must not be NaN in BF16")


class TestStateCheckpointing(unittest.TestCase):
    """测试完备状态保存与恢复"""

    def test_save_and_load_training_state(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            # 创建虚拟模型与优化器
            linear = torch.nn.Linear(4, 4)
            optimizer = torch.optim.Adam(linear.parameters(), lr=1e-3)
            # 模拟执行一步让优化器产生 state (动量缓冲区)
            out = linear(torch.randn(2, 4))
            out.sum().backward()
            optimizer.step()

            # 保存假模型与状态
            class DummyTokenizer:
                def save_pretrained(self, d):
                    pass
            class DummyModel(torch.nn.Module):
                def __init__(self, lin):
                    super().__init__()
                    self.lin = lin
                def save_pretrained(self, d):
                    torch.save(self.lin.state_dict(), os.path.join(d, "weights.pt"))

            dummy_model = DummyModel(linear)
            save_checkpoint(
                model=dummy_model,
                tokenizer=DummyTokenizer(),
                save_dir=tmp_dir,
                optimizer=optimizer,
                step=42,
                config={"lr": 1e-3},
                metrics={"val_acc": 0.85}
            )

            # 验证文件齐全
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "training_state.pt")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "optimizer.pt")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "checkpoint_meta.json")))

            # 恢复状态
            new_linear = torch.nn.Linear(4, 4)
            new_opt = torch.optim.Adam(new_linear.parameters(), lr=1e-3)
            restored = load_training_state(tmp_dir, optimizer=new_opt)

            self.assertEqual(restored.get("step"), 42)
            # 确认优化器 state 已加载
            self.assertGreater(len(new_opt.state_dict()["state"]), 0)

        finally:
            shutil.rmtree(tmp_dir)

    def test_save_and_load_training_state_with_sample_idx(self):
        """测试保存与恢复训练状态时 sample_idx 与随机数状态完全保真"""
        tmp_dir = tempfile.mkdtemp()
        try:
            linear = torch.nn.Linear(4, 4)
            opt = torch.optim.Adam(linear.parameters(), lr=1e-3)

            class DummyModel(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m
                def save_pretrained(self, d):
                    pass
            class DummyTok:
                def save_pretrained(self, d):
                    pass

            save_checkpoint(
                model=DummyModel(linear),
                tokenizer=DummyTok(),
                save_dir=tmp_dir,
                optimizer=opt,
                step=15,
                sample_idx=47,
                config={"lr": 1e-3},
                metrics={"val_acc": 0.40}
            )

            new_opt = torch.optim.Adam(linear.parameters(), lr=1e-3)
            restored = load_training_state(tmp_dir, optimizer=new_opt)
            self.assertEqual(restored.get("step"), 15)
            self.assertEqual(restored.get("sample_idx"), 47)
            self.assertEqual(restored.get("metrics", {}).get("val_acc"), 0.40)
        finally:
            shutil.rmtree(tmp_dir)


class TestEndToEndTrainingIntegration(unittest.TestCase):
    """测试完整端到端训练步 (GRPOTrainer 与 MiMoGRPOTrainer)，验证混合奖励下 Autograd 反向传播与非零梯度更新"""

    def setUp(self):
        cfg = AutoConfig.for_model(
            "qwen2",
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            intermediate_size=128,
            vocab_size=1000
        )
        base_model = AutoModelForCausalLM.from_config(cfg)
        lora_config = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
        self.model = get_peft_model(base_model, lora_config)

        class MockTokenizer:
            pad_token = "<pad>"
            pad_token_id = 0
            eos_token = "<eos>"
            eos_token_id = 1
            padding_side = "left"

            def __call__(self, text, return_tensors="pt", padding=True, truncation=True, max_length=512):
                if isinstance(text, str):
                    text = [text]
                return BatchEncoding({
                    "input_ids": torch.tensor([[10, 20, 30]] * len(text)),
                    "attention_mask": torch.tensor([[1, 1, 1]] * len(text))
                })

            def decode(self, token_ids, skip_special_tokens=True):
                return "test completion"

        class MockTask:
            def __init__(self):
                self.sample_idx = 0
                self.train_problems = [{"prompt": "test prompt", "id": "p1"}]

            def get_batch(self, batch_size=1):
                return [{"prompt": "test prompt", "id": f"p_{self.sample_idx}"}]

            def compute_rewards(self, prompts, completions, metadata):
                # 强制制造非零方差奖励: 一半 1.0, 一半 0.0，杜绝零方差跳过
                return [1.0 if i % 2 == 0 else 0.0 for i in range(len(completions))]

        self.tokenizer = MockTokenizer()
        self.task = MockTask()

    def test_grpo_trainer_backward_and_update(self):
        trainer = GRPOTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            task=self.task,
            group_size=4,
            micro_batch_size=2,
            device="cpu",
            max_new_tokens=5
        )
        stats = trainer.train_step(step_idx=1)
        self.assertFalse(stats["skipped_zero_std"], "Training step should not skip when rewards have variance")
        self.assertGreater(stats["grad_norm"], 0.0, "Gradient norm must be > 0 on valid step")

    def test_mimo_trainer_backward_and_update(self):
        trainer = MiMoGRPOTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            task=self.task,
            group_size=4,
            micro_batch_size=2,
            device="cpu",
            max_new_tokens=5
        )
        stats = trainer.train_global_step(global_step_idx=1)
        self.assertGreater(stats["active_problems"], 0, "active_problems must be > 0 with mixed rewards")
        self.assertGreater(stats["grad_norm"], 0.0, "MiMo gradient norm must be > 0 on valid macro-step")


if __name__ == "__main__":
    unittest.main()
