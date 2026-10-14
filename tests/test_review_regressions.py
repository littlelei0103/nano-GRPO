import copy
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import torch
from tasks.bug_hunter import BugHunterTask, extract_call_args_and_expected, verify_bug_exposure
from tasks.code_exec import execute_test_assertions
from tasks.contracts import CONTRACTS, validate_input
from trainer.grpo_loss import (compute_completion_logprobs, compute_grpo_loss,
                               rollout_generation_config, generate_with_policy)
from trainer.experiment import (prepare_run, initial_selection, select_checkpoint,
                                best_checkpoint_path, validate_checkpoint)
from trainer.utils import save_checkpoint, load_training_state


class RewardRegressionTests(unittest.TestCase):
    def test_frame_and_builtin_oracle_attacks(self):
        attacks = [
            "from sys import _getframe\n_getframe(1).f_globals['_exec'] = lambda *args: None\ndef solution(): return 0",
            "import builtins\nbuiltins.sum = lambda *args: 0\ndef solution(): return 0",
            "def solution(): return 0\nexec = lambda *args: None",
            "def solution(): return 0\nprint('__ALL_TESTS_VERIFIED_SUCCESSFULLY__')\nraise SystemExit(0)",
            "def solution(): return 0\nx = '{0.__class__.__mro__}'.format(1)",
            "from math import sin as __builtins__\ndef solution(): return 0",
            "g = (x for x in [1])\ng.gi_frame.f_globals['exec'] = lambda *args: None\ndef solution(): return 0",
        ]
        for source in attacks:
            with self.subTest(source=source):
                self.assertFalse(execute_test_assertions(source, ["assert solution() == sum([40, 2])"]))

    def test_worker_keeps_ordinary_algorithms_and_test_operators(self):
        source = "from collections import Counter\ndef solution(xs):\n    print('ignored user output')\n    return dict(Counter(xs))"
        self.assertTrue(execute_test_assertions(source, ["assert solution([1, 1, 2]) == {1: 2, 2: 1}"]))
        self.assertTrue(execute_test_assertions("def solution(xs): return sorted(xs)", ["assert solution([3, 1, 2])[-1] in [3, 4]"]))
        self.assertTrue(execute_test_assertions("import math\ndef solution(x): return math.gcd(x, 12)", ["assert solution(18) == 6"]))
        self.assertFalse(execute_test_assertions("def solution(): return 0", ["assert solution() == 42"]))
        self.assertFalse(execute_test_assertions("def solution():\n    while True: pass", ["assert solution() == 42"], timeout_sec=.2))

    def test_all_builtin_witnesses_pass_their_input_contract(self):
        with open("tasks/data/bug_hunter_dataset.json", encoding="utf-8") as f:
            data = json.load(f)
        for problem in [p for split in ("train", "val", "test") for p in data[split]]:
            with self.subTest(problem=problem["id"]):
                self.assertIn(problem["id"], CONTRACTS)
                self.assertEqual(problem["input_contract"], problem["id"])
                args, kwargs, _ = extract_call_args_and_expected(problem["counter_example"])
                self.assertTrue(validate_input(problem["id"], problem["golden_code"], args, kwargs))
                self.assertTrue(verify_bug_exposure(problem["counter_example"], problem["golden_code"], problem["buggy_code"],
                                                   precondition=problem.get("precondition"), input_contract=problem["id"]))

    def test_invalid_domains_receive_no_reward(self):
        task = BugHunterTask()
        problems = {p["problem_id"]: p for p in task.train_problems + task.val_problems + task.test_problems}
        cases = {
            "bs_standard_right_edge": "assert target_func([2, 1, 3], 3) == 2",
            "two_sum_unique_pair": "assert target_func([1, 2, 3, 4], 5) == [1, 2]",
            "sw_max_average_subarray_i": "assert target_func([1, 2], 0) == 0",
            "prefix_range_sum_query": "assert target_func([1, 2], -1, 1) == 2",
            "matrix_diagonal_sum": "assert target_func([[1, 2, 3], [4, 5, 6]]) == 6",
            "eval_rpn_negative_division": "assert target_func(['1', '0', '/']) == 0",
            "bs_single_non_duplicate": "assert target_func([2, 1, 1]) == 2",
            "intervals_meeting_rooms": "assert target_func([[5, 1]]) == True",
        }
        for name, assertion in cases.items():
            with self.subTest(problem=name):
                completion = "Check the specification and edge cases.</think><code>" + assertion + "</code>"
                self.assertEqual(task.compute_rewards([""], [completion], [problems[name]]), [0.0])


class ProbabilityRegressionTests(unittest.TestCase):
    def test_zero_reward_variance_keeps_reference_kl_update(self):
        from test_overhaul import TestEndToEndTrainingIntegration
        from trainer.engine import GRPOTrainer
        from trainer.mimo_engine import MiMoGRPOTrainer
        for cls in (GRPOTrainer, MiMoGRPOTrainer):
            with self.subTest(engine=cls.__name__), redirect_stdout(io.StringIO()):
                fixture = TestEndToEndTrainingIntegration()
                fixture.setUp()
                with torch.no_grad():
                    for name, value in fixture.model.named_parameters():
                        if "lora_B" in name:
                            value.fill_(.25)
                fixture.task.compute_rewards = lambda prompts, completions, metadata: [0.] * len(completions)
                trainer = cls(fixture.model, fixture.tokenizer, fixture.task, group_size=4,
                              micro_batch_size=2, max_new_tokens=5, kl_coef=.04, device="cpu")
                stats = trainer.train_step(1) if cls is GRPOTrainer else trainer.train_global_step(1)
                self.assertGreater(stats["approx_kl"], 0)
                self.assertGreater(stats["grad_norm"], 0)

    def test_rollout_scores_match_teacher_forcing_probabilities(self):
        from transformers import AutoConfig, AutoModelForCausalLM
        torch.manual_seed(7)
        model = AutoModelForCausalLM.from_config(AutoConfig.for_model(
            "qwen2", hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, intermediate_size=64, vocab_size=40))
        model.eval()
        model.generation_config.top_p = .2
        model.generation_config.repetition_penalty = 2.0
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=None)
        config = rollout_generation_config(tokenizer, 3, .7)
        ids = torch.tensor([[2, 3]])
        with torch.no_grad():
            config.output_scores = True
            config.return_dict_in_generate = True
            out = generate_with_policy(model, config, input_ids=ids)
            lp = compute_completion_logprobs(model(out.sequences).logits, out.sequences, 2, temperature=.7)
            sampled_lp = torch.stack([score.log_softmax(-1).gather(1, out.sequences[:, 2+i:3+i]).squeeze(1)
                                      for i, score in enumerate(out.scores)], dim=1)
        torch.testing.assert_close(lp, sampled_lp, atol=1e-6, rtol=1e-6)
        self.assertEqual(config.top_p, 1)
        self.assertEqual(config.top_k, 0)
        with self.assertRaises(ValueError):
            rollout_generation_config(tokenizer, 3, .7, top_p=.95)

    def test_large_kl_still_has_restoring_gradient(self):
        current = torch.tensor([[-12.]], requires_grad=True)
        loss, metrics = compute_grpo_loss(current, current.detach(), torch.tensor([[-1.]]),
                                          torch.tensor([0.]), torch.tensor([[True]]))
        loss.sum().backward()
        self.assertGreater(metrics["approx_kl"], 22000)
        self.assertLess(current.grad.item(), 0)

    def test_overflow_is_rejected_and_masked_tokens_do_not_poison_loss(self):
        current = torch.tensor([[-1000., -1.]], requires_grad=True)
        args = (current, current.detach(), torch.tensor([[0., -1.]]), torch.tensor([0.]))
        with self.assertRaises(FloatingPointError):
            compute_grpo_loss(*args, torch.tensor([[True, True]]))
        loss, _ = compute_grpo_loss(*args, torch.tensor([[False, True]]))
        self.assertTrue(torch.isfinite(loss).all())


class CheckpointRegressionTests(unittest.TestCase):
    def setUp(self):
        class Dummy:
            def save_pretrained(self, path):
                pass
        self.dummy = Dummy()
        self.task = SimpleNamespace(train_problems=[{"id": "a"}], get_val_dataset=lambda: [{"id": "b"}])
        self.config = {"model_name": "tiny", "temperature": .7, "top_p": 1., "seed": 42}

    def test_selection_survives_worse_periodic_and_last_checkpoints(self):
        with tempfile.TemporaryDirectory() as root:
            context = prepare_run(root, self.config, self.task)
            selection = initial_selection()
            select_checkpoint(selection, .8, 6)
            best = os.path.join(context["run_dir"], "checkpoint-best")
            save_checkpoint(self.dummy, self.dummy, best, step=6, selection=selection, run_context=context)
            self.assertFalse(select_checkpoint(selection, .4, 10))
            for name, metrics in [("checkpoint-step-10", {"val_acc": .4}), ("checkpoint-last", {"completed_step": 10})]:
                path = os.path.join(context["run_dir"], name)
                save_checkpoint(self.dummy, self.dummy, path, step=10, sample_idx=10,
                                metrics=metrics, selection=selection, run_context=context)
                restored = load_training_state(path)
                self.assertEqual(restored["selection"]["best_val_acc"], .8)
                self.assertEqual(restored["selection"]["best_step"], 6)
                self.assertEqual(best_checkpoint_path(context, restored["selection"]), best)

    def test_stale_run_and_changed_data_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            old = prepare_run(root, self.config, self.task)
            fresh = prepare_run(root, self.config, self.task)
            path = os.path.join(old["run_dir"], "checkpoint-best")
            save_checkpoint(self.dummy, self.dummy, path, step=1, run_context=old)
            with self.assertRaises(ValueError):
                validate_checkpoint(path, fresh)
            self.task.train_problems.append({"id": "changed"})
            with self.assertRaises(ValueError):
                prepare_run(root, self.config, self.task, path)

    def test_continuous_and_resumed_training_are_identical(self):
        from test_overhaul import TestEndToEndTrainingIntegration
        from trainer.engine import GRPOTrainer
        from trainer.mimo_engine import MiMoGRPOTrainer
        for cls in (GRPOTrainer, MiMoGRPOTrainer):
            with self.subTest(engine=cls.__name__), tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()):
                def build():
                    torch.manual_seed(987)
                    fixture = TestEndToEndTrainingIntegration()
                    fixture.setUp()
                    trainer = cls(fixture.model, fixture.tokenizer, fixture.task, group_size=4,
                                  micro_batch_size=2, max_new_tokens=5, device="cpu")
                    return fixture, trainer
                def step(trainer, n):
                    return trainer.train_step(n) if cls is GRPOTrainer else trainer.train_global_step(n)
                uninterrupted, first = build()
                step(first, 1)
                step(first, 2)
                expected = copy.deepcopy(uninterrupted.model.state_dict())
                interrupted, second = build()
                step(second, 1)
                save_checkpoint(interrupted.model, interrupted.tokenizer if hasattr(interrupted.tokenizer, "save_pretrained") else self.dummy,
                                root, second.optimizer, step=1, sample_idx=1)
                resumed_model = copy.deepcopy(interrupted.model)
                resumed = cls(resumed_model, interrupted.tokenizer, interrupted.task, group_size=4,
                              micro_batch_size=2, max_new_tokens=5, device="cpu")
                torch.rand(10)  # Simulate RNG consumed while reloading the model.
                state = load_training_state(root, resumed.optimizer, require_complete=True)
                self.assertEqual(state["sample_idx"], 1)
                step(resumed, state["step"] + 1)
                for name, value in resumed_model.state_dict().items():
                    torch.testing.assert_close(value, expected[name], rtol=0, atol=0)

    def test_both_entrypoints_preserve_best_run_and_sample_cursor(self):
        from test_overhaul import TestEndToEndTrainingIntegration
        import trainer.run as run_module
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        class TinyTask:
            def __init__(self, **kwargs):
                self.sample_idx = 0
                self.train_problems = [{"id": "p1", "prompt": "one"}, {"id": "p2", "prompt": "two"}]
            def get_batch(self, batch_size):
                item = self.train_problems[self.sample_idx % 2]
                self.sample_idx += 1
                return [item]
            def compute_rewards(self, prompts, completions, metadata):
                return [1. if i % 2 == 0 else 0. for i in range(len(completions))]
            def get_val_dataset(self): return [{"id": "val", "prompt": "val"}]
            def get_test_dataset(self): return [{"id": "test", "prompt": "test"}]

        def setup(model_name, device, resume_checkpoint):
            fixture = TestEndToEndTrainingIntegration()
            fixture.setUp()
            fixture.tokenizer.save_pretrained = lambda path: None
            if resume_checkpoint:
                weights = load_file(os.path.join(resume_checkpoint, "adapter_model.safetensors"))
                set_peft_model_state_dict(fixture.model, weights)
            return fixture.tokenizer, fixture.model

        actual_evaluate = run_module.evaluate_benchmark
        def evaluate(model, tokenizer, dataset, task, *args, **kwargs):
            if dataset and dataset[0]["id"] == "val":
                # A baseline better than every trained checkpoint must remain best.
                return {"problem_pass_rate": .8 if task.sample_idx == 0 else .4,
                        "solved_problems": 4, "total_problems": 10}
            return actual_evaluate(model, tokenizer, dataset, task, *args, **kwargs)

        for macro in (False, True):
            with self.subTest(macro=macro), tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()), \
                    patch.object(run_module, "BugHunterTask", TinyTask), \
                    patch.object(run_module, "setup_lora_and_model", setup), \
                    patch.object(run_module, "evaluate_benchmark", evaluate):
                args = SimpleNamespace(model_name="tiny", task="bug_hunter", data_path="", save_dir=root,
                                       resume_checkpoint="", seed=42, top_p=1., temperature=.7, group_size=4,
                                       micro_batch_size=2, lr=2e-5, kl_coef=.04, clip_eps=.2, max_new_tokens=5,
                                       device="cpu", skip_baseline=False, total_steps=1, global_steps=1,
                                       eval_interval=1, anchor_size=2)
                first = run_module.run_training(args, macro=macro)
                run_dir = os.path.dirname(first["blind_test_evaluation"]["checkpoint_last"]["checkpoint"])
                args.resume_checkpoint = os.path.join(run_dir, "checkpoint-last")
                args.total_steps = args.global_steps = 2
                resumed = run_module.run_training(args, macro=macro)
                self.assertEqual(resumed["run_id"], first["run_id"])
                self.assertEqual(resumed["completed_step"], 2)
                self.assertEqual(resumed["validation_selection"]["best_val_acc"], .8)
                self.assertEqual(resumed["validation_selection"]["best_step"], 0)
                self.assertEqual(resumed["blind_test_evaluation"]["checkpoint_best"]["step"], 0)
                state = load_training_state(args.resume_checkpoint)
                self.assertEqual(state["sample_idx"], 2 if not macro else 0)

    def test_failure_categorization(self):
        """测试细粒度失败归因分类器的准确性"""
        from tasks.bug_hunter import BugHunterTask
        task = BugHunterTask()
        prob = {
            "problem_id": "test_prob",
            "spec": "Given a sorted list of unique ints, return first item.",
            "golden_code": "def target_func(arr):\n    assert isinstance(arr, list) and len(arr) > 0\n    return arr[0]",
            "buggy_code": "def target_func(arr):\n    assert isinstance(arr, list) and len(arr) > 0\n    return arr[-1] if len(arr) > 1 else arr[0]",
            "input_contract": None,
            "precondition": "def validate(arr):\n    return isinstance(arr, list) and len(arr) > 0 and arr == sorted(list(set(arr)))"
        }

        # 1. 格式缺失断言与真实截断解耦测试
        res1_format = task.evaluate_single_completion("Here is my thinking...", prob)
        self.assertEqual(res1_format["category"], "format_malformed")
        self.assertEqual(res1_format["reward"], 0.0)

        res1_trunc = task.evaluate_single_completion("Here is my thinking...", dict(prob, is_truncated=True))
        self.assertEqual(res1_trunc["category"], "truncated")
        self.assertEqual(res1_trunc["reward"], 0.0)


        # 2. 违反前置条件 (未排序数组)
        res2 = task.evaluate_single_completion(
            "<think> Let us test an unsorted array. </think>\n<code>assert target_func([3, 1, 2]) == 3</code>",
            prob
        )
        self.assertEqual(res2["category"], "precondition_violation")
        self.assertEqual(res2["reward"], 0.0)

        # 3. Golden 失败 (预期值错误)
        res3 = task.evaluate_single_completion(
            "<think> Test first element. </think>\n<code>assert target_func([1, 2, 3]) == 999</code>",
            prob
        )
        self.assertEqual(res3["category"], "golden_failed")
        self.assertEqual(res3["reward"], 0.0)

        # 4. 未击穿缺陷 (常规单元素，Buggy 和 Golden 表现一致)
        res4 = task.evaluate_single_completion(
            "<think> Test single element array. </think>\n<code>assert target_func([42]) == 42</code>",
            prob
        )
        self.assertEqual(res4["category"], "missed_bug")
        self.assertEqual(res4["reward"], 0.0)

        # 5. 成功击穿缺陷 (多元素，Buggy 返回最后一个元素，Golden 返回第一个元素)
        res5 = task.evaluate_single_completion(
            "<think> Test multi element array where buggy returns last element. </think>\n<code>assert target_func([1, 2]) == 1</code>",
            prob
        )
        self.assertEqual(res5["category"], "success")
        self.assertEqual(res5["reward"], 1.0)

    def test_problems_per_update_sub_batches(self):
        """测试 problems_per_update 分批更新机制与有效组统计"""
        from test_overhaul import TestEndToEndTrainingIntegration
        from trainer.mimo_engine import MiMoGRPOTrainer

        fixture = TestEndToEndTrainingIntegration()
        fixture.setUp()

        class DummyTask:
            def __init__(self):
                self.train_problems = [
                    {"prompt": f"prompt_{i}", "spec": "s", "golden_code": "def target_func(x): return 1", "buggy_code": "def target_func(x): return 2"}
                    for i in range(4)
                ]
            def compute_rewards_with_details(self, prompts, completions, metadata):
                rewards = [1.0 if i % 2 == 0 else 0.0 for i in range(len(completions))]
                details = [{"reward": r, "category": "success" if r == 1.0 else "golden_failed", "reason": ""} for r in rewards]
                return rewards, details

        dummy_task = DummyTask()

        trainer = MiMoGRPOTrainer(
            model=fixture.model,
            tokenizer=fixture.tokenizer,
            task=dummy_task,
            lr=1e-3,
            group_size=4,
            micro_batch_size=2,
            kl_coef=0.0,
            clip_eps=0.2,
            max_new_tokens=5,
            device="cpu",
            problems_per_update=2
        )

        with redirect_stdout(io.StringIO()):
            stats = trainer.train_global_step(1)

        self.assertEqual(stats["num_updates"], 2)
        self.assertEqual(stats["total_candidates"], 16)
        self.assertEqual(stats["effective_problems"], 4)
        self.assertIn("success", stats["failure_breakdown"])
        self.assertIn("golden_failed", stats["failure_breakdown"])


if __name__ == "__main__":
    unittest.main()

