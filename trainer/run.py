"""Shared experiment lifecycle: identity, resume, validation selection and test."""
import json
import os
import time

from tasks import BugHunterTask, CodeExecTask
from trainer.engine import GRPOTrainer
from trainer.mimo_engine import MiMoGRPOTrainer
from trainer.evaluation import evaluate_benchmark
from trainer.experiment import (prepare_run, initial_selection, select_checkpoint,
                                best_checkpoint_path)
from trainer.utils import (setup_lora_and_model, save_checkpoint, load_training_state,
                           seed_training, reset_gpu_peak, gpu_peak_metrics)


def run_training(args, macro=False):
    mode = "macro" if macro else "streaming"
    if getattr(args, "resample_temp_boost", 0.0) != 0.0:
        raise ValueError("resample_temp_boost must be 0: resampling uses the same policy temperature")
    if args.model_name == "Qwen/Qwen2.5-1.5B-Base":
        args.model_name = "Qwen/Qwen2.5-1.5B"
    if args.top_p != 1.0 or args.temperature <= 0:
        raise ValueError("Use temperature > 0 and top_p=1.0: training uses a full-support policy")
    if args.group_size < 2 or args.micro_batch_size < 1 or args.group_size % args.micro_batch_size:
        raise ValueError("group_size >= 2 must be divisible by micro_batch_size >= 1")
    if not macro and args.eval_interval < 1:
        raise ValueError("eval_interval must be positive")
    if getattr(args, "problems_per_update", None) is None:
        args.problems_per_update = 12 if macro else 1
    task_name = getattr(args, "task", "counterexample" if macro else "bug_hunter")

    if task_name == "counterexample":
        from tasks.counterexample import CounterexampleTask
        task = CounterexampleTask(data_path=args.data_path or None)
    elif task_name == "bug_hunter":
        task = BugHunterTask(data_path=args.data_path or None)
    elif task_name == "bug_differential":
        from tasks.bug_differential import DifferentialBugHunterTask
        task = DifferentialBugHunterTask(data_path=args.data_path or None)
    elif task_name == "code_exec":
        task = CodeExecTask()
    else:
        raise ValueError(f"Unknown task: {task_name}")

    if macro:
        if not 1 <= args.anchor_size <= len(task.train_problems):
            raise ValueError("anchor_size must be between 1 and the available training problem count")
        task.train_problems = task.train_problems[:args.anchor_size]
    if not task.train_problems:
        raise ValueError("Training dataset is empty")
    config = dict(vars(args), training_mode=mode)
    context = prepare_run(args.save_dir, config, task, args.resume_checkpoint)
    seed_training(args.seed)
    model_kwargs = {}
    if getattr(args, "init_adapter", ""):
        model_kwargs["init_adapter"] = args.init_adapter
    tokenizer, model = setup_lora_and_model(args.model_name, device=args.device,
                                            resume_checkpoint=args.resume_checkpoint,
                                            **model_kwargs)
    trainer_cls = MiMoGRPOTrainer if macro else GRPOTrainer
    trainer_kwargs = dict(
        model=model, tokenizer=tokenizer, task=task, lr=args.lr,
        group_size=args.group_size, micro_batch_size=args.micro_batch_size,
        kl_coef=args.kl_coef, clip_eps=args.clip_eps,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        top_p=args.top_p, device=args.device,
        dynamic_resampling=getattr(args, "dynamic_resampling", False),
        max_resample_rounds=getattr(args, "max_resample_rounds", 2),
        resample_temp_boost=getattr(args, "resample_temp_boost", 0.0)
    )
    if macro:
        trainer_kwargs["problems_per_update"] = getattr(args, "problems_per_update", 12)
        trainer_kwargs["seed"] = getattr(args, "seed", 42)
    trainer = trainer_cls(**trainer_kwargs)
    selection = initial_selection()
    completed = 0
    if args.resume_checkpoint:
        restored = load_training_state(args.resume_checkpoint, trainer.optimizer, require_complete=True)
        if not all(key in restored.get("selection", {}) for key in selection):
            raise ValueError("Checkpoint lacks complete validation selection state")
        selection = restored["selection"]
        completed = restored["step"]
        if restored.get("sample_idx") is not None:
            task.sample_idx = restored["sample_idx"]
        best_checkpoint_path(context, selection)
    end_step = args.global_steps if macro else args.total_steps
    if end_step < completed or end_step < 0:
        raise ValueError("Requested final step precedes the restored step")
    val = task.get_val_dataset()
    test = task.get_test_dataset() if hasattr(task, "get_test_dataset") else []
    run_dir = context["run_dir"]
    print(f"[Run Init] Mode: {mode} | Task: {task_name} | Run directory: {run_dir} | Starting Step: {completed + 1}", flush=True)

    def save(name, step, metrics):
        save_checkpoint(model, tokenizer, os.path.join(run_dir, name), trainer.optimizer,
                        step=step, sample_idx=task.sample_idx, config=config,
                        metrics=metrics, selection=selection, run_context=context)

    if not args.resume_checkpoint and not args.skip_baseline and val:
        baseline = evaluate_benchmark(model, tokenizer, val, task, args.max_new_tokens, args.device)
        selection["base_val_acc"] = baseline["problem_pass_rate"]
        select_checkpoint(selection, baseline["problem_pass_rate"], 0)
        save("checkpoint-best", 0, {"val_acc": baseline["problem_pass_rate"]})
        print(f"[Validation Baseline] Initial pass rate: {baseline['problem_pass_rate']:.1%}", flush=True)

    for step in range(completed + 1, end_step + 1):
        reset_gpu_peak(args.device)
        started = time.monotonic()
        stats = trainer.train_global_step(step) if macro else trainer.train_step(step)
        completed = step
        metrics = {key: value for key, value in stats.items() if key not in ("sample_meta", "best_completion", "sample_failures")}
        metrics.update(run_id=context["run_id"], step=step,
                       step_duration_sec=time.monotonic() - started, **gpu_peak_metrics(args.device))
        if "sample_failures" in stats:
            with open(os.path.join(run_dir, "failure_samples.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "step": step,
                    "failure_breakdown": stats.get("failure_breakdown", {}),
                    "samples": stats["sample_failures"]
                }, ensure_ascii=False) + "\n")
        if (macro or step % args.eval_interval == 0) and val:
            result = evaluate_benchmark(model, tokenizer, val, task, args.max_new_tokens, args.device)
            metrics.update(val_acc=result["problem_pass_rate"], val_corr=result["solved_problems"], val_tot=result["total_problems"])
            if select_checkpoint(selection, result["problem_pass_rate"], step):
                save("checkpoint-best", step, metrics)
            save(f"checkpoint-{'global-step' if macro else 'step'}-{step}", step, metrics)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
        with open(os.path.join(run_dir, "metrics.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")

    save("checkpoint-last", completed, {"completed_step": completed})

    report = {
        "run_id": context["run_id"], "model_name": args.model_name,
        "data_sha256": context["data_sha256"], "completed_step": completed,
        "validation_selection": dict(selection),
        "decoding": {"temperature": args.temperature, "top_p": 1.0, "top_k": 0, "max_new_tokens": args.max_new_tokens, "k": 4},
        "blind_test_evaluation": {},
    }
    if test:
        def evaluate_current():
            return {
                "greedy": evaluate_benchmark(model, tokenizer, test, task, args.max_new_tokens, args.device),
                "sample": evaluate_benchmark(model, tokenizer, test, task, args.max_new_tokens, args.device,
                                               do_sample=True, temperature=args.temperature, samples_per_problem=4),
            }
        results = report["blind_test_evaluation"]
        # Base comparison is performed only after training and uses the actual base weights.
        if not args.skip_baseline:
            with model.disable_adapter():
                results["base"] = evaluate_benchmark(model, tokenizer, test, task, args.max_new_tokens, args.device)
        else:
            results["base"] = None
        results["checkpoint_last"] = dict(evaluate_current(), checkpoint=os.path.join(run_dir, "checkpoint-last"))
        best_path = best_checkpoint_path(context, selection)
        results["checkpoint_best"] = None
        if best_path:
            model.load_adapter(best_path, adapter_name="eval_best")
            model.set_adapter("eval_best")
            try:
                results["checkpoint_best"] = dict(evaluate_current(), checkpoint=best_path, step=selection["best_step"])
            finally:
                model.set_adapter("default")
        print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    with open(os.path.join(run_dir, "final_test_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report
