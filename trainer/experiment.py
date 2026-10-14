"""Run identity and validation selection shared by both training entry points."""
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone


RESUME_CONFIG_KEYS = (
    "model_name", "training_mode", "task", "anchor_size", "group_size", "lr", "temperature", "top_p",
    "max_new_tokens", "kl_coef", "clip_eps", "seed", "problems_per_update",
    "dynamic_resampling", "max_resample_rounds", "resample_temp_boost",
)


def task_fingerprint(task):
    datasets = {
        "train": task.train_problems,
        "val": task.get_val_dataset(),
        "test": task.get_test_dataset() if hasattr(task, "get_test_dataset") else [],
    }
    root = os.path.dirname(os.path.dirname(__file__))
    # Registry IDs alone do not identify the executable input/reward contracts.
    datasets["reward_implementation"] = {}
    for name in ("contracts.py", "bug_hunter.py", "bug_differential.py", "code_exec.py", "code_worker.py", "counterexample.py", "semantic_values.py"):
        fpath = os.path.join(root, "tasks", name)
        if os.path.isfile(fpath):
            with open(fpath, "rb") as f:
                datasets["reward_implementation"][name] = hashlib.sha256(f.read()).hexdigest()
    encoded = json.dumps(datasets, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def prepare_run(save_root, config, task, resume_checkpoint=""):
    data_hash = task_fingerprint(task)
    if resume_checkpoint:
        run_dir = os.path.dirname(os.path.abspath(resume_checkpoint))
        path = os.path.join(run_dir, "run_manifest.json")
        if not os.path.isfile(path):
            raise ValueError("Resume requires a complete checkpoint from a run with run_manifest.json")
        with open(path, encoding="utf-8") as f:
            context = json.load(f)
        if context["data_sha256"] != data_hash:
            raise ValueError("Cannot resume: dataset, split or input contracts changed")
        for key in RESUME_CONFIG_KEYS:
            expected_val = context["config"].get(key)
            current_val = config.get(key)
            if expected_val is None and key == "problems_per_update":
                expected_val = 12 if context["config"].get("training_mode") == "macro" else 1
            if current_val is None and key == "problems_per_update":
                current_val = 12 if config.get("training_mode") == "macro" else 1
            if expected_val is None and key == "dynamic_resampling":
                expected_val = False
            if current_val is None and key == "dynamic_resampling":
                current_val = False
            if expected_val is None and key == "max_resample_rounds":
                expected_val = 2
            if current_val is None and key == "max_resample_rounds":
                current_val = 2
            if expected_val is None and key == "resample_temp_boost":
                expected_val = 0.0
            if current_val is None and key == "resample_temp_boost":
                current_val = 0.0
            if expected_val != current_val:
                raise ValueError(f"Cannot resume with changed training option: {key} (expected {expected_val}, got {current_val})")

        context["run_dir"] = run_dir
        validate_checkpoint(resume_checkpoint, context)
        return context

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = os.path.abspath(os.path.join(save_root, run_id))
    os.makedirs(run_dir, exist_ok=False)
    context = {"run_id": run_id, "run_dir": run_dir, "data_sha256": data_hash, "config": config}
    with open(os.path.join(run_dir, "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2, ensure_ascii=False)
    return context


def validate_checkpoint(path, context):
    with open(os.path.join(path, "checkpoint_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    identity = meta.get("run_context", {})
    if any(identity.get(key) != context.get(key) for key in ("run_id", "data_sha256")):
        raise ValueError("Checkpoint belongs to another run or dataset")
    if identity.get("config", {}).get("model_name") != context["config"].get("model_name"):
        raise ValueError("Checkpoint base model does not match this run")
    return meta


def initial_selection():
    return {"best_val_acc": None, "best_step": None, "best_checkpoint": None, "base_val_acc": None}


def select_checkpoint(selection, val_acc, step):
    if val_acc is None:
        return False
    if selection["best_val_acc"] is None or val_acc > selection["best_val_acc"]:
        selection.update(best_val_acc=val_acc, best_step=step, best_checkpoint="checkpoint-best")
        return True
    return False


def best_checkpoint_path(context, selection):
    name = selection.get("best_checkpoint")
    if name is None:
        return None
    if name != "checkpoint-best":
        raise ValueError("Invalid selected checkpoint name")
    path = os.path.join(context["run_dir"], name)
    meta = validate_checkpoint(path, context)
    saved = meta.get("selection", {})
    if any(saved.get(k) != selection.get(k) for k in ("best_step", "best_val_acc")):
        raise ValueError("Selected checkpoint does not match saved validation selection")
    return path
