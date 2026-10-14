"""
trainer/utils.py - 训练基础设施模块 (显存管理、LoRA 适配器生命周期、状态检查点与精确恢复)
"""

import os
import json
import time
import random
import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

from .grpo_loss import (
    compute_completion_mask,
    compute_completion_logprobs,
    get_reference_logprobs,
    compute_grpo_loss
)


def get_gpu_memory_mb() -> float:
    """获取当前 GPU 分配的显存 (MB)"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def print_gpu_memory(tag: str = ""):
    """打印当前 GPU 显存使用快照"""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024 * 1024)
        res = torch.cuda.memory_reserved() / (1024 * 1024)
        print(f"  [Memory Snapshot {tag}] Allocated: {alloc:.1f} MB | Reserved: {res:.1f} MB (Capacity: 8151 MB)", flush=True)


def get_per_token_logps(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """
    通用逐 token log 概率提取 (用于兼容旧版接口)。
    对全序列做 shift 1 位对齐。
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    logprobs = -F.cross_entropy(shift_logits.transpose(1, 2), shift_labels, reduction="none")
    return logprobs


def setup_lora_and_model(
    model_name: str,
    device: str = "cuda",
    lora_r: int = 8,
    lora_alpha: int = 16,
    resume_checkpoint: str = "",
    init_adapter: str = ""
) -> Tuple[AutoTokenizer, torch.nn.Module]:
    """
    初始化基座模型并挂载轻量 LoRA 适配器。
    支持：
      1. resume_checkpoint: 同一实验的断点续训 (后续会恢复优化器与训练状态)
      2. init_adapter: 跨任务/新实验的权重热启动 (全新 run_id，重新初始化优化器)
      3. 全新初始化轻量 LoRA
    面向消费级显存 (8GB VRAM) 采用 bfloat16 混合精度与梯度检查点 (Gradient Checkpointing)。
    """
    print(f"[Model Setup] Loading base tokenizer and model: {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if resume_checkpoint and not os.path.isdir(resume_checkpoint):
        raise FileNotFoundError(resume_checkpoint)
    if init_adapter and not os.path.isdir(init_adapter):
        raise FileNotFoundError(init_adapter)
    torch_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if str(device).startswith("cuda") else torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    ).to(device)

    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"[Model Setup] Resuming LoRA adapter weights from checkpoint: {resume_checkpoint}", flush=True)
        model = PeftModel.from_pretrained(base_model, resume_checkpoint, is_trainable=True)
    elif init_adapter and os.path.exists(init_adapter):
        print(f"[Model Setup] Initializing with pre-trained LoRA adapter (transfer learning): {init_adapter}", flush=True)
        model = PeftModel.from_pretrained(base_model, init_adapter, is_trainable=True)
    else:
        print(f"[Model Setup] Initializing all-linear LoRA adapter (r={lora_r}, alpha={lora_alpha})...", flush=True)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules="all-linear",
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(base_model, lora_config)

    model.print_trainable_parameters()
    print_gpu_memory("Post-Model-Init")
    return tokenizer, model


def save_checkpoint(
    model: torch.nn.Module,
    tokenizer: Any,
    save_dir: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: Optional[int] = None,
    sample_idx: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    selection: Optional[Dict[str, Any]] = None,
    run_context: Optional[Dict[str, Any]] = None
):
    """
    完整持久化保存 LoRA 检查点、Tokenizer、优化器状态及训练断点元数据。
    支持断点精确恢复与实验可复现性。
    """
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # 保存训练状态与随机种子
    state: Dict[str, Any] = {
        "step": step,
        "sample_idx": sample_idx,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rng_state": torch.get_rng_state(),
        "metrics": metrics or {},
        "selection": selection or {},
        "run_context": run_context or {},
        "python_rng_state": random.getstate()
    }
    if torch.cuda.is_available():
        state["cuda_rng_states"] = torch.cuda.get_rng_state_all()

    torch.save(state, os.path.join(save_dir, "training_state.pt"))

    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))

    # 元数据 JSON 记录 (便于无需加载 PyTorch 即可快速检索)
    meta = {
        "step": step,
        "sample_idx": sample_idx,
        "saved_at": state["saved_at"],
        "metrics": metrics or {},
        "config": config or {},
        "selection": selection or {},
        "run_context": run_context or {}
    }
    with open(os.path.join(save_dir, "checkpoint_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[Checkpoint] State saved (weights, optimizer, RNG, metadata) -> {save_dir}", flush=True)


def load_training_state(
    save_dir: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    require_complete: bool = False,
) -> Dict[str, Any]:
    """
    从保存目录恢复优化器状态、随机数种子与训练断点指标。
    """
    state_path = os.path.join(save_dir, "training_state.pt")
    opt_path = os.path.join(save_dir, "optimizer.pt")
    meta_path = os.path.join(save_dir, "checkpoint_meta.json")
    if require_complete and (not os.path.isfile(state_path) or not os.path.isfile(opt_path)):
        raise ValueError("Resume requires both training_state.pt and optimizer.pt")

    restored_state: Dict[str, Any] = {}
    if os.path.exists(state_path):
        restored_state = torch.load(state_path, map_location="cpu")
        if "rng_state" in restored_state:
            torch.set_rng_state(restored_state["rng_state"])
        if "python_rng_state" in restored_state:
            random.setstate(restored_state["python_rng_state"])
        if torch.cuda.is_available() and "cuda_rng_states" in restored_state:
            torch.cuda.set_rng_state_all(restored_state["cuda_rng_states"])
        if torch.cuda.is_available() and "cuda_rng_state" in restored_state and restored_state["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state(restored_state["cuda_rng_state"])
        print(f"[Checkpoint] Resumed training state: Step {restored_state.get('step')}, SampleIdx {restored_state.get('sample_idx')}", flush=True)

    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            restored_state["meta"] = json.load(f)

    if optimizer is not None and os.path.exists(opt_path):
        opt_state = torch.load(opt_path, map_location="cpu")
        optimizer.load_state_dict(opt_state)
        print(f"[Checkpoint] Restored optimizer state and momentum buffers", flush=True)

    return restored_state


def seed_training(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reset_gpu_peak(device):
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)


def gpu_peak_metrics(device):
    if not str(device).startswith("cuda"):
        return {"peak_allocated_mb": None, "peak_reserved_mb": None}
    torch.cuda.synchronize(device)
    return {
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
    }
