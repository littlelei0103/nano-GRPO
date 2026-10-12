import os
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from task_countdown import SYSTEM_PROMPT, find_countdown_solution

# ===================== 配置项 =====================
MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
OUTPUT_DIR = "./checkpoint-sft-warmup"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
EPOCHS = 2
LR = 2e-4
NUM_SAMPLES = 240  # 120 条 3 变量 + 120 条 4 变量

print(f"🛠️ 正在合成 {NUM_SAMPLES} 条高质量 Search & Verify 试错与自检思维链冷启动样本...")


def generate_verified_dataset(num_samples: int = 240):
    dataset = []
    half = num_samples // 2

    # 1. 前一半：3 个数字（基础心算与桥梁课程）
    for _ in range(half):
        while True:
            nums = [random.randint(1, 10) for _ in range(3)]
            target = random.randint(10, 30)
            sol = find_countdown_solution(nums, target)
            if sol is not None:
                expr, steps = sol
                break

        lines = [f"Target: {target}. Available numbers: {nums}."]
        if random.random() < 0.35:
            n1, n2 = nums[0], nums[1]
            fake_res = n1 + n2
            if fake_res != target:
                lines.append(f"Attempt 1: Try {n1} + {n2} = {fake_res}. Remaining numbers cannot reach {target}. Backtracking.")
                lines.append("Attempt 2: Let us re-examine operations:")
        else:
            lines.append("Let us break down the operations step by step:")

        for i, s in enumerate(steps, 1):
            lines.append(f"Step {i}: {s}.")
        lines.append(f"Verification: {expr} = {target}. Exact match!")

        cot = "\n".join(lines)
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"User: Using the numbers {nums}, create an equation that equals {target}.\n"
            f"You can use basic arithmetic operations (+, -, *, /) and each number can only be used once.\n"
            f"Assistant: <think>\n"
        )
        completion = f"{cot}\n</think>\n<answer> {expr} </answer>"
        dataset.append({"prompt": prompt, "completion": completion})

    # 2. 后一半：4 个数字（标准 Countdown）
    for _ in range(num_samples - half):
        while True:
            nums = [random.randint(1, 10) for _ in range(4)]
            target = random.randint(10, 40)
            sol = find_countdown_solution(nums, target)
            if sol is not None:
                expr, steps = sol
                break

        lines = [f"Target: {target}. Available numbers: {nums}."]
        if random.random() < 0.35:
            n1, n2 = nums[0], nums[1]
            fake_res = n1 + n2
            if fake_res != target:
                lines.append(f"Attempt 1: Try {n1} + {n2} = {fake_res}. Remaining numbers cannot reach {target}. Backtracking.")
                lines.append("Attempt 2: Let us re-examine operations:")
        else:
            lines.append("Let us break down the operations step by step:")

        for i, s in enumerate(steps, 1):
            lines.append(f"Step {i}: {s}.")
        lines.append(f"Verification: {expr} = {target}. Exact match!")

        cot = "\n".join(lines)
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"User: Using the numbers {nums}, create an equation that equals {target}.\n"
            f"You can use basic arithmetic operations (+, -, *, /) and each number can only be used once.\n"
            f"Assistant: <think>\n"
        )
        completion = f"{cot}\n</think>\n<answer> {expr} </answer>"
        dataset.append({"prompt": prompt, "completion": completion})

    random.shuffle(dataset)
    return dataset


samples = generate_verified_dataset(NUM_SAMPLES)
print(f"✅ 成功合成 {len(samples)} 条 Search & Verify 样本！")

# 2. 加载模型与 Tokenizer
print("🚀 加载基座模型并挂载 LoRA...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).to(DEVICE)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


# 3. 构造微调 Dataset（只对 Completion 计算损失，标准指令微调）
class CountdownSFTDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=384):
        self.items = []
        for d in data:
            full_text = d["prompt"] + d["completion"]
            enc_full = tokenizer(full_text, truncation=True, max_length=max_len, return_tensors="pt")
            enc_prompt = tokenizer(d["prompt"], truncation=True, max_length=max_len, return_tensors="pt")

            input_ids = enc_full.input_ids[0]
            labels = input_ids.clone()
            prompt_len = enc_prompt.input_ids.shape[1]
            labels[:prompt_len] = -100

            self.items.append({"input_ids": input_ids, "labels": labels})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    input_ids_pad = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels_pad = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    attention_mask = (input_ids_pad != tokenizer.pad_token_id).long()
    return {
        "input_ids": input_ids_pad.to(DEVICE),
        "attention_mask": attention_mask.to(DEVICE),
        "labels": labels_pad.to(DEVICE)
    }


loader = DataLoader(CountdownSFTDataset(samples, tokenizer), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# 4. 执行微调
print(f"🔥 开始微调（总计 {len(loader)} Batches / Epoch）...")
model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    for step, batch in enumerate(loader):
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"🌟 Epoch [{epoch+1}/{EPOCHS}] | SFT Loss: {avg_loss:.4f}")

# 5. 保存黄金冷启动权重
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Search & Verify SFT 冷启动圆满完成！已保存至 {OUTPUT_DIR}")