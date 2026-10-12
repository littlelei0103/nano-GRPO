import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from task_countdown import generate_countdown_sample, compute_reward

MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")

sample = generate_countdown_sample(num_count=4, target_range=(10, 30))
inputs = tokenizer(sample["prompt"], return_tensors="pt").to("cuda")

with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.9)
pred = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

print("=== 题目要求 ===")
print("数字:", sample["numbers"], "目标:", sample["target"])
print("\n=== Base 模型初始输出（无思考链/纯瞎蒙）===")
print(pred)
print("\n初始得分:", compute_reward(pred, sample["numbers"], sample["target"]))