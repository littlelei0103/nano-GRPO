"""Fixed decoding settings shared by validation and final test evaluation."""
import torch
from trainer.grpo_loss import rollout_generation_config, generate_with_policy


def evaluate_benchmark(model, tokenizer, dataset, task, max_new_tokens=350,
                       device="cuda", do_sample=False, temperature=0.7,
                       samples_per_problem=1):
    if samples_per_problem < 1:
        raise ValueError("samples_per_problem must be positive")
    model.eval()
    model.config.use_cache = True
    config = rollout_generation_config(tokenizer, max_new_tokens, temperature, do_sample=do_sample)
    solved = hits = 0
    for item in dataset:
        inputs = tokenizer(item["prompt"], return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]
        problem_hits = 0
        for _ in range(samples_per_problem):
            with torch.no_grad():
                outputs = generate_with_policy(model, config, **inputs)
            completion = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
            reward = task.compute_rewards([item["prompt"]], [completion], [item])[0]
            problem_hits += int(reward >= 1)
        hits += problem_hits
        solved += int(problem_hits > 0)
    total = len(dataset)
    samples = total * samples_per_problem
    return {
        "problem_pass_rate": solved / total if total else None,
        "sample_pass_rate": hits / samples if samples else None,
        "solved_problems": solved, "total_problems": total,
        "total_samples": samples, "correct_samples": hits,
    }
