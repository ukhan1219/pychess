import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import PPOTrainer, AutoModelForCausalLMWithValueHead, PPOConfig
from trl.core import LengthSampler
from datasets import load_dataset

def main(args):
    ppo_config = PPOConfig(
        model_name=args.sft_model_path,
        learning_rate=1.41e-5,
        batch_size=64,
        mini_batch_size=8,
        log_with="tensorboard",
        project_kwargs={"logging_dir": "./logs"}
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_config.model_name).to(device)
    reward_model = AutoModelForSequenceClassification.from_pretrained(args.reward_model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    dataset = dataset.map(lambda x: {"prompt": " ".join(x["text"].split(" ")[:10]}))
    dataset.set_format("torch")
                            
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy_model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=dataset,
    )

    output_length_sampler = LengthSampler(5, 10) # or 3, 6?

    for epoch in tqdm(range(ppo_config.ppo_epochs), "Epoch"):
        for batch in tqdm(ppo_trainer.dataloader, "Batch"):
            query_tensors = batch["input_ids"]

            response_tensors = ppo_trainer.generate(
                query_tensors,
                return_prompt=False,
                length_sampler=output_length_sampler,
                **{"pad_token_id": tokenizer.eos_token_id},
            )

            batch["response"] = tokenizer.batch_decode(response_tensors)


            prompts = tokenizer.batch_decode(query_tensors, skip_special_tokens=True)
            texts = [q + r for q, r in zip(prompts, batch["response"])]
            
            reward_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)

            with torch.no_grad():
                rewards = reward_mode(**reward_inputs).logits

            stats = ppo_trainer.step(query_tensors, response_tensors, list(rewards))
            ppo_trainer.log_stats(stats, batch, rewards.cpu().numpy())


    print(f"Saving RL-tuned model to {args.output_dir}")

    ppo_trainer.save_model(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RL training with PPO")
    parser.add_argument("--sft_model_path", type=str, required=True)
    parser.add_argument("--reward_model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)


"""
    python scripts/05_run_rl_training.py \
    --sft_model_path models/sft_model \
    --reward_model_path models/reward_model \
    --dataset_path data/processed/sft_dataset.jsonl \
    --output_dir models/rl_model
"""
