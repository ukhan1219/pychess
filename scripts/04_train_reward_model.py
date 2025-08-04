import argparse
from numpy import require
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
)
from datasets import load_dataset
from trl import RewardTrainer, RewardConfig
import torch


def main(args):
    use_fp16 = torch.cuda.is_available()
    if use_fp16:
        print("CUDA is available. Using FP16 for training.")

    else:
        print("CUDA is not available. Using FP32 for training.")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    def format_dataset(example):
        example["chosen"] = example["prompt"] + " " + example["chosen"]

        example["rejected"] = example["prompt"] + " " + example["rejected"]

        return example

    dataset = dataset.map(format_dataset)

    training_args = RewardConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=4,
        num_train_epochs=1,
        logging_steps=10,
        learning_rate=2e-5,
        fp16=use_fp16,
        remove_unused_columns=False,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=1,
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print("Starting training...")
    trainer.train()
    print("Training completed.")

    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a reward model.")
    parser.add_argument(
        "--base_model",
        type=str,
        default="distilgpt2",
        help="Name of the pre-trained model.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the preference dataset file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the trained model.",
    )
    args = parser.parse_args()
    main(args)


"""
python scripts/04_train_reward_model.py \           phase2 :: 8m :: ⬡
    --base_model distilgpt2 \
    --dataset_path data/processed/preference_dataset_targeted.jsonl \
    --output_dir models/reward_model_targeted
"""
