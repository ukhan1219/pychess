import argparse
import json
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from datasets import load_dataset
from trl import RewardTrainer, RewardConfig
from trl.trainer.reward_trainer import RewardDataCollatorWithPadding
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
    
    # split dataset for evaluation (90% train, 10% eval for reward model)
    train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    training_args = RewardConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=12,         # increased for better gradient estimates previously 4
        gradient_accumulation_steps=2,          # effective batch size of 32 per device previously not needed
        per_device_eval_batch_size=20,          # larger eval batch for efficiency previously not needed
        num_train_epochs=2,                     # reward models often need more epochs than SFT previously 1
        max_steps=-1,                           # let epochs determine training length previously not needed
        logging_steps=250,                      # less frequent logging for large datasets previously 10
        save_steps=2500,                        # save checkpoints every 1000 steps previously not needed
        save_total_limit=6,                     # keep more checkpoints for reward model analysis previously not needed
        eval_strategy="steps",                  # enable evaluation during training previously not needed
        eval_steps=2500,                        # evaluate every 1000 steps previously not needed
        load_best_model_at_end=True,            # load best model based on eval loss previously not needed
        metric_for_best_model="eval_loss",      # metric for best model previously not needed
        greater_is_better=False,                # greater is better previously not needed
        early_stopping_patience=3,              # stop if no improvement for 3 eval cycles previously not needed
        learning_rate=1e-4,                     # conservative LR for reward model stability previously 2e-5   
        warmup_steps=1000,                      # warmup for training stability previously not needed
        lr_scheduler_type="cosine",             # cosine decay for better convergence previously not needed
        weight_decay=0.01,                      # regularization to prevent overfitting previously not needed
        max_grad_norm=1.0,                      # gradient clipping for stability previously not needed
        fp16=use_fp16,
        bf16=False,
        remove_unused_columns=False,
        dataloader_pin_memory=True,             # faster data loading previously not needed
        report_to="none",                       # disable wandb/tensorboard previously not needed
        seed=42,                                # reproducibility previously not needed
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=1,
        torch_dtype=torch.float16 if use_fp16 else torch.float32,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,            # previously not needed (train_dataset = dataset)
        eval_dataset=eval_dataset,              # previously not needed
        processing_class=tokenizer,
    )

    # print dataset information before training
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Evaluation dataset size: {len(eval_dataset)}")
    print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    
    print("Starting reward model training...")
    train_result = trainer.train()
    print("Training completed.")
    
    # print training metrics
    print(f"Final training loss: {train_result.training_loss:.4f}")
    if hasattr(train_result, 'metrics'):
        for key, value in train_result.metrics.items():
            print(f"{key}: {value}")

    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    
    # save training metrics
    with open(f"{args.output_dir}/training_metrics.json", "w") as f:
        json.dump(train_result.metrics, f, indent=2)


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
python scripts/04_train_reward_model.py \
--base_model distilgpt2 \
--dataset_path data/processed/preference_dataset_targeted.jsonl \
--output_dir models/reward_model_targeted

"""
