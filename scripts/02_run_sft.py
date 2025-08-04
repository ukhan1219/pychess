import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
import torch


def formatting_func(examples):
    """
    This function formats the input examples for supervised fine-tuning.
    """
    return examples["text"]


def main(args):
    use_fp16 = torch.cuda.is_available()
    if use_fp16:
        print("CUDA is available. Using FP16 for training.")

    else:
        print("CUDA is not available. Using FP32 for training.")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.base_model, use_cache=False)

    dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=8,
        num_train_epochs=1,
        logging_steps=100,
        save_steps=1000,
        learning_rate=2e-5,
        fp16=use_fp16,
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        formatting_func=formatting_func,
        args=training_args,
    )

    print("Starting Supervised Fine-Tuning...")

    trainer.train()

    print("Training Finished.")

    print(f"Saving model to {args.output_dir}")

    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Supervised Fine-Tuning (SFT) on a language model."
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="distilgpt2",
        help="Path to the base model or model identifier from Hugging Face Hub.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset in JSON format.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the fine-tuned model.",
    )
    args = parser.parse_args()
    main(args)
