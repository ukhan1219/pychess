import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
import torch

"""
    this script runs supervised fine-tuning (SFT) on a language model using the Hugging Face Transformers library.
    then it saves the fine-tuned model to a specified output directory.
    The script uses the SFTTrainer from the trl library to handle the training process.
    The dataset is expected to be in JSON format, and the text data is extracted from the "text" field of the dataset.
    The script also handles the case where CUDA is available for training, enabling FP16 training.
    and it sets the pad token for the tokenizer if it is not already set.
    it uses the AutoModelForCausalLM and AutoTokenizer classes to load the model and tokenizer.
    The training arguments are defined using the TrainingArguments class, specifying parameters like batch size.
    finally, the script defines a formatting function to extract the text from the dataset examples.
"""


def formatting_func(examples):
    """
    This function formats the input examples for supervised fine-tuning.
    """
    return examples["text"]


def main(args):
    # this function is the main entry point for the script.
    # It sets up the model, tokenizer, dataset, and training arguments for supervised fine-tuning.
    use_fp16 = torch.cuda.is_available()
    if use_fp16:
        print("CUDA is available. Using FP16 for training.")

    else:
        print("CUDA is not available. Using FP32 for training.")

    # we load the tokenizer for the base model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # we set the pad token for the tokenizer if it is not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # we load the model for causal language modeling
    model = AutoModelForCausalLM.from_pretrained(args.base_model, use_cache=False)

    # we check if the model is in half-precision (FP16) mode
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    # the dataset is loaded from the specified JSON file
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=4,
        num_train_epochs=1,
        logging_steps=100,
        save_steps=1000,
        learning_rate=2e-5,
        fp16=use_fp16,
        gradient_checkpointing=True,
    )

    # this sets the training arguments for the SFTTrainer
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


"""
python scripts/02_run_sft.py \
--dataset_path data/processed/sft_dataset_filtered.jsonl \
--output_dir models/sft_model
"""
