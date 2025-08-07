import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
import torch

"""
run supervised fine tuning on a language model using a jsonl dataset and save the trained model
configure tokenizer model dataset and training arguments and train with sft trainer
"""


def formatting_func(examples):
    """
    format input examples for sft
    """
    return examples["text"]


def main(args):
    """
    set up tokenizer model dataset and training configuration for sft and run training then save the model
    """
    # determine whether a cuda gpu is available to enable half precision training
    use_fp16 = torch.cuda.is_available()
    # conditionally print which precision will be used based on gpu availability
    if use_fp16:
        print("CUDA is available. Using FP16 for training.")
    # otherwise print that fp32 precision will be used
    else:
        print("CUDA is not available. Using FP32 for training.")

    # load the tokenizer for the base model from a hub id or local path
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # set a pad token if missing using eos to ensure batching works during training
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # load the causal language model weights for training with cache disabled to support gradient checkpointing
    model = AutoModelForCausalLM.from_pretrained(args.base_model, use_cache=False)

    # load a json dataset from the given file path reading the train split
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    # split the dataset into train and eval using a fixed seed for reproducibility
    train_test_split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    # configure training arguments including batch sizes steps evaluation schedule and optimization parameters
    training_args = TrainingArguments(
        output_dir=args.output_dir,                # where checkpoints and logs are written
        per_device_train_batch_size=12,           # micro-batch size per GPU/CPU
        gradient_accumulation_steps=2,            # number of steps to accumulate grads before optimizer.step()
        num_train_epochs=1,                       # number of passes over the training dataset (ignored if max_steps > 0)
        max_steps=-1,                             # total training steps; -1 means infer from num_train_epochs
        logging_steps=250,                        # step interval for logging metrics
        save_steps=2500,                          # step interval for saving checkpoints
        save_total_limit=3,                       # keep at most this many checkpoints (oldest pruned)
        eval_strategy="steps",                    # run evaluation on a fixed step interval
        eval_steps=2500,                          # step interval used when eval_strategy='steps'
        load_best_model_at_end=True,              # restore best checkpoint (by metric_for_best_model) at end of training
        metric_for_best_model="eval_loss",        # metric name to select the best checkpoint
        greater_is_better=False,                  # whether higher metric is better; False since lower loss is better
        learning_rate=1e-4,                       # base learning rate for the optimizer
        warmup_steps=1000,                        # steps to linearly warm up the learning rate from 0 to lr
        lr_scheduler_type="cosine",               # schedule to decay the learning rate after warmup
        weight_decay=0.01,                        # L2 weight decay applied to optimizer parameters
        fp16=use_fp16,                            # enable float16 training when CUDA is available
        gradient_checkpointing=True,              # trade compute for memory by checkpointing activations
        dataloader_pin_memory=True,               # pin CPU memory in DataLoader for faster host→device copies
        remove_unused_columns=False,              # keep all dataset columns (some trainers require original fields)
        report_to="none",                         # disable reporting to external loggers (e.g., wandb/tensorboard)
    )

    # create the sft trainer by providing the model datasets formatting function and training configuration
    trainer = SFTTrainer(
        model=model,                   # causal LM to fine-tune
        train_dataset=train_dataset,   # training split used for updates
        eval_dataset=eval_dataset,     # evaluation split for periodic validation
        formatting_func=formatting_func,  # function that converts a dataset example to training text
        args=training_args,            # TrainingArguments controlling optimization and logging
    )

    print("Starting Supervised Fine-Tuning...")
    # run the training loop according to the training arguments updating model weights and writing checkpoints
    trainer.train()

    print("Training Finished.")

    print(f"Saving model to {args.output_dir}")
    # save the final trained model to the specified directory creating model and tokenizer files
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

