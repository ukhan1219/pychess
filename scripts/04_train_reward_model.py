import argparse
import json
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from datasets import load_dataset
from trl import RewardTrainer, RewardConfig
import torch

"""
train a reward model using pairwise preference data by concatenating prompt with chosen and rejected and optimizing a sequence classifier
configure tokenizer datasets and reward trainer then save the best model and metrics
"""


def main(args):
    """
    configure precision tokenizer dataset reward training arguments and run training then save model and metrics
    """
    # detect if a cuda gpu is available and whether bf16 is supported then set which precision to use
    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = use_cuda and not use_bf16
    # print which precision mode will be used based on availability of bf16 or fp16 or fallback to fp32
    if use_bf16:
        print("CUDA is available with bf16 support. Using BF16 for training.")
    # when bf16 is not available but a gpu exists print that fp16 will be used
    elif use_fp16:
        print("CUDA is available. Using FP16 for training.")
    # when no gpu is present print that fp32 on cpu will be used
    else:
        print("CUDA is not available. Using FP32 for training.")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    # ensure the tokenizer has a pad token if missing assign eos token so batching works
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # load the preference dataset from a jsonl file path using the train split
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    # concatenate the prompt text with the chosen and rejected texts so the reward model scores full continuations
    def format_dataset(example):
        example["chosen"] = example["prompt"] + " " + example["chosen"]
        example["rejected"] = example["prompt"] + " " + example["rejected"]
        return example
    # apply the formatting to every row then split into train and eval sets with a fixed seed
    dataset = dataset.map(format_dataset)
    train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    # configure reward training arguments such as batch sizes epochs evaluation and optimization settings
    training_args = RewardConfig(
        output_dir=args.output_dir,                 # where checkpoints and logs are written
        per_device_train_batch_size=12,            # micro-batch size per device for training
        gradient_accumulation_steps=2,             # accumulate gradients to simulate larger batch size
        per_device_eval_batch_size=20,             # batch size per device used during evaluation
        num_train_epochs=2,                        # number of passes over the training data (ignored if max_steps > 0)
        max_steps=-1,                              # total training steps; -1 means derive from num_train_epochs
        logging_steps=250,                         # step interval for logging metrics
        save_steps=2500,                           # step interval for saving checkpoints
        save_total_limit=6,                        # keep at most this many checkpoints
        eval_strategy="steps",                     # run evaluation on a fixed step interval
        eval_steps=2500,                           # step interval used when eval_strategy='steps'
        load_best_model_at_end=True,               # load best checkpoint (by metric_for_best_model) after training
        metric_for_best_model="eval_loss",         # metric to track for model selection
        greater_is_better=False,                   # whether a larger metric value is better (False for loss)
        learning_rate=1e-4,                        # base learning rate used by the optimizer
        warmup_steps=1000,                         # steps to warm up learning rate from 0 to lr
        lr_scheduler_type="cosine",                # schedule controlling how lr decays after warmup
        weight_decay=0.01,                         # L2 weight decay factor
        max_grad_norm=1.0 if not use_fp16 else 0.0, # gradient clipping threshold (disabled for fp16 here)
        fp16=use_fp16,                             # enable float16 training on CUDA when bf16 is unavailable
        bf16=use_bf16,                             # enable bfloat16 training when supported
        remove_unused_columns=False,               # keep all dataset columns required by trainer
        dataloader_pin_memory=True,                # pin CPU memory for faster host→device copies
        report_to="none",                          # disable integration with external loggers
        seed=42,                                   # seed for deterministic behavior where possible
    )

    # load a sequence classification model head with a single score output to serve as the reward model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=1,
    )
    # set the pad token id on the model config to align with the tokenizer for correct padding behavior
    model.config.pad_token_id = tokenizer.pad_token_id
    # create the reward trainer that handles batching tokenization and optimization over pairwise preferences
    trainer = RewardTrainer(
        model=model,                 # sequence classifier producing a single scalar reward
        args=training_args,          # RewardConfig controlling optimization and evaluation
        train_dataset=train_dataset, # training split of pairwise preference data
        eval_dataset=eval_dataset,   # evaluation split for periodic validation
        processing_class=tokenizer,  # tokenizer used to preprocess inputs for the model
    )

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Evaluation dataset size: {len(eval_dataset)}")
    print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    
    print("Starting reward model training...")
    # run the training loop which iterates over the dataset computes loss and updates model weights according to the config
    train_result = trainer.train()
    print("Training completed.")
    
    print(f"Final training loss: {train_result.training_loss:.4f}")
    # if the train result includes metrics print each key and value for visibility
    if hasattr(train_result, 'metrics'):
        # iterate and print metrics such as eval loss or training runtime
        for key, value in train_result.metrics.items():
            print(f"{key}: {value}")

    print(f"Saving model to {args.output_dir}")
    # save the trained model weights and tokenizer files to the output directory
    trainer.save_model(args.output_dir)
    # write the metrics json to disk using with which opens the file and ensures it is closed even if an error occurs
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

