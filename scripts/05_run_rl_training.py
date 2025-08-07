import argparse
import torch
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import PPOTrainer, AutoModelForCausalLMWithValueHead, PPOConfig, create_reference_model
from trl.core import LengthSampler
from datasets import load_dataset, Features, Value, Sequence
import logging

"""
run rl fine tuning using ppo with an sft policy model and a reward model over a chess move dataset and save the tuned model
configure tokenizer models dataset and ppo trainer then train and save outputs
"""

logging.getLogger("transformers").setLevel(logging.ERROR)

def main(args):
    """
    set up tokenizer models dataset reward wrapper and ppo configuration then run training and save the model
    """
    # detect gpu availability and supported precision modes to choose compute dtype
    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = use_cuda and not use_bf16
    
    # print which precision mode is being used based on detection above
    if use_bf16:
        print("CUDA is available with bf16 support. Using BF16 for training.")
    # when bf16 is not available but cuda exists use fp16 and print that
    elif use_fp16:
        print("CUDA is available. Using FP16 for training.")
    # when cuda is not available fall back to fp32 on cpu and print that
    else:
        print("CUDA is not available. Using FP32 for training.")
    
    # select device string cuda if available otherwise cpu and print it for visibility
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # load tokenizer for the sft model path and ensure a pad token exists by using eos if needed for batching
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    min_output_length = 2
    max_output_length = 6

    # create a ppo configuration including lr batch sizes epochs stop token and response length
    ppo_config = PPOConfig(
        learning_rate=2e-5,                    # optimizer learning rate for PPO updates
        local_batch_size=8,                    # number of samples collected per update on this process
        local_mini_batch_size=8,               # minibatch size used when optimizing a batch
        num_ppo_epochs=4,                      # how many optimizer passes over each collected batch
        stop_token_id=tokenizer.eos_token_id,  # token id that indicates generation should stop
        total_episodes=1000,                   # total rollout episodes (collections) to run
        response_length=max_output_length,     # max number of tokens generated during rollouts
    )
    
    print("Loading policy model...")
    # load the policy model with a value head from the sft checkpoint for ppo fine tuning
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(args.sft_model_path)
    base_model = policy_model.pretrained_model

    # copy generation config from the base model into the wrapped model if present for consistent generation behavior
    if hasattr(base_model, "generation_config"):
        policy_model.generation_config = base_model.generation_config
    # align base model prefix attributes so the wrapper exposes the same attribute structure as the base model
    if hasattr(base_model, "base_model_prefix"):
        policy_model.base_model_prefix = base_model.base_model_prefix
        setattr(policy_model, base_model.base_model_prefix, base_model)
    
    # move the policy model to the selected device for computation
    policy_model = policy_model.to(device)
    
    print("Loading reward model...")
    # load the reward model for sequence classification and move it to the device and set eval mode
    reward_model = AutoModelForSequenceClassification.from_pretrained(args.reward_model_path)
    reward_model = reward_model.to(device)
    reward_model.eval()
    
    # lightweight wrapper class adds a score method and proxies other calls to the underlying reward model
    class RewardModelWrapper:
        """
        simple wrapper to expose a score method and proxy calls to the underlying reward model
        """
        # store the wrapped model reference for later calls
        def __init__(self, model):
            self.model = model
        # return scores from the classifier head given hidden states or final representation
        def score(self, hidden_states):
            return self.model.classifier(hidden_states)
        # forward calls to the underlying model for normal inference
        def __call__(self, *args, **kwargs):
            return self.model(*args, **kwargs)
        # delegate attribute access to the wrapped model for transparency
        def __getattr__(self, name):
            return getattr(self.model, name)
    # wrap the reward model instance so it exposes the score method used by ppo utilities
    reward_model = RewardModelWrapper(reward_model)
    # define a score method compatible with trl that calls into the reward model wrapper
    def score_method(self, hidden_states):
        return reward_model.model.score(hidden_states)
    
    # monkey patch the policy model to include the score method using types methodtype
    import types
    policy_model.score = types.MethodType(score_method, policy_model)

    # clear any lingering cuda memory and print a quick memory usage snapshot after loading models
    if use_cuda:
        torch.cuda.empty_cache()
        print(f"GPU memory after model loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    print(f"Loading dataset from {args.dataset_path}...")
    # load the training dataset from jsonl and select the train split for ppo prompts
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    # define the output schema for the mapped dataset to avoid type inference issues
    output_features = Features({
        'input_ids': Sequence(Value('int64')),
    })
    # build prompt tokens by taking the first few moves and tokenizing to a fixed length returning ids list
    def format_chess_prompt(example):
        """
        build a prompt from the first moves and return token ids as a list for ppo input
        """
        text = example["text"]
        moves = text.split(" ")
        prompt_length = min(20, max(16, len(moves) // 2))
        prompt = " ".join(moves[:prompt_length])
        tokenized = tokenizer(prompt, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
        return {
            "input_ids": tokenized["input_ids"].squeeze(0).tolist(),
        }
    # apply the prompt formatting to each row remove original columns and enforce the declared features
    dataset = dataset.map(
        format_chess_prompt,
        remove_columns=list(dataset.features),
        features=output_features
    )
    # optionally limit dataset size for faster experiments in resource constrained environments
    if len(dataset) > 50000:
        dataset = dataset.select(range(50000))
        print(f"Limited dataset to {len(dataset)} samples for efficient RL training")
    
    # split dataset into train and eval portions for monitoring ppo progress
    train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]
    
    # set torch formatting so the dataloader yields tensors directly
    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")
    print(f"Dataset prepared with {len(train_dataset)} training samples and {len(eval_dataset)} evaluation samples")

    print("Creating reference model...")
    # create a frozen reference model from the policy to compute kl penalties during ppo
    ref_model = create_reference_model(policy_model)
    
    # construct the ppo trainer with policy reference reward datasets and tokenizer
    ppo_trainer = PPOTrainer(
        args=ppo_config,                        # PPO hyperparameters and rollout settings
        processing_class=tokenizer,             # tokenizer used to prepare prompts/responses
        model=policy_model.pretrained_model,    # policy (actor) used for generation
        ref_model=ref_model.pretrained_model,   # frozen reference model for KL penalty
        reward_model=reward_model.model,        # reward function producing scalar scores
        train_dataset=train_dataset,            # dataset providing prompts for rollouts
        eval_dataset=eval_dataset,              # held-out prompts for evaluation
        value_model=policy_model,               # model that provides the value head for advantage estimation
        data_collator=None,                     # optional batch collation function (None → default)
    )
    
    # define a sampler that will choose output lengths between the given bounds during rollouts
    output_length_sampler = LengthSampler(min_output_length, max_output_length)
    
    # log training configuration values to help understand runtime behavior and parameter choices
    print(f"Starting RL training with:")
    print(f"  - Local batch size: {ppo_config.local_batch_size}")
    print(f"  - Local mini batch size: {ppo_config.local_mini_batch_size}")
    print(f"  - Learning rate: {ppo_config.learning_rate}")
    print(f"  - PPO epochs: {ppo_config.num_ppo_epochs}")
    print(f"  - Total episodes: {ppo_config.total_episodes}")
    print(f"  - Training dataset size: {len(train_dataset)}")
    print(f"  - Evaluation dataset size: {len(eval_dataset)}")
    print(f"  - Output length: {min_output_length}-{max_output_length} tokens")

    # record a start time to compute throughput and elapsed time later
    start_time = time.time()
    
    # define a reward function example which decodes generated ids tokenizes for the reward model and computes a scalar score
    def compute_reward(model_output, **kwargs):
        """
        example placeholder for computing reward from generated ids using the reward model
        """
        # extract ids from model output and decode to text using the tokenizer
        generated_ids = model_output
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        # tokenize the generated text for the reward model and move tensors to the right device
        inputs = tokenizer(generated_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # use with to disable gradient tracking which saves memory and speeds up evaluation then run the reward model to get logits
        # with means enter a context where gradients are not tracked and exit automatically when block ends
        with torch.no_grad():
            reward_output = reward_model(**inputs)
            reward_score = reward_output.logits.squeeze(-1)
        
        return reward_score
    
    print("Starting PPO training...")
    ppo_trainer.train()

    print("Training completed. Saving model...")
    ppo_trainer.save_model(args.output_dir)
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RL training with PPO")
    parser.add_argument("--sft_model_path", type=str, required=True, help="Path to the SFT model")
    parser.add_argument("--reward_model_path", type=str, required=True, help="Path to the reward model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the training dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the RL-tuned model")
    args = parser.parse_args()
    main(args)

"""
python scripts/05_run_rl_training.py \
    --sft_model_path models/sft_model \
    --reward_model_path models/reward_model_targeted \
    --dataset_path data/processed/preference_dataset_targeted.jsonl \
    --output_dir models/rl_model
"""