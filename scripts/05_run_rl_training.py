import argparse
import torch
import gc
import time
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import PPOTrainer, AutoModelForCausalLMWithValueHead, PPOConfig, create_reference_model
from trl.core import LengthSampler
# FIX 1: Import necessary classes from datasets
from datasets import load_dataset, Features, Value, Sequence
from torch.utils.data import DataLoader
import logging

# Set up logging to reduce noise
logging.getLogger("transformers").setLevel(logging.ERROR)

def main(args):
    # Check for CUDA and determine precision
    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = use_cuda and not use_bf16
    
    if use_bf16:
        print("CUDA is available with bf16 support. Using BF16 for training.")
    elif use_fp16:
        print("CUDA is available. Using FP16 for training.")
    else:
        print("CUDA is not available. Using FP32 for training.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Define output length parameters
    min_output_length = 2
    max_output_length = 6

    # Optimized PPO configuration
    ppo_config = PPOConfig(
        learning_rate=2e-5,
        local_batch_size=8,
        local_mini_batch_size=8,
        num_ppo_epochs=4,
        stop_token_id=tokenizer.eos_token_id,
        total_episodes=1000,  # Add total episodes for training
        response_length=max_output_length,
    )
    
    # Load models with memory management
    print("Loading policy model...")
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(args.sft_model_path)
    
    base_model = policy_model.pretrained_model
    if hasattr(base_model, "generation_config"):
        policy_model.generation_config = base_model.generation_config
    if hasattr(base_model, "base_model_prefix"):
        policy_model.base_model_prefix = base_model.base_model_prefix
        setattr(policy_model, base_model.base_model_prefix, base_model)
    
    policy_model = policy_model.to(device)
    
    print("Loading reward model...")
    reward_model = AutoModelForSequenceClassification.from_pretrained(args.reward_model_path)
    reward_model = reward_model.to(device)
    reward_model.eval()
    
    # Create a wrapper class to add the score method
    class RewardModelWrapper:
        def __init__(self, model):
            self.model = model
            
        def score(self, hidden_states):
            # The reward model expects the full sequence, not just hidden states
            # For now, we'll use the logits directly
            return self.model.classifier(hidden_states)
            
        def __call__(self, *args, **kwargs):
            return self.model(*args, **kwargs)
            
        def __getattr__(self, name):
            return getattr(self.model, name)
    
    reward_model = RewardModelWrapper(reward_model)
    
    # Add score method to policy model for reward computation
    def score_method(self, hidden_states):
        # Use the reward model to score the hidden states
        # This is a simplified approach - in practice, you might need to process the hidden states differently
        return reward_model.model.score(hidden_states)
    
    # Monkey patch the score method onto the policy model
    import types
    policy_model.score = types.MethodType(score_method, policy_model)

    if use_cuda:
        torch.cuda.empty_cache()
        print(f"GPU memory after model loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # Load and prepare dataset
    print(f"Loading dataset from {args.dataset_path}...")
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    
    # FIX 2: Define the output features to prevent type inference errors
    output_features = Features({
        'input_ids': Sequence(Value('int64')),
    })

    # FIX 3: Update mapping function to return a list of ints
    def format_chess_prompt(example):
        text = example["text"]
        moves = text.split(" ")
        prompt_length = min(20, max(16, len(moves) // 2))
        prompt = " ".join(moves[:prompt_length])
        tokenized = tokenizer(prompt, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
        return {
            "input_ids": tokenized["input_ids"].squeeze(0).tolist(),
        }
    
    # FIX 4: Pass the explicit features to the .map() call
    dataset = dataset.map(
        format_chess_prompt,
        remove_columns=list(dataset.features),
        features=output_features
    )
    
    if len(dataset) > 50000:
        dataset = dataset.select(range(50000))
        print(f"Limited dataset to {len(dataset)} samples for efficient RL training")
    
    # Split dataset for training and evaluation (90% train, 10% eval)
    train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]
    
    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")
    print(f"Dataset prepared with {len(train_dataset)} training samples and {len(eval_dataset)} evaluation samples")

    # Create reference model for PPO
    print("Creating reference model...")
    ref_model = create_reference_model(policy_model)
    
    ppo_trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=policy_model.pretrained_model,  # Use the base model without value head
        ref_model=ref_model.pretrained_model,  # Use the base reference model
        reward_model=reward_model.model,  # Use the wrapped reward model
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # Add evaluation dataset
        value_model=policy_model,  # Use the full model with value head
        data_collator=None,  # Use default data collator
    )
    
    output_length_sampler = LengthSampler(min_output_length, max_output_length)
    
    print(f"Starting RL training with:")
    print(f"  - Local batch size: {ppo_config.local_batch_size}")
    print(f"  - Local mini batch size: {ppo_config.local_mini_batch_size}")
    print(f"  - Learning rate: {ppo_config.learning_rate}")
    print(f"  - PPO epochs: {ppo_config.num_ppo_epochs}")
    print(f"  - Total episodes: {ppo_config.total_episodes}")
    print(f"  - Training dataset size: {len(train_dataset)}")
    print(f"  - Evaluation dataset size: {len(eval_dataset)}")
    print(f"  - Output length: {min_output_length}-{max_output_length} tokens")

    start_time = time.time()
    
    # Define custom reward function
    def compute_reward(model_output, **kwargs):
        # Extract generated text
        generated_ids = model_output
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Use reward model to score the generated text
        inputs = tokenizer(generated_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            reward_output = reward_model(**inputs)
            reward_score = reward_output.logits.squeeze(-1)
        
        return reward_score
    
    # Use the newer TRL training approach with custom reward
    print("Starting PPO training...")
    
    # Since we can't easily override the reward function, let's use a simpler approach
    # For now, we'll use a dummy reward (chess move quality could be assessed differently)
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
