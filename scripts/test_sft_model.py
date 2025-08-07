import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

"""
test a fine tuned sft chess model by generating the next move from a given prompt and print the first move
"""

def main(args):
    """
    load tokenizer and model prepare inputs generate text and print the first predicted move
    """
    print(f"Loading model from: {args.model_path}")
    # choose cuda when available otherwise cpu for running the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # load tokenizer and model from the given path and move model to the selected device
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)
    
    # ensure the tokenizer has a pad token set using eos and align the model pad token id to avoid generation warnings
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    print("Model loaded successfully. Ready to generate moves.")
    print("-" * 30)

    prompt_text = args.prompt
    
    print(f"Prompt: '{prompt_text}'")
    # tokenize the prompt returning pytorch tensors and move them to the selected device
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    # create an attention mask of ones the same shape as input ids to tell the model to attend to all tokens
    attention_mask = torch.ones(inputs.input_ids.shape, dtype=torch.long)
    # move all input tensors and the attention mask to the device before generation
    inputs = {k: v.to(device) for k, v in inputs.items()}
    attention_mask = attention_mask.to(device)

    # generate a short continuation using sampling strategies to obtain a plausible next move in san form
    output = model.generate(
        inputs['input_ids'],                 # prompt token ids to condition generation
        attention_mask=attention_mask,       # mask indicating valid (1) vs padding (0) tokens in the prompt
        max_new_tokens=5,                    # cap on the number of new tokens to generate
        do_sample=True,                      # enable stochastic sampling instead of greedy/beam decoding
        top_k=50,                            # restrict next-token candidates to the top-k most likely
        top_p=0.95,                          # or use nucleus sampling: smallest set with cumulative prob ≥ p
        pad_token_id=tokenizer.eos_token_id  # token id used for padding sequences in a batch
    )

    # decode the generated ids to text strip the prompt and take the first space separated token as the predicted move
    full_generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    newly_generated_text = full_generated_text.replace(prompt_text, "").strip()
    first_move = newly_generated_text.split(" ")[0]

    print(f"Model's Predicted Next Move: '{first_move}'")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test a fine tuned sft chess model")
    parser.add_argument("--model_path", type=str, default="models/sft_model", help="path to the fine tuned model directory")
    parser.add_argument("--prompt", type=str, default="1. e4 e5 2. Nf3 Nc6 3. Bc4", help="the chess prompt to give the model")
    args = parser.parse_args()
    main(args)

"""
python scripts/test_sft_model.py \
    --model_path models/sft_model \
    --prompt "1. e4 e5 2. Nf3 Nc6 3. Bc4"
"""