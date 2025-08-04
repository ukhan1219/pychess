import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

def main(args):
    print(f"Loading model from: {args.model_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    print("Model loaded successfully. Ready to generate moves.")
    print("-" * 30)

    prompt_text = args.prompt
    
    print(f"Prompt: '{prompt_text}'")

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    
    attention_mask = torch.ones(inputs.input_ids.shape, dtype=torch.long)
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    attention_mask = attention_mask.to(device)

    output = model.generate(
        inputs['input_ids'], 
        attention_mask=attention_mask, 
        max_new_tokens=5,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )


    full_generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    newly_generated_text = full_generated_text.replace(prompt_text, "").strip()
    

    first_move = newly_generated_text.split(" ")[0]

    print(f"Model's Predicted Next Move: '{first_move}'")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a fine-tuned SFT chess model.")
    parser.add_argument("--model_path", type=str, default="models/sft_model", help="Path to the fine-tuned model directory.")
    parser.add_argument("--prompt", type=str, default="1. e4 e5 2. Nf3 Nc6 3. Bc4", help="The chess prompt to give the model.")
    args = parser.parse_args()
    main(args)