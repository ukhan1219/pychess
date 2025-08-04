import torch
import chess
import chess.engine
import argparse
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

STOCKFISH_PATH = "./stockfish/stockfish-macos-x86-64-bmi2"
STOCKFISH_PATH_WSL = "./stockfish/stockfish-ubuntu-x86-64-avx2"

def main(args):
    print("--- Loading Models and Engine ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load all three components
    sft_tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path)
    sft_model = AutoModelForCausalLM.from_pretrained(args.sft_model_path).to(device)

    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path)
    reward_model = AutoModelForSequenceClassification.from_pretrained(args.reward_model_path).to(device)

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH_WSL)

    # Ensure padding tokens are set
    if sft_tokenizer.pad_token is None: sft_tokenizer.pad_token = sft_tokenizer.eos_token
    if reward_tokenizer.pad_token is None: reward_tokenizer.pad_token = reward_tokenizer.eos_token

    print("\n--- Analyzing Position ---")
    prompt_text = args.prompt
    board = chess.Board()
    try:
        # Replay the moves from the prompt to set up the board
        for move_san in prompt_text.split():
            if "." in move_san: continue # Skip move numbers like "1."
            board.push_san(move_san)
    except Exception as e:
        print(f"Error setting up board from prompt: {e}")
        engine.quit()
        return

    print(f"Board FEN: {board.fen()}")
    print(f"Prompt: '{prompt_text}'")

    # 2. Get Stockfish's best move (The Ground Truth)
    result = engine.play(board, chess.engine.Limit(time=0.5))
    stockfish_best_move = board.san(result.move)
    print(f"Stockfish Best Move: {stockfish_best_move}")

    # 3. Get multiple suggestions from our SFT model
    inputs = sft_tokenizer(prompt_text, return_tensors="pt").to(device)
    outputs = sft_model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=5,
        num_return_sequences=5, # Get 5 different suggestions
        do_sample=True,
        top_k=40,
        pad_token_id=sft_tokenizer.eos_token_id
    )
    
    sft_suggestions = []
    for output in outputs:
        new_text = sft_tokenizer.decode(output, skip_special_tokens=True).replace(prompt_text, "").strip()
        move_str = new_text.split(" ")[0]
        try:
            # Add only unique, legal moves
            if board.parse_san(move_str) in board.legal_moves and move_str not in sft_suggestions:
                sft_suggestions.append(move_str)
        except Exception:
            continue

    print(f"SFT Model Suggestions: {sft_suggestions}")

    # 4. Evaluate all interesting moves with our Reward Model
    moves_to_evaluate = list(set([stockfish_best_move] + sft_suggestions))
    if not moves_to_evaluate:
        print("SFT model did not produce any legal moves to evaluate.")
        engine.quit()
        return

    # Prepare batch for RM
    rm_prompts = [prompt_text + " " + move for move in moves_to_evaluate]
    rm_inputs = reward_tokenizer(rm_prompts, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        rewards = reward_model(**rm_inputs).logits.squeeze()

    # 5. Present the results
    results = []
    for i, move in enumerate(moves_to_evaluate):
        results.append({"move": move, "reward_score": rewards[i].item()})

    # Sort results by score, descending
    results.sort(key=lambda x: x["reward_score"], reverse=True)

    print("\n--- Comparison Table ---")
    print(f"{'Move':<10} | {'Reward Score':<15} | {'Notes'}")
    print("-" * 40)
    for res in results:
        notes = []
        if res['move'] == stockfish_best_move:
            notes.append("Stockfish's Pick")
        if res['move'] in sft_suggestions:
            notes.append("SFT Suggestion")
        
        print(f"{res['move']:<10} | {res['reward_score']:<15.4f} | {', '.join(notes)}")

    engine.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test and compare SFT and Reward models against Stockfish.")
    parser.add_argument("--sft_model_path", type=str, default="models/sft_model_filtered", help="Path to the SFT model.")
    parser.add_argument("--reward_model_path", type=str, default="models/reward_model_targeted", help="Path to the Reward model.")
    parser.add_argument("--prompt", type=str, default="1. e4 e5 2. Nf3 Nc6", help="The chess prompt to give the model.")
    args = parser.parse_args()
    main(args)
    
"""
python scripts/test_and_compare.py \
    --sft_model_path models/sft_model \
    --reward_model_path models/reward_model_targeted \
    --prompt "1. e4 e5 2. Nf3 Nc6"
"""