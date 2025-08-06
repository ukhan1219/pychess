import torch
import chess
import chess.engine
import argparse
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Dict, Tuple

STOCKFISH_PATH = "./stockfish/stockfish-macos-x86-64-bmi2"
STOCKFISH_PATH_WSL = "./stockfish/stockfish-ubuntu-x86-64-avx2"

# Test positions covering different phases of the game
TEST_POSITIONS = [
    # Opening positions
    {"name": "Italian Game", "moves": "1. e4 e5 2. Nf3 Nc6"},
    {"name": "Queen's Gambit", "moves": "1. d4 d5 2. c4"},
    {"name": "Sicilian Defense", "moves": "1. e4 c5 2. Nf3 d6 3. d4 cxd4"},
    {"name": "French Defense", "moves": "1. e4 e6 2. d4 d5"},
    
    # Early middlegame positions
    {"name": "King's Indian Setup", "moves": "1. d4 Nf6 2. c4 g6 3. Nc3 Bg7 4. e4 d6"},
    {"name": "Ruy Lopez", "moves": "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6"},
    
    # Tactical positions
    {"name": "Pin Tactic", "moves": "1. e4 e5 2. Nf3 Nc6 3. Bc4 Be7 4. d3 d6 5. Bg5"},
    {"name": "Fork Setup", "moves": "1. e4 e5 2. Nf3 Nc6 3. d4 exd4 4. Nxd4"},
    
    # Endgame-ish positions
    {"name": "Pawn Endgame", "moves": "1. e4 e5 2. f4 exf4 3. Nf3 g5 4. h4 g4 5. Ne5"},
    {"name": "Minor Piece Endgame", "moves": "1. e4 e5 2. Nf3 Nc6 3. Bb5 f5 4. exf5 e4"}
]

def analyze_single_position(position_data: Dict, sft_model, sft_tokenizer, reward_model, reward_tokenizer, engine, device) -> Dict:
    """Analyze a single chess position and return results."""
    prompt_text = position_data["moves"]
    board = chess.Board()
    
    try:
        # Replay the moves from the prompt to set up the board
        for move_san in prompt_text.split():
            if "." in move_san: continue # Skip move numbers like "1."
            board.push_san(move_san)
    except Exception as e:
        return {"error": f"Error setting up board from prompt: {e}"}

    # Get Stockfish's best move (The Ground Truth)
    result = engine.play(board, chess.engine.Limit(time=0.5))
    stockfish_best_move = board.san(result.move)

    # Get multiple suggestions from our SFT model
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

    # Evaluate all interesting moves with our Reward Model
    moves_to_evaluate = list(set([stockfish_best_move] + sft_suggestions))
    if not moves_to_evaluate:
        return {"error": "SFT model did not produce any legal moves to evaluate."}

    # Prepare batch for RM
    rm_prompts = [prompt_text + " " + move for move in moves_to_evaluate]
    rm_inputs = reward_tokenizer(rm_prompts, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        rewards = reward_model(**rm_inputs).logits.squeeze()

    # Prepare results
    results = []
    for i, move in enumerate(moves_to_evaluate):
        results.append({"move": move, "reward_score": rewards[i].item()})

    # Sort results by score, descending
    results.sort(key=lambda x: x["reward_score"], reverse=True)
    
    # Calculate if reward model picked stockfish's move as best
    reward_model_best = results[0]["move"]
    stockfish_rank = next(i for i, r in enumerate(results) if r["move"] == stockfish_best_move) + 1
    
    return {
        "position_name": position_data["name"],
        "prompt": prompt_text,
        "fen": board.fen(),
        "stockfish_best": stockfish_best_move,
        "sft_suggestions": sft_suggestions,
        "results": results,
        "reward_model_best": reward_model_best,
        "stockfish_rank": stockfish_rank,
        "reward_model_agrees": reward_model_best == stockfish_best_move
    }

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

    # Determine which positions to test
    if args.single_prompt:
        # Single position mode (original behavior)
        position_data = {"name": "Custom Position", "moves": args.prompt}
        positions_to_test = [position_data]
    else:
        # Multi-position mode
        positions_to_test = TEST_POSITIONS

    print(f"\n--- Testing {len(positions_to_test)} Position(s) ---")
    
    all_results = []
    for i, position_data in enumerate(positions_to_test, 1):
        print(f"\n[{i}/{len(positions_to_test)}] Analyzing: {position_data['name']}")
        print(f"Position: {position_data['moves']}")
        
        result = analyze_single_position(
            position_data, sft_model, sft_tokenizer, 
            reward_model, reward_tokenizer, engine, device
        )
        
        if "error" in result:
            print(f"Error: {result['error']}")
            continue
            
        all_results.append(result)
        
        # Print individual results
        print(f"Board FEN: {result['fen']}")
        print(f"Stockfish Best: {result['stockfish_best']}")
        print(f"SFT Suggestions: {result['sft_suggestions']}")
        print(f"Reward Model Best: {result['reward_model_best']} ({'✓' if result['reward_model_agrees'] else '✗'})")
        print(f"Stockfish Rank: #{result['stockfish_rank']}")
        
        print(f"\n{'Move':<10} | {'Reward Score':<15} | {'Notes'}")
        print("-" * 40)
        for res in result['results']:
            notes = []
            if res['move'] == result['stockfish_best']:
                notes.append("Stockfish's Pick")
            if res['move'] in result['sft_suggestions']:
                notes.append("SFT Suggestion")
            
            print(f"{res['move']:<10} | {res['reward_score']:<15.4f} | {', '.join(notes)}")
    
    # Calculate and display summary statistics
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY STATISTICS")
        print(f"{'='*60}")
        
        total_positions = len(all_results)
        agreements = sum(1 for r in all_results if r['reward_model_agrees'])
        agreement_rate = agreements / total_positions * 100
        
        avg_stockfish_rank = sum(r['stockfish_rank'] for r in all_results) / total_positions
        
        # Rank distribution
        rank_counts = {}
        for r in all_results:
            rank = r['stockfish_rank']
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        
        print(f"Total Positions Tested: {total_positions}")
        print(f"Reward Model Agreement with Stockfish: {agreements}/{total_positions} ({agreement_rate:.1f}%)")
        print(f"Average Stockfish Rank: {avg_stockfish_rank:.2f}")
        
        print(f"\nStockfish Move Ranking Distribution:")
        for rank in sorted(rank_counts.keys()):
            count = rank_counts[rank]
            percentage = count / total_positions * 100
            print(f"  Rank #{rank}: {count} positions ({percentage:.1f}%)")
        
        print(f"\nPosition-by-Position Results:")
        print(f"{'Position':<25} | {'Agreement':<10} | {'SF Rank':<8} | {'RM Best':<8} | {'SF Best':<8}")
        print("-" * 80)
        for r in all_results:
            agreement = "✓" if r['reward_model_agrees'] else "✗"
            print(f"{r['position_name']:<25} | {agreement:<10} | #{r['stockfish_rank']:<7} | {r['reward_model_best']:<8} | {r['stockfish_best']:<8}")

    engine.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test and compare SFT and Reward models against Stockfish.")
    parser.add_argument("--sft_model_path", type=str, default="models/sft_model_filtered", help="Path to the SFT model.")
    parser.add_argument("--reward_model_path", type=str, default="models/reward_model_targeted", help="Path to the Reward model.")
    parser.add_argument("--prompt", type=str, default="1. e4 e5 2. Nf3 Nc6", help="The chess prompt to give the model (used with --single-prompt).")
    parser.add_argument("--single-prompt", action="store_true", help="Test only a single custom prompt instead of the full test suite.")
    args = parser.parse_args()
    main(args)
    
"""
Test multiple positions (default):
python scripts/test_reward_model.py \
    --sft_model_path models/sft_model \
    --reward_model_path models/reward_model_targeted

Test single custom position:
python scripts/test_reward_model.py \
    --sft_model_path models/sft_model \
    --reward_model_path models/reward_model_targeted \
    --prompt "1. e4 e5 2. Nf3 Nc6" \
    --single-prompt
"""