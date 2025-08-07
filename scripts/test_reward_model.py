import torch
import chess
import chess.engine
import argparse
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict

STOCKFISH_PATH = "./stockfish/stockfish-macos-x86-64-bmi2"
STOCKFISH_PATH_WSL = "./stockfish/stockfish-ubuntu-x86-64-avx2"

"""
test reward model by comparing its ranking of sft and stockfish moves across positions and print summaries
"""
# a small set of named positions to probe different phases and tactics for quick evaluation
TEST_POSITIONS = [
    {"name": "Italian Game", "moves": "1. e4 e5 2. Nf3 Nc6"},
    {"name": "Queen's Gambit", "moves": "1. d4 d5 2. c4"},
    {"name": "Sicilian Defense", "moves": "1. e4 c5 2. Nf3 d6 3. d4 cxd4"},
    {"name": "French Defense", "moves": "1. e4 e6 2. d4 d5"},
    {"name": "King's Indian Setup", "moves": "1. d4 Nf6 2. c4 g6 3. Nc3 Bg7 4. e4 d6"},
    {"name": "Ruy Lopez", "moves": "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6"},
    {"name": "Pin Tactic", "moves": "1. e4 e5 2. Nf3 Nc6 3. Bc4 Be7 4. d3 d6 5. Bg5"},
    {"name": "Fork Setup", "moves": "1. e4 e5 2. Nf3 Nc6 3. d4 exd4 4. Nxd4"},
    {"name": "Pawn Endgame", "moves": "1. e4 e5 2. f4 exf4 3. Nf3 g5 4. h4 g4 5. Ne5"},
    {"name": "Minor Piece Endgame", "moves": "1. e4 e5 2. Nf3 Nc6 3. Bb5 f5 4. exf5 e4"}
]

def analyze_single_position(position_data: Dict, sft_model, sft_tokenizer, reward_model, reward_tokenizer, engine, device) -> Dict:
    """
    analyze a single chess position using sft and reward models and stockfish and return evaluation results
    """
    # extract the moves text from the position data and create a fresh board to replay moves on
    prompt_text = position_data["moves"]
    board = chess.Board()
    
    # try to push each san move onto the board skip tokens like move numbers and if any parsing fails return an error message
    try:
        for move_san in prompt_text.split():
            if "." in move_san:
                continue
            board.push_san(move_san)
    except Exception as e:
        return {"error": f"Error setting up board from prompt: {e}"}

    # ask stockfish for its best move as ground truth using a short time limit then convert to san
    result = engine.play(board, chess.engine.Limit(time=0.5))
    stockfish_best_move = board.san(result.move)

    # tokenize the prompt and run the sft model to sample several continuations for candidate moves
    inputs = sft_tokenizer(prompt_text, return_tensors="pt").to(device)
    # generate multiple short samples to increase the chance of a legal and diverse first move
    outputs = sft_model.generate(
        input_ids=inputs['input_ids'],              # prompt token ids to condition generation
        attention_mask=inputs['attention_mask'],    # mask indicating valid (1) vs padding (0) tokens
        max_new_tokens=5,                           # cap on newly generated tokens
        num_return_sequences=5,                     # number of sampled continuations to return
        do_sample=True,                             # enable stochastic sampling instead of greedy decoding
        top_k=40,                                   # restrict sampling to top-k candidates at each step
        pad_token_id=sft_tokenizer.eos_token_id     # padding id for aligned sequence lengths in batch
    )
    
    # collect unique legal san suggestions from the sft model by decoding each sample and taking the first token
    sft_suggestions = []
    for output in outputs:
        new_text = sft_tokenizer.decode(output, skip_special_tokens=True).replace(prompt_text, "").strip()
        move_str = new_text.split(" ")[0]
        # validate the move by parsing san and checking board legality and ensure we only keep unique entries
        try:
            if board.parse_san(move_str) in board.legal_moves and move_str not in sft_suggestions:
                sft_suggestions.append(move_str)
        # ignore any parse errors and continue collecting other moves
        except Exception:
            continue

    # combine stockfish best and sft suggestions into a unique list of moves to evaluate by the reward model
    moves_to_evaluate = list(set([stockfish_best_move] + sft_suggestions))
    # if there are no legal moves from sft return a helpful error so the caller can skip this case
    if not moves_to_evaluate:
        return {"error": "SFT model did not produce any legal moves to evaluate."}

    # build inputs for the reward model by concatenating the prompt with each move and tokenize as a batch
    rm_prompts = [prompt_text + " " + move for move in moves_to_evaluate]
    rm_inputs = reward_tokenizer(rm_prompts, padding=True, return_tensors="pt").to(device)

    # evaluate rewards in a no grad context which means do not track gradients and free memory automatically at block end
    with torch.no_grad():
        rewards = reward_model(**rm_inputs).logits.squeeze()

    # prepare a list of dicts pairing each move with its reward score for later sorting and display
    results = []
    for i, move in enumerate(moves_to_evaluate):
        results.append({"move": move, "reward_score": rewards[i].item()})

    # sort moves by descending reward choose the top one and compute the rank of stockfish move among them
    results.sort(key=lambda x: x["reward_score"], reverse=True)
    reward_model_best = results[0]["move"]
    stockfish_rank = next(i for i, r in enumerate(results) if r["move"] == stockfish_best_move) + 1
    # return a structured dict summarizing the analysis including fen suggestions ranking and agreement flag
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
    """
    load models and engine test one or more positions compare reward model choice with stockfish and print summaries
    """
    print("--- Loading Models and Engine ---")
    # choose a device string preferring cuda if available for faster inference
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # load sft and reward tokenizers and models moving models to device for inference
    sft_tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path)
    sft_model = AutoModelForCausalLM.from_pretrained(args.sft_model_path).to(device)

    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path)
    reward_model = AutoModelForSequenceClassification.from_pretrained(args.reward_model_path).to(device)

    # launch a stockfish engine process via uci protocol for best move queries
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH_WSL)

    # ensure both tokenizers have pad tokens to support batch tokenization
    if sft_tokenizer.pad_token is None:
        sft_tokenizer.pad_token = sft_tokenizer.eos_token
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token

    # choose which positions to analyze either a single custom prompt or a predefined list
    if args.single_prompt:
        position_data = {"name": "Custom Position", "moves": args.prompt}
        positions_to_test = [position_data]
    # otherwise use the test positions defined above to evaluate across opening tactics and endgames
    else:
        positions_to_test = TEST_POSITIONS

    print(f"\n--- Testing {len(positions_to_test)} Position(s) ---")
    
    all_results = []
    # iterate through positions and analyze each printing errors and continuing when an issue occurs
    for i, position_data in enumerate(positions_to_test, 1):
        print(f"\n[{i}/{len(positions_to_test)}] Analyzing: {position_data['name']}")
        print(f"Position: {position_data['moves']}")
        result = analyze_single_position(
            position_data, sft_model, sft_tokenizer, 
            reward_model, reward_tokenizer, engine, device
        )
        # if the analyzer returns an error skip adding results and move to the next position
        if "error" in result:
            print(f"Error: {result['error']}")
            continue
            
        all_results.append(result)
        
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
    
    # if multiple positions were analyzed compute summary statistics over the set for agreement and ranking
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY STATISTICS")
        print(f"{'='*60}")
        total_positions = len(all_results)
        agreements = sum(1 for r in all_results if r['reward_model_agrees'])
        agreement_rate = agreements / total_positions * 100
        
        avg_stockfish_rank = sum(r['stockfish_rank'] for r in all_results) / total_positions
        
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
        # print per position summary including agreement icon and ranking values for quick inspection
        for r in all_results:
            agreement = "✓" if r['reward_model_agrees'] else "✗"
            print(f"{r['position_name']:<25} | {agreement:<10} | #{r['stockfish_rank']:<7} | {r['reward_model_best']:<8} | {r['stockfish_best']:<8}")

    engine.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test and compare sft and reward models against stockfish")
    parser.add_argument("--sft_model_path", type=str, default="models/sft_model_filtered", help="path to the sft model")
    parser.add_argument("--reward_model_path", type=str, default="models/reward_model_targeted", help="path to the reward model")
    parser.add_argument("--prompt", type=str, default="1. e4 e5 2. Nf3 Nc6", help="the chess prompt to give the model used with single prompt")
    parser.add_argument("--single-prompt", action="store_true", help="test only a single custom prompt instead of the full test suite")
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
