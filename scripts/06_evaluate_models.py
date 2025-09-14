import chess
import chess.engine
import argparse
import os
import shutil
import platform
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from src.chess_utils import resolve_stockfish_path

class HFPlayer:
    """A player class that uses a Hugging Face model to make moves."""

    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_move(self, board):
        # build SAN move history from a fresh board so each move is legal at conversion time
        temp_board = chess.Board()
        sans = []
        for historical_move in board.move_stack:
            san_text = temp_board.san(historical_move)
            sans.append(san_text)
            temp_board.push(historical_move)
        prompt = " ".join(sans) if sans else "1."

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.device)

        # Try a few times to generate a legal move
        for _ in range(5):
            with torch.no_grad():
                output = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=5,
                    pad_token_id=self.tokenizer.eos_token_id,
                    attention_mask=inputs.get("attention_mask", None),
                )
            # Decode only new tokens (exclude the prompt)
            gen_ids = output[0][inputs["input_ids"].shape[1]:]
            new_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            move_str = (new_text.split(" ")[0]) if new_text else ""
            try:
                if move_str:
                    move = board.parse_san(move_str)
                    if move in board.legal_moves:
                        return move
            except Exception:
                continue  # malformed or illegal, try again
        return None  # Failed to generate a legal move


def play_game(player_white, player_black):
    board = chess.Board()
    while not board.is_game_over(claim_draw=True):
        player = player_white if board.turn == chess.WHITE else player_black
        move = player.get_move(board)

        if move is None:  # Player forfeits if it can't produce a legal move
            return "black" if board.turn == chess.WHITE else "white"

        board.push(move)

    result = board.result(claim_draw=True)
    if result == "1-0":
        return "white"
    if result == "0-1":
        return "black"
    return "draw"


def run_tournament(model_player, stockfish_player, num_games):
    results = {"win": 0, "loss": 0, "draw": 0}
    for i in tqdm(range(num_games), "Games"):
        # Alternate colors for fairness
        if i % 2 == 0:
            winner = play_game(model_player, stockfish_player)
            if winner == "white":
                results["win"] += 1
            elif winner == "black":
                results["loss"] += 1
            else:
                results["draw"] += 1
        else:
            winner = play_game(stockfish_player, model_player)
            if winner == "black":
                results["win"] += 1
            elif winner == "white":
                results["loss"] += 1
            else:
                results["draw"] += 1
    return results


def main(args):
    print("--- Setting up Benchmark Opponent (Stockfish) ---")
    engine_path = resolve_stockfish_path()
    if not engine_path:
        raise RuntimeError(
            "Could not locate a Stockfish binary. Install via Homebrew ('brew install stockfish') or ensure an executable exists in './stockfish/'."
        )
    try:
        stockfish_engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to launch Stockfish at '{engine_path}'. Ensure it is executable and compatible with your OS."
        ) from e
    stockfish_engine.configure({"Skill Level": args.skill_level})

    class StockfishPlayer:
        def get_move(self, board):
            res = stockfish_engine.play(board, chess.engine.Limit(time=0.1))
            return res.move

    stockfish_player = StockfishPlayer()

    print("\n--- Evaluating Baseline SFT Model ---")
    sft_player = HFPlayer(args.sft_model_path)
    sft_results = run_tournament(sft_player, stockfish_player, args.num_games)

    print("\n--- Evaluating Final RL-Tuned Model ---")
    rl_player = HFPlayer(args.rl_model_path)
    rl_results = run_tournament(rl_player, stockfish_player, args.num_games)

    stockfish_engine.quit()

    print("\n" + "=" * 30)
    print("       TOURNAMENT RESULTS")
    print("=" * 30)
    print(f"Benchmark: Stockfish Skill Level {args.skill_level}")
    print(f"Number of Games per Model: {args.num_games}\n")
    sft = sft_results
    print(f"SFT Model ('{args.sft_model_path}'):")
    print(f"  Wins: {sft['win']}, Losses: {sft['loss']}, Draws: {sft['draw']}")
    rl = rl_results
    print(f"RL Model ('{args.rl_model_path}'):")
    print(f"  Wins: {rl['win']}, Losses: {rl['loss']}, Draws: {rl['draw']}")
    print("=" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate chess models head-to-head.")
    parser.add_argument("--sft_model_path", type=str, required=True)
    parser.add_argument("--rl_model_path", type=str, required=True)
    parser.add_argument("--num_games", type=int, default=20)
    parser.add_argument(
        "--skill_level", type=int, default=1, help="Stockfish skill level (0-20)"
    )
    args = parser.parse_args()
    main(args)

"""
python scripts/06_evaluate_models.py \
    --sft_model_path models/sft_model \
    --rl_model_path models/rl_model \
    --num_games 50
"""