import chess
import chess.pgn
import chess.engine
import argparse
import json
import sys
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random

STOCKFISH_PATH = "./stockfish/stockfish-macos-x86-64-bmi2"


def get_sft_move(model, tokenizer, board):
    """
    Generate a move using the SFT model.
    """

    prompt = " ".join([board.san(move) for move in board.move_stack])
    if not prompt:
        prompt = "1."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output_sequences = model.generate(
        input_ids=inputs["input_ids"],
        max_new_tokens=5,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

    move_str = generated_text.replace(prompt, "").strip().split(" ")[0]
    return move_str


def main(args):
    print("Loading SFT model...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    sft_model = AutoModelForCausalLM.from_pretrained(args.sft_model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Starting STOCKFISH engine...")
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    if args.input_games_file:
        print(f"Opening PGN file: {args.input_games_file}")
        pgn_stream = open(args.input_games_file)
    else:
        print("No input file provided. Reading from standard input (stdin)...")
        pgn_stream = sys.stdin

    preference_data = []

    pbar = tqdm(total=args.num_samples, desc="Processing games")

    game_iterator = iter(lambda: chess.pgn.read_game(pgn_stream), None)

    for game in game_iterator:
        if len(preference_data) >= args.num_samples:
            break

        if game is None:
            continue

        board = game.board()

        for move in game.mainline_moves():
            board.push(move)

            if random.random() < args.sampling_ratio:
                try:
                    sft_move_san = get_sft_move(sft_model, tokenizer, board)

                    sft_move = board.parse_san(sft_move_san)

                    result = engine.play(board, chess.engine.Limit(time=0.1))
                    stockfish_move = result.move

                    if sft_move != stockfish_move:
                        preference_data.append(
                            {
                                "prompt": " ".join(
                                    [board.san(m) for m in board.move_stack]
                                ),
                                "chosen": board.san(stockfish_move),
                                "rejected": board.san(sft_move),
                            }
                        )

                        pbar.update(1)
                        if len(preference_data) >= args.num_samples:
                            break

                except Exception:
                    continue

    pbar.close()

    engine.quit()

    print(f"Generated {len(preference_data)} preference data samples.")

    with open(args.output_file, "w") as f:
        for item in preference_data:
            f.write(json.dumps(item) + "\n")

    print(f"Preference data saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate preference data from chess games."
    )
    parser.add_argument(
        "--sft_model_path", type=str, required=True, help="Path to the SFT model."
    )

    parser.add_argument(
        "--input_games_file",
        type=str,
        required=False,
        default=None,
        help="Path to the input PGN file containing chess games. If not provided, reads from STDIN.",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the generated preference data.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of preference data samples to generate.",
    )
    parser.add_argument(
        "--sampling_ratio",
        type=float,
        default=0.1,
        help="Ratio of moves to sample from each game.",
    )
    args = parser.parse_args()
    main(args)


"""
    zstdcat data/raw/lichess_db_standard_rated_2024-08.pgn.zst | python scripts/03_generate_preference_data.py \
    --sft_model_path models/sft_model \
    --output_file data/processed/preference_dataset_targeted.jsonl \
    --num_samples 10000
"""
