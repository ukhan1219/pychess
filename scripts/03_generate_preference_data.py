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

"""
    this script generates preference data from chess games by comparing moves made by a supervised fine-tuned (SFT) model against moves suggested by the Stockfish engine.
    by evaluating the differences between these moves, the script creates a dataset that can be used for training or fine-tuning chess AI models.
    and it ensures the quality of the games by filtering based on Elo ratings and move legality.
    then, it saves the generated preference data in a structured format for further use.
"""

STOCKFISH_PATH = "./stockfish/stockfish-macos-x86-64-bmi2"
STOCKFISH_PATH_WSL = "./stockfish/stockfish-ubuntu-x86-64-avx2"

from src.chess_utils import is_game_high_quality


def get_sft_move(model, tokenizer, board, prompt):
    """
    Generate a move using the SFT model.
    """

    # this ensures the prompt is not empty
    if not prompt:
        prompt = "1."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # this generates multiple sequences to increase the chance of getting a legal move
    output_sequences = model.generate(
        input_ids=inputs["input_ids"],
        max_new_tokens=5,
        num_return_sequences=5,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )

    # we decode the generated sequences to text
    full_texts = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

    # we attempt to parse each generated move and check if it's legal
    for text in full_texts:
        try:
            # if the prompt is not found in the text, skip this output
            # else, extract the move string following the prompt
            move_str = text.replace(prompt, "").strip().split(" ")[0]
            move = board.parse_san(move_str)
            if move in board.legal_moves:
                return move_str
        except:
            continue

    return None


def main(args):
    print("Loading SFT model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # we load the SFT model and tokenizer
    sft_model = AutoModelForCausalLM.from_pretrained(args.sft_model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # we start the Stockfish engine
    print("Starting STOCKFISH engine...")
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    # here we open the PGN file or read from stdin
    if args.input_games_file:
        print(f"Opening PGN file: {args.input_games_file}")
        pgn_stream = open(args.input_games_file)
    else:
        print("No input file provided. Reading from standard input (stdin)...")
        pgn_stream = sys.stdin

    # here we process the games and generate preference data
    # and we use a progress bar to track progress
    # iteratively read games from the PGN stream
    preference_data = []
    processed_games = 0
    pbar = tqdm(total=args.num_samples, desc="Processing games")
    game_iterator = iter(lambda: chess.pgn.read_game(pgn_stream), None)

    # for each game, we sample moves and compare SFT model moves with Stockfish moves
    for game in game_iterator:
        if len(preference_data) >= args.num_samples:
            break
        if game is None:
            continue

        # we increment the processed games counter and update the progress bar
        processed_games += 1
        pbar.set_postfix(
            {"Samples Found": len(preference_data), "Processed Games": processed_games}
        )

        # if the game is not high quality, we skip it
        if not is_game_high_quality(game, args.min_elo):
            continue

        # try to sample a move from the game
        # if any error occurs, we skip to the next game
        # else, we compare the moves and store the preference data
        try:
            # mainline_moves gives us the moves in the game which we can index into
            mainline_moves = list(game.mainline_moves())
            if len(mainline_moves) < 5:
                continue

            # move index is chosen randomly but not too close to the start or end
            # because openings and endgames are less informative
            move_index = random.randint(10, len(mainline_moves) - 1)

            board = game.board()

            # for each move up to the sampled index, we push it onto the board
            for i in range(move_index):
                board.push(mainline_moves[i])

            # we create a temporary board to generate the SAN notation for the prompt
            # because the SFT model expects moves in SAN format
            # SAN format is the standard algebraic notation used in chess
            temp_board_for_san = game.board()
            prompt_moves = []

            # for each move up to the sampled index, we convert it to SAN and build the prompt
            for i in range(move_index):
                move = mainline_moves[i]
                prompt_moves.append(temp_board_for_san.san(move))
                temp_board_for_san.push(move)
            prompt = " ".join(prompt_moves)

            # sft move san is the move generated by the SFT model in SAN format
            sft_move_san = get_sft_move(sft_model, tokenizer, board, prompt)
            if sft_move_san is None:
                continue

            # print debug information
            print(f"\n[DEBUG] Board FEN: {board.fen()}")
            print(f"[DEBUG] Prompt: '{prompt}'")
            print(f"[DEBUG] SFT Model Raw Output: '{sft_move_san}'")

            # try to parse the SFT move and check if it's legal
            # if not, we skip to the next game
            # else, we get the Stockfish move and compare
            # whichever move is better (Stockfish's move) is stored as the chosen move
            try:
                sft_move = board.parse_san(sft_move_san)
                if sft_move not in board.legal_moves:
                    print(f"[DEBUG] Result: REJECTED (Illegal Move)")
                    continue  # Skip if the parsed move is not legal
            except (
                chess.InvalidMoveError,
                chess.IllegalMoveError,
                chess.AmbiguousMoveError,
            ) as e:
                print(f"[DEBUG] Result: REJECTED (Parsing Error: {e})")
                continue  # Skip if the move string is malformed

            print(f"[DEBUG] Result: ACCEPTED as legal move.")

            # result is the Stockfish engine's analysis of the current board position
            # and we limit the time to 0.1 seconds for quick evaluation
            result = engine.play(board, chess.engine.Limit(time=0.1))
            stockfish_move = result.move

            # if the moves are different, we store the preference data
            # else, we skip to the next game
            if sft_move != stockfish_move:
                preference_data.append(
                    {
                        "prompt": prompt,
                        "chosen": board.san(stockfish_move),
                        "rejected": board.san(sft_move),
                    }
                )
                pbar.update(1)

        except Exception as e:
            continue

    pbar.close()
    engine.quit()

    print(f"\nGenerated {len(preference_data)} preference data samples.")
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
    parser.add_argument(
        "--min_elo",
        type=int,
        default=2000,
        help="Minimum Elo rating for a game to be considered high quality.",
    )
    args = parser.parse_args()
    main(args)


"""
    zstdcat data/raw/lichess_db_standard_rated_2024-08.pgn.zst | python -m scripts.03_generate_preference_data     --sft_model_path models/sft_model     --output_file data/processed/preference_dataset_targeted.jsonl     --num_samples 100000
"""
