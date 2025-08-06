import chess.pgn
import argparse
import json
from tqdm import tqdm
import sys
import datetime
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

"""
    this script processes chess games in PGN format and prepares them for supervised fine-tuning (SFT).
    then it filters out low-quality games based on various criteria such as player titles, Elo ratings, and game dates.
    then it converts the games into a space-separated string of moves and saves them in a JSONL format.
    and it allows you to specify the maximum number of games to process and the minimum Elo rating for a game to be considered high quality.
    it also supports reading from standard input or a file.and it outputs the processed games to a specified JSONL file.
"""


from src.chess_utils import is_game_high_quality


def game_to_move_sequence(game):
    """
    Converts a chess game to a space separated string of moves.
    """

    moves = []

    board = game.board()

    # this will iterate through the mainline moves of the game
    # and push them to the board
    # mainline moves are the moves played in the game
    # mainline_moves() returns a generator of moves
    for move in game.mainline_moves():
        san_move = board.san(move)
        
        moves.append(san_move)

        board.push(move)

    return " ".join(moves)


def main(args):
    if args.input_file == "-":
        print("Reading from standard input...")
        pgn_stream = sys.stdin
    else:
        print(f"Reading from {args.input_file}...")
        pgn_stream = open(args.input_file)

    processed_games = 0
    skipped_games = 0

    # this will read the PGN file and process each game
    with open(args.output_file, "w") as output_file:
        # the iter() function creates an iterator that reads games from the PGN stream
        # pgn_stream is a file-like object that contains the PGN data
        # PGN is a standard format for recording chess games
        game_iterator = iter(lambda: chess.pgn.read_game(pgn_stream), None)
        for game in tqdm(game_iterator, desc="Processing games"):
            if game is None:
                continue

            # try block to handle any exceptions that may occur during processing
            try:
                if is_game_high_quality(game, args.min_elo):
                    # this will convert the game to a space separated string of moves
                    # and save it to the output file
                    move_sequence = game_to_move_sequence(game)
                    output_file.write(json.dumps({"text": move_sequence}) + "\n")
                    processed_games += 1
                else:
                    skipped_games += 1
            except Exception as e:
                skipped_games += 1
                continue  # Skip this game if any error occurs

            if processed_games >= args.max_games:
                print(f"\nReached max games limit of {args.max_games}.")
                break

    print(f"\n--- Processing Complete ---")
    print(f"Games processed and saved: {processed_games}")
    print(f"Games skipped by quality filters: {skipped_games}")
    print(f"Output saved to: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SFT data from PGN files.")
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to the input PGN file."
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to the output JSONL file."
    )
    parser.add_argument(
        "--max_games",
        type=int,
        default=1000,
        help="Maximum number of games to process.",
    )
    parser.add_argument(
        "--min_elo",
        type=int,
        default=2000,
        help="Minimum Elo rating for a game to be considered high quality.",
    )
    args = parser.parse_args()
    main(args)


# sample command:
# Make sure you have zstd installed (e.g., sudo apt-get install zstd)
"""
MAC:
    zstdcat data/raw/lichess_db_standard_rated_2024-08.pgn.zst | python scripts/01_prepare_sft_data.py \
        --input_file - \
        --output_file data/processed/sft_dataset_filtered.jsonl \
        --max_games 1000000 \
        --min_elo 2000

"""

