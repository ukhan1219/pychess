import chess.pgn
import argparse
import json
from tqdm import tqdm
import sys
import os

# compute the absolute path of this file then go up one directory to get project root and store it
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

"""
prepare sft data from pgn by filtering games for quality and writing jsonl of space separated move sequences
supports reading from stdin or a file and limits by max games and minimum elo
"""

# ensure python can import modules from the project root by adding it to the system path
from src.chess_utils import is_game_high_quality


def game_to_move_sequence(game):
    """
    convert a chess game to a space separated string of moves
    """
    # prepare an empty list to collect san move strings from the game
    moves = []
    board = game.board()
    
    # iterate the main line moves of the game convert each move to san append to list and push the move on the board to advance position
    for move in game.mainline_moves():
        san_move = board.san(move)
        
        moves.append(san_move)

        board.push(move)

    return " ".join(moves)


def main(args):
    """
    read games from input filter by quality and write jsonl lines of move sequences until max games is reached
    """
    # read from standard input when input file is a single dash this allows piping compressed pgn data
    if args.input_file == "-":
        print("Reading from standard input...")
        pgn_stream = sys.stdin
    # otherwise open the provided pgn file path for reading text
    else:
        print(f"Reading from {args.input_file}...")
        pgn_stream = open(args.input_file)

    processed_games = 0
    skipped_games = 0

    # open the output jsonl file using a with block which opens the file and closes it automatically when done even if an error occurs
    with open(args.output_file, "w") as output_file:
        # build an iterator that repeatedly reads the next game from the pgn stream until no game is returned
        game_iterator = iter(lambda: chess.pgn.read_game(pgn_stream), None)
        # iterate over each game and show a progress bar while processing
        for game in tqdm(game_iterator, desc="Processing games"):
            if game is None:
                continue
            # try block attempts to filter and serialize the game and the except block handles any error by skipping the game and continuing
            try:
                # check whether the game meets quality criteria such as minimum elo and metadata then if true write its move sequence to jsonl
                if is_game_high_quality(game, args.min_elo):
                    move_sequence = game_to_move_sequence(game)
                    output_file.write(json.dumps({"text": move_sequence}) + "\n")
                    processed_games += 1
                # otherwise count the game as skipped due to quality filters
                else:
                    skipped_games += 1
            except Exception as e:
                skipped_games += 1
                continue
            # stop processing once the number of processed games reaches the configured maximum
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


"""
zstdcat data/raw/lichess_db_standard_rated_2024-08.pgn.zst | python scripts/01_prepare_sft_data.py \
    --input_file - \
    --output_file data/processed/sft_dataset_filtered.jsonl \
    --max_games 1000000 \
    --min_elo 2000

"""