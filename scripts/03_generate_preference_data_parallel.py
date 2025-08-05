import chess
import chess.pgn
import chess.engine
import argparse
import json
import sys
import re
import random
import multiprocessing as mp
import queue
import time
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging

STOCKFISH_PATH = "./stockfish/stockfish-macos-x86-64-bmi2"
STOCKFISH_PATH_WSL = "./stockfish/stockfish-ubuntu-x86-64-avx2"

from src.chess_utils import is_game_high_quality

logging.getLogger("transformers").setLevel(logging.ERROR)

def get_sft_move_batch(model, tokenizer, positions_data):
    if not positions_data:
        return []
    
    results = []
    
    batch_size = min(8, len(positions_data))
    
    for i in range(0, len(positions_data), batch_size):
        batch = positions_data[i:i + batch_size]
        batch_prompts = [prompt if prompt else "1." for _, prompt in batch]
        
        try:
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            
            with torch.no_grad():
                output_sequences = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=8,
                    num_return_sequences=8,
                    do_sample=True,
                    top_k=40,
                    top_p=0.9,
                    temperature=0.8,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            full_texts = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
            
            texts_per_prompt = len(full_texts) // len(batch_prompts)
            
            for j, (board, prompt) in enumerate(batch):
                start_idx = j * texts_per_prompt
                end_idx = start_idx + texts_per_prompt
                prompt_texts = full_texts[start_idx:end_idx]
                
                move_san = parse_move_from_texts(prompt_texts, prompt, board)
                results.append(move_san)
                
        except Exception as e:
            for board, prompt in batch:
                try:
                    move_san = get_sft_move_single(model, tokenizer, board, prompt)
                    results.append(move_san)
                except:
                    results.append(None)
    
    return results

def get_sft_move_single(model, tokenizer, board, prompt):
    if not prompt:
        prompt = "1."
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_sequences = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=8,
            num_return_sequences=8,
            do_sample=True,
            top_k=40,
            top_p=0.9,
            temperature=0.8,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    full_texts = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
    return parse_move_from_texts(full_texts, prompt, board)

def parse_move_from_texts(texts, prompt, board):
    for text in texts:
        try:
            remaining_text = text.replace(prompt, "").strip()
            
            potential_moves = []
            
            if remaining_text:
                potential_moves.append(remaining_text.split()[0])
            
            move_patterns = re.findall(r'[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[KQRBN])?[+#]?|O-O-O|O-O', remaining_text)
            potential_moves.extend(move_patterns)
            
            for move_str in potential_moves:
                if not move_str:
                    continue
                try:
                    move = board.parse_san(move_str)
                    if move in board.legal_moves:
                        return move_str
                except:
                    continue
        except:
            continue
    
    return None

def worker_process(worker_id, game_queue, result_queue, args):
    try:
        print(f"Worker {worker_id}: Starting up...")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Worker {worker_id}: Using device: {device}")
        sft_model = AutoModelForCausalLM.from_pretrained(args.sft_model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH_WSL)
        
        print(f"Worker {worker_id}: Ready to process games")
        
        games_processed = 0
        samples_found = 0
        
        while True:
            try:
                game_data = game_queue.get(timeout=10)
                if game_data is None:
                    break
                
                game, game_num = game_data
                games_processed += 1
                
                try:
                    if not is_game_high_quality(game, args.min_elo):
                        continue
                    
                    mainline_moves = list(game.mainline_moves())
                    if len(mainline_moves) < 10:
                        continue
                    
                    positions_to_process = []
                    game_length = len(mainline_moves)
                    
                    if game_length <= 20:
                        start_move = 6
                        end_move = game_length - 1
                    else:
                        start_move = max(8, game_length // 6)
                        end_move = min(40, 5 * game_length // 6)

                    if start_move >= end_move:
                        start_move = max(3, game_length // 3)
                        end_move = game_length - 1
                    
                    if start_move >= end_move:
                        continue
                        
                    move_index = random.randint(start_move, end_move)
                    
                    board = game.board()
                    for i in range(move_index):
                        board.push(mainline_moves[i])
                    
                    temp_board_for_san = game.board()
                    prompt_moves = []
                    for i in range(move_index):
                        move = mainline_moves[i]
                        prompt_moves.append(temp_board_for_san.san(move))
                        temp_board_for_san.push(move)
                    prompt = " ".join(prompt_moves)
                    
                    positions_to_process.append((board.copy(), prompt, move_index))
                    
                    if not positions_to_process:
                        continue
                    
                    batch_data = [(board, prompt) for board, prompt, _ in positions_to_process]
                    sft_moves = get_sft_move_batch(sft_model, tokenizer, batch_data)
                    
                    for (board, prompt, move_index), sft_move_san in zip(positions_to_process, sft_moves):
                        if sft_move_san is None:
                            continue
                        
                        try:
                            sft_move = board.parse_san(sft_move_san)
                            if sft_move not in board.legal_moves:
                                continue
                        except:
                            continue
                        
                        try:
                            result = engine.play(board, chess.engine.Limit(time=0.1))
                            stockfish_move = result.move
                            
                            if sft_move != stockfish_move:
                                preference_sample = {
                                    "prompt": prompt,
                                    "chosen": board.san(stockfish_move),
                                    "rejected": board.san(sft_move),
                                }
                                result_queue.put(preference_sample)
                                samples_found += 1
                        except:
                            continue
                
                except Exception as e:
                    continue
                
                if games_processed % 100 == 0:
                    print(f"Worker {worker_id}: Processed {games_processed} games, found {samples_found} samples")
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker {worker_id}: Error processing game: {e}")
                continue
        
        engine.quit()
        print(f"Worker {worker_id}: Shutting down. Processed {games_processed} games, found {samples_found} samples")
        
    except Exception as e:
        print(f"Worker {worker_id}: Fatal error: {e}")
        result_queue.put(None)

def main(args):
    print(f"Starting parallel preference data generation with {args.num_workers} workers...")
    print(f"Target: {args.num_samples} samples")
    
    game_queue = mp.Queue(maxsize=args.num_workers * 10)
    result_queue = mp.Queue()
    
    workers = []
    for i in range(args.num_workers):
        worker = mp.Process(target=worker_process, args=(i, game_queue, result_queue, args))
        worker.start()
        workers.append(worker)
    
    if args.input_games_file:
        print(f"Opening PGN file: {args.input_games_file}")
        pgn_stream = open(args.input_games_file)
    else:
        print("Reading from standard input (stdin)...")
        pgn_stream = sys.stdin
    
    def game_producer():
        games_read = 0
        try:
            while True:
                game = chess.pgn.read_game(pgn_stream)
                if game is None:
                    break
                game_queue.put((game, games_read))
                games_read += 1
                
                if games_read % 1000 == 0:
                    print(f"Read {games_read} games from input")
        except Exception as e:
            print(f"Error reading games: {e}")
        finally:
            for _ in range(args.num_workers):
                game_queue.put(None)
            print(f"Finished reading {games_read} games")
    
    import threading
    producer_thread = threading.Thread(target=game_producer)
    producer_thread.start()
    
    preference_data = []
    pbar = tqdm(total=args.num_samples, desc="Collecting samples")
    
    start_time = time.time()
    last_update = start_time
    
    try:
        while len(preference_data) < args.num_samples:
            try:
                result = result_queue.get(timeout=30)
                if result is None:
                    print("Worker error detected")
                    continue
                
                preference_data.append(result)
                pbar.update(1)
                
                current_time = time.time()
                if current_time - last_update > 30:
                    elapsed = current_time - start_time
                    rate = len(preference_data) / elapsed
                    eta = (args.num_samples - len(preference_data)) / rate if rate > 0 else 0
                    print(f"Progress: {len(preference_data)}/{args.num_samples} samples "
                          f"({rate:.1f} samples/sec, ETA: {eta/3600:.1f}h)")
                    last_update = current_time
                
            except queue.Empty:
                alive_workers = sum(1 for w in workers if w.is_alive())
                if alive_workers == 0:
                    print("All workers finished")
                    break
                continue
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    pbar.close()
    
    producer_thread.join(timeout=10)
    
    for worker in workers:
        worker.terminate()
        worker.join(timeout=5)
    
    print(f"\nGenerated {len(preference_data)} preference data samples.")
    
    with open(args.output_file, "w") as f:
        for item in preference_data:
            f.write(json.dumps(item) + "\n")
    print(f"Preference data saved to {args.output_file}")
    
    elapsed = time.time() - start_time
    print(f"Total time: {elapsed/3600:.2f} hours ({elapsed/60:.1f} minutes)")
    print(f"Average rate: {len(preference_data)/elapsed:.2f} samples/second")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(
        description="Generate preference data from chess games (parallel version)."
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
        default=750000,
        help="Number of preference data samples to generate.",
    )
    parser.add_argument(
        "--min_elo",
        type=int,
        default=2000,
        help="Minimum Elo rating for a game to be considered high quality.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=mp.cpu_count() // 2,
        help="Number of worker processes to use.",
    )
    
    args = parser.parse_args()
    main(args)
    
"""
zstdcat data/raw/lichess_db_standard_rated_2024-08.pgn.zst | python -m scripts.03_generate_preference_data_parallel \
    --sft_model_path models/sft_model \
    --output_file data/processed/preference_dataset_targeted.jsonl \
    --num_samples 750000 \
    --min_elo 2000
"""