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
import os
import glob
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
import psutil

"""
generate preference data by comparing sft model moves to stockfish using parallel workers and write jsonl samples
supports checkpointing resuming and basic memory management
"""
# paths to stockfish engine binaries that the script will launch for move evaluation
STOCKFISH_PATH = "./stockfish/stockfish-macos-x86-64-bmi2"
STOCKFISH_PATH_WSL = "./stockfish/stockfish-ubuntu-x86-64-avx2"

# import the helper that checks if a chess game meets quality requirements such as min elo
from src.chess_utils import is_game_high_quality

# set transformer logging to error to reduce console noise during heavy generation
logging.getLogger("transformers").setLevel(logging.ERROR)

def save_checkpoint(preference_data, output_file, checkpoint_num):
    """
    save preference data to a checkpoint file and return its path
    """
    # create a checkpoint filename by appending a zero padded number to the base output path
    checkpoint_file = f"{output_file}.checkpoint_{checkpoint_num:03d}"
    # open the checkpoint file using with which opens the file and guarantees it closes when the block ends even on error
    with open(checkpoint_file, "w") as f:
        # iterate over preference items and write each as one json line so resume is simple
        for item in preference_data:
            f.write(json.dumps(item) + "\n")
    print(f"Checkpoint saved: {checkpoint_file} ({len(preference_data)} samples)")
    return checkpoint_file

def cleanup_old_checkpoints(output_file, keep_last=3):
    """
    remove old checkpoint files keeping only the last n
    """
    # find all checkpoint files that match the naming pattern for this output
    checkpoint_pattern = f"{output_file}.checkpoint_*"
    checkpoint_files = sorted(glob.glob(checkpoint_pattern))
    # if there are more than keep last checkpoints remove the older ones and keep the newest
    if len(checkpoint_files) > keep_last:
        # select all but the last keep last files in sorted order as candidates for deletion
        files_to_remove = checkpoint_files[:-keep_last]
        # try to remove each old file and continue on filesystem errors
        for file_path in files_to_remove:
            # try except ensures a failure to delete one file does not stop cleanup
            try:
                os.remove(file_path)
                print(f"Removed old checkpoint: {file_path}")
            # if deleting a checkpoint fails catch the os error and warn so cleanup continues
            except OSError as e:
                print(f"Warning: Could not remove {file_path}: {e}")

def load_existing_checkpoints(output_file):
    """
    load the most recent checkpoint if available and return its data list
    """
    # build a glob pattern to search for existing checkpoint files on disk
    checkpoint_pattern = f"{output_file}.checkpoint_*"
    # list files matching the pattern and sort to get chronological order
    checkpoint_files = sorted(glob.glob(checkpoint_pattern))
    # return empty list when no checkpoints found so generation starts fresh
    if not checkpoint_files:
        return []
    # pick the newest checkpoint for resuming work
    latest_checkpoint = checkpoint_files[-1]
    print(f"Loading existing checkpoint: {latest_checkpoint}")
    
    # prepare a list to accumulate loaded preference records
    preference_data = []
    # read the checkpoint safely with try except to ignore parse errors and proceed
    try:
        # open the file using with so it closes automatically at block end
        with open(latest_checkpoint, "r") as f:
            # iterate lines strip whitespace skip empty lines and parse json into python dicts
            for line in f:
                line = line.strip()
                # only process non empty lines to avoid json decode errors
                if line:
                    preference_data.append(json.loads(line))
        print(f"Loaded {len(preference_data)} samples from checkpoint")
    # on any io or json error report and return empty list to start cleanly
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return []
    
    return preference_data

def get_sft_move_batch(model, tokenizer, positions_data):
    """
    generate candidate moves from sft model for a batch of positions and return a parsed legal san move or none per position
    """
    # if there are no positions to process return an empty results list immediately
    if not positions_data:
        return []
    
    results = []
    # choose a batch size up to eight to balance speed and memory
    batch_size = min(8, len(positions_data))
    
    # process positions in chunks of the selected batch size
    for i in range(0, len(positions_data), batch_size):
        # select a slice of positions for this chunk and create prompts defaulting to one dot when missing
        batch = positions_data[i:i + batch_size]
        batch_prompts = [prompt if prompt else "1." for _, prompt in batch]
        # tokenize the batch of prompts move tensors to device and prepare for generation
        try:
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            # run generation without tracking gradients to save memory
            with torch.no_grad():
                # attempt batched generation with sampling parameters
                try:
                    output_sequences = model.generate(
                        input_ids=inputs["input_ids"],           # prompt token ids to condition generation
                        attention_mask=inputs["attention_mask"], # mask indicating which tokens to attend to (1) vs pad (0)
                        max_new_tokens=8,                         # cap on number of tokens to generate beyond the prompt
                        num_return_sequences=8,                    # how many sampled continuations to return per prompt
                        do_sample=True,                            # enable stochastic sampling rather than greedy/beam search
                        top_k=40,                                  # restrict sampling to the top-k highest-probability tokens
                        top_p=0.9,                                 # nucleus sampling threshold: sample from minimal set with prob ≥ p
                        temperature=0.8,                           # softmax temperature; <1 makes distribution sharper
                        repetition_penalty=1.1,                    # penalize repeating tokens to encourage diversity
                        pad_token_id=tokenizer.eos_token_id,       # token id used for padding shorter sequences in a batch
                    )
                except RuntimeError as cuda_error:
                    if "CUDA" in str(cuda_error):
                        print(f"CUDA error in generation, retrying with smaller batch: {cuda_error}")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        raise cuda_error
                    else:
                        raise
            
            # decode generated ids into text and compute how many texts belong to each prompt
            full_texts = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
            texts_per_prompt = len(full_texts) // len(batch_prompts)
            
            # for each item slice its texts parse a san move relative to the board and record it
            for j, (board, prompt) in enumerate(batch):
                start_idx = j * texts_per_prompt
                end_idx = start_idx + texts_per_prompt
                prompt_texts = full_texts[start_idx:end_idx]
                # extract a legal san move from the generated texts if available
                move_san = parse_move_from_texts(prompt_texts, prompt, board)
                results.append(move_san)
            
            del inputs, output_sequences, full_texts
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                except RuntimeError:
                    pass  
        # on any error during batch generation fall back to per item generation and clean up gpu memory if present
        except Exception as e:
            if torch.cuda.is_available():
                # best effort cleanup to free memory before retrying
                try:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                except RuntimeError:
                    pass
            for board, prompt in batch:
                # try single suggestion per board handling failures by appending none
                try:
                    move_san = get_sft_move_single(model, tokenizer, board, prompt)
                    results.append(move_san)
                except:
                    results.append(None)
    
    return results

def get_sft_move_single(model, tokenizer, board, prompt):
    """
    generate candidate moves from sft model for a single position and return a parsed legal san move or none
    """
    if not prompt:
        prompt = "1."
    # tokenize the prompt into tensors and transfer them to the device for generation
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_sequences = model.generate(
            input_ids=inputs["input_ids"],     # prompt token ids to condition generation
            max_new_tokens=8,                   # cap on number of newly generated tokens
            num_return_sequences=8,             # number of sampled continuations to return
            do_sample=True,                     # enable stochastic sampling
            top_k=40,                           # restrict next-token choices to top-k
            top_p=0.9,                          # nucleus sampling threshold
            temperature=0.8,                    # sampling temperature to control randomness
            repetition_penalty=1.1,             # discourage repeated tokens
            pad_token_id=tokenizer.eos_token_id,# padding id for batched decoding safety
        )
    full_texts = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
    result = parse_move_from_texts(full_texts, prompt, board)
    
    del inputs, output_sequences, full_texts
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except RuntimeError:
            pass
    
    return result

def parse_move_from_texts(texts, prompt, board):
    """
    extract the first legal san move from generated texts given the prompt and current board
    """
    for text in texts:
        # try to parse a legal move from the generated text while guarding against malformed strings
        try:
            remaining_text = text.replace(prompt, "").strip()
            
            potential_moves = []
            if remaining_text:
                potential_moves.append(remaining_text.split()[0])
            move_patterns = re.findall(r'[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[KQRBN])?[+#]?|O-O-O|O-O', remaining_text)
            potential_moves.extend(move_patterns)
            for move_str in potential_moves:
                # skip empty candidates and try to parse san using the current board
                if not move_str:
                    continue
                try:
                    move = board.parse_san(move_str)
                    # if the parsed move is in the set of legal moves return its san text
                    if move in board.legal_moves:
                        return move_str
                except:
                    continue
        except:
            continue
    
    return None

def worker_process(worker_id, game_queue, result_queue, args):
    """
    worker loop that loads models reads games from queue generates sft moves compares with stockfish and sends preference samples to the result queue
    """
    try:
        print(f"Worker {worker_id}: Starting up...")
        # choose cuda when available otherwise cpu for running the model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        use_gpu = torch.cuda.is_available()
        
        print(f"Worker {worker_id}: Using device: {device}")
        # if gpu is available set the device to the worker id modulo the number of gpus
        if use_gpu:
            gpu_id = worker_id % torch.cuda.device_count()
            torch.cuda.set_device(gpu_id)
            torch.cuda.empty_cache()
            _ = torch.tensor([1.0]).cuda()
        # try to load the sft model on the selected device
        try:
            sft_model = AutoModelForCausalLM.from_pretrained(args.sft_model_path).to(device)
            # if gpu is available synchronize to ensure all operations are complete
            if use_gpu:
                torch.cuda.synchronize()
        # if loading the model on gpu fails fall back to cpu and warn
        except Exception as e:
            print(f"Worker {worker_id}: Failed to load model on GPU, falling back to CPU: {e}")
            device = "cpu"
            use_gpu = False
            sft_model = AutoModelForCausalLM.from_pretrained(args.sft_model_path).to(device)
        # load the tokenizer for the sft model
        tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path)
        # ensure the tokenizer has a pad token set using eos and align the model pad token id to avoid generation warnings
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # launch the stockfish engine for move evaluation
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH_WSL)
        
        print(f"Worker {worker_id}: Ready to process games")
        # initialize counters for tracking progress and memory cleanup
        games_processed = 0
        samples_found = 0
        memory_cleanup_counter = 0
        
        # main loop that processes games from the queue
        while True:
            # get the next game from the queue with a timeout to avoid blocking
            try:
                game_data = game_queue.get(timeout=10)
                # if the game data is None break the loop to signal termination
                if game_data is None:
                    break
                # unpack the game data into the game object and the game number
                game, game_num = game_data
                games_processed += 1
                memory_cleanup_counter += 1
                
                # if the memory cleanup counter reaches 50 clean up gpu memory
                if memory_cleanup_counter >= 50:
                    # if gpu is available synchronize and empty the cache
                    if use_gpu:
                        try:
                            torch.cuda.synchronize()  
                            torch.cuda.empty_cache()
                        except RuntimeError as e:
                            print(f"Worker {worker_id}: CUDA cleanup error: {e}")
                    # reset the memory cleanup counter
                    memory_cleanup_counter = 0
                
                # try to process the game
                try:
                    # skip low quality or very short games
                    if not is_game_high_quality(game, args.min_elo):
                        continue
                    # get the mainline moves from the game
                    mainline_moves = list(game.mainline_moves())
                    # if the game is too short skip it
                    if len(mainline_moves) < 10:
                        continue
                    
                    # initialize a list to store the positions to process
                    positions_to_process = []
                    game_length = len(mainline_moves)
                    
                    # if the game is too short skip it
                    if game_length <= 20:
                        start_move = 6
                        end_move = game_length - 1
                    # if the game is long enough set the start and end moves
                    else:
                        start_move = max(8, game_length // 6)
                        end_move = min(40, 5 * game_length // 6)

                    # if the start move is greater than the end move skip it
                    if start_move >= end_move:
                        start_move = max(3, game_length // 3)
                        end_move = game_length - 1
                    
                    # if the start move is greater than the end move skip it
                    if start_move >= end_move:
                        continue
                        
                    move_index = random.randint(start_move, end_move)
                    
                    board = game.board()
                    # push the mainline moves onto the board
                    for i in range(move_index):
                        board.push(mainline_moves[i])
                    # create a temporary board for san moves
                    temp_board_for_san = game.board()
                    prompt_moves = []
                    # create a list of prompt moves
                    for i in range(move_index):
                        move = mainline_moves[i]
                        prompt_moves.append(temp_board_for_san.san(move))
                        temp_board_for_san.push(move)

                    # join the prompt moves into a string
                    prompt = " ".join(prompt_moves)
                    # add the board and prompt to the list of positions to process
                    positions_to_process.append((board.copy(), prompt, move_index))
                    # if there are no positions to process skip it
                    if not positions_to_process:
                        continue
                    
                    # create a batch of data for the sft model
                    batch_data = [(board, prompt) for board, prompt, _ in positions_to_process]
                    sft_moves = get_sft_move_batch(sft_model, tokenizer, batch_data)
                    
                    # for each position in the batch get the sft move
                    for (board, prompt, move_index), sft_move_san in zip(positions_to_process, sft_moves):
                        if sft_move_san is None:
                            continue
                        # try to parse the sft move as a san move
                        try:
                            sft_move = board.parse_san(sft_move_san)
                            # if the sft move is not legal skip it
                            if sft_move not in board.legal_moves:
                                continue
                        # if parsing the sft move fails skip it
                        except:
                            continue
                        # get the stockfish move
                        try:
                            result = engine.play(board, chess.engine.Limit(time=0.1, depth=10))
                            stockfish_move = result.move
                            # if the sft move is not the same as the stockfish move add to the preference sample
                            if sft_move != stockfish_move:
                                preference_sample = {
                                    "prompt": prompt,
                                    "chosen": board.san(stockfish_move),
                                    "rejected": board.san(sft_move),
                                }
                                result_queue.put(preference_sample)
                                samples_found += 1
                        # if getting the stockfish move fails skip it
                        except:
                            continue
                # if there is an error processing the game skip it
                except Exception as e:
                    continue
                # if the games processed is a multiple of 100 print the progress
                if games_processed % 100 == 0:
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    gpu_memory_str = ""
                    # if gpu is available get the gpu memory
                    if use_gpu:
                        try:
                            gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                            gpu_memory_str = f", GPU: {gpu_memory_mb:.0f}MB"
                        # if getting the gpu memory fails skip it
                        except RuntimeError:
                            gpu_memory_str = ", GPU: Error"
                    
                    print(f"Worker {worker_id}: Processed {games_processed} games, found {samples_found} samples "
                          f"(RAM: {memory_mb:.0f}MB{gpu_memory_str})")
            # if the queue is empty skip it
            except queue.Empty:
                continue
            # if there is an error processing the game skip it
            except Exception as e:
                print(f"Worker {worker_id}: Error processing game: {e}")
                continue
        # quit the stockfish engine
        engine.quit()
        print(f"Worker {worker_id}: Shutting down. Processed {games_processed} games, found {samples_found} samples")
    # if there is an error in the worker process skip it
    except Exception as e:
        print(f"Worker {worker_id}: Fatal error: {e}")
        result_queue.put(None)

def main(args):
    """
    coordinate multiprocessing for generating preference data handle reading games progress reporting checkpointing and saving output
    """
    print(f"Starting parallel preference data generation with {args.num_workers} workers...")
    print(f"Target: {args.num_samples} samples")
    
    preference_data = load_existing_checkpoints(args.output_file)
    starting_samples = len(preference_data)
    
    # if there are existing samples load them
    if starting_samples > 0:
        print(f"Resuming from checkpoint with {starting_samples} existing samples")
        # if the existing samples are greater than the target return
        if starting_samples >= args.num_samples:
            print(f"Already have {starting_samples} samples, which meets the target of {args.num_samples}")
            return
    # create a queue for the games and a queue for the results
    game_queue = mp.Queue(maxsize=args.num_workers * 10)
    result_queue = mp.Queue()
    workers = []

    # start the worker processes
    for i in range(args.num_workers):
        worker = mp.Process(target=worker_process, args=(i, game_queue, result_queue, args))
        worker.start()
        workers.append(worker)
    # if there is an input file open it
    if args.input_games_file:
        print(f"Opening PGN file: {args.input_games_file}")
        pgn_stream = open(args.input_games_file)
    # if there is no input file read from stdin
    else:
        print("Reading from standard input (stdin)...")
        pgn_stream = sys.stdin
    # define a function to read games from the input stream
    def game_producer():
        games_read = 0
        # read games from the input stream
        try:
            # read games from the input stream
            while True:
                game = chess.pgn.read_game(pgn_stream)
                # if the game is None break the loop
                if game is None:
                    break
                game_queue.put((game, games_read))
                games_read += 1
                # if the games read is a multiple of 1000 print the progress
                if games_read % 1000 == 0:
                    print(f"Read {games_read} games from input")
        # if there is an error reading the games skip it
        except Exception as e:
            print(f"Error reading games: {e}")
        # finally put None into the queue to signal termination
        finally:
            # put None into the queue to signal termination
            for _ in range(args.num_workers):
                game_queue.put(None)
            print(f"Finished reading {games_read} games")
    # start the game producer thread
    import threading
    producer_thread = threading.Thread(target=game_producer)
    producer_thread.start()
    pbar = tqdm(total=args.num_samples, initial=starting_samples, desc="Collecting samples")
    # start the time
    start_time = time.time()
    last_update = start_time
    last_checkpoint = starting_samples
    checkpoint_interval = args.checkpoint_interval
    # try to collect samples
    try:
        # while the number of samples is less than the target
        while len(preference_data) < args.num_samples:
            # try to get a result from the queue
            try:
                result = result_queue.get(timeout=30)
                # if the result is None print an error and continue
                if result is None:
                    print("Worker error detected")
                    continue
                
                preference_data.append(result)
                pbar.update(1)
                # if the number of samples is a multiple of the checkpoint interval save a checkpoint
                if len(preference_data) - last_checkpoint >= checkpoint_interval:
                    checkpoint_num = len(preference_data) // checkpoint_interval
                    save_checkpoint(preference_data, args.output_file, checkpoint_num)
                    cleanup_old_checkpoints(args.output_file, keep_last=3)
                    last_checkpoint = len(preference_data)
                
                current_time = time.time()

                # if the time since the last update is greater than 30 seconds print the progress
                if current_time - last_update > 30:
                    elapsed = current_time - start_time
                    rate = len(preference_data) / elapsed if elapsed > 0 else 0
                    eta = (args.num_samples - len(preference_data)) / rate if rate > 0 else 0
                    print(f"Progress: {len(preference_data)}/{args.num_samples} samples "
                          f"({rate:.1f} samples/sec, ETA: {eta/3600:.1f}h)")
                    last_update = current_time
            # if the queue is empty skip it
            except queue.Empty:
                alive_workers = sum(1 for w in workers if w.is_alive())
                # if all workers are finished break the loop
                if alive_workers == 0:
                    print("All workers finished")
                    break
                continue
    # if the user interrupts skip it
    except KeyboardInterrupt:
        print("Interrupted by user")
        # if the number of samples is greater than the last checkpoint save a checkpoint
        if len(preference_data) > last_checkpoint:
            checkpoint_num = (len(preference_data) // checkpoint_interval) + 1
            save_checkpoint(preference_data, args.output_file, checkpoint_num)
            cleanup_old_checkpoints(args.output_file, keep_last=3)
    # close the progress bar
    pbar.close()
    producer_thread.join(timeout=10)
    
    # terminate the workers
    for worker in workers:
        worker.terminate()
        worker.join(timeout=5)
    
    print(f"\nGenerated {len(preference_data)} preference data samples.")
    
    # save the preference data to the output file
    with open(args.output_file, "w") as f:
        # write each item to the file
        for item in preference_data:
            f.write(json.dumps(item) + "\n")
    print(f"Preference data saved to {args.output_file}")
    
    # if the number of samples is greater than the last checkpoint save a checkpoint
    if len(preference_data) > last_checkpoint:
        final_checkpoint_num = (len(preference_data) - 1) // checkpoint_interval + 1
        save_checkpoint(preference_data, args.output_file, final_checkpoint_num)
        cleanup_old_checkpoints(args.output_file, keep_last=3)
    
    elapsed = time.time() - start_time
    print(f"Total time: {elapsed/3600:.2f} hours ({elapsed/60:.1f} minutes)")
    if elapsed > 0:
        print(f"Average rate: {len(preference_data)/elapsed:.2f} samples/second")

if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  
    
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
        default=min(4, mp.cpu_count() // 2),  
        help="Number of worker processes to use.",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=10000,
        help="Save checkpoint every N samples (default: 10000).",
    )
    
    args = parser.parse_args()
    main(args)
    
"""
zstdcat data/raw/lichess_db_standard_rated_2024-08.pgn.zst | python -m scripts.03_generate_preference_data_parallel \
    --sft_model_path models/sft_model \
    --output_file data/processed/preference_dataset_targeted.jsonl \
    --num_samples 500000 \
    --min_elo 2000 \
    --checkpoint_interval 10000
"""