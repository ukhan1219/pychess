import datetime
import os
import shutil
import platform

STOCKFISH_PATH = "./stockfish/stockfish-macos-x86-64-bmi2"
STOCKFISH_PATH_WSL = "./stockfish/stockfish-ubuntu-x86-64-avx2"

def is_game_high_quality(game, min_elo=2000):
    """
    Applies a series of quality checks to determine if a game is of high quality.
    This is the single source of truth for data quality in the project.
    """
    headers = game.headers

    if headers.get("WhiteTitle") == "BOT" or headers.get("BlackTitle") == "BOT":
        return False

    try:
        if (
            int(headers.get("WhiteElo", 0)) < min_elo
            or int(headers.get("BlackElo", 0)) < min_elo
        ):
            return False
    except (ValueError, TypeError):
        return False

    try:
        game_date_str = headers.get("UTCDate", "1970.01.01")
        game_date = datetime.datetime.strptime(game_date_str, "%Y.%m.%d").date()
        if game_date == datetime.date(2021, 3, 12):
            return False
    except ValueError:
        return False

    if headers.get("Termination", "Normal") != "Normal":
        return False

    return True

def resolve_stockfish_path():
    """Resolve a working Stockfish binary for this platform."""
    candidates = []
    # Paths relative to project root (one level up from this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    mac_bin = os.path.join(project_root, "stockfish", "stockfish-macos-x86-64-bmi2")
    linux_bin = os.path.join(project_root, "stockfish", "stockfish-ubuntu-x86-64-avx2")
    path_binary = shutil.which("stockfish")
    if path_binary:
        candidates.append(path_binary)
    system = platform.system().lower()
    if system == "darwin":
        candidates.extend([mac_bin, STOCKFISH_PATH])
    elif system == "linux":
        candidates.extend([linux_bin, STOCKFISH_PATH_WSL])
    else:
        candidates.extend([mac_bin, linux_bin, STOCKFISH_PATH, STOCKFISH_PATH_WSL])
    for cand in candidates:
        if cand and os.path.exists(cand) and os.access(cand, os.X_OK):
            return cand
    return candidates[0] if candidates else None