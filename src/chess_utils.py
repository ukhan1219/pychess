import datetime

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