"""Engine-wide constants: board, colours, scores and search settings."""


class Defines:
    ENGINE_NAME = "mbo"
    LOG_FILE = "mbo-engine.log"
    BOARD_SIZE = 19
    NOSTONE = 0
    BLACK = 1
    WHITE = 2
    WIN_LENGTH = 6                   # Stones in a row needed to win.
    MAXINT = 20000                   # Score of a won game.
    MININT = -20000                  # Score of a lost game.
    MAX_CANDIDATE_CELLS = 10         # Candidate cells kept; they form at most 45 two-stone moves.
    SEARCH_DEPTH = 2                 # Plies explored by the search (our move + the reply).
    DEFAULT_TIME_LIMIT = 5.0         # Seconds per turn until a `depth` command sets it.
    MAX_TIME_LIMIT = 25.0            # The GUI forfeits a move that takes over 30 s.
