"""Engine-wide constants: board, colours, scores and search settings."""


class Defines:
    ENGINE_NAME = "mbo"
    LOG_FILE = "mbo-engine.log"
    BOARD_SIZE = 19
    NOSTONE = 0
    BLACK = 1
    WHITE = 2
    WIN_LENGTH = 6                   # Stones in a row needed to win.
    MAXINT = 1000000                 # Score of a won game, above any heuristic score.
    MININT = -1000000                # Score of a lost game.
    SEVERITY = (0, 1, 4, 16, 64, 256, 1024)  # Weight of a live window by its stones.
    MAX_CANDIDATE_CELLS = 10         # Candidate cells kept; they form at most 45 two-stone moves.
    MAX_DEPTH = 8                    # Deepest iteration of the iterative deepening search.
    TIME_SAFETY = 0.8                # Share of the time limit the search may use.
    NEXT_ITERATION = 0.25            # No new iteration once this share of the limit is spent.
    DEFAULT_TIME_LIMIT = 5.0         # Seconds per turn until a `depth` command sets it.
    MAX_TIME_LIMIT = 25.0            # The GUI forfeits a move that takes over 30 s.
