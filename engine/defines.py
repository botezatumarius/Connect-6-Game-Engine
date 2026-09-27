class Defines:
    ENGINE_NAME = "mbo"
    LOG_FILE = "mbo-engine.log"
    GRID_NUM = 21                    # 19x19 board plus a border ring.
    PLAYABLE = range(1, GRID_NUM - 1)  # Board indices inside the border.
    NOSTONE = 0
    BLACK = 1
    WHITE = 2
    BORDER = 3
    WIN_LENGTH = 6                   # Stones in a row needed to win.
    MAXINT = 20000                   # Score of a won game.
    MININT = -20000                  # Score of a lost game.
    CANDIDATE_RADIUS = 2             # Max distance from a stone for a candidate cell.
    MAX_CANDIDATE_CELLS = 10         # Candidate cells kept; they form at most 45 two-stone moves.
    SEARCH_DEPTH = 2                 # Plies explored by Minimax (our move + the reply).
    DEFAULT_TIME_LIMIT = 5.0         # Seconds per turn until a `depth` command sets it.
    MAX_TIME_LIMIT = 25.0            # The GUI forfeits a move that takes over 30 s.


class StonePosition:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class StoneMove:
    def __init__(self):
        self.positions = [StonePosition(0, 0), StonePosition(0, 0)]
        self.score = 0
