"""Static evaluation from Black's (first player's) point of view."""
from board import WIN
from defines import Defines

# Weight of the stones the opponent needs to kill a player's winning sets (0 to 3 stones).
KILL_WEIGHTS = (0, 128, 1024, 8192)


def evaluate(board):
    """MAXINT / 0 / MININT if the game is over, else Black's weighted live sets minus White's.

    Live sets (windows with stones of one colour only) are weighted by their stones, and the
    sets that win next turn also by the stones the opponent must use to kill them.
    """
    if board.winner:
        return Defines.MAXINT if board.winner == Defines.BLACK else Defines.MININT
    if board.is_full():
        return 0
    score = 0
    for color, sign in ((Defines.BLACK, 1), (Defines.WHITE, -1)):
        lines = board.lines[color]
        sets = sum(Defines.SEVERITY[stones] * lines[stones] for stones in range(1, WIN))
        score += sign * (sets + KILL_WEIGHTS[board.blockers(color)])
    return score
