"""Naive static evaluation, from Black's (first player's) point of view."""
from board import WIN
from defines import Defines


def evaluate(board):
    """MAXINT / 0 / MININT if the game is over, else Black's living sets minus White's."""
    if board.winner:
        return Defines.MAXINT if board.winner == Defines.BLACK else Defines.MININT
    if board.is_full():
        return 0
    return sum(board.lines[Defines.BLACK][1:WIN]) - sum(board.lines[Defines.WHITE][1:WIN])
