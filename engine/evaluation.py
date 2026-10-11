"""Evaluation function: a weighted sum of position features, seen by the side to move.

The score is returned from Black's point of view, as the Minimax search expects.
"""
from board import opponent
from defines import Defines

# Features, for the side to move ("own") and the other player ("opp"):
#   own_1..own_3  our live sets (windows with only our stones) holding 1, 2 or 3 stones
#   opp_1..opp_3  the opponent's live sets holding 1, 2 or 3 stones
#   opp_blockers  stones we must spend to stop the opponent's threats (1 or 2)
FEATURES = ("own_1", "own_2", "own_3", "opp_1", "opp_2", "opp_3", "opp_blockers")
WEIGHTS = (1, 4, 16, -1, -4, -16, -128)  # Hand-tuned in session 3.


def features(board, color):
    """Feature vector of a quiet position with `color` to move."""
    own, opp = board.lines[color], board.lines[opponent(color)]
    return own[1], own[2], own[3], opp[1], opp[2], opp[3], board.blockers(opponent(color))


def evaluate(board, color, weights=WEIGHTS):
    """Score of the position with `color` to move, from Black's point of view."""
    # Finished game: MAX for a black win, 0 for a draw, MIN for a white win.
    if board.winner:
        return Defines.MAXINT if board.winner == Defines.BLACK else Defines.MININT
    if board.is_full():
        return 0

    sign = 1 if color == Defines.BLACK else -1
    # Imminent results, just below a finished game: the side to move completes a threat,
    # or cannot block the opponent's threats with its two stones.
    if board.threats[color]:
        return sign * (Defines.MAXINT - 1)
    if board.blockers(opponent(color)) >= 3:
        return -sign * (Defines.MAXINT - 2)

    return sign * sum(w * f for w, f in zip(weights, features(board, color)))
