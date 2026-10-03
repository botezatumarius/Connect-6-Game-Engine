"""Move generation: candidates around live stones, ranked by the severity of their threats."""
from itertools import combinations

from board import SIZE, WINDOWS
from defines import Defines

CENTRE = (SIZE // 2) * SIZE + SIZE // 2
# Priority of a window by its stones: each extra stone makes the threat four times more severe.
SEVERITY = (0, 1, 4, 16, 64, 256, 1024)


def rank_cells(board):
    """Score every empty cell of a live window (stones of one colour only) by the windows through it.

    Windows of either colour count, so a cell scores both its attack value (our windows)
    and its defence value (the opponent's windows it would block).
    """
    black, white = board.counts[Defines.BLACK], board.counts[Defines.WHITE]
    scores = {}
    for window, cells in enumerate(WINDOWS):
        b, w = black[window], white[window]
        if (b == 0) == (w == 0):
            continue  # Empty window, or dead because it holds both colours.
        severity = SEVERITY[b or w]
        for cell in cells:
            if not board.cells[cell]:
                scores[cell] = scores.get(cell, 0) + severity
    return scores


def generate_moves(board, limit=Defines.MAX_CANDIDATE_CELLS):
    """Moves ordered from best to worst: pairs of the best cells, by their combined score."""
    if not board.stones:
        return [(CENTRE,)]  # Black's first move is a single stone.

    scores = rank_cells(board)
    if len(scores) < 2:
        # No live window left: any empty cells will do.
        scores = {cell: 0 for cell, stone in enumerate(board.cells) if not stone}
    cells = sorted(scores, key=scores.get, reverse=True)[:limit]
    pairs = combinations(cells, 2)
    return sorted(pairs, key=lambda pair: scores[pair[0]] + scores[pair[1]], reverse=True)
