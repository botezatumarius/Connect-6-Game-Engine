"""Move generation: winning moves, defences against imminent threats, then ranked candidates."""
from itertools import chain, combinations

from board import SIZE, WINDOWS, opponent
from defines import Defines

CENTRE = (SIZE // 2) * SIZE + SIZE // 2


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
        severity = Defines.SEVERITY[b or w]  # Each extra stone makes the threat 4x more severe.
        for cell in cells:
            if not board.cells[cell]:
                scores[cell] = scores.get(cell, 0) + severity
    return scores


def best_cells(board, limit):
    """The `limit` best-scored cells (any empty cells if no live window is left)."""
    scores = rank_cells(board) or {cell: 0 for cell, stone in enumerate(board.cells) if not stone}
    return sorted(scores, key=scores.get, reverse=True)[:limit], scores


def threat_gaps(board, color):
    """Empty cells of each window that `color` can complete next turn."""
    return [{cell for cell in WINDOWS[window] if not board.cells[cell]} for window in board.threats[color]]


def winning_move(board, color, cells):
    """Fill the gaps of one threat window (adding any other cell if a single stone is enough)."""
    gap = sorted(threat_gaps(board, color)[0])
    if len(gap) == 1:
        others = chain(cells, range(len(board.cells)))
        gap.append(next(cell for cell in others if cell != gap[0] and not board.cells[cell]))
    return tuple(gap)


def rank_moves(board, color, moves, scores):
    """Order moves by the strength of the threats they create, then by their cell scores.

    The strength is the number of stones the opponent needs to stop our sets of 4+ stones,
    then how many such sets the move creates.
    """
    def strength(move):
        board.play(move, color)
        key = (board.blockers(color), len(board.threats[color]))
        board.undo()
        return key + (sum(scores.get(cell, 0) for cell in move),)

    return sorted(moves, key=strength, reverse=True)


def generate_moves(board, color, limit=Defines.MAX_CANDIDATE_CELLS):
    """Moves for `color` ordered from best to worst."""
    if not board.stones:
        return [(CENTRE,)]  # Black's first move is a single stone.

    cells, scores = best_cells(board, limit)
    # 1. We can win now: nothing else is worth exploring.
    if board.threats[color]:
        return [winning_move(board, color, cells)]

    # 2. The opponent threatens to win next turn: only moves that block every threat.
    gaps = threat_gaps(board, opponent(color))
    if gaps:
        candidates = set().union(*gaps, cells)
        moves = [move for move in combinations(candidates, 2) if all(gap & set(move) for gap in gaps)]
        # Three or more blockers are needed: the game is lost, block what we can.
        return rank_moves(board, color, moves or list(combinations(candidates, 2)), scores)

    # 3. Pairs of the best cells, moves that create the strongest threats first.
    return rank_moves(board, color, combinations(cells, 2), scores)
