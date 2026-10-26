"""Threat-space search: forced wins made only of moves that create threats.

After a move that creates threats the opponent must block them all, so it has only a few
replies; following such moves alone proves wins far beyond the depth of the full search
(threat-space search, Allis et al.; threats as defined by Wu and Huang for Connect6).
The search is exact for wins within 3 turns, which is the horizon used by the engine.
"""
from itertools import combinations

from board import WINDOWS, opponent


def attack_cells(board, color):
    """Empty cells of live windows where `color` has 2+ stones: the only cells that make fours."""
    counts, other = board.counts[color], board.counts[opponent(color)]
    return sorted({cell for window, cells in enumerate(WINDOWS) if counts[window] >= 2 and not other[window]
                   for cell in cells if not board.cells[cell]})


def winning_attack(board, color, turns, tick=None):
    """A move with which `color` (to move) wins within `turns` turns whatever the defence, or None.

    `tick` is called regularly so the caller can stop a search that runs out of time.
    """
    if turns < 2 or board.threats[color] or board.threats[opponent(color)]:
        return None  # Immediate wins and forced defences are handled by the normal search.
    for move in combinations(attack_cells(board, color), 2):
        if tick:
            tick()
        board.play(move, color)
        try:
            blockers = board.blockers(color)
            # Three blockers cannot be placed with two stones; with fewer, every defence must lose.
            if blockers >= 3 or (blockers and turns > 2 and all_defences_lose(board, color, turns - 1, tick)):
                return move
        finally:
            board.undo()
    return None


def wins_in(board, color, turns, tick=None):
    """True if `color`, to move, has a forced win within `turns` turns."""
    if board.threats[color]:
        return True  # A four or five is completed with one move.
    return winning_attack(board, color, turns, tick) is not None


def all_defences_lose(board, color, turns, tick=None):
    """After a threat by `color`, every defence that blocks it still loses within `turns` turns."""
    defender = opponent(color)
    gaps = [{cell for cell in WINDOWS[w] if not board.cells[cell]} for w in board.threats[color]]
    # A defending stone only matters where it blocks a threat or a future four of the attacker.
    cells = sorted(set().union(*gaps, attack_cells(board, color)))
    defences = [move for move in combinations(cells, 2) if all(gap & set(move) for gap in gaps)]
    defences += [(cell,) for cell in cells if all(cell in gap for gap in gaps)]  # Block + a far stone.
    for defence in defences:
        board.play(defence, defender)
        try:
            if board.threats[defender] or not wins_in(board, color, turns, tick):
                return False
        finally:
            board.undo()
    return True
