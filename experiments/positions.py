"""Test positions of session 5, and a threat-space prover that checks each one.

Attack positions are forced wins in 1, 2 or 3 turns for the side to move. Defence positions give
the opponent such a threat while it is our turn, so we must defend. Every position is written as
the move history the GUI would send.
"""
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))

from board import WINDOWS, Board, opponent  # noqa: E402
from tools import msg2move  # noqa: E402

BLACK, WHITE = 1, 2


def history(black, white, to_move):
    """A legal move history placing `black` and `white` stones, ending with `to_move` to play."""
    black, white = list(black), list(white)
    # Black plays one stone, then both sides two: black has an odd count, white one more or one less.
    assert len(black) % 2 == 1 and len(white) == len(black) + (1 if to_move == BLACK else -1), "bad counts"
    moves = [(BLACK, black.pop(0))]
    while black or white:
        color, stones = (WHITE, white) if moves[-1][0] == BLACK else (BLACK, black)
        moves.append((color, stones.pop(0) + stones.pop(0)))
    return moves


def position(black, white, to_move):
    """A position from space-separated cell lists such as 'JJ JK'."""
    return history(black.split(), white.split(), to_move)


# name: (history, turns until the forced win); the attacker is the side to move.
ATTACKS = {
    "win in 1 (black)": (position("JJ JF JG JH JI KK LL", "AA AS SA SS AJ SJ JA JS", BLACK), 1),
    "win in 1 (white)": (position("AA AS SA SS AJ SJ JA", "JE JF JG JH KK LL", WHITE), 1),
    "win in 2 (black)": (position("JJ FE FF FG NK NL NM", "AA AS SA SS AJ SJ JA JS", BLACK), 2),
    "win in 2 (white)": (position("AA AS SA SS AJ SJ JA", "FE FF FG NK NL NM", WHITE), 2),
    "win in 3 (black)": (position("HM IG IH JJ KH LH MJ", "AA AS SA SS AJ SJ JA JS", BLACK), 3),
    "win in 3 (white)": (position("AA AS SA SS AJ SJ JA", "HJ KG KH KL LJ ML", WHITE), 3),
}

# name: (history, turns until the threatened win); the defender is the side to move.
DEFENCES = {
    "defend in 1 (white)": (position("JJ JF JG JH JI KK LL", "AA AS SA SS AJ SJ", WHITE), 1),
    "defend in 1 (black)": (position("AA AS SA SS AJ", "JE JF JG JH KK LL", BLACK), 1),
    "defend in 2 (white)": (position("JJ FE FF FG NK NL NM", "AA AS SA SS AJ SJ", WHITE), 2),
    "defend in 2 (black)": (position("AA AS SA SS AJ", "FE FF FG NK NL NM", BLACK), 2),
    "defend in 3 (white)": (position("HM IG IH JJ KH LH MJ", "AA AS SA SS AJ SJ", WHITE), 3),
    "defend in 3 (black)": (position("AA AS SA SS AJ", "HJ KG KH KL LJ ML", BLACK), 3),
}


def board_of(moves):
    """Board with the stones of a history."""
    board = Board()
    for color, move in moves:
        board.play(msg2move(move), color)
    return board


def attack_cells(board, color):
    """Empty cells of live windows where `color` has at least 2 stones (the only cells that make fours)."""
    counts, other = board.counts[color], board.counts[opponent(color)]
    return sorted({cell for window, cells in enumerate(WINDOWS) if counts[window] >= 2 and not other[window]
                   for cell in cells if not board.cells[cell]})


def wins_in(board, color, turns):
    """True if `color`, to move, has a forced win within `turns` turns (threat-space proof).

    Exact for wins within 3 turns, the longest in the battery: a defender stone can only matter
    where the attacker could still make a four, i.e. in a window holding two or more of its stones.
    """
    if board.threats[color]:
        return True  # A four or five is completed with one move.
    if turns == 1 or board.threats[opponent(color)]:
        return False  # No forcing attack while the opponent threatens to win.
    for move in combinations(attack_cells(board, color), 2):
        board.play(move, color)
        try:
            blockers = board.blockers(color)
            if blockers >= 3:
                return True  # The opponent cannot block every threat.
            if blockers and turns > 2 and all_defences_lose(board, color, turns - 1):
                return True
        finally:
            board.undo()
    return False


def all_defences_lose(board, color, turns):
    """After a threat by `color`, every defence that blocks it still loses within `turns` turns."""
    defender = opponent(color)
    gaps = [{cell for cell in WINDOWS[w] if not board.cells[cell]} for w in board.threats[color]]
    cells = sorted(set().union(*gaps, attack_cells(board, color)))
    # Two blocking stones, or (for a single threat) one block plus a stone that changes nothing.
    defences = [move for move in combinations(cells, 2) if all(gap & set(move) for gap in gaps)]
    defences += [(cell,) for cell in cells if all(cell in gap for gap in gaps)]
    for defence in defences:
        board.play(defence, defender)
        try:
            if board.threats[defender] or not wins_in(board, color, turns):
                return False
        finally:
            board.undo()
    return True


def exact_win(moves, color, turns):
    """True if `color` wins in exactly `turns` turns: forced within `turns`, not within one fewer."""
    board = board_of(moves)
    return wins_in(board, color, turns) and (turns == 1 or not wins_in(board, color, turns - 1))


def defence_exists(moves, attacker, turns):
    """True if the side to move has a reply after which `attacker` has no forced win within `turns`."""
    board = board_of(moves)
    for move in combinations(attack_cells(board, attacker), 2):
        board.play(move, opponent(attacker))
        try:
            if not wins_in(board, attacker, turns):
                return True
        finally:
            board.undo()
    return False


def to_move(moves):
    """Colour to play after a history."""
    return opponent(moves[-1][0])


if __name__ == "__main__":
    # Certify the battery: exact attack distances, real threats and existing defences.
    for name, (moves, turns) in ATTACKS.items():
        print(f"{name:20s} forced win in exactly {turns}: {exact_win(moves, to_move(moves), turns)}")
    for name, (moves, turns) in DEFENCES.items():
        attacker = moves[-1][0]
        threat = wins_in(board_of(moves), attacker, turns)
        print(f"{name:20s} threatened win in {turns}: {threat}, defence exists: {defence_exists(moves, attacker, turns)}")
