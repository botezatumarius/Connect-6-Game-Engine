"""Opening book: moves computed offline for the first turns, stored once per symmetry class.

A position and its rotations and reflections (the 8 symmetries of the square board) are the
same position, so the book stores only the canonical one: the smallest of the 8 transformed keys.
"""
from board import SIZE
from book_moves import BOOK
from defines import Defines
from tools import LETTERS, msg2move

LAST = SIZE - 1
# The 8 symmetries of the board as functions of (row, col): the 4 rotations and their mirrors.
TRANSFORMS = (
    lambda r, c: (r, c), lambda r, c: (c, LAST - r), lambda r, c: (LAST - r, LAST - c), lambda r, c: (LAST - c, r),
    lambda r, c: (r, LAST - c), lambda r, c: (c, r), lambda r, c: (LAST - r, c), lambda r, c: (LAST - c, LAST - r),
)


def cell_map(transform):
    """The permutation of the cells produced by one symmetry."""
    return tuple(r * SIZE + c for r, c in (transform(*divmod(cell, SIZE)) for cell in range(SIZE * SIZE)))


SYMMETRIES = tuple(cell_map(t) for t in TRANSFORMS)
INVERSES = tuple(tuple(sorted(range(SIZE * SIZE), key=s.__getitem__)) for s in SYMMETRIES)


def name(cells):
    """Cells as sorted protocol coordinates, e.g. 'JJJK'."""
    return "".join(sorted(LETTERS[cell // SIZE] + LETTERS[cell % SIZE] for cell in cells))


def canonical(board):
    """Canonical key of the position ('black|white' stones) and the index of its symmetry."""
    black = [cell for cell, stone in enumerate(board.cells) if stone == Defines.BLACK]
    white = [cell for cell, stone in enumerate(board.cells) if stone == Defines.WHITE]
    return min((f"{name(s[c] for c in black)}|{name(s[c] for c in white)}", i) for i, s in enumerate(SYMMETRIES))


def book_move(board):
    """The book move for this position in the board's own orientation, or None if not in the book."""
    if board.stones > Defines.BOOK_STONES:
        return None
    key, symmetry = canonical(board)
    if key not in BOOK:
        return None
    move = tuple(INVERSES[symmetry][cell] for cell in msg2move(BOOK[key]))
    return move if all(not board.cells[cell] for cell in move) else None
