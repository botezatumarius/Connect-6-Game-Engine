"""Connect6 game rules: end-of-game detection."""
from enum import IntEnum

from defines import Defines

# One direction per line axis (vertical, horizontal, diagonal, anti-diagonal);
# each axis is scanned both ways from the stone.
DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))


class GameResult(IntEnum):
    """Outcome of a position. Win values match the winner's stone colour."""
    ONGOING = 0
    BLACK_WIN = Defines.BLACK
    WHITE_WIN = Defines.WHITE
    DRAW = 3


def count_run(board, x, y, dx, dy, color):
    """Count consecutive `color` stones after (x, y), stepping by (dx, dy).

    The border ring never matches a stone colour, so the scan always stops
    inside the grid and needs no bounds checks.
    """
    count = 0
    x += dx
    y += dy
    while board[x][y] == color:
        count += 1
        x += dx
        y += dy
    return count


def is_winning_stone(board, x, y):
    """Return True if the stone at (x, y) belongs to a line of WIN_LENGTH or more."""
    color = board[x][y]
    if color != Defines.BLACK and color != Defines.WHITE:
        return False
    for dx, dy in DIRECTIONS:
        line = 1 + count_run(board, x, y, dx, dy, color) + count_run(board, x, y, -dx, -dy, color)
        if line >= Defines.WIN_LENGTH:
            return True
    return False


def is_win_by_premove(board, preMove):
    """Return True if either stone of the last move completed a winning line.

    Only the last move can create a new line, so there is no need to scan the whole board.
    """
    return any(is_winning_stone(board, pos.x, pos.y) for pos in preMove.positions)


def is_board_full(board):
    """Return True if no empty intersection is left (border cells are never empty)."""
    return not any(Defines.NOSTONE in row for row in board)


def get_game_result(board, preMove):
    """Classify the position reached after `preMove` has been played on `board`."""
    if is_win_by_premove(board, preMove):
        pos = preMove.positions[0]
        return GameResult(board[pos.x][pos.y])
    return GameResult.DRAW if is_board_full(board) else GameResult.ONGOING
