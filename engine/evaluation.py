"""Naive static evaluation, from Black's (first player's) point of view."""
from itertools import chain
from operator import itemgetter

from defines import Defines
from rules import DIRECTIONS, GameResult, get_game_result

TERMINAL_SCORES = {
    GameResult.BLACK_WIN: Defines.MAXINT,
    GameResult.WHITE_WIN: Defines.MININT,
    GameResult.DRAW: 0,
}
BLACK_STONE = bytes([Defines.BLACK])
WHITE_STONE = bytes([Defines.WHITE])


def build_lines():
    """Return every board line long enough to hold a winning row."""
    playable = Defines.PLAYABLE
    lines = []
    for dx, dy in DIRECTIONS:
        for x in playable:
            for y in playable:
                # Lines start where the previous cell is off the board.
                if x - dx in playable and y - dy in playable:
                    continue
                line = []
                cx, cy = x, y
                while cx in playable and cy in playable:
                    line.append((cx, cy))
                    cx += dx
                    cy += dy
                if len(line) >= Defines.WIN_LENGTH:
                    lines.append(tuple(line))
    return tuple(lines)


# Reads a line's cells from the flattened board in one call.
LINE_GETTERS = tuple(itemgetter(*(x * Defines.GRID_NUM + y for x, y in line)) for line in build_lines())


def open_windows(line, blocker):
    """Count the windows of `line` with no `blocker` stone."""
    return sum(max(0, len(segment) - Defines.WIN_LENGTH + 1) for segment in line.split(blocker))


def count_living_sets(board):
    """Black's living sets minus White's (empty windows cancel out)."""
    flat = bytes(chain.from_iterable(board))
    score = 0
    for get_line in LINE_GETTERS:
        line = bytes(get_line(flat))
        score += open_windows(line, WHITE_STONE) - open_windows(line, BLACK_STONE)
    return score


def evaluate(board, preMove):
    """MAXINT / 0 / MININT if the game is over, else the living-sets balance."""
    result = get_game_result(board, preMove)
    if result != GameResult.ONGOING:
        return TERMINAL_SCORES[result]
    return count_living_sets(board)
