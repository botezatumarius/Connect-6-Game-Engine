"""Naive move generation: proposes the candidate moves the search will explore."""
from itertools import combinations

from defines import Defines, StoneMove, StonePosition

PLAYABLE = Defines.PLAYABLE
CENTRE = (Defines.GRID_NUM // 2, Defines.GRID_NUM // 2)


def to_move(first, second):
    """Build a StoneMove from two (x, y) cells; equal cells mean a single stone."""
    move = StoneMove()
    move.positions = [StonePosition(*first), StonePosition(*second)]
    return move


def occupied_cells(board):
    return [(x, y) for x in PLAYABLE for y in PLAYABLE if board[x][y] != Defines.NOSTONE]


def empty_cells(board):
    return [(x, y) for x in PLAYABLE for y in PLAYABLE if board[x][y] == Defines.NOSTONE]


def candidate_cells(board, stones, limit):
    """Return up to `limit` empty cells, nearest to the existing stones first.

    Only cells within CANDIDATE_RADIUS (Chebyshev distance) of a stone are considered.
    If fewer than two are found, every empty cell is used so a move is always possible.
    """
    radius = Defines.CANDIDATE_RADIUS
    distance = {}
    for x, y in stones:
        for nx in range(max(PLAYABLE.start, x - radius), min(PLAYABLE.stop, x + radius + 1)):
            for ny in range(max(PLAYABLE.start, y - radius), min(PLAYABLE.stop, y + radius + 1)):
                if board[nx][ny] == Defines.NOSTONE:
                    d = max(abs(nx - x), abs(ny - y))
                    if d < distance.get((nx, ny), radius + 1):
                        distance[(nx, ny)] = d

    if len(distance) < 2:
        return empty_cells(board)[:limit]
    return sorted(distance, key=distance.get)[:limit]


def generate_moves(board, limit=Defines.MAX_CANDIDATE_CELLS):
    """Propose the moves to explore from `board`.

    Black opens with a single stone in the centre; every later move is a pair of
    distinct candidate cells, so at most limit * (limit - 1) / 2 moves are returned.
    """
    stones = occupied_cells(board)
    if not stones:
        return [to_move(CENTRE, CENTRE)]
    cells = candidate_cells(board, stones, limit)
    return [to_move(first, second) for first, second in combinations(cells, 2)]
