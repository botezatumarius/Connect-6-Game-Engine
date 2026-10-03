"""Naive move generation: proposes the candidate moves the search will explore."""
from itertools import combinations

from board import SIZE
from defines import Defines

CENTRE = (SIZE // 2) * SIZE + SIZE // 2


def build_neighbours(radius):
    """For each cell, the cells within `radius` (Chebyshev distance) and their distance."""
    neighbours = []
    for row in range(SIZE):
        for col in range(SIZE):
            neighbours.append(tuple(
                (r * SIZE + c, max(abs(r - row), abs(c - col)))
                for r in range(max(0, row - radius), min(SIZE, row + radius + 1))
                for c in range(max(0, col - radius), min(SIZE, col + radius + 1))
                if (r, c) != (row, col)))
    return tuple(neighbours)


NEIGHBOURS = build_neighbours(Defines.CANDIDATE_RADIUS)


def candidate_cells(board, limit):
    """Return up to `limit` empty cells, nearest to the existing stones first."""
    # Distance from each empty cell near a stone to its closest stone.
    distance = {}
    for cell, stone in enumerate(board.cells):
        if stone:
            for near, d in NEIGHBOURS[cell]:
                if not board.cells[near] and d < distance.get(near, Defines.CANDIDATE_RADIUS + 1):
                    distance[near] = d

    # Almost full board: fall back to every empty cell so a move is always possible.
    if len(distance) < 2:
        return [cell for cell, stone in enumerate(board.cells) if not stone][:limit]
    return sorted(distance, key=distance.get)[:limit]


def generate_moves(board, limit=Defines.MAX_CANDIDATE_CELLS):
    """Black opens with one stone in the centre; later moves are pairs of candidate cells."""
    if not board.stones:
        return [(CENTRE,)]
    return list(combinations(candidate_cells(board, limit), 2))
