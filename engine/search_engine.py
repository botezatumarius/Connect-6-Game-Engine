"""Alpha-beta search in negamax form, deepened iteratively until the time for the move runs out.

Below the root only the N best moves are searched (forward pruning, N-best selective search).
"""
import time

from board import opponent
from defines import Defines
from evaluation import WEIGHTS, evaluate
from move_generator import generate_moves

INFINITY = Defines.MAXINT + 1
WIN_BOUND = Defines.MAXINT - 1000  # Scores beyond it are wins or losses at a known distance.


class SearchTimeout(Exception):
    """Raised inside the search when the time for the move is over."""


class SearchEngine():
    """Searches the game tree on the game's board, playing and undoing moves in place."""

    def __init__(self, weights=WEIGHTS, branching=Defines.BRANCHING):
        self.m_weights = weights      # Evaluation weights (tuned by the genetic algorithm).
        self.m_branching = branching  # Moves searched below the root (N-best forward pruning).
        self.m_board = None
        self.m_deadline = None
        self.m_best_move = None
        self.m_depth = 0         # Depth of the last completed iteration.
        self.m_total_nodes = 0   # Nodes expanded in the last search.
        self.m_pruned_nodes = 0  # Sibling moves skipped by alpha-beta cut-offs.

    def search(self, board, color, max_depth=Defines.MAX_DEPTH, time_limit=None):
        """Best move for `color` and its score, searching deeper until time or depth runs out."""
        start = time.perf_counter()
        self.m_board = board
        self.m_deadline = start + time_limit * Defines.TIME_SAFETY if time_limit else None
        self.m_depth = self.m_total_nodes = self.m_pruned_nodes = 0
        best = None
        for depth in range(1, max_depth + 1):
            try:
                score = self.negamax(depth, -INFINITY, INFINITY, color, 0)
            except SearchTimeout:
                break  # Keep the move of the last completed iteration.
            best, self.m_depth = (self.m_best_move, score), depth
            # Stop on a forced result, or when the next (much bigger) iteration cannot finish.
            elapsed = time.perf_counter() - start
            if abs(score) > WIN_BOUND or (time_limit and elapsed > time_limit * Defines.NEXT_ITERATION):
                break
        return best

    def negamax(self, depth, alpha, beta, color, ply):
        """Value of the position for `color` (the side to move) within the (alpha, beta) window."""
        self.m_total_nodes += 1
        # The first iteration always completes, so there is always a move to play.
        if self.m_depth and self.m_deadline and time.perf_counter() > self.m_deadline:
            raise SearchTimeout
        board = self.m_board
        if ply and (depth == 0 or board.winner or board.is_full() or board.threats[color]):
            return self.evaluate(color, ply)

        best = -INFINITY
        moves = generate_moves(board, color, self.m_weights)
        if ply:
            moves = moves[:self.m_branching]  # Forward pruning: the root keeps every candidate.
        for i, move in enumerate(moves):
            board.play(move, color)
            try:
                score = -self.negamax(depth - 1, -beta, -alpha, opponent(color), ply + 1)
            finally:
                board.undo()  # Also when a timeout unwinds the search.
            if score > best:
                best = score
                if ply == 0:
                    self.m_best_move = move
            alpha = max(alpha, score)
            if alpha >= beta:
                self.m_pruned_nodes += len(moves) - i - 1
                break
        return best

    def evaluate(self, color, ply):
        """Static evaluation for `color`; wins found closer to the root score higher."""
        score = evaluate(self.m_board, color, self.m_weights)
        if score > WIN_BOUND:
            return score - ply
        if score < -WIN_BOUND:
            return score + ply
        return score
