"""Fixed-depth Minimax with alpha-beta pruning (Black maximises, White minimises)."""
from board import opponent
from defines import Defines
from evaluation import evaluate
from move_generator import generate_moves


class SearchEngine():
    """Searches the game tree on the game's board, playing and undoing moves in place."""

    def __init__(self):
        self.m_board = None
        self.m_best_move = None
        self.m_total_nodes = 0   # Nodes expanded in the last search.
        self.m_pruned_nodes = 0  # Sibling moves skipped by alpha-beta cut-offs.

    def search(self, board, color, depth):
        """Return the best move for `color` and its score. The board is restored afterwards."""
        self.m_board = board
        self.m_total_nodes = 0
        self.m_pruned_nodes = 0
        score = self.alpha_beta(depth, Defines.MININT - 1, Defines.MAXINT + 1, color)
        return self.m_best_move, score

    def alpha_beta(self, depth, alpha, beta, color, ply=0):
        """Minimax value of the current position within the (alpha, beta) window."""
        self.m_total_nodes += 1
        board = self.m_board
        # Leaf: depth limit reached or game over (the root always searches).
        if ply and (depth == 0 or board.winner or board.is_full()):
            return evaluate(board, color)

        maximizing = color == Defines.BLACK
        moves = generate_moves(board, color)
        for i, move in enumerate(moves):
            board.play(move, color)
            score = self.alpha_beta(depth - 1, alpha, beta, opponent(color), ply + 1)
            board.undo()

            # MAX raises alpha, MIN lowers beta; other moves change nothing.
            if maximizing and score > alpha:
                alpha = score
            elif not maximizing and score < beta:
                beta = score
            else:
                continue
            if ply == 0:
                self.m_best_move = move
            # Cut-off: the opponent will never allow this line, skip the remaining moves.
            if alpha >= beta:
                self.m_pruned_nodes += len(moves) - i - 1
                break
        return alpha if maximizing else beta
