"""Fixed-depth Minimax with alpha-beta pruning (Black maximises, White minimises)."""
from defines import Defines
from evaluation import evaluate
from move_generator import generate_moves
from rules import GameResult, get_game_result, opponent
from tools import make_move, unmake_move


class SearchEngine():
    def __init__(self):
        self.m_board = None
        self.m_chess_type = None
        self.m_depth = None
        self.m_best_move = None
        self.m_total_nodes = 0
        self.m_pruned_nodes = 0

    def before_search(self, board, color, depth):
        self.m_board = [row[:] for row in board]
        self.m_chess_type = color
        self.m_depth = depth
        self.m_total_nodes = 0
        self.m_pruned_nodes = 0

    def alpha_beta_search(self, bestMove):
        """Store our best move in `bestMove` and return its score."""
        score = self.alpha_beta(self.m_depth, Defines.MININT - 1, Defines.MAXINT + 1, self.m_chess_type, None)
        bestMove.positions = self.m_best_move.positions
        return score

    def alpha_beta(self, depth, alpha, beta, color, preMove, ply=0):
        """Minimax value of the position after `preMove` within the (alpha, beta) window."""
        self.m_total_nodes += 1
        if ply and (depth == 0 or get_game_result(self.m_board, preMove) != GameResult.ONGOING):
            return evaluate(self.m_board, preMove)

        maximizing = color == Defines.BLACK
        moves = generate_moves(self.m_board)
        for i, move in enumerate(moves):
            make_move(self.m_board, move, color)
            score = self.alpha_beta(depth - 1, alpha, beta, opponent(color), move, ply + 1)
            unmake_move(self.m_board, move)

            if maximizing and score > alpha:
                alpha = score
            elif not maximizing and score < beta:
                beta = score
            else:
                continue
            if ply == 0:
                self.m_best_move = move
            if alpha >= beta:
                self.m_pruned_nodes += len(moves) - i - 1
                break
        return alpha if maximizing else beta
