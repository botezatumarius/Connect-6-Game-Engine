"""Naive adversarial search: fixed-depth Minimax (Black maximises, White minimises)."""
from operator import itemgetter

from defines import Defines
from evaluation import evaluate
from move_generator import generate_moves
from rules import GameResult, get_game_result, opponent
from tools import make_move, unmake_move


def best_for(color):
    """max for Black, min for White."""
    return max if color == Defines.BLACK else min


class SearchEngine():
    def __init__(self):
        self.m_board = None
        self.m_chess_type = None
        self.m_depth = None
        self.m_total_nodes = 0

    def before_search(self, board, color, depth):
        self.m_board = [row[:] for row in board]
        self.m_chess_type = color
        self.m_depth = depth
        self.m_total_nodes = 0

    def minimax_search(self, bestMove):
        """Store our best move in `bestMove` and return its score."""
        scored = [(self.score_move(move, self.m_depth, self.m_chess_type), move)
                  for move in generate_moves(self.m_board)]
        score, move = best_for(self.m_chess_type)(scored, key=itemgetter(0))
        bestMove.positions = move.positions
        return score

    def score_move(self, move, depth, color):
        """Minimax value of playing `move` for `color`."""
        make_move(self.m_board, move, color)
        score = self.minimax(depth - 1, opponent(color), move)
        unmake_move(self.m_board, move)
        return score

    def minimax(self, depth, color, preMove):
        """Minimax value of the position after `preMove`, with `color` to move."""
        self.m_total_nodes += 1
        if depth == 0 or get_game_result(self.m_board, preMove) != GameResult.ONGOING:
            return evaluate(self.m_board, preMove)
        return best_for(color)(self.score_move(move, depth, color)
                               for move in generate_moves(self.m_board))
