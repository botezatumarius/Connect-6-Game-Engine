import sys
import time

from defines import Defines, StoneMove
from rules import opponent
from search_engine import SearchEngine
from tools import init_board, log_to_file, make_move, move2msg, msg2move, print_board

HELP = """name        - print the name of the game engine.
print       - print the board.
exit/quit   - quit the game.
black XXXX  - place the black stone(s) on XXXX, X is a letter from A to S.
white XXXX  - place the white stone(s) on XXXX.
next        - play the next move for the side to move.
move XXXX   - the opponent played XXXX; play our reply.
new black   - start a new game playing black (plays the first move).
new white   - start a new game playing white.
new xxx     - start a new game with no side assigned.
depth d     - set the time limit per turn to d seconds.
vcf, unvcf  - accepted and ignored.
help        - print this help."""


class GameEngine:
    def __init__(self, name=Defines.ENGINE_NAME):
        self.m_engine_name = name
        self.m_time_limit = Defines.DEFAULT_TIME_LIMIT
        self.m_board = [[Defines.NOSTONE] * Defines.GRID_NUM for _ in range(Defines.GRID_NUM)]
        self.m_search_engine = SearchEngine()
        self.m_commands = {
            "name": self.on_name,
            "print": self.on_print,
            "help": self.on_help,
            "new": self.on_new,
            "black": self.on_place,
            "white": self.on_place,
            "move": self.on_move,
            "next": self.on_next,
            "depth": self.on_depth,
        }
        self.init_game()

    def init_game(self):
        init_board(self.m_board)
        self.m_to_move = Defines.BLACK

    def run(self):
        while True:
            try:
                msg = input().strip()
            except EOFError:
                break
            log_to_file(msg)
            command, _, arg = msg.partition(" ")
            if command in ("exit", "quit"):
                break
            handler = self.m_commands.get(command)
            if handler:
                handler(command, arg.strip())
            sys.stdout.flush()

    def on_name(self, command, arg):
        print(f"name {self.m_engine_name}")

    def on_print(self, command, arg):
        print_board(self.m_board)

    def on_help(self, command, arg):
        print(HELP)

    def on_new(self, command, arg):
        self.init_game()
        if arg == "black":
            self.on_next(command, arg)

    def on_place(self, command, arg):
        self.place(msg2move(arg.upper()), Defines.BLACK if command == "black" else Defines.WHITE)

    def on_move(self, command, arg):
        self.place(msg2move(arg.upper()), self.m_to_move)
        self.on_next(command, arg)

    def on_next(self, command, arg):
        color = self.m_to_move
        move = self.search_a_move(color)
        self.place(move, color)
        print(f"move {move2msg(move)}")

    def on_depth(self, command, arg):
        try:
            seconds = float(arg)
        except ValueError:
            return
        if seconds > 0:
            self.m_time_limit = min(seconds, Defines.MAX_TIME_LIMIT)

    def place(self, move, color):
        make_move(self.m_board, move, color)
        self.m_to_move = opponent(color)

    def search_a_move(self, color):
        start = time.perf_counter()
        best_move = StoneMove()
        self.m_search_engine.before_search(self.m_board, color, Defines.SEARCH_DEPTH)
        score = self.m_search_engine.minimax_search(best_move)
        elapsed = time.perf_counter() - start
        print(f"info time {elapsed:.3f} nodes {self.m_search_engine.m_total_nodes} score {score}")
        return best_move
