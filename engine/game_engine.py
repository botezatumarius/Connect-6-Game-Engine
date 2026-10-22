"""Text interface of the engine: reads the GUI commands from stdin and answers on stdout."""
import sys
import time

from board import Board, opponent
from defines import Defines
from opening_book import book_move
from search_engine import SearchEngine
from tools import log_to_file, move2msg, msg2move, print_board

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
    """Keeps the game position and runs the command loop."""

    def __init__(self, name=Defines.ENGINE_NAME):
        self.m_engine_name = name
        self.m_time_limit = Defines.DEFAULT_TIME_LIMIT
        self.m_search_engine = SearchEngine()
        # Command name -> handler; exit/quit, vcf, unvcf and unknown commands have none.
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
        """Empty board, black to move."""
        self.m_board = Board()
        self.m_to_move = Defines.BLACK

    def run(self):
        """Read and execute commands until exit/quit or the end of the input."""
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
            sys.stdout.flush()  # The GUI reads through a pipe: answer immediately.

    def on_name(self, command, arg):
        """name: print the engine name."""
        print(f"name {self.m_engine_name}")

    def on_print(self, command, arg):
        """print: show the board."""
        print_board(self.m_board)

    def on_help(self, command, arg):
        """help: list the commands."""
        print(HELP)

    def on_new(self, command, arg):
        """new black|white|xxx: start a game; as black, play the first move at once."""
        self.init_game()
        if arg == "black":
            self.on_next(command, arg)

    def on_place(self, command, arg):
        """black/white XXXX: put stones on the board (the GUI sends the whole history)."""
        self.place(msg2move(arg.upper()), Defines.BLACK if command == "black" else Defines.WHITE)

    def on_move(self, command, arg):
        """move XXXX: the opponent (side to move) played XXXX; answer with our move."""
        self.place(msg2move(arg.upper()), self.m_to_move)
        self.on_next(command, arg)

    def on_next(self, command, arg):
        """next: search, play and print a move for the side to move."""
        color = self.m_to_move
        move = self.search_a_move(color)
        self.place(move, color)
        print(f"move {move2msg(move)}")

    def on_depth(self, command, arg):
        """depth d: time limit of d seconds per turn, capped below the GUI timeout."""
        try:
            seconds = float(arg)
        except ValueError:
            return
        if seconds > 0:
            self.m_time_limit = min(seconds, Defines.MAX_TIME_LIMIT)

    def place(self, move, color):
        """Play a move and pass the turn to the other colour."""
        self.m_board.play(move, color)
        self.m_to_move = opponent(color)

    def search_a_move(self, color):
        """Book move, or search within the time limit and print the statistics (time, depth, nodes...)."""
        start = time.perf_counter()
        move = book_move(self.m_board)
        if move:
            print(f"info time {time.perf_counter() - start:.3f} book")
            return move
        engine = self.m_search_engine
        best_move, score = engine.search(self.m_board, color, Defines.MAX_DEPTH, self.m_time_limit)
        elapsed = time.perf_counter() - start
        print(f"info time {elapsed:.3f} depth {engine.m_depth} nodes {engine.m_total_nodes} "
              f"pruned {engine.m_pruned_nodes} score {score:.1f}")
        return best_move
